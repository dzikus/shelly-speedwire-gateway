"""Shelly 3EM to SMA Speedwire Gateway implementation.

This module implements the main gateway class with async operations,
async/await patterns, and error handling.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from pathlib import Path
from typing import Any

import structlog

from shelly_speedwire_gateway.async_mqtt_client import MQTTClient, create_mqtt_client
from shelly_speedwire_gateway.config import load_config, setup_logging_from_config
from shelly_speedwire_gateway.constants import DEFAULT_CONFIG_FILE
from shelly_speedwire_gateway.exceptions import (
    ConfigurationError,
    MQTTConnectionError,
    NetworkError,
)
from shelly_speedwire_gateway.metrics import (
    init_metrics,
    mqtt_connected,
    mqtt_messages_errors,
    mqtt_messages_received,
    mqtt_processing_time,
    shutdown_metrics,
    speedwire_active,
    update_energy_metrics,
)
from shelly_speedwire_gateway.models import GatewaySettings
from shelly_speedwire_gateway.mqtt_processor import (
    MQTTDataProcessor,
    create_mqtt_processor,
)
from shelly_speedwire_gateway.speedwire import SMASpeedwireEmulator

logger = structlog.get_logger(__name__)


class Shelly3EMSpeedwireGateway:
    """Gateway implementation with async operations."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_FILE) -> None:
        """Initialize gateway with configuration."""
        self.config_path = Path(config_path)
        self.running = False
        self.shutdown_event = asyncio.Event()

        try:
            config_dict = load_config(str(self.config_path))
            self.config = GatewaySettings.model_validate(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

        setup_logging_from_config(self.config)

        self.mqtt_client: MQTTClient | None = None
        self.speedwire: SMASpeedwireEmulator | None = None
        self.processor: MQTTDataProcessor | None = None

        self._setup_signal_handlers()

        if self.config.enable_monitoring:
            init_metrics(self.config)
        logger.info(
            "Gateway initialized",
            config_path=str(self.config_path),
            mqtt_broker=f"{self.config.mqtt.broker_host}:{self.config.mqtt.broker_port}",
            speedwire_serial=self.config.speedwire.serial,
            monitoring_enabled=self.config.enable_monitoring,
        )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, _frame: Any = None) -> None:  # ARG005: Unused lambda parameter fixed
            logger.info("Received shutdown signal", signal=signal.Signals(signum).name)
            self.shutdown_event.set()
            self.running = False

            if self.mqtt_client:
                self.mqtt_client.request_shutdown()

            if self.speedwire:
                self.speedwire.running = False

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal_handler)

    def _message_handler_wrapper(self, topic: str, payload: bytes) -> None:
        """Wrapper to make async handler sync for compatibility."""
        with contextlib.suppress(RuntimeError):
            task = asyncio.create_task(self._on_mqtt_message(topic, payload))
            # ARG005: Store task reference to avoid garbage collection warning (fixed by using _)
            task.add_done_callback(lambda _: None)

    async def _on_mqtt_message(self, topic: str, payload: bytes) -> None:
        """Handle incoming MQTT messages."""
        if not self.processor or not self.speedwire:
            return

        mqtt_messages_received.inc()

        with mqtt_processing_time.time():
            try:
                data = self.processor.process_message(topic, payload)
                if data:
                    await self.speedwire.update_data(data)

                    if self.config.enable_monitoring:
                        update_energy_metrics(data.model_dump())

                    logger.debug(
                        "Data updated",
                        device_id=data.device_id,
                        total_power=data.total_power,
                        frequency=data.freq_hz,
                    )
            except Exception as e:
                mqtt_messages_errors.inc()
                logger.exception("Failed to process MQTT message", error=str(e))
                raise

    async def _initialize_components(self) -> None:
        """Initialize MQTT client, Speedwire emulator, and data processor."""
        try:
            self.processor = create_mqtt_processor(
                device_type="shelly3em",
                invert_values=self.config.mqtt.invert_values,
                strict_validation=True,
            )

            mqtt_config_dict = self.config.mqtt.model_dump()
            self.mqtt_client = await create_mqtt_client(
                config=mqtt_config_dict,
                message_handler=self._message_handler_wrapper,
            )

            speedwire_config_dict = self.config.speedwire.model_dump()
            self.speedwire = SMASpeedwireEmulator(speedwire_config_dict)
            await self.speedwire.setup()

            logger.info("All components initialized")

            if self.config.enable_monitoring:
                mqtt_connected.set(1)
                speedwire_active.set(1)

        except Exception as e:
            logger.exception("Failed to initialize components", error=str(e))
            raise

    def _raise_task_exception(self, task_exception: BaseException | None) -> None:
        """Raise task exception if present."""
        if task_exception:
            raise task_exception

    async def _run_main_loops(self) -> None:
        """Run main application loops."""
        if not self.mqtt_client or not self.speedwire:
            raise RuntimeError("Components not initialized")

        self.running = True

        try:
            tasks = [
                asyncio.create_task(self.mqtt_client.run_message_loop(), name="mqtt_message_loop"),
                asyncio.create_task(self.speedwire.tx_loop(self.config.speedwire.interval), name="speedwire_tx_loop"),
                asyncio.create_task(self.speedwire.discovery_loop(), name="speedwire_discovery_loop"),
                asyncio.create_task(self._wait_for_shutdown(), name="shutdown_monitor"),
            ]

            logger.info("All main loops started")

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                logger.debug(f"Cancelling task: {task.get_name()}")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            for task in done:
                task_exception = task.exception()
                if task_exception:
                    logger.error(f"Task {task.get_name()} failed", error=str(task_exception))
                    self._raise_task_exception(task_exception)

        except asyncio.CancelledError:
            logger.info("Main loops cancelled")
        except Exception as e:
            logger.exception("Error in main loops", error=str(e))
            raise

    async def _wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()
        logger.info("Shutdown signal received in main loop")

    async def run(self) -> None:
        """Main gateway execution method."""
        self.running = True

        try:
            logger.info("=== Shelly 3EM to SMA Speedwire Gateway Starting ===")
            logger.info(
                "Configuration summary",
                mqtt_broker=f"{self.config.mqtt.broker_host}:{self.config.mqtt.broker_port}",
                mqtt_topic=self.config.mqtt.base_topic,
                speedwire_interval=self.config.speedwire.interval,
                speedwire_serial=self.config.speedwire.serial,
                log_level=self.config.log_level,
            )

            await self._initialize_components()

            if self.mqtt_client:
                await self.mqtt_client.connect()
                logger.info("MQTT client connected")

            await self._run_main_loops()

        except KeyboardInterrupt:
            logger.info("Gateway interrupted by user")
        except ConfigurationError as e:
            logger.exception("Configuration error", error=str(e))
            raise
        except MQTTConnectionError as e:
            logger.exception("MQTT connection error", error=str(e))
            raise
        except NetworkError as e:
            logger.exception("Network error", error=str(e))
            raise
        except Exception as e:
            logger.exception("Unexpected error in gateway", error=str(e))
            raise
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources and disconnect clients."""
        self.running = False

        logger.info("Starting gateway cleanup")

        cleanup_tasks = []

        if self.mqtt_client:
            cleanup_tasks.append(self._cleanup_mqtt())

        if self.speedwire:
            cleanup_tasks.append(self._cleanup_speedwire())

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except (OSError, ValueError, TypeError) as e:
                logger.warning("Error during cleanup", error=str(e))

        # Cleanup metrics
        if self.config.enable_monitoring:
            mqtt_connected.set(0)
            speedwire_active.set(0)
            await shutdown_metrics()

        logger.info("Gateway cleanup completed")

    async def _cleanup_mqtt(self) -> None:
        """Cleanup MQTT client."""
        try:
            if self.mqtt_client:
                await self.mqtt_client.disconnect()
                logger.debug("MQTT client disconnected")
        except (OSError, ConnectionError) as e:
            logger.warning("Error cleaning up MQTT client", error=str(e))

    async def _cleanup_speedwire(self) -> None:
        """Cleanup Speedwire emulator."""
        try:
            if self.speedwire:
                await self.speedwire.stop()
                logger.debug("Speedwire emulator stopped")
        except (OSError, ConnectionError) as e:
            logger.warning("Error cleaning up Speedwire emulator", error=str(e))

    def get_status(self) -> dict[str, Any]:
        """Get current gateway status."""
        status = {
            "running": self.running,
            "config_path": str(self.config_path),
            "mqtt": {},
            "speedwire": {},
            "processor": {},
        }

        if self.mqtt_client:
            status["mqtt"] = self.mqtt_client.get_stats()

        if self.processor:
            status["processor"] = self.processor.get_processing_stats()

        return status

    async def health_check(self) -> bool:
        """Perform health check on all components."""
        try:
            if not self.mqtt_client or not self.mqtt_client.is_connected:
                return False

            if self.processor:
                stats = self.processor.get_processing_stats()
                time_since_last = time.time() - stats.get("last_update", 0)
                if time_since_last > self.config.speedwire.health_check_timeout:
                    return False

        except (OSError, ValueError) as e:
            logger.warning("Health check failed", error=str(e))
            return False

        return self.running


async def create_gateway(config_path: str = DEFAULT_CONFIG_FILE, **kwargs: Any) -> Shelly3EMSpeedwireGateway:
    """Factory function to create and initialize gateway."""
    gateway = Shelly3EMSpeedwireGateway(config_path)

    for key, value in kwargs.items():
        if hasattr(gateway.config, key):
            setattr(gateway.config, key, value)

    return gateway


class GatewayContext:
    """Async context manager for gateway lifecycle management."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_FILE) -> None:
        """Initialize gateway context with configuration path."""
        self.config_path = config_path
        self.gateway: Shelly3EMSpeedwireGateway | None = None

    async def __aenter__(self) -> Shelly3EMSpeedwireGateway:
        """Enter context and initialize gateway."""
        self.gateway = await create_gateway(self.config_path)
        return self.gateway

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit context and cleanup gateway."""
        if self.gateway:
            await self.gateway.cleanup()


async def run_gateway_from_config(config_path: str = DEFAULT_CONFIG_FILE) -> int:
    """Run gateway with error handling and return exit code."""
    try:
        async with GatewayContext(config_path) as gateway:
            await gateway.run()

    except KeyboardInterrupt:
        logger.info("Gateway interrupted by user")
        return 0

    except ConfigurationError as e:
        logger.exception("Configuration error", error=str(e))
        return 1

    except (MQTTConnectionError, NetworkError) as e:
        logger.exception("Connection error", error=str(e))
        return 2

    except (OSError, ValueError, TypeError) as e:
        logger.exception("Unexpected error", error=str(e))
        return 3

    return 0
