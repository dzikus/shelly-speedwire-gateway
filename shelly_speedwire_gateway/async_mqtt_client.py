"""Async MQTT client using aiomqtt.

This module implements an MQTT client using the
aiomqtt library with async patterns and error handling.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack
from typing import Any

import aiomqtt
import structlog

from shelly_speedwire_gateway.exceptions import (
    GatewayTimeoutError,
    MQTTConnectionError,
    NetworkError,
)
from shelly_speedwire_gateway.models import MQTTConnectionState, MQTTSettings

logger = structlog.get_logger(__name__)


MIN_TOPIC_PARTS_FOR_PARSING = 2
MIN_TOPIC_PARTS_FOR_EMETER = 4
MESSAGE_TIMEOUT_SECONDS = 300


class MQTTClient:
    """Async MQTT client."""

    def __init__(
        self,
        config: MQTTSettings,
        message_handler: Callable[[str, bytes], None] | None = None,
    ) -> None:
        """Initialize the MQTT client.

        Args:
            config: MQTT configuration object
            message_handler: Optional callback for handling messages
        """
        self.config = config
        self.state = MQTTConnectionState()
        self.message_handler = message_handler
        self._client: aiomqtt.Client | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._shutdown_event = asyncio.Event()
        self._running = False

        self.log = logger.bind(
            device_id=self.config.device_id,
            broker=f"{self.config.broker_host}:{self.config.broker_port}",
        )

    async def __aenter__(self) -> MQTTClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def request_shutdown(self) -> None:
        """Request graceful shutdown - synchronous method."""
        self._shutdown_event.set()
        self._running = False
        self.log.info("Shutdown requested")

    async def connect(self) -> None:
        """Connect to MQTT broker with retry logic."""
        self.state.reset_connection()

        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            if self._shutdown_event.is_set():
                raise MQTTConnectionError("Connection cancelled due to shutdown")

            try:
                await self._attempt_connection()
                self.state.mark_connected()
                self.log.info("MQTT connection established", attempt=attempt)
            except (OSError, aiomqtt.MqttError, ConnectionError) as e:
                self.state.reconnect_attempts = attempt

                if attempt == self.config.max_reconnect_attempts:
                    self.log.exception("MQTT connection failed after all attempts", attempts=attempt, error=str(e))
                    raise MQTTConnectionError(
                        f"Failed to connect after {attempt} attempts: {e}",
                        broker_host=self.config.broker_host,
                        broker_port=self.config.broker_port,
                        reconnect_attempts=attempt,
                    ) from e

                backoff_time = self.config.backoff_factor * (2 ** (attempt - 1))
                self.log.warning(
                    "MQTT connection attempt failed, retrying",
                    attempt=attempt,
                    backoff_seconds=backoff_time,
                    error=str(e),
                )

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=backoff_time)
                    raise MQTTConnectionError("Connection cancelled during backoff") from e
                except TimeoutError:
                    continue
            else:
                # TRY300: Move return to else block
                return

    async def _attempt_connection(self) -> None:
        """Attempt single MQTT connection."""
        self._exit_stack = AsyncExitStack()

        client_kwargs: dict[str, Any] = {
            "hostname": self.config.broker_host,
            "port": self.config.broker_port,
            "keepalive": self.config.keepalive,
            "clean_session": True,
            "identifier": f"speedwire_gateway_{self.config.device_id}",
        }

        if self.config.username:
            client_kwargs.update(
                {
                    "username": self.config.username,
                    "password": self.config.password,
                },
            )

        self._client = await self._exit_stack.enter_async_context(aiomqtt.Client(**client_kwargs))

        await self._subscribe_to_topics()

    async def _subscribe_to_topics(self) -> None:
        """Subscribe to required MQTT topics."""
        if not self._client:
            raise RuntimeError("Client not connected")

        topics = [
            f"{self.config.base_topic}/emeter/+/+",
            f"{self.config.base_topic}/online",
        ]

        for topic in topics:
            await self._client.subscribe(topic, qos=self.config.qos)
            self.log.debug("Subscribed to topic", topic=topic)

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker gracefully."""
        self.request_shutdown()
        self.state.connected = False

        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except (OSError, ValueError, RuntimeError) as e:  # BLE001: Use specific exceptions
                self.log.warning("Error during exit stack cleanup", error=str(e))
            finally:
                self._exit_stack = None

        self._client = None
        self.log.info("MQTT disconnected")

    async def run_message_loop(self) -> None:
        """Run the main message processing loop."""
        if not self._client:
            raise RuntimeError("Client not connected")

        self.log.info("Starting MQTT message loop")
        self._running = True

        try:
            await asyncio.gather(
                self._process_messages(),
                self._monitor_connection(),
                self._wait_for_shutdown(),
                return_exceptions=True,
            )

        except Exception as e:
            if isinstance(e, MQTTConnectionError | NetworkError):  # UP038: Use X | Y
                self.log.exception("MQTT error in message loop", error=str(e))
            else:
                self.log.exception("Unexpected error in message loop")
            raise
        finally:
            self._running = False
            self.log.info("MQTT message loop stopped")

    async def _wait_for_shutdown(self) -> None:
        """Wait for MQTT client shutdown signal."""
        await self._shutdown_event.wait()
        self.log.info("Shutdown signal received in message loop")

    async def _process_messages(self) -> None:
        """Process incoming MQTT messages."""
        if not self._client:
            raise RuntimeError("Client not connected")

        try:
            async for message in self._client.messages:
                if self._shutdown_event.is_set() or not self._running:
                    self.log.debug("Breaking message loop due to shutdown")
                    break

                try:
                    await self._handle_message(message)
                    self.state.update_last_message_time()

                except (ValueError, TypeError, OSError) as e:
                    self.log.exception("Error processing MQTT message", topic=str(message.topic), error=str(e))
        except (asyncio.CancelledError, ConnectionError) as e:
            self.log.info("Message processing cancelled", error=str(e))
            raise
        except Exception as e:
            self.log.exception("Unexpected error in message processing", error=str(e))
            raise

    async def _handle_message(self, message: aiomqtt.Message) -> None:
        """Handle individual MQTT message with pattern matching."""
        topic = str(message.topic)
        payload = message.payload

        # Convert payload to bytes safely with type checking
        payload_bytes = self._convert_payload_to_bytes(payload)
        self.log.debug("Received MQTT message", topic=topic, payload_size=len(payload_bytes))

        # Pattern matching for message routing
        topic_parts = topic.split("/")
        if len(topic_parts) >= MIN_TOPIC_PARTS_FOR_PARSING:  # PLR2004: Use constant
            if topic_parts[-1] == "online":
                await self._handle_online_message(payload_bytes)
            elif (
                len(topic_parts) >= MIN_TOPIC_PARTS_FOR_EMETER  # PLR2004: Use constant
                and topic_parts[-3] == "emeter"
            ):
                phase = topic_parts[-2]
                measurement_type = topic_parts[-1]
                await self._handle_emeter_message(phase, measurement_type, payload_bytes)
            else:
                self.log.warning("Unknown message topic", topic=topic)

        # Call external message handler if provided
        if self.message_handler:
            try:
                await self._run_handler_safely(topic, payload_bytes)
            except (ValueError, TypeError, OSError) as e:
                self.log.exception("Error in external message handler", topic=topic, error=str(e))

    def _convert_payload_to_bytes(self, payload: Any) -> bytes:
        """Convert MQTT payload to bytes with type checking."""
        if payload is None:
            return b""
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray | memoryview):
            return bytes(payload)
        if isinstance(payload, str):
            return payload.encode("utf-8")
        if isinstance(payload, int | float):
            return str(payload).encode("utf-8")
        # Fallback - try to convert to string then bytes
        try:
            return str(payload).encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            self.log.warning("Could not convert payload to bytes, using empty bytes", payload_type=type(payload))
            return b""

    async def _handle_online_message(self, payload: bytes) -> None:
        """Handle device online status messages."""
        try:
            status = payload.decode("utf-8")
            self.log.info("Device online status", status=status)
        except UnicodeDecodeError:
            self.log.warning("Invalid online message payload")

    async def _handle_emeter_message(self, phase: str, measurement_type: str, payload: bytes) -> None:
        """Handle energy meter measurement messages."""
        try:
            value = float(payload.decode("utf-8"))
            self.log.debug("Energy measurement received", phase=phase, measurement_type=measurement_type, value=value)
        except (UnicodeDecodeError, ValueError) as e:
            self.log.warning(
                "Invalid emeter message payload",
                phase=phase,
                measurement_type=measurement_type,
                error=str(e),
            )

    async def _run_handler_safely(self, topic: str, payload: bytes) -> None:
        """Run external message handler with timeout."""
        if not self.message_handler:
            return

        try:
            if asyncio.iscoroutinefunction(self.message_handler):
                await asyncio.wait_for(self.message_handler(topic, payload), timeout=self.config.connection_timeout)
            else:
                await asyncio.get_event_loop().run_in_executor(None, self.message_handler, topic, payload)
        except TimeoutError as exc:
            raise GatewayTimeoutError(
                f"Message handler timeout for topic {topic}",
                timeout_seconds=self.config.connection_timeout,
                operation="message_handler",
            ) from exc

    async def _monitor_connection(self) -> None:
        """Monitor connection health and handle reconnections."""
        while self._running and not self._shutdown_event.is_set():
            try:
                if not self.state.connected:
                    self.log.warning("Connection lost, attempting reconnection")
                    # SIM105: Use contextlib.suppress
                    with contextlib.suppress(MQTTConnectionError):
                        await self.connect()

                if self.state.time_since_last_message > MESSAGE_TIMEOUT_SECONDS:  # PLR2004: Use constant
                    self.log.warning(
                        "No messages received recently",
                        seconds_since_last=self.state.time_since_last_message,
                    )

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=30.0)
                    break
                except TimeoutError:
                    continue

            except (ConnectionError, OSError, MQTTConnectionError) as e:
                self.log.exception("Error in connection monitor", error=str(e))
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=60.0)
                    break
                except TimeoutError:
                    continue

        self.log.info("Connection monitor stopped")

    async def publish(
        self,
        topic: str,
        payload: bytes | str,
        qos: int | None = None,
        retain: bool | None = None,  # noqa: FBT001
    ) -> None:
        """Publish message to MQTT broker."""
        if not self._client or not self.state.connected:
            raise MQTTConnectionError("Client not connected")

        try:
            await self._client.publish(topic=topic, payload=payload, qos=qos or self.config.qos, retain=retain or False)
            payload_size = (
                len(payload) if isinstance(payload, bytes | str) else 0  # UP038: Use X | Y
            )
            self.log.debug("Message published", topic=topic, payload_size=payload_size)

        except (OSError, aiomqtt.MqttError) as e:
            self.log.exception("Failed to publish message", topic=topic, error=str(e))
            raise MQTTConnectionError(f"Publish failed: {e}") from e

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.state.connected and self._client is not None

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self.is_connected,
            "reconnect_attempts": self.state.reconnect_attempts,
            "connection_duration": self.state.connection_duration,
            "last_message_time": self.state.last_message_time,
            "time_since_last_message": self.state.time_since_last_message,
            "total_messages_received": self.state.total_messages_received,
            "messages_per_second": self.state.messages_per_second,
        }


async def create_mqtt_client(
    config: dict[str, Any],
    message_handler: Callable[[str, bytes], None] | None = None,
) -> MQTTClient:
    """Factory function to create and configure MQTT client."""
    mqtt_config = MQTTSettings(
        base_topic=config["base_topic"],
        broker_host=config.get("broker_host", "localhost"),
        broker_port=config.get("broker_port", 1883),
        username=config.get("username"),
        password=config.get("password"),
        keepalive=config.get("keepalive", 60),
        invert_values=config.get("invert_values", False),
        qos=config.get("qos", 1),
    )

    return MQTTClient(mqtt_config, message_handler)


@contextlib.asynccontextmanager
async def mqtt_session(
    config: dict[str, Any],
    message_handler: Callable[[str, bytes], None] | None = None,
) -> AsyncGenerator[MQTTClient]:
    """Async context manager for MQTT sessions."""
    client = await create_mqtt_client(config, message_handler)

    try:
        async with client:
            yield client
    finally:
        await client.disconnect()
