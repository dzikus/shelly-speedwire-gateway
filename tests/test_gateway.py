"""Tests for gateway functionality."""
# pylint: disable=redefined-outer-name,protected-access,too-many-public-methods

from __future__ import annotations

import asyncio
import signal
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
import yaml

from shelly_speedwire_gateway.exceptions import (
    ConfigurationError,
    MQTTConnectionError,
    NetworkError,
)
from shelly_speedwire_gateway.gateway import (
    GatewayContext,
    Shelly3EMSpeedwireGateway,
    create_gateway,
    run_gateway_from_config,
)
from shelly_speedwire_gateway.models import PhaseData, Shelly3EMData


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    """Valid configuration dictionary."""
    return {
        "mqtt": {
            "broker_host": "test.mqtt.com",
            "broker_port": 1883,
            "base_topic": "shellies/test-device",
        },
        "speedwire": {
            "serial": 123456789,
            "interval": 1.0,
        },
        "log_level": "INFO",
        "enable_monitoring": False,
    }


@pytest.fixture
def config_file(valid_config_dict: dict[str, Any]) -> Generator[str]:
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(valid_config_dict, f)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestShelly3EMSpeedwireGateway:
    """Test main gateway class."""

    @patch("shelly_speedwire_gateway.gateway.setup_logging_from_config")
    @patch("shelly_speedwire_gateway.gateway.load_config")
    def test_gateway_initialization(
        self,
        mock_load_config: Mock,
        mock_setup_logging: Mock,
        valid_config_dict: dict[str, Any],
    ) -> None:
        """Test gateway initialization with valid config."""
        mock_load_config.return_value = valid_config_dict

        gateway = Shelly3EMSpeedwireGateway("test_config.yaml")

        assert gateway.config_path == Path("test_config.yaml")
        assert gateway.running is False
        assert gateway.mqtt_client is None
        assert gateway.speedwire is None
        assert gateway.processor is None
        mock_setup_logging.assert_called_once_with(gateway.config)

    @patch("shelly_speedwire_gateway.gateway.load_config")
    def test_gateway_init_config_error(self, mock_load_config: Mock) -> None:
        """Test gateway initialization with configuration error."""
        mock_load_config.side_effect = Exception("Config load failed")

        with pytest.raises(ConfigurationError) as exc_info:
            Shelly3EMSpeedwireGateway("invalid_config.yaml")

        assert "Failed to load configuration" in str(exc_info.value)

    @patch("shelly_speedwire_gateway.gateway.setup_logging_from_config")
    @patch("shelly_speedwire_gateway.gateway.init_metrics")
    @patch("shelly_speedwire_gateway.gateway.load_config")
    def test_gateway_init_with_monitoring(
        self,
        mock_load_config: Mock,
        mock_init_metrics: Mock,
        mock_setup_logging: Mock,
        valid_config_dict: dict[str, Any],
    ) -> None:
        """Test gateway initialization with monitoring enabled."""
        config_with_monitoring = valid_config_dict.copy()
        config_with_monitoring["enable_monitoring"] = True
        mock_load_config.return_value = config_with_monitoring

        gateway = Shelly3EMSpeedwireGateway("test_config.yaml")

        assert gateway.config.enable_monitoring is True
        mock_init_metrics.assert_called_once_with(gateway.config)
        mock_setup_logging.assert_called_once_with(gateway.config)

    def test_signal_handler_setup(self, config_file: str) -> None:
        """Test signal handlers are set up correctly."""
        with patch("signal.signal") as mock_signal:
            _ = Shelly3EMSpeedwireGateway(config_file)

            # Verify signal handlers were set up
            assert mock_signal.call_count >= 2
            signal_calls = [call[0][0] for call in mock_signal.call_args_list]
            assert signal.SIGTERM in signal_calls
            assert signal.SIGINT in signal_calls

    def test_signal_handler_execution(self, config_file: str) -> None:
        """Test signal handler behavior."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.mqtt_client = Mock()
        gateway.speedwire = Mock()

        # Simulate signal handler call
        with patch("signal.signal") as mock_signal:
            gateway = Shelly3EMSpeedwireGateway(config_file)

            # Get the signal handler function
            signal_handler = mock_signal.call_args_list[0][0][1]

            # Call the handler
            signal_handler(signal.SIGTERM)

            assert gateway.running is False
            assert gateway.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_message_handler_wrapper(self, config_file: str) -> None:
        """Test MQTT message handler wrapper."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        with patch.object(gateway, "_on_mqtt_message", new_callable=AsyncMock) as mock_handler:
            gateway._message_handler_wrapper("test/topic", b"test payload")

            # Give asyncio a chance to schedule the task
            await asyncio.sleep(0.1)

            mock_handler.assert_called_once_with("test/topic", b"test payload")

    @pytest.mark.asyncio
    async def test_on_mqtt_message_success(self, config_file: str) -> None:
        """Test successful MQTT message processing."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.processor = Mock()
        gateway.speedwire = AsyncMock()

        mock_data = Mock()
        mock_data.device_id = "test-device"
        mock_data.total_power = 1000.0
        mock_data.freq_hz = 50.0
        mock_data.model_dump.return_value = {"test": "data"}

        gateway.processor.process_message.return_value = mock_data

        await gateway._on_mqtt_message("test/topic", b"test payload")

        gateway.processor.process_message.assert_called_once_with("test/topic", b"test payload")
        gateway.speedwire.update_data.assert_called_once_with(mock_data)

    @pytest.mark.asyncio
    async def test_on_mqtt_message_no_components(self, config_file: str) -> None:
        """Test MQTT message handling with no components initialized."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        # Should return early without error
        await gateway._on_mqtt_message("test/topic", b"test payload")

    @pytest.mark.asyncio
    async def test_on_mqtt_message_processing_error(self, config_file: str) -> None:
        """Test MQTT message processing with error."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.processor = Mock()
        gateway.speedwire = AsyncMock()

        gateway.processor.process_message.side_effect = Exception("Processing failed")

        with pytest.raises(Exception, match="Processing failed"):
            await gateway._on_mqtt_message("test/topic", b"test payload")

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.create_mqtt_processor")
    @patch("shelly_speedwire_gateway.gateway.create_mqtt_client")
    @patch("shelly_speedwire_gateway.gateway.SMASpeedwireEmulator")
    async def test_initialize_components_success(
        self,
        mock_speedwire_class: Mock,
        mock_create_client: Mock,
        mock_create_processor: Mock,
        config_file: str,
    ) -> None:
        """Test successful component initialization."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        # Setup mocks
        mock_processor = Mock()
        mock_create_processor.return_value = mock_processor

        mock_client = AsyncMock()
        mock_create_client.return_value = mock_client

        mock_speedwire = AsyncMock()
        mock_speedwire_class.return_value = mock_speedwire

        await gateway._initialize_components()

        assert gateway.processor == mock_processor
        assert gateway.mqtt_client == mock_client
        assert gateway.speedwire == mock_speedwire

        mock_create_processor.assert_called_once()
        mock_create_client.assert_called_once()
        mock_speedwire.setup.assert_called_once()

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.create_mqtt_processor")
    async def test_initialize_components_failure(self, mock_create_processor: Mock, config_file: str) -> None:
        """Test component initialization failure."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        mock_create_processor.side_effect = Exception("Initialization failed")

        with pytest.raises(Exception, match="Initialization failed"):
            await gateway._initialize_components()

    @pytest.mark.asyncio
    async def test_run_main_loops_not_initialized(self, config_file: str) -> None:
        """Test main loops with uninitialized components."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        with pytest.raises(RuntimeError, match="Components not initialized"):
            await gateway._run_main_loops()

    @pytest.mark.asyncio
    async def test_run_main_loops_success(self, config_file: str) -> None:
        """Test successful main loops execution."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.mqtt_client = AsyncMock()
        gateway.speedwire = AsyncMock()

        # Mock the loops to complete quickly
        gateway.mqtt_client.run_message_loop = AsyncMock(side_effect=asyncio.CancelledError())
        gateway.speedwire.tx_loop = AsyncMock(side_effect=asyncio.CancelledError())
        gateway.speedwire.discovery_loop = AsyncMock(side_effect=asyncio.CancelledError())

        # Trigger shutdown immediately
        gateway.shutdown_event.set()

        await gateway._run_main_loops()

        assert gateway.running is True

    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self, config_file: str) -> None:
        """Test shutdown wait functionality."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        # Set shutdown event immediately
        gateway.shutdown_event.set()

        await gateway._wait_for_shutdown()

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.logger")
    async def test_run_keyboard_interrupt(self, mock_logger: Mock, config_file: str) -> None:
        """Test gateway run with keyboard interrupt."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        with patch.object(gateway, "_initialize_components", side_effect=KeyboardInterrupt()):
            await gateway.run()

        mock_logger.info.assert_any_call("Gateway interrupted by user")

    @pytest.mark.asyncio
    async def test_run_configuration_error(self, config_file: str) -> None:
        """Test gateway run with configuration error."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        with (
            patch.object(gateway, "_initialize_components", side_effect=ConfigurationError("Config error")),
            pytest.raises(ConfigurationError),
        ):
            await gateway.run()

    @pytest.mark.asyncio
    async def test_run_mqtt_connection_error(self, config_file: str) -> None:
        """Test gateway run with MQTT connection error."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        with (
            patch.object(gateway, "_initialize_components", side_effect=MQTTConnectionError("MQTT error")),
            pytest.raises(MQTTConnectionError),
        ):
            await gateway.run()

    @pytest.mark.asyncio
    async def test_run_network_error(self, config_file: str) -> None:
        """Test gateway run with network error."""
        gateway = Shelly3EMSpeedwireGateway(config_file)

        with (
            patch.object(gateway, "_initialize_components", side_effect=NetworkError("Network error")),
            pytest.raises(NetworkError),
        ):
            await gateway.run()

    @pytest.mark.asyncio
    async def test_cleanup_all_components(self, config_file: str) -> None:
        """Test cleanup with all components initialized."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.mqtt_client = AsyncMock()
        gateway.speedwire = AsyncMock()

        with patch("shelly_speedwire_gateway.gateway.shutdown_metrics", new_callable=AsyncMock):
            await gateway.cleanup()

            assert gateway.running is False
            gateway.mqtt_client.disconnect.assert_called_once()
            gateway.speedwire.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_monitoring(self, config_file: str) -> None:
        """Test cleanup with monitoring enabled."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.config.enable_monitoring = True

        with patch("shelly_speedwire_gateway.gateway.shutdown_metrics", new_callable=AsyncMock) as mock_shutdown:
            await gateway.cleanup()

            mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_mqtt_error(self, config_file: str) -> None:
        """Test cleanup with MQTT client error."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.mqtt_client = AsyncMock()
        gateway.mqtt_client.disconnect.side_effect = ConnectionError("Disconnect failed")

        # Should not raise exception
        await gateway.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_speedwire_error(self, config_file: str) -> None:
        """Test cleanup with Speedwire error."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.speedwire = AsyncMock()
        gateway.speedwire.stop.side_effect = OSError("Stop failed")

        # Should not raise exception
        await gateway.cleanup()

    def test_get_status(self, config_file: str) -> None:
        """Test status retrieval."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.running = True
        gateway.mqtt_client = Mock()
        gateway.mqtt_client.get_stats.return_value = {"connected": True}
        gateway.processor = Mock()
        gateway.processor.get_processing_stats.return_value = {"messages": 100}

        status = gateway.get_status()

        assert status["running"] is True
        assert status["config_path"] == str(gateway.config_path)
        assert status["mqtt"] == {"connected": True}
        assert status["processor"] == {"messages": 100}

    @pytest.mark.asyncio
    async def test_health_check_success(self, config_file: str) -> None:
        """Test successful health check."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.running = True
        gateway.mqtt_client = Mock()
        gateway.mqtt_client.is_connected = True
        gateway.processor = Mock()
        gateway.processor.get_processing_stats.return_value = {"last_update": 1234567890}

        with patch("time.time", return_value=1234567891):  # 1 second later
            result = await gateway.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_mqtt_disconnected(self, config_file: str) -> None:
        """Test health check with MQTT disconnected."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.running = True
        gateway.mqtt_client = Mock()
        gateway.mqtt_client.is_connected = False

        result = await gateway.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_stale_data(self, config_file: str) -> None:
        """Test health check with stale data."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.running = True
        gateway.mqtt_client = Mock()
        gateway.mqtt_client.is_connected = True
        gateway.processor = Mock()
        gateway.processor.get_processing_stats.return_value = {"last_update": 1234567890}

        # Simulate stale data (more than health check timeout)
        with patch("time.time", return_value=1234567890 + gateway.config.speedwire.health_check_timeout + 1):
            result = await gateway.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, config_file: str) -> None:
        """Test health check with exception."""
        gateway = Shelly3EMSpeedwireGateway(config_file)
        gateway.running = True
        gateway.mqtt_client = Mock()

        # Configure property to raise exception when accessed
        type(gateway.mqtt_client).is_connected = PropertyMock(side_effect=OSError("Connection error"))

        result = await gateway.health_check()

        # The OSError is caught and logged, then it returns False
        assert result is False


class TestCreateGateway:
    """Test gateway factory function."""

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.Shelly3EMSpeedwireGateway")
    async def test_create_gateway_default(self, mock_gateway_class: Mock) -> None:
        """Test gateway creation with defaults."""
        mock_gateway = Mock()
        mock_gateway.config = Mock()
        mock_gateway_class.return_value = mock_gateway

        gateway = await create_gateway()

        mock_gateway_class.assert_called_once_with("shelly_speedwire_gateway_config.yaml")
        assert gateway == mock_gateway

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.Shelly3EMSpeedwireGateway")
    async def test_create_gateway_custom_config(self, mock_gateway_class: Mock) -> None:
        """Test gateway creation with custom config path."""
        mock_gateway = Mock()
        mock_gateway.config = Mock()
        mock_gateway_class.return_value = mock_gateway

        _ = await create_gateway("/custom/config.yaml")

        mock_gateway_class.assert_called_once_with("/custom/config.yaml")

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.Shelly3EMSpeedwireGateway")
    async def test_create_gateway_with_overrides(self, mock_gateway_class: Mock) -> None:
        """Test gateway creation with config overrides."""
        mock_gateway = Mock()
        mock_config = Mock()
        mock_config.log_level = "INFO"
        mock_gateway.config = mock_config
        mock_gateway_class.return_value = mock_gateway

        _ = await create_gateway(log_level="DEBUG")

        mock_gateway_class.assert_called_once()
        assert mock_config.log_level == "DEBUG"


class TestGatewayContext:
    """Test gateway context manager."""

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.create_gateway")
    async def test_context_manager_success(self, mock_create_gateway: Mock) -> None:
        """Test successful context manager usage."""
        mock_gateway = AsyncMock()
        mock_create_gateway.return_value = mock_gateway

        async with GatewayContext("test_config.yaml") as gateway:
            assert gateway == mock_gateway
            mock_create_gateway.assert_called_once_with("test_config.yaml")

        mock_gateway.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.create_gateway")
    async def test_context_manager_with_exception(self, mock_create_gateway: Mock) -> None:
        """Test context manager with exception in body."""
        mock_gateway = AsyncMock()
        mock_create_gateway.return_value = mock_gateway

        # Test exception handling in context manager
        exception_occurred = False
        try:
            async with GatewayContext("test_config.yaml") as gateway:
                assert gateway == mock_gateway
                raise ValueError("Test exception")
        except ValueError:
            exception_occurred = True

        assert exception_occurred, "Expected ValueError to be raised"

        # Verify cleanup was called after exception
        mock_gateway.cleanup.assert_called_once()


class TestRunGatewayFromConfig:
    """Test gateway runner function."""

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.GatewayContext")
    async def test_run_gateway_success(self, mock_context_class: Mock) -> None:
        """Test successful gateway run."""
        mock_gateway = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_gateway
        mock_context_class.return_value = mock_context

        result = await run_gateway_from_config("test_config.yaml")

        assert result == 0
        mock_context_class.assert_called_once_with("test_config.yaml")
        mock_gateway.run.assert_called_once()

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.GatewayContext")
    async def test_run_gateway_keyboard_interrupt(self, mock_context_class: Mock) -> None:
        """Test gateway run with keyboard interrupt."""
        mock_gateway = AsyncMock()
        mock_gateway.run.side_effect = KeyboardInterrupt()
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_gateway
        mock_context_class.return_value = mock_context

        result = await run_gateway_from_config("test_config.yaml")

        assert result == 0

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.GatewayContext")
    async def test_run_gateway_configuration_error(self, mock_context_class: Mock) -> None:
        """Test gateway run with configuration error."""
        mock_gateway = AsyncMock()
        mock_gateway.run.side_effect = ConfigurationError("Config error")
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_gateway
        mock_context_class.return_value = mock_context

        result = await run_gateway_from_config("test_config.yaml")

        assert result == 1

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.GatewayContext")
    async def test_run_gateway_connection_error(self, mock_context_class: Mock) -> None:
        """Test gateway run with connection error."""
        mock_gateway = AsyncMock()
        mock_gateway.run.side_effect = MQTTConnectionError("MQTT error")
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_gateway
        mock_context_class.return_value = mock_context

        result = await run_gateway_from_config("test_config.yaml")

        assert result == 2

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.GatewayContext")
    async def test_run_gateway_network_error(self, mock_context_class: Mock) -> None:
        """Test gateway run with network error."""
        mock_gateway = AsyncMock()
        mock_gateway.run.side_effect = NetworkError("Network error")
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_gateway
        mock_context_class.return_value = mock_context

        result = await run_gateway_from_config("test_config.yaml")

        assert result == 2

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.gateway.GatewayContext")
    async def test_run_gateway_unexpected_error(self, mock_context_class: Mock) -> None:
        """Test gateway run with unexpected error."""
        mock_gateway = AsyncMock()
        mock_gateway.run.side_effect = ValueError("Unexpected error")
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_gateway
        mock_context_class.return_value = mock_context

        result = await run_gateway_from_config("test_config.yaml")

        assert result == 3


class TestGatewayErrorHandling:
    """Test gateway error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_signal_handler_with_clients(self) -> None:
        """Test signal handler when mqtt_client and speedwire exist."""
        config = Mock()
        config.enable_monitoring = False
        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = True
        gateway.shutdown_event = asyncio.Event()

        # Set up mock clients to test signal handler paths
        gateway.mqtt_client = Mock()
        gateway.mqtt_client.request_shutdown = Mock()
        gateway.speedwire = Mock()

        # Mock signal handler function
        with patch("signal.signal") as mock_signal:
            gateway._setup_signal_handlers()

            # Get the signal handler function from the call
            signal_handler = mock_signal.call_args_list[0][0][1]

            # Call signal handler to test lines 88, 91
            signal_handler(signal.SIGTERM.value, None)

            # Verify shutdown was requested
            gateway.mqtt_client.request_shutdown.assert_called_once()
            assert not gateway.running

    @pytest.mark.asyncio
    async def test_monitoring_enabled_paths(self) -> None:
        """Test code paths when monitoring is enabled."""
        config = Mock()
        config.enable_monitoring = True
        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.processor = Mock()
        gateway.speedwire = AsyncMock()

        test_data = Shelly3EMData(
            a=PhaseData(voltage=230.0, current=5.0, power=1000.0, pf=0.9, energy_consumed=100.0, energy_exported=50.0),
            b=PhaseData(voltage=230.0, current=5.0, power=1000.0, pf=0.9, energy_consumed=100.0, energy_exported=50.0),
            c=PhaseData(voltage=230.0, current=5.0, power=1000.0, pf=0.9, energy_consumed=100.0, energy_exported=50.0),
            device_id="test_device",
            freq_hz=50.0,
            timestamp=12345678,
        )

        # Mock processor to return test data
        gateway.processor.process_message = Mock(return_value=test_data)

        # Test _on_mqtt_message with monitoring enabled (line 117)
        with patch("shelly_speedwire_gateway.gateway.update_energy_metrics") as mock_update_metrics:
            await gateway._on_mqtt_message("test/topic", b"test payload")

            # Verify monitoring was called (line 117)
            mock_update_metrics.assert_called_once_with(test_data.model_dump())

        # Test _initialize_components with monitoring (lines 152-153)
        with (
            patch("shelly_speedwire_gateway.gateway.create_mqtt_client") as mock_create_mqtt,
            patch("shelly_speedwire_gateway.gateway.SMASpeedwireEmulator") as mock_speedwire_class,
            patch("shelly_speedwire_gateway.gateway.create_mqtt_processor") as mock_create_processor,
            patch("shelly_speedwire_gateway.gateway.mqtt_connected") as mock_mqtt_metric,
            patch("shelly_speedwire_gateway.gateway.speedwire_active") as mock_speedwire_metric,
        ):
            mock_create_mqtt.return_value = AsyncMock()
            mock_speedwire_emulator = AsyncMock()
            mock_speedwire_class.return_value = mock_speedwire_emulator
            mock_create_processor.return_value = Mock()

            # Test the actual _initialize_components method to hit lines 152-153
            await gateway._initialize_components()

            # Verify metrics were set (lines 152-153)
            mock_mqtt_metric.set.assert_called_with(1)
            mock_speedwire_metric.set.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_raise_task_exception(self) -> None:
        """Test _raise_task_exception method (lines 161-162)."""
        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)

        # Test with no exception
        gateway._raise_task_exception(None)  # Should not raise

        # Test with exception
        test_exception = RuntimeError("Test error")
        with pytest.raises(RuntimeError, match="Test error"):
            gateway._raise_task_exception(test_exception)

    @pytest.mark.asyncio
    async def test_run_main_loops_task_exceptions(self) -> None:
        """Test _run_main_loops task exception handling (lines 184-187, 191-193)."""
        config = Mock()
        config.speedwire = Mock()
        config.speedwire.interval = 1.0

        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = False
        gateway.mqtt_client = AsyncMock()
        gateway.speedwire = AsyncMock()

        # Mock a task that raises an exception
        failed_task = Mock()
        failed_task.get_name.return_value = "failing_task"
        failed_task.exception.return_value = RuntimeError("Task failed")
        failed_task.cancel = Mock()

        # Mock asyncio.create_task to return our failed task
        with (
            patch("asyncio.create_task", return_value=failed_task),
            patch("asyncio.wait") as mock_wait,
            patch("shelly_speedwire_gateway.gateway.logger"),
        ):
            # Simulate task completion with exception
            mock_wait.return_value = ([failed_task], [])

            # Should raise the task exception (lines 191-193)
            with pytest.raises(RuntimeError, match="Task failed"):
                await gateway._run_main_loops()

            # Verify exception handling was called
            failed_task.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_main_loops_cancelled_error(self) -> None:
        """Test _run_main_loops CancelledError handling (lines 197-199)."""
        config = Mock()
        config.speedwire = Mock()
        config.speedwire.interval = 1.0

        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = False
        gateway.mqtt_client = AsyncMock()
        gateway.speedwire = AsyncMock()

        # Mock asyncio.wait to raise CancelledError
        with (
            patch("asyncio.create_task"),
            patch("asyncio.wait", side_effect=asyncio.CancelledError()),
            patch("shelly_speedwire_gateway.gateway.logger") as mock_logger,
        ):
            # Should handle CancelledError gracefully (lines 195-196)
            await gateway._run_main_loops()

            # Verify the cancellation was logged
            mock_logger.info.assert_called_with("Main loops cancelled")

    @pytest.mark.asyncio
    async def test_run_mqtt_connect_path(self) -> None:
        """Test run method MQTT connection path (lines 223-227)."""
        config = Mock()
        config.mqtt = Mock()
        config.mqtt.broker_host = "test.host"
        config.mqtt.broker_port = 1883
        config.mqtt.base_topic = "test/topic"
        config.speedwire = Mock()
        config.speedwire.interval = 1.0
        config.speedwire.serial = 123456
        config.log_level = "INFO"
        config.enable_monitoring = False

        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = True
        gateway.speedwire = None  # Initialize speedwire attribute

        # Mock components
        mqtt_client = AsyncMock()
        mqtt_client.connect = AsyncMock()

        with (
            patch.object(gateway, "_initialize_components") as mock_init,
            patch.object(gateway, "_run_main_loops") as mock_run_main_loops,
            patch("shelly_speedwire_gateway.gateway.logger") as mock_logger,
        ):
            # Set up the mqtt_client after initialization
            async def setup_mqtt() -> None:
                gateway.mqtt_client = mqtt_client

            # Make sure _run_main_loops doesn't return an unawaited coroutine
            mock_run_main_loops.return_value = None
            mock_init.side_effect = setup_mqtt

            await gateway.run()

            # Verify MQTT client connection path was executed (lines 223-225)
            mqtt_client.connect.assert_called_once()
            mock_logger.info.assert_any_call("MQTT client connected")

    @pytest.mark.asyncio
    async def test_run_exception_paths(self) -> None:
        """Test run method exception handling (lines 240-242)."""
        config = Mock()
        config.enable_monitoring = False

        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = True

        # Test general exception handling (lines 240-242)
        with (
            patch.object(gateway, "_initialize_components", side_effect=ValueError("Test error")),
            patch.object(gateway, "cleanup") as mock_cleanup,
            patch("shelly_speedwire_gateway.gateway.logger") as mock_logger,
        ):
            with pytest.raises(ValueError, match="Test error"):
                await gateway.run()

            # Verify exception was logged and cleanup was called
            mock_logger.exception.assert_called_with("Unexpected error in gateway", error="Test error")
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_exception_paths(self) -> None:
        """Test cleanup method exception handling (lines 263-264)."""
        config = Mock()
        config.enable_monitoring = False

        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = True
        gateway.mqtt_client = AsyncMock()
        gateway.speedwire = AsyncMock()

        # Mock asyncio.gather to raise an exception
        with (
            patch("asyncio.gather", side_effect=OSError("Cleanup error")),
            patch("shelly_speedwire_gateway.gateway.logger") as mock_logger,
        ):
            await gateway.cleanup()

            # Verify exception was logged but cleanup continued (lines 263-264)
            mock_logger.warning.assert_called_with("Error during cleanup", error="Cleanup error")

    @pytest.mark.asyncio
    async def test_run_main_loops_with_pending_tasks(self) -> None:
        """Test _run_main_loops with pending tasks that need cancellation (lines 184-187)."""
        config = Mock()
        config.speedwire = Mock()
        config.speedwire.interval = 1.0

        gateway = Shelly3EMSpeedwireGateway.__new__(Shelly3EMSpeedwireGateway)
        gateway.config = config
        gateway.running = False
        gateway.mqtt_client = AsyncMock()
        gateway.speedwire = AsyncMock()

        # Create real async functions and tasks
        async def completed_coro() -> str:
            await asyncio.sleep(0.01)
            return "completed"

        async def pending_coro() -> str:
            await asyncio.sleep(10)  # Long sleep that will be cancelled
            return "pending"

        # Create real tasks
        completed_task = asyncio.create_task(completed_coro(), name="completed_task")
        pending_task1 = asyncio.create_task(pending_coro(), name="pending_task1")
        pending_task2 = asyncio.create_task(pending_coro(), name="pending_task2")

        # Mock asyncio.create_task to return our real tasks
        with (
            patch("asyncio.create_task", side_effect=[completed_task, pending_task1, pending_task2, completed_task]),
            patch("asyncio.wait") as mock_wait,
            patch("shelly_speedwire_gateway.gateway.logger") as mock_logger,
        ):
            # Wait a bit for completed task to finish
            await asyncio.sleep(0.02)

            # Simulate one task completed, two pending
            mock_wait.return_value = ([completed_task], [pending_task1, pending_task2])

            await gateway._run_main_loops()

            # Verify pending tasks were cancelled (lines 184-187)
            mock_logger.debug.assert_any_call("Cancelling task: pending_task1")
            mock_logger.debug.assert_any_call("Cancelling task: pending_task2")

            # Verify tasks are actually cancelled
            assert pending_task1.cancelled()
            assert pending_task2.cancelled()
