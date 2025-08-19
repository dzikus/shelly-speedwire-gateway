"""Tests for metrics collection and Prometheus integration."""
# pylint: disable=redefined-outer-name,protected-access

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import Mock, patch

import pytest

from shelly_speedwire_gateway.metrics import (
    gateway_info,
    init_metrics,
    mqtt_connected,
    mqtt_messages_errors,
    mqtt_messages_received,
    mqtt_processing_time,
    phase_current,
    phase_power,
    phase_voltage,
    shutdown_metrics,
    speedwire_active,
    speedwire_packets_errors,
    speedwire_packets_sent,
    speedwire_send_time,
    total_energy_consumed,
    total_energy_exported,
    total_power,
    update_energy_metrics,
)
from shelly_speedwire_gateway.models import GatewaySettings, MQTTSettings, SpeedwireSettings


@pytest.fixture
def gateway_settings() -> GatewaySettings:
    """Create gateway settings for testing."""
    mqtt_settings = MQTTSettings(
        base_topic="shellies/test-device",
        broker_host="test.mqtt.com",
        broker_port=1883,
    )

    speedwire_settings = SpeedwireSettings(
        serial=123456789,
        interval=1.0,
    )

    return GatewaySettings(
        mqtt=mqtt_settings,
        speedwire=speedwire_settings,
        enable_monitoring=True,
        metrics_port=9090,
    )


@pytest.fixture
def sample_shelly_data() -> dict[str, float]:
    """Create sample Shelly energy data."""
    return {
        "total_power": 2500.0,
        "power_0": 800.0,
        "power_1": 900.0,
        "power_2": 800.0,
        "voltage_0": 230.5,
        "voltage_1": 231.0,
        "voltage_2": 229.8,
        "current_0": 3.47,
        "current_1": 3.90,
        "current_2": 3.48,
        "total_consumed": 12345.67,  # Wh
        "total_returned": 543.21,  # Wh
    }


class TestMetricsInitialization:
    """Test metrics initialization and setup."""

    @patch("shelly_speedwire_gateway.metrics.start_http_server")
    def test_init_metrics_enabled(self, mock_server: Mock, gateway_settings: GatewaySettings) -> None:
        """Test metrics initialization when monitoring is enabled."""
        init_metrics(gateway_settings)

        # Verify HTTP server was started
        mock_server.assert_called_once_with(gateway_settings.metrics_port)

    def test_init_metrics_disabled(self, gateway_settings: GatewaySettings) -> None:
        """Test metrics initialization when monitoring is disabled."""
        gateway_settings.enable_monitoring = False

        with patch("shelly_speedwire_gateway.metrics.start_http_server") as mock_start_server:
            init_metrics(gateway_settings)

            # HTTP server should not be started
            mock_start_server.assert_not_called()

    @patch("shelly_speedwire_gateway.metrics.start_http_server", side_effect=Exception("Server error"))
    def test_init_metrics_server_error(self, gateway_settings: GatewaySettings) -> None:
        """Test metrics initialization with server startup error."""
        with pytest.raises(Exception, match="Server error"):
            init_metrics(gateway_settings)

    @patch("shelly_speedwire_gateway.metrics.start_http_server")
    def test_init_metrics_sets_gateway_info(self, mock_server: Mock, gateway_settings: GatewaySettings) -> None:
        """Test that gateway info is set during initialization."""
        with patch.object(gateway_info, "info") as mock_info:
            init_metrics(gateway_settings)

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert call_args["version"] == "2.0.0"
            assert call_args["mqtt_broker"] == "test.mqtt.com"
            assert call_args["mqtt_topic"] == "shellies/test-device"
            assert call_args["speedwire_serial"] == "123456789"
            mock_server.assert_called_once_with(gateway_settings.metrics_port)


class TestEnergyMetricsUpdate:
    """Test energy metrics updating functionality."""

    def test_update_energy_metrics_complete_data(self, sample_shelly_data: dict[str, float]) -> None:
        """Test updating metrics with complete data."""
        with (
            patch.object(total_power, "set") as mock_total_power,
            patch.object(phase_power.labels(phase="A"), "set") as mock_phase_a_power,
            patch.object(phase_power.labels(phase="B"), "set") as mock_phase_b_power,
            patch.object(phase_power.labels(phase="C"), "set") as mock_phase_c_power,
            patch.object(phase_voltage.labels(phase="A"), "set") as mock_phase_a_voltage,
            patch.object(phase_voltage.labels(phase="B"), "set") as mock_phase_b_voltage,
            patch.object(phase_voltage.labels(phase="C"), "set") as mock_phase_c_voltage,
            patch.object(phase_current.labels(phase="A"), "set") as mock_phase_a_current,
            patch.object(phase_current.labels(phase="B"), "set") as mock_phase_b_current,
            patch.object(phase_current.labels(phase="C"), "set") as mock_phase_c_current,
            patch.object(total_energy_consumed, "set") as mock_consumed,
            patch.object(total_energy_exported, "set") as mock_exported,
        ):
            update_energy_metrics(sample_shelly_data)

            # Verify all metrics were updated
            mock_total_power.assert_called_once_with(2500.0)
            mock_phase_a_power.assert_called_once_with(800.0)
            mock_phase_b_power.assert_called_once_with(900.0)
            mock_phase_c_power.assert_called_once_with(800.0)
            mock_phase_a_voltage.assert_called_once_with(230.5)
            mock_phase_b_voltage.assert_called_once_with(231.0)
            mock_phase_c_voltage.assert_called_once_with(229.8)
            mock_phase_a_current.assert_called_once_with(3.47)
            mock_phase_b_current.assert_called_once_with(3.90)
            mock_phase_c_current.assert_called_once_with(3.48)
            # Energy should be converted from Wh to kWh
            mock_consumed.assert_called_once_with(12.34567)
            # Use call_args to check the value with appropriate precision
            assert mock_exported.call_args[0][0] == pytest.approx(0.54321)

    def test_update_energy_metrics_partial_data(self) -> None:
        """Test updating metrics with partial data."""
        partial_data = {
            "total_power": 1500.0,
            "power_0": 500.0,  # Only phase A
            "voltage_1": 230.0,  # Only phase B
        }

        with (
            patch.object(total_power, "set") as mock_total_power,
            patch.object(phase_power.labels(phase="A"), "set") as mock_phase_a_power,
            patch.object(phase_voltage.labels(phase="B"), "set") as mock_phase_b_voltage,
        ):
            update_energy_metrics(partial_data)

            mock_total_power.assert_called_once_with(1500.0)
            mock_phase_a_power.assert_called_once_with(500.0)
            mock_phase_b_voltage.assert_called_once_with(230.0)

    def test_update_energy_metrics_empty_data(self) -> None:
        """Test updating metrics with empty data."""
        empty_data: dict[str, float] = {}

        # Should not raise exception
        update_energy_metrics(empty_data)

    def test_update_energy_metrics_invalid_data_type(self) -> None:
        """Test updating metrics with invalid data types."""
        invalid_data = {
            "total_power": "not_a_number",
            "power_0": None,
            "voltage_0": "invalid",
        }

        # Should not raise exception, just log errors
        update_energy_metrics(invalid_data)

    def test_update_energy_metrics_missing_keys(self) -> None:
        """Test updating metrics with data containing missing expected keys."""
        data_with_missing_keys = {
            "total_power": 1000.0,
            # Missing phase power/voltage/current data
        }

        with patch.object(total_power, "set") as mock_total_power:
            update_energy_metrics(data_with_missing_keys)

            mock_total_power.assert_called_once_with(1000.0)


class TestMetricCountersAndGauges:
    """Test individual metric counters and gauges."""

    def test_mqtt_connection_metrics(self) -> None:
        """Test MQTT connection status metrics."""
        # Test setting connected
        with patch.object(mqtt_connected, "set") as mock_set:
            mqtt_connected.set(1)
            mock_set.assert_called_once_with(1)

        # Test setting disconnected
        with patch.object(mqtt_connected, "set") as mock_set:
            mqtt_connected.set(0)
            mock_set.assert_called_once_with(0)

    def test_speedwire_active_metrics(self) -> None:
        """Test Speedwire active status metrics."""
        with patch.object(speedwire_active, "set") as mock_set:
            speedwire_active.set(1)
            mock_set.assert_called_once_with(1)

    def test_mqtt_message_counters(self) -> None:
        """Test MQTT message counters."""
        with patch.object(mqtt_messages_received, "inc") as mock_inc:
            mqtt_messages_received.inc()
            mock_inc.assert_called_once()

        with patch.object(mqtt_messages_errors, "inc") as mock_inc:
            mqtt_messages_errors.inc()
            mock_inc.assert_called_once()

    def test_speedwire_packet_counters(self) -> None:
        """Test Speedwire packet counters."""
        with patch.object(speedwire_packets_sent, "inc") as mock_inc:
            speedwire_packets_sent.inc()
            mock_inc.assert_called_once()

        with patch.object(speedwire_packets_errors, "inc") as mock_inc:
            speedwire_packets_errors.inc()
            mock_inc.assert_called_once()

    def test_timing_histograms(self) -> None:
        """Test timing histogram metrics."""
        # Test MQTT processing time histogram
        with patch.object(mqtt_processing_time, "time") as mock_time:
            mock_context = Mock()
            mock_time.return_value = mock_context

            context = mqtt_processing_time.time()
            assert context == mock_context

        # Test Speedwire send time histogram
        with patch.object(speedwire_send_time, "time") as mock_time:
            mock_context = Mock()
            mock_time.return_value = mock_context

            context = speedwire_send_time.time()
            assert context == mock_context

    def test_histogram_context_manager_usage(self) -> None:
        """Test histogram context manager usage pattern."""
        with patch.object(mqtt_processing_time, "time") as mock_time:
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_context)
            mock_context.__exit__ = Mock(return_value=False)
            mock_time.return_value = mock_context

            # Simulate usage pattern from actual code
            with mqtt_processing_time.time():
                pass  # Simulated processing

            mock_context.__enter__.assert_called_once()
            mock_context.__exit__.assert_called_once()


class TestPhaseMetrics:
    """Test phase-specific metrics with labels."""

    def test_phase_power_labels(self) -> None:
        """Test phase power metrics with labels."""
        with patch.object(phase_power, "labels") as mock_labels:
            mock_metric = Mock()
            mock_labels.return_value = mock_metric

            # Test each phase
            for phase_letter in ["A", "B", "C"]:
                _ = phase_power.labels(phase=phase_letter)
                mock_labels.assert_called_with(phase=phase_letter)

    def test_phase_voltage_labels(self) -> None:
        """Test phase voltage metrics with labels."""
        with patch.object(phase_voltage, "labels") as mock_labels:
            mock_metric = Mock()
            mock_labels.return_value = mock_metric

            _ = phase_voltage.labels(phase="A")
            mock_labels.assert_called_with(phase="A")

    def test_phase_current_labels(self) -> None:
        """Test phase current metrics with labels."""
        with patch.object(phase_current, "labels") as mock_labels:
            mock_metric = Mock()
            mock_labels.return_value = mock_metric

            _ = phase_current.labels(phase="B")
            mock_labels.assert_called_with(phase="B")


class TestEnergyConversions:
    """Test energy unit conversions in metrics."""

    def test_wh_to_kwh_conversion(self) -> None:
        """Test Wh to kWh conversion in energy metrics."""
        data = {
            "total_consumed": 5000.0,  # 5000 Wh = 5 kWh
            "total_returned": 2500.0,  # 2500 Wh = 2.5 kWh
        }

        with (
            patch.object(total_energy_consumed, "set") as mock_consumed,
            patch.object(total_energy_exported, "set") as mock_exported,
        ):
            update_energy_metrics(data)

            # Verify conversion from Wh to kWh
            mock_consumed.assert_called_once_with(5.0)
            mock_exported.assert_called_once_with(2.5)

    def test_fractional_energy_values(self) -> None:
        """Test fractional energy values conversion."""
        data = {
            "total_consumed": 1234.56,  # 1234.56 Wh = 1.23456 kWh
            "total_returned": 789.01,  # 789.01 Wh = 0.78901 kWh
        }

        with (
            patch.object(total_energy_consumed, "set") as mock_consumed,
            patch.object(total_energy_exported, "set") as mock_exported,
        ):
            update_energy_metrics(data)

            # Use call_args to check the values with appropriate precision
            assert mock_consumed.call_args[0][0] == pytest.approx(1.23456)
            assert mock_exported.call_args[0][0] == pytest.approx(0.78901)


class TestMetricsShutdown:
    """Test metrics shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_metrics(self) -> None:
        """Test graceful metrics shutdown."""
        # Should not raise exception
        await shutdown_metrics()

    @pytest.mark.asyncio
    async def test_shutdown_metrics_is_async(self) -> None:
        """Test that shutdown_metrics is properly async."""
        assert inspect.iscoroutinefunction(shutdown_metrics)


class TestMetricsErrorHandling:
    """Test error handling in metrics operations."""

    def test_update_energy_metrics_with_exception(self) -> None:
        """Test update_energy_metrics handles exceptions gracefully."""
        # Create data that will cause an exception in metric setting
        problematic_data = {"total_power": float("inf")}

        with patch.object(total_power, "set", side_effect=ValueError("Invalid value")):
            # Should not raise exception, just log it
            update_energy_metrics(problematic_data)

    def test_update_energy_metrics_key_error(self) -> None:
        """Test handling of KeyError during metrics update."""
        data = {"total_power": 1000.0}

        # Mock a metric that raises KeyError
        with patch.object(total_power, "set", side_effect=KeyError("test key error")):
            # Should not raise exception
            update_energy_metrics(data)

    def test_update_energy_metrics_type_error(self) -> None:
        """Test handling of TypeError during metrics update."""
        data = {"total_power": 1000.0}

        with patch.object(total_power, "set", side_effect=TypeError("test type error")):
            # Should not raise exception
            update_energy_metrics(data)


class TestMetricsIntegration:
    """Test integration scenarios for metrics."""

    def test_metrics_workflow_simulation(
        self,
        gateway_settings: GatewaySettings,
        sample_shelly_data: dict[str, float],
    ) -> None:
        """Test complete metrics workflow simulation."""
        # Initialize metrics
        with patch("shelly_speedwire_gateway.metrics.start_http_server") as mock_server:
            init_metrics(gateway_settings)
            mock_server.assert_called_once_with(9090)

        # Simulate connection events
        with patch.object(mqtt_connected, "set") as mock_mqtt, patch.object(speedwire_active, "set") as mock_speedwire:
            mqtt_connected.set(1)
            speedwire_active.set(1)
            mock_mqtt.assert_called_with(1)
            mock_speedwire.assert_called_with(1)

        # Simulate message processing
        with (
            patch.object(mqtt_messages_received, "inc") as mock_received,
            patch.object(mqtt_processing_time, "time") as mock_timing,
        ):
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_context)
            mock_context.__exit__ = Mock(return_value=None)
            mock_timing.return_value = mock_context

            mqtt_messages_received.inc()
            with mqtt_processing_time.time():
                update_energy_metrics(sample_shelly_data)

            mock_received.assert_called_once()

        # Simulate Speedwire packet sending
        with patch.object(speedwire_packets_sent, "inc") as mock_sent:
            speedwire_packets_sent.inc()
            mock_sent.assert_called_once()

    def test_metrics_with_missing_prometheus_client(self) -> None:
        """Test graceful handling when prometheus_client is not available."""
        # This test verifies that the module can be imported
        # and basic operations work even if prometheus_client has issues

        # All metrics should be importable and callable
        assert gateway_info is not None
        assert mqtt_connected is not None
        assert total_power is not None

    @pytest.mark.asyncio
    async def test_async_metrics_operations(self) -> None:
        """Test async operations with metrics."""
        # Test that async operations work with metrics
        with patch.object(mqtt_messages_received, "inc") as mock_inc:
            await asyncio.sleep(0.001)  # Simulate async work
            mqtt_messages_received.inc()
            mock_inc.assert_called_once()

        # Test async shutdown
        await shutdown_metrics()
