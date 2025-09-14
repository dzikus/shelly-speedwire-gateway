"""Test models.py with fixed magic values using constants."""

from __future__ import annotations

import time

import pytest
from pydantic_core import ValidationError

from shelly_speedwire_gateway.constants import (
    DEFAULT_METRICS_PORT,
    TEST_CONSUMED_TOTAL,
    TEST_CURRENT_VALUE,
    TEST_ENERGY_CONSUMED,
    TEST_ENERGY_EXPORTED,
    TEST_EXPORTED_TOTAL,
    TEST_FREQUENCY,
    TEST_KEEPALIVE,
    TEST_MQTT_PORT,
    TEST_PF_MAX,
    TEST_PF_MIN,
    TEST_PHASE_COUNT,
    TEST_POWER_A,
    TEST_POWER_B,
    TEST_POWER_C,
    TEST_POWER_FACTOR,
    TEST_POWER_VALUE,
    TEST_QOS,
    TEST_ROUNDING_PF,
    TEST_ROUNDING_POWER,
    TEST_ROUNDING_VOLTAGE,
    TEST_SERIAL,
    TEST_SERIAL_ALT,
    TEST_SUSY_ID,
    TEST_TOLERANCE,
    TEST_TOTAL_POWER,
    TEST_UNICAST_COUNT,
    TEST_VOLTAGE_VALUE,
)
from shelly_speedwire_gateway.models import (
    DeviceConfig,
    GatewaySettings,
    MQTTConnectionState,
    MQTTSettings,
    NetworkConfig,
    PhaseData,
    Shelly3EMData,
    SpeedwireSettings,
    create_3em_data_from_phases,
    create_phase_data_from_mqtt,
)


class TestPhaseData:
    """Test PhaseData model."""

    def test_create_phase_data(self) -> None:
        """Test creating phase data with valid values."""
        phase = PhaseData(
            power=TEST_POWER_VALUE,
            voltage=TEST_VOLTAGE_VALUE,
            current=TEST_CURRENT_VALUE,
            pf=TEST_POWER_FACTOR,
            energy_consumed=TEST_ENERGY_CONSUMED,
            energy_exported=TEST_ENERGY_EXPORTED,
        )

        assert phase.power == TEST_POWER_VALUE
        assert phase.voltage == TEST_VOLTAGE_VALUE
        assert phase.current == TEST_CURRENT_VALUE
        assert phase.pf == TEST_POWER_FACTOR
        assert phase.energy_consumed == TEST_ENERGY_CONSUMED
        assert phase.energy_exported == TEST_ENERGY_EXPORTED

    def test_computed_fields(self) -> None:
        """Test computed fields calculations."""
        phase = PhaseData(voltage=TEST_VOLTAGE_VALUE, current=TEST_CURRENT_VALUE, pf=TEST_POWER_FACTOR)

        # Apparent power = V * I
        expected_apparent = TEST_VOLTAGE_VALUE * TEST_CURRENT_VALUE
        assert abs(phase.apparent_power - expected_apparent) < TEST_TOLERANCE

        # Reactive power calculation
        expected_reactive = expected_apparent * (1.0 - TEST_POWER_FACTOR * TEST_POWER_FACTOR) ** 0.5
        assert abs(phase.reactive_power - expected_reactive) < TEST_TOLERANCE

    def test_power_factor_validation(self) -> None:
        """Test power factor validation."""
        with pytest.raises(ValidationError):
            PhaseData(pf=1.5)  # Invalid PF > 1.0

        with pytest.raises(ValidationError):
            PhaseData(pf=-1.5)  # Invalid PF < -1.0

    def test_serialization_rounding(self) -> None:
        """Test field serialization with rounding."""
        phase = PhaseData(
            power=TEST_ROUNDING_POWER,
            voltage=TEST_ROUNDING_VOLTAGE,
            pf=TEST_ROUNDING_PF,
        )

        data = phase.model_dump()

        # Check rounding
        assert data["power"] == TEST_ROUNDING_POWER
        assert data["voltage"] == TEST_ROUNDING_VOLTAGE
        assert data["pf"] == TEST_ROUNDING_PF


class TestShelly3EMData:
    """Test Shelly3EMData model."""

    def test_create_3em_data(self) -> None:
        """Test creating 3EM data."""
        data = Shelly3EMData(
            a=PhaseData(power=TEST_POWER_A),
            b=PhaseData(power=TEST_POWER_B),
            c=PhaseData(power=TEST_POWER_C),
            freq_hz=TEST_FREQUENCY,
        )

        assert data.a.power == TEST_POWER_A
        assert data.b.power == TEST_POWER_B
        assert data.c.power == TEST_POWER_C
        assert data.freq_hz == TEST_FREQUENCY

    def test_computed_totals(self) -> None:
        """Test computed total calculations."""
        data = Shelly3EMData(
            a=PhaseData(power=TEST_POWER_VALUE, energy_consumed=500.0, energy_exported=50.0),
            b=PhaseData(power=TEST_POWER_VALUE, energy_consumed=500.0, energy_exported=50.0),
            c=PhaseData(power=TEST_POWER_VALUE, energy_consumed=500.0, energy_exported=50.0),
        )

        assert data.total_power == TEST_TOTAL_POWER
        assert data.total_consumed_wh == TEST_CONSUMED_TOTAL
        assert data.total_exported_wh == TEST_EXPORTED_TOTAL

    def test_get_phase_by_index(self) -> None:
        """Test getting phase by index."""
        phase_a = PhaseData(power=100.0)
        phase_b = PhaseData(power=200.0)
        phase_c = PhaseData(power=300.0)

        data = Shelly3EMData(a=phase_a, b=phase_b, c=phase_c)

        assert data.get_phase(0) == phase_a
        assert data.get_phase(1) == phase_b
        assert data.get_phase(2) == phase_c
        # Invalid index returns default
        assert data.get_phase(3) == PhaseData()

    def test_get_phases_list(self) -> None:
        """Test getting phases as list."""
        phase_a = PhaseData(power=100.0)
        phase_b = PhaseData(power=200.0)
        phase_c = PhaseData(power=300.0)

        data = Shelly3EMData(a=phase_a, b=phase_b, c=phase_c)
        phases = data.get_phases_list()

        assert len(phases) == TEST_PHASE_COUNT
        assert phases[0] == phase_a
        assert phases[1] == phase_b
        assert phases[2] == phase_c

    def test_average_power_factor_calculation(self) -> None:
        """Test average power factor calculation."""
        # Different power factors and apparent powers
        data = Shelly3EMData(
            a=PhaseData(voltage=230.0, current=10.0, pf=0.9, power=2070.0),  # 230*10*0.9
            b=PhaseData(voltage=230.0, current=8.0, pf=0.85, power=1564.0),  # 230*8*0.85
            c=PhaseData(voltage=230.0, current=12.0, pf=0.95, power=2622.0),  # 230*12*0.95
        )

        # Should calculate weighted average based on apparent power
        avg_pf = data.average_power_factor
        assert TEST_PF_MIN < avg_pf < TEST_PF_MAX


class TestMQTTSettings:
    """Test MQTT settings model."""

    def test_default_mqtt_settings(self) -> None:
        """Test default MQTT settings."""
        settings = MQTTSettings()

        assert settings.broker_host == "localhost"
        assert settings.broker_port == TEST_MQTT_PORT
        assert settings.keepalive == TEST_KEEPALIVE
        assert settings.qos == TEST_QOS
        assert settings.invert_values is False


class TestSpeedwireSettings:
    """Test Speedwire settings model."""

    def test_default_speedwire_settings(self) -> None:
        """Test default Speedwire settings."""
        settings = SpeedwireSettings()

        assert settings.interval == 1.0
        assert settings.use_broadcast is False
        assert settings.dualcast is False
        assert settings.serial == TEST_SERIAL
        assert settings.susy_id == TEST_SUSY_ID

    def test_interval_validation(self) -> None:
        """Test interval validation."""
        with pytest.raises(ValidationError):
            SpeedwireSettings(interval=-1.0)  # Negative interval


class TestGatewaySettings:
    """Test Gateway settings model."""

    def test_default_gateway_settings(self) -> None:
        """Test default gateway settings."""
        # Test accepts both default values and config file values
        settings = GatewaySettings()

        assert settings.log_level in ("INFO", "DEBUG")  # Allow either default or config file value
        assert settings.log_format in ("structured", "console", "json")
        assert settings.enable_jit is True
        assert settings.enable_monitoring is False
        assert settings.metrics_port == DEFAULT_METRICS_PORT


class TestNetworkConfig:
    """Test NetworkConfig model."""

    def test_network_config_creation(self) -> None:
        """Test network configuration creation."""
        config = NetworkConfig(
            use_broadcast=True,
            local_ip="192.168.1.100",
            unicast_targets=["192.168.1.101", "192.168.1.102"],
        )

        assert config.use_broadcast is True
        assert config.local_ip == "192.168.1.100"
        assert len(config.unicast_targets) == TEST_UNICAST_COUNT

    def test_default_local_ip(self) -> None:
        """Test default local IP."""
        config = NetworkConfig()
        assert config.local_ip == "0.0.0.0"

    def test_ip_validation(self) -> None:
        """Test IP address validation."""
        with pytest.raises(ValidationError):
            NetworkConfig(local_ip="invalid.ip")

        with pytest.raises(ValidationError):
            NetworkConfig(unicast_targets=["invalid.ip"])


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_phase_data_from_mqtt(self) -> None:
        """Test creating phase data from MQTT values."""
        phase = create_phase_data_from_mqtt(
            power=TEST_POWER_VALUE,
            voltage=TEST_VOLTAGE_VALUE,
            current=TEST_CURRENT_VALUE,
            power_factor=TEST_POWER_FACTOR,
        )

        assert phase.power == TEST_POWER_VALUE
        assert phase.voltage == TEST_VOLTAGE_VALUE
        assert phase.current == TEST_CURRENT_VALUE
        assert phase.pf == TEST_POWER_FACTOR

    def test_create_3em_data_from_phases(self) -> None:
        """Test creating 3EM data from phases."""
        phase_a = PhaseData(power=100.0)
        phase_b = PhaseData(power=200.0)
        phase_c = PhaseData(power=300.0)

        data = create_3em_data_from_phases(phase_a, phase_b, phase_c, frequency=TEST_FREQUENCY, device_id="test123")

        assert data.a == phase_a
        assert data.b == phase_b
        assert data.c == phase_c
        assert data.freq_hz == TEST_FREQUENCY
        assert data.device_id == "test123"


class TestCompleteConfiguration:
    """Test complete configuration validation."""

    def test_complete_gateway_configuration(self) -> None:
        """Test complete gateway configuration."""
        settings = GatewaySettings(
            mqtt=MQTTSettings(broker_host="mqtt.example.com", base_topic="shellies/shellyem3-test"),
            speedwire=SpeedwireSettings(serial=TEST_SERIAL_ALT),
            log_level="DEBUG",
            enable_jit=True,
        )

        assert settings.mqtt.broker_host == "mqtt.example.com"
        assert settings.mqtt.base_topic == "shellies/shellyem3-test"
        assert settings.speedwire.serial == TEST_SERIAL_ALT
        assert settings.log_level == "DEBUG"
        assert settings.enable_jit is True


class TestValidatorsAndComputedFields:
    """Test model validators and computed fields."""

    def test_power_factor_validator_invalid(self) -> None:
        """Test power factor validation with constraint check."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseData(
                voltage=230.0,
                current=5.0,
                power=1000.0,
                pf=1.5,  # Invalid: > 1.0
                energy_consumed=100.0,
                energy_exported=50.0,
            )
        assert "less than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            PhaseData(
                voltage=230.0,
                current=5.0,
                power=1000.0,
                pf=-1.5,  # Invalid: < -1.0
                energy_consumed=100.0,
                energy_exported=50.0,
            )
        assert "greater than or equal to -1" in str(exc_info.value)

    def test_power_factor_validator_edge_case(self) -> None:
        """Test power factor validator with edge values."""
        # Test exact boundary values which should pass constraints but could trigger validator
        phase = PhaseData(
            voltage=230.0,
            current=5.0,
            power=1000.0,
            pf=-1.0,  # Exact boundary
            energy_consumed=100.0,
            energy_exported=50.0,
        )
        assert phase.pf == -1.0

        phase = PhaseData(
            voltage=230.0,
            current=5.0,
            power=1000.0,
            pf=1.0,  # Exact boundary
            energy_consumed=100.0,
            energy_exported=50.0,
        )
        assert phase.pf == 1.0

    def test_computed_fields_shelly3em_data(self) -> None:
        """Test computed fields in Shelly3EMData."""
        phase_a = PhaseData(
            voltage=230.0,
            current=5.0,
            power=1000.0,
            pf=0.9,
            energy_consumed=100.0,
            energy_exported=50.0,
        )
        phase_b = PhaseData(
            voltage=232.0,
            current=4.8,
            power=900.0,
            pf=0.85,
            energy_consumed=120.0,
            energy_exported=60.0,
        )
        phase_c = PhaseData(
            voltage=228.0,
            current=5.2,
            power=1100.0,
            pf=0.95,
            energy_consumed=110.0,
            energy_exported=55.0,
        )

        data = Shelly3EMData(
            a=phase_a,
            b=phase_b,
            c=phase_c,
        )

        # Test computed fields
        expected_reactive_a = 230.0 * 5.0 * (1.0 - 0.9**2) ** 0.5
        expected_reactive_b = 232.0 * 4.8 * (1.0 - 0.85**2) ** 0.5
        expected_reactive_c = 228.0 * 5.2 * (1.0 - 0.95**2) ** 0.5

        assert data.total_reactive_power == pytest.approx(
            expected_reactive_a + expected_reactive_b + expected_reactive_c,
            rel=1e-3,
        )
        assert data.total_apparent_power > 0

        # Test power factor calculation
        assert data.average_power_factor == pytest.approx(3000.0 / data.total_apparent_power, rel=1e-3)

    def test_zero_apparent_power_edge_case(self) -> None:
        """Test edge case when total apparent power is zero."""
        zero_phase = PhaseData(
            voltage=0.0,
            current=0.0,
            power=0.0,
            pf=1.0,
            energy_consumed=0.0,
            energy_exported=0.0,
        )

        data = Shelly3EMData(
            a=zero_phase,
            b=zero_phase,
            c=zero_phase,
        )

        # When total_apparent_power is 0, power factor should default to 1.0
        assert data.total_apparent_power == 0.0
        assert data.average_power_factor == 1.0

    def test_ip_address_validator_invalid(self) -> None:
        """Test IP address validator with invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            NetworkConfig(local_ip="999.999.999.999")  # Invalid IP (passes regex, fails socket check)
        assert "Invalid IP address" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            NetworkConfig(local_ip="not.an.ip")  # Invalid format (fails regex)
        assert "String should match pattern" in str(exc_info.value)

    def test_ip_address_validator_valid(self) -> None:
        """Test IP address validator with valid values."""
        # Test default value
        config = NetworkConfig()
        assert config.local_ip == "0.0.0.0"

        # Test localhost
        config = NetworkConfig(local_ip="127.0.0.1")
        assert config.local_ip == "127.0.0.1"

        # Test regular IP
        config = NetworkConfig(local_ip="192.168.1.100")
        assert config.local_ip == "192.168.1.100"

    def test_phase_data_computed_fields(self) -> None:
        """Test computed fields in PhaseData."""
        phase = PhaseData(
            voltage=230.0,
            current=5.0,
            power=1000.0,
            pf=0.9,
            energy_consumed=100.0,
            energy_exported=50.0,
        )

        # Test computed reactive power
        expected_apparent = 230.0 * 5.0
        expected_reactive = expected_apparent * (1.0 - 0.9**2) ** 0.5

        assert phase.apparent_power == pytest.approx(expected_apparent)
        assert phase.reactive_power == pytest.approx(expected_reactive, rel=1e-3)


class TestValidationEdgeCases:
    """Test validation edge cases and error handling."""

    def test_power_factor_validator_with_extreme_values(self) -> None:
        """Test power factor validation with extreme values."""

        validator = PhaseData.validate_power_factor

        with pytest.raises(ValueError, match="Power factor must be between -1.0 and 1.0, got 2.0"):
            validator(2.0)

        with pytest.raises(ValueError, match="Power factor must be between -1.0 and 1.0, got -2.0"):
            validator(-2.0)

    def test_network_config_invalid_unicast_targets(self) -> None:
        """Test NetworkConfig validation with invalid IP addresses."""

        with pytest.raises(ValidationError) as exc_info:
            NetworkConfig(unicast_targets=["999.999.999.999"])

        error_msg = str(exc_info.value)
        assert "Invalid IP address" in error_msg

    def test_device_config_software_version_edge_cases(self) -> None:
        """Test DeviceConfig software version validation edge cases."""

        validator = DeviceConfig.validate_software_version

        # Wrong number of components
        with pytest.raises(ValueError, match="Software version must have 4 components"):
            validator((1, 2, 3))  # type: ignore[arg-type]

        # Non-integer components
        class FakeInt:
            """Mock class that looks like int but isn't."""

            def __init__(self, value: int) -> None:
                self.value = value

        fake_tuple = (FakeInt(1), 2, 3, "R")
        with pytest.raises(ValueError, match="First three version components must be integers"):
            validator(fake_tuple)  # type: ignore[arg-type]

        # Invalid revision formats
        with pytest.raises(ValueError, match="Revision must be a single character"):
            validator((1, 2, 3, "ABC"))

        with pytest.raises(ValueError, match="Revision must be a single character"):
            validator((1, 2, 3, 123))  # type: ignore[arg-type]

        # Valid case
        result = validator((1, 2, 3, "R"))
        assert result == (1, 2, 3, "R")


class TestMQTTConnectionState:
    """Test MQTTConnectionState behavior."""

    def test_connection_state_lifecycle(self) -> None:
        """Test connection state transitions and computed properties."""

        state = MQTTConnectionState()

        # Initially disconnected
        assert state.connection_duration == 0.0
        assert state.messages_per_second == 0.0
        assert state.time_since_last_message >= 0

        # Reset connection
        state.connected = True
        state.reconnect_attempts = 5
        state.reset_connection()
        assert state.connected is False
        assert state.reconnect_attempts == 0

        # Mark as connected
        state.mark_connected()
        assert state.connected is True
        assert state.reconnect_attempts == 0  # type: ignore[unreachable]

        # Update message metrics
        old_count = state.total_messages_received
        state.update_last_message_time()
        assert state.total_messages_received == old_count + 1

        # Test message rate calculation
        time.sleep(0.01)
        assert state.messages_per_second > 0


class TestMQTTSettingsComputedFields:
    """Test MQTTSettings computed field behavior."""

    def test_device_id_extraction(self) -> None:
        """Test device ID extraction from base topic."""

        # With device ID in topic
        settings = MQTTSettings(base_topic="shellies/shelly3em-ABCDEF123456")
        assert settings.device_id == "ABCDEF123456"

        # Without device ID
        settings = MQTTSettings(base_topic="nodash")
        assert settings.device_id == "default"

    def test_connection_url_generation(self) -> None:
        """Test MQTT connection URL generation."""

        # With credentials
        settings = MQTTSettings(
            broker_host="mqtt.example.com",
            broker_port=1883,
            username="user123",
            password="pass123",
        )
        expected_url = "mqtt://user123:pass123@mqtt.example.com:1883"
        assert settings.connection_url == expected_url

        # Without credentials
        settings = MQTTSettings(
            broker_host="mqtt.example.com",
            broker_port=1883,
        )
        expected_url = "mqtt://mqtt.example.com:1883"
        assert settings.connection_url == expected_url
