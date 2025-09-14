"""Tests for custom exceptions and error handling."""

from __future__ import annotations

from shelly_speedwire_gateway.exceptions import (
    ConfigurationError,
    DataValidationError,
    DeviceError,
    GatewayError,
    GatewayTimeoutError,
    MQTTConnectionError,
    NetworkError,
    ProtocolError,
    ShutdownRequestedError,
)


class TestGatewayError:
    """Test base GatewayError exception."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = GatewayError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}

    def test_error_with_error_code(self) -> None:
        """Test error with error code."""
        error = GatewayError("Test error", error_code=500)

        assert str(error) == "[500] Test error"
        assert error.message == "Test error"
        assert error.error_code == 500

    def test_error_with_context(self) -> None:
        """Test error with additional context."""
        context = {"device_id": "test-device", "port": 1883}
        error = GatewayError("Connection failed", context=context)

        assert str(error) == "Connection failed"
        assert error.message == "Connection failed"
        assert error.context == context

    def test_error_inheritance(self) -> None:
        """Test that GatewayError inherits from Exception."""
        error = GatewayError("Test error")

        assert isinstance(error, Exception)
        assert isinstance(error, GatewayError)

    def test_error_repr(self) -> None:
        """Test error string representation."""
        error = GatewayError("Test message", error_code=404, context={"test": "value"})
        repr_str = repr(error)

        assert "GatewayError" in repr_str
        assert "Test message" in repr_str
        assert "404" in repr_str


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_basic_config_error(self) -> None:
        """Test basic configuration error."""
        error = ConfigurationError("Invalid configuration")

        assert str(error) == "Invalid configuration"
        assert isinstance(error, GatewayError)

    def test_config_error_with_section_and_key(self) -> None:
        """Test configuration error with section and key."""
        error = ConfigurationError(
            "Invalid MQTT settings",
            config_section="mqtt",
            config_key="broker_host",
        )

        assert error.config_section == "mqtt"
        assert error.config_key == "broker_host"
        assert error.context["config_section"] == "mqtt"
        assert error.context["config_key"] == "broker_host"


class TestMQTTConnectionError:
    """Test MQTT connection error."""

    def test_basic_mqtt_error(self) -> None:
        """Test basic MQTT connection error."""
        error = MQTTConnectionError("Connection refused")

        assert str(error) == "Connection refused"
        assert isinstance(error, GatewayError)

    def test_mqtt_error_with_broker_info(self) -> None:
        """Test MQTT error with broker information."""
        error = MQTTConnectionError(
            "Connection timeout",
            broker_host="mqtt.example.com",
            broker_port=8883,
        )

        assert error.broker_host == "mqtt.example.com"
        assert error.broker_port == 8883
        assert error.context["broker_host"] == "mqtt.example.com"
        assert error.context["broker_port"] == 8883

    def test_mqtt_error_with_reconnect_attempts(self) -> None:
        """Test MQTT error with reconnection attempts."""
        error = MQTTConnectionError(
            "Max retries exceeded",
            broker_host="localhost",
            broker_port=1883,
            reconnect_attempts=5,
        )

        assert error.reconnect_attempts == 5
        assert error.broker_host == "localhost"
        assert error.broker_port == 1883
        assert error.context["reconnect_attempts"] == 5


class TestNetworkError:
    """Test NetworkError exception."""

    def test_basic_network_error(self) -> None:
        """Test basic network error."""
        error = NetworkError("Network unreachable")

        assert str(error) == "Network unreachable"
        assert isinstance(error, GatewayError)

    def test_network_error_with_details(self) -> None:
        """Test network error with interface and port."""
        error = NetworkError(
            "Connection refused",
            interface="192.168.1.1",
            port=9522,
            operation="bind",
        )

        assert error.interface == "192.168.1.1"
        assert error.port == 9522
        assert error.operation == "bind"
        assert error.context["interface"] == "192.168.1.1"
        assert error.context["port"] == 9522
        assert error.context["operation"] == "bind"


class TestProtocolError:
    """Test ProtocolError exception."""

    def test_basic_protocol_error(self) -> None:
        """Test basic protocol error."""
        error = ProtocolError("Invalid packet format")

        assert str(error) == "Invalid packet format"
        assert isinstance(error, GatewayError)

    def test_protocol_error_with_packet_data(self) -> None:
        """Test protocol error with packet data."""
        packet_data = b"\x53\x4d\x41\x00" + b"\x00" * 20
        error = ProtocolError(
            "Invalid speedwire packet",
            protocol="speedwire",
            packet_data=packet_data,
            expected_format="SMA speedwire v1.0",
        )

        assert error.protocol == "speedwire"
        assert error.packet_data == packet_data
        assert error.expected_format == "SMA speedwire v1.0"
        assert error.context["protocol"] == "speedwire"
        assert error.context["packet_length"] == len(packet_data)
        assert "packet_header" in error.context

    def test_protocol_error_short_packet(self) -> None:
        """Test protocol error with short packet data."""
        packet_data = b"\x53\x4d\x41"  # Less than 16 bytes
        error = ProtocolError("Short packet", packet_data=packet_data)

        assert error.context["packet_length"] == 3
        assert error.context["packet_header"] == "534d41"  # Full packet as hex


class TestDataValidationError:
    """Test DataValidationError exception."""

    def test_basic_validation_error(self) -> None:
        """Test basic data validation error."""
        error = DataValidationError("Invalid data format")

        assert str(error) == "Invalid data format"
        assert isinstance(error, GatewayError)

    def test_validation_error_with_field_info(self) -> None:
        """Test validation error with field information."""
        error = DataValidationError(
            "Value out of range",
            field_name="voltage",
            field_value=500.0,
            valid_range="0-400V",
        )

        assert error.field_name == "voltage"
        assert error.field_value == 500.0
        assert error.valid_range == "0-400V"
        assert error.context["field_name"] == "voltage"
        assert error.context["field_value"] == 500.0
        assert error.context["valid_range"] == "0-400V"


class TestDeviceError:
    """Test DeviceError exception."""

    def test_basic_device_error(self) -> None:
        """Test basic device error."""
        error = DeviceError("Device not responding")

        assert str(error) == "Device not responding"
        assert isinstance(error, GatewayError)

    def test_device_error_with_details(self) -> None:
        """Test device error with device information."""
        error = DeviceError(
            "Communication timeout",
            device_id="shelly3em-001",
            device_type="shelly3em",
            last_seen=1234567890.0,
        )

        assert error.device_id == "shelly3em-001"
        assert error.device_type == "shelly3em"
        assert error.last_seen == 1234567890.0
        assert error.context["device_id"] == "shelly3em-001"
        assert error.context["device_type"] == "shelly3em"
        assert error.context["last_seen"] == 1234567890.0


class TestGatewayTimeoutError:
    """Test GatewayTimeoutError exception."""

    def test_basic_timeout_error(self) -> None:
        """Test basic timeout error."""
        error = GatewayTimeoutError("Operation timed out")

        assert str(error) == "Operation timed out"
        assert isinstance(error, GatewayError)

    def test_timeout_error_with_duration(self) -> None:
        """Test timeout error with duration information."""
        error = GatewayTimeoutError(
            "Connection timeout",
            timeout_seconds=30.0,
            operation="mqtt_connect",
        )

        assert error.timeout_seconds == 30.0
        assert error.operation == "mqtt_connect"
        assert error.context["timeout_seconds"] == 30.0
        assert error.context["operation"] == "mqtt_connect"


class TestShutdownRequestedError:
    """Test ShutdownRequestedError exception."""

    def test_basic_shutdown_error(self) -> None:
        """Test basic shutdown request."""
        error = ShutdownRequestedError()

        assert str(error) == "Shutdown requested"
        assert error.reason == "Shutdown requested"
        assert error.initiated_by is None

    def test_shutdown_error_with_details(self) -> None:
        """Test shutdown request with details."""
        error = ShutdownRequestedError("User requested shutdown", "main_thread")

        assert str(error) == "User requested shutdown (initiated by main_thread)"
        assert error.reason == "User requested shutdown"
        assert error.initiated_by == "main_thread"

    def test_shutdown_error_inheritance(self) -> None:
        """Test that ShutdownRequestedError inherits from Exception but not GatewayError."""
        error = ShutdownRequestedError()

        assert isinstance(error, Exception)
        assert not isinstance(error, GatewayError)


class TestExceptionChaining:
    """Test exception chaining and context preservation."""

    def test_exception_chaining(self) -> None:
        """Test proper exception chaining."""
        # Create chained exception and verify structure
        msg = "Original error"
        try:
            raise ValueError(msg)
        except ValueError as e:
            chained_error = ConfigurationError("Config error occurred")
            chained_error.__cause__ = e

        # Verify the exception chain
        assert chained_error.__cause__ is not None
        assert isinstance(chained_error.__cause__, ValueError)
        assert str(chained_error.__cause__) == "Original error"

    def test_nested_exception_context(self) -> None:
        """Test nested exception context preservation."""
        original_error = ConnectionError("Network down")
        mqtt_error = MQTTConnectionError(
            "MQTT connection failed",
            broker_host="localhost",
        )
        mqtt_error.__cause__ = original_error

        gateway_error = GatewayError("Gateway initialization failed")
        gateway_error.__cause__ = mqtt_error

        # Verify the chain is preserved
        assert gateway_error.__cause__ == mqtt_error
        assert gateway_error.__cause__.__cause__ == original_error

    def test_context_propagation(self) -> None:
        """Test that context is properly propagated."""
        network_err = NetworkError(
            "UDP socket error",
            interface="239.12.255.254",
            port=9522,
            operation="speedwire_broadcast",
        )

        gateway_err = GatewayError(
            "Speedwire initialization failed",
            context={"component": "speedwire_emulator"},
        )
        gateway_err.__cause__ = network_err

        # Both errors should maintain their context
        assert network_err.interface == "239.12.255.254"
        assert network_err.port == 9522
        assert network_err.operation == "speedwire_broadcast"
        assert gateway_err.context["component"] == "speedwire_emulator"


class TestExceptionErrorCodes:
    """Test error codes in exceptions."""

    def test_error_code_formatting(self) -> None:
        """Test error code formatting in string representation."""
        error = GatewayError("Test error", error_code=404)
        assert str(error) == "[404] Test error"

    def test_no_error_code_formatting(self) -> None:
        """Test string representation without error code."""
        error = GatewayError("Test error")
        assert str(error) == "Test error"

    def test_error_code_in_derived_classes(self) -> None:
        """Test error codes work in derived classes."""
        error = MQTTConnectionError("Connection failed", error_code=503)
        assert str(error) == "[503] Connection failed"
        assert error.error_code == 503


class TestExceptionContext:
    """Test exception context handling."""

    def test_empty_context_default(self) -> None:
        """Test that context defaults to empty dict."""
        error = GatewayError("Test")
        assert error.context == {}

    def test_context_merging_in_derived_classes(self) -> None:
        """Test that derived classes properly merge context."""
        error = MQTTConnectionError(
            "Connection failed",
            broker_host="example.com",
            broker_port=1883,
            context={"custom_field": "custom_value"},
        )

        assert error.context["broker_host"] == "example.com"
        assert error.context["broker_port"] == 1883
        assert error.context["custom_field"] == "custom_value"

    def test_none_values_not_added_to_context(self) -> None:
        """Test that None values are not added to context."""
        error = NetworkError(
            "Test error",
            interface="eth0",
            port=None,  # Should not be added to context
            operation=None,  # Should not be added to context
        )

        assert "interface" in error.context
        assert "port" not in error.context
        assert "operation" not in error.context
