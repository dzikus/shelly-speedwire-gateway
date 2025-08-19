"""Custom exceptions for Shelly 3EM to SMA Speedwire Gateway.

This module defines all custom exception classes used throughout
the application for specific error handling scenarios.
"""

from __future__ import annotations

from typing import Any

PACKET_HEADER_BYTES = 16


class GatewayError(Exception):
    """Base exception for all gateway-related errors.

    Parent class for all custom exceptions in the gateway.
    Common interface for catching any gateway-specific error.

    Attributes:
        message: Error message
        error_code: Optional numeric error code
        context: Optional additional context information
    """

    def __init__(self, message: str, error_code: int | None = None, context: dict[str, Any] | None = None) -> None:
        """Initialize gateway error.

        Args:
            message: Description of the error
            error_code: Optional numeric error code
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code is not None:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        """Return representation of the error."""
        return (
            f"{self.__class__.__name__}(message='{self.message}', error_code={self.error_code}, context={self.context})"
        )


class ConfigurationError(GatewayError):
    """Exception raised for configuration-related errors.

    This exception is raised when there are issues with:
    - Invalid configuration file format
    - Missing required configuration keys
    - Invalid configuration values
    - Configuration validation failures
    """

    def __init__(
        self,
        message: str,
        config_section: str | None = None,
        config_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Description of the configuration error
            config_section: Name of the configuration section with the error
            config_key: Name of the specific configuration key with the error
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if config_section:
            context["config_section"] = config_section
        if config_key:
            context["config_key"] = config_key
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.config_key = config_key


class MQTTConnectionError(GatewayError):
    """Exception raised for MQTT connection-related errors.

    This exception is raised when there are issues with:
    - MQTT broker connection failures
    - Authentication failures
    - Subscription failures
    - Message publishing errors
    - Connection timeouts

    Attributes:
        broker_host: MQTT broker hostname
        broker_port: MQTT broker port
        reconnect_attempts: Number of reconnection attempts made
    """

    def __init__(
        self,
        message: str,
        broker_host: str | None = None,
        broker_port: int | None = None,
        reconnect_attempts: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MQTT connection error.

        Args:
            message: Description of the MQTT error
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            reconnect_attempts: Number of reconnection attempts made
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if broker_host:
            context["broker_host"] = broker_host
        if broker_port:
            context["broker_port"] = broker_port
        if reconnect_attempts is not None:
            context["reconnect_attempts"] = reconnect_attempts
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.reconnect_attempts = reconnect_attempts


class NetworkError(GatewayError):
    """Exception raised for network-related errors.

    This exception is raised when there are issues with:
    - Socket creation or binding failures
    - Network interface problems
    - Multicast/broadcast setup errors
    - Packet transmission failures
    - Network discovery issues

    Attributes:
        interface: Network interface name or IP address
        port: Network port number
        operation: Network operation that failed
    """

    def __init__(
        self,
        message: str,
        interface: str | None = None,
        port: int | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize network error.

        Args:
            message: Description of the network error
            interface: Network interface name or IP address
            port: Network port number
            operation: Network operation that failed (e.g., 'bind', 'send', 'receive')
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if interface:
            context["interface"] = interface
        if port:
            context["port"] = port
        if operation:
            context["operation"] = operation
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.interface = interface
        self.port = port
        self.operation = operation


class ProtocolError(GatewayError):
    """Exception raised for protocol-related errors.

    This exception is raised when there are issues with:
    - Invalid SMA Speedwire packet format
    - OBIS data encoding/decoding errors
    - Protocol version incompatibilities
    - Malformed discovery packets

    Attributes:
        protocol: Protocol name (e.g., 'speedwire', 'obis')
        packet_data: Raw packet data that caused the error
        expected_format: Expected packet format description
    """

    def __init__(
        self,
        message: str,
        protocol: str | None = None,
        packet_data: bytes | None = None,
        expected_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize protocol error.

        Args:
            message: Description of the protocol error
            protocol: Protocol name
            packet_data: Raw packet data that caused the error
            expected_format: Expected packet format description
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if protocol:
            context["protocol"] = protocol
        if packet_data:
            context["packet_length"] = len(packet_data)
            # Use constant for header bytes length
            header_bytes = (
                packet_data[:PACKET_HEADER_BYTES].hex()
                if len(packet_data) >= PACKET_HEADER_BYTES
                else packet_data.hex()
            )
            context["packet_header"] = header_bytes
        if expected_format:
            context["expected_format"] = expected_format
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.protocol = protocol
        self.packet_data = packet_data
        self.expected_format = expected_format


class DataValidationError(GatewayError):
    """Exception raised for data validation errors.

    This exception is raised when there are issues with:
    - Invalid measurement values (voltage, current, power)
    - Out-of-range frequency values
    - Invalid power factor values
    - Corrupted MQTT message data

    Attributes:
        field_name: Name of the field that failed validation
        field_value: Value that failed validation
        valid_range: Description of valid value range
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any | None = None,
        valid_range: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize data validation error.

        Args:
            message: Description of the validation error
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            valid_range: Description of valid value range
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = field_value
        if valid_range:
            context["valid_range"] = valid_range
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.valid_range = valid_range


class ShutdownRequestedError(Exception):  # N818: Add Error suffix
    """Exception raised when graceful shutdown is requested.

    Exception for signaling that the application should perform
    a graceful shutdown. Not an error condition but a control
    flow mechanism.

    Attributes:
        reason: Reason for the shutdown request
        initiated_by: Component that initiated the shutdown
    """

    def __init__(self, reason: str = "Shutdown requested", initiated_by: str | None = None) -> None:
        """Initialize shutdown request.

        Args:
            reason: Reason for the shutdown request
            initiated_by: Component that initiated the shutdown
        """
        super().__init__(reason)
        self.reason = reason
        self.initiated_by = initiated_by

    def __str__(self) -> str:
        """Return string representation of the shutdown request."""
        if self.initiated_by:
            return f"{self.reason} (initiated by {self.initiated_by})"
        return self.reason


class GatewayTimeoutError(GatewayError):
    """Exception raised when operations timeout.

    This exception is raised when operations exceed their timeout limits:
    - MQTT connection timeouts
    - Network operation timeouts
    - Data reception timeouts

    Attributes:
        timeout_seconds: Timeout duration in seconds
        operation: Operation that timed out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Description of the timeout error
            timeout_seconds: Timeout duration in seconds
            operation: Operation that timed out
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if operation:
            context["operation"] = operation
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class DeviceError(GatewayError):
    """Exception raised for device-specific errors.

    This exception is raised when there are issues with:
    - Shelly device communication
    - Device configuration errors
    - Device status reporting issues

    Attributes:
        device_id: Device identifier
        device_type: Type of device (e.g., 'shelly3em')
        last_seen: Timestamp when device was last seen
    """

    def __init__(
        self,
        message: str,
        device_id: str | None = None,
        device_type: str | None = None,
        last_seen: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize device error.

        Args:
            message: Description of the device error
            device_id: Device identifier
            device_type: Type of device
            last_seen: Timestamp when device was last seen
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.get("context", {})
        if device_id:
            context["device_id"] = device_id
        if device_type:
            context["device_type"] = device_type
        if last_seen is not None:
            context["last_seen"] = last_seen
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.device_id = device_id
        self.device_type = device_type
        self.last_seen = last_seen
