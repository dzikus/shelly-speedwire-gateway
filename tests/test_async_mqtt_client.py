"""Tests for async MQTT client functionality."""
# pylint: disable=redefined-outer-name,protected-access,too-many-public-methods

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from shelly_speedwire_gateway.async_mqtt_client import (
    MQTTClient,
    create_mqtt_client,
    mqtt_session,
)
from shelly_speedwire_gateway.exceptions import (
    GatewayTimeoutError,
    MQTTConnectionError,
)
from shelly_speedwire_gateway.models import MQTTSettings


class FailingAsyncIterator:
    """Mock async iterator that raises CancelledError."""

    def __aiter__(self) -> FailingAsyncIterator:
        return self

    async def __anext__(self) -> None:
        raise asyncio.CancelledError("Message loop cancelled")


class TestMQTTClientErrorHandling:
    """Test MQTT client error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_connect_with_shutdown_during_backoff(
        self,
        mqtt_config: MQTTSettings,
        mock_message_handler: Mock,
    ) -> None:
        """Test connection cancelled during backoff (line 109)."""
        client = MQTTClient(mqtt_config, mock_message_handler)
        client.state.shutdown_requested = True
        client._shutdown_event.set()  # Simulate shutdown signal

        with (
            patch.object(client, "_attempt_connection", side_effect=ConnectionError("Connection failed")),
            pytest.raises(MQTTConnectionError, match="Connection cancelled due to shutdown"),
        ):
            await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect_timeout_handling(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test disconnect timeout handling (lines 186-191)."""
        client = MQTTClient(mqtt_config, mock_message_handler)

        # Mock exit stack and client
        mock_exit_stack = AsyncMock()
        mock_exit_stack.aclose.side_effect = TimeoutError()
        client._exit_stack = mock_exit_stack
        client._client = AsyncMock()

        # Call disconnect which should handle the timeout
        await client.disconnect()

        # Verify that the exit stack was attempted to be closed
        mock_exit_stack.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_logging_paths(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test message processing logging paths (lines 204, 209-210)."""
        client = MQTTClient(mqtt_config, mock_message_handler)

        # Test message handler call
        await client._handle_message(Mock(topic="test/topic", payload=b"test payload"))
        mock_message_handler.assert_called_with("test/topic", b"test payload")

        # Test handler exception handling - external handler exceptions should be caught and logged
        mock_message_handler.side_effect = ValueError("Handler error")
        client.log = Mock()

        # This should not raise but should log the error
        await client._handle_message(Mock(topic="test/topic", payload=b"test payload"))
        client.log.exception.assert_called()

    @pytest.mark.asyncio
    async def test_run_message_loop_error_handling(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test message loop error handling (lines 218-223)."""
        client = MQTTClient(mqtt_config, mock_message_handler)
        client._client = AsyncMock()

        # Mock _client.messages to be an async iterator that raises CancelledError
        client._client.messages = FailingAsyncIterator()
        client.log = Mock()

        # Should handle exception and re-raise
        with pytest.raises(asyncio.CancelledError):
            await client._process_messages()

        # Verify cancellation was logged (line 218)
        client.log.info.assert_called_with("Message processing cancelled", error="Message loop cancelled")

    @pytest.mark.asyncio
    async def test_health_check_no_client(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test health check when no client exists (line 253-254)."""
        client = MQTTClient(mqtt_config, mock_message_handler)
        client._client = None

        # Should return False when no client
        result = client.is_connected
        assert not result

    def test_get_stats_no_client(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test get_stats when no client exists (line 299)."""
        client = MQTTClient(mqtt_config, mock_message_handler)
        client._client = None

        stats = client.get_stats()

        # Should return basic stats without client info
        assert "connected" in stats
        assert "reconnect_attempts" in stats
        assert "connection_duration" in stats
        assert stats["connected"] is False

    @pytest.mark.asyncio
    async def test_subscribe_no_client(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test subscribe when no client exists (line 324)."""
        client = MQTTClient(mqtt_config, mock_message_handler)
        client._client = None

        # Should raise RuntimeError when no client
        with pytest.raises(RuntimeError, match="Client not connected"):
            await client._subscribe_to_topics()

    @pytest.mark.asyncio
    async def test_mqtt_session_context_manager_error(self) -> None:
        """Test mqtt_session context manager error handling (lines 332-341)."""
        config = {
            "broker_host": "localhost",
            "broker_port": 1883,
            "base_topic": "test/topic",
        }
        mock_handler = Mock()

        # Mock create_mqtt_client to raise an exception
        with (
            patch(
                "shelly_speedwire_gateway.async_mqtt_client.create_mqtt_client",
                side_effect=RuntimeError("Client creation failed"),
            ),
            pytest.raises(RuntimeError, match="Client creation failed"),
        ):
            async with mqtt_session(config, mock_handler):
                pass  # Should not reach this point


@pytest.fixture
def mqtt_config() -> MQTTSettings:
    """Create MQTT configuration."""
    return MQTTSettings(
        base_topic="shellies/test-device",
        broker_host="localhost",
        broker_port=1883,
        username="testuser",
        password="testpass",
        keepalive=60,
        qos=1,
    )


@pytest.fixture
def mock_message_handler() -> Mock:
    """Create mock message handler."""
    return Mock()


class TestMQTTClient:
    """Test MQTT client functionality."""

    def test_init(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test MQTT client initialization."""
        client = MQTTClient(mqtt_config, mock_message_handler)

        assert client.config == mqtt_config
        assert client.message_handler == mock_message_handler
        assert client._client is None
        assert client._running is False
        assert not client._shutdown_event.is_set()

    def test_request_shutdown(self, mqtt_config: MQTTSettings) -> None:
        """Test shutdown request."""
        client = MQTTClient(mqtt_config)

        client.request_shutdown()

        assert client._shutdown_event.is_set()
        assert client._running is False

    @pytest.mark.asyncio
    @patch("aiomqtt.Client")
    async def test_connect_success(self, mock_client_class: Mock, mqtt_config: MQTTSettings) -> None:
        """Test successful MQTT connection."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        client = MQTTClient(mqtt_config)

        with patch.object(client, "_attempt_connection", new_callable=AsyncMock) as mock_attempt:
            await client.connect()

        assert client.state.connected
        assert client.state.reconnect_attempts == 0
        mock_attempt.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_max_attempts(self, mqtt_config: MQTTSettings) -> None:
        """Test connection failure after max attempts."""
        mqtt_config.max_reconnect_attempts = 2
        client = MQTTClient(mqtt_config)

        with (
            patch.object(client, "_attempt_connection", side_effect=ConnectionError("Connection failed")),
            pytest.raises(MQTTConnectionError, match="Failed to connect after 2 attempts"),
        ):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_cancelled_during_backoff(self, mqtt_config: MQTTSettings) -> None:
        """Test connection cancelled during backoff."""
        mqtt_config.max_reconnect_attempts = 3
        client = MQTTClient(mqtt_config)

        # Fail first attempt, then cancel during backoff
        with patch.object(client, "_attempt_connection", side_effect=ConnectionError("Connection failed")):
            client.request_shutdown()  # Set shutdown event

            with pytest.raises(MQTTConnectionError, match="Connection cancelled due to shutdown"):
                await client.connect()

    @pytest.mark.asyncio
    @patch("shelly_speedwire_gateway.async_mqtt_client.AsyncExitStack")
    @patch("aiomqtt.Client")
    async def test_attempt_connection(
        self,
        mock_client_class: Mock,
        mock_exit_stack_class: Mock,
        mqtt_config: MQTTSettings,
    ) -> None:
        """Test single connection attempt."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_exit_stack = AsyncMock()
        mock_exit_stack.enter_async_context.return_value = mock_client
        mock_exit_stack_class.return_value = mock_exit_stack

        client = MQTTClient(mqtt_config)

        await client._attempt_connection()

        # Verify client was created with correct parameters
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["hostname"] == "localhost"
        assert call_kwargs["port"] == 1883
        assert call_kwargs["username"] == "testuser"
        assert call_kwargs["password"] == "testpass"

        # Verify subscription
        assert mock_client.subscribe.call_count == 2
        topics = [call[0][0] for call in mock_client.subscribe.call_args_list]
        assert "shellies/test-device/emeter/+/+" in topics
        assert "shellies/test-device/online" in topics

    @pytest.mark.asyncio
    async def test_subscribe_to_topics_no_client(self, mqtt_config: MQTTSettings) -> None:
        """Test subscription without connected client."""
        client = MQTTClient(mqtt_config)

        with pytest.raises(RuntimeError, match="Client not connected"):
            await client._subscribe_to_topics()

    @pytest.mark.asyncio
    async def test_disconnect(self, mqtt_config: MQTTSettings) -> None:
        """Test MQTT disconnection."""
        client = MQTTClient(mqtt_config)
        client.state.connected = True

        mock_exit_stack = AsyncMock()
        client._exit_stack = mock_exit_stack
        client._client = Mock()

        await client.disconnect()

        assert not client.state.connected
        assert client._shutdown_event.is_set()
        assert client._running is False
        assert client._client is None
        mock_exit_stack.aclose.assert_called_once()  # type: ignore[unreachable]

    @pytest.mark.asyncio
    async def test_disconnect_with_exit_stack_error(self, mqtt_config: MQTTSettings) -> None:
        """Test disconnection with exit stack error."""
        client = MQTTClient(mqtt_config)
        client.state.connected = True

        mock_exit_stack = AsyncMock()
        mock_exit_stack.aclose.side_effect = RuntimeError("Exit stack error")
        client._exit_stack = mock_exit_stack

        # Should not raise exception
        await client.disconnect()

        assert client._exit_stack is None

    @pytest.mark.asyncio
    async def test_run_message_loop_no_client(self, mqtt_config: MQTTSettings) -> None:
        """Test message loop without connected client."""
        client = MQTTClient(mqtt_config)

        with pytest.raises(RuntimeError, match="Client not connected"):
            await client.run_message_loop()

    @pytest.mark.asyncio
    async def test_run_message_loop_success(self, mqtt_config: MQTTSettings) -> None:
        """Test successful message loop execution."""
        client = MQTTClient(mqtt_config)
        client._client = Mock()

        # Mock the three main tasks to complete immediately
        with (
            patch.object(client, "_process_messages", new_callable=AsyncMock) as mock_process,
            patch.object(client, "_monitor_connection", new_callable=AsyncMock) as mock_monitor,
            patch.object(client, "_wait_for_shutdown", new_callable=AsyncMock) as mock_wait,
        ):
            # Make shutdown wait complete immediately
            client._shutdown_event.set()

            await client.run_message_loop()

            assert not client._running
            mock_process.assert_called_once()
            mock_monitor.assert_called_once()
            mock_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self, mqtt_config: MQTTSettings) -> None:
        """Test shutdown wait functionality."""
        client = MQTTClient(mqtt_config)

        # Set shutdown event immediately
        client._shutdown_event.set()

        await client._wait_for_shutdown()

    @pytest.mark.asyncio
    async def test_process_messages(self, mqtt_config: MQTTSettings, mock_message_handler: Mock) -> None:
        """Test message processing."""
        client = MQTTClient(mqtt_config, mock_message_handler)

        # Create mock client with messages
        mock_client = AsyncMock()
        mock_message1 = Mock()
        mock_message1.topic = "test/topic"
        mock_message1.payload = b"test payload"

        mock_message2 = Mock()
        mock_message2.topic = "test/topic2"
        mock_message2.payload = b"test payload2"

        async def mock_messages() -> AsyncGenerator[Mock]:
            yield mock_message1
            yield mock_message2
            # Then trigger shutdown
            client._shutdown_event.set()

        mock_client.messages = mock_messages()
        client._client = mock_client
        client._running = True

        with patch.object(client, "_handle_message", new_callable=AsyncMock) as mock_handle:
            await client._process_messages()

            assert mock_handle.call_count == 2
            mock_handle.assert_any_call(mock_message1)
            mock_handle.assert_any_call(mock_message2)

    @pytest.mark.asyncio
    async def test_process_messages_with_error(self, mqtt_config: MQTTSettings) -> None:
        """Test message processing with error handling."""
        client = MQTTClient(mqtt_config)

        mock_client = AsyncMock()
        mock_message = Mock()
        mock_message.topic = "test/topic"
        mock_message.payload = b"test payload"

        async def mock_messages() -> AsyncGenerator[Mock]:
            yield mock_message
            client._shutdown_event.set()

        mock_client.messages = mock_messages()
        client._client = mock_client
        client._running = True

        with patch.object(client, "_handle_message", side_effect=ValueError("Processing error")):
            # Should not raise exception, just log it
            await client._process_messages()

    @pytest.mark.asyncio
    async def test_handle_message_online_status(self, mqtt_config: MQTTSettings) -> None:
        """Test handling online status messages."""
        client = MQTTClient(mqtt_config)

        mock_message = Mock()
        mock_message.topic = "shellies/test-device/online"
        mock_message.payload = b"true"

        with patch.object(client, "_handle_online_message", new_callable=AsyncMock) as mock_online:
            await client._handle_message(mock_message)

            mock_online.assert_called_once_with(b"true")

    @pytest.mark.asyncio
    async def test_handle_message_emeter_data(self, mqtt_config: MQTTSettings) -> None:
        """Test handling emeter measurement messages."""
        client = MQTTClient(mqtt_config)

        mock_message = Mock()
        mock_message.topic = "shellies/test-device/emeter/0/voltage"
        mock_message.payload = b"230.5"

        with patch.object(client, "_handle_emeter_message", new_callable=AsyncMock) as mock_emeter:
            await client._handle_message(mock_message)

            mock_emeter.assert_called_once_with("0", "voltage", b"230.5")

    @pytest.mark.asyncio
    async def test_handle_message_unknown_topic(self, mqtt_config: MQTTSettings) -> None:
        """Test handling unknown topic messages."""
        client = MQTTClient(mqtt_config)

        mock_message = Mock()
        mock_message.topic = "unknown/topic"
        mock_message.payload = b"data"

        # Should not raise exception, just log warning
        await client._handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_with_external_handler(self, mqtt_config: MQTTSettings) -> None:
        """Test message handling with external handler."""
        mock_handler = AsyncMock()
        client = MQTTClient(mqtt_config, mock_handler)

        mock_message = Mock()
        mock_message.topic = "test/topic"
        mock_message.payload = b"test data"

        await client._handle_message(mock_message)

        # External handler should be called regardless of topic pattern
        mock_handler.assert_called_once_with("test/topic", b"test data")

    def test_convert_payload_to_bytes_bytes(self, mqtt_config: MQTTSettings) -> None:
        """Test payload conversion from bytes."""
        client = MQTTClient(mqtt_config)

        payload = b"test bytes"
        result = client._convert_payload_to_bytes(payload)

        assert result == b"test bytes"

    def test_convert_payload_to_bytes_string(self, mqtt_config: MQTTSettings) -> None:
        """Test payload conversion from string."""
        client = MQTTClient(mqtt_config)

        payload = "test string"
        result = client._convert_payload_to_bytes(payload)

        assert result == b"test string"

    def test_convert_payload_to_bytes_number(self, mqtt_config: MQTTSettings) -> None:
        """Test payload conversion from number."""
        client = MQTTClient(mqtt_config)

        result_int = client._convert_payload_to_bytes(123)
        result_float = client._convert_payload_to_bytes(45.67)

        assert result_int == b"123"
        assert result_float == b"45.67"

    def test_convert_payload_to_bytes_none(self, mqtt_config: MQTTSettings) -> None:
        """Test payload conversion from None."""
        client = MQTTClient(mqtt_config)

        result = client._convert_payload_to_bytes(None)

        assert result == b""

    def test_convert_payload_to_bytes_bytearray(self, mqtt_config: MQTTSettings) -> None:
        """Test payload conversion from bytearray."""
        client = MQTTClient(mqtt_config)

        payload = bytearray(b"test bytearray")
        result = client._convert_payload_to_bytes(payload)

        assert result == b"test bytearray"

    def test_convert_payload_to_bytes_invalid(self, mqtt_config: MQTTSettings) -> None:
        """Test payload conversion from invalid type."""
        client = MQTTClient(mqtt_config)

        class InvalidPayload:
            """Test class with invalid string conversion."""

            def __str__(self) -> str:
                raise UnicodeEncodeError("test", "", 0, 1, "test error")

        result = client._convert_payload_to_bytes(InvalidPayload())

        assert result == b""

    @pytest.mark.asyncio
    async def test_handle_online_message_valid(self, mqtt_config: MQTTSettings) -> None:
        """Test handling valid online message."""
        client = MQTTClient(mqtt_config)

        await client._handle_online_message(b"true")
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_handle_online_message_invalid(self, mqtt_config: MQTTSettings) -> None:
        """Test handling invalid online message."""
        client = MQTTClient(mqtt_config)

        # Invalid UTF-8 bytes
        await client._handle_online_message(b"\xff\xfe")
        # Should not raise exception, just log warning

    @pytest.mark.asyncio
    async def test_handle_emeter_message_valid(self, mqtt_config: MQTTSettings) -> None:
        """Test handling valid emeter message."""
        client = MQTTClient(mqtt_config)

        await client._handle_emeter_message("0", "voltage", b"230.5")
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_handle_emeter_message_invalid(self, mqtt_config: MQTTSettings) -> None:
        """Test handling invalid emeter message."""
        client = MQTTClient(mqtt_config)

        await client._handle_emeter_message("0", "voltage", b"not_a_number")
        # Should not raise exception, just log warning

    @pytest.mark.asyncio
    async def test_run_handler_safely_sync(self, mqtt_config: MQTTSettings) -> None:
        """Test running synchronous handler safely."""

        def sync_handler(topic: str, _payload: bytes) -> None:
            _ = f"handled {topic}"

        client = MQTTClient(mqtt_config, sync_handler)

        await client._run_handler_safely("test/topic", b"test data")

    @pytest.mark.asyncio
    async def test_run_handler_safely_async(self, mqtt_config: MQTTSettings) -> None:
        """Test running asynchronous handler safely."""

        async def async_handler(topic: str, _payload: bytes) -> None:
            await asyncio.sleep(0.001)
            _ = f"handled {topic}"

        client = MQTTClient(mqtt_config)
        client.message_handler = async_handler  # type: ignore[assignment]

        await client._run_handler_safely("test/topic", b"test data")

    @pytest.mark.asyncio
    async def test_run_handler_safely_timeout(self, mqtt_config: MQTTSettings) -> None:
        """Test handler timeout."""

        async def slow_handler(_topic: str, _payload: bytes) -> None:
            await asyncio.sleep(10)  # Longer than timeout

        mqtt_config.connection_timeout = 0.1  # Very short timeout
        client = MQTTClient(mqtt_config)
        client.message_handler = slow_handler  # type: ignore[assignment]

        with pytest.raises(GatewayTimeoutError, match="Message handler timeout"):
            await client._run_handler_safely("test/topic", b"test data")

    @pytest.mark.asyncio
    async def test_run_handler_safely_no_handler(self, mqtt_config: MQTTSettings) -> None:
        """Test _run_handler_safely when no handler is set."""
        client = MQTTClient(mqtt_config)
        client.message_handler = None

        # Should return without error
        await client._run_handler_safely("test/topic", b"payload")

    @pytest.mark.asyncio
    async def test_monitor_connection(self, mqtt_config: MQTTSettings) -> None:
        """Test connection monitoring."""
        client = MQTTClient(mqtt_config)
        client._running = True
        client.state.connected = True

        # Set shutdown event to exit quickly
        client._shutdown_event.set()

        await client._monitor_connection()

    @pytest.mark.asyncio
    async def test_monitor_connection_reconnect(self, mqtt_config: MQTTSettings) -> None:
        """Test connection monitoring with reconnection."""
        client = MQTTClient(mqtt_config)
        client._running = True
        client.state.connected = False  # Simulate disconnection

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            # Set shutdown after first iteration
            async def delayed_shutdown() -> None:
                await asyncio.sleep(0.1)
                client._shutdown_event.set()

            _ = asyncio.create_task(delayed_shutdown())

            await client._monitor_connection()

            mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_publish_success(self, mqtt_config: MQTTSettings) -> None:
        """Test successful message publishing."""
        mock_client = AsyncMock()
        client = MQTTClient(mqtt_config)
        client._client = mock_client
        client.state.connected = True

        await client.publish("test/topic", b"test payload")

        mock_client.publish.assert_called_once_with(
            topic="test/topic",
            payload=b"test payload",
            qos=1,
            retain=False,
        )

    @pytest.mark.asyncio
    async def test_publish_not_connected(self, mqtt_config: MQTTSettings) -> None:
        """Test publishing when not connected."""
        client = MQTTClient(mqtt_config)

        with pytest.raises(MQTTConnectionError, match="Client not connected"):
            await client.publish("test/topic", b"test payload")

    @pytest.mark.asyncio
    async def test_publish_failure(self, mqtt_config: MQTTSettings) -> None:
        """Test publishing with failure."""
        mock_client = AsyncMock()
        mock_client.publish.side_effect = OSError("Publish failed")

        client = MQTTClient(mqtt_config)
        client._client = mock_client
        client.state.connected = True

        with pytest.raises(MQTTConnectionError, match="Publish failed"):
            await client.publish("test/topic", b"test payload")

    def test_is_connected_property(self, mqtt_config: MQTTSettings) -> None:
        """Test is_connected property."""
        client = MQTTClient(mqtt_config)

        assert not client.is_connected

        client.state.connected = True
        client._client = Mock()

        assert client.is_connected

    def test_get_stats(self, mqtt_config: MQTTSettings) -> None:
        """Test statistics retrieval."""
        client = MQTTClient(mqtt_config)
        client.state.connected = True
        client._client = Mock()  # Mock the client to make is_connected return True
        client.state.reconnect_attempts = 2
        client.state.total_messages_received = 100

        stats = client.get_stats()

        assert stats["connected"] is True
        assert stats["reconnect_attempts"] == 2
        assert stats["total_messages_received"] == 100
        assert "connection_duration" in stats
        assert "messages_per_second" in stats

    @pytest.mark.asyncio
    async def test_context_manager(self, mqtt_config: MQTTSettings) -> None:
        """Test async context manager usage."""
        client = MQTTClient(mqtt_config)

        with (
            patch.object(client, "connect", new_callable=AsyncMock) as mock_connect,
            patch.object(client, "disconnect", new_callable=AsyncMock) as mock_disconnect,
        ):
            async with client as ctx_client:
                assert ctx_client == client
                mock_connect.assert_called_once()

            mock_disconnect.assert_called_once()


class TestCreateMQTTClient:
    """Test MQTT client factory function."""

    @pytest.mark.asyncio
    async def test_create_mqtt_client_minimal_config(self) -> None:
        """Test client creation with minimal config."""
        config = {"base_topic": "shellies/test"}

        client = await create_mqtt_client(config)

        assert client.config.base_topic == "shellies/test"
        assert client.config.broker_host == "localhost"
        assert client.config.broker_port == 1883
        assert client.message_handler is None

    @pytest.mark.asyncio
    async def test_create_mqtt_client_full_config(self) -> None:
        """Test client creation with full config."""
        config = {
            "base_topic": "shellies/test",
            "broker_host": "mqtt.example.com",
            "broker_port": 8883,
            "username": "user",
            "password": "pass",
            "keepalive": 30,
            "qos": 2,
        }

        mock_handler = Mock()
        client = await create_mqtt_client(config, mock_handler)

        assert client.config.broker_host == "mqtt.example.com"
        assert client.config.broker_port == 8883
        assert client.config.username == "user"
        assert client.config.password == "pass"
        assert client.config.keepalive == 30
        assert client.config.qos == 2
        assert client.message_handler == mock_handler


class TestMQTTSession:
    """Test MQTT session context manager."""

    @pytest.mark.asyncio
    async def test_mqtt_session_success(self) -> None:
        """Test successful MQTT session usage."""
        config = {"base_topic": "shellies/test"}

        with patch("shelly_speedwire_gateway.async_mqtt_client.create_mqtt_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            async with mqtt_session(config) as client:
                assert client == mock_client
                mock_client.__aenter__.assert_called_once()

            mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_mqtt_session_with_handler(self) -> None:
        """Test MQTT session with message handler."""
        config = {"base_topic": "shellies/test"}
        mock_handler = Mock()

        with patch("shelly_speedwire_gateway.async_mqtt_client.create_mqtt_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            async with mqtt_session(config, mock_handler) as client:
                assert client == mock_client

            mock_create.assert_called_once_with(config, mock_handler)

    @pytest.mark.asyncio
    async def test_mqtt_session_with_exception(self) -> None:
        """Test MQTT session with exception in body."""
        config = {"base_topic": "shellies/test"}

        with patch("shelly_speedwire_gateway.async_mqtt_client.create_mqtt_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Test exception handling in context manager
            exception_occurred = False
            try:
                async with mqtt_session(config) as client:
                    msg = "Test exception"
                    assert client == mock_client
                    raise ValueError(msg)
            except ValueError:
                exception_occurred = True

            assert exception_occurred, "Expected ValueError to be raised"

        # Verify cleanup was called after exception
        mock_client.disconnect.assert_called_once()
