"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from shelly_speedwire_gateway.config import (
    _deep_merge_dict,
    create_default_config,
    export_config_template,
    load_config,
    load_settings_from_env,
    merge_config_sources,
    setup_logging_from_config,
    validate_config,
    validate_mqtt_connection,
)
from shelly_speedwire_gateway.exceptions import ConfigurationError
from shelly_speedwire_gateway.models import GatewaySettings, MQTTSettings, SpeedwireSettings


class TestConfigValidation:
    """Test configuration validation and error handling."""

    def test_validation_error_handling(self) -> None:
        """Test ValidationError handling with invalid configuration data."""
        invalid_config_data = {
            "mqtt": {
                "broker_host": "test.broker.com",
                "broker_port": "invalid_port",  # Should be int
                "base_topic": "",  # Should not be empty
            },
            "speedwire": {
                "serial": -123456789,  # Should be positive
                "interval": "invalid_interval",  # Should be float
            },
            "log_level": "INVALID_LEVEL",  # Invalid log level
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config(config_path)

            # Verify error includes field paths and messages
            error_msg = str(exc_info.value)
            assert "Configuration validation failed" in error_msg
            assert "mqtt" in error_msg or "speedwire" in error_msg

        finally:
            Path(config_path).unlink()

    def test_load_config_file_errors(self) -> None:
        """Test configuration loading with file system errors."""
        # Test non-existent file - now returns a default config
        result = load_config("non_existent_file.yaml")
        assert isinstance(result, dict)
        assert "mqtt" in result

        # Test invalid YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:\n  - unclosed:")
            config_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Failed to load configuration file"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_validate_config_error_handling(self) -> None:
        """Test configuration validation with various error conditions."""
        # Test ValidationError
        invalid_config = {
            "mqtt": {"broker_port": "invalid"},  # Invalid type
        }

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            validate_config(invalid_config)

        # Test that empty dict validates successfully (uses defaults)
        validate_config({})  # Empty dict should work with defaults

    def test_merge_config_sources_error_paths(self) -> None:
        """Test configuration merging with conflicting or invalid data."""
        base_config = {"mqtt": {"broker_host": "localhost"}}

        # Test with empty env_overrides - returns GatewaySettings object, not dict
        result = merge_config_sources(base_config, "")
        assert isinstance(result, GatewaySettings)
        assert result.mqtt.broker_host == "localhost"

    def test_create_default_config_custom_filename(self) -> None:
        """Test creating default configuration with custom filename."""
        # Test with custom filename
        custom_name = "custom_config.yaml"

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / custom_name

            # Should create config with custom name
            result = create_default_config(custom_path)
            assert isinstance(result, dict)
            assert custom_path.exists()
            assert custom_path.name == custom_name

    def test_setup_logging_custom_configuration(self) -> None:
        """Test logging setup with custom configuration options."""

        # Test with custom log configuration
        config = GatewaySettings(
            mqtt=MQTTSettings(
                broker_host="localhost",
                base_topic="test/topic",
            ),
            speedwire=SpeedwireSettings(
                serial=123456789,
            ),
            log_level="DEBUG",
        )

        # Should handle custom logging setup
        setup_logging_from_config(config)

    def test_config_schema_validation(self) -> None:
        """Test that GatewaySettings has proper schema."""
        # Test that GatewaySettings model can be validated
        schema = GatewaySettings.model_json_schema()
        assert "properties" in schema
        assert isinstance(schema, dict)


class TestLoadConfig:
    """Test configuration file loading."""

    def test_load_valid_config(self) -> None:
        """Test loading a valid configuration file."""
        config_data = {
            "mqtt": {
                "broker_host": "test.broker.com",
                "broker_port": 1883,
                "base_topic": "shellies/test-device",
            },
            "speedwire": {
                "serial": 123456789,
                "interval": 2.0,
            },
            "log_level": "INFO",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loaded_config = load_config(config_path)
            assert loaded_config["mqtt"]["broker_host"] == "test.broker.com"
            assert loaded_config["speedwire"]["serial"] == 123456789
            assert loaded_config["log_level"] == "INFO"
        finally:
            Path(config_path).unlink()

    def test_load_nonexistent_config(self) -> None:
        """Test loading a non-existent configuration file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config("/nonexistent/config.yaml")

        assert "Configuration file not found" in str(exc_info.value) or "Failed to create configuration file" in str(
            exc_info.value,
        )

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:\n  - unbalanced")
            config_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config(config_path)

            assert "Invalid YAML format" in str(exc_info.value) or "Failed to load configuration file" in str(
                exc_info.value,
            )
        finally:
            Path(config_path).unlink()

    def test_load_empty_config(self) -> None:
        """Test loading empty configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config(config_path)
            assert "Configuration file is empty" in str(exc_info.value) or "invalid YAML" in str(exc_info.value)
        finally:
            Path(config_path).unlink()


class TestCreateDefaultConfig:
    """Test default configuration creation."""

    def test_create_default_config_creates_file(self) -> None:
        """Test that default config file is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            create_default_config(config_path)

            assert config_path.exists()

            # Verify it contains expected sections
            with config_path.open() as f:
                config_data = yaml.safe_load(f)

            assert "mqtt" in config_data
            assert "speedwire" in config_data
            assert "log_level" in config_data

    def test_create_default_config_overwrites_existing(self) -> None:
        """Test that default config overwrites existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("existing: content")
            config_path = f.name

        try:
            create_default_config(Path(config_path))

            # Verify file was overwritten with default content
            with Path(config_path).open(encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            assert "existing" not in config_data
            assert "mqtt" in config_data
        finally:
            Path(config_path).unlink()


class TestValidateConfig:
    """Test configuration validation."""

    def test_validate_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config_data = {
            "mqtt": {
                "broker_host": "localhost",
                "broker_port": 1883,
                "base_topic": "shellies/test",
            },
            "speedwire": {
                "serial": 123456789,
                "interval": 1.0,
            },
            "log_level": "INFO",
        }

        # Should not raise exception
        validate_config(config_data)

    def test_validate_invalid_mqtt_port(self) -> None:
        """Test validation with invalid MQTT port."""
        config_data = {
            "mqtt": {
                "broker_host": "localhost",
                "broker_port": 99999,  # Invalid port
            },
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_data)

        error_str = str(exc_info.value)
        assert "validation error" in error_str.lower() or "validation failed" in error_str.lower()

    def test_validate_invalid_serial(self) -> None:
        """Test validation with invalid serial number."""
        config_data = {
            "speedwire": {
                "serial": -1,  # Invalid negative serial
            },
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_data)

        error_str = str(exc_info.value)
        assert "validation error" in error_str.lower() or "validation failed" in error_str.lower()

    def test_validate_invalid_log_level(self) -> None:
        """Test validation with invalid log level."""
        config_data = {
            "log_level": "INVALID_LEVEL",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_data)

        error_str = str(exc_info.value)
        assert "validation error" in error_str.lower() or "validation failed" in error_str.lower()


class TestMergeConfigSources:
    """Test configuration merging functionality."""

    def test_merge_with_defaults(self) -> None:
        """Test merging with default configuration."""
        # Test with no file config (None)
        merged = merge_config_sources(None, "enabled")

        assert isinstance(merged, GatewaySettings)
        # Should have default values
        assert merged.mqtt.broker_host == "localhost"
        assert merged.mqtt.broker_port == 1883

    def test_merge_file_config_enabled_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test merging file config with environment overrides enabled."""
        # Clear any existing environment variables that might interfere
        for key in ["MQTT_BROKER_HOST", "MQTT_BROKER_PORT", "SG_MQTT_BROKER_HOST"]:
            monkeypatch.delenv(key, raising=False)

        file_config = {
            "mqtt": {
                "broker_host": "file.broker.com",
                "broker_port": 1883,
            },
        }

        merged = merge_config_sources(file_config, "enabled")

        assert isinstance(merged, GatewaySettings)
        assert merged.mqtt.broker_host == "file.broker.com"

    def test_merge_file_config_disabled_env_overrides(self) -> None:
        """Test merging file config with environment overrides disabled."""
        file_config = {
            "mqtt": {
                "broker_host": "file.broker.com",
                "broker_port": 1883,
            },
        }

        merged = merge_config_sources(file_config, "disabled")

        assert isinstance(merged, GatewaySettings)
        assert merged.mqtt.broker_host == "file.broker.com"

    def test_merge_invalid_config(self) -> None:
        """Test merging with invalid configuration data."""
        invalid_config = {
            "mqtt": {
                "broker_port": "invalid_port",  # Should be integer
            },
        }

        with pytest.raises(ConfigurationError) as exc_info:
            merge_config_sources(invalid_config, "enabled")

        error_str = str(exc_info.value)
        assert "Failed to create settings" in error_str or "validation failed" in error_str.lower()

    def test_merge_empty_file_config(self) -> None:
        """Test merging with empty file configuration."""
        merged = merge_config_sources({}, "enabled")

        assert isinstance(merged, GatewaySettings)
        # Should have all defaults
        assert merged.mqtt.broker_host == "localhost"
        assert merged.speedwire.interval == 1.0

    def test_merge_with_env_overrides(self) -> None:
        """Test environment variable override behavior."""
        base_config = {"mqtt": {"broker_host": "localhost"}}

        with patch.dict(
            os.environ,
            {
                "MQTT_BROKER_PORT": "1883",
                "SPEEDWIRE_SERIAL": "123456789",
                "LOG_LEVEL": "DEBUG",
            },
        ):
            result = merge_config_sources(base_config, env_overrides="enabled")
            assert result.mqtt.broker_port == 1883
            assert result.speedwire.serial == 123456789
            assert result.log_level == "DEBUG"

    def test_merge_env_access_error_handling(self) -> None:
        """Test graceful handling of environment access errors."""
        base_config = {"mqtt": {"broker_host": "localhost"}}

        class MockEnviron(dict):
            """Mock environment that simulates access errors."""

            def __contains__(self, key: object) -> bool:
                return True

            def __getitem__(self, key: str) -> str:
                raise ValueError("Simulated environment access error")

        mock_environ = MockEnviron()

        with (
            patch("shelly_speedwire_gateway.config.os.environ", mock_environ),
            patch("shelly_speedwire_gateway.config.logger") as mock_logger,
        ):
            result = merge_config_sources(base_config, env_overrides="enabled")
            assert result.mqtt.broker_host == "localhost"
            mock_logger.debug.assert_called_with("No environment variable overrides found")

    def test_deep_merge_dict_functionality(self) -> None:
        """Test _deep_merge_dict helper function."""
        # Test nested dictionary merging
        target = {"a": {"b": 1, "c": 2}, "d": 3}
        source = {"a": {"c": 4, "e": 5}, "f": 6}

        _deep_merge_dict(target, source)

        assert target == {"a": {"b": 1, "c": 4, "e": 5}, "d": 3, "f": 6}

        # Test value replacement when not both dicts
        target = {"a": 1, "b": {"c": 2}}
        source = {"a": 3, "b": 4}

        _deep_merge_dict(target, source)

        assert target == {"a": 3, "b": 4}


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_config_workflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete configuration workflow from file to validation."""
        # Clear environment variables
        for key in ["MQTT_BROKER_HOST", "MQTT_BROKER_PORT", "MQTT_BASE_TOPIC"]:
            monkeypatch.delenv(key, raising=False)
        config_data = {
            "mqtt": {
                "broker_host": "integration.test.com",
                "broker_port": 8883,
                "base_topic": "shellies/integration-test",
                "username": "testuser",
                "password": "testpass",
            },
            "speedwire": {
                "serial": 987654321,
                "interval": 2.5,
                "use_broadcast": True,
            },
            "log_level": "DEBUG",
            "enable_monitoring": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Load config
            loaded_config = load_config(config_path)

            # Validate config
            validate_config(loaded_config)

            # Merge with sources
            merged = merge_config_sources(loaded_config, "enabled")

            # Verify final configuration
            assert merged.mqtt.broker_host == "integration.test.com"
            assert merged.mqtt.broker_port == 8883
            assert merged.mqtt.username == "testuser"
            assert merged.speedwire.serial == 987654321
            assert merged.speedwire.interval == 2.5
            assert merged.speedwire.use_broadcast is True
            assert merged.log_level == "DEBUG"
            assert merged.enable_monitoring is True

        finally:
            Path(config_path).unlink()

    def test_config_with_minimal_required_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration with only minimal required fields."""
        # Clear environment variables
        for key in ["MQTT_BROKER_HOST", "MQTT_BROKER_PORT", "MQTT_BASE_TOPIC"]:
            monkeypatch.delenv(key, raising=False)
        minimal_config = {
            "mqtt": {
                "base_topic": "shellies/minimal-test",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            loaded_config = load_config(config_path)
            validate_config(loaded_config)
            merged = merge_config_sources(loaded_config, "enabled")

            # Should use defaults for missing fields
            assert merged.mqtt.broker_host == "localhost"  # default
            assert merged.mqtt.broker_port == 1883  # default
            assert merged.mqtt.base_topic == "shellies/minimal-test"  # from config

        finally:
            Path(config_path).unlink()


class TestSetupLoggingFromConfig:
    """Test logging setup from configuration."""

    def test_setup_logging_with_gateway_settings(self) -> None:
        """Test logging setup with GatewaySettings object."""

        mqtt = MQTTSettings(base_topic="test", broker_host="test", broker_port=1883)
        speedwire = SpeedwireSettings(interval=1.0, serial=123456789)
        settings = GatewaySettings(
            mqtt=mqtt,
            speedwire=speedwire,
            log_level="DEBUG",
            log_format="json",
        )

        # Should not raise exception
        setup_logging_from_config(settings)

    def test_setup_logging_with_dict_config(self) -> None:
        """Test logging setup with dictionary config."""

        config = {
            "log_level": "WARNING",
            "log_format": "console",
        }

        # Should not raise exception
        setup_logging_from_config(config)

    def test_setup_logging_with_defaults(self) -> None:
        """Test logging setup with default values."""

        config: dict[str, str] = {}  # Empty config should use defaults

        # Should not raise exception
        setup_logging_from_config(config)


class TestValidateMqttConnection:
    """Test MQTT connection validation."""

    def test_validate_mqtt_connection_invalid_host(self) -> None:
        """Test MQTT connection validation with invalid host."""

        mqtt_config = MQTTSettings(
            broker_host="nonexistent.broker.invalid",
            broker_port=1883,
            base_topic="shellies/test",
        )

        # Should return False for invalid host
        result = validate_mqtt_connection(mqtt_config)
        assert result is False

    def test_validate_mqtt_connection_success(self) -> None:
        """Test successful MQTT connection validation."""

        config = GatewaySettings()

        with patch("socket.gethostbyname"), patch("socket.socket") as mock_socket:
            mock_sock = mock_socket.return_value
            mock_sock.connect_ex.return_value = 0

            result = validate_mqtt_connection(config.mqtt)
            assert result is True
            mock_sock.settimeout.assert_called_once_with(5.0)
            mock_sock.close.assert_called_once()

    def test_validate_mqtt_connection_failure(self) -> None:
        """Test MQTT connection validation failure."""

        config = GatewaySettings()

        with patch("socket.gethostbyname"), patch("socket.socket") as mock_socket:
            mock_sock = mock_socket.return_value
            mock_sock.connect_ex.return_value = 1  # Connection failed

            result = validate_mqtt_connection(config.mqtt)
            assert result is False

    def test_validate_mqtt_connection_network_error(self) -> None:
        """Test MQTT connection validation with network error."""

        config = GatewaySettings()

        with patch("socket.gethostbyname", side_effect=socket.gaierror("Name resolution failed")):
            result = validate_mqtt_connection(config.mqtt)
            assert result is False


class TestExportConfigTemplate:
    """Test configuration template export."""

    def test_export_config_template(self, tmp_path: Path) -> None:
        """Test exporting configuration template."""

        output_file = tmp_path / "config_template.yaml"

        export_config_template(output_file)

        # Verify file was created
        assert output_file.exists()

        # Verify it contains valid YAML
        with output_file.open() as f:
            content = yaml.safe_load(f)
            assert isinstance(content, dict)
            assert "mqtt" in content
            assert "speedwire" in content

    def test_export_config_template_creates_directory(self, tmp_path: Path) -> None:
        """Test template export creates directory if needed."""

        output_file = tmp_path / "subdir" / "config_template.yaml"

        export_config_template(output_file)

        # Verify directory and file were created
        assert output_file.parent.exists()
        assert output_file.exists()

    def test_export_config_template_error(self) -> None:
        """Test template export error handling."""
        output_path = Path("/invalid/path/config.yaml")

        with pytest.raises(ConfigurationError) as exc_info:
            export_config_template(output_path)

        error_msg = str(exc_info.value)
        assert "Failed to export configuration template" in error_msg


class TestLoadSettingsFromEnv:
    """Test loading settings from environment variables."""

    def test_load_settings_from_env_defaults(self) -> None:
        """Test loading settings with default values."""

        # Should not raise exception and return default settings
        settings = load_settings_from_env()

        assert isinstance(settings, GatewaySettings)
        assert settings.mqtt.broker_host == "localhost"
        assert settings.speedwire.serial == 1234567890

    def test_load_settings_from_env_with_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading settings with environment overrides."""

        # Set environment variables
        monkeypatch.setenv("MQTT_BROKER_HOST", "custom.broker.com")
        monkeypatch.setenv("SPEEDWIRE_SERIAL", "987654321")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = load_settings_from_env()

        assert settings.mqtt.broker_host == "custom.broker.com"
        assert settings.speedwire.serial == 987654321
        assert settings.log_level == "DEBUG"

    def test_load_settings_from_env_validation_error(self) -> None:
        """Test load_settings_from_env with validation error."""
        with patch("os.environ", {"MQTT_BROKER_PORT": "invalid_port"}):
            with pytest.raises(ConfigurationError) as exc_info:
                load_settings_from_env()

            error_msg = str(exc_info.value)
            assert "Environment configuration validation failed" in error_msg
            assert "mqtt" in error_msg or "broker_port" in error_msg
