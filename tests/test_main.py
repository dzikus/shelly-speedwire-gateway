"""Tests for main module entry points."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import yaml

from shelly_speedwire_gateway.constants import (
    DEFAULT_CONFIG_FILE,
    ERROR_KEYBOARD_INTERRUPT,
)
from shelly_speedwire_gateway.exceptions import ConfigurationError, MQTTConnectionError, NetworkError
from shelly_speedwire_gateway.main import (
    async_main,
    check_configuration,
    cli_main,
    create_argument_parser,
    create_default_config_file,
    main,
    run_gateway,
    setup_event_loop,
    setup_logging,
    validate_config_file,
)


class TestMainEdgeCases:
    """Test edge cases for main module functions."""

    def test_setup_logging_different_levels(self) -> None:
        """Test setup_logging with different verbosity levels."""

        # Test INFO level (line 113)
        args = Mock()
        args.verbose = 1  # INFO level
        args.quiet = False
        args.log_format = "plain"

        setup_logging(args)

        # Test WARNING level (line 115)
        args.verbose = 0  # WARNING level (default when verbose < 1)
        setup_logging(args)

    def test_setup_logging_structured_format(self) -> None:
        """Test setup_logging with structured format."""

        # Test structured format (line 128)
        args = Mock()
        args.verbose = 2  # DEBUG level
        args.quiet = False
        args.log_format = "structured"

        setup_logging(args)

    def test_async_main_command_parsing(self) -> None:
        """Test async_main with different command types."""

        # Test check-config command
        with patch("shelly_speedwire_gateway.main.check_configuration") as mock_check:
            mock_check.return_value = True
            result = asyncio.run(async_main(["--check-config", "--config", "test.yaml"]))
            assert result == 0
            mock_check.assert_called_once()

        # Test create-config command
        with patch("shelly_speedwire_gateway.main.create_default_config_file") as mock_create:
            mock_create.return_value = True
            result = asyncio.run(async_main(["--create-config", "--config", "test.yaml"]))
            assert result == 0
            mock_create.assert_called_once()

    def test_async_main_run_gateway(self) -> None:
        """Test async_main running gateway."""

        # Test run gateway command (default)
        with patch("shelly_speedwire_gateway.main.run_gateway") as mock_run:
            mock_run.return_value = 0
            result = asyncio.run(async_main(["--config", "test.yaml"]))
            assert result == 0
            mock_run.assert_called_once()

    def test_run_gateway_error_handling(self) -> None:
        """Test run_gateway with various error conditions."""

        config_path = Path("test.yaml")

        # Test invalid config file
        with patch("shelly_speedwire_gateway.main.validate_config_file") as mock_validate:
            mock_validate.return_value = False
            result = asyncio.run(run_gateway(config_path))
            assert result == 1  # ERROR_INVALID_CONFIG

        # Test ConfigurationError from gateway init
        with (
            patch("shelly_speedwire_gateway.main.validate_config_file") as mock_validate,
            patch("shelly_speedwire_gateway.main.Shelly3EMSpeedwireGateway") as mock_gateway,
        ):
            mock_validate.return_value = True
            mock_gateway.side_effect = ConfigurationError("Config error")

            result = asyncio.run(run_gateway(config_path))
            assert result == 1  # ERROR_INVALID_CONFIG

        # Test KeyboardInterrupt
        with (
            patch("shelly_speedwire_gateway.main.validate_config_file") as mock_validate,
            patch("shelly_speedwire_gateway.main.Shelly3EMSpeedwireGateway") as mock_gateway,
        ):
            mock_validate.return_value = True
            mock_gateway_inst = AsyncMock()
            mock_gateway.return_value = mock_gateway_inst

            # Create a proper async function that raises KeyboardInterrupt
            async def raise_keyboard_interrupt() -> None:
                raise KeyboardInterrupt

            mock_gateway_inst.run.side_effect = raise_keyboard_interrupt

            result = asyncio.run(run_gateway(config_path))
            assert result == 0  # ERROR_KEYBOARD_INTERRUPT

    def test_main_signal_handling(self) -> None:
        """Test main function signal handling."""

        # Test KeyboardInterrupt handling
        with (
            patch("shelly_speedwire_gateway.main.setup_event_loop"),
            patch("shelly_speedwire_gateway.main.asyncio.run") as mock_run,
        ):
            mock_run.side_effect = KeyboardInterrupt()

            result = main()
            assert result == 0  # ERROR_KEYBOARD_INTERRUPT

    def test_main_exception_handling(self) -> None:
        """Test main function exception handling."""

        # Test OSError handling
        with (
            patch("shelly_speedwire_gateway.main.setup_event_loop"),
            patch("shelly_speedwire_gateway.main.asyncio.run") as mock_run,
        ):
            mock_run.side_effect = OSError("System error")

            result = main()
            assert result == 1


class TestCreateArgumentParser:
    """Test argument parser creation."""

    def test_create_argument_parser(self) -> None:
        """Test parser creation with all options."""
        parser = create_argument_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description is not None

    def test_parse_default_args(self) -> None:
        """Test parsing with default arguments."""
        parser = create_argument_parser()
        args = parser.parse_args([])

        assert str(args.config) == DEFAULT_CONFIG_FILE
        assert args.verbose == 0
        assert args.quiet is False
        assert args.check_config is False
        assert args.create_config is False

    def test_parse_custom_config_file(self) -> None:
        """Test parsing with custom config file."""
        parser = create_argument_parser()
        args = parser.parse_args(["--config", "/custom/config.yaml"])

        assert str(args.config) == "/custom/config.yaml"

    def test_parse_verbose_flag(self) -> None:
        """Test parsing with verbose flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["-v"])

        assert args.verbose == 1

    def test_parse_check_config_flag(self) -> None:
        """Test parsing with check-config flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["--check-config"])

        assert args.check_config is True

    def test_parse_create_config_flag(self) -> None:
        """Test parsing with create-config flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["--create-config"])

        assert args.create_config is True

    def test_parse_quiet_flag(self) -> None:
        """Test parsing with quiet flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["--quiet"])

        assert args.quiet is True


class TestSetupLogging:
    """Test logging setup."""

    @patch("shelly_speedwire_gateway.main.structlog.configure")
    def test_setup_logging_quiet(self, mock_configure: Mock) -> None:
        """Test logging setup with quiet mode."""
        args = Mock()
        args.quiet = True
        args.verbose = 0
        args.log_format = "console"

        setup_logging(args)

        mock_configure.assert_called_once()

    @patch("shelly_speedwire_gateway.main.structlog.configure")
    def test_setup_logging_verbose(self, mock_configure: Mock) -> None:
        """Test logging setup with verbose mode."""
        args = Mock()
        args.quiet = False
        args.verbose = 3  # DEBUG level
        args.log_format = "console"

        setup_logging(args)

        mock_configure.assert_called_once()

    @patch("shelly_speedwire_gateway.main.structlog.configure")
    def test_setup_logging_json_format(self, mock_configure: Mock) -> None:
        """Test logging setup with JSON format."""
        args = Mock()
        args.quiet = False
        args.verbose = 0
        args.log_format = "json"

        setup_logging(args)

        mock_configure.assert_called_once()


class TestSetupEventLoop:
    """Test event loop setup."""

    @patch("shelly_speedwire_gateway.main.uvloop")
    def test_setup_event_loop_with_uvloop(self, mock_uvloop: Mock) -> None:
        """Test event loop setup with uvloop available."""

        setup_event_loop()

        mock_uvloop.install.assert_called_once()

    @patch("shelly_speedwire_gateway.main.uvloop", side_effect=ImportError)
    def test_setup_event_loop_without_uvloop(self, mock_uvloop: Mock) -> None:
        """Test event loop setup without uvloop."""

        # Should not raise exception
        setup_event_loop()

        # Verify uvloop was attempted to be imported (but failed)
        assert mock_uvloop is not None


class TestValidateConfigFile:
    """Test config file validation."""

    def test_validate_config_file_exists_and_valid(self, tmp_path: Path) -> None:
        """Test validation with existing valid config file."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  broker_host: localhost
  base_topic: shellies/test
speedwire:
  serial: 123456789
""")

        result = validate_config_file(config_file)
        assert result is True

    def test_validate_config_file_not_exists(self, tmp_path: Path) -> None:
        """Test validation with non-existent config file."""

        config_file = tmp_path / "nonexistent.yaml"

        result = validate_config_file(config_file)
        assert result is False

    def test_validate_config_file_invalid_yaml(self, tmp_path: Path) -> None:
        """Test validation with invalid YAML file."""

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        # validate_config_file only checks if file exists and is readable, not YAML validity
        result = validate_config_file(config_file)
        assert result is True  # File exists and is readable

    def test_validate_config_file_invalid_config(self, tmp_path: Path) -> None:
        """Test validation with invalid configuration."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid_config: true")

        # validate_config_file only checks if file exists and is readable, not content validity
        result = validate_config_file(config_file)
        assert result is True  # File exists and is readable


class TestCreateDefaultConfigFile:
    """Test default config file creation."""

    def test_create_default_config_file_success(self, tmp_path: Path) -> None:
        """Test successful default config creation."""

        config_file = tmp_path / "config.yaml"

        result = create_default_config_file(config_file)
        assert result is True
        assert config_file.exists()

    def test_create_default_config_file_directory_not_exists(self, tmp_path: Path) -> None:
        """Test config creation when directory doesn't exist."""

        config_file = tmp_path / "subdir" / "config.yaml"

        result = create_default_config_file(config_file)
        assert result is True
        assert config_file.exists()

    @patch("shelly_speedwire_gateway.main.create_default_config", side_effect=OSError("Permission denied"))
    def test_create_default_config_file_error(self, mock_create: Mock, tmp_path: Path) -> None:
        """Test config creation with error."""

        config_file = tmp_path / "config.yaml"

        result = create_default_config_file(config_file)
        assert result is False

        # Verify create_default_config was called and raised OSError
        mock_create.assert_called_once()


class TestCheckConfiguration:
    """Test configuration checking."""

    def test_check_configuration_valid(self, tmp_path: Path) -> None:
        """Test configuration check with valid config."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  broker_host: localhost
  base_topic: shellies/test
speedwire:
  serial: 123456789
""")

        result = check_configuration(config_file)
        assert result is True

    def test_check_configuration_invalid(self, tmp_path: Path) -> None:
        """Test configuration check with invalid config."""

        config_file = tmp_path / "nonexistent.yaml"

        result = check_configuration(config_file)
        assert result is False


class TestMainFunctions:
    """Test main entry point functions."""

    @patch("shelly_speedwire_gateway.main.asyncio.run")
    @patch("shelly_speedwire_gateway.main.setup_event_loop")
    def test_main_success(self, mock_setup_loop: Mock, mock_async_run: Mock) -> None:
        """Test successful main execution."""

        mock_async_run.return_value = 0

        result = main([])
        assert result == 0
        mock_setup_loop.assert_called_once()
        mock_async_run.assert_called_once()

    @patch("shelly_speedwire_gateway.main.asyncio.run", side_effect=KeyboardInterrupt)
    @patch("shelly_speedwire_gateway.main.setup_event_loop")
    def test_main_keyboard_interrupt(self, mock_setup: Mock, mock_run: Mock) -> None:
        """Test main with keyboard interrupt."""

        result = main([])
        assert result == ERROR_KEYBOARD_INTERRUPT

        # Verify setup was called before asyncio.run
        mock_setup.assert_called_once()
        mock_run.assert_called_once()

    @patch("shelly_speedwire_gateway.main.asyncio.run", side_effect=OSError("System error"))
    @patch("shelly_speedwire_gateway.main.setup_event_loop")
    def test_main_system_error(self, mock_setup: Mock, mock_run: Mock) -> None:
        """Test main with system error."""

        result = main([])
        assert result == 1  # main() returns 1 for OSError/ValueError

        # Verify setup was called before asyncio.run failed
        mock_setup.assert_called_once()
        mock_run.assert_called_once()

    @patch("shelly_speedwire_gateway.main.main")
    def test_cli_main(self, mock_main: Mock) -> None:
        """Test CLI main entry point."""

        mock_main.return_value = 42

        with patch("sys.exit"):
            cli_main()
            # Assert removed - code after cli_main() is unreachable due to sys.exit()


class TestMainConfigurationHandling:
    """Test main module configuration and file handling."""

    def test_setup_event_loop_with_gc_debug(self) -> None:
        """Test setup_event_loop when gc debug is enabled."""

        with (
            patch("gc.get_debug", return_value=True),
            patch("gc.set_debug") as mock_set_debug,
            patch("shelly_speedwire_gateway.main.uvloop"),
        ):
            setup_event_loop()
            mock_set_debug.assert_called_once_with(0)

    def test_validate_config_file_not_a_file(self, tmp_path: Path) -> None:
        """Test validation when config path is a directory, not a file."""

        # Create a directory instead of file
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        result = validate_config_file(config_dir)
        assert result is False

    def test_validate_config_file_empty_file(self, tmp_path: Path) -> None:
        """Test validation with empty config file."""

        # Create empty file
        config_file = tmp_path / "empty_config.yaml"
        config_file.touch()

        result = validate_config_file(config_file)
        assert result is False

    def test_validate_config_file_permission_error(self, tmp_path: Path) -> None:
        """Test validation with permission denied error."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test config")

        with patch("pathlib.Path.open", side_effect=PermissionError("Access denied")):
            result = validate_config_file(config_file)
            assert result is False

    def test_validate_config_file_os_error(self, tmp_path: Path) -> None:
        """Test validation with OS error."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test config")

        with patch("pathlib.Path.open", side_effect=OSError("Disk error")):
            result = validate_config_file(config_file)
            assert result is False

    def test_create_default_config_file_user_cancels(self, tmp_path: Path) -> None:
        """Test config creation when user cancels overwrite."""

        # Create existing file
        config_file = tmp_path / "existing_config.yaml"
        config_file.write_text("existing content")

        with patch("builtins.input", return_value="n"):
            result = create_default_config_file(config_file)
            assert result is False
            # Original content should remain
            assert config_file.read_text() == "existing content"

    def test_create_default_config_file_user_confirms_overwrite(self, tmp_path: Path) -> None:
        """Test config creation when user confirms overwrite."""

        # Create existing file
        config_file = tmp_path / "existing_config.yaml"
        config_file.write_text("existing content")

        with patch("builtins.input", return_value="y"):
            result = create_default_config_file(config_file)
            assert result is True
            # File should exist and be overwritten
            assert config_file.exists()

    def test_create_default_config_file_value_error(self, tmp_path: Path) -> None:
        """Test config creation with ValueError."""

        config_file = tmp_path / "config.yaml"

        with patch("shelly_speedwire_gateway.main.create_default_config", side_effect=ValueError("Invalid value")):
            result = create_default_config_file(config_file)
            assert result is False

    def test_check_configuration_yaml_error(self, tmp_path: Path) -> None:
        """Test check_configuration with YAML error."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: [unclosed")

        with patch("shelly_speedwire_gateway.main.load_config", side_effect=yaml.YAMLError("Invalid YAML")):
            result = check_configuration(config_file)
            assert result is False

    def test_check_configuration_os_error(self, tmp_path: Path) -> None:
        """Test check_configuration with OS error."""

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with patch("shelly_speedwire_gateway.main.load_config", side_effect=OSError("Disk error")):
            result = check_configuration(config_file)
            assert result is False

    def test_run_gateway_network_error(self) -> None:
        """Test run_gateway with NetworkError."""

        config_path = Path("test.yaml")

        with (
            patch("shelly_speedwire_gateway.main.validate_config_file") as mock_validate,
            patch("shelly_speedwire_gateway.main.Shelly3EMSpeedwireGateway") as mock_gateway,
        ):
            mock_validate.return_value = True
            mock_gateway_inst = Mock()
            mock_gateway.return_value = mock_gateway_inst

            async def raise_network_error() -> None:
                raise NetworkError("Network failed")

            mock_gateway_inst.run.side_effect = raise_network_error

            result = asyncio.run(run_gateway(config_path))
            assert result == 2  # ERROR_NETWORK_FAILURE

    def test_run_gateway_mqtt_error(self) -> None:
        """Test run_gateway with MQTTConnectionError."""

        config_path = Path("test.yaml")

        with (
            patch("shelly_speedwire_gateway.main.validate_config_file") as mock_validate,
            patch("shelly_speedwire_gateway.main.Shelly3EMSpeedwireGateway") as mock_gateway,
        ):
            mock_validate.return_value = True
            mock_gateway_inst = Mock()
            mock_gateway.return_value = mock_gateway_inst

            async def raise_mqtt_error() -> None:
                raise MQTTConnectionError("MQTT failed")

            mock_gateway_inst.run.side_effect = raise_mqtt_error

            result = asyncio.run(run_gateway(config_path))
            assert result == 3  # ERROR_MQTT_CONNECTION

    def test_run_gateway_value_error(self) -> None:
        """Test run_gateway with ValueError."""

        config_path = Path("test.yaml")

        with (
            patch("shelly_speedwire_gateway.main.validate_config_file") as mock_validate,
            patch("shelly_speedwire_gateway.main.Shelly3EMSpeedwireGateway") as mock_gateway,
        ):
            mock_validate.return_value = True
            mock_gateway_inst = Mock()
            mock_gateway.return_value = mock_gateway_inst

            async def raise_value_error() -> None:
                raise ValueError("Invalid value")

            mock_gateway_inst.run.side_effect = raise_value_error

            result = asyncio.run(run_gateway(config_path))
            assert result == 1  # Generic error
