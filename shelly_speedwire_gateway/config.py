"""Configuration management."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from shelly_speedwire_gateway.exceptions import ConfigurationError
from shelly_speedwire_gateway.models import (
    GatewaySettings,
    MQTTSettings,
    SpeedwireSettings,
)

logger = structlog.get_logger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate configuration from YAML file using Pydantic.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid or file cannot be read
    """
    config_path = Path(config_path)

    try:
        if not config_path.exists():
            logger.warning("Configuration file not found, creating default", path=str(config_path))
            return create_default_config(config_path)

        with config_path.open(encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if not raw_config:
            raise ConfigurationError("Configuration file is empty or invalid YAML")

        try:
            gateway_settings = GatewaySettings.model_validate(raw_config)
            logger.info("Configuration loaded and validated", path=str(config_path))

        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field_path}: {error['msg']}")

            raise ConfigurationError(
                "Configuration validation failed:\n" + "\n".join(error_details),
                config_section="validation",
                context={"errors": error_details},
            ) from e

        # R1720: Remove else after raise
        return dict(gateway_settings.model_dump())

    except (OSError, yaml.YAMLError) as e:
        raise ConfigurationError(
            f"Failed to load configuration file: {e}",
            config_section="file_io",
            context={"path": str(config_path), "error": str(e)},
        ) from e


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration dictionary using Pydantic models.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        GatewaySettings.model_validate(config)
        logger.debug("Configuration validation successful")

    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(f"{field_path}: {error['msg']}")

        raise ConfigurationError(
            "Configuration validation failed:\n" + "\n".join(error_details),
            context={"errors": error_details},
        ) from e


def create_default_config(config_path: Path) -> dict[str, Any]:
    """Create default configuration file using Pydantic models.

    Args:
        config_path: Path where to create the configuration file

    Returns:
        Default configuration as dictionary

    Raises:
        ConfigurationError: If file cannot be created
    """
    try:
        default_mqtt = MQTTSettings(base_topic="shellies/shelly3em-XXXXXXXXXXXXX")

        default_speedwire = SpeedwireSettings()

        default_gateway = GatewaySettings(mqtt=default_mqtt, speedwire=default_speedwire)

        config_dict = default_gateway.model_dump(exclude_none=True, by_alias=True, mode="json")

        config_path.parent.mkdir(parents=True, exist_ok=True)

        yaml_content = _create_commented_yaml(config_dict)

        with config_path.open("w", encoding="utf-8") as f:
            f.write(yaml_content)

        logger.info("Default configuration file created", path=str(config_path))

    except OSError as e:
        raise ConfigurationError(
            f"Failed to create configuration file: {e}",
            config_section="file_creation",
            context={"path": str(config_path), "error": str(e)},
        ) from e

    # R1720: Remove else after raise
    return dict(config_dict)


def _create_commented_yaml(config_dict: dict[str, Any]) -> str:
    """Create YAML configuration with comments.

    Args:
        config_dict: Configuration dictionary

    Returns:
        YAML string with comments
    """
    mqtt_cfg = config_dict["mqtt"]
    speedwire_cfg = config_dict["speedwire"]

    yaml_lines = [
        "# Shelly 3EM to SMA Speedwire Gateway Configuration",
        "#",
        "",
        "# MQTT broker settings",
        "mqtt:",
        f"  broker_host: {mqtt_cfg['broker_host']}  # MQTT broker hostname",
        f"  broker_port: {mqtt_cfg['broker_port']}  # MQTT broker port",
        f"  base_topic: {mqtt_cfg['base_topic']}",
        "    # Base topic (replace XXXXXXXXXXXXX with device ID)",
        f"  keepalive: {mqtt_cfg['keepalive']}  # MQTT keepalive interval in seconds",
        f"  invert_values: {str(mqtt_cfg['invert_values']).lower()}",
        "    # Invert power/energy values",
        f"  qos: {mqtt_cfg['qos']}  # MQTT Quality of Service (0, 1, or 2)",
        "  # username: your_mqtt_username  # Uncomment if authentication required",
        "  # password: your_mqtt_password  # Uncomment if authentication required",
        "",
        "# SMA Speedwire protocol settings",
        "speedwire:",
        f"  interval: {speedwire_cfg['interval']}  # Transmission interval in seconds",
        f"  use_broadcast: {str(speedwire_cfg['use_broadcast']).lower()}",
        "    # Use broadcast instead of multicast",
        f"  dualcast: {str(speedwire_cfg['dualcast']).lower()}",
        "    # Send both multicast and broadcast",
        f"  serial: {speedwire_cfg['serial']}  # SMA device serial number",
        f"  susy_id: {speedwire_cfg['susy_id']}  # SMA SUSy ID",
        "  unicast_targets: []  # List of specific IP addresses for unicast",
        "  # unicast_targets:",
        '  #   - "192.168.1.100"',
        '  #   - "192.168.1.101"',
        "",
        "# Application settings",
        f"log_level: {config_dict['log_level']}",
        "  # DEBUG, INFO, WARNING, ERROR, CRITICAL",
        f"log_format: {config_dict['log_format']}",
        "  # console, json, structured",
        f"enable_jit: {str(config_dict['enable_jit']).lower()}",
        "  # Enable JIT compilation",
        f"enable_monitoring: {str(config_dict['enable_monitoring']).lower()}",
        "  # Enable Prometheus metrics",
        f"metrics_port: {config_dict['metrics_port']}  # Port for metrics endpoint",
    ]

    return "\n".join(yaml_lines)


def load_settings_from_env() -> GatewaySettings:
    """Load configuration from environment variables.

    Returns:
        GatewaySettings instance populated from environment

    Raises:
        ConfigurationError: If required environment variables are missing
    """
    try:
        # Let pydantic-settings handle environment variable loading automatically
        gateway_settings = GatewaySettings()

        logger.info("Configuration loaded from environment variables")

    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(f"{field_path}: {error['msg']}")

        raise ConfigurationError(
            "Environment configuration validation failed:\n" + "\n".join(error_details),
            config_section="environment",
            context={"errors": error_details},
        ) from e

    # R1720: Remove else after raise
    return gateway_settings


def merge_config_sources(  # noqa: PLR0912
    file_config: dict[str, Any] | None = None,
    env_overrides: str = "enabled",
) -> GatewaySettings:
    """Merge configuration from file and environment variables.

    Args:
        file_config: Configuration from file
        env_overrides: Whether environment variables override file settings ("enabled" or "disabled")

    Returns:
        Merged GatewaySettings instance
    """
    if file_config:
        base_config = file_config.copy()

        if env_overrides == "enabled":
            try:
                # Only merge environment variables that are actually set
                env_vars_found = False

                # Check for MQTT environment variables
                mqtt_env_vars = {}
                for key in ["BROKER_HOST", "BROKER_PORT", "USERNAME", "PASSWORD", "BASE_TOPIC"]:
                    env_key = f"MQTT_{key}"
                    if env_key in os.environ:
                        mqtt_env_vars[key.lower()] = os.environ[env_key]
                        env_vars_found = True

                if mqtt_env_vars:
                    base_config.setdefault("mqtt", {}).update(mqtt_env_vars)

                # Check for Speedwire environment variables
                speedwire_env_vars = {}
                for key in ["SERIAL", "INTERVAL"]:
                    env_key = f"SPEEDWIRE_{key}"
                    if env_key in os.environ:
                        speedwire_env_vars[key.lower()] = os.environ[env_key]
                        env_vars_found = True

                if speedwire_env_vars:
                    base_config.setdefault("speedwire", {}).update(speedwire_env_vars)

                # Check for top-level environment variables
                for key in ["LOG_LEVEL", "LOG_FORMAT", "ENABLE_MONITORING"]:
                    if key in os.environ:
                        base_config[key.lower()] = os.environ[key]
                        env_vars_found = True

                if env_vars_found:
                    logger.debug("Applied environment variable overrides")

            except (KeyError, ValueError, OSError):
                logger.debug("No environment variable overrides found")

        try:
            return GatewaySettings.model_validate(base_config)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field_path}: {error['msg']}")

            raise ConfigurationError(
                "Configuration validation failed:\n" + "\n".join(error_details),
                config_section="merge",
                context={"errors": error_details},
            ) from e

    return load_settings_from_env()


def _deep_merge_dict(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Deep merge source dictionary into target dictionary.

    Args:
        target: Target dictionary to merge into
        source: Source dictionary to merge from
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = value


def setup_logging_from_config(config: dict[str, Any] | GatewaySettings) -> None:
    """Configure structured logging from configuration.

    Args:
        config: Configuration dictionary or GatewaySettings instance
    """
    if isinstance(config, GatewaySettings):
        log_level = config.log_level
        log_format = config.log_format
    else:
        log_level = config.get("log_level", "INFO")
        log_format = config.get("log_format", "structured")

    level = getattr(logging, log_level.upper(), logging.INFO)

    processors: list[Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
    ]

    match log_format:
        case "json":
            processors.append(structlog.processors.JSONRenderer())
        case "structured":
            processors.extend(
                [
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.dev.ConsoleRenderer(colors=True),
                ],
            )
        case "console":
            processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s", handlers=[logging.StreamHandler()])

    logging.getLogger("paho").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    logger.info("Logging configured", level=log_level, format=log_format)


def validate_mqtt_connection(mqtt_config: MQTTSettings) -> bool:
    """Validate MQTT connection settings.

    Args:
        mqtt_config: MQTT configuration to validate

    Returns:
        True if connection settings are valid
    """
    try:
        socket.gethostbyname(mqtt_config.broker_host)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        result = sock.connect_ex((mqtt_config.broker_host, mqtt_config.broker_port))
        sock.close()

        if result == 0:
            logger.info("MQTT broker connection test successful")
            return True

        logger.warning("MQTT broker connection test failed", result=result)

    except (socket.gaierror, TimeoutError, OSError) as e:
        logger.warning("MQTT broker validation failed", error=str(e))
        return False

    return False


def export_config_template(output_path: Path) -> None:
    """Export configuration template.

    Args:
        output_path: Path where to save the template
    """
    template_content = """# Shelly 3EM to SMA Speedwire Gateway - Configuration Template
# Configuration template for Shelly 3EM to SMA Speedwire Gateway

# ================================
# MQTT Broker Configuration
# ================================
mqtt:
  # Required: MQTT broker connection details
  broker_host: "localhost"              # MQTT broker hostname or IP address
  broker_port: 1883                     # MQTT broker port (1883=unencrypted, 8883=TLS)
  base_topic: "shellies/shellyem3-XXXXXXXXXXXXX"
    # Replace XXXXXXXXXXXXX with your device ID

  # Optional: Authentication (uncomment if needed)
  # username: "mqtt_user"                # MQTT username
  # password: "mqtt_password"            # MQTT password

  # Connection settings
  keepalive: 60                         # MQTT keepalive interval in seconds
  qos: 1                               # Quality of Service (0=fire&forget, 1=at_least_once, 2=exactly_once)

  # Data processing
  invert_values: false                  # Invert power/energy sign (useful for grid-tied systems)

# ================================
# SMA Speedwire Protocol Configuration
# ================================
speedwire:
  # Transmission settings
  interval: 1.0                         # Packet transmission interval in seconds

  # Network mode selection (choose one approach)
  use_broadcast: false                  # Use broadcast (255.255.255.255) instead of multicast
  dualcast: false                      # Send both multicast AND broadcast (increases network traffic)

  # SMA device emulation
  serial: 1234567890                   # SMA device serial number (must be unique on network)
  susy_id: 349                        # SMA System User ID (typically 349 for energy meters)

  # Targeted transmission (optional)
  unicast_targets: []                  # Send direct packets to specific IPs
  # unicast_targets:
  #   - "192.168.1.100"                # SMA inverter IP
  #   - "192.168.1.101"                # SMA energy manager IP

# ================================
# Application Settings
# ================================
log_level: "INFO"                      # DEBUG, INFO, WARNING, ERROR, CRITICAL
log_format: "structured"               # console, json, structured

# Tuning
enable_jit: true                       # Enable experimental JIT compilation
enable_monitoring: false               # Enable Prometheus metrics endpoint
metrics_port: 8080                     # Port for /metrics endpoint
"""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(template_content)

        logger.info("Configuration template exported", path=str(output_path))

    except OSError as e:
        raise ConfigurationError(
            f"Failed to export configuration template: {e}",
            context={"path": str(output_path), "error": str(e)},
        ) from e
