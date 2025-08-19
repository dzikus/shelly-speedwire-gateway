#!/usr/bin/env python3
"""Main entry point for Shelly 3EM to SMA Speedwire Gateway.

This module provides the command-line interface and main execution logic.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import structlog
import uvloop
import yaml

from shelly_speedwire_gateway.config import (
    create_default_config,
    load_config,
    validate_config,
)
from shelly_speedwire_gateway.constants import (
    DEFAULT_CONFIG_FILE,
    ERROR_INVALID_CONFIG,
    ERROR_KEYBOARD_INTERRUPT,
    ERROR_MQTT_CONNECTION,
    ERROR_NETWORK_FAILURE,
)
from shelly_speedwire_gateway.exceptions import (
    ConfigurationError,
    MQTTConnectionError,
    NetworkError,
)
from shelly_speedwire_gateway.gateway import Shelly3EMSpeedwireGateway

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


VERBOSE_LEVEL_DEBUG = 3
VERBOSE_LEVEL_INFO = 2
VERBOSE_LEVEL_WARNING = 1


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="shelly-speedwire-gateway",
        description="Shelly 3EM to SMA Speedwire Gateway",
        epilog="https://github.com/dzikus/shelly-speedwire-gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_FILE),
        metavar="PATH",
        help=f"Configuration file path (default: {DEFAULT_CONFIG_FILE})",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output except errors")

    parser.add_argument(
        "--log-format",
        choices=["console", "json", "structured"],
        default="console",
        help="Log output format",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

    parser.add_argument("--check-config", action="store_true", help="Validate configuration and exit")

    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")

    return parser


def setup_logging(args: argparse.Namespace) -> None:
    """Configure structured logging based on arguments."""
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= VERBOSE_LEVEL_DEBUG:  # PLR2004: Use constant
        level = logging.DEBUG
    elif args.verbose >= VERBOSE_LEVEL_INFO:  # PLR2004: Use constant
        level = logging.INFO
    elif args.verbose >= VERBOSE_LEVEL_WARNING:  # PLR2004: Use constant
        level = logging.WARNING
    else:
        level = logging.INFO

    processors: list[Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
    ]

    if args.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif args.log_format == "structured":
        processors.extend(
            [
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s", handlers=[logging.StreamHandler(sys.stderr)])

    logging.getLogger("paho").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def setup_event_loop() -> None:
    """Setup event loop for asyncio."""
    # Tune garbage collection
    gc.set_threshold(700, 10, 10)
    if gc.get_debug():
        gc.set_debug(0)
    if sys.platform != "win32":
        try:
            uvloop.install()
            logger.info("uvloop event loop installed")
        except ImportError:
            logger.debug("uvloop not available, using default event loop")


def validate_config_file(config_path: Path) -> bool:
    """Validate configuration file exists and is readable."""
    try:
        if not config_path.exists():
            logger.error("Configuration file not found", path=str(config_path))
            return False

        if not config_path.is_file():
            logger.error("Configuration path is not a file", path=str(config_path))
            return False

        if config_path.stat().st_size == 0:
            logger.error("Configuration file is empty", path=str(config_path))
            return False

        with Path.open(config_path, encoding="utf-8") as f:
            f.read()

        logger.debug("Configuration file validated", path=str(config_path))
    except PermissionError:
        logger.exception("Permission denied reading configuration", path=str(config_path))
        return False
    except OSError as e:
        logger.exception("Error reading configuration file", path=str(config_path), error=str(e))
        return False

    return True


def create_default_config_file(config_path: Path) -> bool:
    """Create a default configuration file."""
    try:
        if config_path.exists():
            response = input(f"Configuration file {config_path} exists. Overwrite? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                # T201: Using sys.stderr.write instead of print
                sys.stderr.write("Configuration creation cancelled.\n")
                return False

        create_default_config(config_path)
        logger.info("Default configuration created", path=str(config_path))
    except (OSError, ValueError) as e:
        logger.exception("Failed to create configuration", path=str(config_path), error=str(e))
        return False

    return True


def check_configuration(config_path: Path) -> bool:
    """Check configuration file validity."""
    try:
        logger.info("Checking configuration", path=str(config_path))

        if not validate_config_file(config_path):
            return False

        config = load_config(str(config_path))
        validate_config(config)

        logger.info("Configuration validation successful")

        mqtt_config = config.get("mqtt", {})
        speedwire_config = config.get("speedwire", {})

        logger.info(
            "Configuration summary",
            mqtt_broker=(f"{mqtt_config.get('broker_host')}:{mqtt_config.get('broker_port')}"),
            mqtt_topic=mqtt_config.get("base_topic"),
            tx_interval=speedwire_config.get("interval"),
            sma_serial=speedwire_config.get("serial"),
        )

    except ConfigurationError as e:
        logger.exception("Configuration validation failed", error=str(e))
        return False
    except (OSError, ValueError, yaml.YAMLError) as e:
        logger.exception("Unexpected configuration error", error=str(e))
        return False

    # R1705: Remove else after return
    return True


async def run_gateway(config_path: Path) -> int:
    """Run the main gateway application."""
    try:
        if not validate_config_file(config_path):
            return ERROR_INVALID_CONFIG

        gateway = Shelly3EMSpeedwireGateway(str(config_path))

        logger.info("Starting Shelly 3EM Speedwire Gateway")

        await gateway.run()

        logger.info("Gateway shutdown complete")

    except KeyboardInterrupt:
        logger.info("Gateway interrupted by user")
        return ERROR_KEYBOARD_INTERRUPT

    except ConfigurationError as e:
        logger.exception("Configuration error", error=str(e))
        return ERROR_INVALID_CONFIG

    except NetworkError as e:
        logger.exception("Network error", error=str(e))
        return ERROR_NETWORK_FAILURE

    except MQTTConnectionError as e:
        logger.exception("MQTT error", error=str(e))
        return ERROR_MQTT_CONNECTION

    except (OSError, ValueError) as e:
        logger.exception("Unexpected error", error=str(e))
        return 1

    return 0


async def async_main(argv: list[str] | None = None) -> int:
    """Async main function with error handling."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    setup_logging(args)

    if args.create_config:
        success = create_default_config_file(args.config)
        return 0 if success else 1

    if args.check_config:
        success = check_configuration(args.config)
        return 0 if success else 1

    return await run_gateway(args.config)


def main(argv: list[str] | None = None) -> int:
    """Main entry point with event loop setup."""
    try:
        setup_event_loop()
        return asyncio.run(async_main(argv))

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return ERROR_KEYBOARD_INTERRUPT
    except (OSError, ValueError) as e:
        logger.exception("Fatal error in main", error=str(e))
        return 1


def cli_main() -> NoReturn:
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    cli_main()
