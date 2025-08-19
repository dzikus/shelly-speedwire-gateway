"""Shelly 3EM to SMA Speedwire Gateway.

A Python gateway for bridging Shelly 3EM energy meters
to SMA Speedwire protocol for integration with SMA inverters.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Grzegorz Sterniczuk"
__license__ = "MIT"

from shelly_speedwire_gateway.exceptions import (
    ConfigurationError,
    DataValidationError,
    GatewayError,
    MQTTConnectionError,
    NetworkError,
    ProtocolError,
)
from shelly_speedwire_gateway.gateway import Shelly3EMSpeedwireGateway
from shelly_speedwire_gateway.models import GatewaySettings, PhaseData, Shelly3EMData

__all__ = [
    "ConfigurationError",
    "DataValidationError",
    "GatewayError",
    "GatewaySettings",
    "MQTTConnectionError",
    "NetworkError",
    "PhaseData",
    "ProtocolError",
    "Shelly3EMData",
    "Shelly3EMSpeedwireGateway",
    "__author__",
    "__license__",
    "__version__",
]
