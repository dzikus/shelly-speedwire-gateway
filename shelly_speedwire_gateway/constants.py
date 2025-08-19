"""Constants for Shelly 3EM to SMA Speedwire Gateway.

This module contains all physical, network, protocol constants
and validation limits.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# NETWORK CONSTANTS
# =============================================================================

# Network addresses and ports
BROADCAST_IP: Final[str] = "255.255.255.255"
MULTICAST_GROUP: Final[str] = "239.12.255.254"
SMA_PORT: Final[int] = 9522
DEFAULT_METRICS_PORT: Final[int] = 8080

# Network interface and binding
LOCALHOST_PREFIX: Final[str] = "127."
DEFAULT_IPV4_ADDR: Final[str] = "0.0.0.0"  # noqa: S104


# =============================================================================
# MQTT CONSTANTS
# =============================================================================


# Topic patterns
EMETER_TOPIC_PATTERN: Final[str] = "/emeter/+/+"
ONLINE_TOPIC_SUFFIX: Final[str] = "/online"

# Message processing
MIN_TOPIC_PARTS_FOR_PARSING: Final[int] = 2
MIN_TOPIC_PARTS_FOR_EMETER: Final[int] = 4

# =============================================================================
# SMA SPEEDWIRE PROTOCOL CONSTANTS
# =============================================================================

# Protocol signatures and IDs
SMA_SIGNATURE: Final[bytes] = b"SMA\x00"
TAG0_ID: Final[int] = 0x02A0
DATA2_ID: Final[int] = 0x0010
END_ID: Final[int] = 0x0000
PROTO_EMETER: Final[int] = 0x6069
PROTOCOL_DISCOVERY: Final[int] = 0x6081
DISCOVERY_RESPONSE: Final[bytes] = bytes.fromhex("534d4100000402a000000001000200000001")
PROTO_BYTES_SIZE: Final[int] = 2

# Device configuration
DEFAULT_SUSY_ID: Final[int] = 0x015D
DEFAULT_SERIAL: Final[int] = 1234567890
SOFTWARE_VERSION_COMPONENTS: Final[int] = 4

# =============================================================================
# PHYSICAL AND ELECTRICAL CONSTANTS
# =============================================================================

# Power calculation thresholds
MIN_CURRENT_THRESHOLD: Final[float] = 0.001
POWER_CONSISTENCY_THRESHOLD: Final[float] = 0.1

# Unit conversions and scaling
WH_TO_WS_MULTIPLIER: Final[int] = 3600  # Wh to Ws conversion
POWER_SCALE_FACTOR: Final[float] = 0.1  # 0.1W units
VOLTAGE_SCALE_FACTOR: Final[int] = 1000  # mV units
CURRENT_SCALE_FACTOR: Final[int] = 1000  # mA units
FREQUENCY_SCALE_FACTOR: Final[int] = 1000  # mHz units
PF_SCALE_FACTOR: Final[int] = 1000  # Power factor scaling
POWER_DECIMAL_PLACES: Final[int] = 10

# Phase configuration
PHASE_NAMES: Final[tuple[str, ...]] = ("l1", "l2", "l3")
PHASE_LETTERS: Final[tuple[str, ...]] = ("a", "b", "c")
PHASE_B_NUM: Final[int] = 2

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Validation ranges
VALID_POWER_FACTOR_RANGE: Final[tuple[float, float]] = (-1.0, 1.0)
VALID_PORT_RANGE: Final[tuple[int, int]] = (1, 65535)
MIN_PARTS_COUNT: Final[int] = 2

# =============================================================================
# TIMING CONSTANTS
# =============================================================================

# Intervals and timeouts
DISCOVERY_LOOP_SLEEP: Final[float] = 0.01
DATA_RECEIVE_TIMEOUT: Final[float] = 0.05

# =============================================================================
# LOGGING AND CONFIGURATION
# =============================================================================

VERBOSE_LEVEL_DEBUG: Final[int] = 3
VERBOSE_LEVEL_INFO: Final[int] = 2
VERBOSE_LEVEL_WARNING: Final[int] = 1

# Configuration files
DEFAULT_CONFIG_FILE: Final[str] = "shelly_speedwire_gateway_config.yaml"

# =============================================================================
# ERROR CODES
# =============================================================================

# Network buffer sizes
RECEIVE_BUFFER_SIZE: Final[int] = 2048
MIN_POWER_FACTOR_THRESHOLD: Final[float] = 0.01

ERROR_INVALID_CONFIG: Final[int] = 1
ERROR_NETWORK_FAILURE: Final[int] = 2
ERROR_MQTT_CONNECTION: Final[int] = 3
ERROR_KEYBOARD_INTERRUPT: Final[int] = 0


# =============================================================================
# OBIS CHANNELS MAPPING
# =============================================================================

OBIS_CHANNELS: Final[dict[str, int]] = {
    # Total values
    "total_energy_import": 1,
    "total_energy_export": 2,
    "total_reactive_energy_q1": 3,
    "total_reactive_energy_q2": 4,
    "total_power_import": 1,
    "total_power_export": 2,
    "total_reactive_power_q1": 3,
    "total_reactive_power_q2": 4,
    "total_apparent_power_import": 9,
    "total_apparent_power_export": 10,
    "power_factor": 13,
    "frequency": 14,
    # L1 Phase values
    "l1_energy_import": 21,
    "l1_energy_export": 22,
    "l1_reactive_energy_q1": 23,
    "l1_reactive_energy_q2": 24,
    "l1_power_import": 21,
    "l1_power_export": 22,
    "l1_reactive_power_q1": 23,
    "l1_reactive_power_q2": 24,
    "l1_apparent_power_import": 29,
    "l1_apparent_power_export": 30,
    "l1_current": 31,
    "l1_voltage": 32,
    "l1_power_factor": 33,
    # L2 Phase values
    "l2_energy_import": 41,
    "l2_energy_export": 42,
    "l2_reactive_energy_q1": 43,
    "l2_reactive_energy_q2": 44,
    "l2_power_import": 41,
    "l2_power_export": 42,
    "l2_reactive_power_q1": 43,
    "l2_reactive_power_q2": 44,
    "l2_apparent_power_import": 49,
    "l2_apparent_power_export": 50,
    "l2_current": 51,
    "l2_voltage": 52,
    "l2_power_factor": 53,
    # L3 Phase values
    "l3_energy_import": 61,
    "l3_energy_export": 62,
    "l3_reactive_energy_q1": 63,
    "l3_reactive_energy_q2": 64,
    "l3_power_import": 61,
    "l3_power_export": 62,
    "l3_reactive_power_q1": 63,
    "l3_reactive_power_q2": 64,
    "l3_apparent_power_import": 69,
    "l3_apparent_power_export": 70,
    "l3_current": 71,
    "l3_voltage": 72,
    "l3_power_factor": 73,
}

# =============================================================================
# TEST CONSTANTS
# =============================================================================

TEST_UNICAST_COUNT: Final[int] = 2
TEST_POWER_VALUE: Final[float] = 1000.0
TEST_VOLTAGE_VALUE: Final[float] = 230.0
TEST_CURRENT_VALUE: Final[float] = 4.35
TEST_POWER_FACTOR: Final[float] = 0.95
TEST_ENERGY_CONSUMED: Final[float] = 1500.0
TEST_ENERGY_EXPORTED: Final[float] = 250.0
TEST_FREQUENCY: Final[float] = 50.0
TEST_TOLERANCE: Final[float] = 0.01
TEST_MQTT_PORT: Final[int] = 1883
TEST_KEEPALIVE: Final[int] = 60
TEST_QOS: Final[int] = 1
TEST_SERIAL: Final[int] = 1234567890
TEST_SUSY_ID: Final[int] = 349
TEST_PHASE_COUNT: Final[int] = 3
TEST_ROUNDING_POWER: Final[float] = 1000.123
TEST_ROUNDING_VOLTAGE: Final[float] = 230.988
TEST_ROUNDING_PF: Final[float] = 0.9877
TEST_TOTAL_POWER: Final[float] = 3000.0
TEST_POWER_A: Final[float] = 1000.0
TEST_POWER_B: Final[float] = 1200.0
TEST_POWER_C: Final[float] = 800.0
TEST_CONSUMED_TOTAL: Final[float] = 1500.0
TEST_EXPORTED_TOTAL: Final[float] = 150.0
TEST_PF_MIN: Final[float] = 0.85
TEST_PF_MAX: Final[float] = 0.95
TEST_SERIAL_ALT: Final[int] = 4294967295
