"""Prometheus metrics for monitoring gateway."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

if TYPE_CHECKING:
    from shelly_speedwire_gateway.models import GatewaySettings

logger = structlog.get_logger(__name__)

# Gateway info
gateway_info = Info("gateway", "Gateway information")

# Connection metrics
mqtt_connected = Gauge("mqtt_connected", "MQTT connection status (1=connected, 0=disconnected)")
speedwire_active = Gauge("speedwire_active", "Speedwire broadcast active (1=active, 0=inactive)")

# Data metrics
mqtt_messages_received = Counter("mqtt_messages_received_total", "Total MQTT messages received")
mqtt_messages_errors = Counter("mqtt_messages_errors_total", "Total MQTT message processing errors")
speedwire_packets_sent = Counter("speedwire_packets_sent_total", "Total Speedwire packets sent")
speedwire_packets_errors = Counter("speedwire_packets_errors_total", "Total Speedwire packet send errors")

# Timing metrics
mqtt_processing_time = Histogram("mqtt_processing_seconds", "Time to process MQTT messages")
speedwire_send_time = Histogram("speedwire_send_seconds", "Time to send Speedwire packets")

# Energy metrics
total_power = Gauge("total_power_watts", "Total power in watts")
phase_power = Gauge("phase_power_watts", "Power per phase in watts", ["phase"])
phase_voltage = Gauge("phase_voltage_volts", "Voltage per phase in volts", ["phase"])
phase_current = Gauge("phase_current_amps", "Current per phase in amps", ["phase"])
total_energy_consumed = Gauge("total_energy_consumed_kwh", "Total energy consumed in kWh")
total_energy_exported = Gauge("total_energy_exported_kwh", "Total energy exported in kWh")


def init_metrics(settings: GatewaySettings) -> None:
    """Initialize Prometheus metrics server.

    Args:
        settings: Gateway settings with metrics configuration
    """
    if not settings.enable_monitoring:
        logger.info("Monitoring disabled, skipping metrics initialization")
        return

    try:
        # Set gateway info
        gateway_info.info(
            {
                "version": "2.0.0",
                "mqtt_broker": settings.mqtt.broker_host,
                "mqtt_topic": settings.mqtt.base_topic,
                "speedwire_serial": str(settings.speedwire.serial),
            },
        )

        # Start metrics server
        start_http_server(settings.metrics_port)
        logger.info(
            "Metrics server started",
            port=settings.metrics_port,
            endpoint=f"http://localhost:{settings.metrics_port}/metrics",
        )
    except Exception as e:
        logger.exception("Failed to start metrics server", error=str(e))
        raise


def update_energy_metrics(data: dict) -> None:
    """Update energy-related metrics from Shelly data.

    Args:
        data: Shelly 3EM data dict
    """
    try:
        if "total_power" in data:
            total_power.set(data["total_power"])

        for phase_num in range(3):
            phase_label = ["A", "B", "C"][phase_num]

            if f"power_{phase_num}" in data:
                phase_power.labels(phase=phase_label).set(data[f"power_{phase_num}"])

            if f"voltage_{phase_num}" in data:
                phase_voltage.labels(phase=phase_label).set(data[f"voltage_{phase_num}"])

            if f"current_{phase_num}" in data:
                phase_current.labels(phase=phase_label).set(data[f"current_{phase_num}"])

        if "total_consumed" in data:
            total_energy_consumed.set(data["total_consumed"] / 1000)  # Convert Wh to kWh

        if "total_returned" in data:
            total_energy_exported.set(data["total_returned"] / 1000)  # Convert Wh to kWh

    except (KeyError, ValueError, TypeError) as e:
        logger.exception("Failed to update energy metrics", error=str(e))


async def shutdown_metrics() -> None:
    """Shutdown metrics server gracefully."""
    logger.info("Shutting down metrics server")
    # prometheus_client doesn't have async shutdown
    await asyncio.sleep(0)
