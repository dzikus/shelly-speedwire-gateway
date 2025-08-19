"""Power calculation utilities for Shelly 3EM to SMA Speedwire Gateway.

This module provides utilities for calculating reactive power, apparent power,
and converting between different power units and formats required by the
SMA Speedwire protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shelly_speedwire_gateway.constants import (
    MIN_CURRENT_THRESHOLD,
    MIN_POWER_FACTOR_THRESHOLD,
    POWER_DECIMAL_PLACES,
    VALID_POWER_FACTOR_RANGE,
)
from shelly_speedwire_gateway.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class PowerComponents:
    """Power components calculation results."""

    active: float
    reactive: float
    apparent: float


@dataclass(frozen=True, slots=True)
class PowerSplit:
    """Import/export power split."""

    import_value: int
    export_value: int


def calculate_reactive_apparent(voltage: float, current: float, power_factor: float) -> tuple[float, float]:
    """Calculate reactive and apparent power from voltage, current, and power factor.

    Args:
        voltage: RMS voltage in Volts
        current: RMS current in Amperes
        power_factor: Power factor (-1.0 to 1.0)

    Returns:
        Tuple of (reactive_power_var, apparent_power_va)

    Raises:
        DataValidationError: If input values are invalid

    Note:
        - Reactive power is positive for inductive loads (lagging PF)
        - Reactive power is negative for capacitive loads (leading PF)
        - Apparent power is always positive
    """
    if voltage < 0:
        raise DataValidationError(
            "Voltage cannot be negative",
            field_name="voltage",
            field_value=voltage,
            valid_range="≥ 0",
        )

    if current < 0:
        raise DataValidationError(
            "Current cannot be negative",
            field_name="current",
            field_value=current,
            valid_range="≥ 0",
        )

    pf_min, pf_max = VALID_POWER_FACTOR_RANGE
    if not pf_min <= power_factor <= pf_max:
        raise DataValidationError(
            "Power factor out of valid range",
            field_name="power_factor",
            field_value=power_factor,
            valid_range=f"{pf_min} to {pf_max}",
        )

    if voltage <= 0 or current <= 0:
        return 0.0, 0.0

    apparent_power = voltage * current
    pf_abs = min(1.0, abs(power_factor))

    if pf_abs < MIN_POWER_FACTOR_THRESHOLD:
        reactive_power = apparent_power
    else:
        reactive_power = apparent_power * math.sqrt(1.0 - pf_abs * pf_abs)

    if power_factor < 0:
        reactive_power = -reactive_power

    return reactive_power, apparent_power


class PowerCalculator:
    """Utility class for electrical power calculations."""

    @staticmethod
    def calculate_power_components(voltage: float, current: float, power_factor: float) -> PowerComponents:
        """Calculate all power components from basic electrical parameters.

        Args:
            voltage: RMS voltage in Volts
            current: RMS current in Amperes
            power_factor: Power factor (-1.0 to 1.0)

        Returns:
            PowerComponents with active, reactive, and apparent power
        """
        reactive, apparent = calculate_reactive_apparent(voltage, current, power_factor)

        active = apparent * power_factor

        return PowerComponents(active=active, reactive=reactive, apparent=apparent)

    @classmethod
    def calculate_phase_power_components_list(cls, phases: list) -> list[PowerComponents]:
        """Calculate power components for multiple phases.

        Args:
            phases: List of PhaseData objects

        Returns:
            List of PowerComponents for each phase
        """
        return [cls.calculate_power_components(phase.voltage, phase.current, phase.pf) for phase in phases]

    @staticmethod
    def split_power(power: float) -> PowerSplit:
        """Split power value into import/export components for SMA protocol.

        The SMA Speedwire protocol represents power as separate import and export
        values, both expressed as positive integers in 0.1W units.

        Args:
            power: Power value in Watts (positive = import, negative = export)

        Returns:
            PowerSplit with import and export values in 0.1W units

        Note:
            - Positive power becomes import_value
            - Negative power becomes export_value (as positive)
            - Values are scaled by POWER_DECIMAL_PLACES (10 for 0.1W units)
        """
        scaled_power = power * POWER_DECIMAL_PLACES

        if power >= 0:
            return PowerSplit(import_value=round(scaled_power), export_value=0)

        return PowerSplit(import_value=0, export_value=round(-scaled_power))

    @staticmethod
    def split_power_tuple(power: float) -> tuple[int, int]:
        """Split power value into import/export tuple for backward compatibility.

        Args:
            power: Power value in Watts

        Returns:
            Tuple of (import_value, export_value) in 0.1W units
        """
        split = PowerCalculator.split_power(power)
        return split.import_value, split.export_value

    @staticmethod
    def calculate_phase_powers(phase_data: Any) -> dict[str, tuple[float, float]]:
        """Calculate reactive and apparent power for a single phase.

        Args:
            phase_data: PhaseData containing voltage, current, and power factor

        Returns:
            Dictionary with 'reactive' and 'apparent' power values as tuples
        """
        reactive, apparent = calculate_reactive_apparent(phase_data.voltage, phase_data.current, phase_data.pf)

        return {"reactive": (reactive, 0.0), "apparent": (apparent, 0.0)}

    @staticmethod
    def get_phase_powers(data: Any) -> dict[str, tuple[float, float]]:
        """Calculate reactive and apparent powers for all phases.

        Args:
            data: Shelly3EMData containing all phase information

        Returns:
            Dictionary mapping phase names to (reactive, apparent) power tuples
        """
        phases = [("a", data.a), ("b", data.b), ("c", data.c)]

        result = {}
        for phase_name, phase_data in phases:
            reactive, apparent = calculate_reactive_apparent(phase_data.voltage, phase_data.current, phase_data.pf)
            result[phase_name] = (reactive, apparent)

        return result

    @staticmethod
    def calculate_total_powers(data: Any) -> PowerComponents:
        """Calculate total power components across all phases.

        Args:
            data: Shelly3EMData containing all phase information

        Returns:
            PowerComponents with total active, reactive, and apparent power
        """
        phase_powers = PowerCalculator.get_phase_powers(data)

        total_reactive = sum(powers[0] for powers in phase_powers.values())
        total_apparent = sum(powers[1] for powers in phase_powers.values())
        total_active = data.total_power

        return PowerComponents(active=total_active, reactive=total_reactive, apparent=total_apparent)

    @staticmethod
    def validate_power_triangle(active: float, reactive: float, apparent: float, tolerance: float = 0.01) -> bool:
        """Validate that power values form a valid power triangle.

        Args:
            active: Active power in Watts
            reactive: Reactive power in VAr
            apparent: Apparent power in VA
            tolerance: Relative tolerance for validation (default 1%)

        Returns:
            True if power triangle is valid within tolerance

        Note:
            For a valid power triangle: S² = P² + Q²
        """
        if apparent <= 0:
            return active == 0 and reactive == 0

        calculated_apparent = math.sqrt(active * active + reactive * reactive)
        relative_error = abs(calculated_apparent - apparent) / apparent

        return relative_error <= tolerance

    @staticmethod
    def calculate_power_factor_from_powers(active: float, apparent: float) -> float:
        """Calculate power factor from active and apparent power.

        Args:
            active: Active power in Watts
            apparent: Apparent power in VA

        Returns:
            Power factor (cosine of phase angle)
        """
        if apparent <= 0:
            return 1.0

        pf = active / apparent
        pf_min, pf_max = VALID_POWER_FACTOR_RANGE

        return max(pf_min, min(pf_max, pf))

    @staticmethod
    def calculate_current_from_power(active_power: float, voltage: float, power_factor: float) -> float:
        """Calculate current from active power, voltage, and power factor.

        Args:
            active_power: Active power in Watts
            voltage: RMS voltage in Volts
            power_factor: Power factor

        Returns:
            RMS current in Amperes
        """
        if voltage <= 0 or abs(power_factor) < MIN_CURRENT_THRESHOLD:
            return 0.0

        return abs(active_power) / (voltage * abs(power_factor))

    @staticmethod
    def watts_to_w01_units(watts: float) -> int:
        """Convert watts to 0.1W units used in SMA protocol.

        Args:
            watts: Power in watts

        Returns:
            Power in 0.1W units (deciwatts)
        """
        return round(watts * POWER_DECIMAL_PLACES)

    @staticmethod
    def w01_units_to_watts(w01_units: int) -> float:
        """Convert 0.1W units to watts.

        Args:
            w01_units: Power in 0.1W units

        Returns:
            Power in watts
        """
        return float(w01_units) / POWER_DECIMAL_PLACES

    @staticmethod
    def calculate_balanced_three_phase_power(
        line_voltage: float,
        line_current: float,
        power_factor: float,
    ) -> PowerComponents:
        """Calculate balanced three-phase power from line values.

        Args:
            line_voltage: Line-to-line voltage in Volts
            line_current: Line current in Amperes
            power_factor: Power factor

        Returns:
            PowerComponents for total three-phase power
        """
        active = math.sqrt(3) * line_voltage * line_current * power_factor
        apparent = math.sqrt(3) * line_voltage * line_current
        reactive = apparent * math.sqrt(1.0 - power_factor * power_factor)

        if power_factor < 0:
            reactive = -reactive

        return PowerComponents(active=active, reactive=reactive, apparent=apparent)
