"""Tests for power calculation utilities."""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from shelly_speedwire_gateway.constants import MIN_POWER_FACTOR_THRESHOLD
from shelly_speedwire_gateway.exceptions import DataValidationError
from shelly_speedwire_gateway.power_calculator import (
    PowerCalculator,
    PowerComponents,
    PowerSplit,
    calculate_reactive_apparent,
)


class TestCalculateReactiveApparent:
    """Test standalone reactive and apparent power calculation function."""

    def test_valid_power_calculation(self) -> None:
        """Test calculation with valid inputs."""
        voltage = 230.0
        current = 5.0
        power_factor = 0.9

        reactive, apparent = calculate_reactive_apparent(voltage, current, power_factor)

        expected_apparent = voltage * current  # 1150.0
        expected_reactive = expected_apparent * math.sqrt(1.0 - power_factor * power_factor)

        assert apparent == pytest.approx(expected_apparent)
        assert reactive == pytest.approx(expected_reactive, rel=1e-6)

    def test_zero_voltage_current(self) -> None:
        """Test with zero voltage or current."""
        reactive, apparent = calculate_reactive_apparent(0.0, 5.0, 0.9)
        assert reactive == 0.0
        assert apparent == 0.0

        reactive, apparent = calculate_reactive_apparent(230.0, 0.0, 0.9)
        assert reactive == 0.0
        assert apparent == 0.0

    def test_low_power_factor_threshold(self) -> None:
        """Test power factor below minimum threshold."""
        voltage = 230.0
        current = 5.0
        power_factor = MIN_POWER_FACTOR_THRESHOLD / 2  # Below threshold

        reactive, apparent = calculate_reactive_apparent(voltage, current, power_factor)

        expected_apparent = voltage * current
        assert apparent == pytest.approx(expected_apparent)
        assert reactive == pytest.approx(expected_apparent)  # Full apparent power

    def test_negative_power_factor(self) -> None:
        """Test with negative power factor (capacitive load)."""
        voltage = 230.0
        current = 5.0
        power_factor = -0.8

        reactive, apparent = calculate_reactive_apparent(voltage, current, power_factor)

        expected_apparent = voltage * current
        expected_reactive = -expected_apparent * math.sqrt(1.0 - 0.8 * 0.8)

        assert apparent == pytest.approx(expected_apparent)
        assert reactive == pytest.approx(expected_reactive, rel=1e-6)
        assert reactive < 0  # Negative for capacitive load

    def test_invalid_voltage(self) -> None:
        """Test with invalid negative voltage."""
        with pytest.raises(DataValidationError) as exc_info:
            calculate_reactive_apparent(-230.0, 5.0, 0.9)

        assert "Voltage cannot be negative" in str(exc_info.value)

    def test_invalid_current(self) -> None:
        """Test with invalid negative current."""
        with pytest.raises(DataValidationError) as exc_info:
            calculate_reactive_apparent(230.0, -5.0, 0.9)

        assert "Current cannot be negative" in str(exc_info.value)

    def test_invalid_power_factor(self) -> None:
        """Test with invalid power factor outside range."""
        with pytest.raises(DataValidationError) as exc_info:
            calculate_reactive_apparent(230.0, 5.0, 1.5)

        assert "Power factor out of valid range" in str(exc_info.value)

        with pytest.raises(DataValidationError) as exc_info:
            calculate_reactive_apparent(230.0, 5.0, -1.5)

        assert "Power factor out of valid range" in str(exc_info.value)


class TestPowerCalculator:
    """Test PowerCalculator utility class."""

    def test_calculate_power_components(self) -> None:
        """Test calculation of all power components."""
        voltage = 230.0
        current = 5.0
        power_factor = 0.9

        components = PowerCalculator.calculate_power_components(voltage, current, power_factor)

        expected_apparent = voltage * current
        expected_active = expected_apparent * power_factor
        expected_reactive = expected_apparent * math.sqrt(1.0 - power_factor * power_factor)

        assert isinstance(components, PowerComponents)
        assert components.apparent == pytest.approx(expected_apparent)
        assert components.active == pytest.approx(expected_active)
        assert components.reactive == pytest.approx(expected_reactive, rel=1e-6)

    def test_split_power_positive(self) -> None:
        """Test power splitting for positive power (import)."""
        power = 1000.0

        split = PowerCalculator.split_power(power)

        assert isinstance(split, PowerSplit)
        assert split.import_value == 10000  # 1000.0 * 10
        assert split.export_value == 0

    def test_split_power_negative(self) -> None:
        """Test power splitting for negative power (export)."""
        power = -1000.0

        split = PowerCalculator.split_power(power)

        assert isinstance(split, PowerSplit)
        assert split.import_value == 0
        assert split.export_value == 10000  # abs(-1000.0) * 10

    def test_split_power_tuple_backward_compatibility(self) -> None:
        """Test backward compatibility tuple method."""
        power = 500.0

        import_val, export_val = PowerCalculator.split_power_tuple(power)

        assert import_val == 5000
        assert export_val == 0

    def test_validate_power_triangle_valid(self) -> None:
        """Test validation of valid power triangle."""
        active = 900.0  # P
        reactive = 435.89  # Q (calculated for PF = 0.9)
        apparent = 1000.0  # S

        is_valid = PowerCalculator.validate_power_triangle(active, reactive, apparent)
        assert is_valid

    def test_validate_power_triangle_invalid(self) -> None:
        """Test validation of invalid power triangle."""
        active = 900.0
        reactive = 1000.0  # Too high for given active/apparent
        apparent = 1000.0

        is_valid = PowerCalculator.validate_power_triangle(active, reactive, apparent)
        assert not is_valid

    def test_validate_power_triangle_zero_apparent(self) -> None:
        """Test validation when apparent power is zero."""
        is_valid = PowerCalculator.validate_power_triangle(0.0, 0.0, 0.0)
        assert is_valid

        is_valid = PowerCalculator.validate_power_triangle(100.0, 0.0, 0.0)
        assert not is_valid

    def test_calculate_power_factor_from_powers(self) -> None:
        """Test power factor calculation from powers."""
        active = 900.0
        apparent = 1000.0
        expected_pf = 0.9

        pf = PowerCalculator.calculate_power_factor_from_powers(active, apparent)
        assert pf == pytest.approx(expected_pf)

    def test_calculate_power_factor_zero_apparent(self) -> None:
        """Test power factor calculation with zero apparent power."""
        pf = PowerCalculator.calculate_power_factor_from_powers(100.0, 0.0)
        assert pf == 1.0

    def test_calculate_current_from_power(self) -> None:
        """Test current calculation from power and other parameters."""
        active_power = 2070.0  # 230V * 9A * 1.0 PF
        voltage = 230.0
        power_factor = 1.0
        expected_current = 9.0

        current = PowerCalculator.calculate_current_from_power(active_power, voltage, power_factor)
        assert current == pytest.approx(expected_current)

    def test_calculate_current_edge_cases(self) -> None:
        """Test current calculation edge cases."""
        # Zero voltage
        current = PowerCalculator.calculate_current_from_power(1000.0, 0.0, 1.0)
        assert current == 0.0

        # Very low power factor
        current = PowerCalculator.calculate_current_from_power(1000.0, 230.0, 0.0001)
        assert current == 0.0

    def test_watts_to_w01_units_conversion(self) -> None:
        """Test watts to 0.1W units conversion."""
        watts = 123.456
        w01_units = PowerCalculator.watts_to_w01_units(watts)
        assert w01_units == 1235  # rounded(123.456 * 10)

    def test_w01_units_to_watts_conversion(self) -> None:
        """Test 0.1W units to watts conversion."""
        w01_units = 1234
        watts = PowerCalculator.w01_units_to_watts(w01_units)
        assert watts == pytest.approx(123.4)

    def test_calculate_balanced_three_phase_power(self) -> None:
        """Test balanced three-phase power calculation."""
        line_voltage = 400.0  # Line-to-line
        line_current = 5.0
        power_factor = 0.9

        components = PowerCalculator.calculate_balanced_three_phase_power(
            line_voltage,
            line_current,
            power_factor,
        )

        expected_active = math.sqrt(3) * line_voltage * line_current * power_factor
        expected_apparent = math.sqrt(3) * line_voltage * line_current
        expected_reactive = expected_apparent * math.sqrt(1.0 - power_factor * power_factor)

        assert components.active == pytest.approx(expected_active)
        assert components.apparent == pytest.approx(expected_apparent)
        assert components.reactive == pytest.approx(expected_reactive, rel=1e-6)

    def test_calculate_balanced_three_phase_power_negative_pf(self) -> None:
        """Test balanced three-phase power with negative power factor."""
        line_voltage = 400.0
        line_current = 5.0
        power_factor = -0.8

        components = PowerCalculator.calculate_balanced_three_phase_power(
            line_voltage,
            line_current,
            power_factor,
        )

        assert components.reactive < 0  # Negative for capacitive
        assert components.active < 0  # Negative PF means negative active power


class TestPowerComponents:
    """Test PowerComponents dataclass."""

    def test_power_components_creation(self) -> None:
        """Test PowerComponents dataclass creation."""
        components = PowerComponents(active=1000.0, reactive=500.0, apparent=1118.0)

        assert components.active == 1000.0
        assert components.reactive == 500.0
        assert components.apparent == 1118.0

    def test_power_components_immutable(self) -> None:
        """Test that PowerComponents is immutable (frozen)."""
        components = PowerComponents(active=1000.0, reactive=500.0, apparent=1118.0)

        with pytest.raises(AttributeError):
            components.active = 2000.0  # type: ignore[misc]  # Should fail due to frozen=True


class TestPowerSplit:
    """Test PowerSplit dataclass."""

    def test_power_split_creation(self) -> None:
        """Test PowerSplit dataclass creation."""
        split = PowerSplit(import_value=5000, export_value=0)

        assert split.import_value == 5000
        assert split.export_value == 0

    def test_power_split_immutable(self) -> None:
        """Test that PowerSplit is immutable (frozen)."""
        split = PowerSplit(import_value=5000, export_value=0)

        with pytest.raises(AttributeError):
            split.import_value = 10000  # type: ignore[misc]  # Should fail due to frozen=True


class TestPowerCalculatorAdditionalMethods:
    """Test uncovered PowerCalculator methods."""

    def test_calculate_phase_power_components_list(self) -> None:
        """Test calculation for list of phases."""

        # Mock phase data objects
        phase1 = Mock(voltage=230.0, current=5.0, pf=0.9)
        phase2 = Mock(voltage=232.0, current=4.8, pf=0.85)
        phase3 = Mock(voltage=228.0, current=5.2, pf=0.95)
        phases = [phase1, phase2, phase3]

        result = PowerCalculator.calculate_phase_power_components_list(phases)

        assert len(result) == 3
        assert all(isinstance(comp, PowerComponents) for comp in result)

        # Check first phase calculation
        expected_apparent = 230.0 * 5.0
        assert result[0].apparent == pytest.approx(expected_apparent)
        assert result[0].active == pytest.approx(expected_apparent * 0.9)

    def test_calculate_phase_powers(self) -> None:
        """Test calculation for single phase data."""

        phase_data = Mock(voltage=230.0, current=5.0, pf=0.9)
        result = PowerCalculator.calculate_phase_powers(phase_data)

        assert "reactive" in result
        assert "apparent" in result

        reactive_power, _ = result["reactive"]
        apparent_power, _ = result["apparent"]

        expected_apparent = 230.0 * 5.0
        expected_reactive = expected_apparent * math.sqrt(1.0 - 0.9 * 0.9)

        assert apparent_power == pytest.approx(expected_apparent)
        assert reactive_power == pytest.approx(expected_reactive, rel=1e-6)

    def test_get_phase_powers(self) -> None:
        """Test get_phase_powers for all phases."""

        phase_a = Mock(voltage=230.0, current=5.0, pf=0.9)
        phase_b = Mock(voltage=232.0, current=4.8, pf=0.85)
        phase_c = Mock(voltage=228.0, current=5.2, pf=0.95)
        data = Mock(a=phase_a, b=phase_b, c=phase_c)

        result = PowerCalculator.get_phase_powers(data)

        assert "a" in result
        assert "b" in result
        assert "c" in result

        # Check phase A
        _reactive_a, apparent_a = result["a"]
        expected_apparent_a = 230.0 * 5.0
        assert apparent_a == pytest.approx(expected_apparent_a)

    def test_calculate_total_powers(self) -> None:
        """Test calculate_total_powers across all phases."""

        phase_a = Mock(voltage=230.0, current=5.0, pf=0.9)
        phase_b = Mock(voltage=232.0, current=4.8, pf=0.85)
        phase_c = Mock(voltage=228.0, current=5.2, pf=0.95)
        data = Mock(a=phase_a, b=phase_b, c=phase_c, total_power=3450.0)

        result = PowerCalculator.calculate_total_powers(data)

        assert isinstance(result, PowerComponents)
        assert result.active == 3450.0
        assert result.reactive > 0  # Should have some reactive power
        assert result.apparent > 0  # Should have some apparent power
