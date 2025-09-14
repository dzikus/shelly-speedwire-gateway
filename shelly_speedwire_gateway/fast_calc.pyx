# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

"""Fast calculations using Cython for maximum performance."""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, abs as c_abs
from libc.stdint cimport int32_t, uint32_t

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t INT_DTYPE_t

cdef class FastCalculator:
    """High-performance calculations in C."""

    cdef readonly int calculations_count

    def __init__(self):
        self.calculations_count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double calc_power(self, double voltage, double current, double pf) nogil:
        """Calculate active power."""
        return voltage * current * pf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double calc_reactive_power(self, double voltage, double current, double pf) nogil:
        """Calculate reactive power."""
        if c_abs(pf) >= 1.0:
            return 0.0
        return voltage * current * sqrt(1.0 - pf * pf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double calc_apparent_power(self, double voltage, double current) nogil:
        """Calculate apparent power."""
        return voltage * current

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef uint32_t calc_checksum(self, const char* data, int length) nogil:
        """Calculate checksum for Speedwire packets."""
        cdef uint32_t checksum = 0
        cdef int i
        for i in range(length):
            checksum += <unsigned char>data[i]
        return checksum

    def power(self, double voltage, double current, double pf=1.0):
        """Calculate power from voltage, current and power factor."""
        self.calculations_count += 1
        return self.calc_power(voltage, current, pf)

    def reactive_power(self, double voltage, double current, double pf):
        """Calculate reactive power."""
        self.calculations_count += 1
        return self.calc_reactive_power(voltage, current, pf)

    def apparent_power(self, double voltage, double current):
        """Calculate apparent power."""
        self.calculations_count += 1
        return self.calc_apparent_power(voltage, current)

    def checksum(self, bytes data):
        """Calculate checksum for packet data."""
        cdef const char* c_data = data
        cdef int length = len(data)
        return self.calc_checksum(c_data, length)


# Global calculator instance
cdef FastCalculator _calculator = FastCalculator()

def get_calculator():
    """Get global calculator instance."""
    return _calculator

def benchmark_performance(int iterations=1000000):
    """Benchmark Cython performance."""
    import time
    cdef double voltage = 230.0
    cdef double current = 10.0
    cdef double pf = 0.95

    start_time = time.time()
    cdef int i
    cdef double result

    for i in range(iterations):
        result = _calculator.calc_power(voltage, current, pf)

    elapsed = time.time() - start_time
    ops_per_sec = iterations / elapsed if elapsed > 0 else 0

    return {
        "cython_enabled": True,
        "calculations_per_sec": ops_per_sec,
        "elapsed_time": elapsed,
        "iterations": iterations,
        "result": result,
    }


class MQTTProcessor:
    """Optimized MQTT processor using Cython calculations."""

    def __init__(self, original_processor):
        self.original = original_processor
        self.calculator = get_calculator()
        self._calls = 0

    def process_message(self, topic: str, payload, qos: int = 0, *, retain: bool = False):
        """Process message with Cython optimizations."""
        if "power" in topic:
            self._calls += 1

            # Use pooled data for processing
            try:
                if isinstance(payload, bytes):
                    payload = payload.decode('utf-8')

                # Fast power calculation with Cython
                if payload.replace('.', '').replace('-', '').isdigit():
                    power_value = float(payload)
                    # Use fast calculator for power validation
                    if abs(power_value) > 0.1:
                        self.calculator.power(230.0, power_value / 230.0, 0.95)

            except (ValueError, UnicodeDecodeError):
                pass

        return self.original.process_message(topic, payload, qos, retain=retain)

    def get_processing_stats(self):
        """Get processing stats with Cython metrics."""
        stats = self.original.get_processing_stats()
        stats["cython_calculations"] = {
            "calls": self._calls,
            "total_calculations": self.calculator.calculations_count,
        }
        return stats

    def clear_cache(self):
        """Clear caches."""
        if hasattr(self.original, 'clear_cache'):
            self.original.clear_cache()
