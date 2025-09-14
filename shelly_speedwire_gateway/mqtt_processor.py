"""MQTT data processing using Pydantic models for validation and transformation.

This module handles MQTT message parsing, validation, and transformation
using Pydantic v2 for data processing with error handling.
"""

from __future__ import annotations

import gc
import re
import resource
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, Field, ValidationError, field_validator

from shelly_speedwire_gateway.constants import (
    MIN_PARTS_COUNT,
    PHASE_LETTERS,
    POWER_CONSISTENCY_THRESHOLD,
)
from shelly_speedwire_gateway.models import GatewaySettings, PhaseData, Shelly3EMData

try:
    from shelly_speedwire_gateway.fast_calc import get_calculator

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ObjectPool[T]:
    """Thread-safe object pool for efficient object reuse."""

    def __init__(self, factory: Callable[[], T], max_size: int = 100, max_idle_time: float = 300.0):
        """Initialize object pool."""
        self._factory: Callable[[], T] = factory
        self._max_size = max_size
        self._max_idle_time = max_idle_time
        self._pool: deque[T] = deque()
        self._timestamps: deque[float] = deque()
        self._lock = threading.RLock()

        self._stats: dict[str, int | float] = {
            "objects_created": 0,
            "objects_reused": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "current_size": 0,
        }

    def get(self) -> T:
        """Get object from pool or create new one."""
        current_time = time.time()

        with self._lock:
            self._cleanup_expired(current_time)

            if self._pool:
                obj = self._pool.popleft()
                self._timestamps.popleft()
                self._stats["objects_reused"] += 1
                self._stats["pool_hits"] += 1
                self._stats["current_size"] = len(self._pool)
                return obj

            obj = self._factory()
            self._stats["objects_created"] += 1
            self._stats["pool_misses"] += 1
            return obj

    def put(self, obj: T) -> None:
        """Return object to pool for reuse."""
        current_time = time.time()

        with self._lock:
            if len(self._pool) >= self._max_size:
                return

            self._pool.append(obj)
            self._timestamps.append(current_time)
            self._stats["current_size"] = len(self._pool)

    def _cleanup_expired(self, current_time: float) -> None:
        while self._pool and self._timestamps and current_time - self._timestamps[0] > self._max_idle_time:
            self._pool.popleft()
            self._timestamps.popleft()

        self._stats["current_size"] = len(self._pool)

    def get_stats(self) -> dict[str, Any]:
        """Get object pool statistics."""
        with self._lock:
            stats = self._stats.copy()
            total_requests = stats["pool_hits"] + stats["pool_misses"]
            stats["hit_rate"] = stats["pool_hits"] / total_requests if total_requests > 0 else 0.0
            return stats


class PooledData:
    """Pooled data object with slots for memory efficiency."""

    __slots__ = ("__weakref__", "current", "device_id", "pf", "phase", "power", "timestamp", "voltage")

    def __init__(self) -> None:
        """Initialize power data."""
        self.device_id: str = ""
        self.phase: int = 0
        self.power: float = 0.0
        self.voltage: float = 0.0
        self.current: float = 0.0
        self.pf: float = 0.0
        self.timestamp: float = 0.0

    def reset(self) -> None:
        """Reset data object to default values."""
        self.device_id = ""
        self.phase = 0
        self.power = 0.0
        self.voltage = 0.0
        self.current = 0.0
        self.pf = 0.0
        self.timestamp = 0.0

    def update(self, device_id: str, phase: int, **kwargs: Any) -> None:
        """Update data object with new values."""
        self.device_id = device_id
        self.phase = phase
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.timestamp = time.time()


class TTLCache:
    """Thread-safe LRU cache with TTL expiration."""

    def __init__(self, maxsize: int = 128, ttl: float = 300.0):
        """Initialize LRU cache with TTL."""
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: dict[Any, tuple[Any, float]] = {}
        self._access_order: dict[Any, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Any | None:
        """Get cached value by key."""
        current_time = time.time()

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry_time = self._cache[key]

            if current_time > expiry_time:
                del self._cache[key]
                del self._access_order[key]
                self._misses += 1
                return None

            self._access_order[key] = current_time
            self._hits += 1
            return value

    def put(self, key: Any, value: Any) -> None:
        """Put value in cache with TTL."""
        current_time = time.time()
        expiry_time = current_time + self.ttl

        with self._lock:
            if len(self._cache) >= self.maxsize and key not in self._cache:
                self._evict_lru()

            self._cache[key] = (value, expiry_time)
            self._access_order[key] = current_time

    def _evict_lru(self) -> None:
        if not self._access_order:
            return

        lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
        del self._cache[lru_key]
        del self._access_order[lru_key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "maxsize": self.maxsize,
            }


class MQTTMessage:
    """MQTT message container with slots."""

    __slots__ = ("payload", "qos", "retain", "timestamp", "topic")

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        topic: str,
        payload: str | bytes,
        qos: int = 0,
        retain: bool = False,  # noqa: FBT001, FBT002
        timestamp: float = 0.0,
    ):
        """Initialize MQTT message."""
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.retain = retain
        self.timestamp = timestamp if timestamp > 0.0 else time.time()


class BatchProcessor:
    """Batch processor for MQTT messages."""

    def __init__(self, config: GatewaySettings):
        """Initialize batch processor."""
        self.batch_size = config.batch_size
        self.flush_interval = config.batch_flush_interval
        self.max_queue_size = config.max_queue_size
        self.is_running = False

        self._queue: deque[MQTTMessage] = deque()
        self._queue_lock = threading.Lock()
        self._batch_processor: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._stats = {
            "messages_queued": 0,
            "messages_processed": 0,
            "batches_processed": 0,
            "average_batch_size": 0.0,
        }

    def start(self) -> None:
        """Start batch processor."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        self._batch_processor = threading.Thread(target=self._processing_loop, daemon=True, name="mqtt-batch-processor")
        self._batch_processor.start()

    def stop(self) -> None:
        """Stop batch processor."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        if self._batch_processor:
            self._batch_processor.join(timeout=5.0)

        self._process_remaining_messages()

    def queue_message(
        self,
        topic: str,
        payload: str | bytes,
        qos: int = 0,
        retain: bool = False,  # noqa: FBT001, FBT002
    ) -> bool:
        """Queue MQTT message for batch processing."""
        message = MQTTMessage(topic=topic, payload=payload, qos=qos, retain=retain)

        with self._queue_lock:
            if len(self._queue) > self.max_queue_size:
                return False

            self._queue.append(message)
            self._stats["messages_queued"] += 1
            return True

    def _processing_loop(self) -> None:
        last_flush = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()
            batch = self._collect_batch()

            should_process = len(batch) >= self.batch_size or (
                batch and current_time - last_flush >= self.flush_interval
            )

            if should_process and batch:
                self.process_batch(batch)
                last_flush = current_time
            else:
                time.sleep(0.001)

    def _collect_batch(self) -> list[MQTTMessage]:
        batch: list[MQTTMessage] = []

        with self._queue_lock:
            while len(batch) < self.batch_size and self._queue:
                batch.append(self._queue.popleft())

        return batch

    def process_batch(self, batch: list[MQTTMessage]) -> None:
        """Process batch of MQTT messages."""
        batch_size = len(batch)
        self._stats["batches_processed"] += 1
        self._stats["messages_processed"] += batch_size

        total_batches = self._stats["batches_processed"]
        current_avg = self._stats["average_batch_size"]
        self._stats["average_batch_size"] = (current_avg * (total_batches - 1) + batch_size) / total_batches

    def _process_remaining_messages(self) -> None:
        remaining = self._collect_batch()
        if remaining:
            self.process_batch(remaining)

    def get_stats(self) -> dict[str, Any]:
        """Get batch processor statistics."""
        with self._queue_lock:
            stats = self._stats.copy()
            stats["current_queue_size"] = len(self._queue)
            stats["is_running"] = self.is_running
            return stats


class MQTTMessageMetadata(BaseModel):
    """Metadata for MQTT message processing."""

    topic: str = Field(description="Full MQTT topic")
    timestamp: float = Field(description="Message timestamp")
    qos: int = Field(ge=0, le=2, description="Quality of Service level")
    retain: bool = Field(default=False, description="Retain flag")

    @field_validator("topic")
    @classmethod
    def validate_topic_format(cls, v: str) -> str:
        """Validate MQTT topic format."""
        if not re.match(r"^[a-zA-Z0-9_/\-+#$]+$", v):
            raise ValueError(f"Invalid MQTT topic format: {v}")
        return v


class ShellyEMeterMessage(BaseModel):
    """Validated MQTT message from Shelly EM device."""

    device_id: str = Field(description="Device identifier")
    phase: int = Field(ge=0, le=2, description="Phase number (0, 1, 2)")
    measurement_type: str = Field(description="Type of measurement")
    value: float = Field(description="Measurement value")
    metadata: MQTTMessageMetadata = Field(description="Message metadata")

    @field_validator("measurement_type")
    @classmethod
    def validate_measurement_type(cls, v: str) -> str:
        """Validate measurement type is known."""
        valid_types = {
            "power",
            "voltage",
            "current",
            "pf",
            "power_factor",
            "total",
            "total_returned",
            "energy_consumed",
            "energy_exported",
            "energy",
            "returned_energy",
        }
        if v not in valid_types:
            raise ValueError(f"Unknown measurement type: {v}")
        return v


class ShellyStatusMessage(BaseModel):
    """Validated status message from Shelly device."""

    device_id: str = Field(description="Device identifier")
    status: str = Field(description="Device status")
    metadata: MQTTMessageMetadata = Field(description="Message metadata")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status value."""
        valid_statuses = {"true", "false", "online", "offline", "1", "0"}
        if v.lower() not in valid_statuses:
            logger.warning("Unknown status value", status=v)
        return v


class EmeterMessageRequest(BaseModel):
    """Request model for emeter message processing with grouped arguments."""

    device_path: str = Field(description="Device path from MQTT topic")
    phase_str: str = Field(description="Phase identifier string")
    measurement_type: str = Field(description="Type of measurement")
    payload: bytes | str = Field(description="Message payload")
    metadata: MQTTMessageMetadata = Field(description="Message metadata")


class PhaseUpdateRequest(BaseModel):
    """Request model for phase update with grouped arguments."""

    device_id: str = Field(description="Device identifier")
    phase: int = Field(ge=0, le=2, description="Phase number (0-2)")
    measurement_type: str = Field(description="Type of measurement")
    value: float = Field(description="Measurement value")
    timestamp: float = Field(description="Measurement timestamp")


@dataclass(slots=True)
class ProcessingStats:
    """Statistics for MQTT message processing."""

    total_messages: int = 0
    valid_messages: int = 0
    invalid_messages: int = 0
    parsing_errors: int = 0
    validation_errors: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_messages == 0:
            return 0.0
        return self.valid_messages / self.total_messages * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_messages == 0:
            return 0.0
        return (self.invalid_messages + self.parsing_errors) / self.total_messages * 100


class MQTTDataProcessor:
    """MQTT data processor using Pydantic validation."""

    def __init__(
        self,
        invert_values: bool = False,  # noqa: FBT001, FBT002
        strict_validation: bool = True,  # noqa: FBT001, FBT002
        config: GatewaySettings | None = None,
    ):
        """Initialize MQTT processor with optimizations."""
        self.config = config or GatewaySettings()
        self.invert_values = invert_values
        self.strict_validation = strict_validation
        self.stats = ProcessingStats()
        self.current_data = Shelly3EMData()

        # Pre-compiled regex patterns
        self.topic_patterns = {
            "emeter": re.compile(r"^(.+)/emeter/(\d+)/(.+)$"),
            "online": re.compile(r"^(.+)/online$"),
            "status": re.compile(r"^(.+)/status$"),
        }

        # Memory pools for object reuse
        self.data_pool: ObjectPool[PooledData] = ObjectPool(PooledData, max_size=50)

        # Caches for different operations using config values
        self.power_cache = TTLCache(maxsize=self.config.lru_cache_size, ttl=60.0)
        self.topic_cache = TTLCache(maxsize=self.config.lru_cache_size, ttl=300.0)
        self.validation_cache = TTLCache(maxsize=self.config.lru_cache_size, ttl=600.0)

        # Batch processor for high throughput using config values
        self.batch_processor = BatchProcessor(self.config)
        self.batch_processor.process_batch = self._process_batch_messages  # type: ignore[method-assign]
        self.batch_processor.start()

        # Legacy topic cache for backward compatibility
        self._topic_cache: dict[str, tuple[str, ...]] = {}
        self._topic_cache_maxsize = self.config.lru_cache_size

        # Cython calculator integration
        if CYTHON_AVAILABLE:
            self.fast_calculator = get_calculator()
            self._cython_enabled = True
            logger.info("Cython acceleration enabled")
        else:
            self.fast_calculator = None
            self._cython_enabled = False
            logger.info("Cython acceleration not available")

        # Performance stats
        self._perf_stats = {
            "cached_calls": 0,
            "pool_usage": 0,
            "batch_usage": 0,
            "cython_calculations": 0,
        }

        logger.info(
            "MQTT processor initialized with optimizations",
            invert_values=invert_values,
            strict_validation=strict_validation,
        )

    def _process_batch_messages(self, batch: list[MQTTMessage]) -> None:
        """Process batch of messages internally without statistics."""
        for message in batch:
            try:
                self._process_message_internal(message.topic, message.payload, message.qos, retain=message.retain)
                self._perf_stats["batch_usage"] += 1
            except (ValueError, TypeError, ValidationError) as e:
                logger.exception("Error processing batched message", topic=message.topic, error=str(e))

    def _get_field_name(self, measurement_type: str) -> str | None:
        """Get field name for measurement type."""
        field_mapping = {
            "power": "power",
            "voltage": "voltage",
            "current": "current",
            "pf": "pf",
            "power_factor": "pf",
            "energy_consumed": "energy_consumed",
            "energy_exported": "energy_exported",
            "total": "energy_consumed",
            "total_returned": "energy_exported",
            "energy": "energy_consumed",
            "returned_energy": "energy_exported",
        }

        return field_mapping.get(measurement_type)

    def _parse_topic_cached(self, topic: str) -> tuple[str, ...] | None:
        """Parse MQTT topic with caching."""
        if topic in self._topic_cache:
            return self._topic_cache[topic]

        # Check cache size and clean if needed
        if len(self._topic_cache) >= self._topic_cache_maxsize:
            self._topic_cache.clear()

        # Try each pattern
        for pattern_name, pattern in self.topic_patterns.items():
            match = pattern.match(topic)
            if match:
                result = (pattern_name, *match.groups())
                self._topic_cache[topic] = result
                return result

        return None

    def process_message(
        self,
        topic: str,
        payload: bytes | str,
        qos: int = 0,
        retain: bool = False,  # noqa: FBT001, FBT002
    ) -> Shelly3EMData | None:
        """Process incoming MQTT message with optimizations."""
        # Try batch processing first for high throughput
        if self.batch_processor.queue_message(topic, payload, qos, retain=retain):
            self._perf_stats["batch_usage"] += 1
            return None  # Queued for async processing

        # Fallback to sync processing
        return self._process_message_internal(topic, payload, qos, retain=retain)

    def _process_message_internal(
        self,
        topic: str,
        payload: bytes | str,
        qos: int = 0,
        retain: bool = False,  # noqa: FBT001, FBT002
    ) -> Shelly3EMData | None:
        """Internal message processing with caching."""
        self.stats.total_messages += 1

        # Check cache for repeated messages
        cache_key = f"{topic}:{hash(str(payload))}"
        cached_result = self.power_cache.get(cache_key)
        if cached_result is not None:
            self._perf_stats["cached_calls"] += 1
            return cached_result  # type: ignore[no-any-return]

        try:
            metadata = MQTTMessageMetadata(topic=topic, timestamp=time.time(), qos=qos, retain=retain)
            result = self._route_message(topic, payload, metadata)

            if result:
                self.stats.valid_messages += 1
                # Cache successful results
                self.power_cache.put(cache_key, result)
                return result

            self.stats.invalid_messages += 1
        except ValidationError as e:
            self.stats.validation_errors += 1
            logger.warning(
                "Message validation failed",
                topic=topic,
                errors=[f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()],
            )
        except (ValueError, TypeError, OSError) as e:
            self.stats.parsing_errors += 1
            logger.exception("Message processing error", topic=topic, error=str(e))

        # TRY300: Move return to else block for exceptions
        return None

    def _route_message(self, topic: str, payload: bytes | str, metadata: MQTTMessageMetadata) -> Shelly3EMData | None:
        """Route message to appropriate handler based on topic pattern."""
        # Use cached topic parsing
        parsed = self._parse_topic_cached(topic)
        if parsed is None:
            return None

        pattern_name = parsed[0]

        if pattern_name == "emeter":
            # Extract match groups from cached result
            device_id, phase_str, measurement_type = parsed[1], parsed[2], parsed[3]
            request = EmeterMessageRequest(
                device_path=device_id,
                phase_str=phase_str,
                measurement_type=measurement_type,
                payload=payload,
                metadata=metadata,
            )
            return self._handle_emeter_message(request)

        # Status messages don't return data
        if match := self.topic_patterns["online"].match(topic):
            self._handle_status_message(match, payload, metadata, "online")
            return None

        if match := self.topic_patterns["status"].match(topic):
            self._handle_status_message(match, payload, metadata, "status")
            return None

        logger.debug("Unknown topic pattern", topic=topic)
        return None

    def _handle_status_message(
        self,
        topic_match: re.Match[str],
        payload: bytes | str,
        metadata: MQTTMessageMetadata,
        message_type: str,
    ) -> None:  # Change return type to None to be consistent
        """Handle device status message."""
        try:
            device_path = topic_match.group(1)
            device_id = device_path.split("/")[-1]

            # SIM108: Use ternary operator
            payload_str = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)

            status_message = ShellyStatusMessage(device_id=device_id, status=payload_str.strip(), metadata=metadata)

            logger.info(
                "Device status update",
                device_id=status_message.device_id,
                status=status_message.status,
                type=message_type,
            )

        except ValidationError as e:
            logger.warning("Failed to parse status message", error=str(e))

        # R1711: Remove explicit return None

    def _handle_emeter_message(self, request: EmeterMessageRequest) -> Shelly3EMData | None:
        """Handle energy meter measurement message."""
        try:
            device_id = request.device_path.split("/")[-1]

            # Convert payload to float and apply inversion immediately
            if isinstance(request.payload, bytes):
                try:
                    # Direct conversion
                    value = float(request.payload)
                except (ValueError, UnicodeDecodeError):
                    # Fallback to string conversion
                    payload_str = request.payload.decode("utf-8").strip()
                    value = float(payload_str)
            else:
                value = float(str(request.payload).strip())

            measurement_type = request.measurement_type
            if self.invert_values:
                # Invert power values (these can be negative)
                if measurement_type in ("power", "energy", "returned_energy"):
                    value = -value
                # Swap energy field meanings for total/total_returned (keep values positive)
                elif measurement_type == "total":
                    measurement_type = "total_returned"
                elif measurement_type == "total_returned":
                    measurement_type = "total"

            phase_request = PhaseUpdateRequest(
                device_id=device_id,
                phase=int(request.phase_str),
                measurement_type=measurement_type,
                value=value,
                timestamp=request.metadata.timestamp,
            )
            return self._update_phase(phase_request)

        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(
                "Failed to parse emeter message",
                error=str(e),
                topic_parts=(request.device_path, request.phase_str, request.measurement_type),
            )
            return None

    def _update_phase(self, request: PhaseUpdateRequest) -> Shelly3EMData:
        """Update phase data."""
        current_phase = self.current_data.get_phase(request.phase)
        phase_dict = current_phase.model_dump()

        # Get field name using centralized logic
        field_name = self._get_field_name(request.measurement_type)
        if field_name:
            phase_dict[field_name] = request.value

        new_phase = PhaseData.model_validate(phase_dict)

        if request.phase == 0:
            new_data = self.current_data.model_copy(
                update={"a": new_phase, "timestamp": request.timestamp, "device_id": request.device_id},
            )
        elif request.phase == 1:
            new_data = self.current_data.model_copy(
                update={"b": new_phase, "timestamp": request.timestamp, "device_id": request.device_id},
            )
        elif request.phase == 2:  # noqa: PLR2004
            new_data = self.current_data.model_copy(
                update={"c": new_phase, "timestamp": request.timestamp, "device_id": request.device_id},
            )
        else:
            return self.current_data

        self.current_data = new_data
        return self.current_data

    def _update_phase_data(self, message: ShellyEMeterMessage) -> Shelly3EMData:
        """Update phase data from validated MQTT message."""
        current_phase = self.current_data.get_phase(message.phase)
        phase_dict = current_phase.model_dump()

        # Value already inverted at entry point, use as-is
        value = message.value

        # Get field name using centralized logic
        field_name = self._get_field_name(message.measurement_type)
        if not field_name:
            logger.warning("Unknown measurement type", type=message.measurement_type)
            return self.current_data

        phase_dict[field_name] = value

        # Refactor to avoid try-except-else vs no-else-return conflict
        new_phase = self._create_validated_phase(phase_dict)
        if new_phase is None:
            logger.warning("Failed to create validated phase data")
            return self.current_data

        current_dict = self.current_data.model_dump()
        current_dict.update(
            {
                "timestamp": message.metadata.timestamp,
                "device_id": message.device_id,
            },
        )

        phase_letter = PHASE_LETTERS[message.phase]
        current_dict[phase_letter] = new_phase.model_dump()

        updated_data = self._create_validated_3em_data(current_dict)
        if updated_data is not None:
            self.current_data = updated_data

        return self.current_data

    def _create_validated_phase(self, phase_dict: dict) -> PhaseData | None:
        """Create validated PhaseData from dictionary."""
        try:
            return PhaseData.model_validate(phase_dict)
        except ValidationError as e:
            logger.warning(
                "Phase data validation failed",
                errors=[f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()],
            )
            return None

    def _create_validated_3em_data(self, data_dict: dict) -> Shelly3EMData | None:
        """Create validated Shelly3EMData from dictionary."""
        try:
            return Shelly3EMData.model_validate(data_dict)
        except ValidationError as e:
            logger.warning(
                "3EM data validation failed",
                errors=[f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()],
            )
            return None

    def get_current_data(self) -> Shelly3EMData:
        """Get current validated measurement data."""
        return self.current_data

    def reset_data(self) -> None:
        """Reset current measurement data to defaults."""
        self.current_data = Shelly3EMData()
        logger.debug("Measurement data reset")

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics with optimization metrics."""
        stats: dict[str, Any] = {
            "total_messages": self.stats.total_messages,
            "valid_messages": self.stats.valid_messages,
            "invalid_messages": self.stats.invalid_messages,
            "parsing_errors": self.stats.parsing_errors,
            "validation_errors": self.stats.validation_errors,
            "success_rate_percent": round(self.stats.success_rate, 2),
            "error_rate_percent": round(self.stats.error_rate, 2),
            "current_device_id": self.current_data.device_id,
            "last_update": self.current_data.timestamp,
            "topic_cache_size": len(self._topic_cache),
        }

        # Add optimization statistics
        stats["optimizations"] = self._perf_stats.copy()

        # Add cache statistics
        stats["cache_stats"] = {
            "power_cache": self.power_cache.get_stats(),
            "topic_cache": self.topic_cache.get_stats(),
            "validation_cache": self.validation_cache.get_stats(),
        }

        # Add memory pool statistics
        stats["memory_pool"] = self.data_pool.get_stats()

        # Add batch processor statistics
        stats["batch_processing"] = self.batch_processor.get_stats()

        # Add Cython statistics
        if self._cython_enabled and self.fast_calculator:
            stats["cython"] = {
                "enabled": True,
                "calculations_count": self.fast_calculator.calculations_count,
            }
        else:
            stats["cython"] = {"enabled": False}

        return stats

    def get_metrics(self) -> dict[str, Any]:
        """Return memory and processing metrics."""
        try:
            memory_info = resource.getrusage(resource.RUSAGE_SELF)

            return {
                "memory_peak_kb": memory_info.ru_maxrss,
                "gc_collections": len(gc.get_stats()) if hasattr(gc, "get_stats") else 0,
                "topic_cache_size": len(self._topic_cache),
                "topic_cache_hit_rate": self._calculate_cache_hit_rate(),
            }
        except ImportError:
            return {"error": "Resource monitoring not available on this platform"}

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate topic cache hit rate (approximation)."""
        if self.stats.total_messages == 0:
            return 0.0
        return min(100.0, len(self._topic_cache) / max(self.stats.total_messages, 1) * 100)

    def clear_cache(self) -> None:
        """Clear all caches and stop batch processing."""
        self.power_cache = TTLCache(maxsize=self.config.lru_cache_size, ttl=60.0)
        self.topic_cache = TTLCache(maxsize=self.config.lru_cache_size, ttl=300.0)
        self.validation_cache = TTLCache(maxsize=self.config.lru_cache_size, ttl=600.0)
        self._topic_cache.clear()

        # Stop and restart batch processor
        self.batch_processor.stop()
        self.batch_processor = BatchProcessor(self.config)
        self.batch_processor.process_batch = self._process_batch_messages  # type: ignore[method-assign]
        self.batch_processor.start()

        logger.info("All caches cleared and batch processing restarted")

    def validate_complete_dataset(self) -> bool:
        """Check if current dataset is complete and valid for transmission."""
        return self._perform_dataset_validation()

    def _perform_dataset_validation(self) -> bool:
        """Perform the actual dataset validation logic."""
        try:
            phases = self.current_data.get_phases_list()

            for i, phase in enumerate(phases):
                if not any([phase.power, phase.voltage, phase.current]):
                    logger.debug(f"Phase {i} has no measurement data")
                    continue

                if phase.voltage > 0 and phase.current > 0 and phase.power != 0:
                    calculated_power = phase.voltage * phase.current * abs(phase.pf)
                    power_error = abs(abs(phase.power) - calculated_power) / calculated_power

                    if power_error > POWER_CONSISTENCY_THRESHOLD:
                        logger.warning(
                            "Phase power inconsistency detected",
                            phase=i,
                            measured_power=phase.power,
                            calculated_power=calculated_power,
                            error_percent=round(power_error * 100, 1),
                        )
            return True

        except (ValueError, ZeroDivisionError, TypeError) as e:
            logger.exception("Dataset validation failed", error=str(e))
            return False

    def create_partial_data_from_cache(self, max_age_seconds: float = 300.0) -> Shelly3EMData | None:
        """Create data object from cached values if recent enough.

        Args:
            max_age_seconds: Maximum age of cached data in seconds

        Returns:
            Shelly3EMData if cache is valid, None if too old
        """
        if (time.time() - self.current_data.timestamp) > max_age_seconds:
            logger.warning("Cached data too old", age_seconds=time.time() - self.current_data.timestamp)
            return None

        return self.current_data


def create_mqtt_processor(
    device_type: str = "shelly3em",
    invert_values: bool = False,  # noqa: FBT001, FBT002
    strict_validation: bool = True,  # noqa: FBT001, FBT002
) -> MQTTDataProcessor:
    """Factory function to create MQTT processor with device-specific settings.

    Args:
        device_type: Type of device ('shelly3em' supported)
        invert_values: Whether to invert power/energy signs
        strict_validation: Whether to use strict validation

    Returns:
        Configured MQTTDataProcessor instance with caching and fast calculations
    """
    if device_type != "shelly3em":
        raise ValueError(f"Unsupported device type: {device_type}")

    processor = MQTTDataProcessor(invert_values=invert_values, strict_validation=strict_validation)

    logger.info(
        "MQTT processor created",
        device_type=device_type,
        invert_values=invert_values,
        strict_validation=strict_validation,
    )

    return processor


class BatchMQTTProcessor:
    """Process MQTT messages from multiple devices."""

    def __init__(self, max_devices: int = 10):
        """Initialize batch processor.

        Args:
            max_devices: Maximum number of devices to track
        """
        self.processors: dict[str, MQTTDataProcessor] = {}
        self.max_devices = max_devices

    def process_message(
        self,
        topic: str,
        payload: bytes | str,
        qos: int = 0,
        retain: bool = False,  # noqa: FBT001, FBT002
    ) -> Shelly3EMData | None:
        """Process message and route to appropriate device processor."""
        device_id = self._extract_device_id(topic)
        if not device_id:
            return None

        if device_id not in self.processors:
            if len(self.processors) >= self.max_devices:
                oldest_device = next(iter(self.processors))
                del self.processors[oldest_device]
                logger.debug("Removed oldest device processor", device=oldest_device)

            self.processors[device_id] = MQTTDataProcessor()
            logger.info("Created processor for new device", device_id=device_id)

        return self.processors[device_id].process_message(topic, payload, qos, retain=retain)

    def _extract_device_id(self, topic: str) -> str | None:
        """Extract device ID from MQTT topic."""
        parts = topic.split("/")
        if len(parts) >= MIN_PARTS_COUNT:
            return parts[1]
        return None

    def get_all_current_data(self) -> dict[str, Shelly3EMData]:
        """Get current data from all device processors."""
        return {device_id: processor.get_current_data() for device_id, processor in self.processors.items()}

    def get_combined_stats(self) -> dict[str, Any]:
        """Get combined statistics from all processors."""
        combined = ProcessingStats()

        for processor in self.processors.values():
            stats = processor.stats
            combined.total_messages += stats.total_messages
            combined.valid_messages += stats.valid_messages
            combined.invalid_messages += stats.invalid_messages
            combined.parsing_errors += stats.parsing_errors
            combined.validation_errors += stats.validation_errors

        return {
            "devices": len(self.processors),
            "total_messages": combined.total_messages,
            "valid_messages": combined.valid_messages,
            "invalid_messages": combined.invalid_messages,
            "parsing_errors": combined.parsing_errors,
            "validation_errors": combined.validation_errors,
            "success_rate_percent": round(combined.success_rate, 2),
            "error_rate_percent": round(combined.error_rate, 2),
        }
