"""SMA Speedwire protocol implementation for energy meter emulation.

This module implements the SMA Speedwire EMETER protocol with
async patterns and packet building capabilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
import time
from typing import Any

import structlog

try:
    from shelly_speedwire_gateway.metrics import (
        speedwire_packets_errors,
        speedwire_packets_sent,
        speedwire_send_time,
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

from shelly_speedwire_gateway.constants import (
    BROADCAST_IP,
    DATA2_ID,
    DEFAULT_IPV4_ADDR,
    DISCOVERY_RESPONSE,
    END_ID,
    MULTICAST_GROUP,
    OBIS_CHANNELS,
    PROTO_EMETER,
    PROTOCOL_DISCOVERY,
    RECEIVE_BUFFER_SIZE,
    SMA_PORT,
    SMA_SIGNATURE,
    TAG0_ID,
)
from shelly_speedwire_gateway.exceptions import NetworkError
from shelly_speedwire_gateway.models import DeviceConfig, NetworkConfig, Shelly3EMData
from shelly_speedwire_gateway.power_calculator import PowerCalculator

MIN_PACKET_SIZE = 20
PROTO_BYTES_SIZE = 2

logger = structlog.get_logger(__name__)


class SpeedwirePacketBuilder:
    """Builds SMA Speedwire protocol packets with OBIS data."""

    def __init__(self, device_config: DeviceConfig) -> None:
        """Initialize builder with device configuration."""
        self.susy_id = device_config.susy_id & 0xFFFF
        self.serial = device_config.serial & 0xFFFFFFFF
        self.sw_version = device_config.software_version

    @staticmethod
    def _obis_id_bytes(channel_b: int, index_c: int, type_d: int, tariff_e: int = 0) -> bytes:
        """Create OBIS identifier bytes."""
        return struct.pack(">BBBB", channel_b & 0xFF, index_c & 0xFF, type_d & 0xFF, tariff_e & 0xFF)

    @staticmethod
    def _u32(value: int) -> bytes:
        """Pack 32-bit unsigned integer."""
        return struct.pack(">I", value & 0xFFFFFFFF)

    @staticmethod
    def _u64(value: int) -> bytes:
        """Pack 64-bit unsigned integer."""
        return struct.pack(">Q", value & 0xFFFFFFFFFFFFFFFF)

    def _device_addr_and_time(self) -> bytes:
        """Create device address and timestamp header."""
        addr = struct.pack(">HI", self.susy_id, self.serial)
        ticker = int(time.time() * 1000) & 0xFFFFFFFF
        return addr + struct.pack(">I", ticker)

    def _add_energy(self, parts: list[bytes], c_index: int, ws: int) -> None:
        """Add energy value (Ws) to packet."""
        parts.append(self._obis_id_bytes(0, c_index, 8, 0))
        parts.append(self._u64(max(0, int(ws))))

    def _add_power(self, parts: list[bytes], c_index: int, w01: int) -> None:
        """Add power value (0.1W units) to packet."""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(max(0, int(w01))))

    def _add_current_ma(self, parts: list[bytes], c_index: int, amps: float) -> None:
        """Add current value (mA) to packet."""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(int(max(0.0, amps) * 1000)))

    def _add_voltage_mv(self, parts: list[bytes], c_index: int, volts: float) -> None:
        """Add voltage value (mV) to packet."""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(int(max(0.0, volts) * 1000)))

    def _add_pf(self, parts: list[bytes], pf: float) -> None:
        """Add total power factor."""
        value = round(max(-1.0, min(1.0, pf)) * 1000.0)
        parts.append(self._obis_id_bytes(0, OBIS_CHANNELS["power_factor"], 4, 0))

        if value < 0:
            value = (0x100000000 + value) & 0xFFFFFFFF

        parts.append(self._u32(value))

    def _add_pf_phase(self, parts: list[bytes], c_index: int, pf: float) -> None:
        """Add phase power factor."""
        value = round(max(-1.0, min(1.0, pf)) * 1000.0)
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))

        if value < 0:
            value = (0x100000000 + value) & 0xFFFFFFFF

        parts.append(self._u32(value))

    def _add_freq(self, parts: list[bytes], hz: float) -> None:
        """Add frequency value (mHz) to packet."""
        value = round(max(0.0, hz) * 1000.0)
        parts.append(self._obis_id_bytes(0, OBIS_CHANNELS["frequency"], 4, 0))
        parts.append(self._u32(value))

    def _add_reactive_power(self, parts: list[bytes], c_index: int, var01: int) -> None:
        """Add reactive power (0.1VAr units)."""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(max(0, int(var01))))

    def _add_apparent_power(self, parts: list[bytes], c_index: int, va01: int) -> None:
        """Add apparent power (0.1VA units)."""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(max(0, int(va01))))

    def _add_sw_version(self, parts: list[bytes]) -> None:
        """Add software version (channel 36864)."""
        parts.append(self._obis_id_bytes(144, 0, 0, 0))
        major, minor, build, rev_char = self.sw_version

        version = ((major & 0xFF) << 24) | ((minor & 0xFF) << 16) | ((build & 0xFF) << 8) | (ord(rev_char) & 0xFF)

        parts.append(self._u32(version))

    def _add_total_values(self, parts: list[bytes], data: Shelly3EMData) -> None:
        """Add total power and energy values to packet."""
        power_components = PowerCalculator.calculate_total_powers(data)

        active_split = PowerCalculator.split_power(power_components.active)
        reactive_split = PowerCalculator.split_power(power_components.reactive)
        apparent_split = PowerCalculator.split_power(power_components.apparent)

        p_imp = active_split.import_value
        p_exp = active_split.export_value
        q_imp = reactive_split.import_value
        q_exp = reactive_split.export_value
        s_imp = apparent_split.import_value
        s_exp = apparent_split.export_value

        e_imp = round(max(0.0, data.total_consumed_wh) * 3600)
        e_exp = round(max(0.0, data.total_exported_wh) * 3600)

        self._add_energy(parts, OBIS_CHANNELS["total_energy_import"], e_imp)
        self._add_energy(parts, OBIS_CHANNELS["total_energy_export"], e_exp)

        self._add_power(parts, OBIS_CHANNELS["total_power_import"], p_imp)
        self._add_power(parts, OBIS_CHANNELS["total_power_export"], p_exp)
        self._add_reactive_power(parts, OBIS_CHANNELS["total_reactive_power_q1"], q_imp)
        self._add_reactive_power(parts, OBIS_CHANNELS["total_reactive_power_q2"], q_exp)
        self._add_apparent_power(parts, OBIS_CHANNELS["total_apparent_power_import"], s_imp)
        self._add_apparent_power(parts, OBIS_CHANNELS["total_apparent_power_export"], s_exp)

    def _add_phase_values(self, parts: list[bytes], phase_data: Any, phase_prefix: str) -> None:
        """Add values for a single phase."""
        power_components = PowerCalculator.calculate_power_components(
            phase_data.voltage,
            phase_data.current,
            phase_data.pf,
        )

        active_split = PowerCalculator.split_power(power_components.active)
        reactive_split = PowerCalculator.split_power(power_components.reactive)
        apparent_split = PowerCalculator.split_power(power_components.apparent)

        p_imp = active_split.import_value
        p_exp = active_split.export_value
        q_imp = reactive_split.import_value
        q_exp = reactive_split.export_value
        s_imp = apparent_split.import_value
        s_exp = apparent_split.export_value

        e_imp = round(max(0.0, phase_data.energy_consumed) * 3600)
        e_exp = round(max(0.0, phase_data.energy_exported) * 3600)

        self._add_energy(parts, OBIS_CHANNELS[f"{phase_prefix}_energy_import"], e_imp)
        self._add_energy(parts, OBIS_CHANNELS[f"{phase_prefix}_energy_export"], e_exp)
        self._add_power(parts, OBIS_CHANNELS[f"{phase_prefix}_power_import"], p_imp)
        self._add_power(parts, OBIS_CHANNELS[f"{phase_prefix}_power_export"], p_exp)
        self._add_reactive_power(parts, OBIS_CHANNELS[f"{phase_prefix}_reactive_power_q1"], q_imp)
        self._add_reactive_power(parts, OBIS_CHANNELS[f"{phase_prefix}_reactive_power_q2"], q_exp)
        self._add_apparent_power(parts, OBIS_CHANNELS[f"{phase_prefix}_apparent_power_import"], s_imp)
        self._add_apparent_power(parts, OBIS_CHANNELS[f"{phase_prefix}_apparent_power_export"], s_exp)

    def _add_voltage_current_pf(self, parts: list[bytes], data: Shelly3EMData) -> None:
        """Add voltage, current, and power factor values."""
        phases = [data.a, data.b, data.c]
        phase_names = ["l1", "l2", "l3"]

        for phase, phase_name in zip(phases, phase_names, strict=False):
            current = abs(phase.current) * (1 if phase.power >= 0 else -1)

            self._add_current_ma(parts, OBIS_CHANNELS[f"{phase_name}_current"], current)
            self._add_voltage_mv(parts, OBIS_CHANNELS[f"{phase_name}_voltage"], phase.voltage)
            self._add_pf_phase(parts, OBIS_CHANNELS[f"{phase_name}_power_factor"], phase.pf)

        pf_values = [data.a.pf, data.b.pf, data.c.pf]
        worst_pf = min(pf_values, key=abs)
        self._add_pf(parts, worst_pf)
        self._add_freq(parts, data.freq_hz)

    def build_emeter_payload(self, data: Shelly3EMData, include_vipf: bool = True) -> bytes:  # noqa: FBT001, FBT002
        """Build EMETER protocol payload."""
        parts = [self._device_addr_and_time()]

        self._add_total_values(parts, data)

        self._add_phase_values(parts, data.a, "l1")
        self._add_phase_values(parts, data.b, "l2")
        self._add_phase_values(parts, data.c, "l3")

        if include_vipf:
            self._add_voltage_current_pf(parts, data)

        self._add_sw_version(parts)

        return b"".join(parts)

    def build_packet(self, payload_data2: bytes) -> bytes:
        """Build complete Speedwire packet with headers."""
        tag0_len = struct.pack(">H", 4)
        tag0_id = struct.pack(">H", TAG0_ID)
        group = struct.pack(">I", 1)

        data2_id = struct.pack(">H", DATA2_ID)
        proto = struct.pack(">H", PROTO_EMETER)
        data2_payload = proto + payload_data2
        data2_len = struct.pack(">H", len(data2_payload))

        end_len = struct.pack(">H", 0)
        end_id = struct.pack(">H", END_ID)

        return SMA_SIGNATURE + tag0_len + tag0_id + group + data2_len + data2_id + data2_payload + end_len + end_id


class NetworkManager:
    """Manages network sockets for Speedwire communication."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize network manager."""
        self.config = config
        self.sock_send: socket.socket | None = None
        self.sock_recv: socket.socket | None = None

    async def setup(self) -> None:
        """Initialize network sockets asynchronously."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._setup_sender_socket)
            await asyncio.get_event_loop().run_in_executor(None, self._setup_receiver_socket)
            logger.info("Network sockets initialized")
        except OSError as e:
            raise NetworkError(f"Failed to setup network: {e}") from e

    def _setup_sender_socket(self) -> None:
        """Setup sender socket for transmitting data."""
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_send.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)

        with contextlib.suppress(OSError):
            self.sock_send.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(self.config.local_ip),
            )

        if self.config.use_broadcast or self.config.dualcast:
            self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def _setup_receiver_socket(self) -> None:
        """Setup receiver socket for discovery requests."""
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        with contextlib.suppress(OSError, AttributeError):
            self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock_recv.bind(("", SMA_PORT))
        # Set socket to non-blocking mode
        is_blocking = False
        self.sock_recv.setblocking(is_blocking)

        try:
            mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(DEFAULT_IPV4_ADDR)
            self.sock_recv.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError:
            pass

    async def send_packet(self, packet: bytes) -> None:
        """Send packet via configured method."""
        if not self.sock_send:
            return

        def _send() -> None:
            if not self.sock_send:
                return
            try:
                if self.config.use_broadcast:
                    self.sock_send.sendto(packet, (BROADCAST_IP, SMA_PORT))
                else:
                    self.sock_send.sendto(packet, (MULTICAST_GROUP, SMA_PORT))
                    if self.config.dualcast:
                        self.sock_send.sendto(packet, (BROADCAST_IP, SMA_PORT))

                for ip in self.config.unicast_targets:
                    with contextlib.suppress(OSError):
                        self.sock_send.sendto(packet, (ip, SMA_PORT))
            except OSError as e:
                logger.exception("Packet transmission failed", error=str(e))

        await asyncio.get_event_loop().run_in_executor(None, _send)

    async def send_discovery_response(self, addr: tuple[str, int]) -> None:
        """Send discovery response to requester."""
        if not self.sock_send:
            return

        def _send_discovery() -> None:
            if not self.sock_send:
                return
            try:
                self.sock_send.sendto(DISCOVERY_RESPONSE, (addr[0], SMA_PORT))
                logger.debug("Discovery response sent", target=addr[0])
            except OSError as e:
                logger.warning("Discovery response failed", error=str(e))

            if self.config.use_broadcast or self.config.dualcast:
                try:
                    self.sock_send.sendto(DISCOVERY_RESPONSE, (BROADCAST_IP, SMA_PORT))
                    logger.debug("Discovery response broadcast sent")
                except OSError:
                    pass

        await asyncio.get_event_loop().run_in_executor(None, _send_discovery)

    async def receive_data(self) -> tuple[bytes, tuple[str, int]] | None:
        """Receive data from socket asynchronously."""
        if not self.sock_recv:
            return None

        def _receive() -> tuple[bytes, tuple[str, int]] | None:
            if not self.sock_recv:
                return None
            try:
                return self.sock_recv.recvfrom(RECEIVE_BUFFER_SIZE)
            except (BlockingIOError, InterruptedError, OSError):
                return None

        return await asyncio.get_event_loop().run_in_executor(None, _receive)

    async def close(self) -> None:
        """Close sockets."""

        def _close() -> None:
            if self.sock_send:
                self.sock_send.close()
            if self.sock_recv:
                self.sock_recv.close()

        await asyncio.get_event_loop().run_in_executor(None, _close)


class SMASpeedwireEmulator:
    """Emulates SMA Energy Meter via Speedwire protocol with async support."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize emulator with configuration."""
        self.device_config = DeviceConfig(
            susy_id=int(config.get("susy_id", 0x015D)),
            serial=int(config.get("serial", 1234567890)),
        )

        self.network_config = NetworkConfig(
            use_broadcast=bool(config.get("use_broadcast", False)),
            dualcast=bool(config.get("dualcast", False)),
            local_ip=self._detect_local_ip(),
            unicast_targets=list(config.get("unicast_targets", [])),
        )

        self.builder = SpeedwirePacketBuilder(self.device_config)
        self.network = NetworkManager(self.network_config)
        self.data = Shelly3EMData()
        self.running = False

    @staticmethod
    def _detect_local_ip() -> str:
        """Detect local IP address on the network interface."""
        try:
            hostname = socket.gethostname()
            ips = socket.gethostbyname_ex(hostname)[2]
            filtered = [ip for ip in ips if not ip.startswith("127.")]
            if filtered:
                return filtered[0]
        except (socket.gaierror, socket.herror, OSError):
            pass
        return DEFAULT_IPV4_ADDR

    async def setup(self) -> None:
        """Initialize network sockets."""
        await self.network.setup()
        self._log_setup_info()

    def _log_setup_info(self) -> None:
        """Log setup information."""
        tx_mode = "broadcast" if self.network_config.use_broadcast else "multicast"
        if self.network_config.dualcast:
            tx_mode += " + broadcast"

        logger.info(
            "Speedwire emulator setup complete",
            tx_mode=tx_mode,
            susy_id=f"0x{self.device_config.susy_id:04X}",
            serial=self.device_config.serial,
            local_ip=self.network_config.local_ip,
        )

    async def update_data(self, data: Shelly3EMData) -> None:
        """Update internal data from MQTT."""
        self.data = data

    @staticmethod
    def _is_discovery_query(data: bytes) -> bool:
        """Check if packet is a discovery query."""
        if len(data) < MIN_PACKET_SIZE or data[:4] != SMA_SIGNATURE:
            return False

        try:
            proto_bytes = data[16:18]
            if len(proto_bytes) < PROTO_BYTES_SIZE:
                return False
            proto = struct.unpack(">H", proto_bytes)[0]
        except (struct.error, IndexError):
            return False

        # R1705: Remove else after return
        return bool(proto == PROTOCOL_DISCOVERY)

    async def discovery_loop(self) -> None:
        """Handle discovery requests."""
        self.running = True
        while self.running:
            try:
                result = await self.network.receive_data()
                if result is None:
                    await asyncio.sleep(0.05)
                    continue

                data, addr = result
                if self._is_discovery_query(data):
                    logger.debug("Discovery request received", source=addr[0], packet_size=len(data))
                    await self.network.send_discovery_response(addr)
            except (OSError, ConnectionError, ValueError) as e:
                logger.warning("Error in discovery loop", error=str(e))

            await asyncio.sleep(0.01)

    async def _build_and_send(self) -> None:
        """Build and send energy data packet."""
        if METRICS_ENABLED:
            with speedwire_send_time.time():
                try:
                    payload = self.builder.build_emeter_payload(self.data, include_vipf=True)
                    packet = self.builder.build_packet(payload)

                    logger.debug("Sending EMETER packet", packet_size=len(packet))
                    await self.network.send_packet(packet)
                    speedwire_packets_sent.inc()
                except Exception as e:
                    speedwire_packets_errors.inc()
                    logger.exception("Failed to send Speedwire packet", error=str(e))
                    raise
        else:
            payload = self.builder.build_emeter_payload(self.data, include_vipf=True)
            packet = self.builder.build_packet(payload)

            logger.debug("Sending EMETER packet", packet_size=len(packet))
            await self.network.send_packet(packet)

    async def tx_loop(self, interval: float) -> None:
        """Main transmission loop."""
        self.running = True
        while self.running:
            await self._build_and_send()
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        """Stop the emulator."""
        self.running = False
        await self.network.close()
        logger.info("Speedwire emulator stopped")
