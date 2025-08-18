#!/usr/bin/env python3
"""
Shelly EM3 to SMA Speedwire Gateway
Emulates SMA Energy Meter protocol for integration with SMA inverters

Protocol implementation based on:
- https://github.com/Waldmensch1/venus.dbus-sma-smartmeter/
- https://github.com/Wired-Square/homeassistant-sma-sw/
- SMA EMETER Protocol Technical Information v1.0
- OBIS code structure for energy metering

Author: Grzegorz Sterniczuk
License: MIT
"""

import asyncio
import logging
import signal
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import threading
import yaml
import paho.mqtt.client as mqtt


@dataclass
class ShellyEM3Data:
    """Container for Shelly EM3 three-phase energy meter data"""

    # Instantaneous power per phase (W)
    power_a: float = 0.0
    power_b: float = 0.0
    power_c: float = 0.0

    # Voltage (V) and current (A) per phase
    voltage_a: float = 0.0
    voltage_b: float = 0.0
    voltage_c: float = 0.0
    current_a: float = 0.0
    current_b: float = 0.0
    current_c: float = 0.0

    # Power factor per phase
    pf_a: float = 1.0
    pf_b: float = 1.0
    pf_c: float = 1.0

    # Grid frequency
    freq_hz: float = 50.0

    # Energy counters (Wh)
    energy_consumed_a: float = 0.0
    energy_consumed_b: float = 0.0
    energy_consumed_c: float = 0.0
    energy_exported_a: float = 0.0
    energy_exported_b: float = 0.0
    energy_exported_c: float = 0.0

    @property
    def total_power(self) -> float:
        """Total instantaneous power across all phases"""
        return self.power_a + self.power_b + self.power_c

    @property
    def total_consumed_wh(self) -> float:
        """Total energy consumed from grid"""
        return self.energy_consumed_a + self.energy_consumed_b + self.energy_consumed_c

    @property
    def total_exported_wh(self) -> float:
        """Total energy exported to grid"""
        return self.energy_exported_a + self.energy_exported_b + self.energy_exported_c


class SpeedwireBuilder:
    """Builds SMA Speedwire protocol packets"""

    SIG = b"SMA\x00"
    TAG0_ID = 0x02A0
    DATA2_ID = 0x0010
    END_ID = 0x0000

    PROTO_EMETER = 0x6069
    PROTOCOL_DISCOVERY = 0x6081

    def __init__(self, susy_id: int, serial_number: int):
        self.susy_id = susy_id & 0xFFFF
        self.serial = serial_number & 0xFFFFFFFF

    @staticmethod
    def _obis_id_bytes(
        channel_b: int, index_c: int, type_d: int, tariff_e: int = 0
    ) -> bytes:
        """Create OBIS identifier bytes"""
        return struct.pack(
            ">BBBB", channel_b & 0xFF, index_c & 0xFF, type_d & 0xFF, tariff_e & 0xFF
        )

    @staticmethod
    def _u32(v: int) -> bytes:
        """Pack 32-bit unsigned integer"""
        return struct.pack(">I", v & 0xFFFFFFFF)

    @staticmethod
    def _u64(v: int) -> bytes:
        """Pack 64-bit unsigned integer"""
        return struct.pack(">Q", v & 0xFFFFFFFFFFFFFFFF)

    def _device_addr_and_time(self) -> bytes:
        """Create device address and timestamp header"""
        addr = struct.pack(">HI", self.susy_id, self.serial)
        ticker = int(time.time() * 1000) & 0xFFFFFFFF
        return addr + struct.pack(">I", ticker)

    def add_energy(self, parts: list, c_index: int, ws: int):
        """Add energy value (Ws) to packet"""
        parts.append(self._obis_id_bytes(0, c_index, 8, 0))
        parts.append(self._u64(max(0, int(ws))))

    def add_power(self, parts: list, c_index: int, w01: int):
        """Add power value (0.1W units) to packet"""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(max(0, int(w01))))

    def add_current_ma(self, parts: list, c_index: int, amps: float):
        """Add current value (mA) to packet"""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(int(max(0.0, amps) * 1000.0)))

    def add_voltage_mv(self, parts: list, c_index: int, volts: float):
        """Add voltage value (mV) to packet"""
        parts.append(self._obis_id_bytes(0, c_index, 4, 0))
        parts.append(self._u32(int(max(0.0, volts) * 1000.0)))

    def add_pf(self, parts: list, pf: float):
        """Add power factor value to packet"""
        v = int(round(max(0.0, min(1.0, pf)) * 1000.0))
        parts.append(self._obis_id_bytes(0, 13, 4, 0))
        parts.append(self._u32(v))

    def add_freq(self, parts: list, hz: float):
        """Add frequency value (mHz) to packet"""
        v = int(round(max(0.0, hz) * 1000.0))
        parts.append(self._obis_id_bytes(0, 14, 4, 0))
        parts.append(self._u32(v))

    def add_sw_version(
        self, parts: list, major: int, minor: int, build: int, rev_char: str
    ):
        """Add software version to packet"""
        parts.append(self._obis_id_bytes(144, 0, 0, 0))
        version = (
            ((major & 0xFF) << 24)
            | ((minor & 0xFF) << 16)
            | ((build & 0xFF) << 8)
            | (ord(rev_char) & 0xFF)
        )
        parts.append(self._u32(version))

    def build_emeter_payload(
        self,
        d: "ShellyEM3Data",
        include_vipf: bool = False,
        flip_import_export: bool = False,
    ) -> bytes:
        """Build EMETER protocol payload"""
        parts = [self._device_addr_and_time()]

        # total power and energy
        p_imp = int(round(max(0.0, d.total_power) * 10.0))
        p_exp = int(round(max(0.0, -d.total_power) * 10.0))
        e_imp = int(round(max(0.0, d.total_consumed_wh) * 3600.0))
        e_exp = int(round(max(0.0, d.total_exported_wh) * 3600.0))

        # SMA uses reversed channel mapping
        if not flip_import_export:
            p_imp, p_exp = p_exp, p_imp
            e_imp, e_exp = e_exp, e_imp

        self.add_energy(parts, 1, e_imp)
        self.add_energy(parts, 2, e_exp)
        self.add_power(parts, 1, p_imp)
        self.add_power(parts, 2, p_exp)

        # phase power split
        def split(w):
            if w >= 0:
                return int(round(w * 10.0)), 0
            else:
                return 0, int(round(-w * 10.0))

        p1_imp, p1_exp = split(d.power_a)
        p2_imp, p2_exp = split(d.power_b)
        p3_imp, p3_exp = split(d.power_c)

        # phase energy values (Wh to Ws)
        e1_imp = int(round(max(0.0, d.energy_consumed_a) * 3600.0))
        e2_imp = int(round(max(0.0, d.energy_consumed_b) * 3600.0))
        e3_imp = int(round(max(0.0, d.energy_consumed_c) * 3600.0))
        e1_exp = int(round(max(0.0, d.energy_exported_a) * 3600.0))
        e2_exp = int(round(max(0.0, d.energy_exported_b) * 3600.0))
        e3_exp = int(round(max(0.0, d.energy_exported_c) * 3600.0))

        if not flip_import_export:
            p1_imp, p1_exp = p1_exp, p1_imp
            p2_imp, p2_exp = p2_exp, p2_imp
            p3_imp, p3_exp = p3_exp, p3_imp
            e1_imp, e1_exp = e1_exp, e1_imp
            e2_imp, e2_exp = e2_exp, e2_imp
            e3_imp, e3_exp = e3_exp, e3_imp

        # Phase L1
        self.add_energy(parts, 21, e1_imp)
        self.add_energy(parts, 22, e1_exp)
        self.add_power(parts, 21, p1_imp)
        self.add_power(parts, 22, p1_exp)

        # Phase L2
        self.add_energy(parts, 41, e2_imp)
        self.add_energy(parts, 42, e2_exp)
        self.add_power(parts, 41, p2_imp)
        self.add_power(parts, 42, p2_exp)

        # Phase L3
        self.add_energy(parts, 61, e3_imp)
        self.add_energy(parts, 62, e3_exp)
        self.add_power(parts, 61, p3_imp)
        self.add_power(parts, 62, p3_exp)

        if include_vipf:
            # currents & voltages
            self.add_current_ma(parts, 31, d.current_a)
            self.add_voltage_mv(parts, 32, d.voltage_a)
            self.add_current_ma(parts, 51, d.current_b)
            self.add_voltage_mv(parts, 52, d.voltage_b)
            self.add_current_ma(parts, 71, d.current_c)
            self.add_voltage_mv(parts, 72, d.voltage_c)

            # pf and freq
            min_pf = min(max(d.pf_a, 0.0), max(d.pf_b, 0.0), max(d.pf_c, 0.0))
            self.add_pf(parts, min_pf)
            self.add_freq(parts, d.freq_hz)

        # Software version
        self.add_sw_version(parts, 4, 3, 7, "R")

        return b"".join(parts)

    def build_packet(self, payload_data2: bytes) -> bytes:
        """Build complete Speedwire packet with headers"""
        tag0_len = struct.pack(">H", 4)
        tag0_id = struct.pack(">H", self.TAG0_ID)
        group = struct.pack(">I", 1)

        data2_id = struct.pack(">H", self.DATA2_ID)
        proto = struct.pack(">H", self.PROTO_EMETER)
        data2_payload = proto + payload_data2
        data2_len = struct.pack(">H", len(data2_payload))

        end_len = struct.pack(">H", 0)
        end_id = struct.pack(">H", self.END_ID)

        return (
            self.SIG
            + tag0_len
            + tag0_id
            + group
            + data2_len
            + data2_id
            + data2_payload
            + end_len
            + end_id
        )


class SMASpeedwireEmulator:
    """Emulates SMA Energy Meter via Speedwire protocol"""

    BROADCAST_IP = "255.255.255.255"
    MULTICAST_GRP = "239.12.255.254"
    PORT = 9522
    DISCOVERY_RESP = bytes.fromhex("534d4100000402a000000001000200000001")

    def __init__(self, config: dict):
        self.config = config
        self.use_broadcast = bool(config.get("use_broadcast", False))
        self.dualcast = bool(config.get("dualcast", False))
        self.include_vi_pf_freq = bool(config.get("include_voltage_current", True))
        self.flip_import_export = bool(config.get("flip_import_export", False))

        self.susy_id = int(config.get("susy_id", 0x015D))
        self.serial = int(config.get("serial", 1234567890))

        self.builder = SpeedwireBuilder(self.susy_id, self.serial)

        self.data = ShellyEM3Data()
        self.data_lock = threading.Lock()
        self.running = True

        self.sock_send: Optional[socket.socket] = None
        self.sock_recv: Optional[socket.socket] = None
        self.local_ip = self._detect_local_ip()
        self.unicast_targets = list(config.get("unicast_targets", []))

    @staticmethod
    def _detect_local_ip() -> str:
        """Detect local IP address on the network interface"""
        try:
            hostname = socket.gethostname()
            ips = socket.gethostbyname_ex(hostname)[2]
            filtered = [ip for ip in ips if not ip.startswith("127.")]
            # Return first non-loopback IP
            if filtered:
                return filtered[0]
        except Exception:
            pass
        return "0.0.0.0"

    def setup(self):
        """Initialize network sockets"""
        # Sender socket
        self.sock_send = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_send.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        try:
            self.sock_send.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(self.local_ip),
            )
        except OSError:
            pass
        if self.use_broadcast or self.dualcast:
            self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # Receiver socket for discovery
        self.sock_recv = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except Exception:
            pass
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock_recv.bind(("", self.PORT))
        self.sock_recv.setblocking(False)

        try:
            mreq = socket.inet_aton(self.MULTICAST_GRP) + socket.inet_aton("0.0.0.0")
            self.sock_recv.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError:
            pass

        logging.info(
            f"TX: {'broadcast' if self.use_broadcast else 'multicast'}{' + broadcast' if self.dualcast else ''}, SUSy=0x{self.susy_id:04X}, Serial={self.serial}, LocalIP={self.local_ip}"
        )

    def update_data(self, data: ShellyEM3Data):
        """Update internal data from MQTT"""
        with self.data_lock:
            self.data = data

    def _is_discovery_query(self, data: bytes) -> bool:
        """Check if packet is a discovery query"""
        if len(data) < 20 or data[:4] != b"SMA\x00":
            return False
        try:
            proto = struct.unpack(">H", data[16:18])[0]
        except struct.error:
            return False
        return proto == SpeedwireBuilder.PROTOCOL_DISCOVERY

    def _send_discovery_response(self, src_addr: tuple[str, int]):
        """Send discovery response to requester"""
        try:
            self.sock_send.sendto(self.DISCOVERY_RESP, (src_addr[0], self.PORT))
            logging.info(f"Discovery response sent to {src_addr[0]}")
        except Exception as e:
            logging.warning(f"Discovery response failed: {e}")
            return

        if self.use_broadcast or self.dualcast:
            try:
                self.sock_send.sendto(
                    self.DISCOVERY_RESP, (self.BROADCAST_IP, self.PORT)
                )
                logging.debug("Discovery response broadcast sent")
            except Exception as be:
                logging.debug(f"Discovery broadcast skipped: {be}")

    async def discovery_loop(self):
        """Handle discovery requests"""
        while self.running:
            try:
                try:
                    data, addr = self.sock_recv.recvfrom(2048)
                except (BlockingIOError, InterruptedError):
                    await asyncio.sleep(0.05)
                    continue
                if self._is_discovery_query(data):
                    logging.info(f"Discovery request from {addr[0]} len={len(data)}")
                    self._send_discovery_response(addr)
            except Exception:
                pass
            await asyncio.sleep(0.01)

    def _build_and_send(self):
        """Build and send energy data packet"""
        with self.data_lock:
            d = self.data

        payload = self.builder.build_emeter_payload(
            d,
            include_vipf=self.include_vi_pf_freq,
            flip_import_export=self.flip_import_export,
        )
        packet = self.builder.build_packet(payload)

        logging.debug("EMETER TX tick")
        try:
            if self.use_broadcast:
                self.sock_send.sendto(packet, (self.BROADCAST_IP, self.PORT))
            else:
                self.sock_send.sendto(packet, (self.MULTICAST_GRP, self.PORT))
                if self.dualcast:
                    self.sock_send.sendto(packet, (self.BROADCAST_IP, self.PORT))

            for ip in self.unicast_targets:
                try:
                    self.sock_send.sendto(packet, (ip, self.PORT))
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"TX failed: {e}")

    async def tx_loop(self, interval: float):
        """Main transmission loop"""
        while self.running:
            self._build_and_send()
            await asyncio.sleep(interval)


class ShellyEM3MQTTClient:
    """MQTT client for Shelly EM3 data reception"""

    def __init__(self, config: dict, on_update):
        self.config = config
        self.on_update = on_update
        self.base_topic = config["base_topic"].rstrip("/")
        self.client = None
        self.data = ShellyEM3Data()
        self.lock = threading.Lock()

    def setup(self):
        """Initialize MQTT client"""
        device_id = (
            self.base_topic.split("-")[-1] if "-" in self.base_topic else "default"
        )
        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"speedwire_gateway_{device_id}",
            )
            self.client.on_connect = self._on_connect_v2
        except Exception:
            self.client = mqtt.Client(client_id=f"speedwire_gateway_{device_id}")
            self.client.on_connect = self._on_connect_v1

        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        if "username" in self.config:
            self.client.username_pw_set(
                self.config["username"], self.config.get("password", "")
            )

    def _on_connect_v2(self, client, userdata, flags, reason_code, properties):
        """MQTT v2 connection callback"""
        logging.info(f"MQTT connected: {reason_code}")
        self._subscribe(client)

    def _on_connect_v1(self, client, userdata, flags, rc):
        """MQTT v1 connection callback"""
        logging.info(f"MQTT connected (rc={rc})")
        self._subscribe(client)

    def _subscribe(self, client):
        """Subscribe to Shelly topics"""
        emeter_topic = f"{self.base_topic}/emeter/+/+"
        client.subscribe(emeter_topic, qos=1)
        logging.info(f"Subscribed to {emeter_topic}")

        online_topic = f"{self.base_topic}/online"
        client.subscribe(online_topic, qos=1)

    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="ignore")

        if topic.endswith("/online"):
            logging.info(f"Shelly online: {payload}")
            return

        try:
            value = float(payload)
        except ValueError:
            return

        if "/emeter/" in topic:
            try:
                relative_topic = topic[len(self.base_topic) + 1 :]
                parts = relative_topic.split("/")

                if len(parts) >= 3 and parts[0] == "emeter":
                    phase = int(parts[1])
                    key = parts[2]

                    with self.lock:
                        d = self.data
                        if phase == 0:
                            if key == "power":
                                d.power_a = value
                            elif key == "voltage":
                                d.voltage_a = value
                            elif key == "current":
                                d.current_a = value
                            elif key == "pf":
                                d.pf_a = abs(value)
                            elif key == "total_returned":
                                d.energy_consumed_a = value
                            elif key == "total":
                                d.energy_exported_a = value
                        elif phase == 1:
                            if key == "power":
                                d.power_b = value
                            elif key == "voltage":
                                d.voltage_b = value
                            elif key == "current":
                                d.current_b = value
                            elif key == "pf":
                                d.pf_b = abs(value)
                            elif key == "total_returned":
                                d.energy_consumed_b = value
                            elif key == "total":
                                d.energy_exported_b = value
                        elif phase == 2:
                            if key == "power":
                                d.power_c = value
                            elif key == "voltage":
                                d.voltage_c = value
                            elif key == "current":
                                d.current_c = value
                            elif key == "pf":
                                d.pf_c = abs(value)
                            elif key == "total_returned":
                                d.energy_consumed_c = value
                            elif key == "total":
                                d.energy_exported_c = value

                        if self.on_update:
                            self.on_update(self.data)
            except Exception as e:
                logging.debug(f"Topic parse error: {e}")

    def connect(self):
        """Connect to MQTT broker"""
        self.client.connect(
            self.config["broker_host"],
            int(self.config.get("broker_port", 1883)),
            int(self.config.get("keepalive", 60)),
        )
        self.client.loop_start()

    def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def _on_disconnect(self, client, userdata, rc, *args):
        """MQTT disconnect callback"""
        logging.warning(f"MQTT disconnected rc={rc}")


class ShellyEM3SpeedwireGateway:
    """Main gateway application"""

    def __init__(self, config_path: str = "config_speedwire.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()

        self.mqtt = ShellyEM3MQTTClient(self.config["mqtt"], self._on_data_update)
        self.sw = SMASpeedwireEmulator(self.config["speedwire"])

        self.running = True
        self.last_update = time.time()
        self.count = 0

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _load_config(self, cfg_path: str) -> dict:
        """Load configuration from YAML file"""
        p = Path(cfg_path)

        yaml_path = p.with_suffix(".yaml")
        yml_path = p.with_suffix(".yml")

        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        elif yml_path.exists():
            with open(yml_path, "r") as f:
                return yaml.safe_load(f)
        else:
            default = {
                "mqtt": {
                    "broker_host": "localhost",
                    "broker_port": 1883,
                    "base_topic": "shellies/shellyem3-XXXXXXXXXXXXX",
                    "keepalive": 60,
                    "invert_power": True,
                },
                "speedwire": {
                    "interval": 1.0,
                    "use_broadcast": False,
                    "dualcast": False,
                    "push_on_update": True,
                    "min_send_interval": 0.2,
                    "heartbeat_interval": 10.0,
                    "flip_import_export": False,
                    "include_voltage_current": True,
                    "include_sw_version": True,
                    "serial": 1234567890,
                    "susy_id": 349,
                },
                "logging": {"level": "INFO", "file": "speedwire_gateway.log"},
            }

            with open(yaml_path, "w") as f:
                yaml.dump(default, f, default_flow_style=False, sort_keys=False)

            logging.info(f"Created default config: {yaml_path}")
            return default

    def _setup_logging(self):
        """Configure logging"""
        cfg = self.config.get("logging", {})
        level = getattr(logging, cfg.get("level", "INFO"))
        handlers = [logging.StreamHandler()]
        if "file" in cfg:
            handlers.append(logging.FileHandler(cfg["file"]))
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=handlers,
        )
        logging.getLogger("paho").setLevel(logging.WARNING)

    def _on_data_update(self, data: ShellyEM3Data):
        """Handle data update from MQTT"""
        self.sw.update_data(data)
        self.last_update = time.time()
        self.count += 1
        if self.count % 10 == 0:
            logging.info(
                "Psum=%+6.1f W | L1=%+6.1f W L2=%+6.1f W L3=%+6.1f W | E+=%.3f kWh E-=%.3f kWh",
                data.total_power,
                data.power_a,
                data.power_b,
                data.power_c,
                data.total_consumed_wh / 1000.0,
                data.total_exported_wh / 1000.0,
            )

    async def run(self):
        """Main gateway execution loop"""
        logging.info("=== Shelly EM3 to SMA Speedwire Gateway ===")
        logging.info(
            f"MQTT: {self.config['mqtt']['broker_host']}:{self.config['mqtt'].get('broker_port',1883)}"
        )
        logging.info(f"TX interval: {self.config['speedwire'].get('interval',1.0)} s")
        try:
            self.mqtt.setup()
            self.sw.setup()
            self.mqtt.connect()

            await asyncio.gather(
                self.sw.tx_loop(self.config["speedwire"].get("interval", 1.0)),
                self.sw.discovery_loop(),
            )
        finally:
            self.mqtt.disconnect()

    def _shutdown(self, *_):
        """Graceful shutdown handler"""
        logging.info("Shutting down...")
        self.sw.running = False
        self.running = False


def main():
    gw = ShellyEM3SpeedwireGateway("config_speedwire.yaml")
    try:
        asyncio.run(gw.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception("Fatal error: %s", e)


if __name__ == "__main__":
    main()
