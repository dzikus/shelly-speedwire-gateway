# Shelly EM3 to SMA Speedwire Gateway

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/dzikus/shelly-speedwire-gateway)
[![Docker Pulls](https://img.shields.io/docker/pulls/dzikus99/shelly-speedwire-gateway)](https://hub.docker.com/r/dzikus99/shelly-speedwire-gateway)
[![Docker Image Size](https://img.shields.io/docker/image-size/dzikus99/shelly-speedwire-gateway/latest)](https://hub.docker.com/r/dzikus99/shelly-speedwire-gateway)

A Python-based gateway that enables Shelly EM3 three-phase energy meters to communicate with SMA inverters by emulating the SMA Energy Meter protocol via Speedwire.

## Features

- **Real-time energy monitoring** - Forwards power, voltage, current, and energy data from Shelly EM3 to SMA devices
- **Three-phase support** - Full support for all three phases with individual measurements
- **SMA Speedwire protocol** - Complete implementation of SMA EMETER protocol v1.0
- **Multiple deployment options** - Run as standalone Python script, systemd service, or Docker container
- **Flexible networking** - Supports multicast, broadcast, and direct unicast communication
- **Auto-discovery** - Responds to SMA device discovery requests
- **Low latency** - Configurable transmission intervals with immediate updates on data change

## Requirements

### Hardware
- **Shelly EM3** energy meter connected to your electrical installation
- **SMA inverter** or device that supports Speedwire protocol
- Network connectivity between Shelly EM3, MQTT broker, and SMA devices

### Software
- Python 3.7+ (for standalone installation)
- Docker (for containerized deployment)
- MQTT broker (e.g., Mosquitto)

## Quick Start

### Docker (Recommended)

```yaml
version: '3.8'

services:
  shelly-speedwire-gateway:
    image: dzikus99/shelly-speedwire-gateway:latest
    container_name: shelly-speedwire
    restart: unless-stopped
    network_mode: host  # REQUIRED for multicast/broadcast
    environment:
      - MQTT_BROKER_HOST=192.168.1.123
      - MQTT_BASE_TOPIC=shellies/shellyem3-XXXXXXXXXXXXX
      - SPEEDWIRE_SERIAL=1234567890
```

```bash
docker-compose up -d
```

### Docker Run

```bash
docker run -d \
  --name shelly-speedwire \
  --network host \
  --restart unless-stopped \
  -e MQTT_BROKER_HOST=192.168.1.123 \
  -e MQTT_BASE_TOPIC=shellies/shellyem3-XXXXXXXXXXXXX \
  -e SPEEDWIRE_SERIAL=1234567890 \
  -e LOG_LEVEL=INFO \
  dzikus99/shelly-speedwire-gateway:latest
```

## Installation

### Option 1: Docker Container

Pull the image from Docker Hub:
```bash
docker pull dzikus99/shelly-speedwire-gateway:latest
```

Or build locally:
```bash
docker build -t shelly-speedwire-gateway .
```

### Option 2: Systemd Service (Linux)

1. **Clone the repository:**
```bash
git clone https://github.com/dzikus/shelly-speedwire-gateway
cd shelly-speedwire-gateway
```

2. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

3. **Copy files to system location:**
```bash
sudo mkdir -p /opt/shelly-speedwire-gateway
sudo cp shelly_speedwire_gateway.py /opt/shelly-speedwire-gateway/
sudo cp shelly_speedwire_gateway_config.yaml /opt/shelly-speedwire-gateway/
```

4. **Install systemd service:**
```bash
sudo cp shelly-speedwire-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable shelly-speedwire-gateway
sudo systemctl start shelly-speedwire-gateway
```

5. **Check status:**
```bash
sudo systemctl status shelly-speedwire-gateway
sudo journalctl -u shelly-speedwire-gateway -f
```

### Option 3: Standalone Python

1. **Install dependencies:**
```bash
pip3 install paho-mqtt pyyaml
```

2. **Configure the gateway** (see Configuration section)

3. **Run the gateway:**
```bash
python3 shelly_speedwire_gateway.py
```

## Configuration

The gateway can be configured using either a YAML configuration file or environment variables (Docker only).

### Configuration File

Create or edit `shelly_speedwire_gateway_config.yaml`:

```yaml
mqtt:
  broker_host: 192.168.1.123          # Your MQTT broker IP
  broker_port: 1883
  base_topic: shellies/shellyem3-XXXXXXXXXXXXX  # Your Shelly EM3 topic
  keepalive: 60
  # username: mqtt_user               # Optional MQTT auth
  # password: mqtt_pass

speedwire:
  interval: 1.0                        # Send interval in seconds
  use_broadcast: false                 # Use broadcast instead of multicast
  dualcast: false                      # Send both multicast and broadcast
  push_on_update: true                 # Send immediately on MQTT update
  min_send_interval: 0.2               # Minimum seconds between packets
  heartbeat_interval: 10.0             # Maximum seconds without sending
  flip_import_export: false            # Reverse import/export mapping
  serial: 1234567890                   # Unique serial number
  susy_id: 349                         # SUSy ID (349 = standard EMETER-20)
  include_voltage_current: true        # Include V, I, PF, Hz data
  include_sw_version: true             # Include software version

  # Optional: Direct unicast to specific devices
  # unicast_targets:
  #   - 192.168.1.50                  # SMA inverter IP
  #   - 192.168.1.51

logging:
  level: INFO                          # DEBUG, INFO, WARNING, ERROR
```

### Environment Variables (Docker)

When using Docker, you can configure the gateway using environment variables instead of a configuration file.

#### MQTT Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `MQTT_BROKER_HOST` | `localhost` | MQTT broker IP address or hostname | No |
| `MQTT_BROKER_PORT` | `1883` | MQTT broker port | No |
| `MQTT_BASE_TOPIC` | `shellies/shellyem3-XXXXXXXXXXXXX` | Your Shelly EM3 MQTT topic. Format: `shellies/shellyem3-[DEVICE_ID]` | Yes |
| `MQTT_KEEPALIVE` | `60` | MQTT connection keepalive interval in seconds | No |
| `MQTT_USERNAME` | - | MQTT broker username for authentication | No |
| `MQTT_PASSWORD` | - | MQTT broker password for authentication | No |

#### Speedwire Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `SPEEDWIRE_INTERVAL` | `1.0` | How often to send data packets in seconds. Lower values = more frequent updates | No |
| `SPEEDWIRE_USE_BROADCAST` | `false` | Use UDP broadcast (255.255.255.255) instead of multicast. Enable if multicast doesn't work in your network | No |
| `SPEEDWIRE_DUALCAST` | `false` | Send both multicast and broadcast packets. Useful for mixed network environments | No |
| `SPEEDWIRE_PUSH_ON_UPDATE` | `true` | Send packet immediately when new MQTT data arrives (reduces latency) | No |
| `SPEEDWIRE_MIN_SEND_INTERVAL` | `0.2` | Minimum seconds between packets when push_on_update is enabled | No |
| `SPEEDWIRE_HEARTBEAT_INTERVAL` | `10.0` | Maximum seconds without sending a packet (ensures SMA device knows gateway is alive) | No |
| `SPEEDWIRE_FLIP_IMPORT_EXPORT` | `false` | Reverse import/export power mapping in SMA device. Set to `true` if SMA device shows consumption as production and vice versa | No |
| `SPEEDWIRE_SERIAL` | `1234567890` | Unique serial number for the emulated energy meter. **Change this to a unique value!** | Yes |
| `SPEEDWIRE_SUSY_ID` | `349` | SUSy ID for device identification. 349 (0x015D) = standard SMA EMETER-20 | No |
| `SPEEDWIRE_INCLUDE_VOLTAGE_CURRENT` | `true` | Include voltage, current, power factor, and frequency in packets | No |
| `SPEEDWIRE_INCLUDE_SW_VERSION` | `true` | Include emulated software version (4.3.7.R) in packets | No |

#### Logging Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` | No |

### Using Configuration File with Docker

To use a configuration file instead of environment variables:

```yaml
version: '3.8'

services:
  shelly-speedwire-gateway:
    image: dzikus99/shelly-speedwire-gateway:latest
    container_name: shelly-speedwire
    restart: unless-stopped
    network_mode: host
    volumes:
      - ./shelly_speedwire_gateway_config.yaml:/app/shelly_speedwire_gateway_config.yaml:ro
```

**Note:** When mounting a config file, environment variables are ignored.

## Network Configuration

### Network Requirements

- **UDP Port 9522** - Must be open for Speedwire communication
- **Multicast** - Address 239.12.255.254 (default mode)
- **Broadcast** - Address 255.255.255.255 (alternative mode)
- **Network mode** - Docker containers MUST use `network_mode: host`

### Network Modes

1. **Multicast (Default)**
   - Most efficient for multiple SMA devices
   - Requires multicast routing enabled
   - Uses address 239.12.255.254

2. **Broadcast**
   - Works in all network configurations
   - Higher network load
   - Enable with `SPEEDWIRE_USE_BROADCAST=true`

3. **Dualcast**
   - Sends both multicast and broadcast
   - Maximum compatibility
   - Enable with `SPEEDWIRE_DUALCAST=true`

4. **Unicast**
   - Direct communication to specific devices
   - Configure via `unicast_targets` in config file
   - Most reliable for single device setups

## Data Mapping

The gateway translates Shelly EM3 MQTT data to SMA Speedwire protocol using OBIS codes:

| Shelly EM3 Data | SMA Speedwire Field | OBIS Code | Unit | Description |
|-----------------|---------------------|-----------|------|-------------|
| Total Power | Total Active Power | 1.4.0 / 2.4.0 | W | Positive = Import, Negative = Export |
| Phase A Power | L1 Active Power | 21.4.0 / 22.4.0 | W | Phase 1 instantaneous power |
| Phase B Power | L2 Active Power | 41.4.0 / 42.4.0 | W | Phase 2 instantaneous power |
| Phase C Power | L3 Active Power | 61.4.0 / 62.4.0 | W | Phase 3 instantaneous power |
| Phase A Voltage | L1 Voltage | 32.4.0 | V | Phase 1 voltage |
| Phase B Voltage | L2 Voltage | 52.4.0 | V | Phase 2 voltage |
| Phase C Voltage | L3 Voltage | 72.4.0 | V | Phase 3 voltage |
| Phase A Current | L1 Current | 31.4.0 | A | Phase 1 current |
| Phase B Current | L2 Current | 51.4.0 | A | Phase 2 current |
| Phase C Current | L3 Current | 71.4.0 | A | Phase 3 current |
| Power Factor | Minimum PF | 13.4.0 | - | Lowest PF of all phases |
| Grid Frequency | Frequency | 14.4.0 | Hz | Grid frequency |
| Energy Consumed | Import Energy | 1.8.0 / 21.8.0 / 41.8.0 / 61.8.0 | Wh | Cumulative energy from grid |
| Energy Exported | Export Energy | 2.8.0 / 22.8.0 / 42.8.0 / 62.8.0 | Wh | Cumulative energy to grid |

### Power Flow Direction

By default, the gateway assumes:
- **Positive power** = Energy consumption from grid
- **Negative power** = Energy export to grid (solar feed-in)

If your setup shows inverted values use `SPEEDWIRE_FLIP_IMPORT_EXPORT` if SMA device shows reversed flow

## Finding Your Shelly EM3 Topic

### Method 1: MQTT Explorer
1. Connect to your MQTT broker using MQTT Explorer
2. Look for topics starting with `shellies/shellyem3-`
3. The device ID is the part after `shellyem3-`

### Method 2: Command Line
```bash
mosquitto_sub -h YOUR_BROKER_IP -t "shellies/#" -v
```

### Method 3: Shelly Web Interface
1. Access your Shelly EM3 web interface
2. Go to Settings → Device Info
3. The device ID is shown there

## Troubleshooting

### Gateway Not Starting

**Check MQTT connectivity:**
```bash
ping YOUR_BROKER_IP
mosquitto_sub -h YOUR_BROKER_IP -t "shellies/shellyem3-XXXXXXXXXXXXX/#" -v
```

**Enable debug logging to see detailed error messages:**
```bash
# Docker
docker run -e LOG_LEVEL=DEBUG dzikus99/shelly-speedwire-gateway:latest

# Systemd
# Edit /opt/shelly-speedwire-gateway/shelly_speedwire_gateway_config.yaml
# Set logging.level to DEBUG, then restart:
sudo systemctl restart shelly-speedwire-gateway

# Standalone
# Set LOG_LEVEL=DEBUG in config file or run directly
```

**Check logs:**
```bash
# Docker
docker logs shelly-speedwire

# Systemd
journalctl -u shelly-speedwire-gateway -f

# Standalone
python3 shelly_speedwire_gateway.py
```

### SMA Device Not Receiving Data

**Try different network modes:**
1. Enable broadcast: `SPEEDWIRE_USE_BROADCAST=true`
2. Enable dualcast: `SPEEDWIRE_DUALCAST=true`
3. Add device IP to unicast targets

### Wrong Power Flow Direction

**Check these settings in order:**
1. `SPEEDWIRE_FLIP_IMPORT_EXPORT` - Toggle if SMA shows reversed
2. Verify Shelly EM3 CT clamp orientation

### Debug Commands

```bash
# Monitor all Shelly MQTT messages
mosquitto_sub -h BROKER_IP -t "shellies/#" -v

# Check if gateway is sending packets
tcpdump -i any -n udp port 9522

# Check Docker network mode
docker inspect shelly-speedwire | grep NetworkMode
```

## Protocol Implementation

### SMA Speedwire Protocol

The gateway implements the SMA Speedwire EMETER protocol v1.0:

- **Protocol ID**: 0x6069 (EMETER)
- **Discovery ID**: 0x6081
- **Port**: UDP 9522
- **Multicast**: 239.12.255.254
- **Update Rate**: Configurable (default 1s)

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/dzikus/shelly-speedwire-gateway
cd shelly-speedwire-gateway

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run gateway
python3 shelly_speedwire_gateway.py
```

### Docker Build

```bash
# Single architecture
docker build -t shelly-speedwire-gateway .

# Multi-architecture
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64,linux/riscv64,linux/s390x,linux/386,linux/arm/v7,linux/arm/v6 \
  --tag your-registry/shelly-speedwire-gateway:latest --push .
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Guidelines

1. Test your changes with real hardware if possible
2. Update documentation for new features
3. Follow Python PEP 8 style guidelines
4. Add debug logging for troubleshooting

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Protocol implementation inspired by:
  - [venus.dbus-sma-smartmeter](https://github.com/Waldmensch1/venus.dbus-sma-smartmeter/)
  - [homeassistant-sma-sw](https://github.com/Wired-Square/homeassistant-sma-sw/)
- Thanks to the SMA and Shelly communities for protocol documentation

## Support

- **Issues**: [GitHub Issues](https://github.com/dzikus/shelly-speedwire-gateway/issues)
- **Docker Hub**: [dzikus99/shelly-speedwire-gateway](https://hub.docker.com/r/dzikus99/shelly-speedwire-gateway)

## Changelog

### v1.0.0 (2025)
- Initial release
- Full Shelly EM3 support
- SMA Speedwire EMETER protocol implementation
- Docker, systemd, and standalone deployment options
- Multicast, broadcast, and unicast transmission modes
- Automatic discovery response
- Configurable via environment variables or YAML

---

**Note**: This is an unofficial gateway not affiliated with Shelly or SMA. Use at your own risk.
