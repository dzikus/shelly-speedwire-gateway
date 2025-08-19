# Shelly 3EM to SMA Speedwire Gateway

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/dzikus/shelly-speedwire-gateway)
[![Docker Pulls](https://img.shields.io/docker/pulls/dzikus99/shelly-speedwire-gateway)](https://hub.docker.com/r/dzikus99/shelly-speedwire-gateway)
[![Docker Image Size](https://img.shields.io/docker/image-size/dzikus99/shelly-speedwire-gateway/latest)](https://hub.docker.com/r/dzikus99/shelly-speedwire-gateway)

A Python gateway that enables Shelly 3EM three-phase energy meters to communicate with SMA inverters by translating MQTT data to the SMA Speedwire protocol.

## Features

- Energy monitoring - forwards power, voltage, current, and energy data from Shelly 3EM to SMA devices
- Three-phase support - three-phase measurements with individual phase data
- SMA Speedwire protocol implementation - EMETER protocol v1.0 with reactive and apparent power calculations
- CT clamp orientation support - handles backwards-mounted current transformers with the invert_values option
- Deployment options - run as standalone Python script, systemd service, or Docker container
- Networking - supports multicast, broadcast, and direct unicast communication
- Discovery - responds to SMA device discovery requests
- Configurable transmission intervals with immediate updates on data changes
- **High-performance optimizations** - Cython compilation, memory pooling, LRU caching for minimal resource usage, Batch MQTT processing

## Requirements

### Hardware
- **Shelly 3EM** energy meter connected to your electrical installation
- **SMA inverter** or device that supports the Speedwire protocol
- Network connectivity between Shelly 3EM, MQTT broker, and SMA devices

### Software
- Python 3.13+ (for standalone installation)
- Docker (for containerized deployment)
- MQTT broker (Mosquitto)
- UV package manager

## Getting Started

### Docker

```yaml
version: '3.8'

services:
  shelly-speedwire-gateway:
    image: dzikus99/shelly-speedwire-gateway:latest
    container_name: shelly-speedwire
    restart: unless-stopped
    network_mode: host  # Required for multicast/broadcast
    environment:
      - MQTT_BROKER_HOST=192.168.1.123
      - MQTT_BASE_TOPIC=shellies/shelly3em-XXXXXXXXXXXXX
      - SPEEDWIRE_SERIAL=1234567890
      - MQTT_INVERT_VALUES=false  # Set to true if CT clamps are backwards
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
  -e MQTT_BASE_TOPIC=shellies/shelly3em-XXXXXXXXXXXXX \
  -e SPEEDWIRE_SERIAL=1234567890 \
  -e LOG_LEVEL=INFO \
  -e MQTT_MAX_RECONNECT_ATTEMPTS=5 \
  -e SPEEDWIRE_MIN_SEND_INTERVAL=0.5 \
  -e GATEWAY_BATCH_SIZE=100 \
  dzikus99/shelly-speedwire-gateway:latest
```

## Installation Options

### Option 1: Docker Container

Pull the image:
```bash
docker pull dzikus99/shelly-speedwire-gateway:latest
```

Or build locally:
```bash
docker build -t shelly-speedwire-gateway .
```

### Option 2: Systemd Service (Linux)

1. Clone the repository:
```bash
git clone https://github.com/dzikus/shelly-speedwire-gateway
cd shelly-speedwire-gateway
```

2. Install dependencies:
```bash
# Using UV (recommended)
pip install uv
uv sync

# Or using pip
pip install aiomqtt pydantic pydantic-settings structlog pyyaml uvloop psutil
```

3. Install the package:
```bash
sudo mkdir -p /opt/shelly-speedwire-gateway
sudo cp -r shelly_speedwire_gateway/ /opt/shelly-speedwire-gateway/
sudo cp scripts/run_gateway.py /opt/shelly-speedwire-gateway/
sudo cp shelly_speedwire_gateway_config.yaml /opt/shelly-speedwire-gateway/
```

4. Install systemd service:
```bash
sudo cp shelly-speedwire-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable shelly-speedwire-gateway
sudo systemctl start shelly-speedwire-gateway
```

5. Check status:
```bash
sudo systemctl status shelly-speedwire-gateway
sudo journalctl -u shelly-speedwire-gateway -f
```

### Option 3: Standalone Python

1. Install dependencies:
```bash
# Using UV (recommended)
pip install uv
uv sync

# Or using pip
pip install aiomqtt pydantic pydantic-settings structlog pyyaml uvloop psutil
```

2. Configure the gateway (see Configuration section)

3. Run the gateway:
```bash
python3 scripts/run_gateway.py
```

## Configuration

The gateway supports two configuration methods: YAML files or environment variables (Docker).

### Configuration File

Create or edit `shelly_speedwire_gateway_config.yaml`:

```yaml
mqtt:
  broker_host: 192.168.1.123
  broker_port: 1883
  base_topic: shellies/shelly3em-XXXXXXXXXXXXX
  keepalive: 60
  invert_values: false  # Set to true if CT clamps are mounted backwards
  # username: mqtt_user
  # password: mqtt_pass

speedwire:
  interval: 1.0                       # Send interval in seconds
  use_broadcast: false                # Use broadcast instead of multicast
  dualcast: false                     # Send both multicast and broadcast
  include_voltage_current: true       # Include V, I, PF, Hz data
  serial: 1234567890                  # Unique serial number
  susy_id: 349                        # SUSy ID

  # Optional: Direct unicast to specific devices
  # unicast_targets:
  #   - 192.168.1.50
  #   - 192.168.1.51

log_level: INFO
log_format: structured
enable_monitoring: false
metrics_port: 8080
```

### Environment Variables (Docker)

#### MQTT Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `MQTT_BROKER_HOST` | `localhost` | MQTT broker IP address or hostname | No |
| `MQTT_BROKER_PORT` | `1883` | MQTT broker port | No |
| `MQTT_BASE_TOPIC` | `shellies/shelly3em-XXXXXXXXXXXXX` | Your Shelly 3EM MQTT topic | Yes |
| `MQTT_KEEPALIVE` | `60` | MQTT connection keepalive interval in seconds | No |
| `MQTT_INVERT_VALUES` | `false` | Set to `true` if CT clamps are mounted backwards | No |
| `MQTT_USERNAME` | - | MQTT broker username for authentication | No |
| `MQTT_PASSWORD` | - | MQTT broker password for authentication | No |
| `MQTT_QOS` | `1` | MQTT Quality of Service level (0, 1, or 2) | No |
| `MQTT_MAX_RECONNECT_ATTEMPTS` | `3` | Maximum number of MQTT reconnection attempts | No |
| `MQTT_CONNECTION_TIMEOUT` | `10.0` | MQTT connection timeout in seconds | No |
| `MQTT_MESSAGE_TIMEOUT` | `300.0` | Maximum time to wait for MQTT messages in seconds | No |

#### Speedwire Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `SPEEDWIRE_INTERVAL` | `1.0` | Data packet transmission interval in seconds | No |
| `SPEEDWIRE_USE_BROADCAST` | `false` | Use UDP broadcast instead of multicast | No |
| `SPEEDWIRE_DUALCAST` | `false` | Send both multicast and broadcast packets | No |
| `SPEEDWIRE_SERIAL` | `1234567890` | Unique serial number for the emulated energy meter | Yes |
| `SPEEDWIRE_SUSY_ID` | `349` | SUSy ID for device identification | No |
| `SPEEDWIRE_MIN_SEND_INTERVAL` | `0.1` | Minimum time between packet transmissions in seconds | No |
| `SPEEDWIRE_HEARTBEAT_INTERVAL` | `30.0` | Heartbeat interval for keep-alive packets in seconds | No |
| `SPEEDWIRE_DISCOVERY_LOOP_SLEEP` | `0.01` | Sleep interval in discovery loop in seconds | No |
| `SPEEDWIRE_DATA_RECEIVE_TIMEOUT` | `0.05` | Timeout for data reception in seconds | No |
| `SPEEDWIRE_HEALTH_CHECK_TIMEOUT` | `300` | Health check timeout in seconds | No |
| `SPEEDWIRE_HEALTH_CHECK_INTERVAL` | `60.0` | Health check interval in seconds | No |
| `SPEEDWIRE_RECEIVE_BUFFER_SIZE` | `2048` | UDP receive buffer size in bytes | No |
| `SPEEDWIRE_MAX_PACKET_SIZE` | `1500` | Maximum packet size in bytes | No |
| `SPEEDWIRE_MIN_PACKET_SIZE` | `64` | Minimum packet size in bytes | No |
| `SPEEDWIRE_MAX_RETRIES` | `5` | Maximum number of retry attempts | No |
| `SPEEDWIRE_BACKOFF_FACTOR` | `0.3` | Exponential backoff factor for retries | No |
| `SPEEDWIRE_CONNECTION_POOL_SIZE` | `10` | Connection pool size for network connections | No |

#### Gateway Performance Settings

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `GATEWAY_LRU_CACHE_SIZE` | `128` | LRU cache size for performance optimization | No |
| `GATEWAY_BATCH_SIZE` | `50` | Batch processing size for MQTT messages | No |
| `GATEWAY_BATCH_FLUSH_INTERVAL` | `0.05` | Batch flush interval in seconds | No |
| `GATEWAY_MAX_QUEUE_SIZE` | `10000` | Maximum queue size for message processing | No |

#### Gateway Physical Limits

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `GATEWAY_MAX_VOLTAGE` | `500.0` | Maximum voltage limit in volts | No |
| `GATEWAY_MAX_CURRENT` | `100.0` | Maximum current limit in amperes | No |
| `GATEWAY_MAX_POWER` | `50000.0` | Maximum power limit in watts | No |
| `GATEWAY_MIN_FREQUENCY` | `45.0` | Minimum grid frequency in Hz | No |
| `GATEWAY_MAX_FREQUENCY` | `65.0` | Maximum grid frequency in Hz | No |
| `GATEWAY_DEFAULT_FREQUENCY` | `50.0` | Default grid frequency in Hz | No |
| `GATEWAY_MIN_POWER_FACTOR_THRESHOLD` | `0.01` | Minimum power factor threshold | No |

#### Gateway Device Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `GATEWAY_DEFAULT_SW_VERSION` | `2.3.4.R` | Default software version to emulate | No |
| `GATEWAY_MQTT_LOG_LEVEL` | `INFO` | MQTT-specific logging level | No |

## Network Configuration

### Requirements

- **UDP Port 9522** - required for Speedwire communication
- **Multicast** - address 239.12.255.254 (default mode)
- **Broadcast** - address 255.255.255.255 (alternative mode)
- **Network mode** - Docker containers use `network_mode: host`

### Network Modes

1. **Multicast (Default)** - requires multicast routing
2. **Broadcast** - works in all configurations, higher network load
3. **Dualcast** - sends both multicast and broadcast for better compatibility
4. **Unicast** - direct communication to specific devices

## CT Clamp Orientation

### Important: Current Transformer Direction

The Shelly 3EM uses CT clamps to measure current flow. If these are installed backwards, the power readings will be inverted (consumption shows as generation and vice versa).

### Using the invert_values Parameter

If your CT clamps are mounted backwards, set `invert_values: true`:

```yaml
mqtt:
  invert_values: true
```

Or with Docker:
```bash
-e MQTT_INVERT_VALUES=true
```

The invert_values option:
- Reverses the sign of all power values
- Swaps total and total_returned energy counters
- Corrects power factor signs
- Does not affect voltage, current, or frequency values

## Data Mapping

The gateway translates Shelly 3EM MQTT data to SMA Speedwire protocol using OBIS codes:

| Shelly 3EM Data | SMA Speedwire Field | OBIS Code | Description |
|-----------------|---------------------|-----------|-------------|
| Total Power | Total Active Power | 1.4.0 / 2.4.0 | Positive = Import, Negative = Export |
| Phase A/B/C Power | L1/L2/L3 Active Power | 21/41/61.4.0 | Per-phase instantaneous power |
| Calculated Total | Total Reactive Power | 3.4.0 / 4.4.0 | Calculated from V, I, PF |
| Phase A/B/C Voltage | L1/L2/L3 Voltage | 32/52/72.4.0 | Phase voltages |
| Phase A/B/C Current | L1/L2/L3 Current | 31/51/71.4.0 | Phase currents |
| Phase A/B/C PF | L1/L2/L3 Power Factor | 33/53/73.4.0 | Per-phase power factors |
| Grid Frequency | Frequency | 14.4.0 | Grid frequency |
| Energy Consumed/Exported | Import/Export Energy | Various | Cumulative energy counters |

### Power Flow Direction

- **Positive power** = Energy consumption from grid
- **Negative power** = Energy export to grid (solar feed-in)

## Finding Your Shelly 3EM Topic

### Method 1: MQTT Explorer
1. Connect to your MQTT broker using MQTT Explorer
2. Look for topics starting with `shellies/shelly3em-`
3. The device ID is the part after `shelly3em-`

### Method 2: Command Line
```bash
mosquitto_sub -h YOUR_BROKER_IP -t "shellies/#" -v
```

### Method 3: Shelly Web Interface
1. Access your Shelly 3EM web interface
2. Go to Settings → Device Info
3. The device ID is shown there

## Troubleshooting

### Gateway Not Starting

Check MQTT connectivity:
```bash
ping YOUR_BROKER_IP
mosquitto_sub -h YOUR_BROKER_IP -t "shellies/shelly3em-XXXXXXXXXXXXX/#" -v
```

Enable debug logging:
```bash
# Docker
docker run -e LOG_LEVEL=DEBUG dzikus99/shelly-speedwire-gateway:latest

# Systemd
# Edit config file, set log_level to DEBUG, then restart
sudo systemctl restart shelly-speedwire-gateway
```

Check logs:
```bash
# Docker
docker logs shelly-speedwire

# Systemd
journalctl -u shelly-speedwire-gateway -f
```

### SMA Device Not Receiving Data

Try different network modes:
1. Enable broadcast: `SPEEDWIRE_USE_BROADCAST=true`
2. Enable dualcast: `SPEEDWIRE_DUALCAST=true`
3. Add device IP to unicast targets

### Wrong Power Flow Direction

1. Check CT clamp orientation - verify installation direction
2. Enable `invert_values` if CT clamps are backwards
3. Set `invert_values: true` in configuration

### Debug Commands

```bash
# Monitor Shelly MQTT messages
mosquitto_sub -h BROKER_IP -t "shellies/#" -v

# Check network packets
tcpdump -i any -n udp port 9522

# Check Docker network mode
docker inspect shelly-speedwire | grep NetworkMode

# Monitor specific power values
mosquitto_sub -h BROKER_IP -t "shellies/shelly3em-XXXXXXXXXXXXX/emeter/+/power" -v
```

## Protocol Implementation

### SMA Speedwire Protocol

SMA Speedwire EMETER protocol v1.0:

- **Protocol ID**: 0x6069 (EMETER)
- **Discovery ID**: 0x6081
- **Port**: UDP 9522
- **Multicast**: 239.12.255.254
- **Update Rate**: Configurable (default 1s)
- **Reactive/apparent power support**
- **Per-phase power factor with negative value support**

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/dzikus/shelly-speedwire-gateway
cd shelly-speedwire-gateway

# Install UV
pip install uv

# Install dependencies and build Cython extensions
uv sync
uv run python setup.py build_ext --inplace

# Run gateway
python3 scripts/run_gateway.py
```

### Docker Build

```bash
# Single architecture
docker build -t shelly-speedwire-gateway .

# Multi-architecture
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64,linux/riscv64,linux/s390x,linux/386,linux/arm/v7 \
  --tag your-registry/shelly-speedwire-gateway:latest --push .
```

## Contributing

Contributing:

1. Test changes with real hardware when available
2. Update documentation for new features
3. Use Python PEP 8 style guidelines
4. Add debug logging for troubleshooting

## Changelog

### v2.0.0 (2025)
- Added `invert_values` parameter for backwards CT clamp installations
- Reactive power (VAr) calculation and transmission
- Apparent power (VA) calculation and transmission
- Per-phase power factor support with negative values
- Complete SMA EMETER protocol implementation (all OBIS channels)
- Software version emulation updated to 2.3.4.R
- Removed `flip_import_export` parameter (replaced by `invert_values`)
- **Performance optimizations**: Cython compilation, memory pooling, LRU caching, Batch processing
### v1.0.0 (2025)
- Initial release
- Full Shelly 3EM support
- SMA Speedwire EMETER protocol implementation
- Docker, systemd, and standalone deployment options
- Multicast, broadcast, and unicast transmission modes
- Automatic discovery response
- Configurable via environment variables or YAML

## Acknowledgments

Protocol implementation inspired by:
- [venus.dbus-sma-smartmeter](https://github.com/Waldmensch1/venus.dbus-sma-smartmeter/)
- [homeassistant-sma-sw](https://github.com/Wired-Square/homeassistant-sma-sw/)
- [SMA-EM speedwire decoder](https://github.com/datenschuft/SMA-EM/)

Protocol documentation from SMA and Shelly communities.

## Support

- **Issues**: [GitHub Issues](https://github.com/dzikus/shelly-speedwire-gateway/issues)
- **Docker Hub**: [dzikus99/shelly-speedwire-gateway](https://hub.docker.com/r/dzikus99/shelly-speedwire-gateway)

---

This gateway is not affiliated with Shelly or SMA. Use at your own discretion.
