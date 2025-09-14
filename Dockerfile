# Use Alpine for smaller image size
FROM python:3.13-alpine AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    UV_CACHE_DIR=/tmp/uv-cache

# Install uv for fast dependency management
RUN pip install uv

RUN apk add --no-cache \
        build-base \
        gcc \
        g++ \
        musl-dev \
        linux-headers

# Create virtual environment with uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy project files needed for installation
COPY pyproject.toml setup.py README.md ./
COPY shelly_speedwire_gateway/ ./shelly_speedwire_gateway/

# Install dependencies with uv (much faster than pip)
# Include monitoring dependencies for production use
RUN uv pip install -e .[monitoring] && \
    uv pip install setuptools && \
    rm -rf /tmp/uv-cache

# Build Cython extensions
RUN python setup.py build_ext --inplace

# Compile Python bytecode
RUN python -m compileall shelly_speedwire_gateway/

# Clean up Cython build artifacts but keep .so files
RUN find . -name "*.c" -delete && \
    find . -name "*.html" -delete && \
    rm -rf build/

FROM python:3.13-alpine AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONOPTIMIZE=2 \
    PYTHONGC=1

RUN apk add --no-cache \
        tini \
        procps \
        && \
    adduser -D -u 1000 -s /bin/sh gateway

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY --from=builder /app/shelly_speedwire_gateway/ ./shelly_speedwire_gateway/
COPY scripts/run_gateway.py ./scripts/run_gateway.py

RUN printf '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Function to generate config from environment\n\
generate_config() {\n\
    python3 -c "\n\
import os\n\
import yaml\n\
\n\
config = {\n\
    \"mqtt\": {\n\
        \"broker_host\": os.getenv(\"MQTT_BROKER_HOST\", \"localhost\"),\n\
        \"broker_port\": int(os.getenv(\"MQTT_BROKER_PORT\", \"1883\")),\n\
        \"base_topic\": os.getenv(\"MQTT_BASE_TOPIC\", \"shellies/shellyem3-XXXXXXXXXXXX\"),\n\
        \"keepalive\": int(os.getenv(\"MQTT_KEEPALIVE\", \"60\")),\n\
        \"invert_values\": os.getenv(\"MQTT_INVERT_VALUES\", \"false\").lower() == \"true\",\n\
        \"qos\": int(os.getenv(\"MQTT_QOS\", \"1\"))\n\
    },\n\
    \"speedwire\": {\n\
        \"interval\": float(os.getenv(\"SPEEDWIRE_INTERVAL\", \"1.0\")),\n\
        \"use_broadcast\": os.getenv(\"SPEEDWIRE_USE_BROADCAST\", \"false\").lower() == \"true\",\n\
        \"dualcast\": os.getenv(\"SPEEDWIRE_DUALCAST\", \"false\").lower() == \"true\",\n\
        \"serial\": int(os.getenv(\"SPEEDWIRE_SERIAL\", \"1234567890\")),\n\
        \"susy_id\": int(os.getenv(\"SPEEDWIRE_SUSY_ID\", \"349\")),\n\
        \"unicast_targets\": []\n\
    },\n\
    \"log_level\": os.getenv(\"LOG_LEVEL\", \"INFO\"),\n\
    \"log_format\": os.getenv(\"LOG_FORMAT\", \"structured\"),\n\
    \"enable_monitoring\": os.getenv(\"ENABLE_MONITORING\", \"false\").lower() == \"true\",\n\
    \"metrics_port\": int(os.getenv(\"METRICS_PORT\", \"8080\"))\n\
}\n\
\n\
# Add optional MQTT authentication\n\
if os.getenv(\"MQTT_USERNAME\"):\n\
    config[\"mqtt\"][\"username\"] = os.getenv(\"MQTT_USERNAME\")\n\
if os.getenv(\"MQTT_PASSWORD\"):\n\
    config[\"mqtt\"][\"password\"] = os.getenv(\"MQTT_PASSWORD\")\n\
\n\
# Add unicast targets if specified\n\
if os.getenv(\"SPEEDWIRE_UNICAST_TARGETS\"):\n\
    targets = os.getenv(\"SPEEDWIRE_UNICAST_TARGETS\").split(\",\")\n\
    config[\"speedwire\"][\"unicast_targets\"] = [t.strip() for t in targets if t.strip()]\n\
\n\
# Write YAML config\n\
with open(\"/app/shelly_speedwire_gateway_config.yaml\", \"w\", encoding=\"utf-8\") as f:\n\
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n\
"\n\
}\n\
\n\
# Main execution\n\
main() {\n\
    echo "Starting Shelly 3EM to SMA Speedwire Gateway v2.0.0"\n\
    \n\
    # Check if config file exists (mounted via volume)\n\
    if [ -f "/app/shelly_speedwire_gateway_config.yaml" ] && [ "${USE_ENV_CONFIG:-false}" != "true" ]; then\n\
        echo "Using existing configuration file"\n\
    else\n\
        echo "Generating configuration from environment variables"\n\
        generate_config\n\
    fi\n\
    \n\
    # Validate required environment variables\n\
    if [ "${MQTT_BASE_TOPIC:-}" = "shellies/shellyem3-XXXXXXXXXXXX" ]; then\n\
        echo "WARNING: Using default MQTT base topic. Please set MQTT_BASE_TOPIC environment variable."\n\
    fi\n\
    \n\
    if [ "${SPEEDWIRE_SERIAL:-1234567890}" = "1234567890" ]; then\n\
        echo "WARNING: Using default Speedwire serial number. Please set SPEEDWIRE_SERIAL to a unique value."\n\
    fi\n\
    \n\
    # Start the gateway\n\
    echo "Configuration loaded, starting gateway..."\n\
    exec python /app/scripts/run_gateway.py "$@"\n\
}\n\
\n\
# Handle signals properly\n\
trap "echo \"Received signal, shutting down...\"; exit 0" TERM INT\n\
\n\
# Run main function\n\
main "$@"\n' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

RUN chown -R gateway:gateway /app /opt/venv

USER gateway

ENV MQTT_BROKER_HOST=localhost \
    MQTT_BROKER_PORT=1883 \
    MQTT_BASE_TOPIC=shellies/shellyem3-XXXXXXXXXXXX \
    MQTT_KEEPALIVE=60 \
    MQTT_INVERT_VALUES=false \
    MQTT_QOS=1 \
    SPEEDWIRE_INTERVAL=1.0 \
    SPEEDWIRE_USE_BROADCAST=false \
    SPEEDWIRE_DUALCAST=false \
    SPEEDWIRE_SERIAL=1234567890 \
    SPEEDWIRE_SUSY_ID=349 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=structured \
    ENABLE_MONITORING=false \
    METRICS_PORT=8080

EXPOSE 9522/udp 8080/tcp

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD pgrep -f "run_gateway.py" > /dev/null || exit 1

LABEL maintainer="Grzegorz Sterniczuk <grzegorz@sternicz.uk>" \
      version="2.0.0" \
      description="Shelly 3EM to SMA Speedwire Gateway" \
      org.opencontainers.image.title="Shelly Speedwire Gateway" \
      org.opencontainers.image.description="Gateway between Shelly 3EM energy meters and SMA Speedwire protocol" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="Grzegorz Sterniczuk" \
      org.opencontainers.image.source="https://github.com/dzikus/shelly-speedwire-gateway" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD []
