FROM python:3.13-slim

RUN useradd -m -u 1000 -s /bin/bash gateway

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt

COPY shelly_speedwire_gateway.py .

RUN printf '#!/bin/sh\n\
set -e\n\
\n\
# Check if config file already exists (e.g., mounted via volume)\n\
if [ -f /app/shelly_speedwire_gateway_config.yaml ]; then\n\
    echo "Using existing config file"\n\
else\n\
    echo "Generating config from environment variables"\n\
    cat > /app/shelly_speedwire_gateway_config.yaml << EOL\n\
# Auto-generated configuration from environment variables\n\
mqtt:\n\
  broker_host: ${MQTT_BROKER_HOST}\n\
  broker_port: ${MQTT_BROKER_PORT}\n\
  base_topic: ${MQTT_BASE_TOPIC}\n\
  keepalive: ${MQTT_KEEPALIVE}\n\
  invert_power: ${MQTT_INVERT_POWER}\n\
$([ -n "$MQTT_USERNAME" ] && echo "  username: ${MQTT_USERNAME}")\n\
$([ -n "$MQTT_PASSWORD" ] && echo "  password: ${MQTT_PASSWORD}")\n\
\n\
speedwire:\n\
  interval: ${SPEEDWIRE_INTERVAL}\n\
  use_broadcast: ${SPEEDWIRE_USE_BROADCAST}\n\
  dualcast: ${SPEEDWIRE_DUALCAST}\n\
  push_on_update: ${SPEEDWIRE_PUSH_ON_UPDATE}\n\
  min_send_interval: ${SPEEDWIRE_MIN_SEND_INTERVAL}\n\
  heartbeat_interval: ${SPEEDWIRE_HEARTBEAT_INTERVAL}\n\
  flip_import_export: ${SPEEDWIRE_FLIP_IMPORT_EXPORT}\n\
  serial: ${SPEEDWIRE_SERIAL}\n\
  susy_id: ${SPEEDWIRE_SUSY_ID}\n\
  include_voltage_current: ${SPEEDWIRE_INCLUDE_VOLTAGE_CURRENT}\n\
  include_sw_version: ${SPEEDWIRE_INCLUDE_SW_VERSION}\n\
\n\
logging:\n\
  level: ${LOG_LEVEL}\n\
EOL\n\
fi\n\
\n\
# Run the application\n\
exec python3 /app/shelly_speedwire_gateway.py\n' \
> /usr/local/bin/docker-entrypoint.sh

RUN chmod 755 /usr/local/bin/docker-entrypoint.sh && \
    chmod 644 /app/shelly_speedwire_gateway.py && \
    chown gateway:gateway /app && \
    chmod 755 /app && \
    chown root:root /app/shelly_speedwire_gateway.py

USER gateway

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV MQTT_BROKER_HOST=localhost
ENV MQTT_BROKER_PORT=1883
ENV MQTT_BASE_TOPIC=shellies/shellyem3-XXXXXXXXXXXX
ENV MQTT_KEEPALIVE=60
ENV MQTT_INVERT_POWER=true
ENV MQTT_USERNAME=""
ENV MQTT_PASSWORD=""

ENV SPEEDWIRE_INTERVAL=1.0
ENV SPEEDWIRE_USE_BROADCAST=false
ENV SPEEDWIRE_DUALCAST=false
ENV SPEEDWIRE_PUSH_ON_UPDATE=true
ENV SPEEDWIRE_MIN_SEND_INTERVAL=0.2
ENV SPEEDWIRE_HEARTBEAT_INTERVAL=10.0
ENV SPEEDWIRE_FLIP_IMPORT_EXPORT=false
ENV SPEEDWIRE_SERIAL=1234567890
ENV SPEEDWIRE_SUSY_ID=349
ENV SPEEDWIRE_INCLUDE_VOLTAGE_CURRENT=true
ENV SPEEDWIRE_INCLUDE_SW_VERSION=true

ENV LOG_LEVEL=INFO

EXPOSE 9522/udp

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD pgrep -f shelly_speedwire_gateway.py || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
