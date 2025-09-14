.PHONY: help install install-dev run test lint format clean setup-config check-config docker-build docker-push docker-build-push

# Default target
help:
	@echo "Available commands:"
	@echo "  install         - Install production dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  run             - Run the gateway"
	@echo "  test            - Run tests"
	@echo "  lint            - Run linting (ruff, pylint, mypy)"
	@echo "  format          - Format code (ruff)"
	@echo "  build-cython-debug  - Build Cython with debug symbols"
	@echo "  build-cython-prod   - Build Cython optimized and stripped"
	@echo "  clean           - Clean up build artifacts and Cython files"
	@echo "  setup-config    - Create default configuration"
	@echo "  check-config    - Validate configuration"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-push     - Push Docker image to registry"
	@echo "  docker-build-push - Build and push Docker image"

# Installation
install:
	uv sync

install-dev:
	uv sync --dev

# Running
run:
	uv run python scripts/run_gateway.py

setup-config:
	uv run python scripts/run_gateway.py --create-config

check-config:
	uv run python scripts/run_gateway.py --check-config

# Development
test:
	uv run pytest tests/ -v --cov=shelly_speedwire_gateway

lint:
	uv run ruff check shelly_speedwire_gateway/ tests/
	uv run pylint shelly_speedwire_gateway/
	uv run mypy shelly_speedwire_gateway/

format:
	uv run ruff format shelly_speedwire_gateway/ tests/
	uv run ruff check shelly_speedwire_gateway/ tests/ --fix

# Cython build targets
build-cython-debug:
	CYTHON_DEBUG=1 uv run python setup.py build_ext --inplace
	@echo "Debug build complete with symbols and annotations"

build-cython-prod:
	CYTHON_DEBUG=0 uv run python setup.py build_ext --inplace
	@echo "Production build complete (optimized and stripped)"
	@if [ -f shelly_speedwire_gateway/*.so ]; then \
		file shelly_speedwire_gateway/*.so | grep -q "stripped" && echo "✅ Binary is properly stripped" || echo "⚠️  Binary is NOT stripped"; \
	fi

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	# Clean Cython generated files
	rm -f shelly_speedwire_gateway/*.c
	rm -f shelly_speedwire_gateway/*.html
	rm -f shelly_speedwire_gateway/*.so
	uv cache clean

# Docker commands
DOCKER_REGISTRY ?= dzikus99
DOCKER_IMAGE ?= shelly-speedwire-gateway
DOCKER_TAG ?= latest
DOCKER_FULL_NAME = $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-build:
	docker build -t $(DOCKER_FULL_NAME) .

docker-push:
	docker push $(DOCKER_FULL_NAME)

docker-build-push: docker-build docker-push

# Multi-architecture build (requires buildx)
docker-build-multiarch:
	docker buildx create --use --name multiarch-builder 2>/dev/null || true
	docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 \
		--tag $(DOCKER_FULL_NAME) \
		--push .

docker-run:
	docker run --rm --network host \
		-e MQTT_BROKER_HOST=localhost \
		-e MQTT_BASE_TOPIC=shellies/shellyem3-test \
		-e SPEEDWIRE_SERIAL=1234567890 \
		$(DOCKER_FULL_NAME)
