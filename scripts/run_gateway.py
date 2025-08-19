#!/usr/bin/env python3
"""Launcher script for Shelly 3EM to SMA Speedwire Gateway.

Script for running the gateway from the project root directory
without requiring package installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import and run main
    from shelly_speedwire_gateway.main import cli_main

    if __name__ == "__main__":
        cli_main()

except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print("\nRun this script from the project root directory.")
    print("Expected project structure:")
    print("  project_root/")
    print("  ├── shelly_speedwire_gateway/")
    print("  │   ├── __init__.py")
    print("  │   ├── main.py")
    print("  │   └── ...")
    print("  └── scripts/")
    print("      └── run_gateway.py")
    sys.exit(1)
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
