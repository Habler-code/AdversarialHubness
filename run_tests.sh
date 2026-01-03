#!/bin/bash
# Quick test runner script

set -euo pipefail

echo "Installing dependencies..."
python3 -m pip install -e . > /dev/null 2>&1 || python3 -m pip install -r requirements.txt -r requirements-dev.txt

echo "Running tests..."
pytest tests/ -v "$@"
