#!/bin/bash
# Quick test runner script

echo "Installing dependencies..."
pip install -e . > /dev/null 2>&1 || pip install -r requirements.txt

echo "Running tests..."
pytest tests/ -v "$@"
