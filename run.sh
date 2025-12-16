#!/usr/bin/env bash

# Create venv if it doesn't exist (uses pyenv-selected Python)
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Show Python version should be 3.10.14
python --version

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Run app
python main.py