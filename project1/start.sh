#!/bin/bash

# Creating .venv
python3 -m venv .venv
source .venv/bin/activate

# Installing dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Installed Dependencies"