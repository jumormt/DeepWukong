#!/bin/bash

# Install PyTorch first (torch-geometric depends on it)
pip install "$(head -n 1 requirements.txt)"

# Install torch-geometric and its dependencies
pip install torch-geometric

# Install remaining requirements
pip install -r requirements.txt
