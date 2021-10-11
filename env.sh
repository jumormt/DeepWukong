#!/bin/bash

# 1. torch-geometric has dependency to torch-scatter without mentioning it in setup.py :(
# 2. torch-scatter's setup.py import torch, so it have to be installed already :(
# 3. ad-hoc solution: install first line of requirements.txt that is hardcoded to be torch :crazy:

pip install "$(head -n 1 requirements.txt)"
pip install -r requirements.txt