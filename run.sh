#! /usr/bin/env bash

FILES_PATH='/path/to/code'
PYTHON_ENV='/path/to/python3.8'

PTH=${FILES_PATH}/"train.py"

$PYTHON_ENV "$PTH" --debug
