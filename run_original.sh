#! /usr/bin/env bash

FILES_PATH='/path/to/code'
PYTHON_ENV='/path/to/python3.8'

PTH=${FILES_PATH}/"handleIEMOCAP.py"
echo "$PTH"
echo "$PYTHON_ENV"
$PYTHON_ENV "$PTH"

PTH=${FILES_PATH}/"train_original.py"
echo "$PTH"
echo "$PYTHON_ENV"
$PYTHON_ENV "$PTH"
