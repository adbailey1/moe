#! /usr/bin/env bash

FILES_PATH='/home/andrew/PycharmProjects/moe/original_run'
PYTHON_ENV='/home/andrew/miniconda3/envs/cuda11/bin/python3.8'

PTH=${FILES_PATH}/"handleIEMOCAP.py"
echo "$PTH"
echo "$PYTHON_ENV"
$PYTHON_ENV "$PTH"

PTH=${FILES_PATH}/"train_original.py"
echo "$PTH"
echo "$PYTHON_ENV"
$PYTHON_ENV "$PTH"
