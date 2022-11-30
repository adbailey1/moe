#! /usr/bin/env bash

FILES_PATH='/home/andrew/PycharmProjects/MoE_Multi'
PYTHON_ENV='/home/andrew/miniconda3/envs/cuda11/bin/python3.8'

$PYTHON_ENV ${FILES_PATH}/"utilities/organise_dataset.py"

$PYTHON_ENV PTH=${FILES_PATH}/"train.py"
