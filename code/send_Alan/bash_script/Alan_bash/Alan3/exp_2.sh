#!/bin/bash

    cd /home/jpierre/v2/new_runs/Alan/Alan3/mt-simplest_noisy_scaleL1-0.0001_dropout-0
    eval "$(conda shell.bash hook)"
    conda activate myenvPy
    python training_main.py
    