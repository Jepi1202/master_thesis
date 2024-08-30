#!/bin/bash

    cd /home/jpierre/v2/new_runs/Alan/Alan3/mt-baseline_normal_dt-0.001_tr-rollout_NLayers-0_dropout-0_layerNorm-0
    eval "$(conda shell.bash hook)"
    conda activate myenvPy
    python training_main.py
    