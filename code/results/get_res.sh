#!/bin/bash

    cd /master/code/results/1-step/normal_0-01_v
    ls
    python stats_gen_steps.py

    cd /master/code/results/1-step/normal_0-01_delta
    python stats_gen_steps.py

    cd /master/code/results/1-step/normal_0-001_v
    python stats_gen_steps.py

    cd /master/code/results/1-step/normal_0-001_delta
    python stats_gen_steps.py


    cd /master/code/results/1-step/noisy_0-01_v
    python stats_gen_steps.py

    cd /master/code/results/1-step/noisy_0-01_delta
    python stats_gen_steps.py

    cd /master/code/results/1-step/noisy_0-001_v
    python stats_gen_steps.py

    cd /master/code/results/1-step/noisy_0-001_delta
    python stats_gen_steps.py

    