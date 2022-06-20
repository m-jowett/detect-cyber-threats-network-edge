#!/bin/bash

# Compare 2 best performing centralised hyperparameters in federated environment

bash ../run_simulation_nodocker.sh 9 1 5 25 3 "set2" 0 0 "fedAVG" 3 0
bash ../run_simulation_nodocker.sh 9 2 6 52 3 "set2" 0 0 "fedAVG" 3 0
