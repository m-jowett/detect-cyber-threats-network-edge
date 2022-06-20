#!/bin/bash

# Centralised: each combination of hidden size, epochs 1-3, sequence length 5-6.

HSLIST=(10 25 30 40 50 52 60)

for D_SEQLEN in $(seq 5 6); do
	for D_EPOCHS in $(seq 1 3); do
		for D_HS in "${HSLIST[@]}"; do
			echo "${D_EPOCHS} ${D_SEQLEN} ${D_HS}"
			bash ../run_simulation_nodocker.sh 1 $D_EPOCHS $D_SEQLEN $D_HS 1 "set1" 0 0 "fedAVG" 3 0
		done
	done
done
