#!/bin/bash

echo "1 Node, Max Rounds"

for D_ROUNDS in $(seq 1 29); do
	echo "1 $D_ROUNDS"
	bash ../run_simulation_nodocker.sh 1 2 6 52 "$D_ROUNDS" "set3" 0 0 "fedAVG" 3 0
done
