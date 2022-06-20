#!/bin/bash

echo "Max Nodes, 1 Round"

for D_NODES in $(seq 1 29); do
	echo "$D_NODES 1"
	bash ../run_simulation_nodocker.sh $D_NODES 2 6 52 1 "set4" 0 0 "fedAVG" 3 0
done

