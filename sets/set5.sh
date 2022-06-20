#!/bin/bash

echo "Max Nodes, Max Rounds"

ROUNDLIST=(29 14 9 7 5 4 4 3 3 2)

for D_NODES in $(seq 1 10); do
	D_ROUNDS=${ROUNDLIST[$((D_NODES - 1))]}
	echo "$D_NODES $D_ROUNDS"

	bash ../run_simulation_nodocker.sh $D_NODES 2 6 52 $D_ROUNDS "set5" 0 0 "fedAVG" 3 0
done

