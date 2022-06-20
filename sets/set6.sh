#!/bin/bash

echo "Nodes, Rounds, FedAVG"

ROUNDLIST=(5 9 3 1)
NODELIST=(2 3 6 8)

for D_COUNT in $(seq 0 3); do
	D_ROUNDS=${ROUNDLIST[$D_COUNT]}
	D_NODES=${NODELIST[$D_COUNT]}

	echo "$D_NODES $D_ROUNDS"

	bash ../run_simulation_nodocker.sh $D_NODES 2 6 52 $D_ROUNDS "set6" 0 0 "fedAVG" 3 0
done


