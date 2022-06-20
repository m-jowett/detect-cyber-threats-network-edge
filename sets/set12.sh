#!/bin/bash

echo "Nodes, Rounds, FedAVG, Pareto Distribution, Benign"

ROUNDLIST=(1 3 6 5)
NODELIST=(19 6 4 2)

for D_COUNT in $(seq 0 3); do
	D_ROUNDS=${ROUNDLIST[$D_COUNT]}
	D_NODES=${NODELIST[$D_COUNT]}

	echo "$D_NODES $D_ROUNDS"

	bash ../run_simulation_nodocker.sh $D_NODES 2 6 52 $D_ROUNDS "set12" 0 1 "fedAVG" 3 1
done

