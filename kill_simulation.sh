#!/bin/bash

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


# If a simulation script was terminated pre-maturely this can be used to kill the server/client containers

# Arguments
NUMNODES=$1

echo "Dumping logs to console"

docker logs "fyp-server-0"

for UUID in $(seq 1 $NUMNODES); do
	docker logs "fyp-client-${UUID}"
done


echo "Stopping and removing containers"

docker stop -t 1 "fyp-server-0"
for UUID in $(seq 1 $NUMNODES); do
	docker stop -t 1 "fyp-client-${UUID}"
done

docker rm "fyp-server-0" -v
for UUID in $(seq 1 $NUMNODES); do
	docker rm "fyp-client-${UUID}" -v
done
