FROM ubuntu:20.04

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


# Docker image for server container

# Install dependencies

RUN apt-get update
RUN apt-get install -y python3 python3-pip

RUN pip3 install requests sklearn numpy aiohttp
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Mount point for data

RUN mkdir /fypdata

# Copy implementation into place

RUN mkdir /fypscripts

COPY server.py /fypscripts/server.py
COPY model.py /fypscripts/model.py
COPY statistics.py /fypscripts/statistics.py

# Define environment variables

ENV NODE_UUID=""

ENV NODE_THREADLIMIT=""

ENV NODE_SEED=""

# Entrypoint into implementation

ENTRYPOINT python3 -u /fypscripts/server.py $NODE_UUID $NODE_THREADLIMIT $NODE_SEED
