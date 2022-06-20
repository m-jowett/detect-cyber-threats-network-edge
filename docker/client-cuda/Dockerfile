FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


# Docker image for client containers, CUDA support

# Install dependencies

RUN apt-get update
RUN apt-get install -y python3 python3-pip

RUN pip3 install requests sklearn numpy aiohttp
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113

#RUN pip3 install numba

# Mount point for data

RUN mkdir /fypdata

# Copy implementation into place

RUN mkdir /fypscripts

COPY client.py /fypscripts/client.py
COPY model.py /fypscripts/model.py
COPY statistics.py /fypscripts/statistics.py

# Define environment variables

ENV NODE_UUID=""

# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

# tcmalloc and GNU OpenMP disabled for CUDA

#RUN apt-get install -y libtcmalloc-minimal4
#
#ENV OMP_NUM_THREADS=12
#ENV OMP_SCHEDULE=STATIC
#ENV OMP_PROC_BIND=CLOSE
#ENV GOMP_CPU_AFFINITY="0-11"
#
#ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD

# CUDA determinism
# script will error out otherwise due to PyTorch flag set
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

ENV NODE_OFFLOAD=""

ENV NODE_SEED=""

# Entrypoint into implementation

ENTRYPOINT python3 -u /fypscripts/client.py $NODE_UUID "cuda" $NODE_OFFLOAD $NODE_SEED
