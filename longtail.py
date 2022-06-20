#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


import numpy as np
from pathlib import Path

# Provide functionality to the data generation script for long tail distribution between nodes.

# see development/longtail.ipynb for selecting co-efficient and visualisation of distribution

# load in configuration
from json import load as json_load
configData = {}
with open(Path('config.json'), 'r') as f:
	configData = json_load(f)

# load parameters of distribution from config

maxNodes = configData['longtail']['maxNodes']
lowerBound = configData['longtail']['lowerBound']
coeff = configData['longtail']['coefficient']

def pareto(x):
	return (lowerBound/x)**coeff

def paretoNodes(numNodes):
	# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

	# place nodes at equally spaced intervals through distribution

	x = np.linspace(
		lowerBound,
		maxNodes+lowerBound,
		numNodes,
	)

	x = pareto(x)

	return x

def paretoNodesRatio(numNodes):
	# normalise the sizes from the distribution to produce a ratio

	x = paretoNodes(numNodes)
	t = sum(x)
	x = x/t
	return x
