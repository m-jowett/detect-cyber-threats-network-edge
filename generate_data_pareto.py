#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from math import floor, ceil
import random
import torch
from sys import argv
from longtail import paretoNodesRatio
from json import load as json_load
from pathlib import Path

# Script to prepare data to be passed to each node for training and validation.

# This is a modified copy of generated_data_main.py for Pareto distribution.
# Although they could have been merged into one it would have been inefficient and hard to read

# load in configuration
configData = {}
with open(Path('config.json'), 'r') as f:
	configData = json_load(f)

# command line arguments
# these are usually passed by the simulation script: run_simulation.sh

numNodes       = int(argv[1])
numRounds      = int(argv[2])
sequenceLength = int(argv[3])

# TODO
splitAttack = bool(int(argv[4]))
splitBenign = bool(int(argv[5]))
###

#if (numNodes * numRounds) > 29:
#	print('ERROR: nr outside normal bounds')
#	exit()

# load other parameters from config.json where applicable

seed = configData['default']['seed']
splitTrain = configData['data']['splitTrain']
splitTest  = configData['data']['splitTest']

nodeStore = Path(configData['default']['dataPath'])
dataPath = Path(configData['default']['featurePath'])

dataSets = configData['data']['datasets']

selectedFeatures = configData['data']['features']

#####################################

# https://numpy.org/doc/stable/reference/
# https://scikit-learn.org/stable/modules/classes.html
# https://pytorch.org/docs/stable/index.html
# https://docs.python.org/3/library/random.html

# used to normalise features
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

scaler = MinMaxScaler(
	feature_range = (-1,1),
	copy = True,
	clip = False,
)

# see reproducibility chapter of report

# https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html

np.random.seed(seed)
random.seed(seed)

rng = np.random.default_rng(seed=seed)

numDatasets = len(dataSets)
inputSize = len(selectedFeatures)

####################################
print('Loading datasets')

# dataX
#	datasets

# dataY
#	datasets

# dataL
#	datasets

dataX = []
dataY = []
dataL = []

for fileName, sentiment in dataSets:
	#print(fileName)

	fn = Path(dataPath, fileName)
	fn = fn.with_suffix('.pkl')

	data = pd.read_pickle(
		filepath_or_buffer = fn,
	)

	# drop unwanted features if applicable
	# selected features to keep in config.json

	data.drop(
		labels = data.columns.difference(selectedFeatures),
		axis = 1,
		inplace = True,
	)

	# TCP/UDP only, remove empty packets etc.
	data = data[data['protocol'].isin([6, 17])]

	# dataframe to numpy ndarray
	# the order of the columns/selected features matter if you want to inspect the data
	data = data.to_numpy(dtype=np.float32)

	dataX.append(data)
	dataY.append(sentiment)
	dataL.append(fileName)

print('Sentiment to float32')

# [dataset sentiment, ...]
dataY = np.asarray(
	a = dataY,
	dtype = np.float32,
)


########################################
print('Taking test split from each dataset at interval')

# testDataX
# 	datasets

# round up validation split since there is an abundance of data available

testDataX = []
for d in range(numDatasets):
	testDataX.append([])

for d in range(numDatasets):
	interval = floor(1/splitTest)
	testSize = floor((1/interval) * len(dataX[d]))
	trainSize = len(dataX[d]) - testSize
	dropped  = ceil(splitTest * len(dataX[d])) - testSize

	if dropped > 0:
		print('WARN: {0} omitted from test set of {1}'.format(dropped, dataL[d]))
	
	if dropped < 0:
		print('WARN: {0} extra to test set of {1}'.format(dropped*-1, dataL[d]))

	if trainSize < sequenceLength:
		print('ERROR: train size less than seq length {0} {1}'.format(trainSize, dataL[d]))
		exit()
	
	if testSize < sequenceLength:
		print('ERROR: test size less than seq length {0} {1}'.format(testSize, dataL[d]))
		exit()
	
	testIndexes = np.arange(0, len(dataX[d]), interval)
	testDataX[d] = dataX[d][testIndexes]

	dataX[d] = np.delete(
		arr=dataX[d],
		obj=testIndexes,
		axis=0,
	)

##########################################
print('Generate Pareto ratio')

nodesRatio = paretoNodesRatio(numNodes)

##########################################
print('Determine split size for each node')

# sizeSplit
# node, datasets sizes

sizesSplit = []
for n in range(numNodes):
	sizesSplit.append([])
	for d in range(numDatasets):
		sizesSplit[n].append([])

for d in range(numDatasets):

	totalSplit = 0

	for n in range(numNodes):
		splitSize = floor(len(dataX[d]) * nodesRatio[n])

		sizesSplit[n][d] = splitSize

		totalSplit += splitSize
	
	dropped = len(dataX[d]) - totalSplit

	if dropped > 0:
		print('WARN: {0} total dropped rows {1}'.format(dataL[d], dropped))


##########################################
print('Split datasets between each node')

nodesDataX = []
for n in range(numNodes):
	nodesDataX.append([])
	for d in range(numDatasets):
		nodesDataX[n].append([])

for d in range(numDatasets):

	pos = 0
	for n in random.sample(range(numNodes), numNodes):
		nodesDataX[n][d] = dataX[d][pos:pos+sizesSplit[n][d]]
		pos += sizesSplit[n][d]


#############################################
# split each nodes datasets data into training rounds

# nodesDataX
# 	nodes datasets

# nodesDataX
#	nodes datasets rounds

print('Split each nodes datasets data into training rounds')

for d in range(numDatasets):
	for n in range(numNodes):

		splitSize = floor(sizesSplit[n][d] / numRounds)

		new = []

		if splitSize < sequenceLength:
			print('SUPPRESSED ERROR: split size less than sequence length {0} {1} {2}'.format(sizesSplit[n][d], dataL[d], n))
			nodesDataX[n][d] = new
			continue
		
		dropped = sizesSplit[n][d] - (splitSize*numRounds)
		if dropped > 0:
			print('WARN: dropped {0} from nodes data of {1} {2}'.format(dropped, dataL[d], n))
		
		pos = 0
		for _ in range(numRounds):
			new.append(nodesDataX[n][d][pos:pos+splitSize])
			pos += splitSize
		
		nodesDataX[n][d] = new

################################################

# nodesDataX
#	nodes datasets rounds

# testDataX
#	datasets rounds

# https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

print('Sliding window')

for d in range(numDatasets):
	for n in range(numNodes):
		
		if len(nodesDataX[n][d]) == 0:
			print('SUPPRESSED ERROR: Skipping {0} {1}'.format(n, dataL[d]))
			continue

		for r in range(numRounds):

			nodesDataX[n][d][r] = np.lib.stride_tricks.sliding_window_view(
				x=nodesDataX[n][d][r],
				window_shape=(sequenceLength, inputSize)
			)

			# squash the axis, use batch first
			nodesDataX[n][d][r] = nodesDataX[n][d][r].squeeze(axis=1)

for d in range(numDatasets):
	testDataX[d] = np.lib.stride_tricks.sliding_window_view(
		x=testDataX[d],
		window_shape=(sequenceLength, inputSize)
	)

	testDataX[d] = testDataX[d].squeeze(axis=1)

#################################################
print('Generate train stats for each node-round')

# store the dataset sizes for each round, for testing purposes

# nodeTrainStats
# nodes rounds datasetsizes

nodeTrainStats = []
for n in range(numNodes):
	nodeTrainStats.append([])
	for r in range(numRounds):
		nodeTrainStats[n].append([])
		for d in range(numDatasets):
			nodeTrainStats[n][r].append(0)

for d in range(numDatasets):
	for n in range(numNodes):
		for r in range(numRounds):
			if len(nodesDataX[n][d]) > 0:
				nodeTrainStats[n][r][d] = len(nodesDataX[n][d][r])

##################################################

# nodesDataX
#	nodes datasets rounds -> nodes rounds

# nodesDataY
#	nodes rounds

print('Merge round train data, dataset test data. Generate train sentiment sets.')

# no longer need to distinguish between the different datasets making up training data
# squash them

nodesDataY = []
new = []

for n in range(numNodes):
	nodesDataY.append([])
	new.append([])
	for r in range(numRounds):
		nodesDataY[n].append([])
		new[n].append([])


# nodes rounds datasets
# nodes rounds

for n in range(numNodes):
	for r in range(numRounds):

		ySizes = []
		
		for d in range(numDatasets):
			if len(nodesDataX[n][d]) > 0:
				new[n][r].append(nodesDataX[n][d][r])
				ySizes.append(len(nodesDataX[n][d][r]))
			else:
				print('SUPPRESSED ERROR: Skipping {0} {1} {2}'.format(n, dataL[d], r))
				ySizes.append(0)

		nodesDataY[n][r] = np.repeat(
			a=dataY,
			repeats=ySizes,
			axis=0,
		)

		new[n][r] = np.vstack(tup=new[n][r])

nodesDataX = new

########################################################

# nodesDataX
#	nodes rounds

# testDataX
#	datasets

print('Scale')

# up until now views have been used of the original data in memory
# fit_transform will generate a copy, be careful with memory usage for large datasets

for n in range(numNodes):
	for r in range(numRounds):
		x = nodesDataX[n][r].shape
		# flatten windows so that sklearn can process
		# then place back after
		nodesDataX[n][r] = nodesDataX[n][r].reshape(x[0]*x[1], x[2])
		nodesDataX[n][r] = scaler.fit_transform(X=nodesDataX[n][r])
		nodesDataX[n][r] = nodesDataX[n][r].reshape(x)

for d in range(numDatasets):
	x = testDataX[d].shape
	testDataX[d] = testDataX[d].reshape(x[0]*x[1], x[2])
	testDataX[d] = scaler.fit_transform(X=testDataX[d])
	testDataX[d] = testDataX[d].reshape(x)

#######################################################

print('Shuffle')

# reinitialise the random generator to be consistent with previous versions of implementation
rng = np.random.default_rng(seed=seed)

for n in range(numNodes):
	for r in range(numRounds):
		newIndexes = rng.permutation(len(nodesDataX[n][r]))
		nodesDataX[n][r] = nodesDataX[n][r][newIndexes]
		nodesDataY[n][r] = nodesDataY[n][r][newIndexes]

#####################################################

# nodesDataX
#	nodes rounds

# nodesDataY
#	nodes rounds

# testDataX
#	datasets

# dataY
#	datasets

# dataL
#	datasets

print('To Tensor + Store')

for n in range(numNodes):
	for r in range(numRounds):
		nodesDataX[n][r] = torch.Tensor(nodesDataX[n][r])
		nodesDataY[n][r] = torch.Tensor(nodesDataY[n][r])

		torch.save(nodesDataX[n][r], Path(nodeStore, 'nodes{0}-rounds{1}-trainX-{2}-{3}-{4}-{5}-{6}.nodedata'.format(numNodes, numRounds, n+1, r+1, splitBenign, splitAttack, True)))
		torch.save(nodesDataY[n][r], Path(nodeStore, 'nodes{0}-rounds{1}-trainY-{2}-{3}-{4}-{5}-{6}.nodedata'.format(numNodes, numRounds, n+1, r+1, splitBenign, splitAttack, True)))

for d in range(numDatasets):
	testDataX[d] = torch.Tensor(testDataX[d])
	torch.save(testDataX[d], Path(nodeStore, 'nodes{0}-rounds{1}-testX-{2}.nodedata'.format(numNodes, numRounds, dataL[d])))

dataY = torch.Tensor(dataY)
torch.save(dataY, Path(nodeStore, 'nodes{0}-rounds{1}-testY.nodedata'.format(numNodes, numRounds)))

for n in range(numNodes):
	for r in range(numRounds):
		torch.save(nodeTrainStats[n][r], Path(nodeStore, 'nodes{0}-rounds{1}-trainS-{2}-{3}-{4}-{5}-{6}.nodedata'.format(numNodes, numRounds, n+1, r+1, splitBenign, splitAttack, True)))
