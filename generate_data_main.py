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
from json import load as json_load
from pathlib import Path

# Script to prepare data to be passed to each node for training and validation.

# load in configuration
configData = {}
with open(Path('config.json'), 'r') as f:
	configData = json_load(f)


# command line arguments
# these are usually passed by the simulation script: run_simulation.sh

numNodes       = int(argv[1])
numRounds      = int(argv[2])
sequenceLength = int(argv[3])

# split attack and/or benign datasets
# 0/1
# both 0 for equal split between all nodes
splitAttack = bool(int(argv[4]))
splitBenign = bool(int(argv[5]))

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

#######################################
# if unique split, decide what nodes get which traffic types

# split for each node attack/benign
# [] each represents dataset, numbers inside are nodes that will receive data from it

splitDatasetsNodes = []
for d in range(numDatasets):
	splitDatasetsNodes.append([])

# 1. same number as nodes as datasets, each gets one
# 2. more nodes than datasets, some get none
# 3. less nodes than datasets, some get more than one

# shuffle the order of the datasets, so that they are not grouped in order
# shuffle the nodes that will be assigned, so that how many datasets they get is random

print('Splitting attack datasets between nodes if specified')

# get index position of each attack dataset
attackIndexes = np.flatnonzero(dataY==1)

if splitAttack:
	nIndexes = random.sample(range(numNodes), numNodes)
	
	rng.shuffle(attackIndexes)
	attackIndexes = np.array_split(attackIndexes, numNodes)

	for n, i in enumerate(attackIndexes):
		# nIndexes[n] gets datasets in i
		for d in i:
			splitDatasetsNodes[d].append(nIndexes[n])
else:
	for d in attackIndexes:
		splitDatasetsNodes[d] = list(range(numNodes))

print('Splitting benign datasets between nodes if specified')

benignIndexes = np.flatnonzero(dataY==0)

if splitBenign:
	nIndexes = random.sample(range(numNodes), numNodes)
	
	
	rng.shuffle(benignIndexes)
	benignIndexes = np.array_split(benignIndexes, numNodes)

	for n, i in enumerate(benignIndexes):
		# nIndexes[n] gets datasets in i
		for d in i:
			splitDatasetsNodes[d].append(nIndexes[n])
else:
	for d in benignIndexes:
		splitDatasetsNodes[d] = list(range(numNodes))

##########################################
# split each dataset between each node if needed
# equal split
# which part is assigned to what node is randomised

# dataX
# 	datasets

# nodesDataX
# 	nodes datasets

print('Split datasets between each node')

nodesDataX = []
for n in range(numNodes):
	nodesDataX.append([])
	for d in range(numDatasets):
		nodesDataX[n].append([])

sizesSplit = []

for d in range(numDatasets):
	#print(dataL[d])

	splitSize = floor(len(dataX[d]) / len(splitDatasetsNodes[d]))

	#print(splitSize)

	dropped = len(dataX[d]) - (splitSize * len(splitDatasetsNodes[d]))

	if dropped > 0:
		print('WARN: {0} dropped rows {1}'.format(dataL[d], dropped))

	if splitSize < sequenceLength:
		print('ERROR: split size less than seq length {0} {1}'.format(splitSize, dataL[d]))
		exit()

	sizesSplit.append(splitSize)

	pos = 0
	for n in random.sample(splitDatasetsNodes[d], len(splitDatasetsNodes[d])):
		nodesDataX[n][d] = dataX[d][pos:pos+splitSize]
		pos += splitSize


# sizesSplit
# 	datasets

#############################################

# nodesDataX
# 	nodes datasets

# nodesDataX
#	nodes datasets rounds

print('Split each nodes datasets data into training rounds')

sizesRounds = [floor(i / numRounds) for i in sizesSplit]

for d in range(numDatasets):
	if sizesRounds[d] < sequenceLength:
		print('ERROR: round size less than seq length {0} {1}'.format(sizesRounds[d], dataL[d]))
		exit()

for d in range(numDatasets):

	#print(dataL[d])
	#print(sizesRounds[d])

	dropped = sizesSplit[d] - (sizesRounds[d]*numRounds)
	
	if dropped > 0:
		print('WARN: dropped {0} from each nodes data of {1}'.format(dropped, dataL[d]))

	for n in splitDatasetsNodes[d]:

		new = []

		pos = 0
		for _ in range(numRounds):
			new.append(nodesDataX[n][d][pos:pos+sizesRounds[d]])
			pos += sizesRounds[d]

		nodesDataX[n][d] = new


################################################

# nodesDataX
#	nodes datasets rounds

# testDataX
#	datasets rounds

# https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

print('Sliding window')

for d in range(numDatasets):
	for n in splitDatasetsNodes[d]:
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
	for n in splitDatasetsNodes[d]:
		for r in range(numRounds):
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
			if n in splitDatasetsNodes[d]:
				new[n][r].append(nodesDataX[n][d][r])
				ySizes.append(len(nodesDataX[n][d][r]))
			else:
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

		torch.save(nodesDataX[n][r], Path(nodeStore, 'nodes{0}-rounds{1}-trainX-{2}-{3}-{4}-{5}-{6}.nodedata'.format(numNodes, numRounds, n+1, r+1, splitBenign, splitAttack, False)))
		torch.save(nodesDataY[n][r], Path(nodeStore, 'nodes{0}-rounds{1}-trainY-{2}-{3}-{4}-{5}-{6}.nodedata'.format(numNodes, numRounds, n+1, r+1, splitBenign, splitAttack, False)))

for d in range(numDatasets):
	testDataX[d] = torch.Tensor(testDataX[d])
	torch.save(testDataX[d], Path(nodeStore, 'nodes{0}-rounds{1}-testX-{2}.nodedata'.format(numNodes, numRounds, dataL[d])))

dataY = torch.Tensor(dataY)
torch.save(dataY, Path(nodeStore, 'nodes{0}-rounds{1}-testY.nodedata'.format(numNodes, numRounds)))

for n in range(numNodes):
	for r in range(numRounds):
		torch.save(nodeTrainStats[n][r], Path(nodeStore, 'nodes{0}-rounds{1}-trainS-{2}-{3}-{4}-{5}-{6}.nodedata'.format(numNodes, numRounds, n+1, r+1, splitBenign, splitAttack, False)))
