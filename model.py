#!/usr/bin/python3


# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022

# Model module, used by both the server and client

# However the server only uses it for storage of (hyper)parameters

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from math import floor


# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
# https://pytorch.org/docs/stable/notes/randomness.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# https://pytorch.org/docs/stable/generated/torch.no_grad.html
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

class Model(torch.nn.Module):
	def __init__(self,
		hiddenSize,
		inputSize,
		numEpochs,
		batchSize,
		sequenceLength,
		learnRate,
		seed,	
	):
		super(Model, self).__init__() # inherit functions from nn.Module
		# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

		# seed, deterministic flag. performance loss, remove in real-world.
		# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
		# https://pytorch.org/docs/stable/notes/randomness.html
		self.seed = seed
		torch.manual_seed(self.seed)
		torch.use_deterministic_algorithms(True)
		#np.random.seed(seed)

		#torch.autograd.set_detect_anomaly(False)
		#torch.autograd.profiler.profile(False)

		# hyperparameters
		
		self.hiddenSize = hiddenSize
		self.inputSize = inputSize

		self.numEpochs = numEpochs
		self.batchSize = batchSize
		
		self.sequenceLength = sequenceLength

		self.learnRate = learnRate

		self.numLayers = 1

		# important: batch first input to network
		self.lstm = torch.nn.LSTM(
			input_size=self.inputSize,
			hidden_size=self.hiddenSize,
			num_layers=self.numLayers,
			bias=True,
			batch_first=True,
			dropout=0,
			bidirectional=False,
			proj_size=0,
		)

		#self.layerDropout = nn.Dropout(p=0.2)

		# aka. fully connected layer
		self.layerLinear = torch.nn.Linear(self.hiddenSize, 1)

		self.layerSigmoid = torch.nn.Sigmoid()
		
		# store the sizes of each dataset trained upon for each training round.
		self.trainingSizes = []

		self.trainingRound = 1

		print('Model initiated.')

	# perform a forward pass on the model
	def forward(self, batchData, device):
		
		hiddenState = torch.zeros(self.numLayers, len(batchData), self.hiddenSize, device=device,)
		cellState = torch.zeros(self.numLayers, len(batchData), self.hiddenSize, device=device,)

		output, (hiddenState, cellState) = self.lstm(batchData, (hiddenState, cellState))

		output = output[:, -1:, :] # last cell only

		output = torch.squeeze(output, 1) # remove extra dimension of 1

		output = self.layerLinear(output)
	
		output = torch.squeeze(output, 1) # again remove dimension

		output = self.layerSigmoid(output)

		return output
	
	# perform a training round
	# trainDataX and trainDataY being tensors
	# data, expected output for each window
	# sizes are also passed through since the model is unaware of the context of the training data
	def runTraining(self, trainDataX, trainDataY, trainDataS, device):
		# training round

		# trainDataX, trainDataY
		# all datasets
		
		print('Model training.')

		optimiser = torch.optim.Adam(self.parameters(), lr=self.learnRate)
		lossFunction = torch.nn.BCELoss()

		batchDrop = len(trainDataX) % self.batchSize
		print('WARN: last {0} rows, small batch size'.format(batchDrop))

		numRows = len(trainDataX)

		# store and return the train loss from every epoch
		epochLosses = []

		# put the model in training mode
		self.train()

		for e in range(1, self.numEpochs+1):
			
			# calculate loss accumulatively over all batches
			totalLoss = 0.0
			numBatchesRun = 0

			for b in range(0, numRows, self.batchSize):
				
				# set to null rather than zero
				# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
				optimiser.zero_grad(set_to_none=True,)

				outputs = self.forward(trainDataX[b:b+self.batchSize], device)

				loss = lossFunction(outputs, trainDataY[b:b+self.batchSize])

				loss.backward()

				optimiser.step()

				totalLoss += loss.item()
				numBatchesRun += 1

			epochLosses.append(totalLoss / numBatchesRun)
		
		self.trainingSizes.append(trainDataS)

		self.trainingRound += 1

		del(trainDataX)
		del(trainDataY)

		print(epochLosses)

		return epochLosses

	# perform validation of the model, using a dataset, not all
	def runValidation(self, validDataX, validDataY, device):
		print('Model validation.')

		# https://pytorch.org/docs/stable/generated/torch.no_grad.html
		# no need to calculate gradients during validation, since no backprop
		with torch.no_grad():

			lossFunction = torch.nn.BCELoss()

			# put the model in eval mode
			self.eval()

			totalLoss = 0.0
			numBatchesRun = 0
			# store the output tensors for each batch, so that they can be compared to expected
			# to generate metrics
			allOutputs = []

			for b in range(0, len(validDataX), self.batchSize):

				outputs = self.forward(validDataX[b:b+self.batchSize], device)

				totalLoss += lossFunction(outputs, validDataY[b:b+self.batchSize]).item()
				numBatchesRun += 1

				outputs = outputs.to('cpu')

				# if sigmoid output is greater than 0.5, attack. else benign.
				allOutputs.append(torch.gt(outputs, 0.5))


			validSentiment = 0
			
			# do not use exact number due to some inaccuracy with np.float e.g. 1.0005
			if validDataY[0] >= 0.5:
				validSentiment = 1

			validDataY = torch.gt(validDataY, 0.5)
			validDataY = validDataY.to('cpu')

			# combine array of tensors into a tensor, each batch into one tensor
			allOutputs = torch.cat(allOutputs, dim=0)

			totalLoss = totalLoss / numBatchesRun

			# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics


			# if validSentiment is not specified then recall will be incorrect for
			# benign dataset, since for them positive == 0

			# precision and f1-score will not return meaningful data unless
			# validation is run over all datasets combined, rather than just one.

			accuracy = accuracy_score(
				y_true=validDataY,
				y_pred=allOutputs,
				normalize=True # between 0 and 1, not sum
			)

			precision = precision_score(
				y_true=validDataY,
				y_pred=allOutputs,
				pos_label=validSentiment,
				average='binary',
			)

			recall = recall_score(
				y_true=validDataY,
				y_pred=allOutputs,
				pos_label=validSentiment,
				average='binary',
			)

			f1 = f1_score(
				y_true=validDataY,
				y_pred=allOutputs,
				pos_label=validSentiment,
				average='binary',
			)

		del(validDataX)
		del(validDataY)
		del(allOutputs)

		print(totalLoss, accuracy, precision, recall, f1)

		return(totalLoss, accuracy, precision, recall, f1)




