#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


# Client module, used in both the CPU and CUDA client node

from model import Model
from aiohttp import web
from sys import argv
from statistics import postStats
import time
from io import BytesIO
import torch
from pathlib import Path

# https://pytorch.org/docs/stable/generated/torch.save.html
# https://docs.aiohttp.org/en/stable/index.html
# https://docs.aiohttp.org/en/stable/streams.html
# https://docs.python.org/3/library/io.html
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
# https://pytorch.org/docs/stable/notes/randomness.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# https://docs.aiohttp.org/en/stable/client.html

# mountpoint of node data, relative to inside the container
flDataPath = Path('/fypdata/')

class Client():

	def __init__(self, uuid, deviceLabel, offload, seed):

		self.uuid = uuid

		self.model = None

		# semaphore, prevent multiple conflicting requests at same time
		self.locked = False

		# if True, after training/validation move the model parameters back into CPU memory.
		self.offload = offload

		# what device will data be loaded/ the model stored/ training/validation take place
		self.device = torch.device(deviceLabel)

		# seed, deterministic flag. performance loss, remove in real-world.
		# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
		# https://pytorch.org/docs/stable/notes/randomness.html
		self.seed = seed
		torch.manual_seed(self.seed)
		torch.use_deterministic_algorithms(True)

		print('Client alive.')
		return

	def checkLock(self):

		if self.locked:
			print('ERROR: attempted to run while locked.')
			exit()
		else:
			self.locked = True
		
		return

	def checkUnlock(self):

		if not self.locked:
			print('ERROR: attempted to unlock while unlocked, something has crashed.')
			exit()
		else:
			self.locked = False
		
		return

	# initialise the model with selected hyperparameters
	async def initModel(self, req):
		self.checkLock()

		print('Init Model')

		data = await req.post()

		hiddenSize     = int(data['hiddenSize'])
		inputSize      = int(data['inputSize'])
		numEpochs      = int(data['numEpochs'])
		batchSize      = int(data['batchSize'])
		sequenceLength = int(data['sequenceLength'])
		learnRate      = float(data['learnRate'])


		self.model = Model(
			hiddenSize,
			inputSize,
			numEpochs,
			batchSize,
			sequenceLength,
			learnRate,
			self.seed,
		)

		# if offload selected defer moving model onto device until required
		if not self.offload:
			self.model = self.model.to(self.device)

		self.checkUnlock()
		return web.Response(status=200, text='OK')

	# returns the current model parameters and last training round sizes as a file
	def getParams(self, req):
		self.checkLock()

		print('Params requested.')

		# https://docs.aiohttp.org/en/stable/streams.html
		# https://docs.python.org/3/library/io.html

		# bytes buffer
		b = BytesIO()

		torch.save({
			'state_dict': self.model.state_dict(),
			'lastSize': self.model.trainingSizes[-1],
		}, b)

		self.checkUnlock()
		return web.Response(status=200, body=b.getvalue())

	# recieve the new global model and replace the parameters of the clients model
	async def newParams(self, req):
		self.checkLock()

		print('New params.')

		# https://docs.aiohttp.org/en/stable/streams.html
		# https://docs.python.org/3/library/io.html

		# bytes buffer, seek to start of file
		data = await req.read()
		data = BytesIO(data)
		data.seek(0)

		# THIS IS DANGEROUS IF YOU DO NOT 100% TRUST THE SERVER
		if self.offload:
			# params always cpu from server
			self.model.load_state_dict(
				torch.load(data)
			)
		else:
			# move onto specified device, e.g. VRAM
			self.model.load_state_dict(
				torch.load(data, map_location=self.device,)
			)

		self.checkUnlock()
		return web.Response(status=200, text='OK')

	# training trigger, the current training round is stored in self.model
	# read data corresponding to UUID from mountpoint
	async def training(self, req):
		self.checkLock()
		print('Training')

		data = await req.post()

		if self.offload:
			print('offload: move onto gpu')
			self.model = self.model.to(self.device)

		# ensure all options present, else 500
		# since neither the server or client has access to this info it is passed through
		# so that the stats can be appropriately tagged
		trainParams = {
			'numNodes':    data.get('numNodes'),
			'numRounds':   data.get('numRounds'),
			'statsSet':    data.get('statsSet'),
			'attackSplit': bool(int(data.get('attackSplit'))),
			'benignSplit': bool(int(data.get('benignSplit'))),
			'fedMethod':   data.get('fedMethod'),
			'fedWeight':   data.get('fedWeight'),
			'longTail':    bool(int(data.get('longTail'))),
		}

		pathX = Path(flDataPath, 'nodes{0}-rounds{1}-trainX-{2}-{3}-{4}-{5}-{6}.nodedata'.format(
			trainParams['numNodes'],
			trainParams['numRounds'],
			self.uuid,
			self.model.trainingRound,
			trainParams['benignSplit'],
			trainParams['attackSplit'],
			trainParams['longTail']
		))
		pathY = Path(flDataPath, 'nodes{0}-rounds{1}-trainY-{2}-{3}-{4}-{5}-{6}.nodedata'.format(
			trainParams['numNodes'],
			trainParams['numRounds'],
			self.uuid,
			self.model.trainingRound,
			trainParams['benignSplit'],
			trainParams['attackSplit'],
			trainParams['longTail']
		))

		# sizes of each traffic type are also passed through since the model is unaware of the context of the training data

		pathS = Path(flDataPath, 'nodes{0}-rounds{1}-trainS-{2}-{3}-{4}-{5}-{6}.nodedata'.format(
			trainParams['numNodes'],
			trainParams['numRounds'],
			self.uuid,
			self.model.trainingRound,
			trainParams['benignSplit'],
			trainParams['attackSplit'],
			trainParams['longTail']
		))

		print('X {0}'.format(pathX))
		print('Y {0}'.format(pathY))
		print('S {0}'.format(pathS))
		
		dataX = torch.load(pathX, map_location=self.device,)
		dataY = torch.load(pathY, map_location=self.device,)
		dataS = torch.load(pathS)

		print('X {0}'.format(dataX.shape))
		print('Y {0}'.format(dataY.shape))
		print('S {0}'.format(len(dataS)))
		
		# training function will return train loss for each epoch
		# [...]
		results = self.model.runTraining(dataX, dataY, dataS, self.device,)

		# post each train loss to stats database
		for e, l in enumerate(results):
			postStats(
				trainParams['statsSet'],
				{
					'uuid':            self.uuid,
					'epochs':          e+1,
					'sequence_length': self.model.sequenceLength,
					'hidden_size':     self.model.hiddenSize,
					'batch_size':      self.model.batchSize,
					'learn_rate':      self.model.learnRate,
					'total_nodes':     trainParams['numNodes'],
					'total_rounds':    trainParams['numRounds'],
					'time':            int(time.time()),
					'current_round':   self.model.trainingRound-1,
					'seed':            self.model.seed,
					'device':          self.device.type,
					'dataSize':        sum(dataS),

					'train_loss':      l,
					'attackSplit':     trainParams['attackSplit'],
					'benignSplit':     trainParams['benignSplit'],

					'fedMethod':       trainParams['fedMethod'],
					'fedWeight':       trainParams['fedWeight'],

					'longTail':        trainParams['longTail'],
				},
			)

		if self.offload:
			print('offload: move off gpu')
			self.model = self.model.to('cpu')
		
		# delete data from memory now its no longer needed
		del(dataX)
		del(dataY)

		# if using GPU flush reserved space for data from VRAM
		if self.device.type == 'cuda':
			torch.cuda.empty_cache()

		self.checkUnlock()
		return web.Response(status=200, text='OK')

	# evaluate the performance of the model
	async def validation(self, req):
		self.checkLock()
		print('Validation.')

		data = await req.post()

		if self.offload:
			print('offload: move onto gpu')
			self.model = self.model.to(self.device)
		
		# ensure all options present, else 500
		# since neither the server or client has access to this info it is passed through
		# so that the stats can be appropriately tagged
		validParams = {
			'numNodes':      data.get('numNodes'),
			'numRounds':     data.get('numRounds'),
			'datasets':      data.get('datasets').split(','), # ORDER MATTERS
			'statsSet':      data.get('statsSet'),
			'mergeComplete': data.get('mergeComplete'),
			'attackSplit':   bool(int(data.get('attackSplit'))),
			'benignSplit':   bool(int(data.get('benignSplit'))),
			'fedMethod':     data.get('fedMethod'),
			'fedWeight':     data.get('fedWeight'),
			'longTail':      bool(int(data.get('longTail'))),
		}

		# the sentiment for each dataset is stored in the same file
		# [dataset sentiment, ...] float32
		# y-axis is generated on the fly to save storage/memory usage

		pathY = Path(flDataPath, 'nodes{0}-rounds{1}-testY.nodedata'.format(
			validParams['numNodes'],
			validParams['numRounds'],
		))
		print('vY {0}'.format(pathY))
		valuesY = torch.load(pathY, map_location=self.device,)
		print('vY {0}'.format(valuesY.shape))

		for d, dataset in enumerate(validParams['datasets']):

			pathX = Path(flDataPath, 'nodes{0}-rounds{1}-testX-{2}.nodedata'.format(
				validParams['numNodes'],
				validParams['numRounds'],
				dataset,
			))
			print('X {0}'.format(pathX))
			dataX = torch.load(pathX, map_location=self.device,)
			print('X {0}'.format(dataX.shape))

			dataY = torch.full(
				size=(len(dataX),),
				fill_value=valuesY[d],
				#dtype=np.float32,
				device=self.device,
			)
			print('Y {0}'.format(dataY.shape))

			#valueY = dataY[d]

			loss, accuracy, precision, recall, f1 = self.model.runValidation(dataX, dataY, self.device,)

			# post model performance of each dataset to stats db
			postStats(
				validParams['statsSet'],
				{
					'uuid':            self.uuid,
					'epochs':          self.model.numEpochs,
					'sequence_length': self.model.sequenceLength,
					'hidden_size':     self.model.hiddenSize,
					'batch_size':      self.model.batchSize,
					'learn_rate':      self.model.learnRate,
					'total_nodes':     validParams['numNodes'],
					'total_rounds':    validParams['numRounds'],
					'time':            int(time.time()),
					'current_round':   self.model.trainingRound-1,
					'seed':            self.model.seed,
					'device':          self.device.type,
					'dataSize':        len(dataX),

					'mergeComplete': bool(int(validParams['mergeComplete'])), # str 0/1

					'valid_loss': loss,
					'accuracy':   accuracy,
					'precision':  precision,
					'f1':         recall,
					'recall':     f1,
					'dataset':    dataset,

					'attackSplit': validParams['attackSplit'],
					'benignSplit': validParams['benignSplit'],

					'fedMethod': validParams['fedMethod'],
					'fedWeight': validParams['fedWeight'],

					'longTail': validParams['longTail'],
				},
			)

		if self.offload:
			print('offload: move off gpu')
			self.model = self.model.to('cpu')
		
		del(dataX)
		del(dataY)

		if self.device.type == 'cuda':
			torch.cuda.empty_cache()

		self.checkUnlock()
		return web.Response(status=200, text='OK')


def main(uuid, deviceLabel, offload, seed):
	client = Client(uuid, deviceLabel, offload, seed)

	# list of entrypoints for aiohttp
	# https://docs.aiohttp.org/en/stable/index.html
	app = web.Application()
	app.add_routes([
		web.post('/initModel', client.initModel),
		web.get('/getParams', client.getParams),
		web.post('/newParams', client.newParams),
		web.post('/training', client.training),
		web.post('/validation', client.validation),
	])
	web.run_app(app, port=8080)

if __name__ == '__main__':
	# environment variables pass through command line
	# in docker these are passed during container start
	uuid = int(argv[1])
	deviceLabel = argv[2]
	offload = bool(int(argv[3]))
	seed = int(argv[4])
	print('Client: {0}'.format(uuid))
	main(uuid, deviceLabel, offload, seed)
