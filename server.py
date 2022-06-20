#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


# Server module, used in Docker server node

from model import Model
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
from sys import argv
from asyncio import gather as asyncio_gather
from io import BytesIO
from json import loads as json_loads
import torch


# https://pytorch.org/docs/stable/generated/torch.save.html
# https://docs.aiohttp.org/en/stable/index.html
# https://docs.aiohttp.org/en/stable/streams.html
# https://docs.python.org/3/library/io.html
# https://stackoverflow.com/questions/64534844/python-asyncio-aiohttp-timeout
# https://docs.python.org/3/library/asyncio-task.html
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.aiohttp.org/en/stable/client.html

class Server():

	def __init__(self, uuid, threadlimit, seed):

		self.uuid = uuid

		self.model = None

		# UUID map to IP/Port 
		self.clients = {}
		self.numClients = 0

		# will store nodes parameters waiting for aggregation
		self.paramsQueue = []

		# prevent TCP/HTTP timeout while waiting for nodes to respond
		# https://stackoverflow.com/questions/64534844/python-asyncio-aiohttp-timeout
		self.timeout = ClientTimeout(total=None, sock_connect=3600, sock_read=3600)

		# server does not train/valid therefore CPU
		# defined so that GPU nodes params can be mapped to CPU mem
		self.device = torch.device('cpu')

		# how many different nodes can be simultaneously issued an action
		# use to cap number of nodes to max CPU/GPU usage
		self.threadlimit = threadlimit

		# seed, deterministic flag. performance loss, remove in real-world.
		# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
		# https://pytorch.org/docs/stable/notes/randomness.html
		self.seed = seed
		torch.manual_seed(self.seed)
		torch.use_deterministic_algorithms(True)

		print('Server alive.')

	#def checkLock(self, name):
	#	if self.locks[name] > 0:
	#		print('ERROR: {0} tried to run while locked.'.format(name))
	#		exit()
	#	else:
	#		for l in self.toLock[name]:
	#			self.locks[l] += 1

	#def checkUnlock(self, name):
	#	for l in self.toLock[name]:
	#		self.locks[l] -= 1

	# initialise the server's model not clients
	# the server will then transfer the same hyperparameters to client if initModelClient called
	async def initModel(self, req):
		#self.checkLock('initModel')
		print('Init server model.')

		# get required params from POST form
		data = await req.post()

		hiddenSize     = int(data['hiddenSize'])
		inputSize      = int(data['inputSize'])
		numEpochs      = int(data['numEpochs'])
		batchSize      = int(data['batchSize'])
		sequenceLength = int(data['sequenceLength'])
		learnRate      = float(data['learnRate'])

		# create model
		self.model = Model(
			hiddenSize,
			inputSize,
			numEpochs,
			batchSize,
			sequenceLength,
			learnRate,
			self.seed,
		)

		#self.checkUnlock('initModel')
		return web.Response(status=200, text='OK')

	# link a client node to the server
	# the server will pass forward commands to these nodes
	# whenever an all command is issued
	async def registerClient(self, req):
		print('Register new client.')
		#self.checkLock('registerClient')

		data = await req.post()
		a = int(data.get('uuid'))
		b = data.get('ip')
		c = int(data.get('port'))

		print(a,b,c)

		# UUID map to IP/Port
		self.clients[a] = [b,c]

		self.numClients += 1

		#self.checkUnlock('registerClient')
		return web.Response(status=200, text='OK')

	# request and buffer the model parameters of a node
	async def gatherParams(self, uuid, session):
		print('Requesting client {0} for params'.format(uuid))

		# type cast just in case
		ip   = str(self.clients[uuid][0])
		port = str(self.clients[uuid][1])
		url  = 'http://' + ip + ':' + port + '/getParams'

		# https://docs.aiohttp.org/en/stable/streams.html
		# https://docs.python.org/3/library/io.html

		# bytes buffer, seek to start of file
		results = await session.get(url)
		data = await results.read()
		data = BytesIO(data)
		data.seek(0)

		# INSECURE, do not use if you cannot trust the client
		data = torch.load(data, map_location=self.device,)

		self.paramsQueue.append([
			data['state_dict'], # model parameters
			data['lastSize'], # training sizes
		])

		return results.status

	# request and buffer parameters of all nodes
	async def gatherAllParams(self, req):
		#self.checkLock('gatherAllParams')
		
		print('Gather params from all clients.')
		
		self.paramsQueue = [] # clear paramsQueue, should be empty already
		
		async with ClientSession(timeout=self.timeout) as session:
			# request all clients registered with the server
			tasks = [self.gatherParams(uuid, session) for uuid in self.clients]
			# wait for each client to return a status code
			taskResults = await asyncio_gather(*tasks)

			for u, s in enumerate(taskResults):
				# check for non-200, error codes
				u = list(self.clients.keys())[u]
				if s != 200:
					#self.checkUnlock('gatherAllParams')
					return web.Response(status=500, text='Client {0} unsuccessful.'.format(u))

		#self.checkUnlock('gatherAllParams')
		return web.Response(status=200, text='OK')
	
	# apply weighting to buffered params and merge
	# new global model is applied to the server model (for storage/compat check only)
	def fedApply(self, nodeWeights):
		print('Applying weighting to recieved params.')
		# paramsQueue
		# [[params, [size1, size2, ...]]]

		globalParams = None

		for n, (params, _) in enumerate(self.paramsQueue):
			
			for name in params:
				params[name] = params[name] * nodeWeights[n]

			if globalParams == None:
				globalParams = params
			else:
				for name in globalParams:
					globalParams[name] = globalParams[name] + params[name]
		
		# apply new global model parameters to server model (for storage/compat check only)
		self.model.load_state_dict(globalParams)

		# clear the parameters buffer now that they are processed
		self.paramsQueue = []

		return


	# using trainingSizes from nodes calculate the weight that should be applied
	# to that nodes parameters

	# Communication-Efficient Learning of Deep Networks from Decentralized Data, H. Brendan McMahan
	# https://arxiv.org/pdf/1602.05629.pdf

	async def fedAVG(self, req):
		print('FedAVG.')
		# paramsQueue
		# [[params, [size1, size2, ...]]]

		# since fedAVG is only concerned with total size rather than each dataset
		for n in range(len(self.paramsQueue)):
			self.paramsQueue[n][1] = sum(self.paramsQueue[n][1])

		totalSize = 0.0
		for _, size in self.paramsQueue:
			totalSize += size
		
		# generate the weighting
		# total node data size / all nodes total data size
		nodeWeights = []
		for _, size in self.paramsQueue:
			nodeWeights.append(size/totalSize)
		
		print(nodeWeights)

		# go to aggregation
		self.fedApply(nodeWeights)
		
		return web.Response(status=200, text='OK')
	

	async def fedAVGDiverse(self, req):
		print('FedDiverse.')
		# paramsQueue
		# [[params, [size1, size2, ...]]]
		
		# number of nodes parameters to process
		nc = len(self.paramsQueue)
		# number of datasets
		# assumes all node training sizes have same num of datasets
		nd = len(self.paramsQueue[0][1])

		# total number of rows for each dataset
		totalSizes = [0.0] * nd
		for n in range(nc):
			for d in range(nd):
				totalSizes[d] += self.paramsQueue[n][1][d]
		
		nodeWeights = [0.0] * nc
		for n in range(nc):
			for d in range(nd):
				# each node, how much of a dataset compared to others
				# sum fractions for each node
				nodeWeights[n] += (self.paramsQueue[n][1][d] / totalSizes[d])
		
		print(nodeWeights)

		# normalise/ratio, so add up to 1
		totalWeight = sum(nodeWeights)
		for n in range(nc):
			nodeWeights[n] = nodeWeights[n] / totalWeight
		
		print(nodeWeights)
		# go to aggregation
		self.fedApply(nodeWeights)

		return web.Response(status=200, text='OK')
	
	

	async def fedAVGWeighted(self, req):
		print('FedWeighted.')
		# paramsQueue
		# [[params, [size1, size2, ...]]]

		# convert the weights and dataset indexes from string to int if required
		data = await req.text()

		newdata = {}
		for d, multiplier in json_loads(data).items():
			newdata[int(d)] = float(multiplier)
		data = newdata

		print(data)

		# data
		# [datasetID: multiplier, ]
		
		# number of parameters to aggregate
		nc = len(self.paramsQueue)

		# number of datasets
		# assumes all node training sizes have same num of datasets
		nd = len(self.paramsQueue[0][1])

		# total number of flows, all types, all nodes
		totalSize = 0.0
		for n in range(nc):
			for d in range(nd):
				totalSize += self.paramsQueue[n][1][d]
		
		# store the fraction for each dataset separate
		nodeWeights = []
		for n in range(nc):
			nodeWeights.append([0.0] * nd)
		
		# generate the fraction for each node, for each dataset
		for n in range(nc):
			for d in range(nd):
				nodeWeights[n][d] = self.paramsQueue[n][1][d] / totalSize
		
		print(nodeWeights)
		
		# multiply the specified dataset fractions
		for n in range(nc):
			for d, multiplier in data.items():
				nodeWeights[n][d] *= multiplier
		
		# add up the fractions for each node
		for n in range(nc):
			nodeWeights[n] = sum(nodeWeights[n])
		
		# new total size after multiplying
		totalSize = sum(nodeWeights)

		# normalise
		for n in range(nc):
			nodeWeights[n] = nodeWeights[n] / totalSize

		print(nodeWeights)
		# go to aggregation
		self.fedApply(nodeWeights)

		return web.Response(status=200, text='OK')

	# send the server/global model parameters to a client
	async def sendParams(self, uuid, session):
		print('Sending params to client: {0}'.format(uuid))

		# type cast to be safe
		ip   = str(self.clients[uuid][0])
		port = str(self.clients[uuid][1])
		url  = 'http://' + ip + ':' + port + '/newParams'

		# https://docs.aiohttp.org/en/stable/streams.html
		# new bytes buffer for sending file
		b = BytesIO()
		torch.save(self.model.state_dict(), b)

		results = await session.post(url, data=b.getvalue())
		return results.status

	# send the global model to all clients registered with the server
	async def sendParamsAll(self, req):
		print('Sending params to all clients.')
		#self.checkLock('sendParamsAll')

		async with ClientSession(timeout=self.timeout) as session:
			# send new params to all clients
			tasks = [self.sendParams(uuid, session) for uuid in self.clients]
			# wait for each client to return a status code
			taskResults = await asyncio_gather(*tasks)

			# check all nodes returned status code 200, OK
			for u, s in enumerate(taskResults):
				u = list(self.clients.keys())[u]
				if s != 200:
					return web.Response(status=500, text='Client {0} unsuccessful.'.format(u))


		#self.checkUnlock('sendParamsAll')
		return web.Response(status=200, text='OK')

	# ask a node to begin training on its local data
	async def triggerTraining(self, uuid, trainParams, session):
		print('Trigger training client: {0}'.format(uuid))

		# type cast to be safe
		ip   = str(self.clients[uuid][0])
		port = str(self.clients[uuid][1])
		url  = 'http://' + ip + ':' + port + '/training'

		# wait for completion then return status code
		results = await session.post(url, data=trainParams)
		return results.status

	# ask all registered nodes to train
	async def triggerTrainingAll(self, req):
		print('Trigger training for all clients.')

		#self.checkLock('triggerTrainingAll')

		# POST form
		data = await req.post()

		print('Training all num nodes: {0}'.format(self.numClients))

		# ensure all options present, else 500
		# since neither the server or client has access to this info it is passed through
		# so that the stats can be appropriately tagged
		trainParams = {
			'numNodes':  self.numClients,
			'numRounds': data.get('numRounds'),
			'statsSet':  data.get('statsSet'),
			'attackSplit': data.get('attackSplit'),
			'benignSplit': data.get('benignSplit'),
			'fedMethod': data.get('fedMethod'),
			'fedWeight': data.get('fedWeight'),
			'longTail': data.get('longTail'),
		}

		# enforce timeout and thread limit
		async with ClientSession(timeout=self.timeout, connector=TCPConnector(limit=self.threadlimit)) as session:
			# request all nodes registered with server
			tasks = [self.triggerTraining(uuid, trainParams, session) for uuid in self.clients]
			# wait for all status codes to be returned
			taskResults = await asyncio_gather(*tasks)

			# ensure all 200, no errors
			for u, s in enumerate(taskResults):
				u = list(self.clients.keys())[u]
				if s != 200:
					#self.checkUnlock('triggerTrainingAll')
					return web.Response(status=500, text='Client {0} unsuccessful.'.format(u))

		#self.checkUnlock('triggerTrainingAll')
		return web.Response(status=200, text='OK')

	# ask a node to validate its performance, report stats to DB with tags specified in validParams
	async def triggerValidation(self, uuid, validParams, session):
		print('Trigger validation client: {0}'.format(uuid))

		ip   = str(self.clients[uuid][0])
		port = str(self.clients[uuid][1])
		url  = 'http://' + ip + ':' + port + '/validation'

		results = await session.post(url, data=validParams)
		return results.status
	
	# alternative entrypoint for validation
	async def triggerValidationManual(self, req):

		data = await req.post()

		# neither the client or server has access to this information
		# hence it is required to be passed in the POST form
		# report stats to DB with tags specified
		validParams = {
			'numNodes':      self.numClients,
			'numRounds':     data.get('numRounds'),
			'datasets':      data.get('datasets'),
			'statsSet':      data.get('statsSet'),
			'mergeComplete': data.get('mergeComplete'),
			'attackSplit':   data.get('attackSplit'),
			'benignSplit':   data.get('benignSplit'),
			'fedMethod':     data.get('fedMethod'),
			'fedWeight':     data.get('fedWeight'),
			'longTail':      data.get('longTail'),
		}

		uuid = int(data.get('uuid'))
		print('Manual validation trigger for {0}'.format(uuid))

		# enforce timeout and thread limit
		async with ClientSession(timeout=self.timeout, connector=TCPConnector(limit=self.threadlimit)) as session:
			# request all nodes registered with server
			tasks = [self.triggerValidation(uuid, validParams, session)]
			# wait for all status codes to be returned
			taskResults = await asyncio_gather(*tasks)

			# ensure all 200, no errors
			for u, s in enumerate(taskResults):
				u = list(self.clients.keys())[u]
				if s != 200:
					return web.Response(status=500, text='Client {0} unsuccessful.'.format(u))

		return web.Response(status=200, text='OK')

	# ask all nodes to validate their performance
	# can be used before parameter aggregation to evaluate before/after
	async def triggerValidationAll(self, req):
		print('Trigger validation for all clients.')

		data = await req.post()

		# neither the client or server has access to this information
		# hence it is required to be passed in the POST form
		# report stats to DB with tags specified
		validParams = {
			'numNodes':      self.numClients,
			'numRounds':     data.get('numRounds'),
			'datasets':      data.get('datasets'),
			'statsSet':      data.get('statsSet'),
			'mergeComplete': data.get('mergeComplete'),
			'attackSplit':   data.get('attackSplit'),
			'benignSplit':   data.get('benignSplit'),
			'fedMethod':     data.get('fedMethod'),
			'fedWeight':     data.get('fedWeight'),
			'longTail':      data.get('longTail'),
		}

		# enforce timeout and thread limit
		async with ClientSession(timeout=self.timeout, connector=TCPConnector(limit=self.threadlimit)) as session:
			# request all nodes registered with server
			tasks = [self.triggerValidation(uuid, validParams, session) for uuid in self.clients]
			# wait for all status codes to be returned
			taskResults = await asyncio_gather(*tasks)
			
			# ensure all 200, no errors
			for u, s in enumerate(taskResults):
				u = list(self.clients.keys())[u]
				if s != 200:
					return web.Response(status=500, text='Client {0} unsuccessful.'.format(u))

		return web.Response(status=200, text='OK')

	# ask a node to initialise its model using the same hyperparameters as the server
	# this can be used as a reset rather than pulling the docker containers down
	async def initModelClient(self, uuid, modelParams, session):
		print('Initialise client model: {0}'.format(uuid))

		ip   = str(self.clients[uuid][0])
		port = str(self.clients[uuid][1])
		url  = 'http://' + ip + ':' + port + '/initModel'

		results = await session.post(url, data=modelParams)
		return results.status

	# not including self
	# ask all clients to initialise their models
	async def initModelClients(self, req):
		print('Initialising all client models.')

		data = await req.post()

		# gather hyperparameters from global model/self
		modelParams = {
			'hiddenSize':     self.model.hiddenSize,
			'inputSize':      self.model.inputSize,
			'numEpochs':      self.model.numEpochs,
			'batchSize':      self.model.batchSize,
			'sequenceLength': self.model.sequenceLength,
			'learnRate':      self.model.learnRate,
		}

		async with ClientSession(timeout=self.timeout) as session:
			# request all nodes registered with server
			tasks = [self.initModelClient(uuid, modelParams, session) for uuid in self.clients]
			# wait for all status codes to be returned
			taskResults = await asyncio_gather(*tasks)
			
			# ensure all 200, no errors
			for u, s in enumerate(taskResults):
				u = list(self.clients.keys())[u]
				if s != 200:
					return web.Response(status=500, text='Client {0} unsuccessful.'.format(u))


		return web.Response(status=200, text='OK')


def main(uuid, threadlimit, seed):
	server = Server(uuid, threadlimit, seed)

	# list of entrypoints for aiohttp web server
	# https://docs.aiohttp.org/en/stable/index.html
	app = web.Application()
	app.add_routes([
		web.post('/initModel', server.initModel),
		web.post('/registerClient', server.registerClient),
		web.get('/gatherAllParams', server.gatherAllParams),
		
		web.get('/sendParamsAll', server.sendParamsAll),

		web.post('/triggerTrainingAll', server.triggerTrainingAll),
		web.post('/triggerValidationAll', server.triggerValidationAll),

		web.post('/triggerValidationManual', server.triggerValidationManual),

		web.post('/initModelClients', server.initModelClients),

		web.post('/FedAVG', server.fedAVG),
		web.post('/FedDiverse', server.fedAVGDiverse),
		web.post('/FedWeighted', server.fedAVGWeighted),
	])
	web.run_app(app, port=8080)

if __name__ == '__main__':
	# environment variables pass through command line
	# in docker these are passed during container start
	uuid = int(argv[1])
	threadlimit = int(argv[2])
	seed = int(argv[3])
	print('Server: {0}'.format(uuid))
	main(uuid, threadlimit, seed)
