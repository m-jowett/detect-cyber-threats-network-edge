#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


import requests

# Provides nodes ability to post training/validation statistics to database.
# Also to initialise the indexes and data types for each set.

STATS_URL="http://fyp-opensearch:9200"

# if running simulations on host e.g. without docker replace above with localhost in order for nodes to access DB
#STATS_URL="http://localhost:9200"

# throw error if fail to post stats
# if deployed set to False, function of the model more important than stats
# implemented to prevent long simulations being run to find out they were not recorded
STATS_ERROR=True

# requests is used over aiohttp/asyncio (rest of implementation communication is asynchronous)
# https://docs.python-requests.org/en/latest/

# initiate indexes with data types
def init():
	# not using a range since the sets do not have to be numeric
	# numbers were used for simplicity only
	for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '20', '21', '22', '23', 'test']:
		
		# https://www.elastic.co/guide/en/elasticsearch/reference/current/sql-data-types.html

		# TRAINING index
		data = {
			"mappings": {
				"properties": {
					'uuid':            {'type': 'byte'},
					'epochs':          {'type': 'byte'},
					'sequence_length': {'type': 'byte'},
					'hidden_size':     {'type': 'byte'},
					'batch_size':      {'type': 'short'},
					'learn_rate':      {'type': 'float'},
					'total_nodes':     {'type': 'byte'},
					'total_rounds':    {'type': 'byte'},
					'time':            {'type': 'date', "format": "epoch_second"},
					'current_round':   {'type': 'byte'},
					'seed':            {'type': 'integer'},
					'device':          {'type': 'keyword'},
					'dataSize':        {'type': 'integer'},
					'train_loss':      {'type': 'double'},
					'attackSplit':     {'type': 'boolean'},
					'benignSplit':     {'type': 'boolean'},
					'fedMethod':       {'type': 'keyword'},
					'fedWeight':       {'type': 'keyword'},
					'longTail':        {'type': 'boolean'},
				}
			}
		}

		r = requests.put(
			url=STATS_URL + '/set'+ i +'_training',
			json=data,
		)
		print(r.status_code)
		print(r.json())

		# VALIDATION index
		data = {
			"mappings": {
				"properties": {
					'uuid':            {'type': 'byte'},
					'epochs':          {'type': 'byte'},
					'sequence_length': {'type': 'byte'},
					'hidden_size':     {'type': 'byte'},
					'batch_size':      {'type': 'short'},
					'learn_rate':      {'type': 'float'},
					'total_nodes':     {'type': 'byte'},
					'total_rounds':    {'type': 'byte'},
					'time':            {'type': 'date', "format": "epoch_second"},
					'current_round':   {'type': 'byte'},
					'seed':            {'type': 'integer'},
					'device':          {'type': 'keyword'},
					'dataSize':        {'type': 'integer'},
					'mergeComplete':   {'type': 'boolean'},
					'valid_loss':      {'type': 'double'},
					'accuracy':        {'type': 'double'},
					'precision':       {'type': 'double'},
					'f1':              {'type': 'double'},
					'recall':          {'type': 'double'},
					'dataset':         {'type': 'keyword'},
					'attackSplit':     {'type': 'boolean'},
					'benignSplit':     {'type': 'boolean'},
					'fedMethod':       {'type': 'keyword'},
					'fedWeight':       {'type': 'keyword'},
					'longTail':        {'type': 'boolean'},
				}
			}
		}

		r = requests.put(
			url=STATS_URL + '/set'+ i +'_validation',
			json=data,
		)

		print(r.status_code)
		print(r.json())
	
	return

# submit stats to server
# if indexes are not initiated then it will still pass
# but data types and time etc. will not be categorised
# data - dictionary key/value
def postStats(index, data):
	r = requests.post(
		url=STATS_URL + '/' + index + '/_doc/',
		json=data,
	)

	print('{0}: {1}'.format(index, r.status_code))
	
	if r.status_code != 200 and r.status_code != 201:
		print(r.json())
		if STATS_ERROR:
			raise ConnectionError('postStats non 200')
	
	return

# Run this script from command line to generate indexes

if __name__ == "__main__":
	print('Initiating indexes.')
	STATS_URL="http://localhost:9200"
	init()
