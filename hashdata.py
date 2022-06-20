#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


import pandas as pd
import torch
import numpy as np
from hashlib import sha256
from sys import argv
from pathlib import Path

# Produce SHA256 checksums of the various stored data types used throughout implementation
# used to check for reproducibility and during testing of features

# define the device tensors will be mapped to, since data from GPU node may have been saved
torchDevice = torch.device('cpu')

# https://stackoverflow.com/questions/49883236/how-to-generate-a-hash-or-checksum-value-on-python-dataframe-created-from-a-fix
# https://stackoverflow.com/questions/70386735/calculating-a-checksum-within-a-numpy-array

# generate SHA256sum of pandas dataframe
def hashDataframe(x):
	x = sha256(pd.util.hash_pandas_object(x).values).hexdigest()
	return(x)

# load pandas dataframe from disk then hash
def hashBinaryDataframe(filePath):
	x = pd.read_pickle(
		filepath_or_buffer=filePath,
	)
	x = hashDataframe(x)
	return(x)

# sha256sum of numpy ndarray
def hashNumpy(x):
	x.flags.writeable = False
	x = sha256(x.data).hexdigest()
	return(x)

# sha256sum of pytorch tensors
# equivalent to numpy ndarray, convert then hash
def hashTorch(x):
	x = x.numpy()
	x = hashNumpy(x)
	return(x)

# load PyTorch tensors from disk then hash
def hashFileTorch(filePath):
	x = torch.load(
		f=filePath,
		map_location=torchDevice
	)
	x = hashTorch(x)
	return(x)

# If run from command line, arguments:
# p - path to file to load in and checksum
# t - the type of file to be hashed, dictates how it will be handled
if __name__ == '__main__':
	p = Path(argv[1])
	t = argv[2]

	if t == 'torch':
		x = hashFileTorch(p)
	elif t == 'pandas':
		x = hashBinaryDataframe(p)
	else:
		print('ERROR: unrecognised')
		exit()
	
	print(x)
