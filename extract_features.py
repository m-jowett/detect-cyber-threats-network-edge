#!/usr/bin/python3

# m-jowett/detect-cyber-threats-network-edge
# 04/05/2022


from nfstream import NFStreamer
from pathlib import Path

# https://www.nfstream.org/docs/api
# https://docs.python.org/3.9/library/pathlib.html

# load in configuration
from json import load as json_load
configData = {}
with open(Path('config.json'), 'r') as f:
	configData = json_load(f)

rawPath    = Path(configData['default']['pcapPath'])
outputPath = Path(configData['default']['featurePath'])

# generate list of paths to PCAP files for processing

pcapList = configData['data']['datasets']
for i in range(len(pcapList)):
	pcapList[i].append(Path(rawPath, pcapList[i][0]).with_suffix('.pcap'))


# For CICDDoS2019 some PCAP files contain multiple sets, see dataset processing chapter of report
# times from development/cicddos_dates_filter.ipynb
# bidirectional_first_seen_ms
# bidirectional_last_seen_ms
# These were listed retrospectively, please verify if reproducing
filterList = {
	'cicddos2019_day1_NetBIOS': [1520773200.0, 1520773740.0],
	'cicddos2019_day1_LDAP':    [1520774460.0, 1520775000.0],
	'cicddos2019_day1_UDP':     [1520776380.0, 1520776980.0],
	'cicddos2019_day1_SYN':     [1520778480.0, 1520800500.0],
	'cicddos2019_day2_NTP':     [1515767700.0, 1515768300.0],
	'cicddos2019_day2_DNS':     [1515768720.0, 1515769500.0],
	'cicddos2019_day2_LDAP':    [1515770520.0, 1515771120.0],
	'cicddos2019_day2_NetBIOS': [1515772200.0, 1515772800.0],
	'cicddos2019_day2_UDP':     [1515775500.0, 1515776940.0],
	'cicddos2019_day2_SYN':     [1515778140.0, 1515778440.0],
}

for p in pcapList:
	# name, sentiment, path to PCAP

	print(p[2])

	# output path for pickled dataframe
	np = Path(outputPath, p[2].with_suffix('.pkl').name)

	print('-> {0}'.format(np))

	# feature extraction of PCAP
	# to pandas dataframe

	# NFStream documentation referenced in selecting parameters of streamer
	# https://www.nfstream.org/docs/api
	# the PCAP file can be replaced with a live network device if required

	streamer = NFStreamer(
		source=p[2],
		accounting_mode=1, # IP - sizes etc
		statistical_analysis=True, # in retrospect analyse streams etc
		n_dissections=0, # disable L7 (application layer) features
	).to_pandas()

	if p[0] in filterList.keys():
		print('Time filtering needed.')
		s = filterList[p[0]][0]
		e = filterList[p[0]][1]
		streamer = streamer[streamer['bidirectional_first_seen_ms'] >= s]
		streamer = streamer[streamer['bidirectional_last_seen_ms'] <= e]

	print('To disk')

	streamer.to_pickle(
		path=np,
	)
	

