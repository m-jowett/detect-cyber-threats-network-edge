#!/bin/bash

# Some commands used in pre-processing sets of PCAP files, for reference in reproducing datasets

# https://www.wireshark.org/docs/man-pages/mergecap.html

# combine multiple PCAPs
# 	mergecap *.pcap -w "out.pcap"

# get the frame time (UNIX) of the first and last packet in file
# 	capinfos -a -e -r -T "$f"

