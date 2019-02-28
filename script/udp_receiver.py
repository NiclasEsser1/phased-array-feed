#!/usr/bin/env python

import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import struct

FITS_TIME_STAMP_LEN = 28
NCHAN  = 199584
NCHUNK = 231

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_address = ('134.104.70.90', 17106)
sock.bind(server_address)
while (1):
    data, server = sock.recvfrom(1<<16)
    print "The length of the packet is {} bytes.\n".format(len(data))
    
    nchan      = np.fromstring(data[8 + FITS_TIME_STAMP_LEN : 12 +FITS_TIME_STAMP_LEN], dtype='int32')[0]
    nchunk     = np.fromstring(data[28 + FITS_TIME_STAMP_LEN : 32 +FITS_TIME_STAMP_LEN], dtype='int32')[0]
    nchan_per_chunk = nchan/nchunk
    print nchan_per_chunk
    print nchan
    print nchunk
    
    unpack_data = struct.unpack("i28cfiffiiii{}f".format(nchan_per_chunk), data)
    print unpack_data
