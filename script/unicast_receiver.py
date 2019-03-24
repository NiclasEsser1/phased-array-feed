#!/usr/bin/env python

import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import struct

FITS_TIME_STAMP_LEN = 28

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

#server_address = ('134.104.70.95', 17106)
server_address = ('134.104.70.95', 17110)
sock.bind(server_address)
data, server = sock.recvfrom(1<<16)
nchan      = np.fromstring(data[8 + FITS_TIME_STAMP_LEN : 12 +FITS_TIME_STAMP_LEN], dtype='int32')[0]
nchunk     = np.fromstring(data[28 + FITS_TIME_STAMP_LEN : 32 +FITS_TIME_STAMP_LEN], dtype='int32')[0]
nchan_per_chunk = nchan/nchunk
print "The length of the packet is {} bytes.\n".format(len(data))
print "NCHAN is", nchan
print "NCHAN_PER_CHUNK is", nchan_per_chunk
print "NCHUNK is", nchunk

unpack_data = struct.unpack("i28cfiffiiii{}f".format(nchan_per_chunk), data)
print unpack_data[0:37]
spectral = unpack_data[37:-1]
#print spectral
#plt.figure()
#plt.plot(spectral)
#plt.show()
#
while (1):
    data, server = sock.recvfrom(1<<16)
    unpack_data = struct.unpack("i28cfiffiiii{}f".format(nchan_per_chunk), data)
    print unpack_data[0:37]
    spectral = unpack_data[37:-1]
    #print spectral
    #plt.figure()
    #plt.plot(spectral)
    #plt.show()
