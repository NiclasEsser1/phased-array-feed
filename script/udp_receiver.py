#!/usr/bin/env python

import socket
import sys
import numpy as np
import matplotlib.pyplot as plt

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('134.104.70.90', 17106)
#server_address = ("10.17.0.2", 17100)
sock.bind(server_address)
while (1):
    data, server = sock.recvfrom(1<<16)
    print len(data)
    
    nchan    = np.fromstring(data[0 : 4], dtype='int32')
    flux     = np.fromstring(data[4 : (nchan[0] * 4 + 4)], dtype='float32')
    tsamp    = np.fromstring(data[(nchan[0] * 4 + 4) : (nchan[0] * 4 + 8)], dtype='float32')
    utc_time = np.fromstring(data[(nchan[0] * 4 + 8) : (nchan[0] * 4 + 36)], dtype='c')
    beam     = np.fromstring(data[(nchan[0] * 4 + 36) : (nchan[0] * 4 + 40)], dtype='int32')    
    ra       = np.fromstring(data[(nchan[0] * 4 + 40) : (nchan[0] * 4 + 44)], dtype='float32')
    dec      = np.fromstring(data[(nchan[0] * 4 + 44) : (nchan[0] * 4 + 48)], dtype='float32')
    mjd_i    = np.fromstring(data[(nchan[0] * 4 + 48) : (nchan[0] * 4 + 52)], dtype='int32')
    mjd_f    = np.fromstring(data[(nchan[0] * 4 + 52) : (nchan[0] * 4 + 56)], dtype='float32')
    az       = np.fromstring(data[(nchan[0] * 4 + 56) : (nchan[0] * 4 + 60)], dtype='float32')
    el       = np.fromstring(data[(nchan[0] * 4 + 60) : (nchan[0] * 4 + 64)], dtype='float32')
    
    print nchan[0], tsamp[0], utc_time, beam[0], ra[0], dec[0], mjd_i[0], mjd_f[0], az[0], el[0]
    
    plt.figure()
    plt.plot(flux)
    plt.show()
