#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

beam_id = 0

fdir  = "/beegfs/DENG/beam{:02d}".format(beam_id)
fname = "2019-03-26-18:13:35_0000000000000000.000000.dada"
fname = "2019-03-26-18:41:11_0000000000000000.000000.dada"
fname = "2019-03-26-18:47:11_0000000000000000.000000.dada"
fname = "2019-03-26-18:50:52_0000000000000000.000000.dada"
#fname = "2019-03-27-10:50:15_0000000000000000.000000.dada"
#fname = "2019-03-27-05:39:41_0000000000000000.000000.dada"
#fname = "2019-03-27-03:56:36_0000000000000000.000000.dada"
#fname = "2019-03-27-03:44:29_0000000000000000.000000.dada"
#fname = "2019-03-27-03:01:52_0000000000000000.000000.dada"
#fname = "2019-03-27-02:51:22_0000000000000000.000000.dada"
fname = "2019-03-24-13:34:03_0000000000000000.000000.dada"

fname = os.path.join(fdir, fname)

nchan = 512
hdrsz = 4096
dsize = 1
nsamp = 10000
nsamp_seek = 0 #162000
blksize    = dsize * nchan

f = open(fname, "r")
f.seek(hdrsz + nsamp_seek * blksize)
while True:
    data = np.array(np.fromstring(f.read(nsamp * blksize), dtype='uint8'))
    data  = np.reshape(data, (nsamp, nchan)).T

    plt.figure()
    plt.imshow(data, aspect='auto',interpolation='none',cmap='binary')
    #plt.imshow(data, aspect='auto',interpolation='none')
    plt.show()

f.close()
