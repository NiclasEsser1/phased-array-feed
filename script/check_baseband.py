#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

nsamp_seek = 0
nsamp      = 1

hdrsize    = 4096
dsize      = 1
ndata_samp = 4
nchan      = 336

ndata      = ndata_samp * nsamp * nchan
freq0      = 1340.5 - nchan/2.0
freq       = freq0 + np.arange(int(nchan))

fdir  = "/beegfs/DENG/pacifix8_numa0_process1/"
fdir  = "/beegfs/DENG/pacifix7_numa0_process0/"
fname = "2018-08-30-19:37:27_0000000000000000.000000.dada"
fname = "2018-08-31-01:11:19_0000000000000000.000000.dada"
#fname = "2018-08-30-20:11:41_0000000000000000.000000.dada"

blksize = ndata * dsize
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seek * blksize)# * ndata_samp)
sample  = np.array(np.fromstring(f.read(blksize), dtype='int8'))
sample  = np.reshape(sample, (ndata_samp, nchan))

plt.figure()
plt.plot(freq, sample[0,:])
plt.plot(freq, sample[1,:])
plt.plot(freq, sample[2,:])
plt.plot(freq, sample[3,:])
plt.show()

f.close()
