#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

nsamp_seak = 10
nsamp      = 1

hdrsize    = 4096
dsize      = 4

bw         = 336.0
#nchan_1mhz = 54
nchan_1mhz = 864
c_freq     = 1340.5

nchan_bw   = int(bw * nchan_1mhz)
ndata      = nsamp * nchan_bw
freq0      = 1340.5 - bw/2.0
freq       = freq0 + np.arange(int(nchan_bw))/float(nchan_1mhz)

#fdir  = "/beegfs/DENG/AUG/"
#fname = 'J0332+5434-spectral.dada'
#fname = 'J1939+2134-spectral.dada'
#fname = 'J1713+0747-spectral.dada'
#fname = 'J1819-1458-spectral.dada'

fdir  = "/beegfs/DENG/pacifix8_numa0_process0/"
fname = "2018-08-30-19:37:27_0000000000000000.000000.dada"

blksize = int(ndata * dsize)
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seak * blksize * 4)
sample  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
#sample  = np.reshape(sample, (nsamp, nchan_bw))
#sample = np.mean(sample, axis = 0)

plt.figure()
#plt.plot(freq, sample)
plt.plot(sample)
plt.show()

f.close()
