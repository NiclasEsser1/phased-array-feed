#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

repeat     = 1
nsamp      = 1
nchan      = 336 * 54
ndata      = nsamp * nchan 
hdrsize    = 4096
dsize      = 4
fdir  = "/beegfs/DENG/AUG/"
fname = '2018-08-30-19:37:27_0000000000000000.000000.dada'
fname = '2018-08-30-20:11:41_0000000000000000.000000.dada'
#fname = '2018-08-31-01:11:19_0000000000000000.000000.dada'
fname = 'J0332+5434-power.dada'
fname = 'J0332+5434-spectral.dada'

blksize = ndata * dsize
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + repeat * blksize)
sample  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
sample  = np.reshape(sample, (nsamp, nchan))
sample = np.mean(sample, axis = 0)

f.close()
plt.figure()
plt.plot(sample)
plt.show()
