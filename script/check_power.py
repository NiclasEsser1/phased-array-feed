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
#plt.subplot(2,1,1)
plt.plot(sample[0:-1:54])

nchan      = 336
ndata      = nsamp * nchan 
hdrsize    = 4096
dsize      = 4
fdir  = "/beegfs/DENG/AUG/"
fname = 'J0332+5434-power.dada'

blksize = ndata * dsize
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + repeat * blksize)
sample  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
sample  = np.reshape(sample, (nsamp, nchan))
sample = np.mean(sample, axis = 0)

#plt.subplot(2,1,2)
plt.plot(5*sample, '--')
plt.show()
