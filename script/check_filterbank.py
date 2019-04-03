#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

nsamp_seek = 0
nsamp      = 50

hdrsize    = 4096
dsize      = 1

bw         = 512
nchan_chan = 1
c_freq     = 1337.0

nchan_band = int(bw * nchan_chan)
ndata      = nsamp * nchan_band
freq0      = c_freq - bw/2.0
freq       = freq0 + np.arange(int(nchan_band))/float(nchan_chan)

fdir  = "/beegfs/DENG/beam00/"
fname = "2019-03-27-07:52:11_0000000000000000.000000.dada"

blksize = ndata * dsize
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seek * nchan_band)
sample  = np.array(np.fromstring(f.read(blksize), dtype='uint8'))
sample  = np.reshape(sample, (nsamp, nchan_band))

plt.figure()
for i in range(nsamp):
    plt.plot(sample[i,:])
plt.show()

f.close()
