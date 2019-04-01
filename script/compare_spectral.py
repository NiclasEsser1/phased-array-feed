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
dsize      = 4
ndata_samp = 4

#bw         = 336.0
bw         = 231
nchan_chan = 864
#c_freq     = 1340.5
c_freq     = 1337

nchan_band = int(bw * nchan_chan)
ndata      = ndata_samp * nsamp * nchan_band
freq0      = c_freq - bw/2.0
freq       = freq0 + np.arange(int(nchan_band))/float(nchan_chan)

#fdir  = "/beegfs/DENG/pacifix8_numa0_process0/"
blksize = ndata * dsize

# First file wrong data
fdir  = "/beegfs/DENG/beam00/"
fname   = "2019-03-27-10:50:15_0000000000000000.000000.dada"#"2018-08-31-01:11:19_0000000000000000.000000.dada.before"
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seek * nchan_band * ndata_samp)# * ndata_samp)
sample  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
sample  = np.reshape(sample, (nsamp, ndata_samp, nchan_band))
f.close()

# Second file right data
fdir  = "/beegfs/DENG/beam01/"
fname   = "2019-03-27-10:50:15_0000000000000000.000000.dada"#"2018-08-31-01:11:19_0000000000000000.000000.dada"
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seek * nchan_band * ndata_samp)# * ndata_samp)
sample1  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
sample1  = np.reshape(sample1, (nsamp, ndata_samp, nchan_band))
f.close()

extra = sample[0,0]
extra1 = sample1[0,0]
for i in range(nsamp - 1):
    plt.figure()
    samp = sample[i+1,0] - sample[i,0] - 0.5 * extra
    extra += samp

    samp1 = sample1[i+1,0] - sample1[i,0] - 0.5 * extra1
    extra1 += samp1
    
    plt.plot(freq, samp)
    plt.plot(freq, samp1)
    plt.show()
