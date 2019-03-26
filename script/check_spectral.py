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
dsize      = 4
ndata_samp = 4

bw         = 336.0
#bw         = 14.0
bw         = 231
#bw         = 35
nchan_chan = 864
#c_freq     = 1340.5
c_freq     = 1337
#c_freq     = 1421.

nchan_band = int(bw * nchan_chan)
ndata      = ndata_samp * nsamp * nchan_band
freq0      = c_freq - bw/2.0
freq       = freq0 + np.arange(int(nchan_band))/float(nchan_chan)

#fdir  = "/beegfs/DENG/pacifix8_numa0_process1/"
#fdir  = "/beegfs/DENG/pacifix8_numa0_process0/"
#fdir  = "/beegfs/DENG/pacifix7_numa0_process0/"
#fdir  = "/beegfs/DENG/pacifix0_numa0_process1/"
fdir  = "/beegfs/DENG/beam08/"
#fname = "2018-08-30-19:37:27_0000000000000000.000000.dada"
#fname = "2018-08-31-01:11:19_0000000000000000.000000.dada"
#fname = "2018-08-30-20:11:41_0000000000000000.000000.dada"
fname = "2019-03-25-20:07:49_0000000000000000.000000.dada" # DATA from on-sky, frequency is not right; beam00 and beam01, 0, 0;
#fname = "2019-03-25-20:23:32_0000000000000000.000000.dada"  # DATA from on-sky, beam10 and beam11, +1 -1;
fname = "2019-03-25-20:34:29_0000000000000000.000000.dada"  # DATA from on-sky, beam10 and beam11, +1, +1;
fname = "2019-03-26-10:47:13_0000000000000000.000000.dada"   # DATA from on-sky, beam05, full band
fname = "2019-03-26-10:55:41_0000000000000000.000000.dada"   # DATA from on-sky, beam10 and beam11;
fname = "2019-03-26-11:10:05_0000000000000000.000000.dada"
fname = "2019-03-26-13:51:59_0000000000000000.000000.dada"   # DATA from on-sky, beam09 and beam08, after capture fix, 33 chunks;
#fname = "2019-03-26-13:54:21_0000000000000000.000000.dada"   # DATA from on-sky, beam09 and beam04, after capture fix, 48 chunks;

blksize = ndata * dsize
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seek * blksize)# * ndata_samp)
sample  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
sample  = np.reshape(sample, (ndata_samp, nchan_band))

plt.figure()
#plt.plot(freq, sample[0,:])
plt.plot(freq, sample[1,:])
#plt.plot(freq, sample[2,:])
#plt.plot(freq, sample[3,:])
plt.show()

f.close()
