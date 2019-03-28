#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import sys
import numpy.fft as fft

nsamp_seek = 0
nsamp      = 10

hdrsize    = 4096
dsize      = 4
ndata_samp = 4

bw         = 231.0
bw         = 336.0
nchan_chan = 864
c_freq     = 1340.5
#c_freq     = 1337

nchan_band = int(bw * nchan_chan)
ndata      = ndata_samp * nsamp * nchan_band
freq0      = c_freq - bw/2.0
freq       = freq0 + np.arange(int(nchan_band))/float(nchan_chan)

#fdir  = "/beegfs/DENG/pacifix8_numa0_process1/"
fdir  = "/beegfs/DENG/pacifix8_numa0_process0/"
#fdir  = "/beegfs/DENG/pacifix7_numa0_process0/"
#fdir  = "/beegfs/DENG/pacifix0_numa0_process1/"
#fdir  = "/beegfs/DENG/beam00/"
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
fname = "dump_beam00.bin"
fname = "2018-08-31-01:11:19_0000000000000000.000000.dada.before"
#fname = "2018-08-31-01:11:19_0000000000000000.000000.dada"
#fname = "2019-03-27-10:50:15_0000000000000000.000000.dada"

blksize = ndata * dsize
fname   = os.path.join(fdir, fname)
f       = open(fname, "r")
f.seek(hdrsize + nsamp_seek * nchan_band * ndata_samp)# * ndata_samp)
sample  = np.array(np.fromstring(f.read(blksize), dtype='float32'))
sample  = np.reshape(sample, (nsamp, ndata_samp, nchan_band))

plt.figure()
result = []
#extra = sample[0,0,171000:173000]
extra = sample[0,0]
for i in range(nsamp - 1):
    #samp = sample[i+1,0,171000:173000] - sample[i,0,171000:173000] - 0.5 * extra
    samp = sample[i+1,0] - sample[i,0] - 0.5 * extra
    result.append(samp)
    extra += samp
    plt.plot(samp)
plt.show()
plt.figure()
result = np.array(result)
np.savetxt("result1.txt", result)

#result = []
#plt.figure()
#for i in range(nsamp - 1):
#    result.append(sample[i+1,0])
#    plt.plot(sample[i,0])
#plt.show()
#result = np.array(result)
#np.savetxt("result2.txt", result)

f.close()
