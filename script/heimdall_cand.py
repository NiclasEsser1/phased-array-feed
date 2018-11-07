#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

ddir  = "/beegfs/DENG/AUG"

#cand_fname = "cand-J1819-1458-zap0511-snr10.txt"
#cand_fname = "cand-J1819-1458-snr10.txt"
#cand_fname = "cand-J1819-1458-snr10-swap.txt"
#cand_fname = "cand-J1819-1458-zap0511-snr10-swap.txt"
#fil_fname  = "J1819-1458-1024chan-64scl.fil"

#cand_fname = "cand-J0332+5434-zap0511-snr10.txt"
#fil_fname  = "J0332+5434-1024chan-64scl.fil"
#cand_fname = "cand-J1939+2134-zap0511-snr10.txt"
#fil_fname  = "J1939+2134-1024chan-64scl.fil"

#cand_fname = "cand-J1819-1458-zap5121023-snr10-swap.txt"
cand_fname = "cand-J1819-1458-zap5121023-zap304310-snr10-swap.txt"
fil_fname  = "J1819-1458-1024chan-64scl-swap.fil"

#cand_fname = "cand.txt"
#fil_fname  = "J1819-1458-1024chan-64scl.fil"

cand_fname = "{:s}/{:s}".format(ddir, cand_fname)
cand       = np.loadtxt(cand_fname)
sample0    = cand[:,7]
sample1    = cand[:,8]
ncand      = len(cand)
tsamp      = 54.0E-6

fil_fname  = "{:s}/{:s}".format(ddir, fil_fname)

hdrsz      = 342
nchan      = 1024
dsize      = 1

for i in range(ncand):
    print cand[i,5]
    fil = open(fil_fname, "r")
    fil.seek(hdrsz + nchan * (int(sample0[i]) - 1))
    nsamp  = int(sample1[i] - sample0[i] + 3000)
    sample = np.array(np.fromstring(fil.read(nchan * nsamp), dtype='uint8'))
    sample = np.reshape(sample, (nsamp, nchan)).T#[0:512, :]#[512:-1, :]
    fil.close()

    #yticks = np.array(range(nsamp)) * tsamp
    
    plt.figure()
    plt.imshow(sample, interpolation = 'hanning', aspect="auto", origin = 'lower')
    #plt.yticks(yticks)
    #plt.imshow(sample, interpolation = 'hanning', aspect="auto")
    plt.show()
