#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

ddir      = "/beegfs/DENG/AUG"
fname     = "2018-08-30-20:11:41_0000000000000000.000000.dada"
scl_fname = "2018-08-30-20:11:41_scale.txt"
scale     = np.loadtxt("{:s}/{:s}".format(ddir, scl_fname))

nchan      = 1024
#nsamp_seek = 1000000
nsamp_seek = 0
nsamp      = 1000
hdrsz      = 4096
fname      = "{:s}/{:s}".format(ddir, fname)
f          = open(fname, "r")

f.seek(hdrsz + nchan * nsamp_seek)

sample = np.array(np.fromstring(f.read(nchan * nsamp), dtype='uint8'))
sample = np.reshape(sample, (nsamp, nchan))

print np.std(sample[:,1])
plt.figure()
plt.plot(sample[0,:] * scale[:,1] + scale[:,0])
plt.show()

f.close()
