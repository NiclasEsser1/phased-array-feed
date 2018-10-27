#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

ddir      = "/beegfs/DENG/AUG"
#fname     = "2018-08-30-20:11:41_0000000000000000.000000.dada"
#scl_fname = "2018-08-30-20:11:41_scale.txt"
#fname     = "2018-08-31-01:11:19_0000000000000000.000000.dada"
#scl_fname = "2018-08-31-01:11:19_scale.txt"

#fname     = "2018-08-30-20:11:41_0000000000000000.filterbank.dada.4"
#scl_fname = "2018-08-30-20:11:41_scale.txt"

#fname     = "I_58295.92240879_beam_12_8bit.fil"
#scl_fname = "I_58295.92240879_beam_12_8bit.fil.scale"

fname     = "J0332+5434-512chan-64scl.fil"
scl_fname = "J0332+5434-512chan-64scl.txt"

scale     = np.loadtxt("{:s}/{:s}".format(ddir, scl_fname))

nchan      = 512
nsamp_seek = 3000000
#nsamp_seek = 10000000
#nsamp_seek = 0
nsamp      = 1000
#hdrsz      = 4096
hdrsz     = 342
fname      = "{:s}/{:s}".format(ddir, fname)
f          = open(fname, "r")

f.seek(hdrsz + nchan * nsamp_seek)

sample = np.array(np.fromstring(f.read(nchan * nsamp), dtype='uint8'))
sample = np.reshape(sample, (nsamp, nchan))
plt.figure()
plt.plot(sample[:,256])
plt.show()

print np.std(sample[:,0]), np.mean(sample[:,0])
sample = np.mean(sample, axis = 0)

plt.figure()
plt.plot(sample * scale[:,1] + scale[:,0])
plt.show()

f.close()
