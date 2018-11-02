#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

ddir      = "/beegfs/DENG/AUG"

fname     = "J0332+5434-512chan-64scl.fil"
scl_fname = "J0332+5434-512chan-64scl.txt"

scale      = np.loadtxt("{:s}/{:s}".format(ddir, scl_fname))
nchan      = 512
nsamp_seek = 30000
nsamp      = 1
hdrsz      = 342
fname      = "{:s}/{:s}".format(ddir, fname)

#f          = open(fname, "r")
#
#f.seek(hdrsz + nchan * nsamp_seek)
#
#sample = np.array(np.fromstring(f.read(nchan * nsamp), dtype='uint8'))
#sample = np.reshape(sample, (nsamp, nchan))
##plt.figure()
##plt.plot(sample[:,511])
##plt.show()
#
#print np.std(sample[:,0]), np.mean(sample[:,0])
#mean_all = np.mean(sample, axis = 0)
#std_all  = np.std(sample, axis = 0)
#sample = np.mean(sample, axis = 0)
#
#f.close()
#
#plt.figure()
#plt.subplot(3,1,1)
#plt.plot(sample * scale[:,1] + scale[:,0])
##plt.show()
#
##plt.figure()
#plt.subplot(3,1,2)
#plt.plot(mean_all)
##plt.show()
#
##plt.figure()
#plt.subplot(3,1,3)
#plt.plot(std_all)
#plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(scale[:,0])
plt.subplot(2,1,2)
plt.plot(scale[:,1])
plt.show()
