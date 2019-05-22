#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os

source = "J1819-1458"
#source = "J0332+5434"
#source = "J1939+2134"
#source = "J1713+0747"

nchunk = 30
#nchunk = 48
#nchunk = 24
nchan  = 512
tsamp  = 216

ddir   = "/beegfs/DENG/AUG/{}".format(source)
cand_fname = "{}-{}chan-{}us-{}chunks.cand".format(source, nchan, tsamp, nchunk)
fil_fname  = "{}-{}chan-{}us-{}chunks.fil".format(source, nchan, tsamp, nchunk)

fil_fname  = os.path.join(ddir, fil_fname)
cand_fname = os.path.join(ddir, cand_fname)
cand       = np.loadtxt(cand_fname)
ncand      = len(cand)
shift      = 0.05

for i in range(ncand):
    tstart   = cand[i,2] - shift
    duration = 2 * shift

    dm       = cand[i,5]
    #dm       = 26.794137
    command  = "waterfaller.py {:s} -T {:f} -t {:f} -d {:f} --show-ts --show-spec".format(fil_fname, tstart, duration, dm)
    print command, ", SNR:", cand[i,0]
    os.system(command)
    
    #dm = 0
    #command = "waterfaller.py {:s} -T {:f} -t {:f} -d {:f} --show-ts --show-spec".format(fil_fname, tstart, duration, dm)
    #print command, ", SNR:", cand[i,0]
    #os.system(command)
    #
    #dm = 195.786
    #command = "waterfaller.py {:s} -T {:f} -t {:f} -d {:f} --show-ts --show-spec".format(fil_fname, tstart, duration, dm)
    #print command, ", SNR:", cand[i,0]
    #os.system(command)
