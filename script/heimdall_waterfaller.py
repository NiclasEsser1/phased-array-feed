#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os

ddir  = "/beegfs/DENG/AUG"
#ddir  = "./"

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
mask_fname = "J1819-1458-1024chan-64scl-swap_rfifind.mask"

#cand_fname = "cand.txt"
#fil_fname  = "J1819-1458-1024chan-64scl.fil"

fil_fname  = os.path.join(ddir, fil_fname)
cand_fname = os.path.join(ddir, cand_fname)
mask_fname = os.path.join(ddir, mask_fname)
cand       = np.loadtxt(cand_fname)
ncand      = len(cand)
tsamp      = 54.0E-6
dm         = 196.0

for i in range(ncand):
    tstart   = cand[i,2] - 0.2
    duration = 0.4

    command = "waterfaller.py {:s} -T {:f} -t {:f} -d {:f} --show-ts --show-spec --bandpass --maskfile=\"{:s}\"".format(fil_fname, tstart, duration, dm, mask_fname)
    print command, ", SNR:", cand[i,0]
    os.system(command)
    exit(1)
