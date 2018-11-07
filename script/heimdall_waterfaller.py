#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os

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

fil_fname  = "{:s}/{:s}".format(ddir, fil_fname)
cand_fname = "{:s}/{:s}".format(ddir, cand_fname)
cand       = np.loadtxt(cand_fname)
ncand      = len(cand)

for i in range(ncand):
    tstart   = cand[i,2]
    duration = 0.2
    #print "waterfaller.py {:s} -T {:f} -t {:f} --colour-map=\"Blues\" --show-colour-bar".format(fil_fname, tstart, duration)
    #os.system("waterfaller.py {:s} -T {:f} -t {:f} --colour-map=\"Blues\" --show-colour-bar".format(fil_fname, tstart, duration))

    print "waterfaller.py {:s} -T {:f} -t {:f}".format(fil_fname, tstart, duration)
    os.system("waterfaller.py {:s} -T {:f} -t {:f}".format(fil_fname, tstart, duration))
