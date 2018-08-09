#!/usr/bin/env python

import os

length  = 20
numa    = 1
#memsize = 80000000000
memsize = 100000000000
hdir    = '/home/pulsar/'
ddir    = '/beegfs/DENG/docker'
uid     = 50000
gid     = 50000
dname   = 'paf-base'

os.system('./do_launch.py -a {:f} -b {:d} -c {:d} -d {:s} -e {:s} -f {:d} -g {:d} -i {:s}'.format(length, numa, memsize, ddir, hdir, uid, gid, dname))
