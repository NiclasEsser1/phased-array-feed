#!/usr/bin/env python

import os

numa    = 1
hdir    = '/home/pulsar/'
ddir    = '/beegfs/DENG/docker'
uid     = 50000
gid     = 50000
dname   = 'paf-base'

os.system('./do_launch.py -a {:d} -b {:s} -c {:s} -d {:d} -e {:d} -f {:s}'.format(numa, ddir, hdir, uid, gid, dname))
