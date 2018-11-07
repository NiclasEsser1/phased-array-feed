#!/usr/bin/env python

import os, argparse

parser = argparse.ArgumentParser(description='Launch container for PAF pipeline development')
parser.add_argument('-a', '--numa_id', type=int, nargs='+',
                    help='The ID of NUMA node')

args    = parser.parse_args()
numa_id = args.numa_id[0]
hdir    = '/home/pulsar/'
ddir    = '/beegfs/'
uid     = 50000
gid     = 50000
dname   = 'paf-base'

os.system('./do_launch.py -a {:d} -b {:s} -c {:s} -d {:d} -e {:d} -f {:s}'.format(numa_id, ddir, hdir, uid, gid, dname))
