#!/usr/bin/env python

import socket, os, argparse, parser

parser = argparse.ArgumentParser(description='To transfer data from shared memeory to disk with a docker container')
parser.add_argument('-a', '--src_name', type=str, nargs='+',
                    help='The name of source')    

args     = parser.parse_args()
src_name = args.src_name[0]

node_id     = socket.gethostname()[-1]
directory = "/beegfs/DENG/AUG/baseband/{:s}/{:s}".format(src_name, node_id)
command = "/home/pulsar/xinping/phased-array-feed/script/dada_dbdisk.py -a /home/pulsar/xinping/phased-array-feed/config/pipeline.conf -b 0 -c {:s} -d {:s}".format(node_id, directory)
print command
os.system(command)
