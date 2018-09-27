#!/usr/bin/env python

from pprint import pprint
from pssh.clients.native import ParallelSSHClient
import os, socket, argparse, parser

parser = argparse.ArgumentParser(description='To transfer data from shared memeory to disk with a docker container')
parser.add_argument('-a', '--src_name', type=str, nargs='+',
                    help='The name of source')    

args     = parser.parse_args()
src_name = args.src_name[0]

hosts = ['pacifix0', 'pacifix1', 'pacifix2', 'pacifix3', 'pacifix4', 'pacifix5']
client = ParallelSSHClient(hosts)

output = client.run_command('/home/pulsar/xinping/phased-array-feed/script/baseband_dada_dbdisk_entry.py -a {:s}'.format(src_name))
for host, host_output in output.items():
    for line in host_output.stdout:
        print(line)

