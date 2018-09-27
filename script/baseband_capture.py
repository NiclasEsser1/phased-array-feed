#!/usr/bin/env python

from pprint import pprint
from pssh.clients.native import ParallelSSHClient
import os, socket

hosts = ['pacifix0', 'pacifix1', 'pacifix2', 'pacifix3', 'pacifix4', 'pacifix5']
#hosts = ['pacifix2', 'pacifix3']
client = ParallelSSHClient(hosts)

output = client.run_command('/home/pulsar/xinping/phased-array-feed/script/baseband_capture_entry.py')
for host, host_output in output.items():
    for line in host_output.stdout:
        print(line)
