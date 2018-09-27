#!/usr/bin/env python

import socket, os

node_id     = socket.gethostname()[-1]
command = "/home/pulsar/xinping/phased-array-feed/script/capture.py -a /home/pulsar/xinping/phased-array-feed/config/system.conf -b /home/pulsar/xinping/phased-array-feed/config/pipeline.conf -c 0 -d {:s} -e 1 -f 1".format(node_id)
print command
os.system(command)
