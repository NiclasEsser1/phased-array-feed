#!/usr/bin/env python

import socket
import os
import time
import parser
import argparse

# ./control.py -a 0 -b 0 -c START-OF-DATA:0:0
# ./control.py -a 0 -b 0 -c END-OF-DATA
# ./control.py -a 0 -b 0 -c STATUS-OF-TRAFFIC
# ./control.py -a 0 -b 0 -c END-OF-CAPTURE

parser = argparse.ArgumentParser(description='To control the capture')

parser.add_argument('-a', '--beam', type=int, nargs='+',
                    help='beam to control')
parser.add_argument('-b', '--length', type=int, nargs='+',
                    help='length for valid data in seconds')

args    = parser.parse_args()
beam    = args.beam[0]
length  = args.length[0]

address = "capture.socket{:d}".format(beam)

start_buf = 0
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

command = "START-OF-DATA:PSR J1939+2134:06 05 56.34:+23 23 40.00:{:d}".format(start_buf)
print command
sock.sendto("{:s}\n".format(command), address)

print "sleep for {:d} seconds ...\n".format(length)
time.sleep(length)

command = "END-OF-DATA"
print command
sock.sendto("{:s}\n".format(command), address)

sock.close()
