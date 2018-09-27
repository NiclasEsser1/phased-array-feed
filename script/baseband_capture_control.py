#!/usr/bin/env python

import socket
import os
import time
import parser
import argparse
import threading

parser = argparse.ArgumentParser(description='To control data capture of different parts')
parser.add_argument('-a', '--src_name', type=str, nargs='+',
                    help='The name of source')    
args     = parser.parse_args()
src_name = args.src_name[0]

length = 600
beam   = 0

node_id     = socket.gethostname()[-1]
part        = int(node_id)

J1939info = "PSR J1939+2134:19 39 38.5612138:+21 34 59.12627"
J0332info = "PSR J0332+5434:03:32:59.3679935:54:34:43.570012"
J1819info = "RRAT J1819-1458:18 19 34.173:-14 58 03.57"
J1713info = "PSR J1713+0747:17 13 49.53053:+07 47 37.5264"

if src_name == "J1939+2134":
    info = J1939info
if src_name == "J0332+5434":
    info = J0332info
if src_name == "J1819-1458":
    info = J1819info
if src_name == "J1713+0747":
    info = J1713info

address = "capture.beam{:02d}part{:02d}.socket".format(beam, part)

start_buf = 0

sock      = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
command   = "START-OF-DATA:{:s}:{:d}".format(info, start_buf)    
print command
sock.sendto("{:s}\n".format(command), address)

print "going to sleep for {:d} seconds\n".format(length)
time.sleep(length)
sock.sendto("END-OF-DATA\n", address)

sock.close()
    
