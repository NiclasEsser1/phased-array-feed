#!/usr/bin/env python

import socket
import os
import time
import parser
import argparse

parser = argparse.ArgumentParser(description='To control the capture')

parser.add_argument('-a', '--command', type=str, nargs='+',
                    help='Command to send')

args    = parser.parse_args()
command = args.command[0]

server_address = "/tmp/capture_socket"
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.sendto("{:s}:0:0\n".format(command), server_address)

sock.close()
