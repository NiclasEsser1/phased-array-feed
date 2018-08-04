#!/usr/bin/env python

import socket
import os
import time

server_address = "/tmp/capture_socket"
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#sock.sendto("START-OF-DATA:0:0\n", server_address)
#sock.sendto("END-OF-DATA\n", server_address)

sock.sendto("END-OF-CAPTURE\n", server_address)
#sock.sendto("STATUS-OF-TRAFFIC\n", server_address)
sock.close()
