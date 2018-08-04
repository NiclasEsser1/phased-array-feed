#!/usr/bin/env python

import socket
import os

server_address = "/tmp/capture_socket"
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.sendto("MONITOR, HELLO!\n", server_address)
sock.close()
