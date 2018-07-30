#!/usr/bin/env python

import os
import socket
import struct
import json
import sys
import subprocess

def multicast2obsinfo(beam):
    # Multicast group, port and address
    MCAST_GRP  = '224.1.1.1'
    MCAST_PORT = 5007
    MCAST_ADDR = ('', MCAST_PORT)
    
    # Create the socket and get ready to receive data
    sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create the socket
    group = socket.inet_aton(MCAST_GRP)
    mreq  = struct.pack('4sL', group, socket.INADDR_ANY)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(MCAST_ADDR)                                    # Bind to the server address
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)  # Tell the operating system to add the socket to the multicast group on all interfaces.
    
    # Get data packet
    pkt, address = sock.recvfrom(1<<16)
    data         = json.loads(pkt)
    
    # Get center frequency in MHz
    freq = float(data['sky_frequency'])
    
    # Get target_name 
    if beam == 0: # If we capture data from boresight beam (beam 0), we use the target name from metadata
        target_name = data['target_name']
        if target_name == ' ':
            print "Can not get target name from metadata ..."
    else:         # If we capture data from other beams, we set the target name to be empty
        target_name = ' '
    print "The target name is {:s}".format(target_name)
    
    # RA and DEC
    ra  = float(data['beams_direction']['beam{:02d}'.format(beam + 1)][0]) # RA in decimal radian, we may need to convert it to other unit
    dec = float(data['beams_direction']['beam{:02d}'.format(beam + 1)][1]) # DEC in decimal radian, we may need to convert it to other unit
    print "RA and DEC in decimal radian are {:f} and {:f}".format(ra, dec)

    # check routing table
    routing_table = data['routing_table']
    address       = []
    for table_line in routing_table:
        ip   = table_line.split(',')[3 * beam + 2]
        port = table_line.split(',')[3 * beam + 3]
        address.append("{:s}:{:s}".format(ip, port))
        
    # Add chunk numbers to address
    address       = {x:address.count(x) for x in address}
    address_chunk = []
    node          = []
    for i in range(len(address)):
        address_chunk.append("{:s}:{:d}".format(address.keys()[i], address.values()[i]))
        node.append(float(address.keys()[i].split('.')[2]))
    print "We receive data from beam {:d} with {:s} (ip:port:number-of-frequency-chunks)".format(beam, sorted(address_chunk))

    # Check if we receive data on the same GPU node, later we may need to confirm that we receive data on the same NUMA node
    if len(list(set(node)))>1:
        print "Bad configuration, the data from beam {:d} arrives at multiple GPU node, current capture software can not handle this ...".format(beam)
        exit()

    # Return result and close socket
    return target_name, ra, dec, freq, node[0], sorted(address_chunk)
    sock.close()
    
def main():
    beam = 0  # Beam index count from 0

    name, ra, dec, freq, node, address = multicast2obsinfo(beam)
    # name is the target name, ra and dec is the beam ra and dec, freq is center frequency of observing band, address is the ip:port:number-of-frequency-chunks format of address
      
    # At current stage, if we want to change between modes, we need to update source code;
    # If we want to use different configuration, we need to update source code and also script configuration;

    hdr    = 0
    freq   = 1340.5 # We need to remove this line in real observation
    length = 100
    directory = "/beegfs/DENG/docker"

    com_line = "./capture.py -a capture.conf -b {:s} -c {:f} -d {:s} -e {:f} -f {:d}".format(' '.join(address), length, directory, freq, hdr)
    print com_line
    os.system("./capture.py -a capture.conf -b {:s} -c {:f} -d {:s} -e {:f} -f {:d}".format(' '.join(address), length, directory, freq, hdr))

if __name__ == "__main__":
    main()
