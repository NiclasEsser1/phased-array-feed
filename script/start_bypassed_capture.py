import socket
import time
import numpy as np
import struct
import os
import subprocess
import argparse
from argparse import RawTextHelpFormatter

PAF_DF_PACKETSZ = 7232
PAF_PERIOD = 27




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options', formatter_class=RawTextHelpFormatter)
    # parser.add_argument('--key', '-k', action = "store", default ="dada", dest = "key", help = "Shared memory key")
    parser.add_argument('--node_id', '-i', action = "store", default =0, dest = "nodeid", help = "ID of the numa node")
    # parser.add_argument('--time_ref', '-ref', action = "store", dest = "time_ref", help = "Time reference string epoch_seconds_dataframe")
    # parser.add_argument('--ip_addr', '-ip', action = "store", dest = "ip_addr", help = "Ip address")
    # parser.add_argument('--freq', '-f', action = "store", dest = "freq", help = "Base frequency of channel group")
    # parser.add_argument('--capture_conf', '-c', action = "store", dest = "capture_conf", help = "Capture conf of single capture thread: ip_port_exepectedbeams_actualbeams_cpuid")
    # parser.add_argument('--packets', '-p', action = "store", dest = "packets", help = "Number of packets stored in one ringerbuffer block. Note: packet*nof_beam*packets_size = ringubffer_size")
    # parser.add_argument('--temp_packets', '-tp', action = "store", dest = "temp_packets", default=128, help = "Number of packets stored temporary buffer. Note: temp_packet*nof_beam*packets_size = tempbuffer_size")

    # key = parser.parse_args().key
    nodeid = parser.parse_args().nodeid
    # time_ref = parser.parse_args().time_ref
    # freq = parser.parse_args().freq
    # capture_conf = parser.parse_args().capture_conf
    # packets = int(parser.parse_args().packets)
    # temp_packets = parser.parse_args().temp_packets
    # ip_addr = parser.parse_args().ip_addr

    print(nodeid)
    block_size = 16384*36*PAF_DF_PACKETSZ
    nof_blocks = 8
    disk_folder = "/beegfsEDD/NESSER"
    os.system("dada_db -k "+key+" -d")
    os.system("numactl -m "+str(nodeid)+" dada_db -k "+key+" -l -p -b "+str(block_size)+" -n "+str(nof_blocks))
    os.system("numactl -m 0 dada_dbdisk -k "+key+" -D "+disk_folder+" -W -d -s")

    # p = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
