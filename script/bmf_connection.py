#!/usr/bin/env python

import numpy as np
import socket
import struct
import parser
import argparse
import inspect
from subprocess import check_output

#ONE_BEAM = {"nchunk_per_port":        16,
#            "ports":                  [[17100, 17101, 17102]]
#}
#
#TWO_BEAM = {"nchunk_per_port":       11,
#            "ports":                 [[17100, 17101, 17102], [17103, 17104, 17105]]
#}

ONE_BEAM = {"nchunk_per_port":        48,
            "ports":                  [[17100]]
}

TWO_BEAM = {"nchunk_per_port":       33,
            "ports":                 [[17100], [17101]]
}

PAF_DF_PKTSZ = 7232

class PipelineError(Exception):
    pass

def check_port_connection(ip, port, ndf_check_chk):
    """
    To check the connection of single port
    """
    alive = 1
    nchk_alive = 0
    data = bytearray(PAF_DF_PKTSZ)
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Force to timeout after one data frame period
    sock.settimeout(1)
    server_address = (ip, port)
    sock.bind(server_address)
    
    try:
        nbyte, address = sock.recvfrom_into(data, PAF_DF_PKTSZ)
        if (nbyte != PAF_DF_PKTSZ):
            alive = 0
        else:
            source = []
            alive = 1
            for i in range(ndf_check_chk):
                buf, address = sock.recvfrom(PAF_DF_PKTSZ)
                
                data_uint64 = np.fromstring(str(buf), 'uint64')
                hdr_uint64 = np.uint64(struct.unpack(
                    "<Q", struct.pack(">Q", data_uint64[2]))[0])
                
                source.append(address)
            nchk_alive = len(set(source))
        sock.close()
    except Exception as error:
        raise PipelineError("{} fail".format(inspect.stack()[0][3]))
    return alive, nchk_alive

def check_beam_connection(destination, ndf_check_chk):
    """
    To check the connection of one beam with given ip and port numbers
    """
    nport = len(destination)
    alive = np.zeros(nport, dtype=int)
    nchk_alive = np.zeros(nport, dtype=int)
    
    destination_dead = []   # The destination where we can not receive data
    destination_alive = []   # The destination where we can receive data
    for i in range(nport):
        ip = destination[i].split("_")[0]
        port = int(destination[i].split("_")[1])
        alive, nchk_alive = check_port_connection(
            ip, port, ndf_check_chk)
        
        if alive == 1:
            destination_alive.append(
                destination[i] + "_{}".format(nchk_alive))
        else:
            destination_dead.append(destination[i])
            
    if (len(destination_alive) == 0):  # No alive ports, error
        raise PipelineError("The stream is not alive")

    return destination_alive, destination_dead

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='To run the pipeline for my test')    
    parser.add_argument('-a', '--numa', type=int, nargs='+',
                        help='The ID of numa node')
    parser.add_argument('-b', '--nbeam', type=int, nargs='+',
                        help='The number of beams')
    args  = parser.parse_args()
    numa  = args.numa[0]
    nbeam = args.nbeam[0]
    
    host_id = check_output("hostname").strip()[-1]
    ip      = "10.17.{}.{}".format(host_id, numa + 1)

    check_ndf_per_chunk = 10240

    if nbeam == 1:
        nchunk_per_port = ONE_BEAM["nchunk_per_port"]
        ports           = ONE_BEAM["ports"]
    if nbeam == 2:
        nchunk_per_port = TWO_BEAM["nchunk_per_port"]
        ports           = TWO_BEAM["ports"]
        
        
    for i in range(nbeam):
        destination = []
        for port in ports[i]:
            destination.append("{}_{}_{}".format(ip, port, nchunk_per_port))
        destination_alive, dead_info = check_beam_connection(destination, check_ndf_per_chunk)
        print destination_alive
