#!/usr/bin/env python

import numpy as np
import socket, ConfigParser, struct, os, argparse

SECDAY = 86400.

def ConfigSectionMap(fname, section):
    # Play with configuration file
    Config = ConfigParser.ConfigParser()
    Config.read(fname)
    
    dict_conf = {}
    options = Config.options(section)
    for option in options:
        try:
            dict_conf[option] = Config.get(section, option)
            if dict_conf[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict_conf[option] = None
    return dict_conf

def check_port(ip, port, pktsz, ndf_check):
    active = 1
    nchunk_active = 0
    data = bytearray(pktsz) 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (ip, port)
    sock.bind(server_address)
    
    try:
        nbyte, address = sock.recvfrom_into(data, pktsz)
        if (nbyte != pktsz):
            active = 0
        else:
            source = []
            active = 1
            for i in range(ndf_check):
                buf, address = sock.recvfrom(pktsz)
                source.append(address)
            nchunk_active = len(set(source))
    except:
        active = 0        

    return active, nchunk_active

def check_all_ports(destination, pktsz, sec_prd, ndf_check):
    nport = len(destination)
    active = np.zeros(nport, dtype = int)
    nchunk_active = np.zeros(nport, dtype = int)
    socket.setdefaulttimeout(sec_prd)  # Force to timeout after one data frame period
    
    for i in range(nport):
        active[i], nchunk_active[i] = check_port(destination[i].split(":")[0], int(destination[i].split(":")[1]), pktsz, ndf_check)
    destination_active = []   # The destination where we can receive data
    destination_dead   = []   # The destination where we can not receive data
    for i in range(nport):
        if active[i] == 1:
            destination_active.append("{:s}:{:s}:{:s}:{:d}".format(destination[i].split(":")[0], destination[i].split(":")[1], destination[i].split(":")[2], nchunk_active[i]))
        else:
            destination_dead.append("{:s}:{:s}:{:s}".format(destination[i].split(":")[0], destination[i].split(":")[1], destination[i].split(":")[2]))
    return destination_active, destination_dead

def capture_refinfo(destination, pktsz, system_conf):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (destination.split(":")[0], int(destination.split(":")[1]))
    sock.bind(server_address)
    buf, address = sock.recvfrom(pktsz) # raw packet

    df_res   = float(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_res'])
    data     = np.fromstring(buf, 'uint64')
    hdr_part = np.uint64(struct.unpack("<Q", struct.pack(">Q", data[0]))[0])
    sec_ref  = (hdr_part & np.uint64(0x3fffffff00000000)) >> np.uint64(32)
    idf_ref  = hdr_part & np.uint64(0x00000000ffffffff)

    hdr_part  = np.uint64(struct.unpack("<Q", struct.pack(">Q", data[1]))[0])
    epoch     = (hdr_part & np.uint64(0x00000000fc000000)) >> np.uint64(26)    
    epoch_ref = float(ConfigSectionMap(system_conf, "EpochBMF")['{:d}'.format(epoch)])
    sec_prd   = idf_ref * df_res

    sec        = int(np.floor(sec_prd) + sec_ref + epoch_ref * SECDAY)
    picosecond = int(1.0E6 * round(1.0E6 * (sec_prd - np.floor(sec_prd))))
    
    #return sec, picosecond
    return epoch_ref, sec_ref, idf_ref

# ./capture_2beams1nic_36beams_36chunks.py -a ../config/system.conf -b ../config/pipeline.conf -c 10.17.0.1 -d 17100 17101 17102 -e 12 -f 0 -g 1 -i 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To capture data with given ip and port')
    
    parser.add_argument('-a', '--system_conf', type=str, nargs='+',
                        help='The configuration of PAF system')
    parser.add_argument('-b', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-c', '--ip', type=str, nargs='+',
                        help='The ip to capture data')
    parser.add_argument('-d', '--ports', type=int, nargs='+',
                        help='The ports to capture data')
    parser.add_argument('-e', '--nchunk', type=int, nargs='+',
                        help='Number of chunks to record from each port')
    parser.add_argument('-f', '--hdr', type=int, nargs='+',
                        help='Record packet header or not')
    parser.add_argument('-g', '--bind', type=int, nargs='+',
                        help='Bind threads to cpu or not')
    parser.add_argument('-i', '--beam', type=int, nargs='+',
                        help='Beam id')
    
    args          = parser.parse_args()
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    ip            = args.ip[0]
    ports         = args.ports
    nchunk        = args.nchunk[0]
    hdr           = args.hdr[0]
    bind          = args.bind[0]
    beam          = args.beam[0]
    
    ddir           = "/beegfs/DENG"
    dvolume        = "{:s}:{:s}".format(ddir, ddir)
    hvolume        = "/home/pulsar:/home/pulsar"
    uid            = 50000
    gid            = 50000
    dname          = "phased-array-feed"
    key            = "dada"

    nreader        = 1
    nblk           = 8
    ndf            = 10240
    pktsz          = 7232
    ndf_check      = 1024
    period         = 27
    nport          = len(ports)
    nchan          = nport * nchunk * 7
    nchunk_all     = nchunk * nport
    freq           = 1340.5
    ncpu_numa      = 10
    
    if hdr:
        pktoff = 0
    else:
        pktoff = 64

    blksz          = (pktsz - pktoff) * ndf * nchunk_all
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key, blksz, nblk, nreader))
    
    destination = []
    for port in ports:
        destination.append("{:s}:{:d}:{:d}".format(ip, port, nchunk))    
    destination_active = check_all_ports(destination, pktsz, period, ndf_check)[0]
    destination_dead   = check_all_ports(destination, pktsz, period, ndf_check)[1]

    destination_alive = []
    cpu = 0
    for info in destination_active:
        if beam % 2 ==0:
            destination_alive.append("{:s}:{:d}".format(info, cpu))
            cpu = cpu + 1
        else:
            destination_alive.append("{:s}:{:d}".format(info, cpu + ncpu_numa/2))
            cpu = cpu + 1    
    print destination_alive
    
    refinfo = capture_refinfo(destination_active[0], pktsz, system_conf)
    capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -f {:f} -g {:d} -i {:f}:{:d}:{:d} -j {:s} -k {:d} -l {:d} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:d} -s {:s} -t {:s} -u {:s}".format(key, pktsz, pktoff, " -d ".join(destination_active), freq, nchan, refinfo[0], refinfo[1], refinfo[2], ddir, cpu + ncpu_numa/2, cpu + 1 + ncpu_numa/2, bind, period, nchunk_all, ndf, 250, 250000, "capture.socket{:d}".format(beam), "../config/header_16bit.txt", "PAF-BMF")
    os.system(capture_command)
    
    os.system("dada_db -d -k {:s}".format(key))
