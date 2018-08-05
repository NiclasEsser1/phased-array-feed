#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json, os, subprocess, threading, datetime, time
import numpy as np

SECDAY      = 86400.0
DADA_TIMSTR = "%Y-%m-%d-%H:%M:%S"
MJD1970     = 40587.0

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

def captureinfo(pipeline_conf, system_conf, destination, nchan, hdr, beam, part):
    # Get pipeline configuration from configuration file
    ndf_chk_rbuf = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_chk_rbuf'])
    ndf_check    = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_check'])
    nblk         = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['nblk'])
    key          = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "CAPTURE")['key']), 0), 'x')
    kfile_prefix = ConfigSectionMap(pipeline_conf, "CAPTURE")['kfname_prefix']
    kfname       = "{:s}_beam{:02d}_part{:02d}.key".format(kfile_prefix, beam, part)
    nreader      = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['nreader'])
     
    # Record the key to a key file with kfname
    kfile = open(kfname, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key))
    kfile.close()
    
    # Get system configuration from configuration file
    sec_prd      = float(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['sec_prd'])
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    nchan_chk    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
    df_hdrsz     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
    pktsz        = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df + df_hdrsz
    if hdr == 1:
        blksz    = ndf_chk_rbuf * (nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan + df_hdrsz * nchan / nchan_chk)
    else:
        blksz    = ndf_chk_rbuf * nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan
    
    # Check the connection
    destination_active, destination_dead = check_all_ports(destination, pktsz, sec_prd, ndf_check)
    
    if (len(destination_active) == 0):
        print "There is no active port for beam {:02d}, have to abort ...".format(beam)
        exit(1)
    print "The active destination \"[IP:PORT:NCHUNK_EXPECT:NCHUNK_ACTUAL]\" are: ", destination_active
    print "The dead destination \"[IP:PORT:NCHUNK_EXPECT]\" are:                 ", destination_dead
    
    # Create PSRDADA buffer
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key, blksz, nblk, nreader))

    # Get reference timestamp of capture
    refinfo = capture_refinfo(destination_active[0], pktsz, system_conf)
    print "The reference timestamp \"(DF_SEC, DF_IDF)\"for current capture is: ", refinfo
    
    return destination_active, destination_dead, refinfo, key
