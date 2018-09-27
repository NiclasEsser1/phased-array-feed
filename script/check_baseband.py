#!/usr/bin/env python

import numpy as np
import struct
import ConfigParser
import time
import os

def keyword_value(data_file, nline_header, key_word):
    data_file.seek(0)  # Go to the beginning of DADA file
    for iline in range(nline_header):
        line = data_file.readline()
        if key_word in line and line[0] != '#':
            return line.split()[1]
    print "Can not find the keyword \"{:s}\" in header ...".format(key_word)
    exit(1)

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

dada_hdrsz  = 4096
pkt_hdrsz   = 64
pktsz       = 7232
MJD1970     = 40587.0
SECDAY      = 86400.0
fname       = "/beegfs/DENG/AUG/baseband/J0332+5434/J0332+5434-baseband.dada"
fname       = "/beegfs/DENG/AUG/baseband/J1939+2134/J1939+2134-baseband.dada"
#fname       = "/beegfs/DENG/AUG/baseband/J1713+0747/J1713+0747-baseband.dada"
#fname       = "/beegfs/DENG/AUG/baseband/J1819-1458/J1819-1458-baseband.dada"

system_conf = "../config/system.conf"

# To get timestamp from the first packet
f = open(fname, "r")
f.seek(dada_hdrsz)
while True:
    pkt_hdr = f.read(pkt_hdrsz)
    f.seek(pktsz - pkt_hdrsz, os.SEEK_CUR)
    
    if pkt_hdr =='':
        break
    else:
        data     = np.fromstring(pkt_hdr, 'uint64')
        
        hdr_part = np.uint64(struct.unpack("<Q", struct.pack(">Q", data[0]))[0])
        second   = (hdr_part & np.uint64(0x3fffffff00000000)) >> np.uint64(32)
        idf      = hdr_part & np.uint64(0x00000000ffffffff)
    
        hdr_part = np.uint64(struct.unpack("<Q", struct.pack(">Q", data[2]))[0])
        freq     = (hdr_part & np.uint64(0x00000000ffff0000)) >> np.uint64(16)    
        
        print second, idf, freq
f.close()
