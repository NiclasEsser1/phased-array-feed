#!/usr/bin/env python

import numpy as np
import struct
import ConfigParser
import time

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
fname       = "/beegfs/DENG/docker/2018-08-07-12:07:28_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-07-12:10:54_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-07-12:12:23_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-07-12:13:55_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-09:49:30_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-13:13:43_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:06:03_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:09:10_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:13:47_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:18:41_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:31:07_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:35:40_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:49:16_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-10-15:56:03_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-17-15:22:49_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-18-07:01:54_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-08:57:32_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-08:56:19_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-09:22:40_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-09:20:13_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-12:29:21_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-12:28:43_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-12:33:20_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-12:35:17_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-18:07:38_0000000000000000.000000.dada"
fname       = "/beegfs/DENG/docker/2018-08-24-18:10:02_0000000000000000.000000.dada"
#fname       = "/beegfs/DENG/docker/2018-08-24-18:11:37_0000000000000000.000000.dada"
#fname       = "/beegfs/DENG/AUG/baseband/J1939+2134/0/append0.dada"
#fname       = "/beegfs/DENG/AUG/baseband/J0332+5434/1/J0332+5434-baseband1.dada"
#fname       = "/beegfs/DENG/AUG/baseband/J0332+5434/3/J0332+5434-baseband3.dada"
#fname       = "/beegfs/DENG/AUG/baseband/J0332+5434/J0332+5434-baseband.dada"
fname       = "/beegfs/DENG/AUG/baseband/J1939+2134/J1939+2134-baseband.dada"
fname       = "/beegfs/DENG/AUG/baseband/J1713+0747/J1713+0747-baseband.dada"
fname       = "/beegfs/DENG/AUG/baseband/J1819-1458/J1819-1458-baseband.dada"

system_conf = "../config/system.conf"

# To get header information
f = open(fname, "r")
nline_header = 50
mjd_start    = float(keyword_value(f, nline_header, "MJD_START"))
utc_start    = keyword_value(f, nline_header, "UTC_START")
picoseconds  = int(keyword_value(f, nline_header, "PICOSECONDS"))
print "{:.10f}\t{:s}\t{:d}".format(mjd_start, utc_start, picoseconds)
f.close()

# To get timestamp from the first packet
f = open(fname, "r")
f.seek(dada_hdrsz)
pkt_hdr = f.read(pkt_hdrsz)

data     = np.fromstring(pkt_hdr, 'uint64')
hdr_part = np.uint64(struct.unpack("<Q", struct.pack(">Q", data[0]))[0])
sec_ref  = (hdr_part & np.uint64(0x3fffffff00000000)) >> np.uint64(32)
idf_ref  = hdr_part & np.uint64(0x00000000ffffffff)

hdr_part  = np.uint64(struct.unpack("<Q", struct.pack(">Q", data[1]))[0])
epoch     = (hdr_part & np.uint64(0x00000000fc000000)) >> np.uint64(26)    
epoch_ref = float(ConfigSectionMap(system_conf, "EpochBMF")['{:d}'.format(epoch)])
df_res    = float(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_res'])

sec_prd   = df_res * idf_ref

sec         = int(np.floor(sec_prd) + sec_ref + epoch_ref * SECDAY)
picoseconds = int(1.0E6 * round(1.0E6 * (sec_prd - np.floor(sec_prd))))
utc_start   = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime(sec))
mjd_start   = sec / SECDAY + MJD1970

print "{:.10f}\t{:s}\t{:d}".format(mjd_start, utc_start, picoseconds)

f.close()
