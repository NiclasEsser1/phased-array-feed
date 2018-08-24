#!/usr/bin/env python

import captureinfo, metadata2streaminfo
import argparse, ConfigParser, os
import threading, time
import subprocess

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

def b2b(capture_container_name, dvolume, hvolume, uid, gid, b2b_container_name, b2b_dname, system_conf, pipeline_conf, beam, part, hdr):
    while True:
        s = subprocess.check_output('docker ps', shell=True) # Wait until the previous container is ready
        if s.find(capture_container_name) != -1:
            com_line = "docker run --ipc=shareable --ipc=container:{:s} --rm -it --net=host -v {:s} -v {:s} -u {:d}:{:d} --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --name {:s} xinpingdeng/{:s} -a {:s} -b {:s} -c {:d} -d {:d} -e {:d}".format(capture_container_name, dvolume, hvolume, uid, gid, b2b_container_name, b2b_dname, system_conf, pipeline_conf, beam, part, hdr)
            print com_line
            os.system(com_line)
            break
        
def dbdisk(b2b_container_name, dvolume, uid, gid, dbdisk_container_name, key, directory):
    while True:
        s = subprocess.check_output('docker ps', shell=True) # Wait until the previous container is ready
        if s.find(b2b_container_name) != -1:
            com_line = "docker run --rm -it --ipc=container:{:s} -v {:s} -u {:d}:{:d} --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --name {:s} xinpingdeng/paf-base dada_dbdisk -k {:s} -D {:s}".format(b2b_container_name, dvolume, uid, gid, dbdisk_container_name, key, directory)
            print com_line
            os.system(com_line)
            break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To pass baseband data from a ring buffer into another')    
    parser.add_argument('-a', '--system_conf', type=str, nargs='+',
                        help='The configuration of PAF system')
    parser.add_argument('-b', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')    
    parser.add_argument('-c', '--beam', type=int, nargs='+',
                        help='The beam id from 0')
    parser.add_argument('-d', '--part', type=int, nargs='+',
                        help='The part id from 0')
    parser.add_argument('-e', '--hdr', type=int, nargs='+',
                        help='Record packet header or not')    
    
    args          = parser.parse_args()
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = args.part[0]
    hdr           = args.hdr[0]
    
    uid = 50000
    gid = 50000
    ddir = ConfigSectionMap(pipeline_conf, "CAPTURE")['dir']
    hdir = "/home/pulsar"
    dvolume = '{:s}:{:s}'.format(ddir, ddir)
    hvolume = '{:s}:{:s}'.format(hdir, hdir)

    capture_dname          = "paf-capture"
    b2b_dname              = "paf-baseband2baseband-demo"
    dbdisk_dname           = "paf-dbdisk"
    
    capture_container_name = "{:s}.beam{:02d}part{:02d}".format(capture_dname, beam, part)
    b2b_container_name     = "{:s}.beam{:02d}part{:02d}".format(b2b_dname, beam, part)
    dbdisk_container_name  = "{:s}.beam{:02d}part{:02d}".format(dbdisk_dname, beam, part)

    nodes, address_nchks, freqs, nchans = metadata2streaminfo.metadata2streaminfo(system_conf)
    nchan = nchans[beam][part]
    
    ndf_chk_rbuf = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['ndf_chk_rbuf'])
    nblk         = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nblk'])
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    nchan_chk    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
    df_hdrsz     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
    directory    = ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['dir']
    key           = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['key']), 0), 'x')
    pktsz    = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df + df_hdrsz
    if hdr == 1:
        blksz    = ndf_chk_rbuf * (nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan + df_hdrsz * nchan / nchan_chk)
    else:
        blksz    = ndf_chk_rbuf * nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan

    t_b2b    = threading.Thread(target = b2b, args=(capture_container_name, dvolume, hvolume, uid, gid, b2b_container_name, b2b_dname, system_conf, pipeline_conf, beam, part, hdr))
    t_dbdisk = threading.Thread(target = dbdisk, args=(b2b_container_name, dvolume, uid, gid, dbdisk_container_name, key, directory))
    
    t_b2b.start()
    t_dbdisk.start()
    
    t_b2b.join()
    t_dbdisk.join()
