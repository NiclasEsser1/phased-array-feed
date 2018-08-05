#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json, os, subprocess, threading, datetime, time
import numpy as np
import captureinfo, metadata2streaminfo

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

def main(system_conf, pipeline_conf, bind, hdr, beam, part):
    nodes, address_nchks, freqs, nchans = metadata2streaminfo.metadata2streaminfo(system_conf)
    dir_capture = ConfigSectionMap(pipeline_conf, "CAPTURE")['dir']
    destination_active, destination_dead, refinfo, key = captureinfo.captureinfo(pipeline_conf, system_conf, address_nchks[beam][part], nchans[beam][part], hdr, beam, part)
    
    # To set up cpu cores if we decide to bind threads
    ncpu_numa    = int(ConfigSectionMap(system_conf, "NUMA")['ncpu_numa'])
    node = int(destination_active[0].split(":")[0].split(".")[3])
    for i in range(len(destination_active)):
        cpu = (node - 1) * ncpu_numa + i
        destination_active[i] = "{:s}:{:d}".format(destination_active[i], cpu)

    # To setup buffer size for single packet and reading start of it    
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    nchan_chk    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
    df_hdrsz     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
    ndf_chk_prd  = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndf_chk_prd'])
    ndf_chk_rbuf = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_chk_rbuf'])
    ndf_chk_tbuf = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_chk_tbuf'])
    pktsz        = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df + df_hdrsz
    if(hdr == 0):
        pktoff = df_hdrsz                                                 # The start point of each BMF packet
    else:
        pktoff = 0
#    exit()

    # Do the real work here
    sec_prd = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['sec_prd'])
    nchunk = nchans[beam][part]/nchan_chk;
    if (len(destination_dead) == 0):
        capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -f {:f} -g {:d} -i {:f}:{:d}:{:d} -j {:s} -k {:d} -l {:d} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:d} -s /tmp/capture.beam{:d}.part{:d}".format(key, pktsz, pktoff, " -d ".join(destination_active), freqs[beam][part], nchans[beam][part], refinfo[0], refinfo[1], refinfo[2], dir_capture, cpu + 1, cpu + 2, bind, sec_prd, nchunk, ndf_chk_rbuf, ndf_chk_tbuf, ndf_chk_prd, beam, part)
    else:
        capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -e {:s} -f {:f} -g {:d} -i {:f}:{:d}:{:d} -j {:s} -k {:d} -l {:d} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:d} -s /tmp/capture.beam{:d}.part{:d}".format(key, pktsz, pktoff, " -d ".join(destination_active), " -e ".join(destination_dead[beam][part]), freqs[beam][part], nchans[beam][part], refinfo[0], refinfo[1], refinfo[2], dir_capture, cpu + 1, cpu + 2, bind, sec_prd, nchunk, ndf_chk_rbuf, ndf_chk_tbuf, ndf_chk_prd, beam, part)
    print capture_command
    os.system(capture_command)
    
    # Delete PSRDADA buffer
    os.system("dada_db -d {:s}".format(key))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To capture data from given beam (with given part if the data arrives with multiple parts)')
    
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
    parser.add_argument('-f', '--bind', type=int, nargs='+',
                        help='Bind threads to cpu or not')
    
    args          = parser.parse_args()
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = args.part[0]
    hdr           = args.hdr[0]
    bind          = args.bind[0]
    
    main(system_conf, pipeline_conf, bind, hdr, beam, part)
