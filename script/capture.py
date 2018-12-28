#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json, os, subprocess, threading, datetime, time
import numpy as np
import metadata2streaminfo
import datetime

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

# ./capture.py -a ../config/system.conf -b ../config/pipeline.conf -c 0 -d 0 -e 0 -f 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To capture data from given beam with a docker container')
    
    parser.add_argument('-a', '--system_conf', type=str, nargs='+',
                        help='The configuration of PAF system')
    parser.add_argument('-b', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')    
    parser.add_argument('-c', '--beam', type=int, nargs='+',
                        help='The beam id from 0')
    parser.add_argument('-d', '--hdr', type=int, nargs='+',
                        help='Record packet header or not')
    parser.add_argument('-e', '--bind', type=int, nargs='+',
                        help='Bind threads to cpu or not')
    
    args          = parser.parse_args()
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = 0
    hdr           = args.hdr[0]
    bind          = args.bind[0]
    numa          = (beam % 4) / 2
    cpu0          = (beam % 4) * 5
    cpu1          = (beam % 4 + 1) * 5 - 1
    uid = 50000
    gid = 50000
    ddir = ConfigSectionMap(pipeline_conf, "CAPTURE")['dir']
    hdir = "/home/pulsar"
    dvolume = '{:s}:{:s}'.format(ddir, ddir)
    hvolume = '{:s}:{:s}'.format(hdir, hdir)

    dname          = "phased-array-feed"
    container_name = "paf-capture.beam{:02d}part{:02d}".format(beam, part)
    script_name    = "capture_entry.py"
    
    nodes, address_nchks, freqs, nchans = metadata2streaminfo.metadata2streaminfo(system_conf)
    freq = freqs[beam][part]
    nchan = nchans[beam][part]

    ndf_chk_rbuf = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_chk_rbuf'])
    nblk         = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['nblk'])
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    nchan_chk    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
    df_hdrsz     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
    
    #instrument = "PAF-BEAM{:02d}PART{:02d}".format(beam, part)
    instrument = "BEAM{:02d}".format(beam)
    ctrl_socket = "./capture.beam{:02d}part{:02d}.socket".format(beam, part)
    address_nchk = " ".join(address_nchks[beam][part])
    
    #com_line = "docker run --ipc=shareable --rm -it --net=host --cpuset-mems={:d} --cpuset-cpus={:s} -v {:s} -v {:s} -u {:d}:{:d} --cap-add=IPC_LOCK --ulimit memlock=-1:-1 -e DISPLAY -v /tmp:/tmp --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm --name {:s} xinpingdeng/{:s} \"./{:s} -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:f} -g {:s} -i {:s} -j {:s} -k {:d} -l {:d}\"".format(numa, "{:d}-{:d}".format(cpu0, cpu1), dvolume, hvolume, uid, gid, container_name, dname, script_name, system_conf, pipeline_conf, hdr, bind, nchan, freq, address_nchk, ctrl_socket, instrument, beam, part)
    com_line = "docker run --ipc=shareable --rm -it --net=host --cpuset-mems={:d} -v {:s} -v {:s} -u {:d}:{:d} --cap-add=IPC_LOCK --ulimit memlock=-1:-1 -e DISPLAY -v /tmp:/tmp --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm --name {:s} xinpingdeng/{:s} \"./{:s} -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:f} -g {:s} -i {:s} -j {:s} -k {:d} -l {:d}\"".format(numa, dvolume, hvolume, uid, gid, container_name, dname, script_name, system_conf, pipeline_conf, hdr, bind, nchan, freq, address_nchk, ctrl_socket, instrument, beam, part)
    
    print com_line
    os.system(com_line)
    os.system("rm -f {:s}".format(ctrl_socket))
