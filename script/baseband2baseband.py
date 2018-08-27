#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json, os, subprocess, threading, datetime, time
import numpy as np
import captureinfo, metadata2streaminfo
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

# ./baseband2baseband.py -a ../config/pipeline.conf -b 0 -c 0 -d 8
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To capture data from given beam (with given part if the data arrives with multiple parts) with a docker container')
    
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--beam', type=int, nargs='+',
                        help='The beam id from 0')
    parser.add_argument('-c', '--part', type=int, nargs='+',
                        help='The part id from 0')
    parser.add_argument('-d', '--cpu', type=int, nargs='+',
                        help='Bind threads to cpu')
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = args.part[0]
    cpu           = args.cpu[0]
    
    uid = 50000
    gid = 50000
    ddir = ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['dir']
    hdir = "/home/pulsar"
    dvolume = '{:s}:{:s}'.format(ddir, ddir)
    hvolume = '{:s}:{:s}'.format(hdir, hdir)

    dname                   = "phased-array-feed"
    previous_container_name = "paf-capture.beam{:02d}part{:02d}".format(beam, part)
    current_container_name  = "paf-baseband2baseband.beam{:02d}part{:02d}".format(beam, part)
    software_name           = "baseband2baseband_main"
    
    directory      = ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['dir']
    key_capture    = ConfigSectionMap(pipeline_conf, "CAPTURE")['key']
    ndf_chk_rbuf   = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['ndf_chk_rbuf'])
    key_b2b        = ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['key']
    nstream        = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nstream'])
    ndf_chk_stream = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['ndf_chk_stream'])
    nrepeat        = ndf_chk_rbuf / (ndf_chk_stream * nstream)
    
    com_line = "docker run --ipc=container:{:s} --rm -it --net=host -v {:s} -v {:s} -u {:d}:{:d} --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --name {:s} xinpingdeng/{:s} \"taskset -c {:d} /home/pulsar/xinping/phased-array-feed/src/baseband2baseband/{:s} -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:s}\"".format(previous_container_name, dvolume, hvolume, uid, gid, current_container_name, dname, cpu, software_name, key_capture, key_b2b, ndf_chk_rbuf, nrepeat, nstream, ndf_chk_stream, directory)
    
    print com_line
    os.system(com_line)
