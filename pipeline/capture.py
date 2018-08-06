#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json, os, subprocess, threading, datetime, time
import numpy as np
import captureinfo, metadata2streaminfo, capture_beam_part

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
    
    nodes, address_nchks, freqs, nchans = metadata2streaminfo.metadata2streaminfo(system_conf)
    instrument = "PAF-BEAM{:02d}PART{:02d}".format(beam, part)
    ctrl_socket = "/tmp/capture.beam{:02d}.part{:02d}".format(beam, part)
    
    capture_beam_part.capture_beam_part(system_conf, pipeline_conf, bind, hdr, nchans[beam][part], freqs[beam][part], address_nchks[beam][part], ctrl_socket, instrument, beam, part)
