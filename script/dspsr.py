#!/usr/bin/env python

import os, parser, argparse, ConfigParser

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

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='To transfer data from shared memeory to disk with a docker container')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')    
    parser.add_argument('-b', '--beam', type=int, nargs='+',
                        help='The beam id from 0')
    parser.add_argument('-c', '--part', type=int, nargs='+',
                        help='The part id from 0')
    parser.add_argument('-d', '--directory', type=str, nargs='+',
                        help='Directory to put the data')
    
    uid = 50000
    gid = 50000

    hdir          = "/home/pulsar"
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = args.part[0]
    directory     = args.directory[0]
    key           = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['key']), 0), 'x')
    hvolume       = '{:s}:{:s}'.format(hdir, hdir)
    dvolume       = '{:s}:{:s}'.format(directory, directory)
    
    previous_container_name = "paf-baseband2baseband-demo.beam{:02d}part{:02d}".format(beam, part)
    current_container_name  = "paf-dsspr.beam{:02d}part{:02d}".format(beam, part)
    kfname_b2b              = "baseband2baseband-demo.beam{:02d}part{:02d}.key".format(beam, part)
    
    com_line = "docker run --rm -it --ipc=container:{:s} -v {:s} -v {:s} -u {:d}:{:d} --name {:s} xinpingdeng/paf-base dspsr -E  ../config/J1939+2134.par {:s}".format(previous_container_name, dvolume, hvolume, uid, gid, current_container_name, kfname_b2b)
    print com_line
    os.system(com_line)
