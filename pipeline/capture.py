#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json
import numpy as np
import subprocess

# ./capture.py -a ../config/pipeline.conf -b ../config/system.conf -c 1340.5 -d 336 -e '10.17.0.1:17100:8', '10.17.0.1:17101:8', '10.17.0.1:17102:8', '10.17.0.1:17103:8', '10.17.0.1:17104:8', '10.17.0.1:17105:8' -f 0 -g 0

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

def main(args):
    # Read pipeline input
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    system_conf   = args.system_conf[0]
    destination   = args.destination
    nchan         = args.nchan[0]
    hdr           = args.hdr[0]

    # Get pipeline configuration from configuration file
    ndf_blk      = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_blk'])
    nblk         = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['nblk'])
    hdr_fname    = ConfigSectionMap(pipeline_conf, "CAPTURE")['hdr_fname']
    key          = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "CAPTURE")['key']), 0), 'x')
    kfile_prefix = ConfigSectionMap(pipeline_conf, "CAPTURE")['kfname_prefix']

    # Get system configuration from configuration file
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    if hdr == 1:
        nchan_chk = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
        df_hdrsz  = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
        blksz     = ndf_blk * (nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan + df_hdrsz * nchan / nchan_chk)
    else:
        blksz   = ndf_blk * nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan
    print blksz
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create DADA buffer and run capture with given parameters')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The name of configuration file which defines the pipeline configurations')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The name of configuration file which defines the system configurations')
    parser.add_argument('-c', '--freq', type=float, nargs='+',
                        help='The center frequency')
    parser.add_argument('-d', '--nchan', type=int, nargs='+',
                        help='The number of channels')
    parser.add_argument('-e', '--destination', type=str, nargs='+',
                        help='The destination')
    parser.add_argument('-f', '--part', type=int, nargs='+', # Count from zero
                        help='Which part of the capture for given beam')
    parser.add_argument('-g', '--hdr', type=int, nargs='+', # Count from zero
                        help='To capture packet header or not')
    
    args = parser.parse_args()
    main(args)
