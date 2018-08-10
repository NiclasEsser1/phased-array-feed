#!/usr/bin/env python

import captureinfo, metadata2streaminfo
import argparse, ConfigParser, os

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

#def main():

# ./baseband2baseband_demo.py -a ../config/system.conf -b ../config/pipeline.conf -c 0 -d 0 -e 1
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

    current_dname           = "paf-baseband2baseband-demo"
    previous_dname          = "paf-capture"
    current_container_name  = "{:s}.beam{:02d}part{:02d}".format(current_dname, beam, part)
    previous_container_name = "{:s}.beam{:02d}part{:02d}".format(previous_dname, beam, part)
    
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
    pktsz        = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df + df_hdrsz
    if hdr == 1:
        blksz    = ndf_chk_rbuf * (nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan + df_hdrsz * nchan / nchan_chk)
    else:
        blksz    = ndf_chk_rbuf * nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan
    memsize = blksz * (nblk + 1) * 2  # + 1 to be safe
    nchunk = nchan/nchan_chk
    com_line = "docker run --ipc=shareable --ipc=container:{:s} --rm -it --net=host -v {:s} -v {:s} -u {:d}:{:d} --ulimit memlock={:d} --name {:s} xinpingdeng/{:s} -a {:s} -b {:s} -c {:d} -d {:d} -e {:d}".format(previous_container_name, dvolume, hvolume, uid, gid, memsize, current_container_name, current_dname, system_conf, pipeline_conf, beam, part, hdr)
    print com_line

    os.system(com_line)
