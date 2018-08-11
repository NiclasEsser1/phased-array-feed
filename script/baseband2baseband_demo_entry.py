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
                        help='To capture header or not')
    
    args          = parser.parse_args()
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = args.part[0]
    hdr           = args.hdr[0]

    key_capture      = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "CAPTURE")['key']), 0), 'x')
    key_b2b          = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['key']), 0), 'x')
    kfname_b2b       = "baseband2baseband-demo.beam{:02d}part{:02d}.key".format(beam, part)
    kfile = open(kfname_b2b, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key_b2b))
    kfile.close()
    
    nodes, address_nchks, freqs, nchans = metadata2streaminfo.metadata2streaminfo(system_conf)
    nchan = nchans[beam][part]
    
    ndf_chk_rbuf_b2b = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['ndf_chk_rbuf'])
    nblk_b2b         = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nblk'])
    nreader_b2b      = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nreader'])
    
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    nchan_chk    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
    df_hdrsz     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
    if hdr == 1:
        pktsz        = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df + df_hdrsz
        blksz    = ndf_chk_rbuf_b2b * (nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan + df_hdrsz * nchan / nchan_chk)
    else:
        pktsz        = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df
        blksz    = ndf_chk_rbuf_b2b * nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan
    nchunk = nchan/nchan_chk

    # Create PSRDADA buffer
    blksz = pktsz
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key_b2b, blksz, nblk_b2b, nreader_b2b))

    baseband2baseband_demo_command = "../src/demo/baseband2baseband_demo -a {:s} -b {:s} -c {:d} -d {:d} -e {:d}".format(key_capture, key_b2b, ndf_chk_rbuf_b2b, nchunk, pktsz)
    print baseband2baseband_demo_command
    
    os.system(baseband2baseband_demo_command)
    
    # Delete PSRDADA buffer
    os.system("dada_db -d -k {:s}".format(key_b2b))
