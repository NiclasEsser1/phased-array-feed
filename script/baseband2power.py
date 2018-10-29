#!/usr/bin/env python

import os, parser, argparse, ConfigParser, time
import numpy as np

BLKSZ = 1024

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

# docker run --ipc=container:disk2db --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name baseband2power xinpingdeng/phased-array-feed "./baseband2power.py -a ../config/pipeline.conf -b ../config/system.conf -c 0.016"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert baseband data to power with original channels')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The configuration of system')
    parser.add_argument('-c', '--tsamp', type=float, nargs='+',
                        help='The sampleing time in seconds of converted power data')

    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    system_conf   = args.system_conf[0]
    tsamp         = args.tsamp[0]

    df_res         = float(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_res'])
    nsamp_df       = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    ndf_chk_rbuf   = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['ndf_chk_rbuf'])
    nchan          = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['nchan_out'])
    nbyte          = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['nbyte_out'])
    nstream        = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['nstream'])
    nreader        = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['nreader'])
    nblk           = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['nblk'])
    ddir           = ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['dir']
    key_output     = ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['key']
    key_input      = ConfigSectionMap(pipeline_conf, "DISK2DB")['key']
    
    # Find the number of data frames for each stream
    # It is the maximum which can be devided by ndf_chk_rbuf/nstream
    if(ndf_chk_rbuf%nstream):
        print "Please reset the NDF_CHK_RBUF and NSTREAM ... !"
        print exit(1)
    ndf_chk_stream_max     = int(tsamp / df_res)
    ndf_chk_stream_attempt = np.arange(ndf_chk_stream_max, 0, -1)
    for ndf_chk_stream in ndf_chk_stream_attempt:
        if ((ndf_chk_rbuf/nstream) % ndf_chk_stream == 0):
            break
    print ndf_chk_stream

    # To get the parameters for sum
    nsum = int(ndf_chk_stream * nsamp_df)
    print nsum
    if (nsum < (2 * BLKSZ)):
        twice_sum = 0
        sum1_blksz = nsum / 2
    else:
        twice_sum = 1
        sum1_blksz = nsum
        while (sum1_blksz > BLKSZ):
            sum1_blksz = sum1_blksz / 2
    print sum1_blksz, twice_sum

    # How many runs to finish one incoming buffer
    nrepeat = int(ndf_chk_rbuf/(nstream * ndf_chk_stream))

    # How many byte for each output buffer
    blksz = nbyte * nchan * nrepeat * nstream

    # Create output buffer
    db_create = "dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key_output, blksz, nblk, nreader)
    print db_create
    os.system(db_create)

    # Do the work
    b2p = "../src/baseband2power/baseband2power_main -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:d} -i {:d} -j {:s}".format(key_input, key_output, ndf_chk_rbuf, nrepeat, nstream, ndf_chk_stream, twice_sum, sum1_blksz, ddir)
    #b2p = "nvprof ../src/baseband2power/baseband2power_main -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:d} -i {:d} -j {:s}".format(key_input, key_output, ndf_chk_rbuf, nrepeat, nstream, ndf_chk_stream, twice_sum, sum1_blksz, ddir)
    #b2p = "cuda-memcheck ../src/baseband2power/baseband2power_main -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:d} -i {:d} -j {:s}".format(key_input, key_output, ndf_chk_rbuf, nrepeat, nstream, ndf_chk_stream, twice_sum, sum1_blksz, ddir)
    
    print b2p
    os.system(b2p)
    
    # Destroy output buffer
    time.sleep(10)
    db_destroy = "dada_db -d -k {:s}".format(key_output)
    os.system(db_destroy)
