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

# docker run --rm -it --ipc=container:baseband2power -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --name db2disk xinpingdeng/phased-array-feed "./db2disk.py -a ../config/pipeline.conf -b BASEBAND2POWER"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer data from memeory to disk')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--data_source', type=str, nargs='+',
                        help='The source of data')
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    data_source   = args.data_source[0]

    ddir          = ConfigSectionMap(pipeline_conf, "{:s}".format(data_source))['dir']
    key           = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "{:s}".format(data_source))['key']), 0), 'x')

    db2disk = "dada_dbdisk -k {:s} -D {:s} -s -W".format(key, ddir)
    print db2disk
    os.system(db2disk)
