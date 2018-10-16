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

# ./dada_diskdb.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J0332+5434/J0332+5434.dada -d 0 -e /beegfs/DENG/AUG/baseband/J0332+5434/ -f 0
# ./dada_diskdb.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J1713+0747/J1713+0747.dada -d 0 -e /beegfs/DENG/AUG/baseband/J1713+0747/ -f 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To transfer data from shared memeory to disk with a docker container')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The configuration of system')  
    parser.add_argument('-c', '--fname', type=str, nargs='+',
                        help='file name with directory')
    parser.add_argument('-d', '--byte', type=int, nargs='+',
                        help='Byte to seek into the file')
    parser.add_argument('-e', '--directory', type=str, nargs='+',
                        help='The directory of data')
    parser.add_argument('-f', '--baseband', type=int, nargs='+',
                        help='To fold baseband data or not')
        
    uid = 50000
    gid = 50000
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    system_conf   = args.system_conf[0]
    fname         = args.fname[0]
    byte          = int(args.byte[0])
    directory     = args.directory[0]
    baseband      = args.baseband[0]
    
    diskdb_container_name  = "paf-diskdb"
    nblk          = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nblk'])
    dvolume       = '{:s}:{:s}'.format(directory, directory)
    hvolume       = "/home/pulsar:/home/pulsar"
    pktsz         = int(ConfigSectionMap(pipeline_conf, "DISKDB")['pktsz'])
    ndf_chk_rbuf  = int(ConfigSectionMap(pipeline_conf, "DISKDB")['ndf_chk_rbuf'])
    nreader       = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nreader'])
    nchk_beam     = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nchk_beam'])
    blksz         = pktsz * ndf_chk_rbuf * nchk_beam
    script_name   = "/home/pulsar/xinping/phased-array-feed/script/dada_diskdb_entry.py"

    com_line = "docker run --ipc=shareable --rm -it -v {:s} -v {:s} -u {:d}:{:d} --ulimit memlock=-1:-1 --name {:s} xinpingdeng/phased-array-feed \"{:s} -a {:s} -b {:s} -c {:s} -d {:d} -e {:d}\"".format(dvolume, hvolume, uid, gid, diskdb_container_name, script_name, pipeline_conf, system_conf, fname, byte, baseband)
    print com_line
    os.system(com_line)
