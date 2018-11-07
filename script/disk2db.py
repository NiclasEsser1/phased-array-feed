#!/usr/bin/env python

import os, parser, argparse, ConfigParser, time

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

# docker run --ipc=shareable --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name disk2db xinpingdeng/phased-array-feed "./disk2db.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J1819-1458/J1819-1458.dada -d 0"
# docker run --ipc=shareable --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name disk2db xinpingdeng/phased-array-feed "./disk2db.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J1713+0747/J1713+0747.dada -d 0"
# docker run --ipc=shareable --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name disk2db xinpingdeng/phased-array-feed "./disk2db.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J1939+2134/J1939+2134.dada -d 0"
# docker run --ipc=shareable --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name disk2db xinpingdeng/phased-array-feed "./disk2db.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J0332+5434/J0332+5434.dada -d 0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer data from disk to memory')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The configuration of system')  
    parser.add_argument('-c', '--fname', type=str, nargs='+',
                        help='file name with full path')
    parser.add_argument('-d', '--byte', type=int, nargs='+',
                        help='Byte to seek into the file')
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    system_conf   = args.system_conf[0]
    fname         = args.fname[0]
    byte          = args.byte[0]
    
    nblk          = int(ConfigSectionMap(pipeline_conf, "DISK2DB")['nblk'])
    pktsz         = int(ConfigSectionMap(pipeline_conf, "DISK2DB")['pktsz'])
    nreader       = int(ConfigSectionMap(pipeline_conf, "DISK2DB")['nreader'])
    nchk_beam     = int(ConfigSectionMap(pipeline_conf, "DISK2DB")['nchk_beam'])
    ndf_chk_rbuf  = int(ConfigSectionMap(pipeline_conf, "DISK2DB")['ndf_chk_rbuf'])
    key           = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "DISK2DB")['key']), 0), 'x')
    blksz         = pktsz * ndf_chk_rbuf * nchk_beam

    # Do the work
    db_create = "dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key, blksz, nblk, nreader)
    print db_create
    os.system(db_create)
    
    disk2db = "dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(key, fname, byte)
    print disk2db
    os.system(disk2db)    

    time.sleep(10)
    db_destroy = "dada_db -d -k {:s}".format(key)
    print db_destroy
    os.system(db_destroy)
