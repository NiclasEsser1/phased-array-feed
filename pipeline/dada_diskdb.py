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

# ./dada_dbdisk.py -a ../config/pipeline.conf -b 0 -c 0 -d /beegfs/DENG/docker
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
    parser.add_argument('-e', '--fname', type=str, nargs='+',
                        help='The name of DADA file')
    parser.add_argument('-f', '--byte', type=int, nargs='+',
                        help='Byte to seek into the file')
    
    uid = 50000
    gid = 50000
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    beam          = int(args.beam[0])
    part          = int(args.part[0])
    directory     = args.directory[0]
    fname         = args.fname[0]
    byte          = int(args.byte[0])
    
    diskdb_container_name  = "paf-diskdb.beam{:02d}part{:02d}".format(beam, part)
    nblk          = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nblk'])
    dvolume       = '{:s}:{:s}'.format(directory, directory)
    hvolume       = "/home/pulsar:/home/pulsar"
    pktsz         = int(ConfigSectionMap(pipeline_conf, "DISKDB")['pktsz'])
    ndf_chk_rbuf  = int(ConfigSectionMap(pipeline_conf, "DISKDB")['ndf_chk_rbuf'])
    nreader       = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nreader'])
    nchk_beam     = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nchk_beam'])
    blksz         = pktsz * ndf_chk_rbuf * nchk_beam

    memsize = blksz * (nblk + 1)  # + 1 to be safe
    com_line = "docker run --ipc=shareable --rm -it -v {:s} -v {:s} -u {:d}:{:d} --ulimit memlock={:d} --name {:s} xinpingdeng/paf-general -a {:s} -b {:d} -c {:d} -d {:s} -e {:s} -f {:d}".format(dvolume, hvolume, uid, gid, memsize, diskdb_container_name, pipeline_conf, beam, part, directory, fname, byte)
    print com_line
    os.system(com_line)
