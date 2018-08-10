#!/usr/bin/env python
import os, parser, argparse, ConfigParser
# ./baseband2baseband_demo_entry.py -a ../config/system.conf -b ../config/pipeline.conf  -c 0 -d 0 -e 1

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
    parser.add_argument('-e', '--fname', type=str, nargs='+',
                        help='The name of DADA file')
    parser.add_argument('-f', '--byte', type=int, nargs='+',
                        help='Byte to seek into the file')
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    part          = args.part[0]
    directory     = args.directory[0]
    fname         = args.fname[0]
    byte          = args.byte[0]
    
    key           = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "DISKDB")['key']), 0), 'x')
    nblk          = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nblk'])
    fname         = '{:s}/{:s}'.format(directory, fname)
    pktsz         = int(ConfigSectionMap(pipeline_conf, "DISKDB")['pktsz'])
    ndf_chk_rbuf  = int(ConfigSectionMap(pipeline_conf, "DISKDB")['ndf_chk_rbuf'])
    nreader       = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nreader'])
    nchk_beam     = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nchk_beam'])
    blksz         = pktsz * ndf_chk_rbuf * nchk_beam

    kfname       = "diskdb.beam{:02d}part{:02d}.key".format(beam, part)
    kfile = open(kfname, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key))
    kfile.close()
    
    os.system("dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key, blksz, nblk, nreader))
    print "DADA creat done"
    os.system("dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(key, fname, byte))
    os.system("dada_db -d -k {:s}".format(key))
