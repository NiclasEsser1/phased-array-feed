#!/usr/bin/env python

import argparse
import ConfigParser
from os.path import join

HDR_SIZE        = 4096
PKT_SIZE        = 7168
NCHK_DROP1      = 18
NCHK_DROP2      = 0
NCHK_KEEP       = 30
FIRST_SKIP_SIZE = NCHK_DROP1 * PKT_SIZE
SKIP_SIZE       = (NCHK_DROP1 + NCHK_DROP2) * PKT_SIZE
KEEP_SIZE       = NCHK_KEEP * PKT_SIZE

# ./chunks48to30.py -a /beegfs/DENG/AUG/baseband/J1819-1458 -b J1819-1458_48chunks.dada -c J1819-1458_30chunks.dada
# ./chunks48to30.py -a /beegfs/DENG/AUG/baseband/J0332+5434 -b J0332+5434_48chunks.dada -c J0332+5434_30chunks.dada
# ./chunks48to30.py -a /beegfs/DENG/AUG/baseband/J1939+2134 -b J1939+2134_48chunks.dada -c J1939+2134_30chunks.dada
# ./chunks48to30.py -a /beegfs/DENG/AUG/baseband/J1713+0747 -b J1713+0747_48chunks.dada -c J1713+0747_30chunks.dada

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To convert baseband data from 48 chunks to 33 chunks")
    parser.add_argument('-a', '--dname', type=str, nargs='+',
                        help='The name of directory')
    parser.add_argument('-b', '--iname', type=str, nargs='+',
                        help='The name of input file')    
    parser.add_argument('-c', '--oname', type=str, nargs='+',
                        help='The name of output file')
    
    args     = parser.parse_args()
    dname = args.dname[0]
    iname = args.iname[0]
    oname = args.oname[0]

    iname = join(dname, iname)
    oname = join(dname, oname)

    ofile = open(oname, "w")
    keep_size = KEEP_SIZE
    skip_size = SKIP_SIZE
    
    with open(iname, "r") as ifile:
        ofile.write(ifile.read(HDR_SIZE))
        ifile.read(FIRST_SKIP_SIZE)
        while (keep_size == KEEP_SIZE) and (skip_size == SKIP_SIZE):
            keep_data = ifile.read(KEEP_SIZE)
            keep_size = len(keep_data)
            ofile.write(keep_data)
            
            skip_data = ifile.read(SKIP_SIZE)
            skip_size = len(skip_data)
