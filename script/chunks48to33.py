#!/usr/bin/env python

import argparse
import ConfigParser
from os.path import join

HDR_SIZE        = 4096
PKT_SIZE        = 7168
NCHK_DROP1      = 7
NCHK_DROP2      = 8
NCHK_KEEP       = 33
FIRST_SKIP_SIZE = NCHK_DROP1 * PKT_SIZE
SKIP_SIZE       = (NCHK_DROP1 + NCHK_DROP2) * PKT_SIZE
KEEP_SIZE       = NCHK_KEEP * PKT_SIZE

# ./chunks48to33.py -a /beegfs/DENG/AUG/baseband/J1819-1458 -b J1819-1458.dada -c J1819-1458_33chunks.dada

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
    with open(iname, "r") as ifile:
        ofile.write(ifile.read(HDR_SIZE))
        ifile.read(FIRST_SKIP_SIZE)
        while ifile:
            ofile.write(ifile.read(KEEP_SIZE))
            ifile.read(SKIP_SIZE)
        
