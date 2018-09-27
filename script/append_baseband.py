#!/usr/bin/env python 

import argparse
import glob
import os

# ./append_baseband.py -a /beegfs/DENG/AUG/baseband -b J1939+2134 -c 0 -d 44581 -e 2018-08-30-20:11:41 -f 459700000000
# ./append_baseband.py -a /beegfs/DENG/AUG/baseband -b J1939+2134 -c 1 -d 34727 -e 2018-08-30-20:11:41 -f 459700000000
# ./append_baseband.py -a /beegfs/DENG/AUG/baseband -b J1939+2134 -c 2 -d 29011 -e 2018-08-30-20:11:41 -f 459700000000
# ./append_baseband.py -a /beegfs/DENG/AUG/baseband -b J1939+2134 -c 3 -d 21871 -e 2018-08-30-20:11:41 -f 459700000000
# ./append_baseband.py -a /beegfs/DENG/AUG/baseband -b J1939+2134 -c 4 -d 11271 -e 2018-08-30-20:11:41 -f 459700000000
# ./append_baseband.py -a /beegfs/DENG/AUG/baseband -b J1939+2134 -c 5 -d 0     -e 2018-08-30-20:11:41 -f 459700000000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To append multiple baseband data files into single baseband data file')
    
    parser.add_argument('-a', '--base_dir', type=str, nargs='+',
                        help='The base directory name')
    parser.add_argument('-b', '--psr_name', type=str, nargs='+',
                        help='The pulsar name')    
    parser.add_argument('-c', '--chunk', type=int, nargs='+',
                        help='The chunk id')
    parser.add_argument('-d', '--shift', type=int, nargs='+',
                        help='To shift how many packets')    
    parser.add_argument('-e', '--utc_start', type=str, nargs='+',
                        help='To align to UTC_START')    
    parser.add_argument('-f', '--picoseconds', type=int, nargs='+',
                        help='To align to picoseconds')
    
    args        = parser.parse_args()
    base_dir    = args.base_dir[0]
    psr_name    = args.psr_name[0]
    chunk       = args.chunk[0]
    shift       = args.shift[0]
    utc_start   = args.utc_start[0]
    picoseconds = args.picoseconds[0]

    directory = "{:s}/{:s}/{:d}".format(base_dir, psr_name, chunk)
    glob_key = "{:s}/*.000000.dada".format(directory)
    files = sorted(glob.glob(glob_key))
    
    print "Working on the data files in {:s} ...".format(directory)

    dada_hdrsz = 4096
    pktsz      = 7232
    dtsz       = 7168
    nchunk     = 8
    out_fname  = "{:s}/append{:d}.dada".format(directory, chunk)
    out        = open(out_fname, "w")
    first      = True        
    for fname in files:
        print "Start {:s}".format(fname)
        with open(fname,"r") as f:
            if first:
                out.write(f.read(dada_hdrsz))
                f.seek(nchunk * shift * pktsz, 1)   # To align the data
                while True:
                    f.seek(pktsz - dtsz, 1)
                    dt = f.read(dtsz)
                    if dt == '':
                        break
                    else:
                        out.write(dt)                                        
                first = False
            else:
                f.seek(dada_hdrsz)                
                while True:
                    f.seek(pktsz - dtsz, 1)
                    dt = f.read(dtsz)
                    if dt == '':
                        break
                    else:
                        out.write(dt)
            f.close()            
        print "Finish {:s}".format(fname)
    out.close()

    os.system("dada_install_header -p UTC_START={:s} {:s}".format(utc_start, out_fname))
    os.system("dada_install_header -p UTC_START={:d} {:s}".format(picoseconds, out_fname))
