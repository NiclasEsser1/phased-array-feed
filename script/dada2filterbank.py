#!/usr/bin/env python

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='To replace dada header with filterbank header')
    
    parser.add_argument('-a', '--dada_fname', type=str, nargs='+',
                        help='The dada file name')   
    parser.add_argument('-b', '--filterbank_fname', type=str, nargs='+',
                        help='The filterbank file name')   
    parser.add_argument('-c', '--header_fname', type=str, nargs='+',
                        help='The filterbank header file name')   

    ddir             = "/beegfs/DENG/pacifix7_numa1_process0"
    args             = parser.parse_args()
    dada_fname       = args.dada_fname[0]
    filterbank_fname = args.filterbank_fname[0]
    header_fname     = args.header_fname[0]
    
    d_fname = "{:s}/{:s}".format(ddir, dada_fname)
    f_fname = "{:s}/{:s}".format(ddir, filterbank_fname)
    
    f_hdrsz = 342
    d_hdrsz = 4096
    pktsz   = 1024 * 1024
    
    # Read in filterbank header 
    h_fname   = "{:s}/{:s}".format(ddir, header_fname)
    h_file    = open(h_fname, "r")
    hdr       = h_file.read(f_hdrsz)
    h_file.close()

    # Do the work
    f_file = open(f_fname, "w")
    f_file.write(hdr)

    d_file = open(d_fname, "r")
    d_file.seek(d_hdrsz)
    
    while True:
        dt = d_file.read(pktsz)
        if dt == '':
            break
        else:
            f_file.write(dt)
       
    d_file.close()
    f_file.close()
