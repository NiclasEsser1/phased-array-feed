#!/usr/bin/env python

import argparse, os, datetime

def keyword_dada(dada_fname, keyword):    
    dada_hdrsz = 4096
    dada_file  = open(dada_fname)

    dada_hdr = dada_file.read(dada_hdrsz).split('\n')
    for dada_hdr_line in dada_hdr:
        if keyword in dada_hdr_line and dada_hdr_line[0] != '#':
            value = dada_hdr_line.split()[1]
            dada_file.close()
            return value
        
    print "Can not find the keyword \"{:s}\" in header ...".format(keyword)
    dada_file.close()
    exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To replace dada header with filterbank header')
    
    parser.add_argument('-a', '--dada_fname', type=str, nargs='+',
                        help='The dada file name')   
    parser.add_argument('-b', '--filterbank_fname', type=str, nargs='+',
                        help='The filterbank file name')   

    ddir             = "/beegfs/DENG/AUG/"
    args             = parser.parse_args()
    dada_fname       = os.path.join(ddir, args.dada_fname[0])
    filterbank_fname = os.path.join(ddir, args.filterbank_fname[0])

    utc_start   = keyword_dada(dada_fname, "UTC_START")
    mjd_start   = float(keyword_dada(dada_fname, "MJD_START"))
    picoseconds = float(keyword_dada(dada_fname, "PICOSECONDS"))
    tsamp       = float(keyword_dada(dada_fname, "TSAMP"))
    freq        = float(keyword_dada(dada_fname, "FREQ"))
    nchan       = float(keyword_dada(dada_fname, "NCHAN"))
    #bw          = float(keyword_dada(dada_fname, "BW"))
    bw          = -256.0 * 32.0 / 27.0
    
    mjd_start = mjd_start + picoseconds / 86400.0E12
    fch1      = freq - 0.5 * bw / nchan * (nchan - 1)
    foff      = bw / nchan
    utc_date  = datetime.datetime.strptime(utc_start, '%Y-%m-%d-%H:%M:%S').strftime('%Y/%m/%d')

    print fch1, foff
    
