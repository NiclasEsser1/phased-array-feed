#!/usr/bin/env python

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='To replace dada header with filterbank header')
    
    parser.add_argument('-a', '--input_source', type=str, nargs='+',
                        help='The input source')   
    parser.add_argument('-b', '--filterbank_fname', type=str, nargs='+',
                        help='The filterbank file name')   
    parser.add_argument('-c', '--directory', type=str, nargs='+',
                        help='The directory to put files')   
    
    args             = parser.parse_args()
    input_source     = args.input_source[0]
    filterbank_fname = args.filterbank_fname[0]
    directory        = args.directory[0]
    filterbank_fname = "{}/{}".format(directory, filterbank_fname)

    os.system("/home/pulsar/xinping/phased-array-feed/src/dada2filterbank_main -a {} -b {} -c {}".format(input_source, filterbank_fname, directory))
