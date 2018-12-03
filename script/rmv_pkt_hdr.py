#!/usr/bin/env python 

parser = argparse.ArgumentParser(description='To remove header of each packet from appended baseband data file')
    
    parser.add_argument('-a', '--base_dir', type=str, nargs='+',
                        help='The base directory name')
    parser.add_argument('-b', '--psr_name', type=str, nargs='+',
                        help='The pulsar name')    
    parser.add_argument('-c', '--chunk', type=int, nargs='+',
                        help='The chunk id')

    args          = parser.parse_args()
    base_dir      = args.base_dir[0]
    psr_name      = args.psr_name[0]
    chunk         = args.chunk[0]
    directory     = "{:s}/{:s}/{:d}".format(base_dir, psr_name, chunk)
    fname         = ""
