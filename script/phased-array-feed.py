#!/usr/bin/env python

import parser, argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To run a script specified by the command line')
    parser.add_argument('command_line', metavar='c', type=str, nargs='+',
                    help='The command line')
    args         = parser.parse_args()
    command_line = args.command_line[0]
    
    print "We will run", command_line
    os.system(command_line)
