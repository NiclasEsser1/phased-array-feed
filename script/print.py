#!/usr/bin/env python

import parser, argparse

print "HERE\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To check the status of BMF stream')
    parser.add_argument('-a', '--eth', type=str, nargs='+',
                        help='The NiC interface')
    parser.add_argument('-b', '--port', type=int, nargs='+',
                        help='The port number inuse on given NiC')
    
    args = parser.parse_args()
    eth  = args.eth[0]
    port = args.port

    print eth, port
