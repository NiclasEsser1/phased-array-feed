#!/usr/bin/env python

import parser, argparse, socket, fcntl, struct

NDF   = 1024
PKTSZ = 7232

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

def port_status(ip, port):
    active = 1
    nchunk_active = 0
    data = bytearray(PKTSZ) 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (ip, int(port))
    sock.bind(server_address)
    
    try:
        nbyte, address = sock.recvfrom_into(data, PKTSZ)
        if (nbyte != PKTSZ):
            active = 0
        else:
            source = []
            active = 1
            for i in range(NDF):
                buf, address = sock.recvfrom(PKTSZ)
                source.append(address)
            nchunk_active = len(set(source))
    except:
        active = 0        
    return nchunk_active

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To check the status of BMF stream')
    parser.add_argument('-a', '--eth', type=str, nargs='+',
                        help='The NiC interface')
    parser.add_argument('-b', '--port', type=int, nargs='+',
                        help='The port number inuse on given NiC')

    args = parser.parse_args()
    eth  = args.eth[0]
    port = args.port
    
    ip = get_ip_address(eth)
    active = []
    for p in port:
        active.append(port_status(ip, p))
    print "The number of active frequency chunks are:", active
