#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json
import numpy as np

def ConfigSectionMap(conf_fname, section):
    # Play with configuration file
    Config = ConfigParser.ConfigParser()
    Config.read(conf_fname)
    
    dict_conf = {}
    options = Config.options(section)
    for option in options:
        try:
            dict_conf[option] = Config.get(section, option)
            if dict_conf[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict_conf[option] = None
    return dict_conf

def available_beam(multicast_data, nbeam):
    routing_table = multicast_data['routing_table']
    beams = []
    for beam in range(nbeam):
        ip = []
        for table_line in routing_table:
            chk_ip   = table_line.split(',')[3 * beam + 2]
            if(chk_ip != "0.0.0.0"):
                ip.append(chk_ip)
        if(len(ip)!=0):
            beams.append(beam)            
    return np.array(beams)  
    
def destination(multicast_data, conf_fname, beam):    
    # check routing table
    routing_table = multicast_data['routing_table']
    chk           = 0
    address       = []
    ip_chk        = []
    ip            = []
    for table_line in routing_table:
        chk_ip   = table_line.split(',')[3 * beam + 2]
        chk_port = table_line.split(',')[3 * beam + 3]
        if(chk_ip != "0.0.0.0" and chk_port != "0"):
            address.append("{:s}:{:s}".format(chk_ip, chk_port))
            ip_chk.append("{:s}:{:d}".format(chk_ip, chk))
            ip.append(chk_ip)
        
        chk = chk + 1

    # Get software input information from the routing table information
    # IP, port, the number of chunks on each port, chunk maximum and minimum, etc ...
    nchk_beam = int(ConfigSectionMap(conf_fname, "EthernetInterfaceBMF")['nchk_beam'])
    address   = {x:address.count(x) for x in address}        
    ip        = sorted(list((set(ip))))       # Which ip we receive data
    nip       = len(ip)
    max_min_chk  = []   # The maximum and minimum frequency chunks of each IP
    address_nchk = []   # Port and number of frequency chunks on each port of each IP
    node         = []
    for item_ip in ip:
        min_chk = nchk_beam
        max_chk = 0
        address_nchk_temp = []
        for i in range(len(address)):  
            if item_ip == address.keys()[i].split(":")[0]:
                address_nchk_temp.append("{:s}:{:d}".format(address.keys()[i], address.values()[i]))
        address_nchk.append(address_nchk_temp)

        for i in range(len(ip_chk)):
            if item_ip == ip_chk[i].split(":")[0]:
                if min_chk > int(ip_chk[i].split(":")[1]):
                    min_chk = int(ip_chk[i].split(":")[1])
                if max_chk < int(ip_chk[i].split(":")[1]):
                    max_chk = int(ip_chk[i].split(":")[1])
        max_min_chk.append([min_chk, max_chk])
        node.append("pulsar@pacifix{:d}.mpifr-bonn.mpg.de".format(int(item_ip.split(".")[2])))
        
    for i in range(nip):
        address_nchk[i] = sorted(address_nchk[i]) # Sort it to make it in order

    print "For beam {:d}, we will receive data with {:d} NiC, the detail with the format [IP:PORT:NCHUNK] is as follow ...".format(beam, nip)
    for i in range(nip):
        print "\t{:s}".format(address_nchk[i])
    
    # Get the number of channel on each IP
    freq0     = float(multicast_data['sky_frequency'])
    nchan_chk = int(ConfigSectionMap(conf_fname, "EthernetInterfaceBMF")['nchan_chk'])
    nchan     = np.zeros(nip, dtype=int)
    freq      = np.zeros(nip, dtype=float)
    for i in range(nip):
        nchan[i] = (max_min_chk[i][1] - max_min_chk[i][0] + 1) * nchan_chk;
        freq[i]  = freq0 - 0.5 * (nchk_beam - (max_min_chk[i][1] + max_min_chk[i][0] + 1)) * nchan_chk 
        print "The center frequency of data from {:s} is {:.1f} MHz with {:d} channels, the login detail is \"{:s}\".".format(ip[i], freq[i], nchan[i], node[i])
    print "\n"
    return node, address_nchk
    
def main(args):
    args       = parser.parse_args()
    conf_fname = args.conf_fname[0]

    # Configure metadata interface
    MCAST_GRP  = ConfigSectionMap(conf_fname, "MetadataInterfaceTOS")['ip']
    MCAST_PORT = int(ConfigSectionMap(conf_fname, "MetadataInterfaceTOS")['port'])
    MCAST_ADDR = ('', MCAST_PORT)
    
    # Create the socket and get ready to receive data
    sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create the socket
    group = socket.inet_aton(MCAST_GRP)
    mreq  = struct.pack('4sL', group, socket.INADDR_ANY)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(MCAST_ADDR)                                    # Bind to the server address
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)  # Tell the operating system to add the socket to the multicast group on all interfaces.
    
    # Get data packet
    pkt, addr      = sock.recvfrom(1<<16)
    multicast_data = json.loads(pkt)

    # Get information from metadata packet, which is shared with all beams
    target_name = multicast_data['target_name']
    ra  = float(multicast_data['beams_direction']['beam01'][0]) # RA in decimal radian, we may need to convert it to other unit
    dec = float(multicast_data['beams_direction']['beam01'][1]) # DEC in decimal radian, we may need to convert it to other unit

    # To get available beams 
    nbeam = 36
    beams = available_beam(multicast_data, nbeam)
    print "The available beams are {:s}, counting from 0 ...\n".format(beams)
    
    # Get the desination of a given beam
    for beam in beams:
        node, address_nchk = destination(multicast_data, conf_fname, beam)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read TOS metadata and get required information')
    parser.add_argument('-a', '--conf_fname', type=str, nargs='+',
                        help='The name of configuration file')

    args = parser.parse_args()
    main(args)
