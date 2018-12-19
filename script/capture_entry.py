#!/usr/bin/env python

import ConfigParser, parser, argparse, socket, struct, json, os, subprocess, threading, datetime, time
import numpy as np
import captureinfo, metadata2streaminfo

def ConfigSectionMap(fname, section):
    # Play with configuration file
    Config = ConfigParser.ConfigParser()
    Config.read(fname)
    
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

def main(system_conf, pipeline_conf, bind, hdr, nchan, freq, address_nchk, ctrl_socket, instrument, beam, part):
    dir_capture = ConfigSectionMap(pipeline_conf, "CAPTURE")['dir']
    destination_alive, destination_dead, refinfo, key = captureinfo.captureinfo(pipeline_conf, system_conf, address_nchk, nchan, hdr)
    print(datetime.datetime.now())
        
    if (len(destination_alive) == 0):
        print "There is no alive port for beam {:02d}, part {:02d}, have to abort ...".format(beam, part)
        exit(1)
        
    # Put the key into a file    
    #kfile_prefix = ConfigSectionMap(pipeline_conf, "CAPTURE")['kfname_prefix']
    #kfname       = "{:s}.beam{:02d}part{:02d}.key".format(kfile_prefix, beam, part)
    kfname       = "capture.beam{:02d}part{:02d}.key".format(beam, part)
    kfile = open(kfname, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key))
    kfile.close()
    
    # To set up cpu cores if we decide to bind threads
    ncpu_numa = int(ConfigSectionMap(system_conf, "NUMA")['ncpu_numa'])
    port0     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['port0'])
    node      = int(destination_alive[0].split(":")[0].split(".")[3])
    cpus      = []
    for i in range(len(destination_alive)):  # The cpu thing here is not very smart
        #cpu = (node - 1) * ncpu_numa + (int(destination_alive[i].split(":")[1]) - port0)%ncpu_numa
        cpu = (beam % 4) * ncpu_numa / 2 + i 
        destination_alive[i] = "{:s}:{:d}".format(destination_alive[i], cpu)
        cpus.append(cpu)
    cpus = np.array(cpus)

    # To setup buffer size for single packet and reading start of it    
    nsamp_df     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    nchan_chk    = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nchan_chk'])
    df_hdrsz     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_hdrsz'])
    ndf_chk_prd  = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndf_chk_prd'])
    ndf_chk_rbuf = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_chk_rbuf'])
    ndf_chk_tbuf = int(ConfigSectionMap(pipeline_conf, "CAPTURE")['ndf_chk_tbuf'])
    ddir         = "{:s}/beam{:02d}".format(ConfigSectionMap(pipeline_conf, "CAPTURE")['dir'], beam)
    hdr_fname    = ConfigSectionMap(pipeline_conf, "CAPTURE")['hdr_fname']
    pktsz        = npol_samp * ndim_pol * nbyte_dim * nchan_chk * nsamp_df + df_hdrsz
    if(hdr == 0):
        pktoff = df_hdrsz                                                 # The start point of each BMF packet
    else:
        pktoff = 0

    period = 27
    
    # Do the real work here
    sec_prd = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['sec_prd'])
    nchunk = nchan/nchan_chk
    if (len(destination_dead) == 0):
        #capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -f {:f} -g {:d} -i {:f}:{:d}:{:d} -j {:s} -k {:d} -l {:d} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:d} -s {:s} -t {:s} -u {:s}".format(key, pktsz, pktoff, " -d ".join(destination_alive), freq, nchan, refinfo[0], refinfo[1], refinfo[2], dir_capture, (node - 1) * ncpu_numa + (max(cpus) + 1)%ncpu_numa, (node - 1) * ncpu_numa + (max(cpus) + 1)%ncpu_numa, bind, sec_prd, nchunk, ndf_chk_rbuf, ndf_chk_tbuf, ndf_chk_prd, ctrl_socket, hdr_fname, instrument)
        capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -f {:f} -g {:d} -i {:d}:{:d}:{:d} -j {:s} -k {:d} -l {:s} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:s} -s {:s} -t {:s}".format(key, pktsz, pktoff, " -d ".join(destination_alive), freq, nchan_chk, int(refinfo[0]), refinfo[1], refinfo[2], ddir, max(cpus)+1, "1:{:d}:capture.socket{:02d}".format(max(cpus)+1, beam), bind, period, ndf_chk_rbuf, 250, 250000, "../config/header_16bit.txt", "PAF-BMF", "UNKNOW:00 00 00.00:00 00 00.00")
    else:
        #capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -e {:s} -f {:f} -g {:d} -i {:f}:{:d}:{:d} -j {:s} -k {:d} -l {:d} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:d} -s {:s} -t {:s} -u {:s}".format(key, pktsz, pktoff, " -d ".join(destination_alive), " -e ".join(destination_dead), freq, nchan, refinfo[0], refinfo[1], refinfo[2], dir_capture, (node - 1) * ncpu_numa + (max(cpus) + 1)%ncpu_numa, (node - 1) * ncpu_numa + (max(cpus) + 1)%ncpu_numa, bind, sec_prd, nchunk, ndf_chk_rbuf, ndf_chk_tbuf, ndf_chk_prd, ctrl_socket, hdr_fname, instrument)
        capture_command = "../src/capture/capture_main -a {:s} -b {:d} -c {:d} -d {:s} -f {:f} -g {:d} -i {:d}:{:d}:{:d} -j {:s} -k {:d} -l {:s} -m {:d} -n {:d} -o {:d} -p {:d} -q {:d} -r {:s} -s {:s} -t {:s}".format(key, pktsz, pktoff, " -d ".join(destination_alive), freq, nchan_chk, int(refinfo[0]), refinfo[1], refinfo[2], ddir, max(cpus)+1, "1:{:d}:capture.socket{:02d}".format(max(cpus)+1, beam), bind, period, ndf_chk_rbuf, 250, 250000, "../config/header_16bit.txt", "PAF-BMF", "UNKNOW:00 00 00.00:00 00 00.00")
    
    print capture_command
    os.system(capture_command)
    
    # Delete PSRDADA buffer
    os.system("dada_db -d -k {:s}".format(key))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To capture data from given beam and given part')    
    parser.add_argument('-a', '--system_conf', type=str, nargs='+',
                        help='The configuration of PAF system')
    parser.add_argument('-b', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-c', '--hdr', type=int, nargs='+',
                        help='Record packet header or not')
    parser.add_argument('-d', '--bind', type=int, nargs='+',
                        help='Bind threads to cpu or not')
    parser.add_argument('-e', '--nchan', type=int, nargs='+',
                        help='The number of channels')
    parser.add_argument('-f', '--freq', type=float, nargs='+',
                        help='The center frequency in MHz')
    parser.add_argument('-g', '--address_nchk', type=str, nargs='+',
                        help='The address for current capture')
    parser.add_argument('-i', '--ctrl_socket', type=str, nargs='+',
                        help='The control socket')
    parser.add_argument('-j', '--instrument', type=str, nargs='+',
                        help='The control socket')
    parser.add_argument('-k', '--beam', type=int, nargs='+',
                        help='The beam id from 0')
    parser.add_argument('-l', '--part', type=int, nargs='+',
                        help='The part id from 0')
    
    args          = parser.parse_args()
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    hdr           = args.hdr[0]
    bind          = args.bind[0]
    nchan         = args.nchan[0]
    freq          = args.freq[0]
    address_nchk  = args.address_nchk
    ctrl_socket   = args.ctrl_socket[0]
    instrument    = args.instrument[0]
    beam          = args.beam[0]
    part          = args.part[0]

    main(system_conf, pipeline_conf, bind, hdr, nchan, freq, address_nchk, ctrl_socket, instrument, beam, part)
