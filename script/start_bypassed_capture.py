import socket
import time
import numpy as np
import struct
import os
import subprocess
from astropy.time import Time
import argparse
from argparse import RawTextHelpFormatter

PAF_DF_PACKETSZ = 7232
PAF_PERIOD = 27

# EPOCH in BMF packet header]
EPOCHS = [
    [Time("2025-07-01T00:00:00", format='isot', scale='utc'), 51],
    [Time("2025-01-01T00:00:00", format='isot', scale='utc'), 50],
    [Time("2024-07-01T00:00:00", format='isot', scale='utc'), 49],
    [Time("2024-01-01T00:00:00", format='isot', scale='utc'), 48],
    [Time("2023-07-01T00:00:00", format='isot', scale='utc'), 47],
    [Time("2023-01-01T00:00:00", format='isot', scale='utc'), 46],
    [Time("2022-07-01T00:00:00", format='isot', scale='utc'), 45],
    [Time("2022-01-01T00:00:00", format='isot', scale='utc'), 44],
    [Time("2021-07-01T00:00:00", format='isot', scale='utc'), 43],
    [Time("2021-01-01T00:00:00", format='isot', scale='utc'), 42],
    [Time("2020-07-01T00:00:00", format='isot', scale='utc'), 41],
    [Time("2020-01-01T00:00:00", format='isot', scale='utc'), 40],
    [Time("2019-07-01T00:00:00", format='isot', scale='utc'), 39],
    [Time("2019-01-01T00:00:00", format='isot', scale='utc'), 38],
    [Time("2018-07-01T00:00:00", format='isot', scale='utc'), 37],
    [Time("2018-01-01T00:00:00", format='isot', scale='utc'), 36],
]

def refinfo(ip, port):
    """
    To get reference information for capture
    """
    data = bytearray(PAF_DF_PACKETSZ)
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Force to timeout after one data frame period
    sock.settimeout(1)
    server_address = (ip, port)
    sock.bind(server_address)

    nbyte, address = sock.recvfrom_into(data, PAF_DF_PACKETSZ)
    data = np.fromstring(str(data), dtype='uint64')

    hdr_part = np.uint64(struct.unpack(
        "<Q", struct.pack(">Q", data[0]))[0])
    sec_ref = (hdr_part & np.uint64(
        0x3fffffff00000000)) >> np.uint64(32)
    idf_ref = hdr_part & np.uint64(0x00000000ffffffff)

    hdr_part = np.uint64(struct.unpack(
        "<Q", struct.pack(">Q", data[1]))[0])
    epoch_idx = (hdr_part & np.uint64(
        0x00000000fc000000)) >> np.uint64(26)

    hdr_part = np.uint64(struct.unpack(
        "<Q", struct.pack(">Q", data[2]))[0])
    freq = (hdr_part & np.uint64(
        0x00000000ffff0000)) >> np.uint64(16)

    for epoch in EPOCHS:
        if epoch[1] == epoch_idx:
            break
    epoch_ref = int(epoch[0].unix / 86400.0)

    sock.close()

    return epoch_ref, int(sec_ref), int(idf_ref), float(freq)

def synced_refinfo(utc_start_capture, ip, port):
        # Capture one packet to see what is current epoch, seconds and idf
        # We need to do that because the seconds is not always matched with
        # estimated value
        epoch_ref, sec_ref, idf_ref, freq = refinfo(ip, port)

        while utc_start_capture.unix > (epoch_ref * 86400.0 + sec_ref + PAF_PERIOD):
            sec_ref = sec_ref + PAF_PERIOD
        while utc_start_capture.unix < (epoch_ref * 86400.0 + sec_ref):
            sec_ref = sec_ref - PAF_PERIOD

        idf_ref = (utc_start_capture.unix - epoch_ref *
                   86400.0 - sec_ref) / PAF_PERIOD

        return epoch_ref, sec_ref, int(idf_ref), freq



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options', formatter_class=RawTextHelpFormatter)
    # parser.add_argument('--key', '-k', action = "store", default ="dada", dest = "key", help = "Shared memory key")
    parser.add_argument('--node_id', '-nid', action = "store", default =0, dest = "nodeid", help = "ID of the numa node")
    # parser.add_argument('--time_ref', '-ref', action = "store", dest = "time_ref", help = "Time reference string epoch_seconds_dataframe")
    # parser.add_argument('--ip_addr', '-ip', action = "store", dest = "ip_addr", help = "Ip address")
    # parser.add_argument('--freq', '-f', action = "store", dest = "freq", help = "Base frequency of channel group")
    # parser.add_argument('--capture_conf', '-c', action = "store", dest = "capture_conf", help = "Capture conf of single capture thread: ip_port_exepectedbeams_actualbeams_cpuid")
    # parser.add_argument('--packets', '-p', action = "store", dest = "packets", help = "Number of packets stored in one ringerbuffer block. Note: packet*nof_beam*packets_size = ringubffer_size")
    # parser.add_argument('--temp_packets', '-tp', action = "store", dest = "temp_packets", default=128, help = "Number of packets stored temporary buffer. Note: temp_packet*nof_beam*packets_size = tempbuffer_size")

    # key = parser.parse_args().key
    nodeid = parser.parse_args().nodeid
    # time_ref = parser.parse_args().time_ref
    # freq = parser.parse_args().freq
    # capture_conf = parser.parse_args().capture_conf
    # packets = int(parser.parse_args().packets)
    # temp_packets = parser.parse_args().temp_packets
    # ip_addr = parser.parse_args().ip_addr

    print(nodeid)
    # block_size = packets*36*PAF_DF_PACKETSZ
    # nof_blocks = 8
    # disk_folder = "/beegfsEDD/NESSER"
    # # print(block_size)
    # # print("numactl -m "+str(nodeid)+" dada_db -k "+key+" -l -p -b "+str(block_size)+" -n "+str(nof_blocks))
    # os.system("dada_db -k "+key+" -d")
    # os.system("numactl -m "+str(nodeid)+" dada_db -k "+key+" -l -p -b "+str(block_size)+" -n "+str(nof_blocks))
    # os.system("numactl -m 0 dada_dbdisk -k "+key+" -D "+disk_folder+" -W -d -s")
    #
    # _utc_start_capture = Time(Time.now(), format='isot', scale='utc')
    # epoch_ref, sec_ref, idf_ref, freq = synced_refinfo(_utc_start_capture, ip_addr, 17100)
    # time_ref = str(epoch_ref)+"_"+str(sec_ref)+"_"+str(idf_ref)
    #
    # if nodeid == 0:
    #     capture_conf = "-c 10.17.1.1_17100_9_9_2 -c 10.17.1.1_17101_9_9_3 -c 10.17.1.1_17102_9_9_4 -c 10.17.1.1_17103_9_9_5"
    #     capture_ctrl_cpu = "0_1"
    #     buffer_ctrl_cpu = "0"
    # else:
    #     capture_conf = "-c 10.17.1.2_17100_9_9_12 -c 10.17.1.2_17101_9_9_13 -c 10.17.1.2_17102_9_9_14 -c 10.17.1.2_17103_9_9_15"
    #     capture_ctrl_cpu = "0_11"
    #     buffer_ctrl_cpu = "10"
    #
    # cmd = "numactl -m "+str(nodeid)+" /home/pulsar/nesser/Projects/phased-array-feed/src/capture_bypassed_bmf/capture_main " + "-a "+key+" -b 0 "+capture_conf+" -e 1337.0 -f "+time_ref+" -g ../../log/numa1_pacifix1 -i "+str(buffer_ctrl_cpu)+" -j "+str(capture_ctrl_cpu)+" -k 1 -l "+str(packets)+" -m "+str(temp_packets)+" -n header_dada.txt -o UNKOWN_00:00:00.00_00:00:00.00 -p 0 -q "+str(freq)
    # print(cmd)
    # p = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
