import socket
import time
import numpy as np
import struct
import os
from astropy.time import Time

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

    for epoch in EPOCHS:
        if epoch[1] == epoch_idx:
            break
    epoch_ref = int(epoch[0].unix / 86400.0)

    sock.close()

    return epoch_ref, int(sec_ref), int(idf_ref)

def synced_refinfo(utc_start_capture, ip, port):
        # Capture one packet to see what is current epoch, seconds and idf
        # We need to do that because the seconds is not always matched with
        # estimated value
        epoch_ref, sec_ref, idf_ref = refinfo(ip, port)

        while utc_start_capture.unix > (epoch_ref * 86400.0 + sec_ref + PAF_PERIOD):
            sec_ref = sec_ref + PAF_PERIOD
        while utc_start_capture.unix < (epoch_ref * 86400.0 + sec_ref):
            sec_ref = sec_ref - PAF_PERIOD

        idf_ref = (utc_start_capture.unix - epoch_ref *
                   86400.0 - sec_ref) / PAF_PERIOD

        return epoch_ref, sec_ref, int(idf_ref)



_utc_start_capture = Time(Time.now(), format='isot', scale='utc')
# epoch_ref, sec_ref, idf_ref = _synced_refinfo("10.17.1.1", 17100)
epoch_ref, sec_ref, idf_ref = synced_refinfo(_utc_start_capture, "10.17.1.1", 17100)
command = "./capture_main -a dada -b 0 -c 10.17.1.1_17100_9_9_3 -c 10.17.1.1_17101_9_9_3 -c 10.17.1.1_17102_9_9_3 -c 10.17.1.1_17103_9_9_3 -e 1337.0 -f "+str(epoch_ref)+"_"+str(sec_ref)+"_"+str(idf_ref)+" -g ../../log/numa1_pacifix1 -i 1 -j 0_2 -k 1 -l 16384 -m 128 -n header_dada.txt -o UNKOWN_00:00:00.00_00:00:00.00 -p 0 -q 1340"
print(command)
os.system(command)
