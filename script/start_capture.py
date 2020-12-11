import subprocess
import time
import os
import pty
import argparse
import socket
from argparse import RawTextHelpFormatter
from subprocess import Popen, PIPE

DISK = "/beegfsEDD/NESSER"
CONF_DIR="/phased-array-feed/config/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch container for PAF pipeline development')
    parser.add_argument('-n', '--numa_name', action="store", dest="name", help='The ID of NUMA node')
    parser.add_argument('-c', '--cmd_file', action="store", dest="cmd_file", help='The ID of NUMA node')
    parser.add_argument('-t', '--header_file', action="store", dest="header_file", help='The ID of NUMA node')

    dockername = parser.parse_args().name
    cmd_file = parser.parse_args().cmd_file
    header_file = str(parser.parse_args().header_file)
    with open(cmd_file) as f:
        cmd_list = f.read().splitlines()
    f.close()
    os.remove(cmd_file)
    capture_main_cmd = "docker exec -it "+dockername+" "+cmd_list[3]
    print(capture_main_cmd)
    # os.remove(cmd_file)
    os.system("docker cp " + header_file + " " + dockername + ":" + CONF_DIR)
    os.system(capture_main_cmd)
