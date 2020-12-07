import subprocess
import time
import os
import pty
import argparse
import socket
from argparse import RawTextHelpFormatter
from subprocess import Popen, PIPE


DOCKERIMAGE = "edd01:5000/capture_bypassed_bmf_2"
DISK = "/beegfsEDD/NESSER"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch container for PAF pipeline development')
    parser.add_argument('-n', '--numa_name', action="store", dest="name", help='The ID of NUMA node')
    parser.add_argument('-c', '--cmd_file', action="store", dest="cmd_file", help='The ID of NUMA node')

    dockername = parser.parse_args().name
    cmd_file = parser.parse_args().cmd_file
    with open(cmd_file) as f:
        cmd_list = f.read().splitlines()
    #
    remove_dada_cmd = cmd_list[0]
    setup_dada_cmd = cmd_list[1]
    setup_dada_disk_cmd = cmd_list[2]
    print(remove_dada_cmd)
    print(setup_dada_cmd)
    print(setup_dada_disk_cmd)
    f.close()
    # os.remove(cmd_file)

    docker_cmd = "docker run --name="+dockername+" --rm \
        --privileged=true \
        --ipc=shareable \
        --cap-add=IPC_LOCK \
        --ulimit memlock=-1:-1 \
        --net=host \
        -v "+DISK+":"+DISK+" \
        -e DISPLAY \
        -e USER=root \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --cap-add=SYS_PTRACE \
        -it "+DOCKERIMAGE+" /bin/bash -ic 'cd /phased-array-feed/;git pull;cd /phased-array-feed/src/capture_bypassed_bmf/;make;"+remove_dada_cmd+";"+setup_dada_cmd+";"+setup_dada_disk_cmd+";bash'"
    print(docker_cmd)
    os.system(docker_cmd)
