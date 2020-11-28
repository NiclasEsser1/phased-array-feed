import subprocess
import time
import os
import pty
import argparse
import socket
from argparse import RawTextHelpFormatter

DOCKERIMAGE = "edd01:5000/capture_bypassed_bmf"
DISK = "/beegfsEDD/NESSER"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch container for PAF pipeline development')
    parser.add_argument('-n', '--numa_name', action="store", dest="name", help='The ID of NUMA node')
    parser.add_argument('-c', '--cmd_file', action="store", dest="cmd_file", help='The ID of NUMA node')

    dockername="capture_bypassed_bmf_"+parser.parse_args().name
    cmd_file=parser.parse_args().cmd_file
    args_file = open(cmd_file)
    capture_main_args = args_file.readline()
    args_file.close()
    os.remove(cmd_file)

    print("SOS"+capture_main_args+"EOS")
    cmd = "docker run --name="+dockername+" --rm \
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
        -it "+DOCKERIMAGE+" /bin/bash -ic 'cd phased-array-feed/;git pull; bash'"
    #
    #
    # pty, tty = pty.openpty()
    # print(cmd)
    #
    # # p1 = subprocess.Popen(cmd, shell=True)#,stdin=tty, stdout=tty, stderr=tty)
    # print("Entering docker " + dockername)
    os.system(cmd)


    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(host_ip, host_port)
    data = recv(1024)
    os.system("docker container stop "+dockername)


    # print("dockername)
    # raw_input("Press key to stop...")
    # time.sleep(20)

    #
    # print("container stopped")
