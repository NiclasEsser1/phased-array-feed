import subprocess
import time
import os
import pty
import argparse
from argparse import RawTextHelpFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch container for PAF pipeline development')
    parser.add_argument('-i', '--numa_id', action="store", dest="nid", help='The ID of NUMA node')

    disk="/beegfsEDD/NESSER"
    dockerimage="edd01:5000/capture_bypassed_bmf"
    dockername="niclas_numa_"+parser.parse_args().nid

    cmd = "docker run --name="+dockername+" --rm \
        --privileged=true \
        --ipc=shareable \
        --cap-add=IPC_LOCK \
        --ulimit memlock=-1:-1 \
        --net=host \
        -v "+disk+":"+disk+" \
        -e DISPLAY \
        -e USER=root \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --cap-add=SYS_PTRACE \
        -it "+dockerimage+" /bin/bash -ic 'cd phased-array-feed/;git pull;python script/start_bypassed_capture.py -i "+parser.parse_args().nid+";bash'"


    pty, tty = pty.openpty()
    print(cmd)

    p1 = subprocess.Popen(cmd, shell=True,stdin=tty, stdout=tty, stderr=tty)

    raw_input("Press key to stop...")

    os.system("docker container stop "+dockername)
    print("container stopped")
