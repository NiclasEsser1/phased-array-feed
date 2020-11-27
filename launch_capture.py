import subprocess
import time
import os
import pty


DOCKERIMAGE="xinpingdeng/paf-base"
name_numa0 = "capture_numa0"
name_numa1 = "capture_numa1"
numa0 = "docker run --name="+name_numa0+" --rm \
        --privileged=true \
        --ipc=shareable \
        --cap-add=IPC_LOCK \
        --ulimit memlock=-1:-1 \
        --net=host \
        -v /home/pulsar/nesser:/home/pulsar/nesser \
        -v /beegfsEDD/NESSER:/beegfsEDD/NESSER \
        -e DISPLAY \
        -e USER=root \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --cap-add=SYS_PTRACE \
        -it xinpingdeng/paf-base /bin/bash -ic 'bash; pip install astropy; python /home/pulsar/nesser/Projects/phased-array-feed/src/capture_bypassed_bmf/start_capture.py -k dada -nid 0 -ip 10.17.1.1 -p 16384'"

numa1 = "docker run --name="+name_numa1+" --rm \
        --privileged=true \
        --ipc=shareable \
        --cap-add=IPC_LOCK \
        --ulimit memlock=-1:-1 \
        --net=host \
        -v /home/pulsar/nesser:/home/pulsar/nesser \
        -v /beegfsEDD/NESSER:/beegfsEDD/NESSER \
        -e DISPLAY \
        -e USER=root \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --cap-add=SYS_PTRACE \
        -it xinpingdeng/paf-base /bin/bash -ic 'bash; pip install astropy;python /home/pulsar/nesser/Projects/phased-array-feed/src/capture_bypassed_bmf/start_capture.py -k dadc -nid 1 -ip 10.17.1.2 -p 16384'"

pty, tty = pty.openpty()
#
p1 = subprocess.Popen(numa0, shell=True,stdin=tty, stdout=tty, stderr=tty)
p2 = subprocess.Popen(numa1, shell=True,stdin=tty, stdout=tty, stderr=tty)
# time.sleep(2)

raw_input("Press key to stop...")

os.system("docker container stop "+name_numa0)
os.system("docker container stop "+name_numa1)
print("container stopped")
