#!/usr/bin/env python

# nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --rm -it --ulimit memlock=40000000000 -v host_dir:doc_dir --net=host fold_mode
# nvidia-docker run tells the container that we will use nvidia/cuda library at runtime;
# --rm means the container will be released once it finishs;
# -i means it is a interactive container;
# -t oallocate a pseudo-TTY;
# as single character can be combined, we use -it instead of -i and -t here;
# --ulimit memlock=XX tell container to use XX bytes locked shared memory, NOTE, here is the size of shared memory used by all docker containers on host machine, not the current one;
# --net=host let container to use host network configuration;
# -e NVIDIA_DRIVER_CAPABILITIES controls which driver libraries/binaries will be mounted inside the container;
# -e NVIDIA_VISIBLE_DEVICES which GPUs will be made accessible inside the container;
# -v maps the host directory with the directory inside container, if the directories do not exist, docker will create them;
# Detail on how to setup nvidia docker image can be found at https://github.com/NVIDIA/nvidia-container-runtime;

import os, argparse

# Read in command line arguments
parser = argparse.ArgumentParser(description="Launch the pipeline to catpure and fold data stream from BMF or from PSRDADA file")
parser.add_argument("-a", "--image", type=str, nargs="+",
                    help="The name of docker image")
parser.add_argument("-b", "--numa", type=int, nargs="+",
                    help="The index of numa node")
parser.add_argument("-c", "--root", type=int, nargs="+",
                    help="To run the docker as root or not")

args  = parser.parse_args()
numa  = args.numa[0]
image = args.image[0]
root  = args.root[0]
uid   = 50000
gid   = 50000

container_name = "{}.{}".format(image, numa)
image       = "xinpingdeng/{}".format(image)
xauth       = "/home/pulsar/.Xauthority"
sshauth     = "/home/pulsar/.ssh"
dev_dir     = "/home/pulsar/xinping/"
data_dir    = "/beegfs/"

data_mount  = "{}:{}".format(data_dir, data_dir)
dev_mount   = "{}:{}".format(dev_dir, dev_dir)
xauth_mount = "{}:{}".format(xauth, xauth)
sshauth_mount = "{}:{}".format(sshauth, sshauth)

if(numa == 0):
    cpuset_cpus = "0-9"
if(numa == 1):
    cpuset_cpus = "10-19"
    
if root:
    #comline = "docker run --runtime=nvidia --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it --rm --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm -e DISPLAY --net=host -v {} -v {} -v {} -v {} -e NVIDIA_VISIBLE_DEVICES={} -e NVIDIA_DRIVER_CAPABILITIES=all --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --cpuset-mems={} --cpuset-cpus={} --name {} {}".format(data_mount, xauth_mount, dev_mount, sshauth_mount, numa, numa, cpuset_cpus, container_name, image)
    comline = "docker run --runtime=nvidia --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it --rm -e DISPLAY --net=host -v {} -v {} -v {} -v {} -e NVIDIA_VISIBLE_DEVICES={} -e NVIDIA_DRIVER_CAPABILITIES=all --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --cpuset-mems={} --cpuset-cpus={} --name {} {}".format(data_mount, xauth_mount, dev_mount, sshauth_mount, numa, numa, cpuset_cpus, container_name, image)
else:    
    comline = "docker run --runtime=nvidia --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it --rm -e DISPLAY --net=host -v {} -v {} -v {} -v {} -u {}:{} -e NVIDIA_VISIBLE_DEVICES={} -e NVIDIA_DRIVER_CAPABILITIES=all --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --cpuset-mems={} --cpuset-cpus={} --name {} {}".format(data_mount, xauth_mount, dev_mount, sshauth_mount, uid, gid, numa, numa, cpuset_cpus, container_name, image)
    
print comline
print "\nYou are going to a docker container with the name {}{}!\n".format(image, numa)

os.system(comline)
