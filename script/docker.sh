#/usr/bin/env sh

# Just a example of docker run
docker run --rm -ti \
       # Remove the container on finish and enable the interaction
       --ipc=shareable --ipc=previous_container_name --cap-add=IPC_LOCK --ulimit memlock=-1:-1\
       # IPC namespace can be accessed with the name of the container with the name of current_container_name, it also access the namespace of the container with the name of previous_container_name
       # IPC_LOCK enabled and memlock is unlimited
       --net=host \
       # Use the network configuration of host
       -v  host_dir:container_dir\
       # Mount container_dir to host_dir
       -u 50000:50000\
       # Use pulsar user in group of psr
       --name current_container_name\
       # Give name to the current container
       -e DISPLAY \
       # Enable the X of container, should be used with -v /home:/home
       --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=gpu_id -e NVIDIA_DRIVER_CAPABILITIES=all\
       # To use nvidia feature inside container, set the visible device and the capabilities of it
       --cap-add=SYS_PTRACE \
       # To enable GDB inside docker
       image_name
