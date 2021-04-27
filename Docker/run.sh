DOCKERIMAGE="edd01:5000/capture_bypassed_bmf"
DOCKERNAME="capture_bypassed_bmf_numa0"
docker run --name="$DOCKERNAME" --rm \
        --privileged=true \
        --ipc=shareable \
        --cap-add=IPC_LOCK \
        --ulimit memlock=-1:-1 \
        --net=host \
        -v /beegfsEDD/NESSER:/beegfsEDD/NESSER \
        -e DISPLAY \
        -e USER=root \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --cap-add=SYS_PTRACE \
        -it $DOCKERIMAGE /bin/bash -ic "bash;"
