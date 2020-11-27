while getopts n:nid:ip:p:k option; do
   case "${option}" in
      n) DOCKERNAME=${OPTARG};;
      nid) NODEID=${OPTARG};;
      ip) IPADDRESS=${OPTARG};;
      p) PACKETS=${OPTARG};;
      k) DADA_KEY=${OPTARG};;
   esac
done

DOCKERIMAGE="edd01:5000/capture_bypassed_bmf"

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
        -it $DOCKERIMAGE /bin/bash -ic "cd phased-array-feed/;git pull;script/start_bypassed_capture.py -k $DADA_KEY -nid $NODEID -ip $IPADDRESS -p $PACKETS;bash;"
