#!/bin/bash
while getopts d:i:k:n:p option; do
   case ${option} in
      d) DOCKERNAME=${OPTARG};;
      i) IPADDRESS=${OPTARG};;
      k) DADA_KEY=${OPTARG};;
      n) NODEID=${OPTARG};;
      p) PACKETS=${OPTARG};;
   esac
done

DOCKERIMAGE="edd01:5000/capture_bypassed_bmf"
echo "DOCKERNAME: $DOCKERNAME"
echo "NODEID: $NODEID"
echo "IPADDRESS: $IPADDRESS"
echo "PACKETS: $PACKETS"
echo "DADA_KEY: $DADA_KEY"
echo "cd phased-array-feed/;git pull;python script/start_bypassed_capture.py --key $DADA_KEY -nid $NODEID -ip $IPADDRESS -p $PACKETS;bash;"
# docker run --name="$DOCKERNAME" --rm \
#         --privileged=true \
#         --ipc=shareable \
#         --cap-add=IPC_LOCK \
#         --ulimit memlock=-1:-1 \
#         --net=host \
#         -v /beegfsEDD/NESSER:/beegfsEDD/NESSER \
#         -e DISPLAY \
#         -e USER=root \
#         --runtime=nvidia \
#         -e NVIDIA_VISIBLE_DEVICES=0 \
#         -e NVIDIA_DRIVER_CAPABILITIES=all \
#         --cap-add=SYS_PTRACE \
#         -it $DOCKERIMAGE /bin/bash -ic "cd phased-array-feed/;git pull;python script/start_bypassed_capture.py --key $DADA_KEY -nid $NODEID -ip $IPADDRESS -p $PACKETS;bash;"
