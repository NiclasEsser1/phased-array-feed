!/bin/bash
# Make sure that no dada buffers with the same key are alive. Destroy them if necessary...
echo "Destroying existing dada buffer.."
dada_db -k dada -d;
sleep 2
# Creating dada buffer
echo "Creating new dada buffer.."
numactl -m 1 dada_db -k dada -l -p -b 266674176 -n 16
sleep 2
# Reading client
echo "Connecting dada_dbdisk to dada buffer"
dada_dbdisk -k dada -D /beegfsEDD/NESSER -W -d -s # Problem is here: No errors thrown, but data aren't written to a file. Also tested with -D /beegfsEDD/NESSER/test.dada which throws "No such file or directory"
sleep 2
# Monitor
#echo "Conncetion dada_dbmonitor to dada buffer"
#dada_dbmonitor -k dada -d -v
#sleep 2
# Writing client for UDP capturing
echo "Starting capturing.."
# dada_junkdb -k dada -b 17218142208 -c f -r 2432.666 header_dada.txt
./capture_main -a dada -b 7298 -c 64 -d 10.17.1.1:17100:1:1:2 -d 10.17.1.1:17101:1:1:3 -d 10.17.1.1:17102:1:1:4 -d 10.17.1.1:17103:1:1:5 -f 1140.500000 -g 7 -j ../../log/ -k 6 -l 7 -m 1 -n 27 -o 1 -p 36864 -q 256 -r 250000 -s ./capture.beam17part00.socket -t header_dada.txt
