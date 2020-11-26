
# Make sure that no dada buffers with the same key are alive. Destroy them if necessary...
 echo "Destroying existing dada buffer.."
 dada_db -k dada -d;
# sleep 2
# # Creating dada buffer
 echo "Creating new dada buffer.."
 numactl -m 1 dada_db -k dada -l -p -b 266600448 -n 16
#
# # Reading client
 echo "Connecting dada_dbdisk to dada buffer"
 dada_dbdisk -k dada -D /beegfsEDD/NESSER -W -d -s # Problem is here: No errors thrown, but data aren't written to a file. Also tested with -D /beegfsEDD/NESSER/test.dada which throws "No such file or directory"
#
# # Monitor
# #echo "Conncetion dada_dbmonitor to dada buffer"
dada_dbmonitor -k dada -d -v
# #sleep 2
# # Writing client for UDP capturing

# dada_junkdb -k dada -b 17218142208 -c f -r 2432.666 header_dada.txt
#./capture_main -a dada -b 0 -c 10.17.1.1_17100_9_9_3 -c 10.17.1.1_17101_9_9_3 -c 10.17.1.1_17102_9_9_3 -c 10.17.1.1_17103_9_9_3 -e 1337.0 -f ${EPOCH_REF}_${SEC_PACKET}_248140 -g ../../log/numa1_pacifix1 -i 1 -j 0_2 -k 1 -l 16384 -m 128 -n header_dada.txt -o UNKOWN_00:00:00.00_00:00:00.00 -p 0 -q 1340
