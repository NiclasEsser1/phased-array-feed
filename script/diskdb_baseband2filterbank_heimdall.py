#!/usr/bin/env python

import os, parser, argparse, ConfigParser, threading, time

def ConfigSectionMap(fname, section):
    # Play with configuration file
    Config = ConfigParser.ConfigParser()
    Config.read(fname)
    
    dict_conf = {}
    options = Config.options(section)
    for option in options:
        try:
            dict_conf[option] = Config.get(section, option)
            if dict_conf[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict_conf[option] = None
    return dict_conf

def baseband2filterbank(args):    
    uid = 50000
    gid = 50000
    
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    cpu           = args.cpu[0]
    gpu           = args.gpu[0]
    
    ddir = ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['dir']
    hdir = "/home/pulsar"
    
    dname                   = "phased-array-feed"
    previous_container_name = "paf-diskdb.beam{:02d}".format(beam)
    current_container_name  = "paf-baseband2filterbank.beam{:02d}".format(beam)
    software_name           = "baseband2filterbank_main"
    
    ddir           = ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['dir']
    key_diskdb     = ConfigSectionMap(pipeline_conf, "DISKDB")['key']
    ndf_chk_rbuf   = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['ndf_chk_rbuf'])
    key_b2f        = ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['key']
    nstream        = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nstream'])
    ndf_chk_stream = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['ndf_chk_stream'])
    nrepeat        = ndf_chk_rbuf / (ndf_chk_stream * nstream)

    dvolume = '{:s}:{:s}'.format(ddir, ddir)
    hvolume = '{:s}:{:s}'.format(hdir, hdir)

    com_line = "docker run --ipc=container:{:s} --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={:d} -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY --net=host -v {:s} -v {:s} -u {:d}:{:d} --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --name {:s} xinpingdeng/{:s} \"taskset -c {:d} /home/pulsar/xinping/phased-array-feed/src/baseband2filterbank/{:s} -a {:s} -b {:s} -c {:d} -d {:d} -e {:d} -f {:d} -g {:s}\"".format(previous_container_name, gpu, dvolume, hvolume, uid, gid, current_container_name, dname, cpu, software_name, key_diskdb, key_b2f, ndf_chk_rbuf, nrepeat, nstream, ndf_chk_stream, ddir)

    time.sleep(5)
    print com_line
    os.system(com_line)

def heimdall(args):
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    cpu           = args.cpu[0]
    gpu           = args.gpu[0]
    
    ddir                    = ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['dir']
    key_b2f                 = ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['key']
    previous_container_name = "paf-baseband2filterbank.beam{:02d}".format(beam)
    current_container_name  = "paf-heimdall.beam{:02d}".format(beam)

    hdir = "/home/pulsar"
    dvolume = '{:s}:{:s}'.format(ddir, ddir)
    hvolume = '{:s}:{:s}'.format(hdir, hdir)
    dname   = "paf-base"
    software_name = "heimdall"

    com_line = "docker run --ipc=container:{:s} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={:d} -e NVIDIA_DRIVER_CAPABILITIES=all --rm -it -u 50000:50000 -v {:s} --ulimit memlock=-1:-1 --name {:s} xinpingdeng/{:s} taskset -c {:d} {:s} -k {:s} -output_dir {:s} -dm 1 1000 -zap_chans 512 1023 -zap_chans 304 310 -detect_thresh 10".format(previous_container_name, gpu, dvolume, current_container_name, dname, cpu, software_name, key_b2f, ddir)

    time.sleep(10)
    print com_line
    os.system(com_line)
    #docker run --ipc=container:diskdb --rm -it -u 50000:50000 -v /beegfs/:/beegfs --ulimit memlock=-1:-1 --name heimdall xinpingdeng/paf-base heimdall -k dada -output_dir /beegfs/DENG/AUG/ -dm 1 1000 -zap_chans 512 1023 -zap_chans 304 310 -detect_thresh 10

def diskdb(args):
    system_conf   = args.system_conf[0]
    pipeline_conf = args.pipeline_conf[0]
    beam          = args.beam[0]
    cpu           = args.cpu[0]
    
    current_container_name  = "paf-diskdb,beam{:02d}".format(beam)
    
    ddir = ConfigSectionMap(pipeline_conf, "DISKDB")['dir']
    hdir = "/home/pulsar"
    dvolume = '{:s}:{:s}'.format(ddir, ddir)
    hvolume = '{:s}:{:s}'.format(hdir, hdir)

    current_container_name = "paf-diskdb.beam{:02d}".format(beam)
    dname                  = "phased-array-feed"
    software_name          = "diskdb.py"
    dfname                 = "/beegfs/DENG/AUG/baseband/J1819-1458/J1819-1458.dada"

    com_line = "docker run --ipc=shareable --rm -it -v {:s} -v {:s} -u 50000:50000 --ulimit memlock=-1:-1 --name {:s} xinpingdeng/{:s} \"taskset -c {:d} /home/pulsar/xinping/phased-array-feed/script/{:s} -a {:s} -b {:s} -c {:s} -d 0\"".format(dvolume, hvolume, current_container_name, dname, cpu, software_name, pipeline_conf, system_conf, dfname) 
    print com_line
    os.system(com_line)

    #docker run --ipc=shareable --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name diskdb xinpingdeng/phased-array-feed "./diskdb.py -a ../config/pipeline.conf -b ../config/system.conf -c /beegfs/DENG/AUG/baseband/J0332+5434/J0332+5434.dada -d 0"
    
# ./diskdb_baseband2filterbank_heimdall.py -a ../config/pipeline.conf -b ../config/system.conf -c 0 -d 4 -e 0
# ./diskdb_baseband2filterbank_heimdall.py -a ../config/pipeline.conf -b ../config/system.conf -c 0 -d 9 -e 0

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='To process baseband data and run heimdall')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The configuration of system')    
    parser.add_argument('-c', '--beam', type=int, nargs='+',
                        help='The beam id from 0')
    parser.add_argument('-d', '--cpu', type=int, nargs='+',
                        help='Bind threads to cpu')
    parser.add_argument('-e', '--gpu', type=int, nargs='+',
                        help='Bind threads to GPU')    
    
    args          = parser.parse_args()
    
    t_diskdb              = threading.Thread(target = diskdb, args = (args,))
    t_baseband2filterbank = threading.Thread(target = baseband2filterbank, args = (args,))
    t_heimdall            = threading.Thread(target = heimdall, args = (args,))

    t_diskdb.start()
    t_baseband2filterbank.start()
    t_heimdall.start()
    
    t_diskdb.join()
    t_baseband2filterbank.join()
    t_heimdall.join()
