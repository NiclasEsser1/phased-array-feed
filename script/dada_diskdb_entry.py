#!/usr/bin/env python
import os, parser, argparse, ConfigParser, time, threading

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

def diskdb_db(pipeline_conf, system_conf):    
    key           = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "DISKDB")['key']), 0), 'x')
    nblk          = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nblk'])
    pktsz         = int(ConfigSectionMap(pipeline_conf, "DISKDB")['pktsz'])
    ndf_chk_rbuf  = int(ConfigSectionMap(pipeline_conf, "DISKDB")['ndf_chk_rbuf'])
    nreader       = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nreader'])
    nchk_beam     = int(ConfigSectionMap(pipeline_conf, "DISKDB")['nchk_beam'])
    blksz         = pktsz * ndf_chk_rbuf * nchk_beam

    kfname       = "diskdb.key"
    kfile = open(kfname, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key))
    kfile.close()

    os.system("dada_db -l -p -k {:s} -b {:d} -n {:d} -r {:d}".format(key, blksz, nblk, nreader))

def b2b_db(pipeline_conf, system_conf):    
    nbyte_in        = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nbyte_in'])
    nbyte_out       = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nbyte_out'])    
    nchan_in        = float(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nchan_in'])
    nchan_keep_chan = float(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nchan_keep_chan'])
    nchan_keep_band = float(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nchan_keep_band'])
    osamp_ratei     = float(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['osamp_ratei'])
    ndf_chk_rbuf    = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['ndf_chk_rbuf'])
    nblk_b2b     = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nblk'])
    nreader_b2b  = int(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['nreader'])
    key_b2b      = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['key']), 0), 'x')
    
    nchan_ratei     = nchan_keep_chan * nchan_in / nchan_keep_band
    nsamp_df      = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    npol_samp     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['npol_samp'])
    ndim_pol      = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['ndim_pol'])
    nbyte_dim     = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nbyte_dim'])
    blksz_b2b    = int(ndf_chk_rbuf * nsamp_df * npol_samp * ndim_pol * nbyte_dim * nchan_in * nbyte_out * osamp_ratei / (nchan_ratei * nbyte_in))
    
    kfname       = "baseband2baseband.beam00part00.key"
    kfile = open(kfname, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key_b2b))
    kfile.close()
    os.system("dada_db -l p -k {:s} -b {:d} -n {:d} -r {:d}".format(key_b2b, blksz_b2b, nblk_b2b, nreader_b2b))

def b2f_db(pipeline_conf, system_conf):    
    nbyte_in        = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nbyte_in'])
    nbyte_out       = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nbyte_out'])
    nchan_in        = float(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nchan_in'])
    nchan_out        = float(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nchan_out'])
    nchan_keep_chan = float(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nchan_keep_chan'])
    nchan_keep_band = float(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nchan_keep_band'])
    osamp_ratei     = float(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['osamp_ratei'])
    ndf_chk_rbuf    = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['ndf_chk_rbuf'])
    nsamp_ave       = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nsamp_ave'])
    nchan_ratei     = nchan_keep_chan * nchan_in / nchan_keep_band
    
    nsamp_df      = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    blksz_b2f    = int(ndf_chk_rbuf * nchan_in * nsamp_df * osamp_ratei / nchan_ratei * nchan_out / nchan_keep_band * nbyte_out)
    
    nblk_b2f     = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nblk'])
    nreader_b2f  = int(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['nreader'])
    key_b2f      = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['key']), 0), 'x')
    kfname       = "baseband2filterbank.beam00part00.key"
    kfile = open(kfname, "w")
    kfile.writelines("DADA INFO:\n")
    kfile.writelines("key {:s}\n".format(key_b2f))
    kfile.close()
    
    os.system("dada_db -l p -k {:s} -b {:d} -n {:d} -r {:d}".format(key_b2f, blksz_b2f, nblk_b2f, nreader_b2f))    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To transfer data from shared memeory to disk with a docker container')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The configuration of system')  
    parser.add_argument('-c', '--fname', type=str, nargs='+',
                        help='file name with directory')
    parser.add_argument('-d', '--byte', type=int, nargs='+',
                        help='Byte to seek into the file')
    
    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    system_conf = args.system_conf[0]
    fname         = args.fname[0]
    byte          = args.byte[0]

    t_diskdb_db = threading.Thread(target = diskdb_db, args=(pipeline_conf, system_conf, ))
    t_b2b_db    = threading.Thread(target = b2b_db, args=(pipeline_conf, system_conf, ))
    t_b2f_db    = threading.Thread(target = b2f_db, args=(pipeline_conf, system_conf, ))

    t_diskdb_db.start()
    #t_b2b_db.start()
    t_b2f_db.start()

    t_diskdb_db.join()
    #t_b2b_db.join()
    t_b2f_db.join()

    key_diskdb = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "DISKDB")['key']), 0), 'x')
    #key_b2b    = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2BASEBAND")['key']), 0), 'x')
    key_b2f    = format(int("0x{:s}".format(ConfigSectionMap(pipeline_conf, "BASEBAND2FILTERBANK")['key']), 0), 'x')
    
    os.system("dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(key_diskdb, fname, byte))

    time.sleep(10)
    
    os.system("dada_db -d -k {:s}".format(key_diskdb))    
    #os.system("dada_db -d -k {:s}".format(key_b2b))
    os.system("dada_db -d -k {:s}".format(key_b2f))
    
