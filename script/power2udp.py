#!/usr/bin/env python
import parser, argparse, ConfigParser, os, datetime, pytz
import numpy as np

BLKSZ = 1024
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

# Leap seconds definition, [UTC datetime, MJD, DUTC]
LEAPSECONDS = [
    [datetime.datetime(2017, 1, 1, tzinfo=pytz.utc), 57754, 37],
    [datetime.datetime(2015, 7, 1, tzinfo=pytz.utc), 57205, 36],
    [datetime.datetime(2012, 7, 1, tzinfo=pytz.utc), 56109, 35],
    [datetime.datetime(2009, 1, 1, tzinfo=pytz.utc), 54832, 34],
    [datetime.datetime(2006, 1, 1, tzinfo=pytz.utc), 53736, 33],
    [datetime.datetime(1999, 1, 1, tzinfo=pytz.utc), 51179, 32],
    [datetime.datetime(1997, 7, 1, tzinfo=pytz.utc), 50630, 31],
    [datetime.datetime(1996, 1, 1, tzinfo=pytz.utc), 50083, 30],
    [datetime.datetime(1994, 7, 1, tzinfo=pytz.utc), 49534, 29],
    [datetime.datetime(1993, 7, 1, tzinfo=pytz.utc), 49169, 28],
    [datetime.datetime(1992, 7, 1, tzinfo=pytz.utc), 48804, 27],
    [datetime.datetime(1991, 1, 1, tzinfo=pytz.utc), 48257, 26],
    [datetime.datetime(1990, 1, 1, tzinfo=pytz.utc), 47892, 25],
    [datetime.datetime(1988, 1, 1, tzinfo=pytz.utc), 47161, 24],
    [datetime.datetime(1985, 7, 1, tzinfo=pytz.utc), 46247, 23],
    [datetime.datetime(1993, 7, 1, tzinfo=pytz.utc), 45516, 22],
    [datetime.datetime(1982, 7, 1, tzinfo=pytz.utc), 45151, 21],
    [datetime.datetime(1981, 7, 1, tzinfo=pytz.utc), 44786, 20],
    [datetime.datetime(1980, 1, 1, tzinfo=pytz.utc), 44239, 19],
    [datetime.datetime(1979, 1, 1, tzinfo=pytz.utc), 43874, 18],
    [datetime.datetime(1978, 1, 1, tzinfo=pytz.utc), 43509, 17],
    [datetime.datetime(1977, 1, 1, tzinfo=pytz.utc), 43144, 16],
    [datetime.datetime(1976, 1, 1, tzinfo=pytz.utc), 42778, 15],
    [datetime.datetime(1975, 1, 1, tzinfo=pytz.utc), 42413, 14],
    [datetime.datetime(1974, 1, 1, tzinfo=pytz.utc), 42048, 13],
    [datetime.datetime(1973, 1, 1, tzinfo=pytz.utc), 41683, 12],
    [datetime.datetime(1972, 7, 1, tzinfo=pytz.utc), 41499, 11],
    [datetime.datetime(1972, 1, 1, tzinfo=pytz.utc), 41317, 10],
]

def utc_now():
    """
    Return current UTC as a timezone aware datetime.
    :rtype: datetime
    """
    dt=datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    return dt

def getDUTCDt(dt=None):
    """
    Get the DUTC value in seconds that applied at the given (datetime)
    timestamp.
    """
    dt = dt or utc_now()
    for i in LEAPSECONDS:
        if i[0] < dt:
            return i[2]
    # Different system used before then
    return 0

# docker run --ipc=container:baseband2power --rm -it -v /beegfs:/beegfs -v /home/pulsar:/home/pulsar -u 50000:50000 --ulimit memlock=-1:-1 --name power2udp xinpingdeng/phased-array-feed "./power2udp.py -a ../config/pipeline.conf -b ../config/system.conf -c 0.016"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert baseband data to power with original channels')
    parser.add_argument('-a', '--pipeline_conf', type=str, nargs='+',
                        help='The configuration of pipeline')
    parser.add_argument('-b', '--system_conf', type=str, nargs='+',
                        help='The configuration of system')
    parser.add_argument('-c', '--tsamp', type=float, nargs='+',
                        help='The sampleing time in seconds of converted power data')

    args          = parser.parse_args()
    pipeline_conf = args.pipeline_conf[0]
    system_conf   = args.system_conf[0]
    tsamp         = args.tsamp[0]
    leap          = getDUTCDt()
    
    df_res    = float(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['df_res'])
    nsamp_df  = int(ConfigSectionMap(system_conf, "EthernetInterfaceBMF")['nsamp_df'])
    fits_ip   = ConfigSectionMap(system_conf, "EthernetInterfacePACIFIX")['fits_ip']
    fits_port = int(ConfigSectionMap(system_conf, "EthernetInterfacePACIFIX")['fits_port'])
    meta_ip   = ConfigSectionMap(system_conf, "EthernetInterfacePACIFIX")['meta_ip']
    meta_port = int(ConfigSectionMap(system_conf, "EthernetInterfacePACIFIX")['meta_port'])

    ddir      = ConfigSectionMap(pipeline_conf, "POWER2UDP")['dir']
    key       = ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['key']
    nstream   = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['nstream'])
    ndf_chk_rbuf   = int(ConfigSectionMap(pipeline_conf, "BASEBAND2POWER")['ndf_chk_rbuf'])
    
        # Find the number of data frames for each stream
    # It is the maximum which can be devided by ndf_chk_rbuf/nstream
    if(ndf_chk_rbuf%nstream):
        print "Please reset the NDF_CHK_RBUF and NSTREAM ... !"
        print exit(1)
    ndf_chk_stream_max     = int(tsamp / df_res)
    ndf_chk_stream_attempt = np.arange(ndf_chk_stream_max, 0, -1)
    for ndf_chk_stream in ndf_chk_stream_attempt:
        if ((ndf_chk_rbuf/nstream) % ndf_chk_stream == 0):
            break

    # How many runs to finish one incoming buffer
    nrepeat = int(ndf_chk_rbuf/ndf_chk_stream)

    p2u = "../src/power2udp/power2udp_main -a {:s} -b {:s} -c {:s} -d {:d} -e {:s} -f {:d} -g {:d} -i {:d}".format(key, ddir, fits_ip, fits_port, meta_ip, meta_port, nrepeat, leap)
    print p2u

    os.system(p2u)
    
