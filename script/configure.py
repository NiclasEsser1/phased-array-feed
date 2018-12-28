#!/usr/bin/env python

import ConfigParser
import threading

class configure:    
    def ConfigSectionMap(self, fname, section):
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

    def __init__(self, system_conf, pipeline_conf, mode, nbeam, nchan, freq):
        self.prd = self.ConfigSectionMap(system_conf, "EthernetInterfaceBMF")["df_prd"]
        
        if((nbeam == 36) and (nchan == 336)):
            raise Exception("For now, we do not support 36 beams with full bandwidth")

        self.nprocess = int(nbeam/18)  # 2 or 1

        
if __name__ == "__main__":
    print "HERE"
    system_conf   = "../config/system-new.conf"
    pipeline_conf = "../config/pipeline-new.conf"
    mode          = "filterbank"
    nbeam         = 18
    nchan         = 336
    freq          = 1340.5
    
    conf = configure(system_conf, pipeline_conf, mode, nbeam, nchan, freq)
    print conf.ConfigSectionMap(pipeline_conf, "CAPTURE")["ndf_chk_rbuf"]
    
