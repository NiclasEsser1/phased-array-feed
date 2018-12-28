#!/usr/bin/env python

import ConfigParser
import threading
import json
#from astropy.time import Time

SECDAY      = 86400.0
MJD1970     = 40587.0

class Pipeline(object):
    def __init__(self):
        pass

    def configure(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def deconfigure(self):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError

class SearchModeStreamTwoProcess(Pipeline):
    def __init__(self, system_conf, pipeline_conf, utc_start, ip, freq):
        super(Pipeline, self).__init__()
        self.system_conf   = system_conf
        self.pipeline_conf = pipeline_conf
        self.nprocess      = 2
        self.utc_start     = utc_start
        self.ip            = ip
        self.freq          = freq

class SearchModeStreamOneProcess(Pipeline):
    def __init__(self, system_conf, pipeline_conf, utc_start, ip, freq):
        super(Pipeline, self).__init__()
        self.system_conf   = system_conf
        self.pipeline_conf = pipeline_conf
        self.nprocess      = 1
        self.utc_start     = utc_start
        self.ip            = ip
        self.freq          = freq
    
class SearchModeFileTwoProcess(Pipeline):
    def __init__(self, system_conf, pipeline_conf, fname):
        super(Pipeline, self).__init__()
        self.system_conf   = system_conf
        self.pipeline_conf = pipeline_conf
        self.nprocess      = 2
        self.fname         = fname
        
class SearchModeFileOneProcess(Pipeline):
    def __init__(self, system_conf, pipeline_conf, fname):
        super(Pipeline, self).__init__()
        self.system_conf   = system_conf
        self.pipeline_conf = pipeline_conf
        self.nprocess      = 1
        self.fname         = fname
        
if __name__ == "__main__":
    system_conf   = "../config/system-new.conf"
    pipeline_conf = "../config/pipeline-new.conf"
    mode          = "filterbank"
    nbeam         = 18
    nchan         = 336
    freq          = 1340.5
    ip            = "10.17.8.1"
    fname         = "/beegfs/DENG/AUG/baseband/J1819-1458/J1819-1458.dada"
    utc_start     = "2018-08-30T19:37:27"
    
    #search = Pipeline(system_conf, pipeline_conf, mode, nbeam, nchan, freq, utc_start, ip)
    ##pipeline = Pipeline(system_conf, pipeline_conf, mode, nbeam, nchan, freq)
    ##print Pipeline.ConfigSectionMap(pipeline_conf, "CAPTURE")["ndf_chk_rbuf"]
    ##print Pipeline.ConfigSectionMap(pipeline_conf, "B2F_F")["key"].split(",")
    #search.configure()

    search_mode_pipeline = SearchModeFileTwoProcess(system_conf, pipeline_conf, fname)
