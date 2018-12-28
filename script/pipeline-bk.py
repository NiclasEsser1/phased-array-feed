#!/usr/bin/env python

import ConfigParser
import threading
import json
from astropy.time import Time

SECDAY      = 86400.0
MJD1970     = 40587.0

class Pipeline(object):
    """
    A class to define the behaviour of pipeline
    including, configure, start, stop and deconfigure and other support functions
    """
    def ConfigSectionMap(self, fname, section):
        """
        Play with configuration file and get required parameters
        """
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
    
    def utc2ref(self):
        """
        To convert UTC_START to reference information for data capture
        """
        t = Time(self.utc_start, format='isot', scale='utc')
        print t.mjd
        
    def __init__(self, system_conf, pipeline_conf, mode, nbeam, nchan, freq, utc_start, ip = None, fname = None):
        """
        Init the class
        """
        if((nbeam == 36) and (nchan == 336)):
            raise Exception("For now, we do not support 36 beams with full bandwidth ...\n")

        # Two docker containers will run on each server and here definie how many processes we will run inside each container, 2 or 1
        self.nprocess  = int(nbeam/18)  
        self.freq      = freq
        self.nchan     = nchan
        self.mode      = mode
        self.utc_start = utc_start
        
        # To see where we get data, from file or from NiC
        if(((ip == None) and (fname == None)) or ((ip != None) and (fname != None))):
            raise Exception("Please provide the source of data, either ip address or file name ... \n")
        if(ip != None):
            self.ip       = ip
            self.fname    = None
        else:
            self.fname    = fname
            self.ip       = None
            
        # To get the configuration section in the pipeline configuration file
        if(self.mode == "baseband"):
            self.process_section = "B2B"
        else:
            if(self.mode == "filterbank"):
                conf1 = "B2F"
            else:
                conf1 = "B2S"
            if(self.nprocess == 2):
                conf2 = "_P"
            else:
                conf2 = "_F"                
            self.process_section = conf1 + conf2
            
    def configure(self):
        # Create ring buffer for capture and launch the capture software with sod disabled at the beginning
        # Create ring buffer for disk2db and run disk2db if we will read a file to get data
        # Data should comes to ring buffer, but no processing happen
        self.utc2ref()
        
        if self.mode == "filterbank":
            # Create ring buffer for filterbank output (process1) and for FITSwriter data (process2)
            return True
        if self.mode == "spectral" or self.mode == "baseband":
            # Create ring buffer for spectral/baseband output (process1)
            return True
        
        return True

    def start(self):              
        if self.mode == "filterbank":
            # Run the baseband2filterbank, heimdall and spectral2udp
            # Enable sod of ring buffer
            # Should start to process
            return True
        if self.mode == "spectral":
            # Run the baseband2spectral and spectral2udp
            # Enable sod of ring buffer
            # Should start to process
            return True
        if self.mode == "baseband":
            # Run the baseband2baseband
            # Enable sod of ring buffer
            # Should start to process
            return True
        
        return True

    def stop(self):                  
        if self.mode == "filterbank":
            # Disable sod of ring buffer
            # Stop the baseband2filterbank, heimdall and spectral2udp
            return True
        if self.mode == "spectral":
            # Disable sod of ring buffer
            # stop the baseband2spectral and spectral2udp
            return True
        if self.mode == "baseband":
            # Disable sod of ring buffer
            # stop the baseband2baseband and dspsr
            return True
        
        return True

    def deconfigure(self):
        # Disable sod of ring buffer
        # Stop the capture
        # delete capture ring buffer
        
        if self.mode == "filterbank":
            # Delete ring buffer for filterbank output (process1) and for FITSwriter data (process2)
            return True
        if self.mode == "spectral" or self.mode == "baseband":
            # Delete ring buffer for spectral/baseband output (process1)
            return True
        
        return True
        
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
    
    search = Pipeline(system_conf, pipeline_conf, mode, nbeam, nchan, freq, utc_start, ip)
    #pipeline = Pipeline(system_conf, pipeline_conf, mode, nbeam, nchan, freq)
    #print Pipeline.ConfigSectionMap(pipeline_conf, "CAPTURE")["ndf_chk_rbuf"]
    #print Pipeline.ConfigSectionMap(pipeline_conf, "B2F_F")["key"].split(",")
    search.configure()
