#!/usr/bin/env python

from inspect import currentframe, getframeinfo
import ConfigParser
import threading
import json
import os
from astropy.time import Time
import astropy.units as units
import multiprocessing
import numpy as np
import socket
import struct
from subprocess import check_output, PIPE, Popen, check_call
import time
import shlex
import parser
import argparse
import logging

log = logging.getLogger("phased-array-feed.paf_pipeline")

EXECUTE        = True
#EXECUTE        = False

#NVPROF         = True
NVPROF         = False

#SOD            = False   # To start filterbank data or not
SOD            = True   # To start filterbank data or not

HEIMDALL       = True   # To run heimdall on filterbank file or not
#HEIMDALL       = False   # To run heimdall on filterbank file or not

MEMCHECK       = True
#MEMCHECK       = False

#DBDISK         = True   # To run dbdisk on filterbank file or not
DBDISK         = False   # To run dbdisk on filterbank file or not

PAF_ROOT       = "/home/pulsar/xinping/phased-array-feed/"
DATA_ROOT      = "/beegfs/DENG/"
BASEBAND_ROOT  = "{}/AUG/baseband/".format(DATA_ROOT)

PAF_CONFIG = {"instrument_name":    "PAF-BMF",
              "nchan_chk":    	     7,        # MHz
              "samp_rate":    	     0.84375,
              "prd":                 27,       # Seconds
              "df_res":              1.08E-4,  # Seconds
              "ndf_prd":             250000,
              
              "df_dtsz":      	     7168,
              "df_pktsz":     	     7232,
              "df_hdrsz":     	     64,
              
              "nbyte_baseband":      2,
              "npol_samp_baseband":  2,
              "ndim_pol_baseband":   2,
              
              "ncpu_numa":           10,
              "port0":               17100,
}

SEARCH_CONFIG_GENERAL = {"rbuf_baseband_ndf_chk":   16384,                 
                         "rbuf_baseband_nblk":      4,
                         "rbuf_baseband_nread":     1,                 
                         "tbuf_baseband_ndf_chk":   128,
                         
                         "rbuf_filterbank_ndf_chk": 16384,
                         "rbuf_filterbank_nblk":    2,
                         "rbuf_filterbank_nread":   (HEIMDALL + DBDISK) if (HEIMDALL + DBDISK) else 1,
                         
                         "nchan_filterbank":        512,
                         "cufft_nx":                128,
                        
                         "nbyte_filterbank":        1,
                         "npol_samp_filterbank":    1,
                         "ndim_pol_filterbank":     1,
                         
                         "ndf_stream":      	    1024,
                         "nstream":                 2,
                         
                         "bind":                    1,
                         "seek_byte":               0,
                         
                         "pad":                     0,
                         "ndf_check_chk":           1024,
                         
                         "detect_thresh":           10,
                         "dm":                      [1, 1000],
                         "zap_chans":               [],
}

SEARCH_CONFIG_1BEAM = {"rbuf_baseband_key":      ["dada"],
                       "rbuf_filterbank_key":    ["dade"],
                       "nchan_keep_band":        32768,
                       "nbeam":                  1,
                       "nport_beam":             3,
                       "nchk_port":              16,
}

SEARCH_CONFIG_2BEAMS = {"rbuf_baseband_key":       ["dada", "dadc"],
                        "rbuf_filterbank_key":     ["dade", "dadg"],
                        "nchan_keep_band":         24576,
                        "nbeam":                   2,
                        "nport_beam":              3,
                        "nchk_port":               11,
}

PIPELINE_STATES = ["idle", "configuring", "ready",
                   "starting", "running", "stopping",
                   "deconfiguring", "error"]

class PipelineError(Exception):
    pass

class Pipeline(object):
    def __init__(self):
        self.state               = "idle"   # Idle at the very beginning
        self.diskdb_process       = []
        self.createrbuf_process   = []
        self.deleterbuf_process   = []
        
    def __del__(self):
        class_name = self.__class__.__name__
        
    def configure(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def deconfigure(self):
        raise NotImplementedError
    
    def kill_process(self, process_name):
        try:
            cmd = "pkill -f {}".format(process_name)
            check_call(shlex.split(cmd))
        except:
            pass # We only kill running process
        
    def create_rbuf(self, key, blksz, nblk, nreader):                    
        cmd = ("dada_db -l -p"
               " -k {:} -b {:}" 
               " -n {:} -r {:}").format(key,
                                       blksz,
                                       nblk,
                                       nreader)
        log.info(cmd)
        print cmd
        if EXECUTE:
            try:
                proc = Popen(shlex.split(cmd))#,
                             #stderr=PIPE,
                             #stdout=PIPE)
                self.createrbuf_process.append(proc)
            except:
                self.state = "error"
                raise PipelineError("Can not create ring buffer")

    def delete_rbuf(self, key):
        cmd = "dada_db -d -k {:}".format(key)
        log.info(cmd)
        print cmd
        if EXECUTE:
            try: 
                proc = Popen(shlex.split(cmd))#,
                             #stderr=PIPE,
                             #stdout=PIPE)
                self.deleterbuf_process.append(proc)
            except:
                self.state = "error"
                raise PipelineError("Can not delete ring buffer")

    def diskdb(self, key, fname, seek_byte):               
        cmd = "dada_diskdb -k {} -f {} -o {} -s".format(key,fname, seek_byte)            
        
        log.info(cmd)
        print cmd
        if EXECUTE:
            try:
                proc = Popen(shlex.split(cmd))#,
                             #stderr=PIPE,
                             #stdout=PIPE)
                self.diskdb_process.append(proc)
            except:
                self.state = "error"
                raise PipelineError("dada_diskdb fail")

class SearchFile(Pipeline):
    def __init__(self):
        super(SearchFile, self).__init__()        
        self.heimdall_process               = []
        self.dbdisk_process                 = []
        self.baseband2filterbank_process    = []
        
    def configure(self, fname, ip, general_config, pipeline_config):
        check_call(["ipcrm", "-a"])
        self.kill_process("dada_diskdb")
        self.kill_process("baseband2filterbank_main")
        self.kill_process("heimdall")
        self.kill_process("dada_dbdisk")
        
        if (self.state != "idle"):
            raise PipelineError(
                "Can only configure pipeline in idle state")

        self.state             = "configuring"
        self.general_config    = general_config
        self.pipeline_config   = pipeline_config
        self.ip                = ip
        self.node              = int(ip.split(".")[2])
        self.numa              = int(ip.split(".")[3]) - 1
        self.fname             = fname

        self.prd                = PAF_CONFIG["prd"]
        self.port0              = PAF_CONFIG["port0"]
        self.df_res             = PAF_CONFIG["df_res"]
        self.df_dtsz            = PAF_CONFIG["df_dtsz"]
        self.df_pktsz           = PAF_CONFIG["df_pktsz"]
        self.df_hdrsz           = PAF_CONFIG["df_hdrsz"]
        self.ncpu_numa          = PAF_CONFIG["ncpu_numa"]
        self.nchan_chk          = PAF_CONFIG["nchan_chk"]
        self.nbyte_baseband     = PAF_CONFIG["nbyte_baseband"]
        self.ndim_pol_baseband  = PAF_CONFIG["ndim_pol_baseband"]
        self.npol_samp_baseband = PAF_CONFIG["npol_samp_baseband"]

        self.nbeam               = self.pipeline_config["nbeam"]
        self.nchk_port           = self.pipeline_config["nchk_port"]
        self.nport_beam          = self.pipeline_config["nport_beam"]
        self.nchan_keep_band     = self.pipeline_config["nchan_keep_band"]
        self.rbuf_baseband_key   = self.pipeline_config["rbuf_baseband_key"]
        self.rbuf_filterbank_key = self.pipeline_config["rbuf_filterbank_key"]
        
        self.dm                      = self.general_config["dm"],
        self.pad                     = self.general_config["pad"]
        self.bind                    = self.general_config["bind"]
        self.nstream                 = self.general_config["nstream"]
        self.cufft_nx                = self.general_config["cufft_nx"]
        self.zap_chans               = self.general_config["zap_chans"]
        self.ndf_stream              = self.general_config["ndf_stream"]
        self.detect_thresh           = self.general_config["detect_thresh"]
        self.ndf_check_chk           = self.general_config["ndf_check_chk"]
        self.nchan_filterbank        = self.general_config["nchan_filterbank"]
        self.nbyte_filterbank        = self.general_config["nbyte_filterbank"]
        self.rbuf_baseband_nblk      = self.general_config["rbuf_baseband_nblk"]
        self.ndim_pol_filterbank     = self.general_config["ndim_pol_filterbank"]
        self.rbuf_baseband_nread     = self.general_config["rbuf_baseband_nread"]
        self.npol_samp_filterbank    = self.general_config["npol_samp_filterbank"]
        self.rbuf_filterbank_nblk    = self.general_config["rbuf_filterbank_nblk"]
        self.rbuf_baseband_ndf_chk   = self.general_config["rbuf_baseband_ndf_chk"]
        self.rbuf_filterbank_nread   = self.general_config["rbuf_filterbank_nread"]
        self.tbuf_baseband_ndf_chk   = self.general_config["tbuf_baseband_ndf_chk"]        
        self.rbuf_filterbank_ndf_chk = self.general_config["rbuf_filterbank_ndf_chk"]

        self.blk_res                 = self.df_res * self.rbuf_baseband_ndf_chk
        self.nchk_beam               = self.nchk_port*self.nport_beam
        self.nchan_baseband          = self.nchan_chk*self.nchk_beam
        self.ncpu_pipeline           = self.ncpu_numa/self.nbeam
        self.rbuf_baseband_blksz     = self.nchk_port*self.nport_beam*self.df_dtsz*\
                                       self.rbuf_baseband_ndf_chk
        
        if self.rbuf_baseband_ndf_chk%(self.ndf_stream * self.nstream):
            self.state = "error"
            raise PipelineError("data frames in ring buffer block can not "
                                "be processed by filterbank with integer repeats")

        self.rbuf_filterbank_blksz   = int(self.nchan_filterbank * self.rbuf_baseband_blksz*
                                         self.nbyte_filterbank*self.npol_samp_filterbank*
                                         self.ndim_pol_filterbank/
                                         float(self.nbyte_baseband*
                                               self.npol_samp_baseband*
                                               self.ndim_pol_baseband*
                                               self.nchan_baseband*self.cufft_nx))
        
        # Create ring buffers
        for i in range(self.nbeam):
            self.create_rbuf(self.rbuf_baseband_key[i],
                             self.rbuf_baseband_blksz,
                             self.rbuf_baseband_nblk,
                             self.rbuf_baseband_nread)
            self.create_rbuf(self.rbuf_filterbank_key[i],
                             self.rbuf_filterbank_blksz,
                             self.rbuf_filterbank_nblk,
                             self.rbuf_filterbank_nread)
        if EXECUTE:
            for i in range(self.nbeam * 2):
                self.createrbuf_process[i].communicate()
                if self.createrbuf_process[i].returncode:
                    self.state = "error"
                    raise PipelineError("Failed to create ring buffer")
                
        self.runtime_dir = []
        for i in range(self.nbeam):
            self.runtime_dir.append("{}/pacifix{}_numa{}_process{}".format(DATA_ROOT, self.node, self.numa, i))
        
        self.state = "ready"
        
    def start(self):
        if self.state != "ready":
            raise PipelineError(
                "Pipeline can only be started from ready state")
        
        self.state = "starting"
        # Start diskdb, baseband2filterbank and heimdall software
        for i in range(self.nbeam):
            self.diskdb(self.pipeline_config["rbuf_baseband_key"][i],
                        self.fname, self.general_config["seek_byte"])            
            self.baseband2filterbank(self.rbuf_baseband_key[i],
                                     self.rbuf_filterbank_key[i],
                                     self.runtime_dir[i])
            if HEIMDALL:
                self.heimdall(self.rbuf_filterbank_key[i],
                              self.runtime_dir[i])
            if DBDISK:                
                self.dbdisk(self.rbuf_filterbank_key[i],
                             self.runtime_dir[i])

        self.state = "running"
        
        if EXECUTE:
            for i in range(self.nbeam):
                self.diskdb_process[i].communicate()
                if self.diskdb_process[i].returncode:
                    self.state = "error"
                    raise PipelineError("Failed to diskdb")

                self.baseband2filterbank_process[i].communicate()
                if self.baseband2filterbank_process[i].returncode:
                    print self.baseband2filterbank_process[i].returncode
                    self.state = "error"
                    raise PipelineError("Failed to baseband2filterbank")
                
                if HEIMDALL:
                    self.heimdall_process[i].communicate()
                    if self.heimdall_process[i].returncode:
                        self.state = "error"
                        raise PipelineError("Failed to heimdall")
                                    
                if DBDISK:                
                    self.dbdisk_process[i].communicate()
                    if self.dbdisk_process[i].returncode:
                        self.state = "error"
                        raise PipelineError("Failed to dbdisk")                
                    
    def stop(self):
        if self.state != "running":
            raise PipelineError("Can only stop a running pipeline")
        self.state = "stopping"
        # For this mode, it will stop automatically
        self.state = "ready"

    def deconfigure(self):
        if self.state not in ["ready", "error"]:
            raise PipelineError(
                "Pipeline can only be deconfigured from ready state")
        
        self.state = "deconfiguring"
        if self.state == "ready": # Normal deconfigure
            self.delete_rbuf(self.rbuf_baseband_key[i])
            self.delete_rbuf(self.rbuf_filterbank_key[i])
            
            for i in range(self.nbeam * 2):
                    self.deleterbuf_process[i].communicate()                                    
                    if self.deleterbuf_process[i].returncode:
                        self.state = "error"
                        raise PipelineError("Failed to delete ring buffer")
        else:
            os.system("ipcrm -a")
            self.kill_process("heimdall")
            self.kill_process("dada_dbdisk")
            self.kill_process("dada_diskdb")        
            self.kill_process("baseband2filterbank_main")
            
        self.state = "idle"
    
    def dbdisk(self, key, runtime_dir):
        cmd = "dada_dbdisk -k {} -D {} -o -s -z".format(key, runtime_dir)
        
        log.info(cmd)
        print cmd
        if EXECUTE:
            try:
                proc = Popen(shlex.split(cmd))#,
                             #stderr=PIPE,
                             #stdout=PIPE)
                self.dbdisk_process.append(proc)
            except:
                self.state = "error"
                raise PipelineError("dada_dbdisk fail")

    def baseband2filterbank(self, key_in, key_out, runtime_dir):                            
        software = "{}/src/baseband2filterbank_main".format(PAF_ROOT)
        if NVPROF:
            cmd = "nvprof "
        else:
            cmd = ""

        if MEMCHECK:
            cmd += "cuda-memcheck "
            
        cmd += ("{} -a {} -b {} -c {} -d {} -e {} " 
                " -f {} -i {} -j {} -k {} -l {} ").format(
                    software, key_in, key_out,
                    self.rbuf_filterbank_ndf_chk, self.nstream,
                    self.ndf_stream, runtime_dir, self.nchk_beam,
                    self.cufft_nx, self.nchan_filterbank, self.nchan_keep_band)
        if SOD:
            cmd += "-g 1"
        else:
            cmd += "-g 0"
            
        log.info(cmd)
        print cmd
        if EXECUTE:
            try:
                proc = Popen(shlex.split(cmd))#,
                             #stderr=PIPE,
                             #stdout=PIPE)
                self.baseband2filterbank_process.append(proc)
            except:
                self.state = "error"
                raise PipelineError("baseband2filterbank fail")

    def heimdall(self, key, runtime_dir):
        zap = ""
        for zap_chan in self.zap_chans:
            zap += " -zap_chans {} {}".format(zap_chan[0], zap_chan[1])

        if NVPROF:
            cmd = "nvprof "
        else:
            cmd = ""

        cmd += ("heimdall -k {} -dm {} {} {} " 
                " -detect_thresh {} -output_dir {}").format(
                    key, self.dm[0][0], self.dm[0][1], zap,
                    self.detect_thresh, runtime_dir)
        log.info(cmd)
        print cmd
        if EXECUTE:
            try:
                proc = Popen(shlex.split(cmd))#,
                             #stderr=PIPE,
                             #stdout=PIPE)
                self.heimdall_process.append(proc)
            except:
                self.state = "error"
                raise PipelineError("Heimdall fail")
            
class SearchFile2Beams(SearchFile):
    def __init__(self):
        super(SearchFile2Beams, self).__init__()

    def configure(self, fname, ip):
        super(SearchFile2Beams, self).configure(fname, ip, SEARCH_CONFIG_GENERAL, SEARCH_CONFIG_2BEAMS)
                                            
    def start(self):
        super(SearchFile2Beams, self).start()
        
    def stop(self):
        super(SearchFile2Beams, self).stop()
        
    def deconfigure(self):
        super(SearchFile2Beams, self).deconfigure()
        
class SearchFile1Beam(SearchFile):
    def __init__(self):
        super(SearchFile1Beam, self).__init__()

    def configure(self, fname, ip):
        super(SearchFile1Beam, self).configure(fname, ip, SEARCH_CONFIG_GENERAL, SEARCH_CONFIG_1BEAM) 
        
    def start(self):
        super(SearchFile1Beam, self).start()
        
    def stop(self):
        super(SearchFile1Beam, self).stop()
        
    def deconfigure(self):
        super(SearchFile1Beam, self).deconfigure()
      
if __name__ == "__main__":
    host_id       = check_output("hostname").strip()[-1]
    parser = argparse.ArgumentParser(description='To run the pipeline for my test')
    parser.add_argument('-a', '--numa', type=int, nargs='+',
                        help='The ID of numa node')
    args     = parser.parse_args()
    numa     = args.numa[0]    
    ip       = "10.17.{}.{}".format(host_id, numa + 1)
    fname    = "{}/J1819-1458/J1819-1458_48chunks.dada".format(BASEBAND_ROOT)
    
    print "\nCreate pipeline ...\n"
    search_mode = SearchFile1Beam()
    
    print "\nConfigure it ...\n"
    search_mode.configure(fname, ip)

    print "\nStart it ...\n"
    search_mode.start()
    print "\nStop it ...\n"
    search_mode.stop()
    print "\nDeconfigure it ...\n"
    search_mode.deconfigure()
