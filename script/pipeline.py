#!/usr/bin/env python

import coloredlogs
import ConfigParser
import json
import numpy as np
import socket
import struct
import time
import shlex
from subprocess import PIPE, Popen, check_output
from inspect import currentframe, getframeinfo
import logging
import argparse
import threading
import inspect
import os
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK

# Spectrometer output currently can only be network output or disk output;
# Monitor output currently can only be network output

# Configuration of input for different number of beams 
INPUT_1BEAM = {"input_nbeam":                  1,
               "input_nchunk":                 48,
}

INPUT_2BEAMS = {"input_nbeam":                  2,
                "input_nchunk":                 33,
}

# We can turn the configuratio directory here to json in future
# Configuration of PAF system, including GPU servers
SYSTEM_CONFIG = {"paf_nchan_per_chunk":    	     7,        # MHz
                 "paf_over_samp_rate":               (32.0/27.0),
                 "paf_period":                       27,       # Seconds
                 "paf_ndf_per_chunk_per_period":     250000,
                 "paf_nsamp_per_df":                 128,
                 
                 "paf_df_res":                       1.08E-4,  # Seconds
                 "paf_df_dtsz":      	             7168,     # Bytes
                 "paf_df_pktsz":     	             7232,
                 "paf_df_hdrsz":     	             64,
                 
                 "pacifix_ncpu_per_numa_node":       10,  
                 "pacifix_memory_limit_per_numa_node":  60791751475, # has 10% spare
}

# Configuration for pipelines
PIPELINE_CONFIG = {"execution":                    1,
                   "root_software":                "/home/pulsar/xinping/phased-array-feed/",
                   "root_runtime":                 "/beegfs/DENG/",
                   "root_data":                    "/beegfs/DENG/AUG/baseband/",
                   "rbuf_ndf_per_chunk_per_block": 16384,# For all ring buffers

                   # Configuration of input
                   "input_source_name":          "J1819-1458",
                   #"input_source_name":          "J0332+5434",
                   #"input_source_name":          "J1939+2134",
                   "input_keys":                 ["dada", "dadc"], # To put baseband data from file
                   "input_nblk":                 5,
                   "input_nreader":              1,
                   "input_nbyte":                2,
                   "input_npol":                 2,
                   "input_ndim":                 2,

                   # Configuration of GPU kernels
                   "gpu_ndf_per_chunk_per_stream": 1024,
                   "gpu_nstream":                  2,
                   
                   "search_keys":         ["dbda", "dbdc"], # To put filterbank data 
                   "search_nblk":         2,
                   "search_nchan":        1024,
                   "search_nchan":        512,
                   "search_cufft_nx":     128,
                   "search_cufft_nx":     256,                   
                   "search_nbyte":        1,
                   "search_npol":         1,
                   "search_ndim":         1,
                   "search_heimdall":     1,
                   "search_dbdisk":       1,
                   "search_monitor":      1,
                   "search_spectrometer": 1,
                   "search_detect_thresh":10,
                   "search_dm":           [1, 3000],
                   "search_zap_chans":    [],
                   "search_software_name":  "baseband2filterbank_main",
                   
                   "spectrometer_keys":           ["dcda", "dcdc"], # To put independent or commensal spectral data
                   "spectrometer_nblk":           2,
                   "spectrometer_nreader":        1,
                   "spectrometer_cufft_nx":       1024,
                   "spectrometer_nbyte":          4,
                   "spectrometer_ndata_per_samp": 4,
                   "spectrometer_ptype":          2,
                   "spectrometer_ip":    	  '134.104.70.95',
                   "spectrometer_port":    	  17110,
                   #"spectrometer_ip":      	  '239.3.1.2',
                   #"spectrometer_port":           2,
                   "spectrometer_dbdisk":         1,
                   "spectrometer_monitor":        1,
                   "spectrometer_accumulate_nblk": 1,
                   "spectrometer_software_name":  "baseband2spectral_main",
                   
                   # Spectral parameters for the simultaneous spectral output from fold and search mode
                   # The rest configuration of this output is the same as the normal spectrometer mode
                   "simultaneous_spectrometer_start_chunk":    26,
                   "simultaneous_spectrometer_nchunk":         5,
                   
                   "fold_keys":           ["ddda", "dddc"], # To put processed baseband data 
                   "fold_nblk":           2,
                   "fold_cufft_nx":       64,
                   "fold_nbyte":          1,
                   "fold_npol":           2,
                   "fold_ndim":           2,
                   "fold_dspsr":          1,
                   "fold_dbdisk":         0,
                   "fold_monitor":        1,
                   "fold_spectrometer":   1,
                   "fold_subint":         10,
                   "fold_software_name":  "baseband2baseband_main",
                   
                   "monitor_keys":            ["deda", "dedc"], # To put monitor data
                   "monitor_ip":      	      '134.104.70.95',
                   "monitor_port":     	      17111,
                   #"monitor_ip":      	      '239.255.255.250',
                   #"monitor_port":     	      1901,
                   #"monitor_ip":      	      '239.3.1.1',
                   #"monitor_port":           2,
                   "monitor_ptype":           2,
}

class PipelineError(Exception):
    pass

PIPELINES = {}

def register_pipeline(name):
    def _register(cls):
        PIPELINES[name] = cls
        return cls
    return _register

class ExecuteCommand(object):

    def __init__(self, command, execution = True, process_index = None):
        self._command = command
        self._process_index = process_index
        self._execution = execution
        self.stdout_callbacks = set()
        self.returncode_callbacks = set()
        self._monitor_threads = []
        self._process = None
        self._executable_command = None
        self._stdout = None
        self._returncode = None
        
        log.debug(self._command)
        self._executable_command = shlex.split(self._command)

        if self._execution:
            try:
                self._process = Popen(self._executable_command,
                                      stdout=PIPE,
                                      stderr=PIPE,
                                      bufsize=1,
                                      universal_newlines=True)
                flags = fcntl(self._process.stdout, F_GETFL)  # Noblock 
                fcntl(self._process.stdout, F_SETFL, flags | O_NONBLOCK)
                flags = fcntl(self._process.stderr, F_GETFL) 
                fcntl(self._process.stderr, F_SETFL, flags | O_NONBLOCK)
                
            except Exception as error:
                log.exception("Error while launching command: {} with error {}".format(
                    self._command, error))
                self.returncode = self._command + "; RETURNCODE is: ' 1'"
            if self._process == None:
                self.returncode = self._command + "; RETURNCODE is: ' 1'"

            # Start monitors
            self._monitor_threads.append(
                threading.Thread(target=self._process_monitor))
            
            for thread in self._monitor_threads:
                thread.start()

    def __del__(self):
        class_name = self.__class__.__name__

    def finish(self):
        if self._execution:
            for thread in self._monitor_threads:
                thread.join()

    def stdout_notify(self):
        for callback in self.stdout_callbacks:
            callback(self._stdout, self)

    @property
    def stdout(self):
        return self._stdout

    @stdout.setter
    def stdout(self, value):
        self._stdout = value
        self.stdout_notify()

    def returncode_notify(self):
        for callback in self.returncode_callbacks:
            callback(self._returncode, self)

    @property
    def returncode(self):
        return self._returncode

    @returncode.setter
    def returncode(self, value):
        self._returncode = value
        self.returncode_notify()

    def _process_monitor(self):
        if self._execution:
            while self._process.poll() == None:
                try:
                    stdout = self._process.stdout.readline().rstrip("\n\r")
                    if stdout != b"":
                        if self._process_index != None:
                            self.stdout = stdout + \
                                          "; PROCESS_INDEX is " + \
                                          str(self._process_index)
                        else:
                            self.stdout = stdout
                except:
                    pass
                
                try:
                    stderr = self._process.stderr.readline().rstrip("\n\r")
                except:
                    pass

            if self._process.returncode:
                self.returncode = self._command + \
                                  "; RETURNCODE is: " +\
                                  str(self._process.returncode)
            log.error("Successful finish execution")
            
class Pipeline(object):
    def __init__(self):
        self._paf_period                     	 = SYSTEM_CONFIG["paf_period"]
        self._paf_df_res                     	 = SYSTEM_CONFIG["paf_df_res"]
        self._paf_df_dtsz                    	 = SYSTEM_CONFIG["paf_df_dtsz"]
        self._paf_df_pktsz                   	 = SYSTEM_CONFIG["paf_df_pktsz"]
        self._paf_df_hdrsz                   	 = SYSTEM_CONFIG["paf_df_hdrsz"]
        self._paf_over_samp_rate             	 = SYSTEM_CONFIG["paf_over_samp_rate"]
        self._paf_nsamp_per_df             	 = SYSTEM_CONFIG["paf_nsamp_per_df"]
        self._paf_nchan_per_chunk            	 = SYSTEM_CONFIG["paf_nchan_per_chunk"]
        self._paf_ndf_per_chunk_per_period       = SYSTEM_CONFIG["paf_ndf_per_chunk_per_period"]
        self._pacifix_ncpu_per_numa_node         = SYSTEM_CONFIG["pacifix_ncpu_per_numa_node"]
        self._pacifix_memory_limit_per_numa_node = SYSTEM_CONFIG["pacifix_memory_limit_per_numa_node"]

        self._execution                    = PIPELINE_CONFIG["execution"]
        self._root_software                = PIPELINE_CONFIG["root_software"]
        self._root_runtime                 = PIPELINE_CONFIG["root_runtime"]
        self._root_data                    = PIPELINE_CONFIG["root_data"]
        self._rbuf_ndf_per_chunk_per_block = PIPELINE_CONFIG["rbuf_ndf_per_chunk_per_block"]
        self._rbuf_blk_res                 = self._paf_df_res * self._rbuf_ndf_per_chunk_per_block
        self._rbuf_nsamp_per_chan_per_block = self._rbuf_ndf_per_chunk_per_block * self._paf_nsamp_per_df
        
        self._input_source_name            = PIPELINE_CONFIG["input_source_name"]                
        self._input_keys                   = PIPELINE_CONFIG["input_keys"]                
        self._input_nblk                   = PIPELINE_CONFIG["input_nblk"]                 
        self._input_nreader                = PIPELINE_CONFIG["input_nreader"]
        self._input_nbyte                  = PIPELINE_CONFIG["input_nbyte"]
        self._input_npol                   = PIPELINE_CONFIG["input_npol"]
        self._input_ndim                   = PIPELINE_CONFIG["input_ndim"]
        self._gpu_ndf_per_chunk_per_stream = PIPELINE_CONFIG["gpu_ndf_per_chunk_per_stream"]
        self._gpu_nstream                  = PIPELINE_CONFIG["gpu_nstream"]                    
        
        self._monitor_keys  = PIPELINE_CONFIG["monitor_keys"]
        self._monitor_ip    = PIPELINE_CONFIG["monitor_ip"]
        self._monitor_port  = PIPELINE_CONFIG["monitor_port"]
        self._monitor_ptype = PIPELINE_CONFIG["monitor_ptype"]      
        
        self._spectrometer_keys     = PIPELINE_CONFIG["spectrometer_keys"]
        self._spectrometer_nblk     = PIPELINE_CONFIG["spectrometer_nblk"]
        self._spectrometer_nreader  = PIPELINE_CONFIG["spectrometer_nreader"]
        self._spectrometer_cufft_nx = PIPELINE_CONFIG["spectrometer_cufft_nx"]
        self._spectrometer_nbyte    = PIPELINE_CONFIG["spectrometer_nbyte"]
        self._spectrometer_ptype    = PIPELINE_CONFIG["spectrometer_ptype"]
        self._spectrometer_ndata_per_samp = PIPELINE_CONFIG["spectrometer_ndata_per_samp"]
        self._spectrometer_ip       = PIPELINE_CONFIG["spectrometer_ip"]
        self._spectrometer_port     = PIPELINE_CONFIG["spectrometer_port"]
        self._spectrometer_dbdisk   = PIPELINE_CONFIG["spectrometer_dbdisk"]
        self._spectrometer_monitor  = PIPELINE_CONFIG["spectrometer_monitor"]
        self._spectrometer_accumulate_nblk = PIPELINE_CONFIG["spectrometer_accumulate_nblk"]
        self._spectrometer_nchan_keep_per_chan = self._spectrometer_cufft_nx / self._paf_over_samp_rate;
        if self._spectrometer_dbdisk:
            self._spectrometer_sod = 1
            self._spectrometer_dbdisk_commands = []
            self._spectrometer_create_rbuf_commands = []
            self._spectrometer_delete_rbuf_commands = []
            self._spectrometer_dbdisk_execution_instances = []
        self._spectrometer_software_name = PIPELINE_CONFIG["spectrometer_software_name"]
        self._spectrometer_main = "{}/src/{}".format(self._root_software, self._spectrometer_software_name)

        self._simultaneous_spectrometer_start_chunk  = PIPELINE_CONFIG["simultaneous_spectrometer_start_chunk"]
        self._simultaneous_spectrometer_nchunk       = PIPELINE_CONFIG["simultaneous_spectrometer_nchunk"]
        self._simultaneous_spectrometer_nchan        = self._simultaneous_spectrometer_nchunk *\
                                                       self._paf_nchan_per_chunk
        
        self._search_keys          = PIPELINE_CONFIG["search_keys"]        
        self._search_nblk          = PIPELINE_CONFIG["search_nblk"]         
        self._search_nchan         = PIPELINE_CONFIG["search_nchan"]        
        self._search_cufft_nx      = PIPELINE_CONFIG["search_cufft_nx"]                       
        self._search_nbyte         = PIPELINE_CONFIG["search_nbyte"]        
        self._search_npol          = PIPELINE_CONFIG["search_npol"]         
        self._search_ndim          = PIPELINE_CONFIG["search_ndim"] 
        self._search_heimdall      = PIPELINE_CONFIG["search_heimdall"]
        self._search_dbdisk        = PIPELINE_CONFIG["search_dbdisk"]
        self._search_monitor       = PIPELINE_CONFIG["search_monitor"]
        self._search_spectrometer  = PIPELINE_CONFIG["search_spectrometer"]
        self._search_sod           = self._search_heimdall or self._search_dbdisk
        self._search_nreader       = (self._search_heimdall + self._search_dbdisk) \
                                     if (self._search_heimdall + self._search_dbdisk) else 1             
        self._search_detect_thresh = PIPELINE_CONFIG["search_detect_thresh"]    
        self._search_dm            = PIPELINE_CONFIG["search_dm"]
        self._search_zap_chans     = PIPELINE_CONFIG["search_zap_chans"]
        self._search_nchan_keep_per_chan = self._search_cufft_nx / self._paf_over_samp_rate;
        if self._search_heimdall:
            self._search_heimdall_commands = []
            self._search_heimdall_execution_instances = []
        if self._search_dbdisk:
            self._search_dbdisk_commands = []
            self._search_dbdisk_execution_instances = []
        self._search_software_name = PIPELINE_CONFIG["search_software_name"]
        self._search_main = "{}/src/{}".format(self._root_software, self._search_software_name)
        
        self._fold_keys       = PIPELINE_CONFIG["fold_keys"]
        self._fold_nblk       = PIPELINE_CONFIG["fold_nblk"]
        self._fold_cufft_nx   = PIPELINE_CONFIG["fold_cufft_nx"]
        self._fold_nbyte      = PIPELINE_CONFIG["fold_nbyte"]
        self._fold_npol       = PIPELINE_CONFIG["fold_npol"]
        self._fold_ndim       = PIPELINE_CONFIG["fold_ndim"]
        self._fold_dspsr      = PIPELINE_CONFIG["fold_dspsr"]
        self._fold_dbdisk     = PIPELINE_CONFIG["fold_dbdisk"]
        self._fold_monitor    = PIPELINE_CONFIG["fold_monitor"]
        self._fold_spectrometer = PIPELINE_CONFIG["fold_spectrometer"]
        self._fold_subint     = PIPELINE_CONFIG["fold_subint"]
        self._fold_nchan_keep_per_chan = self._fold_cufft_nx / self._paf_over_samp_rate;
        self._fold_sod       = self._fold_dspsr or self._dolf_dbdisk
        self._fold_nreader   = (self._fold_dspsr + self._fold_dbdisk) \
                               if (self._fold_dspsr + self._fold_dbdisk) else 1 
        if self._fold_dspsr:
            self._fold_dspsr_commands = []
            self._fold_dspsr_execution_instances = []
        if self._fold_dbdisk:
            self._fold_dbdisk_commands = []
            self._fold_dbdisk_execution_instances = []
        self._fold_software_name = PIPELINE_CONFIG["fold_software_name"]
        self._fold_main       = "{}/src/{}".format(self._root_software, self._fold_software_name)

        self._pipeline_runtime_directory = []

        self._input_create_rbuf_commands = []
        self._fold_create_rbuf_commands = []
        self._search_create_rbuf_commands = []
        self._input_delete_rbuf_commands = []
        self._fold_delete_rbuf_commands = []
        self._search_delete_rbuf_commands = []

        self._input_commands = []        
        self._fold_commands = []
        self._search_commands = []
        self._spectrometer_commands = []

        self._fold_execution_instances = []
        self._search_execution_instances = []
        self._spectrometer_execution_instances = []
        self._input_execution_instances = []

        # To see if we can process input data with integer repeats
        if self._rbuf_ndf_per_chunk_per_block % \
           (self._gpu_ndf_per_chunk_per_stream * self._gpu_nstream):
            raise PipelineError("data in input ring buffer block can only "
                                "be processed by baseband2baseband with integer repeats")

        
        self._cleanup_commands_at_config = ["pkill -9 -f capture",
                                            "pkill -9 -f dspsr",
                                            "pkill -9 -f dada_db",
                                            "pkill -9 -f heimdall",
                                            "pkill -9 -f dada_diskdb",
                                            "pkill -9 -f dada_dbdisk",
                                            "pkill -9 -f baseband2filter", # process name, maximum 16 bytes (15 bytes visiable)
                                            "pkill -9 -f baseband2spectr", # process name, maximum 16 bytes (15 bytes visiable)
                                            "pkill -9 -f baseband2baseba", # process name, maximum 16 bytes (15 bytes visiable)
                                            "ipcrm -a"]
        self._cleanup_commands_at_start = ["pkill -9 -f dspsr",
                                           "pkill -9 -f dada_db",
                                           "pkill -9 -f heimdall",
                                           "pkill -9 -f dada_diskdb",
                                           "pkill -9 -f dada_dbdisk",
                                           "pkill -9 -f baseband2filter", # process name, maximum 16 bytes (15 bytes visiable)
                                           "pkill -9 -f baseband2spectr", # process name, maximum 16 bytes (15 bytes visiable)
                                           "pkill -9 -f baseband2baseba"]
        
        # Cleanup at very beginning
        self._cleanup(self._cleanup_commands_at_config)       
        
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

    def _handle_execution_stdout(self, stdout, callback):
        if self._execution:
            log.debug(stdout)

    def _handle_execution_returncode(self, returncode, callback):
        if self._execution and returncode:
            log.debug(returncode)
            self._cleanup(self._cleanup_commands_at_config)
            raise PipelineError(returncode)

    def _cleanup(self, cleanup_commands):
        # Kill existing process and free shared memory if there is any
        execution_instances = []
        for command in cleanup_commands:
            execution_instances.append(ExecuteCommand(command, self._execution))            
        for execution_instance in execution_instances:         # Wait until the cleanup is done
            execution_instance.finish()

@register_pipeline("Fold")
class Fold(Pipeline):
    def __init__(self):
        super(Fold, self).__init__()

    def configure(self, input_ip, input_source, length, input_freq):
        # Setup parameters of the pipeline
        self._input_ip     = input_ip
        self._input_freq   = input_freq
        self._input_source = input_source
        self._numa         = int(self._input_ip.split(".")[3]) - 1
        self._server       = int(self._input_ip.split(".")[2])
        
        self._input_nbeam      = self._input_source["input_nbeam"]
        self._input_nchunk     = self._input_source["input_nchunk"]
        self._input_nchan      = self._input_nchunk * self._paf_nchan_per_chunk
        self._input_dada_fname = "{}/{}/{}_{}chunks.dada".format(self._root_data,
                                                                 self._input_source_name,
                                                                 self._input_source_name,
                                                                 self._input_nchunk)
        
        self._input_blksz = self._input_nchunk * self._paf_df_dtsz * self._rbuf_ndf_per_chunk_per_block
        self._fold_blksz  = int(self._input_blksz * self._fold_nbyte * self._fold_npol * self._fold_ndim /
                                (self._paf_over_samp_rate * self._input_nbyte * self._input_npol * self._input_ndim))
        
        # To see if we have enough memory
        self._simultaneous_spectrometer_blksz = self._simultaneous_spectrometer_nchan * \
                                                self._spectrometer_nchan_keep_per_chan * \
                                                self._spectrometer_ndata_per_samp * \
                                                self._spectrometer_nbyte * \
                                                (self._spectrometer_dbdisk and self._fold_spectrometer)
        if self._input_nbeam*(self._input_blksz * self._input_nblk +
                              self._fold_blksz * self._fold_nblk + 
                              self._simultaneous_spectrometer_blksz * self._spectrometer_nblk) >\
                              self._pacifix_memory_limit_per_numa_node:
            raise PipelineError("We do not have enough shared memory for the setup "
                                "Try to reduce the ring buffer block number, or  "
                                "reduce the number of packets in each ring buffer block, or "
                                "reduce the number of frequency chunks for spectral (if there is any)")

        # To check the dada file and setup file_size to meet the length
        self._input_file_size = int(self._input_nchunk * self._paf_df_dtsz / \
                                    self._paf_df_res * length)
        log.debug("Required FILE_SIZE is {}".format(self._input_file_size))

        # To see if we can fit FFT into input
        if self._rbuf_nsamp_per_chan_per_block % self._fold_cufft_nx:
            raise PipelineError("self._rbuf_nsamp_per_chan_per_block should be multiple times of self._fold_cufft_nx")
        if self._fold_spectrometer and (self._rbuf_nsamp_per_chan_per_block % self._spectrometer_cufft_nx):
            raise PipelineError("self._rbuf_nsamp_per_chan_per_block should be multiple times of self._spectrometer_cufft_nx")

        # To check output pol type
        if self._fold_monitor and (self._monitor_ptype not in [1, 2, 4]):
            log.error("Monitor pol type should be 1, 2 or 4, but it is {}".format(self._monitor_ptype))
            raise PipelineError("Monnitor pol type should be 1, 2 or 4, but it is {}".format(self._monitor_ptype))
        if self._fold_spectrometer and (self._spectrometer_ptype not in [1, 2, 4]):
            log.error("Spectrometer pol type should be 1, 2 or 4, but it is {}".format(self._spectrometer_ptype))
            raise PipelineError("Spectrometer pol type should be 1, 2 or 4, but it is {}".format(self._spectrometer_ptype))

        # To see if the DADA data file is there
        if not os.path.isfile(self._input_dada_fname):
            raise PipelineError("DADA file {} is not there".format(self._input_dada_fname))

        # To see if the software is available
        if not os.path.isfile(self._fold_main):
            raise PipelineError("{} is not there".format(self._fold_main))

        # To set up rest configurations
        for i in range(self._input_nbeam):      
            # To get directory for runtime information
            pipeline_runtime_directory = "{}/pacifix{}_numa{}_process{}".format(self._root_runtime,
                                                                                self._server,
                                                                                self._numa,
                                                                                i)
            if not os.path.isdir(pipeline_runtime_directory):
                try:
                    os.makedirs(pipeline_runtime_directory)
                except Exception as error:
                    log.exception(error)
                    raise PipelineError("{} is not there and fail to create it".format(pipeline_runtime_directory))
            self._pipeline_runtime_directory.append(pipeline_runtime_directory)
 
            # Command to create input ring buffer
            self._input_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                     "-b {:} -n {:} -r {:}").format(self._input_keys[i],
                                                                                    self._input_blksz,
                                                                                    self._input_nblk,
                                                                                    self._input_nreader))

            # command to create fold ring buffer
            self._fold_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                    "-b {:} -n {:} -r {:}").format(self._fold_keys[i],
                                                                                   self._fold_blksz,
                                                                                   self._fold_nblk,
                                                                                   self._fold_nreader))
            
            # input command
            self._input_commands.append("dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(self._input_keys[i],
                                                                                        self._input_dada_fname,
                                                                                        0))

            # Setup baseband2baseband
            command = ("{} -a {} -b {} -c {} -d {} "
                       "-e {} -f {} -g {} -i {} ").format(self._fold_main, self._input_keys[i], 
                                                          self._fold_keys[i], self._rbuf_ndf_per_chunk_per_block,
                                                          self._gpu_nstream, self._gpu_ndf_per_chunk_per_stream,
                                                          self._pipeline_runtime_directory[i],
                                                          self._input_nchunk, self._fold_cufft_nx)
            if self._fold_sod:
                command += "-j 1 "
            else:
                command += "-j 0 "
            if self._fold_monitor:
                command += "-k Y_{}_{}_{} ".format(self._monitor_ip,
                                                   self._monitor_port,
                                                   self._monitor_ptype)
            else:
                command += "-k N "
            self._fold_commands.append(command)

            # Command to run dspsr
            if self._fold_dspsr:
                kfname = "{}/dspsr.key".format(self._pipeline_runtime_directory[i])
                kfile = open(kfname, "w")
                kfile.writelines("DADA INFO:\n")
                kfile.writelines("key {:s}\n".format(self._fold_keys[i]))
                kfile.close()            
                if not os.path.isfile(kfname):
                    raise PipelineError("{} is not exist".format(kfname))
                pfname = "/home/pulsar/xinping/phased-array-feed/config/{}.par".format(self._input_source_name)
                if not os.path.isfile(pfname):
                    raise PipelineError("{} is not exist".format(pfname))
                self._fold_dspsr_commands.append(("dspsr -b 1024 -L {} -A "
                                                  "-E {} -cuda 0,0 {}").format(self._fold_subint,
                                                                               pfname,
                                                                               kfname))  

            if self._fold_dbdisk:
                self._fold_dbdisk_commands.append("dada_dbdisk -W -k {} -D "
                                                  "{} -o -s -z".format(self._fold_keys[i],
                                                                       self._pipeline_runtime_directory[i]))
                
            # command to delete input ring buffer
            self._input_delete_rbuf_commands.append(
                "dada_db -d -k {:}".format(self._input_keys[i]))

            # command to delete fold ring buffer
            self._fold_delete_rbuf_commands.append(
                "dada_db -d -k {:}".format(self._fold_keys[i]))

    def start(self):
        self._cleanup(self._cleanup_commands_at_start)
        
        # Update FILE_SIZE
        command = "dada_install_header -p FILE_SIZE={} {}".format(self._input_file_size,
                                                                  self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update RECEIVER
        command = "dada_install_header -p RECEIVER=0 {}".format(self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update NCHAN
        command = "dada_install_header -p NCHAN={} {}".format(self._input_nchan,
                                                              self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update BW
        command = "dada_install_header -p BW={} {}".format(self._input_nchan,
                                                           self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()        
        # Update FREQ
        command = "dada_install_header -p FREQ={} {}".format(self._input_freq,
                                                             self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()

        # Create input ring buffer
        process_index = 0
        execution_instances = []
        for command in self._input_create_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution,
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
        
        # Create ring buffer for fold data
        process_index = 0
        execution_instances = []
        for command in self._fold_create_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution,
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:         # Wait until the buffer creation is done
            execution_instance.finish()
            
        # Execute the input
        process_index = 0
        self._input_execution_instances = []
        for command in self._input_commands:
            execution_instance = ExecuteCommand(command, self._execution, process_index)
            #self._execution_instances.append(execution_instance)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._input_execution_instances.append(execution_instance)
            process_index += 1
            
        # Run dspsr
        if self._fold_dspsr:
            process_index = 0
            self._fold_dspsr_execution_instances = []
            for command in self._fold_dspsr_commands:
                execution_instance = ExecuteCommand(command, self._execution, process_index)
                execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                self._fold_dspsr_execution_instances.append(execution_instance)
                process_index += 1 

        # Run dbdisk
        if self._fold_dbdisk:
            process_index = 0
            self._fold_dbdisk_execution_instances = []
            for command in self._fold_dbdisk_commands:
                execution_instance = ExecuteCommand(command, self._execution, process_index)
                execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                self._fold_dbdisk_execution_instances.append(execution_instance)
                process_index += 1 

        # Run baseband2baseband
        process_index = 0
        self._fold_execution_instances = []
        for command in self._fold_commands:
            execution_instance = ExecuteCommand(command, self._execution, process_index)
            execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._fold_execution_instances.append(execution_instance)
            process_index += 1
            
    def stop(self):
        if self._fold_dspsr:
            for execution_instance in self._fold_dspsr_execution_instances:
                execution_instance.finish()
                
        if self._fold_dbdisk:
            for execution_instance in self._fold_dbdisk_execution_instances:
                execution_instance.finish()

        for execution_instance in self._fold_execution_instances:
            execution_instance.finish()

        for execution_instance in self._input_execution_instances:
            execution_instance.finish()
            
        # To delete baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._fold_delete_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, 
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
        
        # To delete input ring buffer
        process_index = 0
        execution_instances = []
        for command in self._input_delete_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, 
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
            
    def deconfigure(self):
        pass

@register_pipeline("Fold2Beams")
class Fold2Beams(Fold):

    def __init__(self):
        super(Fold2Beams, self).__init__()

    def configure(self, input_ip, length, input_freq):
        super(Fold2Beams, self).configure(input_ip,
                                          INPUT_2BEAMS,
                                          length,
                                          input_freq)

    def start(self):
        super(Fold2Beams, self).start()

    def stop(self):
        super(Fold2Beams, self).stop()

    def deconfigure(self):
        super(Fold2Beams, self).deconfigure()

@register_pipeline("Fold1Beam")
class Fold1Beam(Fold):

    def __init__(self):
        super(Fold1Beam, self).__init__()

    def configure(self, input_ip, length, input_freq):
        super(Fold1Beam, self).configure(input_ip,
                                         INPUT_1BEAM,
                                         length,
                                         input_freq)

    def start(self):
        super(Fold1Beam, self).start()

    def stop(self):
        super(Fold1Beam, self).stop()

    def deconfigure(self):
        super(Fold1Beam, self).deconfigure()

@register_pipeline("Search")
class Search(Pipeline):
    def __init__(self):
        super(Search, self).__init__()
        
    def configure(self, input_ip, input_source, length, input_freq):
        # Setup parameters of the pipeline
        self._input_ip = input_ip
        self._input_freq   = input_freq
        self._input_source = input_source
        self._numa = int(ip.split(".")[3]) - 1
        self._server = int(ip.split(".")[2])

        self._input_nbeam  = self._input_source["input_nbeam"]
        self._input_nchunk = self._input_source["input_nchunk"]
        self._input_nchan  = self._input_nchunk * self._paf_nchan_per_chunk
        self._input_dada_fname = "{}/{}/{}_{}chunks.dada".format(self._root_data,
                                                                 self._input_source_name,
                                                                 self._input_source_name,
                                                                 self._input_nchunk)
                
        self._input_blksz  = self._input_nchunk *\
                             self._paf_df_dtsz * \
                             self._rbuf_ndf_per_chunk_per_block
        self._search_blksz = int(self._input_blksz * self._search_nchan *
                                 self._search_nbyte * self._search_npol *
                                 self._search_ndim / float(self._input_nbyte *
                                                           self._input_npol *
                                                           self._input_ndim *
                                                           self._input_nchan *
                                                           self._search_cufft_nx))
        
        # To see if we have enough memory        
        self._simultaneous_spectrometer_blksz = self._simultaneous_spectrometer_nchan * \
                                                self._spectrometer_nchan_keep_per_chan * \
                                                self._spectrometer_ndata_per_samp * \
                                                self._spectrometer_nbyte * \
                                                (self._spectrometer_dbdisk and self._search_spectrometer)
        #log.error(self._simultaneous_spectrometer_blksz)
        if self._input_nbeam*(self._input_blksz * self._input_nblk +\
                              self._search_blksz * self._search_nblk +\
                              self._simultaneous_spectrometer_blksz * self._spectrometer_nblk) >\
                              self._pacifix_memory_limit_per_numa_node:
            raise PipelineError("We do not have enough shared memory for the setup "
                                "Try to reduce the ring buffer block number, or  "
                                "reduce the number of packets in each ring buffer block, or "
                                "reduce the number of frequency chunks for spectral (if there is any)")
        
        # To change the FILE_SIZE in dada file, in this we control the length of run
        self._input_file_size = int(self._input_nchunk * self._paf_df_dtsz / self._paf_df_res * length)
        log.debug("Required FILE_SIZE is {}".format(self._input_file_size))

        # To see if we can fit FFT into input samples
        if self._rbuf_nsamp_per_chan_per_block % self._search_cufft_nx:
            raise PipelineError("self._rbuf_nsamp_per_chan_per_block should be multiple times of self._search_cufft_nx")
        if self._search_spectrometer and (self._rbuf_nsamp_per_chan_per_block % self._spectrometer_cufft_nx):
            raise PipelineError("self._rbuf_nsamp_per_chan_per_block should be multiple times of self._spectrometer_cufft_nx")

        # To check pol type
        if self._search_monitor and (self._monitor_ptype not in [1, 2, 4]):
            log.error("Monitor pol type should be 1, 2 or 4, but it is {}".format(self._monitor_ptype))
            raise PipelineError("Monnitor pol type should be 1, 2 or 4, but it is {}".format(self._monitor_ptype))
        if self._search_spectrometer and (self._spectrometer_ptype not in [1, 2, 4]):
            log.error("Spectrometer pol type should be 1, 2 or 4, but it is {}".format(self._spectrometer_ptype))
            raise PipelineError("Spectrometer pol type should be 1, 2 or 4, but it is {}".format(self._spectrometer_ptype))

        # To see if the DADA data file is there
        if not os.path.isfile(self._input_dada_fname):
            raise PipelineError("DADA file {} is not there".format(self._input_dada_fname))

        # To see if the software is available
        if not os.path.isfile(self._search_main):
            raise PipelineError("{} is not there".format(self._search_main))

        # To setup commands for each process
        for i in range(self._input_nbeam):      
            # To get directory for runtime information
            pipeline_runtime_directory = "{}/pacifix{}_numa{}_process{}".format(self._root_runtime,
                                                                                self._server,
                                                                                self._numa,
                                                                                i)
            if not os.path.isdir(pipeline_runtime_directory):
                try:
                    os.makedirs(pipeline_runtime_directory)
                except Exception as error:
                    log.exception(error)
                    raise PipelineError("Fail to create {}".format(pipeline_runtime_directory))
            self._pipeline_runtime_directory.append(pipeline_runtime_directory)

            # command to create baseband ring buffer
            self._input_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                     "-b {:} -n {:} -r {:}").format(self._input_keys[i],
                                                                                    self._input_blksz,
                                                                                    self._input_nblk,
                                                                                    self._input_nreader))
            # Command to create filterbank ring buffer
            self._search_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                      "-b {:} -n {:} -r {:}").format(self._search_keys[i],
                                                                                     self._search_blksz,
                                                                                     self._search_nblk,
                                                                                     self._search_nreader))
            
            if self._search_spectrometer and self._spectrometer_dbdisk:
                self._spectrometer_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                                "-b {:} -n {:} -r {:}").format(self._spectrometer_keys[i],
                                                                                               self._simultaneous_spectrometer_blksz,
                                                                                               self._spectrometer_nblk,
                                                                                               self._spectrometer_nreader))
                
            # input command
            self._input_commands.append("dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(self._input_keys[i],
                                                                                        self._input_dada_fname,
                                                                                        0))
            
            # Command to run heimdall
            if self._search_heimdall:
                command = ("heimdall -k {} -detect_thresh {} -output_dir {} ").format(
                    self._search_keys[i],
                    self._search_detect_thresh,
                    self._pipeline_runtime_directory[i])
                if self._search_zap_chans:
                    zap = ""
                    for zap_chan in self._search_zap_chans:
                        zap += " -zap_chans {} {}".format(
                            zap_chan[0], zap_chan[1])
                    command += zap
                if self._search_dm:
                    command += "-dm {} {}".format(self._search_dm[0],
                                                  self._search_dm[1])
                self._search_heimdall_commands.append(command)

            if self._search_dbdisk:
                # Command to run dbdisk
                command = "dada_dbdisk -W -k {} -D {} -o -s -z".format(self._search_keys[i],
                                                                       self._pipeline_runtime_directory[i])
                self._search_dbdisk_commands.append(command)
            
            if self._search_spectrometer and self._spectrometer_dbdisk:
                command = "dada_dbdisk -W -k {} -D {} -o -s -z".format(self._spectrometer_keys[i],
                                                                       self._pipeline_runtime_directory[i])
                self._spectrometer_dbdisk_commands.append(command)
                
            # baseband2filterbank command
            command = ("{} -a {} -b {} -c {} -d {} -e {} "
                       "-f {} -i {} -j {} -k {} ").format(self._search_main, self._input_keys[i],
                                                                         self._search_keys[i], self._rbuf_ndf_per_chunk_per_block, self._gpu_nstream,
                                                                         self._gpu_ndf_per_chunk_per_stream, self._pipeline_runtime_directory[i],
                                                                         self._input_nchunk, self._search_cufft_nx,
                                                                         self._search_nchan)

            if self._search_spectrometer:                
                if self._spectrometer_dbdisk:
                    command += "-m k_{}_{}_{}_{}_{}_{}_{} ".format(self._spectrometer_keys[i],
                                                                   self._spectrometer_sod,
                                                                   self._spectrometer_ptype,
                                                                   self._simultaneous_spectrometer_start_chunk,
                                                                   self._simultaneous_spectrometer_nchunk,
                                                                   self._spectrometer_accumulate_nblk,
                                                                   self._spectrometer_cufft_nx)
                else:
                    command += "-m n_{}_{}_{}_{}_{}_{}_{} ".format(self._spectrometer_ip,
                                                                   self._spectrometer_port,
                                                                   self._spectrometer_ptype,
                                                                   self._simultaneous_spectrometer_start_chunk,
                                                                   self._simultaneous_spectrometer_nchunk,
                                                                   self._spectrometer_accumulate_nblk,
                                                                   self._spectrometer_cufft_nx)
            else:
                command += "-m N "                    
            
            if self._search_sod:
                command += "-g 1 "
            else:
                command += "-g 0 "
            if self._search_monitor:
                command += "-l Y_{}_{}_{} ".format(self._monitor_ip,
                                                   self._monitor_port,
                                                   self._monitor_ptype)
            else:
                command += "-l N "                
            self._search_commands.append(command)

            # command to delete search ring buffer
            self._search_delete_rbuf_commands.append(
                "dada_db -d -k {:}".format(self._search_keys[i]))

            # command to delete baseband ring buffer
            self._input_delete_rbuf_commands.append(
                "dada_db -d -k {:}".format(self._input_keys[i]))
            
            if self._search_spectrometer and self._spectrometer_dbdisk:
                self._spectrometer_delete_rbuf_commands.append(
                    "dada_db -d -k {:}".format(self._spectrometer_keys[i]))

    def start(self):
        self._cleanup(self._cleanup_commands_at_start)
        
        # Update FILE_SIZE
        command = "dada_install_header -p FILE_SIZE={} {}".format(self._input_file_size,
                                                                  self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update RECEIVER
        command = "dada_install_header -p RECEIVER=0 {}".format(self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update NCHAN
        command = "dada_install_header -p NCHAN={} {}".format(self._input_nchan,
                                                              self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update BW
        command = "dada_install_header -p BW={} {}".format(self._input_nchan,
                                                           self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()        
        # Update FREQ
        command = "dada_install_header -p FREQ={} {}".format(self._input_freq,
                                                             self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        
        # Create input ring buffer
        process_index = 0
        execution_instances = []
        for command in self._input_create_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, 
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()

        # Create ring buffer for simultaneous spectrometer output
        if self._search_spectrometer and self._spectrometer_dbdisk:
            process_index = 0
            execution_instances = []
            for command in self._spectrometer_create_rbuf_commands:
                execution_instances.append(ExecuteCommand(command, self._execution, 
                                                          process_index))
                process_index += 1
            for execution_instance in execution_instances:
                execution_instance.finish()
            
        # Create ring buffer for search data
        process_index = 0
        execution_instances = []
        for command in self._search_create_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, 
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:         # Wait until the buffer creation is done
            execution_instance.finish()

        # Execute the diskdb
        process_index = 0
        self._input_execution_instances = []
        for command in self._input_commands:
            execution_instance = ExecuteCommand(command, self._execution, 
                                                process_index)
            #self._execution_instances.append(execution_instance)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._input_execution_instances.append(execution_instance)
            process_index += 1
            
        # Run baseband2filterbank
        process_index = 0
        self._search_execution_instances = []
        for command in self._search_commands:
            execution_instance = ExecuteCommand(command, self._execution, process_index)
            execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._search_execution_instances.append(execution_instance)
            process_index += 1 

        if self._search_heimdall:  # run heimdall if required
            process_index = 0
            self._search_heimdall_execution_instances = []
            for command in self._search_heimdall_commands:
                execution_instance = ExecuteCommand(command, self._execution, 
                                                    process_index)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                self._search_heimdall_execution_instances.append(execution_instance)
                process_index += 1

        if self._search_dbdisk:
            process_index = 0
            self._search_dbdisk_execution_instances = []
            for command in self._search_dbdisk_commands:
                execution_instance = ExecuteCommand(command, self._execution, 
                                                    process_index)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                self._search_dbdisk_execution_instances.append(execution_instance)
                process_index += 1
        
        if self._search_spectrometer and self._spectrometer_dbdisk:
            process_index = 0
            self._spectrometer_dbdisk_execution_instances = []
            for command in self._spectrometer_dbdisk_commands:
                execution_instance = ExecuteCommand(command, self._execution, 
                                                    process_index)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                self._spectrometer_dbdisk_execution_instances.append(execution_instance)
                process_index += 1

    def stop(self):
        if self._search_dbdisk:
            for execution_instance in self._search_dbdisk_execution_instances:
                execution_instance.finish()
                
        if self._search_heimdall:
            for execution_instance in self._search_heimdall_execution_instances:
                execution_instance.finish()
                
        if self._search_spectrometer and self._spectrometer_dbdisk:
            for execution_instance in self._spectrometer_dbdisk_execution_instances:
                execution_instance.finish()
                
        for execution_instance in self._search_execution_instances:
            execution_instance.finish()

        for execution_instance in self._input_execution_instances:
            execution_instance.finish()

        # To delete simultaneous spectral output buffer
        if self._search_spectrometer and self._spectrometer_dbdisk:
            process_index = 0
            execution_instances = []
            for command in self._spectrometer_delete_rbuf_commands:
                execution_instances.append(ExecuteCommand(command, self._execution, 
                                                          process_index))
                process_index += 1
            for execution_instance in execution_instances:
                execution_instance.finish()
                
        # To delete search ring buffer
        process_index = 0
        execution_instances = []
        for command in self._search_delete_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, 
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
        
        # To delete input ring buffer
        process_index = 0
        execution_instances = []
        for command in self._input_delete_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, 
                                                      process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
            
    def deconfigure(self):
        pass
        
@register_pipeline("Search2Beams")
class Search2Beams(Search):

    def __init__(self):
        super(Search2Beams, self).__init__()

    def configure(self, input_ip, length, input_freq):
        super(Search2Beams, self).configure(input_ip,
                                            INPUT_2BEAMS,
                                            length,
                                            input_freq)

    def start(self):
        super(Search2Beams, self).start()

    def stop(self):
        super(Search2Beams, self).stop()

    def deconfigure(self):
        super(Search2Beams, self).deconfigure()


@register_pipeline("Search1Beam")
class Search1Beam(Search):

    def __init__(self):
        super(Search1Beam, self).__init__()

    def configure(self, input_ip, length, input_freq):
        super(Search1Beam, self).configure(input_ip,
                                           INPUT_1BEAM,
                                           length,
                                           input_freq)

    def start(self):
        super(Search1Beam, self).start()

    def stop(self):
        super(Search1Beam, self).stop()

    def deconfigure(self):
        super(Search1Beam, self).deconfigure()

@register_pipeline("Spectral")
class Spectrometer(Pipeline):
    def __init__(self):
        super(Spectrometer, self).__init__()
        
    def configure(self, input_ip, input_source, length, input_freq):
        # Setup parameters of the pipeline
        self._input_ip = input_ip
        self._input_freq   = input_freq
        self._input_source = input_source
        self._numa = int(ip.split(".")[3]) - 1
        self._server = int(ip.split(".")[2])
        
        self._input_nbeam = self._input_source["input_nbeam"]
        self._input_nchunk = self._input_source["input_nchunk"]
        self._input_nchan  = self._input_nchunk * self._paf_nchan_per_chunk
        self._input_nbeam  = self._input_source["input_nbeam"]
        self._input_dada_fname = "{}/{}/{}_{}chunks.dada".format(self._root_data,
                                                                 self._input_source_name,
                                                                 self._input_source_name,
                                                                 self._input_nchunk)
        
        self._input_blksz        = self._input_nchunk * self._paf_df_dtsz * self._rbuf_ndf_per_chunk_per_block
        self._spectrometer_blksz = int(self._spectrometer_ndata_per_samp * self._input_nchan * 
                                       self._spectrometer_cufft_nx /
                                       self._paf_over_samp_rate *
                                       self._spectrometer_nbyte *
                                       self._spectrometer_dbdisk)

        # To check pol type
        if self._spectrometer_ptype not in [1, 2, 4]:  # We can only have three possibilities
            log.error("Spectrometer pol type should be 1, 2 or 4, but it is {}".format(self._spectrometer_ptype))
            raise PipelineError("Spectrometer pol type should be 1, 2 or 4, but it is {}".format(self._spectrometer_ptype))
        if self._spectrometer_monitor and (self._monitor_ptype not in [1, 2, 4]):
            log.error("Monitor pol type should be 1, 2 or 4, but it is {}".format(self._monitor_ptype))
            raise PipelineError("monitor pol type should be 1, 2 or 4, but it is {}".format(self._monitor_ptype))
        
        # To see if we have enough memory
        if self._input_nbeam*(self._input_blksz * self._input_nblk + \
                        self._spectrometer_blksz * self._spectrometer_nblk) > \
                        self._pacifix_memory_limit_per_numa_node:
            raise PipelineError("We do not have enough shared memory for the setup "
                                "Try to reduce the ring buffer block number "
                                "or reduce the number of packets in each ring buffer block")
        
        # Get the required file size
        self._input_file_size = int(self._input_nchunk * self._paf_df_dtsz / self._paf_df_res * length)
        
        # To check the existing of files
        if not os.path.isfile(self._spectrometer_main):
            raise PipelineError("{} is not exist".format(self._spectrometer_main))
        if not os.path.isfile(self._input_dada_fname):
            raise PipelineError("{} is not exist".format(self._input_dada_fname))
        
        for i in range(self._input_nbeam):      
            # To get directory for runtime information
            pipeline_runtime_directory = "{}/pacifix{}_numa{}_process{}".format(self._root_runtime,
                                                                                self._server,
                                                                                self._numa, i)
            if not os.path.isdir(pipeline_runtime_directory):
                    try:
                        os.makedirs(pipeline_runtime_directory)
                    except Exception as error:
                        log.exception(error)
                        raise PipelineError("Fail to create {}".format(pipeline_runtime_directory))
            self._pipeline_runtime_directory.append(pipeline_runtime_directory)

            # diskdb command
            self._input_commands.append("dada_diskdb -k {} -f {} -o {} -s".format(self._input_keys[i],
                                                                                  self._input_dada_fname,
                                                                                  0))

            # spectrometer command
            command = "{} -a {} -c {} -d {} -e {} -f {} -g {} -i {} -j {} -k {} ".format(
                self._spectrometer_main,
                self._input_keys[i],self._rbuf_ndf_per_chunk_per_block,
                self._gpu_nstream, self._gpu_ndf_per_chunk_per_stream,
                self._pipeline_runtime_directory[i], self._input_nchunk,
                self._spectrometer_cufft_nx, self._spectrometer_ptype, self._spectrometer_accumulate_nblk)
            if self._spectrometer_dbdisk:
                command += "-b k_{}_{} ".format(self._spectrometer_keys[i], self._spectrometer_sod)
            else:
                command += "-b n_{}_{} ".format(self._spectrometer_ip, self._spectrometer_port)
            
            if self._spectrometer_monitor:
                command += "-l Y_{}_{}_{} ".format(self._monitor_ip,
                                                   self._monitor_port,
                                                   self._monitor_ptype)
            else:
                command += "-l N "    
            self._spectrometer_commands.append(command)

            # Command to create spectral ring buffer
            if self._spectrometer_dbdisk:
                self._spectrometer_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                                "-b {:} -n {:} -r {:}").format(
                                                                    self._spectrometer_keys[i],
                                                                    self._spectrometer_blksz,
                                                                    self._spectrometer_nblk,
                                                                    self._spectrometer_nreader))

            # command to create baseband ring buffer
            self._input_create_rbuf_commands.append(("dada_db -l -p -k {:} "
                                                     "-b {:} -n {:} -r {:}").format(
                                                         self._input_keys[i],
                                                         self._input_blksz,
                                                         self._input_nblk,
                                                         self._input_nreader))
            
            # command to delete spectral ring buffer
            if self._spectrometer_dbdisk:
                self._spectrometer_delete_rbuf_commands.append(
                    "dada_db -d -k {:}".format(
                    self._spectrometer_keys[i]))
                
            # command to delete baseband ring buffer
            self._input_delete_rbuf_commands.append(
                "dada_db -d -k {:}".format(
                    self._input_keys[i]))

            # Command to run dbdisk
            if self._spectrometer_dbdisk:
                command = "dada_dbdisk -W -k {} -D {} -o -s -z".format(self._spectrometer_keys[i], self._pipeline_runtime_directory[i])
                self._spectrometer_dbdisk_commands.append(command)

    def start(self):
        self._cleanup(self._cleanup_commands_at_start)
        
        # Update FILE_SIZE
        command = "dada_install_header -p FILE_SIZE={} {}".format(self._input_file_size,
                                                                  self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update RECEIVER
        command = "dada_install_header -p RECEIVER=0 {}".format(self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update NCHAN
        command = "dada_install_header -p NCHAN={} {}".format(self._input_nchan,
                                                              self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        # Update BW
        command = "dada_install_header -p BW={} {}".format(self._input_nchan,
                                                           self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()        
        # Update FREQ
        command = "dada_install_header -p FREQ={} {}".format(self._input_freq,
                                                             self._input_dada_fname)
        execution_instance = ExecuteCommand(command, self._execution)
        execution_instance.finish()
        
        # Create baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._input_create_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution,  process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()

        # Create ring buffer for spectral data
        if self._spectrometer_dbdisk:
            process_index = 0
            execution_instances = []
            for command in self._spectrometer_create_rbuf_commands:
                execution_instances.append(ExecuteCommand(command, self._execution,  process_index))
                process_index += 1
            for execution_instance in execution_instances:         # Wait until the buffer creation is done
                execution_instance.finish()
        
        # Execute the diskdb
        process_index = 0
        self._input_execution_instances = []
        for command in self._input_commands:
            execution_instance = ExecuteCommand(command, self._execution,  process_index)
            #self._execution_instances.append(execution_instance)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._input_execution_instances.append(execution_instance)
            process_index += 1
            
        # Run baseband2spectral
        process_index = 0
        self._spectrometer_execution_instances = []
        for command in self._spectrometer_commands:
            execution_instance = ExecuteCommand(command, self._execution,  process_index)
            execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._spectrometer_execution_instances.append(execution_instance)
            process_index += 1 
                
        if self._spectrometer_dbdisk:
            process_index = 0
            self._spectrometer_dbdisk_execution_instances = []
            for command in self._spectrometer_dbdisk_commands:
                execution_instance = ExecuteCommand(command, self._execution,  process_index)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                self._spectrometer_dbdisk_execution_instances.append(execution_instance)
                process_index += 1

    def stop(self):
        if self._spectrometer_dbdisk:
            for execution_instance in self._spectrometer_dbdisk_execution_instances:
                execution_instance.finish()

        for execution_instance in self._spectrometer_execution_instances:
            execution_instance.finish()

        for execution_instance in self._input_execution_instances:
            execution_instance.finish()
            
        # To delete spectral ring buffer
        if self._spectrometer_dbdisk:
            process_index = 0
            execution_instances = []
            for command in self._spectrometer_delete_rbuf_commands:
                execution_instances.append(ExecuteCommand(command, self._execution, process_index))
                process_index += 1
            for execution_instance in execution_instances:
                execution_instance.finish()
        
        # To delete baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._input_delete_rbuf_commands:
            execution_instances.append(ExecuteCommand(command, self._execution, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
            
    def deconfigure(self):
        pass

@register_pipeline("Spectrometer1Beam")
class Spectrometer1Beam(Spectrometer):
    def configure(self, input_ip, length, input_freq):
        super(Spectrometer1Beam, self).configure(input_ip, INPUT_1BEAM, length, input_freq)

@register_pipeline("Spectrometer2Beams")
class Spectrometer2Beams(Spectrometer):
    def configure(self, input_ip, length, input_freq):
        super(Spectrometer2Beams, self).configure(input_ip, INPUT_2BEAMS, length, input_freq)

# nvprof -f --profile-child-processes -o profile.nvprof%p ./pipeline.py -a 0 -b 2 -c search -d 1 -e 10
# cuda-memcheck ./pipeline.py -a 0 -b 2 -c search -d 1 -e 10
if __name__ == "__main__":
    logging.getLogger().addHandler(logging.NullHandler())
    log = logging.getLogger("mpikat")
    coloredlogs.install(
        fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level="DEBUG",
        logger=log)

    host_id = check_output("hostname").strip()[-1]

    parser = argparse.ArgumentParser(
        description='To run the pipeline for my test')
    parser.add_argument('-a', '--numa', type=int, nargs='+',
                        help='The ID of numa node')
    parser.add_argument('-b', '--beam', type=int, nargs='+',
                        help='The number of beams')
    parser.add_argument('-c', '--pipeline', type=str, nargs='+',
                        help='The pipeline to run')
    parser.add_argument('-d', '--nconfigure', type=int, nargs='+',
                        help='How many times to repeat the configure')
    parser.add_argument('-e', '--length', type=int, nargs='+',
                        help='How many seconds to run')

    args = parser.parse_args()
    numa = args.numa[0]
    beam = args.beam[0]
    pipeline = args.pipeline[0]    
    nconfigure = args.nconfigure[0]
    length = args.length[0]
    
    ip = "10.17.{}.{}".format(host_id, numa + 1)

    if beam == 1:
        freq = 1340.5
    if beam == 2:
        freq = 1337.0

    if pipeline == "fold":
        for i in range(nconfigure):
            log.info("Create pipeline ...")
            if beam == 1:
                fold_mode = Fold1Beam()
            if beam == 2:
                fold_mode = Fold2Beams()

            log.info("Configure it ...")
            fold_mode.configure(ip, length, freq)
        
            log.info("Start it ...")
            fold_mode.start()
        
            log.info("Stop it ...")
            fold_mode.stop()
        
            log.info("Deconfigure it ...")
            fold_mode.deconfigure()
        
    if pipeline == "search":
        for i in range(nconfigure):
            log.info("Create pipeline ...")
            if beam == 1:
                search_mode = Search1Beam()
            if beam == 2:
                search_mode = Search2Beams()

            log.info("Configure it ...")
            search_mode.configure(ip, length, freq)
        
            log.info("Start it ...")
            search_mode.start()
        
            log.info("Stop it ...")
            search_mode.stop()
        
            log.info("Deconfigure it ...")
            search_mode.deconfigure()
        
    if pipeline == "spectrometer":
        for i in range(nconfigure):
            log.info("Create pipeline ...")
            if beam == 1:
                spectrometer_mode = Spectrometer1Beam()
            if beam == 2:
                spectrometer_mode = Spectrometer2Beams()
    
            log.info("Configure it ...")
            spectrometer_mode.configure(ip, length, freq)
        
            log.info("Start it ...")
            spectrometer_mode.start()
            log.info("Stop it ...")
            spectrometer_mode.stop()
        
            log.info("Deconfigure it ...")
            spectrometer_mode.deconfigure()
