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

EXECUTE = True
#EXECUTE        = False

FITSWRITER = True
#FITSWRITER = False

SOD = True   # Start filterbank data
# SOD  = False  # Do not start filterbank data

HEIMDALL = False   # To run heimdall on filterbank file or not
#HEIMDALL       = True   # To run heimdall on filterbank file or not

DBDISK = True   # To run dbdisk on processed data or not
#DBDISK         = False   # To run dbdisk on processed data or not

PAF_ROOT       = "/home/pulsar/xinping/phased-array-feed/"
DATA_ROOT      = "/beegfs/DENG/"
DADA_ROOT      = "{}/AUG/baseband/".format(DATA_ROOT)
SOURCE         = "J1819-1458"

PAF_CONFIG = {"instrument_name":    "PAF-BMF",
              "nchan_chk":    	     7,        # MHz
              "over_samp_rate":      (32.0/27.0),
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
              "mem_node":            60791751475, # has 10% spare
              "first_port":          17100,
}

SEARCH_CONFIG_GENERAL = {"rbuf_baseband_ndf_chk":   16384,                 
                         "rbuf_baseband_nblk":      5,
                         "rbuf_baseband_nread":     1,                 
                         "tbuf_baseband_ndf_chk":   128,

                         "rbuf_filterbank_ndf_chk": 16384,
                         "rbuf_filterbank_nblk":    2,
                         "rbuf_filterbank_nread":   (HEIMDALL + DBDISK) if (HEIMDALL + DBDISK) else 1,

                         "nchan_filterbank":        512,
                         #"cufft_nx":                128,
                         "cufft_nx":                256,

                         "nbyte_filterbank":        1,
                         "npol_samp_filterbank":    1,
                         "ndim_pol_filterbank":     1,

                         "ndf_stream":      	    1024,
                         "nstream":                 2,

                         "bind":                    1,

                         "pad":                     0,
                         "ndf_check_chk":           1024,
                         
                         "ip":                      '134.104.70.90',
                         "port":                    17106,
                         
                         "detect_thresh":           10,
                         "dm":                      [1, 10000],
                         "zap_chans":               [],
                         "ptype":                   2,
}

SEARCH_CONFIG_1BEAM = {"dada_fname":             "{}/{}/{}_48chunks.dada".format(DADA_ROOT, SOURCE, SOURCE),
                       "rbuf_baseband_key":      ["dada"],
                       "rbuf_filterbank_key":    ["dade"],
                       #"nchan_keep_band":        32768,
                       "nchan_keep_band":        72192,
                       "nbeam":                  1,
                       "nport_beam":             3,
                       "nchk_port":              16,
}

SEARCH_CONFIG_2BEAMS = {"dada_fname":              "{}/{}/{}_33chunks.dada".format(DADA_ROOT, SOURCE, SOURCE),
                        "rbuf_baseband_key":       ["dada", "dadc"],
                        "rbuf_filterbank_key":     ["dade", "dadg"],
                        #"nchan_keep_band":         24576,
                        "nchan_keep_band":         49664,
                        "nbeam":                   2,
                        "nport_beam":              3,
                        "nchk_port":               11,
}

SPECTRAL_CONFIG_GENERAL = {"rbuf_baseband_ndf_chk":   16384,                 
                           "rbuf_baseband_nblk":      5,
                           "rbuf_baseband_nread":     1,                 
                           "tbuf_baseband_ndf_chk":   128,
                           "nblk_accumulate":         2,
                           "rbuf_spectral_ndf_chk":   16384,
                           "rbuf_spectral_nblk":      2,
                           "rbuf_spectral_nread":     1,
                           
                           "cufft_nx":                1024,
                           "nbyte_spectral":          4,
                           "ndf_stream":      	      1024,
                           "nstream":                 2,
                           
                           "bind":                    1,
                           "pad":                     0,
                           "ndf_check_chk":           1024,
                           "ip":                      '134.104.70.90',
                           "port":                    17106,
}

SPECTRAL_CONFIG_1BEAM = {"dada_fname":             "{}/{}/{}_48chunks.dada".format(DADA_ROOT, SOURCE, SOURCE),
                         "rbuf_baseband_key":      ["dada"],
                         "rbuf_spectral_key":      ["dade"],
                         "nbeam":                  1,
                         "nport_beam":             3,
                         "nchk_port":              16,
}

SPECTRAL_CONFIG_2BEAMS = {"dada_fname":              "{}/{}/{}_33chunks.dada".format(DADA_ROOT, SOURCE, SOURCE),
                          "rbuf_baseband_key":       ["dada", "dadc"],
                          "rbuf_spectral_key":       ["dade", "dadg"],
                          "nbeam":                   2,
                          "nport_beam":              3,
                          "nchk_port":               11,
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
    def __init__(self, command, process_index = None):
        self._command = command
        self.stdout_callbacks = set()
        self.stderr_callbacks = set()
        self.returncode_callbacks = set()
        self._monitor_threads = []
        self._process = None
        self._executable_command = None
        self._stdout = None
        self._stderr = None
        self._returncode = None
        self._process_index = process_index
        
        log.debug(self._command)
        self._executable_command = shlex.split(self._command)

        if EXECUTE:
            try:
                self._process = Popen(self._executable_command,
                                      stdout=PIPE,
                                      stderr=PIPE,
                                      bufsize=1,
                                      universal_newlines=True)
            except Exception as error:
                log.exception(error)
                self.returncode = self._command + "; RETURNCODE is: ' 1'"
            if self._process == None:
                self.returncode = self._command + "; RETURNCODE is: ' 1'"

            # Start monitors
            self._monitor_threads.append(threading.Thread(target=self._stdout_monitor))
            self._monitor_threads.append(threading.Thread(target=self._stderr_monitor))
            
            for thread in self._monitor_threads:
                thread.daemon = True
                thread.start()
                                    
    def __del__(self):
        class_name = self.__class__.__name__

    def finish(self):
        if EXECUTE:
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

    def stderr_notify(self):
        for callback in self.stderr_callbacks:
            callback(self._stderr, self)

    @property
    def stderr(self):
        return self._stderr

    @stderr.setter
    def stderr(self, value):
        self._stderr = value
        self.stderr_notify()

    
    def _stdout_monitor(self):
        if EXECUTE:
            while self._process.poll() == None:
                stdout = self._process.stdout.readline().rstrip("\n\r")
                if stdout != b"":
                    if self._process_index != None:
                        self.stdout = stdout + "; PROCESS_INDEX is " + str(self._process_index)
                    else:
                        self.stdout = stdout
                    #log.debug(self._command + " " + self.stdout)
                    log.debug(stdout)
            if self._process.returncode and self._process.stderr.readline().rstrip("\n\r") == b"":
                self.returncode = self._command + "; RETURNCODE is: " + str(self._process.returncode)
            
    def _stderr_monitor(self):
        if EXECUTE:
            while self._process.poll() == None:
                stderr = self._process.stderr.readline().rstrip("\n\r")
                if stderr != b"":
                    if self._process_index != None:
                        self.stderr = self._command + "; STDERR is: " + stderr + "; PROCESS_INDEX is " + str(self._process_index)
                    else:
                        self.stderr = self._command + "; STDERR is: " + stderr
                    log.debug(stderr)
                    
class Pipeline(object):
    def __init__(self):
        self._prd = PAF_CONFIG["prd"]
        self._first_port = PAF_CONFIG["first_port"]
        self._df_res = PAF_CONFIG["df_res"]
        self._df_dtsz = PAF_CONFIG["df_dtsz"]
        self._df_pktsz = PAF_CONFIG["df_pktsz"]
        self._df_hdrsz = PAF_CONFIG["df_hdrsz"]
        self._ncpu_numa = PAF_CONFIG["ncpu_numa"]
        self._nchan_chk = PAF_CONFIG["nchan_chk"]
        self._nbyte_baseband = PAF_CONFIG["nbyte_baseband"]
        self._ndim_pol_baseband = PAF_CONFIG["ndim_pol_baseband"]
        self._npol_samp_baseband = PAF_CONFIG["npol_samp_baseband"]
        self._mem_node           = PAF_CONFIG["mem_node"]
        self._over_samp_rate     = PAF_CONFIG["over_samp_rate"]
        self._cleanup_commands   = []
        
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
        if EXECUTE:
            log.debug(stdout)

    def _handle_execution_returncode(self, returncode, callback):
        if EXECUTE:
            log.debug(returncode)
            if returncode:
                raise PipelineError(returncode)

    def _handle_execution_stderr(self, stderr, callback):
        if EXECUTE:
            log.error(stderr)
            #raise PipelineError(stderr)

    def cleanup(self):
        # Kill existing process and free shared memory if there is any
        execution_instances = []
        for command in self._cleanup_commands:
            execution_instances.append(ExecuteCommand(command))            
        for execution_instance in execution_instances:         # Wait until the cleanup is done
            execution_instance.finish()

@register_pipeline("Search")
class Search(Pipeline):
    def __init__(self):
        super(Search, self).__init__()
        self._runtime_directory = []

        self._dbdisk_commands = []
        self._diskdb_commands = []
        self._heimdall_commands = []
        self._baseband2filterbank_commands = []
        self._baseband_create_buffer_commands = []
        self._baseband_delete_buffer_commands = []
        self._filterbank_create_buffer_commands = []
        self._filterbank_delete_buffer_commands = []

        self._dbdisk_execution_instances = []
        self._diskdb_execution_instances = []
        self._baseband2filterbank_execution_instances = []
        self._heimdall_execution_instances = []

        self._dm = SEARCH_CONFIG_GENERAL["dm"]
        self._pad = SEARCH_CONFIG_GENERAL["pad"]
        self._ip_udp = SEARCH_CONFIG_GENERAL["ip"]
        self._bind = SEARCH_CONFIG_GENERAL["bind"]
        self._ptype = SEARCH_CONFIG_GENERAL["ptype"]
        self._port_udp = SEARCH_CONFIG_GENERAL["port"]
        self._nstream = SEARCH_CONFIG_GENERAL["nstream"]
        self._cufft_nx = SEARCH_CONFIG_GENERAL["cufft_nx"]
        self._zap_chans = SEARCH_CONFIG_GENERAL["zap_chans"]
        self._ndf_stream = SEARCH_CONFIG_GENERAL["ndf_stream"]
        self._detect_thresh = SEARCH_CONFIG_GENERAL["detect_thresh"]
        self._ndf_check_chk = SEARCH_CONFIG_GENERAL["ndf_check_chk"]
        self._nchan_filterbank = SEARCH_CONFIG_GENERAL["nchan_filterbank"]
        self._nbyte_filterbank = SEARCH_CONFIG_GENERAL["nbyte_filterbank"]
        self._rbuf_baseband_nblk = SEARCH_CONFIG_GENERAL["rbuf_baseband_nblk"]
        self._ndim_pol_filterbank = SEARCH_CONFIG_GENERAL["ndim_pol_filterbank"]
        self._rbuf_baseband_nread = SEARCH_CONFIG_GENERAL["rbuf_baseband_nread"]
        self._npol_samp_filterbank = SEARCH_CONFIG_GENERAL["npol_samp_filterbank"]
        self._rbuf_filterbank_nblk = SEARCH_CONFIG_GENERAL["rbuf_filterbank_nblk"]
        self._rbuf_baseband_ndf_chk = SEARCH_CONFIG_GENERAL["rbuf_baseband_ndf_chk"]
        self._rbuf_filterbank_nread = SEARCH_CONFIG_GENERAL["rbuf_filterbank_nread"]
        self._tbuf_baseband_ndf_chk = SEARCH_CONFIG_GENERAL["tbuf_baseband_ndf_chk"]
        self._rbuf_filterbank_ndf_chk = SEARCH_CONFIG_GENERAL["rbuf_filterbank_ndf_chk"]

        self._cleanup_commands = ["pkill -9 -f dada_diskdb",
                                  "pkill -9 -f baseband2filter", # process name, maximum 16 bytes (15 bytes visiable)
                                  "pkill -9 -f heimdall",
                                  "pkill -9 -f dada_dbdisk",
                                  "pkill -9 -f dada_db",
                                  "ipcrm -a"]

    def configure(self, ip, pipeline_config):
        # Setup parameters of the pipeline
        self._ip = ip
        self._pipeline_config = pipeline_config
        self._numa = int(ip.split(".")[3]) - 1
        self._server = int(ip.split(".")[2])
        
        self._nbeam = self._pipeline_config["nbeam"]
        self._nchk_port = self._pipeline_config["nchk_port"]
        self._dada_fname = self._pipeline_config["dada_fname"]
        self._nport_beam = self._pipeline_config["nport_beam"]
        self._nchan_keep_band = self._pipeline_config["nchan_keep_band"]
        self._rbuf_baseband_key = self._pipeline_config["rbuf_baseband_key"]
        self._rbuf_filterbank_key = self._pipeline_config["rbuf_filterbank_key"]
        self._blk_res = self._df_res * self._rbuf_baseband_ndf_chk
        self._nchk_beam = self._nchk_port * self._nport_beam
        self._nchan_baseband = self._nchan_chk * self._nchk_beam
        self._ncpu_pipeline = self._ncpu_numa / self._nbeam
        self._rbuf_baseband_blksz = self._nchk_port * \
            self._nport_beam * self._df_dtsz * self._rbuf_baseband_ndf_chk
        self._rbuf_filterbank_blksz = int(self._nchan_filterbank * self._rbuf_baseband_blksz *
                                          self._nbyte_filterbank * self._npol_samp_filterbank *
                                          self._ndim_pol_filterbank / float(self._nbyte_baseband *
                                                                            self._npol_samp_baseband *
                                                                            self._ndim_pol_baseband *
                                                                            self._nchan_baseband *
                                                                            self._cufft_nx))
        
        # To see if we can process baseband data with integer repeats
        if self._rbuf_baseband_ndf_chk % (self._ndf_stream * self._nstream):
            raise PipelineError("data in baseband ring buffer block can only "
                                "be processed by baseband2filterbank with integer repeats")

        # To see if we have enough memory
        if self._nbeam*(self._rbuf_filterbank_blksz + self._rbuf_baseband_blksz) > self._mem_node:
            raise PipelineError("We do not have enough shared memory for the setup "
                                "Try to reduce the ring buffer block number "
                                "or reduce the number of packets in each ring buffer block")
        
        # To be safe, kill all related softwares and free shared memory
        self.cleanup()
                
        # To setup commands for each process
        baseband2filterbank = "{}/src/baseband2filterbank_main".format(PAF_ROOT)
        if not os.path.isfile(baseband2filterbank):
            raise PipelineError("{} is not exist".format(baseband2filterbank))
        if not os.path.isfile(self._dada_fname):
            raise PipelineError("{} is not exist".format(self._dada_fname))                
        for i in range(self._nbeam):      
            if EXECUTE:
                # To get directory for runtime information
                runtime_directory = "{}/pacifix{}_numa{}_process{}".format(DATA_ROOT, self._server, self._numa, i)
                if not os.path.isdir(runtime_directory):
                    try:
                        os.makedirs(directory)
                    except Exception as error:
                        log.exception(error)
                        raise PipelineError("Fail to create {}".format(runtime_directory))
            else:
                runtime_directory = None                
            self._runtime_directory.append(runtime_directory)

            # diskdb command
            diskdb_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline                                      
            self._diskdb_commands.append("dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(self._rbuf_baseband_key[i], self._dada_fname, 0))

            # baseband2filterbank command
            baseband2filterbank_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 1
            command = ("{} -a {} -b {} -c {} -d {} -e {} "
                       "-f {} -i {} -j {} -k {} -l {} "
                       "-m {} -n {}_{} ").format(baseband2filterbank, self._rbuf_baseband_key[i],
                                                 self._rbuf_filterbank_key[i], self._rbuf_filterbank_ndf_chk, self._nstream,
                                                 self._ndf_stream, self._runtime_directory[i], self._nchk_beam, self._cufft_nx,
                                                 self._nchan_filterbank, self._nchan_keep_band, self._ptype, self._ip_udp, self._port_udp)
            if SOD:
                command += "-g 1"
            else:
                command += "-g 0"
            self._baseband2filterbank_commands.append(command)

            # Command to create filterbank ring buffer
            dadadb_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 2
            self._filterbank_create_buffer_commands.append(("dada_db -l -p -k {:} "
                                                            "-b {:} -n {:} -r {:}").format(self._rbuf_filterbank_key[i],
                                                                                           self._rbuf_filterbank_blksz,
                                                                                           self._rbuf_filterbank_nblk,
                                                                                           self._rbuf_filterbank_nread))

            # command to create baseband ring buffer
            self._baseband_create_buffer_commands.append(("dada_db -l -p -k {:} "
                                                          "-b {:} -n {:} -r {:}").format(self._rbuf_baseband_key[i],
                                                                                         self._rbuf_baseband_blksz,
                                                                                         self._rbuf_baseband_nblk,
                                                                                         self._rbuf_baseband_nread))

            # command to delete filterbank ring buffer
            self._filterbank_delete_buffer_commands.append(
                "dada_db -d -k {:}".format(self._rbuf_filterbank_key[i]))

            # command to delete baseband ring buffer
            self._baseband_delete_buffer_commands.append(
                "dada_db -d -k {:}".format(self._rbuf_baseband_key[i]))

            # Command to run heimdall
            heimdall_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 3
            command = ("heimdall -k {} -detect_thresh {} -output_dir {} ").format(
                self._rbuf_filterbank_key[i],
                self._detect_thresh, runtime_directory)
            if self._zap_chans:
                zap = ""
                for zap_chan in self._zap_chans:
                    zap += " -zap_chans {} {}".format(
                        self._zap_chan[0], self._zap_chan[1])
                command += zap
                if self._dm:
                    command += "-dm {} {}".format(self._dm[0], self._dm[1])
            self._heimdall_commands.append(command)

            # Command to run dbdisk
            dbdisk_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 4
            command = "dada_dbdisk -b {} -k {} -D {} -o -s -z".format(
                dbdisk_cpu, self._rbuf_filterbank_key[i], self._runtime_directory[i])
            self._dbdisk_commands.append(command)

    def start(self):
        # Create baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._baseband_create_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()

        # Create ring buffer for filterbank data
        process_index = 0
        execution_instances = []
        for command in self._filterbank_create_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:         # Wait until the buffer creation is done
            execution_instance.finish()

        # Execute the diskdb
        process_index = 0
        self._diskdb_execution_instances = []
        for command in self._diskdb_commands:
            execution_instance = ExecuteCommand(command, process_index)
            #self._execution_instances.append(execution_instance)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._dbdisk_execution_instances.append(execution_instance)
            process_index += 1
            
        # Run baseband2filterbank
        process_index = 0
        self._baseband2filterbank_execution_instances = []
        for command in self._baseband2filterbank_commands:
            execution_instance = ExecuteCommand(command, process_index)
            execution_instance.stderr_callbacks.add(self._handle_execution_stderr)
            #execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
            #execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._baseband2filterbank_execution_instances.append(execution_instance)
            process_index += 1 

        if HEIMDALL:  # run heimdall if required
            process_index = 0
            self._heimdall_execution_instances = []
            for command in self._heimdall_commands:
                execution_instance = ExecuteCommand(command, process_index)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                self._heimdall_execution_instances.append(execution_instance)
                process_index += 1
                
        if DBDISK:   # Run dbdisk if required
            process_index = 0
            self._dbdisk_execution_instances = []
            for command in self._dbdisk_commands:
                execution_instance = ExecuteCommand(command, process_index)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                self._dbdisk_execution_instances.append(execution_instance)
                process_index += 1

    def stop(self):
        if DBDISK:
            for execution_instance in self._dbdisk_execution_instances:
                execution_instance.finish()
        if HEIMDALL:
            for execution_instance in self._heimdall_execution_instances:
                execution_instance.finish()
                
        for execution_instance in self._baseband2filterbank_execution_instances:
            execution_instance.finish()

        for execution_instance in self._diskdb_execution_instances:
            execution_instance.finish()
            
        # To delete filterbank ring buffer
        process_index = 0
        execution_instances = []
        for command in self._filterbank_delete_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
        
        # To delete baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._baseband_delete_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
            
    def deconfigure(self):
        pass
        
@register_pipeline("Search2Beams")
class Search2Beams(Search):

    def __init__(self):
        super(Search2Beams, self).__init__()

    def configure(self, ip):
        super(Search2Beams, self).configure(ip, SEARCH_CONFIG_2BEAMS)

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

    def configure(self, ip):
        super(Search1Beam, self).configure(ip, SEARCH_CONFIG_1BEAM)

    def start(self):
        super(Search1Beam, self).start()

    def stop(self):
        super(Search1Beam, self).stop()

    def deconfigure(self):
        super(Search1Beam, self).deconfigure()

@register_pipeline("Spectral")
class Spectral(Pipeline):
    def __init__(self):
        super(Spectral, self).__init__()
        self._runtime_directory = []

        self._dbdisk_commands = []
        self._diskdb_commands = []
        self._baseband2spectral_commands = []
        self._baseband_create_buffer_commands = []
        self._baseband_delete_buffer_commands = []
        self._spectral_create_buffer_commands = []
        self._spectral_delete_buffer_commands = []

        self._dbdisk_execution_instances = []
        self._diskdb_execution_instances = []
        self._baseband2spectral_execution_instances = []

        self._ip_out   = SPECTRAL_CONFIG_GENERAL["ip"]
        self._port_out = SPECTRAL_CONFIG_GENERAL["port"]
        self._pad  = SPECTRAL_CONFIG_GENERAL["pad"]
        self._bind = SPECTRAL_CONFIG_GENERAL["bind"]
        self._nstream = SPECTRAL_CONFIG_GENERAL["nstream"]
        self._cufft_nx = SPECTRAL_CONFIG_GENERAL["cufft_nx"]
        self._ndf_stream = SPECTRAL_CONFIG_GENERAL["ndf_stream"]
        self._ndf_check_chk = SPECTRAL_CONFIG_GENERAL["ndf_check_chk"]
        self._nbyte_spectral = SPECTRAL_CONFIG_GENERAL["nbyte_spectral"]
        self._nblk_accumulate = SPECTRAL_CONFIG_GENERAL["nblk_accumulate"]
        self._rbuf_baseband_nblk = SPECTRAL_CONFIG_GENERAL["rbuf_baseband_nblk"]
        self._rbuf_baseband_nread = SPECTRAL_CONFIG_GENERAL["rbuf_baseband_nread"]
        self._rbuf_spectral_nblk = SPECTRAL_CONFIG_GENERAL["rbuf_spectral_nblk"]
        self._rbuf_baseband_ndf_chk = SPECTRAL_CONFIG_GENERAL["rbuf_baseband_ndf_chk"]
        self._rbuf_spectral_nread = SPECTRAL_CONFIG_GENERAL["rbuf_spectral_nread"]
        self._tbuf_baseband_ndf_chk = SPECTRAL_CONFIG_GENERAL["tbuf_baseband_ndf_chk"]
        self._rbuf_spectral_ndf_chk = SPECTRAL_CONFIG_GENERAL["rbuf_spectral_ndf_chk"]

        if not (self._ndf_stream % self._cufft_nx == 0):
            log.error("ndf_stream should be multiple times of cufft_nx")
            raise PipelineError("ndf_stream should be multiple times of cufft_nx")
        
        self._cleanup_commands = ["pkill -9 -f dada_diskdb",
                                  "pkill -9 -f baseband2spectr", # process name, maximum 16 bytes (15 bytes visiable)
                                  "pkill -9 -f dada_dbdisk",
                                  "pkill -9 -f dada_db",
                                  "ipcrm -a"]

    def configure(self, ip, ptype, pipeline_config):
        # Setup parameters of the pipeline
        self._ip = ip
        self._pipeline_config = pipeline_config
        self._numa = int(ip.split(".")[3]) - 1
        self._server = int(ip.split(".")[2])
        self._ptype = ptype

        if self._ptype not in [1, 2, 4]:  # We can only have three possibilities
            log.error("Pol type should be 1, 2 or 4, but it is {}".format(self._ptype))
            raise PipelineError("Pol type should be 1, 2 or 4, but it is {}".format(self._ptype))
        
        self._nbeam = self._pipeline_config["nbeam"]
        self._nchk_port = self._pipeline_config["nchk_port"]
        self._dada_fname = self._pipeline_config["dada_fname"]
        self._nport_beam = self._pipeline_config["nport_beam"]
        self._rbuf_baseband_key = self._pipeline_config["rbuf_baseband_key"]
        self._rbuf_spectral_key = self._pipeline_config["rbuf_spectral_key"]
        self._blk_res = self._df_res * \
                        self._rbuf_baseband_ndf_chk
        self._nchk_beam = self._nchk_port * \
                          self._nport_beam
        self._nchan_baseband = self._nchan_chk * \
                               self._nchk_beam
        self._ncpu_pipeline = self._ncpu_numa / self._nbeam
        self._rbuf_baseband_blksz = self._nchk_port * \
                                    self._nport_beam * \
                                    self._df_dtsz * \
                                    self._rbuf_baseband_ndf_chk
        
        self._rbuf_spectral_blksz = int(4 * self._nchan_baseband * # Replace 4 with true pol numbers if we do not pad 0
                                        self._cufft_nx /
                                        self._over_samp_rate *
                                        self._nbyte_spectral)
        
        # To see if we can process baseband data with integer repeats
        if self._rbuf_baseband_ndf_chk % (self._ndf_stream * self._nstream):
            raise PipelineError("data in baseband ring buffer block can only "
                                "be processed by baseband2spectral with integer repeats")

        # To see if we have enough memory
        if self._nbeam*(self._rbuf_spectral_blksz + self._rbuf_baseband_blksz) > self._mem_node:
            raise PipelineError("We do not have enough shared memory for the setup "
                                "Try to reduce the ring buffer block number "
                                "or reduce the number of packets in each ring buffer block")
        
        # To be safe, kill all related softwares and free shared memory
        self.cleanup()
                
        # To setup commands for each process
        baseband2spectral = "{}/src/baseband2spectral_main".format(PAF_ROOT)
        if not os.path.isfile(baseband2spectral):
            raise PipelineError("{} is not exist".format(baseband2spectral))
        if not os.path.isfile(self._dada_fname):
            raise PipelineError("{} is not exist".format(self._dada_fname))                
        for i in range(self._nbeam):      
            if EXECUTE:
                # To get directory for runtime information
                runtime_directory = "{}/pacifix{}_numa{}_process{}".format(DATA_ROOT, self._server, self._numa, i)
                if not os.path.isdir(runtime_directory):
                    try:
                        os.makedirs(directory)
                    except Exception as error:
                        log.exception(error)
                        raise PipelineError("Fail to create {}".format(runtime_directory))
            else:
                runtime_directory = None                
            self._runtime_directory.append(runtime_directory)

            # diskdb command
            diskdb_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline                                      
            self._diskdb_commands.append("dada_diskdb -k {:s} -f {:s} -o {:d} -s".format(
                self._rbuf_baseband_key[i], self._dada_fname, 0))

            # baseband2spectral command
            baseband2spectral_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 1
            command = "{} -a {} -c {} -d {} -e {} -f {} -g {} -i {} -j {} -k {} ".format(
                baseband2spectral,
                self._rbuf_baseband_key[i],self._rbuf_spectral_ndf_chk,
                self._nstream, self._ndf_stream,
                self._runtime_directory[i], self._nchk_beam,
                self._cufft_nx, self._ptype, self._nblk_accumulate)
            if not FITSWRITER:
                if SOD:
                    command += "-b k_{}_1".format(self._rbuf_spectral_key[i])
                else:
                    command += "-b k_{}_0".format(self._rbuf_spectral_key[i])
            else:
                command += "-b n_{}_{}".format(self._ip_out, self._port_out)
            self._baseband2spectral_commands.append(command)

            # Command to create spectral ring buffer
            dadadb_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 2
            if not FITSWRITER:
                self._spectral_create_buffer_commands.append(("dada_db -l -p -k {:} "
                                                              "-b {:} -n {:} -r {:}").format(
                                                                  dadadb_cpu, self._rbuf_spectral_key[i],
                                                                  self._rbuf_spectral_blksz,
                                                                  self._rbuf_spectral_nblk,
                                                                  self._rbuf_spectral_nread))

            # command to create baseband ring buffer
            self._baseband_create_buffer_commands.append(("dada_db -l -p -k {:} "
                                                          "-b {:} -n {:} -r {:}").format(
                                                              self._rbuf_baseband_key[i],
                                                              self._rbuf_baseband_blksz,
                                                              self._rbuf_baseband_nblk,
                                                              self._rbuf_baseband_nread))
            
            # command to delete spectral ring buffer
            if not FITSWRITER:
                self._spectral_delete_buffer_commands.append(
                    "dada_db -d -k {:}".format(
                        self._rbuf_spectral_key[i]))

            # command to delete baseband ring buffer
            self._baseband_delete_buffer_commands.append(
                "dada_db -d -k {:}".format(
                    self._rbuf_baseband_key[i]))

            # Command to run dbdisk
            dbdisk_cpu = self._numa * self._ncpu_numa + i * self._ncpu_pipeline + 4
            command = "dada_dbdisk -b {} -k {} -D {} -o -s -z".format(
                dbdisk_cpu, self._rbuf_spectral_key[i], self._runtime_directory[i])
            self._dbdisk_commands.append(command)

    def start(self):
        # Create baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._baseband_create_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()

        # Create ring buffer for spectral data
        process_index = 0
        execution_instances = []
        for command in self._spectral_create_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:         # Wait until the buffer creation is done
            execution_instance.finish()
        
        # Execute the diskdb
        process_index = 0
        self._diskdb_execution_instances = []
        for command in self._diskdb_commands:
            execution_instance = ExecuteCommand(command, process_index)
            #self._execution_instances.append(execution_instance)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._dbdisk_execution_instances.append(execution_instance)
            process_index += 1
            
        # Run baseband2spectral
        process_index = 0
        self._baseband2spectral_execution_instances = []
        for command in self._baseband2spectral_commands:
            execution_instance = ExecuteCommand(command, process_index)
            execution_instance.stderr_callbacks.add(self._handle_execution_stderr)
            #execution_instance.returncode_callbacks.add(self._handle_execution_returncode)
            execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
            self._baseband2spectral_execution_instances.append(execution_instance)
            process_index += 1 
                
        if not FITSWRITER:   # Run dbdisk if required
            process_index = 0
            self._dbdisk_execution_instances = []
            for command in self._dbdisk_commands:
                execution_instance = ExecuteCommand(command, process_index)
                execution_instance.stdout_callbacks.add(self._handle_execution_stdout)
                execution_instance.returncode_callbacks.add(
                    self._handle_execution_returncode)
                self._dbdisk_execution_instances.append(execution_instance)
                process_index += 1

    def stop(self):
        if not FITSWRITER:
            for execution_instance in self._dbdisk_execution_instances:
                execution_instance.finish()

        for execution_instance in self._baseband2spectral_execution_instances:
            execution_instance.finish()

        for execution_instance in self._diskdb_execution_instances:
            execution_instance.finish()
            
        # To delete spectral ring buffer
        process_index = 0
        execution_instances = []
        for command in self._spectral_delete_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
        
        # To delete baseband ring buffer
        process_index = 0
        execution_instances = []
        for command in self._baseband_delete_buffer_commands:
            execution_instances.append(ExecuteCommand(command, process_index))
            process_index += 1
        for execution_instance in execution_instances:
            execution_instance.finish()
            
    def deconfigure(self):
        pass


@register_pipeline("Spectral1Beam1Pol")
class Spectral1Beam1Pol(Spectral):
    def configure(self, ip):
        super(Spectral1Beam1Pol, self).configure(ip, 1, SPECTRAL_CONFIG_1BEAM)

@register_pipeline("Spectral2Beams1Pol")
class Spectral2Beams1Pol(Spectral):
    def configure(self, ip):
        super(Spectral2Beams1Pol, self).configure(ip, 1, SPECTRAL_CONFIG_2BEAMS)

@register_pipeline("Spectral1Beam2Pols")
class Spectral1Beam2Pols(Spectral):
    def configure(self, ip):
        super(Spectral1Beam2Pols, self).configure(ip, 2, SPECTRAL_CONFIG_1BEAM)

@register_pipeline("Spectral2Beams2Pols")
class Spectral2Beams2Pols(Spectral):
    def configure(self, ip):
        super(Spectral2Beams2Pols, self).configure(ip, 2, SPECTRAL_CONFIG_2BEAMS)

@register_pipeline("Spectral1Beam4Pols")
class Spectral1Beam4Pols(Spectral):
    def configure(self, ip):
        super(Spectral1Beam4Pols, self).configure(ip, 4, SPECTRAL_CONFIG_1BEAM)

@register_pipeline("Spectral2Beams4Pols")
class Spectral2Beams4Pols(Spectral):
    def configure(self, ip):
        super(Spectral2Beams4Pols, self).configure(ip, 4, SPECTRAL_CONFIG_2BEAMS)

# nvprof -f --profile-child-processes -o profile.nvprof%p ./pipeline.py -a 0 -b 2 -c search -d 1
# cuda-memcheck ./pipeline.py -a 0 -b 2 -c search -d 1
if __name__ == "__main__":
    logging.getLogger().addHandler(logging.NullHandler())
    log = logging.getLogger('mpikat')
    coloredlogs.install(
        fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level='DEBUG',
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

    args = parser.parse_args()
    numa = args.numa[0]
    beam = args.beam[0]
    pipeline = args.pipeline[0]    
    nconfigure = args.nconfigure[0]
    
    ip = "10.17.{}.{}".format(host_id, numa + 1)

    if beam == 1:
        freq = 1340.5
    if beam == 2:
        freq = 1337.0

    if pipeline == "search":
        for i in range(nconfigure):
            log.info("Create pipeline ...")
            if beam == 1:
                search_mode = Search1Beam()
            if beam == 2:
                search_mode = Search2Beams()

            log.info("Configure it ...")
            search_mode.configure(ip)
        
            log.info("Start it ...")
            search_mode.start()
        
            log.info("Stop it ...")
            search_mode.stop()
        
            log.info("Deconfigure it ...")
            search_mode.deconfigure()
        
    if pipeline == "spectral":
        for i in range(nconfigure):
            log.info("Create pipeline ...")
            if beam == 1:
                #spectral_mode = Spectral1Beam1Pol()
                spectral_mode = Spectral1Beam2Pols()
                #spectral_mode = Spectral1Beam4Pols()
            if beam == 2:
                #spectral_mode = Spectral2Beams1Pol()
                spectral_mode = Spectral2Beams2Pols()
                #spectral_mode = Spectral2Beams4Pols()
    
            log.info("Configure it ...")
            spectral_mode.configure(ip)
        
            log.info("Start it ...")
            spectral_mode.start()
            log.info("Stop it ...")
            spectral_mode.stop()
        
            log.info("Deconfigure it ...")
            spectral_mode.deconfigure()
