#!/usr/bin/env python
import os
import sys
import signal
import tornado
from tornado.gen import coroutine
from katcp import Sensor, AsyncDeviceServer, AsyncReply
from katcp.kattypes import request, return_reply, Float, Str, Int, Bool
from optparse import OptionParser
from subprocess import check_output

pacifix     = ['pacifix0', 'pacifix1', 'pacifix3', 'pacifix4', 'pacifix5', 'pacifix6', 'pacifix7', 'pacifix8']
switch      = ['switch4', 'switch5']
port        = ['17100', '17101', '17102', '17103', '17104', '17105']
execute_dir = "/home/pulsar/xinping/phased-array-feed/script"

class PafBackendController(AsyncDeviceServer):
    VERSION_INFO = ("paf-backend-controller-api", 0, 1)
    BUILD_INFO = ("paf-backend-controller-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port):
        super(PafBackendController, self).__init__(ip, port)

    def setup_sensors(self):
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of PafBackendController",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        
        self.add_sensor(self._device_status)

    def start(self):
        super(PafBackendController, self).start()

    @request(Int(), Float(), Bool())
    @return_reply(Int(), Float())
    def request_myreq(self, req, my_int, my_float, my_bool):
        '''?myreq my_int my_float my_bool'''
        return ("ok", my_int + 1, my_float / 2.0)

    @request(Str())
    @return_reply(Str())
    def request_uptime(self, req, machine):
        """
        @brief      ssh to a machine and run "uptime" there
        """
        self.machine = machine
        
        if self.machine == 'pacifix':      # For all pacifix nodes
            self.out = check_output("dsh -g {:s} uptime".format(self.machine), shell=True)
        elif self.machine == 'switch':     # For all switches
            self.out = check_output("dsh -g {:s} uptime".format(self.machine), shell=True)
        else:                              # For each pacifix node and switch
            if self.machine in pacifix:
                self.out = check_output("ssh pulsar@{:s} uptime".format(self.machine), shell=True)
            elif self.machine == 'switch4':
                self.out = check_output("ssh admin@134.104.79.145 uptime".format(self.machine), shell=True)
            elif self.machine == 'switch5':
                self.out = check_output("ssh admin@134.104.79.154 uptime".format(self.machine), shell=True)
            else:
                return("fail", "%s"%(self.machine))
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))
        
    @request(Str())
    @return_reply(Str())
    def request_nvidiasmi(self, req, machine):
        """
        @brief      ssh to a machine and run "nvidia-smi" there
        """
        self.machine = machine
        
        if self.machine == 'pacifix':   # For all pacifix nodes
            self.out = check_output("dsh -g {:s} nvidia-smi".format(self.machine), shell=True)
        elif self.machine in pacifix:   # For each pacifix node
            self.out = check_output("ssh pulsar@{:s} nvidia-smi".format(self.machine), shell=True)
        else:
            return("fail", "%s"%(self.machine))                
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))
    
    @request(Str())
    @return_reply(Str())
    def request_dockerps(self, req, machine):
        """
        @brief      ssh to a machine and "run docker ps" there
        """
        self.machine = machine
        
        if self.machine == 'pacifix':   # For all pacifix nodes
            self.out = check_output("dsh -g {:s} docker ps".format(self.machine), shell=True)
        elif self.machine in pacifix:   # For each pacifix node
            self.out = check_output("ssh pulsar@{:s} docker ps".format(self.machine), shell=True)
        else:
            return("fail", "%s"%(self.machine))                
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))
    
    @request(Str())
    @return_reply(Str())
    def request_freemh(self, req, machine):
        """
        @brief      ssh to a machine and run "free -mh" there
        """
        self.machine = machine
        
        if self.machine == 'pacifix':   # For all pacifix nodes
            self.out = check_output("dsh -g {:s} free -mh".format(self.machine), shell=True)
        elif self.machine in pacifix:   # For each pacifix node
            self.out = check_output("ssh pulsar@{:s} free -mh".format(self.machine), shell=True)
        else:
            return("fail", "%s"%(self.machine))                
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))
    
    @request(Str(), Str())
    @return_reply(Str())
    def request_streamstatus(self, req, machine, eth):
        """
        @brief      ssh to a machine and check stream status there
        """
        self.machine     = machine
        self.eth         = eth
        self.port        = port
        self.execute_dir = execute_dir
                
        if self.machine == 'pacifix':   # For all pacifix nodes
            self.out = check_output("dsh -g {:s} \"{:s}/stream_status.py -a {:s} -b {:s}\"".format(self.machine, self.execute_dir, self.eth, " ".join(self.port)), shell=True)
        elif self.machine in pacifix:   # For each pacifix node
            self.out = check_output("ssh pulsar@{:s} \"{:s}/stream_status.py -a {:s} -b {:s}\"".format(self.machine, self.execute_dir, self.eth, " ".join(self.port)), shell=True)
        else:
            return("fail", "%s %s"%(self.machine, self.eth))                
            
        return("ok", "%s %s \n\n%s"%(self.machine, self.eth, self.out))
    
    @request(Str())
    @return_reply(Str())
    def request_ipcs(self, req, machine):
        """
        @brief      ssh to a machine and run "ipcs -a" there
        """
        self.machine = machine
        
        if self.machine == 'pacifix':   # For all pacifix nodes
            self.out = check_output("dsh -g {:s} ipcs -a".format(self.machine), shell=True)
        elif self.machine in pacifix:   # For each pacifix node
            self.out = check_output("ssh pulsar@{:s} ipcs -a".format(self.machine), shell=True)
        else:
            return("fail", "%s"%(self.machine))                
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))

    @request(Str(), Str())
    @return_reply(Str())
    def request_ipcrm(self, req, machine, key):
        """
        @brief      ssh to a machine and run "ipcrm" there
        """
        self.machine = machine
        self.key     = key
        
        if self.machine == 'pacifix':   # For all pacifix nodes
            if self.key == 'a':
                self.out = check_output("dsh -g {:s} ipcrm -a".format(self.machine), shell=True)
            else:
                self.out = check_output("dsh -g {:s} ipcrm -M {:s}".format(self.machine, self.key), shell=True)
                
        elif self.machine in pacifix:   # For each pacifix node
            if self.key == 'a':
                self.out = check_output("ssh pulsar@{:s} ipcrm -a".format(self.machine), shell=True)
            else:
                self.out = check_output("ssh pulsar@{:s} ipcrm -M {:s}".format(self.machine, self.key), shell=True)
        else:
            return("fail", "%s"%(self.machine))                
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))

    
    @request(Str())
    @return_reply(Str())
    def request_rebootswitch(self, req, machine):
        """
        @brief      ssh to a switch and reboot it
        """
        self.machine = machine

        if self.machine == 'switch':     # For all switches
            self.out = check_output("dsh -g {:s} echo plx9080|sudo -S reboot".format(self.machine), shell=True)
        elif self.machine == 'switch4':
            self.out = check_output("ssh admin@134.104.79.145 echo plx9080|sudo -S reboot".format(self.machine), shell=True)
        elif self.machine == 'switch5':
            self.out = check_output("ssh admin@134.104.79.154 echo plx9080|sudo -S reboot".format(self.machine), shell=True)
        else:
            return("fail", "%s"%(self.machine))
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))
    
    @request(Str())
    @return_reply(Str())
    def request_reconfigswitch(self, req, machine):
        """
        @brief      ssh to a switch and reconfigure it
        """
        self.machine = machine

        if self.machine == 'switch':     # For all switches
            self.out = check_output("dsh -g {:s} echo plx9080|sudo -S \"icos-cfg -a config_good.txt\"".format(self.machine), shell=True)
        elif self.machine == 'switch4':
            self.out = check_output("ssh admin@134.104.79.145 echo plx9080|sudo -S \"icos-cfg -a config_good.txt\"".format(self.machine), shell=True)
        elif self.machine == 'switch5':
            self.out = check_output("ssh admin@134.104.79.154 echo plx9080|sudo -S \"icos-cfg -a config_good.txt\"".format(self.machine), shell=True)
        else:
            return("fail", "%s"%(self.machine))
            
        return("ok", "%s \n\n%s"%(self.machine, self.out))
    
@coroutine
def on_shutdown(ioloop, server):
    yield server.stop()
    ioloop.stop()

def main(host, port):
    server = PafBackendController(host, port)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))
    def start_and_display():
        server.start()
        #server.setup_sensors()
        print "Server started at", server.bind_address
    ioloop = tornado.ioloop.IOLoop.current()
    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', '--host', dest='host', type="string", default="", metavar='HOST',
                      help='attach to server HOST (default="" - localhost)')
    parser.add_option('-p', '--port', dest='port', type=int, default=17107, metavar='N',
                      help='attach to server port N (default=17107)')
    (opts, args) = parser.parse_args()
    sys.argv = sys.argv[:1]
    
    main(opts.host, opts.port)
