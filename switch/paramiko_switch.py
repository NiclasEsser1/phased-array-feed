#!/usr/bin/env python

import paramiko
import time
import argparse
import threading
import logging
import socket

log = logging.getLogger("mpikat.paf_switch")

PARAMIKO_BUFSZ  = 10240
SWITCH_PASSWORD = "plx9080"
SWITCH_USERNAME = "admin"
SLEEP_TIME      = 180

IP = {"switch4": "134.104.79.145",
      "switch5": "134.104.79.154",}

class RemoteAccessError(Exception):
    pass

class RemoteAccess(object):
    def __init__(self, ip, username, bufsz=PARAMIKO_BUFSZ, password=None):
        self.ip         = ip
        self.username   = username
        self.bufsz      = bufsz
        self.password   = password

    def __enter__(self):        
        # Setup paramiko clinet
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=self.ip, username=self.username, password=self.password)
        log.info("Successful connection {} ...".format(self.ip))
        
        log.info("Invoke shell from {}... ".format(self.ip))
        self.remote_shell = self.ssh_client.invoke_shell()
        
        return self
    
    def reboot(self):
        self.remote_shell.sendall("sudo reboot\n"+self.password+"\n")        
        output = ""
        while True:
            try:
                data = self.remote_shell.recv(1<<15)
            except socket.timeout:
                continue
            output += data
            if output.find("The system is going down for reboot NOW!") != -1:
                log.info(output)
                print output
                return
            
    def control(self, cmd):        
        self.remote_shell.sendall(cmd+"\n"+self.password+"\n")   
        output = ""
        while True:
            try:
                data = self.remote_shell.recv(1<<15)
            except socket.timeout:
                continue
            output += data
            if output.find("Completed apply operation.") != -1:
                log.info(output)
                print output
                return
        
    def scp(self, src, dst):        
        log.info("Create SCP connection with {}... ".format(self.ip))
        self.sftp_client = self.ssh_client.open_sftp()        
        log.info("Copy {} to {}".format(src, dst))
        self.sftp_client.put(src, dst)
        log.info("Close scp channel")
        self.sftp_client.close()
        
    def __exit__(self, *args):
        log.info("Disconnect from {} ...".format(self.ip))
        self.ssh_client.close()

def switch_control(ip):
    print "Reboot switch {}".format(ip)
    with RemoteAccess(ip = ip, username = SWITCH_USERNAME, password = SWITCH_PASSWORD) as switch:
        switch.reboot()

    print "Sleep for {} seconds for reboot".format(SLEEP_TIME)
    time.sleep(SLEEP_TIME)
    print "Sleep DONE"

    print "Configure switch {}".format(ip)
    with RemoteAccess(ip = ip, username = SWITCH_USERNAME, password = SWITCH_PASSWORD) as switch:
        switch.control("sudo icos-cfg -a config_good.txt")
    print "DONE\n\n\n"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To reboot and configure swtich with one go')
    # Need to get the status of switch also
    
    parser.add_argument('-a', '--switch', type=int, nargs='+',
                        help='The ID of switch')
    args     = parser.parse_args()

    switches   = args.switch

    threads = []
    for switch in switches:
        switch_name = "switch{}".format(switch)
        ip = IP[switch_name]
        threads.append(threading.Thread(target = switch_control, args = (ip,)))
        
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()
