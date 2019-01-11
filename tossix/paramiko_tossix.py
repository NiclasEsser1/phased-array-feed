#!/usr/bin/env python

import paramiko
import time
import argparse
import os

class RemoteControl(object):
    def __init__(self):
        pass
    
    def connect(self, ip, username, bufsz, password=None):
        self.ip         = ip
        self.username   = username
        self.bufsz      = 10240

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=ip, username=username)
        print "Successful connection {} ...".format(self.ip)

        print "Invoke shell from {}... ".format(self.ip)
        self.remote_connection = self.ssh_client.invoke_shell()
        
    def control(self, command, sleep):
        print "Run \"{}\" on {}\n".format(command, ip)
        self.remote_connection.send("{}\n".format(command))
        time.sleep(sleep)
        print self.remote_connection.recv(bufsz)

    def disconnect(self):
        print "Disconnect from {} ...".format(self.ip)
        self.ssh_client.close()

    def __del__(self):
        class_name = self.__class__.__name__
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To run script on tossix remotely')

    bufsz    = 10240
    username = "pulsar"
    ip       = "134.104.74.36"
    
    tossix = RemoteControl()
    
    tossix.connect(ip, username, bufsz)

    tossix.control("bash\n", 1)
    tossix.control(". /home/pulsar/aaron/askap-trunk/initaskap.sh", 1)
    tossix.control(". /home/pulsar/aaron/askap-trunk/Code/Components/OSL/scripts/osl_init_env.sh", 1)
    tossix.control("cd /home/pulsar/aaron/askap-trunk/Code/Components/OSL/scripts/ade", 1)
    tossix.control("python osl_a_metadata_streaming.py", 10) 
    #tossix.control("python osl_a_abf_config_stream.py --alt=4beams --param 'alt.4beams.ade_bmf.stream10G.streamSetup=stream_full_18beams.csv' ", 10)
    #tossix.control("python osl_a_abf_config_stream.py --alt=4beams", 10)
    tossix.control("python osl_a_abf_config_stream.py --param 'ade_bmf.stream10G.streamSetup=stream_full_18beams.csv' ", 10)

    #tossix.control("python osl_a_adx_get_coarse_spectra.py", 10)
    #tossix.control("y", 1)
    #tossix.control("scp coarse_spectra.png pulsar@pacifix0:{}".format(os.getcwd()), 1)

    #os.system("display coarse_spectra.png")
    tossix.disconnect()
