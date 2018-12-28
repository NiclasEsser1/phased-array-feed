#!/usr/bin/env python

import paramiko
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To reboot and configure swtich with one go')
    # Need to get the status of switch also
    
    parser.add_argument('-a', '--switch', type=int, nargs='+',
                        help='The ID of switch')
    args     = parser.parse_args()
    
    switch   = args.switch[0]
    
    bufsz    = 10240
    username = "admin"
    password = "plx9080"
    if switch == 4:
        ip = "134.104.79.145"
    if switch == 5:
        ip = "134.104.79.154"    

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, username=username)
    
    print "Successful connection: {:s}".format(ip)
    print "The switch was used to be connected with pacifix{:d}\n".format(switch)
    
    remote_connection = ssh_client.invoke_shell()
    remote_connection.send("sudo reboot\n")
    remote_connection.send(password + "\n")
    time.sleep(1)
    
    print remote_connection.recv(bufsz)
    ssh_client.close()
    
    print "\nPlease wait for 3 minutes ...\n"
    time.sleep(180)  # Wait 3 minutes for switch to back
    
    print "Start to configure the switch ...\n"
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, username=username)
    
    print "Successful connection: {:s}".format(ip)
    print "The switch was used to be connected with pacifix{:d}\n".format(switch)
    
    remote_connection = ssh_client.invoke_shell()
    remote_connection.send("sudo icos-cfg -a config_good.txt\n")
    remote_connection.send(password + "\n")
    time.sleep(10)
    
    print remote_connection.recv(bufsz)
    print "\nFinish the configuration"
    ssh_client.close()
