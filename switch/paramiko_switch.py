#!/usr/bin/env python

import paramiko
import time
import argparse
import threading

USERNAME = "admin"
PASSWORD = "plx9080"
BUFSZ    = 10240

IP = {"switch4": "134.104.79.145",
      "switch5": "134.104.79.154",}

def switch_control(ip):    
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, username=USERNAME)
    
    print "Successful connection: {:s}".format(ip)
    print "The switch was used to be connected with pacifix{:d}\n".format(switch)
    
    remote_connection = ssh_client.invoke_shell()
    remote_connection.send("sudo reboot\n")
    remote_connection.send(PASSWORD + "\n")
    time.sleep(1)
    
    print remote_connection.recv(BUFSZ)
    ssh_client.close()
    
    print "\nPlease wait for 3 minutes ...\n"
    time.sleep(180)  # Wait 3 minutes for switch to back
    
    print "Start to configure the switch ...\n"
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, username=USERNAME)
    
    print "Successful connection: {:s}".format(ip)
    print "The switch was used to be connected with pacifix{:d}\n".format(switch)
    
    remote_connection = ssh_client.invoke_shell()
    remote_connection.send("sudo icos-cfg -a config_good.txt\n")
    remote_connection.send(PASSWORD + "\n")
    time.sleep(10)
    
    print remote_connection.recv(BUFSZ)
    print "\nFinish the configuration"
    ssh_client.close()


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
        print switch_name
        ip = IP[switch_name]
        print ip
        threads.append(threading.Thread(target = switch_control, args = (ip,)))
        
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()
