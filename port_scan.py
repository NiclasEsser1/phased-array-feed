from inc.ssh_client import *
from inc.config import *
import subprocess
from subprocess import Popen, PIPE
import time

def sniff_packet(node, port_idx = 0):
    print("Sniffing packet on "+node.host+":"+str(node.ports[port_idx])+" to determine time reference")
    sniffer = (SSHConnector(host=node.host, user=USER, gss_auth=True, gss_kex=True, logfile="log/snifflogger_" + node.node_name))
    # sniffer = SSHConnector(host="192.168.0.3", user=USER, logfile="log/snifflogger_"+ node.node_name+".log")
    sniffer.connect()
    sniffer.open_shell()
    out = sniffer.execute(
        "python "+config.data["script_dir"]+config.data["time_ref_script"]["name"]+ " -i " + str(node.ip) + " -p " + str(node.ports[port_idx]),
        config.data["time_ref_script"]["expected_start"],
        config.data["time_ref_script"]["failed"])
    sniffer.close()
    if out != "failed":
        time_ref = string_between(out, config.data["time_ref_script"]["expected_start"],config.data["time_ref_script"]["expected_end"])
        print("Time reference is: "+time_ref)
        return time_ref
    return out

config = Config("./config", "config.json")
script_dir = config.data["script_dir"]

client_list = []
process_list = []
cmdline_file = config.data["remote_config_dir"]+ "command_line"

# Scan ports by sniffing packets. If a node has a dead port it can be be removed from list
for node in config.numa_list:
    for port_idx in range(len(node.ports)):
        out = sniff_packet(node, port_idx)
        if out == "failed":
            print("\n\nPort " +node.ip+ ":" + str(node.ports[port_idx]) + " on " + node.node_name + " is dead\n\n")
