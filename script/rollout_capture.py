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


print("Launching capture dockers on pacifix nodes")
for node in config.node_list:
    # Create a storage directory for each node
    if not os.path.exists(node.storage_dir):
        os.mkdir(node.storage_dir)

    # Add time reference to each capture command pattern
    # Establish SSH connection to each node (Kerberos)
    node.ssh_client = SSHConnector(host=node.host, user=USER, password=PASSWORD, gss_auth=True, gss_kex=True, logfile="log/logger_" + node.node_name)
    node.ssh_client.connect()
    node.ssh_client.open_sftp()
    node.ssh_client.upload(config.data["local_config_dir"] + config.data["template_dada_header"], config.data["remote_config_dir"] + config.data["template_dada_header"])
    node.ssh_client.open_shell()

    # Write commands executed from inside Docker into a file
    out = node.ssh_client.execute("echo 'dada_db -k " +node.key+ " -d' > " +cmdline_file + str(node.node_name) + ".txt", "", "fail")
    out = node.ssh_client.execute("echo 'numactl -m " +str(node.id)+ " dada_db -k " +node.key+ " -l -p -b " +str(config.data["dada_block_size"])+ " -n " +str(config.data["nof_dada_blocks"])+ "' >> " +cmdline_file + str(node.node_name) + ".txt", "", "fail")
    out = node.ssh_client.execute("echo 'numactl -m " +str(node.id)+ " dada_dbdisk -k " +node.key+ " -D " +node.storage_dir+ " -W -d -s' >> " +cmdline_file + str(node.node_name) + ".txt", "", "fail")

    # Launching dockers on each node
    launch_cmd = "python " + script_dir + config.data["launch_script"] + " -n " +str(node.dockername) + " -c " + cmdline_file + str(node.node_name) + ".txt " + "-d " + config.data["dockerimage"]
    p1 = Popen(node.ssh_client.execute(launch_cmd, "", "failed"), shell=True,  stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # node.ssh_client.close()
    node.ssh_client2 = SSHConnector(host=node.host, user=USER, password=PASSWORD, gss_auth=True, gss_kex=True, logfile="log/logger_" + node.node_name)
    node.ssh_client2.connect()

# print("Wait for init...")
time.sleep(10)
while raw_input("Start capture?y/n") == "y":
    time_ref = sniff_packet(config.node_list[0])
    for node in config.node_list:
        node.cmd_pattern += " -f " + str(time_ref)
        # node.cmd_pattern = "dada_junkdb -k " +node.key+ " -b 8531214336 -c f -r 2432.666 "+ config.data["docker_config_dir"] + config.data["template_dada_header"]

        node.ssh_client2.open_shell()
        node.ssh_client2.execute("echo '"+node.cmd_pattern+"' >> " +cmdline_file + str(node.node_name) + ".txt", "", "fail")
        start_cmd = "python " + script_dir + config.data["start_script"] + " -n " +str(node.dockername) + " -c " + cmdline_file + str(node.node_name) + ".txt" + " -t " + config.data["remote_config_dir"] + config.data["template_dada_header"]
        node.ssh_client2.execute(start_cmd, "", "failed")


for node in config.node_list:
    node.ssh_client.execute("docker stop "+node.dockername, "", "failed")
    node.ssh_client.execute("docker container rm "+node.dockername, "", "failed")
    print("Docker " +  node.dockername + " stopped")

for node in config.node_list:
    node.ssh_client.close()
