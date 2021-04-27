import pandas as pd
import fnmatch
import json
import os
import re
import numpy as np

UNSET_PORT=0
UNSET_MAC="0x020000000000"
UNSET_IP="00.00.0.0"
USER="nesser"
PASSWORD=""


inet = {
    "10.17.0.1" : "pacifix0",
    "10.17.0.2" : "pacifix0",
    "10.17.1.1" : "pacifix1",
    "10.17.1.2" : "pacifix1",
    "10.17.2.1" : "pacifix2",
    "10.17.2.2" : "pacifix2",
    "10.17.3.1" : "pacifix3",
    "10.17.3.2" : "pacifix3",
    "10.17.4.1" : "pacifix4",
    "10.17.4.2" : "pacifix4",
    "10.17.5.1" : "pacifix5",
    "10.17.5.2" : "pacifix5",
    "10.17.6.1" : "pacifix6",
    "10.17.6.2" : "pacifix6",
    "10.17.7.1" : "pacifix7",
    "10.17.7.2" : "pacifix7",
    "10.17.8.1" : "pacifix8",
    "10.17.8.2" : "pacifix8"
}


class ConfigError(Exception):
    pass

class Config:
    def __init__(self, dir, fname):
        self.fname = fname
        self.dir = dir
        if self.dir[-1] != "/":
            self.dir += "/"
        self.conf_path = self.dir + self.fname
        print(self.conf_path)
        # Load all config files
        try:
            self.config_file = open(os.path.relpath(self.conf_path), 'r')
            self.data = json.load(self.config_file)
        except Exception as e:
            raise ConfigError("Failed to init json config: " + str(e.__class__) + str(e))
        try:
            self.rt_path = self.dir + self.data["routing_table"]
            self.df = pd.read_csv(self.rt_path)
            self.rt = self.df.T.to_dict()
        except Exception as e:
            raise ConfigError("Failed to load routing table: " + str(e.__class__) + str(e))
        # Parse them
        # Parse routing table, determine which mac, ip corresponse to which ports
        numa_id = "numa0"
        self.node_dict = {numa_id:{}}
        self.mac_list = []
        self.nof_nodes = 0
        for key_out in self.rt.keys():
            for key_in,val in self.rt[key_out].items():
                if "MAC" in key_in and val != UNSET_MAC:#
                    beamid = int(key_in.replace("MAC",""))
                    if val not in self.mac_list:
                        self.mac_list.append(val)
                        numa_id = re.sub(r'\d', "", numa_id)
                        numa_id += str(str(int(self.nof_nodes)))
                        self.node_dict[numa_id] = {}
                        self.node_dict[numa_id]["beams"] = []
                        self.nof_nodes += 1
                    self.node_dict[numa_id]["mac"] = val
                    self.node_dict[numa_id]["ip"] = self.rt[key_out]["IP"+str(beamid)]
                    self.node_dict[numa_id]["beams"].append([beamid,self.rt[key_out]["PORT"+str(beamid)]])
                    self.node_dict[numa_id]["bandid"] = self.rt[key_out]["BANDID"]
        # print(self.node_dict)
        f = open(os.path.relpath('../tmp/parsed_numa_config.json'), 'w')
        json.dump(self.node_dict, f, indent=4)
        # construct command line pattern for argument 'c' of capture_main program (ip_port_expectedbeams_actualbeams_cpu)
        self.node_list = []
        for idx, key in enumerate(self.node_dict.keys()):
            self.node_list.append(NumaNode(idx, key, self.data, self.node_dict[key]))
            self.node_list[-1].construct_pattern()
        self.node_list.sort(key=lambda x: x.ip)

class NumaNode:
    def __init__(self, id, node_name, config, dictionary):
        self.id = id%2
        self.node_name = node_name
        self.dockername = config["dockername"]+self.node_name
        self.config = config
        self.mac = dictionary["mac"]
        self.ip = dictionary["ip"]
        self.host = inet[self.ip]
        self.bandid = dictionary["bandid"]
        self.freq = float(config["center_freq"] + (self.bandid - config["band_groups"]/2)*config["bandwidth_group"]) +config["bandwidth_group"]/2
        self.beam_port = np.asarray(dictionary["beams"])
        self.storage_dir = config["root_storage_dir"] + self.node_name + "/"
        self.cmd_pattern = config["capture_main"]["directory"] + config["capture_main"]["name"]
        if self.id%2:
            self.start_cpu = self.config["cpus_per_node"]
            self.key = self.config["dada_key_odd"]
        else:
            self.start_cpu = 0
            self.key = self.config["dada_key_even"]
    def construct_pattern(self):
        self.cmd_pattern += " -a " + self.key
        self.cmd_pattern += " -b " + str(self.config["packet_start"])
        self.ports = np.unique(self.beam_port[:,1])
        for idx in range(self.ports.shape[0]):
            self.cmd_pattern += " -c "+str(self.ip)+"_"
            self.cmd_pattern += str(self.ports[idx]) + "_"
            self.cmd_pattern += str(np.count_nonzero(self.beam_port[:,1] == self.ports[idx])) + "_"
            self.cmd_pattern += str(np.count_nonzero(self.beam_port[:,1] == self.ports[idx])) + "_"
            self.cmd_pattern += str(idx+2+self.start_cpu) + " "
        self.cmd_pattern += " -e " + str(self.config["center_freq"])
        self.cmd_pattern += " -g " + self.config["log_dir"]
        self.cmd_pattern += " -i " + str(self.start_cpu)
        self.cmd_pattern += " -j " + str(self.config["capture_control"]) + "_" + str(self.start_cpu + 1)
        self.cmd_pattern += " -k " + str(self.config["bind_cpu"])
        self.cmd_pattern += " -l " + str(self.config["nof_data_frame_per_block"])
        self.cmd_pattern += " -m " + str(self.config["nof_data_frame_tmp_buffer"])
        self.cmd_pattern += " -n " + self.config["docker_config_dir"] + self.config["template_dada_header"]
        self.cmd_pattern += " -o " + self.config["soure_information"]
        self.cmd_pattern += " -p " + str(self.config["padding"])
        self.cmd_pattern += " -q " + str(self.freq)
        self.cmd_pattern += " -r " + str(self.config["nof_data_frame_per_capture"])
        # print(self.cmd_pattern)
