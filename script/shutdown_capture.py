import subprocess
import time
import os
import pty
import argparse
from argparse import RawTextHelpFormatter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutdown container for PAF bmf bypassing')
    parser.add_argument('-n', '--numa_name', action="store", dest="name", help='The ID of NUMA node')
    dockername="capture_bypassed_bmf_"+parser.parse_args().name
    print("Stopping container: "+dockername)
    os.system("docker container stop "+dockername)
    print("SOS")
