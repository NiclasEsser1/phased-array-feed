# Max-Planck-Institution for Radioastronmy (MPIfR),
# Bonn, Auf dem Huegel 69
#
# Beam weight calculation tool.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import logging
import paramiko
import base64
import sys
import os
import time
import socket
from os import path
from time import gmtime, strftime
import re


"""
 Description:
 ------------
    Simple SSH client based on paramiko

Institution: Max-Planck Institution for Radioastronomy (MPIfR-Bonn)
    Auf dem Huegel 69, Bonn, Germany

Author: Niclas Eesser <nesser@mpifr-bonn.mpg.de>

Changelog :
    - Ver 1.0000   : Initial version (2020 08 28)
"""


def resolve_directory(dir):
    """
    Description:
    ------------
       Parses a string that contains a username, address, directory
    Parameters:
    -----------
        :param dir: Directory to parse (dtype str)
    """
    list = re.split('@|:', dir)
    if len(list) == 3:
        username = list[0]
        host = list[1]
        folder = list[2]
    elif len(list) == 2:
        username = list[0]
        host = list[0]
    elif len(list) == 1:
        username = None
        host = None
        folder = list[0]
    else:
        print("Error: Could not resolve directory")
        sys.exit(1)
    return username, host, folder

def string_between(string, start, end):
    r = re.compile(start+'(.*?)'+end)
    m = r.search(string)
    if m:
        return m.group(1)

class SSHConnectorError(Exception):
    pass

class SSHConnector():
    """
    Description:
    ------------
        A class that allows to connect via SSH.

    Attributes
    ----------
        host : str
            A string that represents the remote host address
        username : str
            Name of user
        password : str
            Password of user
        port : int
            Port to connect to remote host
        gss_auth: bool
            Kerberos authentication based on gssapi. If desired set to True
        client: SSHClient
            Instance of paramiko's SSHClient class. Provides the connection to a remote server.
        sftp: SFTPClient
            Instance of paramiko's SFTPClient class. Allows to down- and upload files.

    Methods
    -------
        connect()
            Connects to a server by given attributes that are passed to constructor
        download(src_file, dst_file)
            Uses the SFTP client to download one file
        upload(src_file, dst_file)
            Uses the SFTP client to upload one file
        download_files(src_dir, dst_dir, fnames)
            Uses SFTPClient to download several files
        upload_files(src_dir, dst_dir, fnames)
            Uses SFTPClient to upload several files
        close()
            Closes the connection of SFTPClient and SSHClient
    """
    def __init__(self, host=None, user=None, password=None, port=22, gss_auth=False, gss_kex=False, log_level="INFO", logfile="", timeout=15, client_id=0):
        """
        Description:
        ------------
            Constructor of class SSHConnector

        Parameters:
        -----------
            All parameters passed to constructor already described in class description
        """
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.gss_auth = gss_auth
        self.gss_kex = gss_kex
        self.id = paramiko.util.get_thread_id()
        self.log_level = log_level
        self.timeout = timeout
        self.shell_active = False
        self.sftp_active = False

        if logfile == "":
            self.logfile = "log/client" +str(self.id)+ "_" +strftime("%Y%m%d_%H%M%S", gmtime())+ ".log"
        else:
            self.logfile = logfile
        # Set logging
        try:
            self.log = logging.getLogger("paramiko")
            self.log.setLevel(self.log_level)
            handler = logging.StreamHandler(open(self.logfile, "a"))
            frm = "%(levelname)-.3s [%(asctime)s.%(msecs)03d] thr="+str(self.id)+"-3d"
            frm += " %(name)s: %(message)s"
            handler.setFormatter(logging.Formatter(frm, "%Y%m%d-%H:%M:%S"))
            self.log.addHandler(handler)
        except Exception as e:
            raise SSHConnectorError("Failed with:" + str(e.__class__) + str(e))
    def connect(self):
        """
        Description:
        ------------
            Connects to a server by given attributes that are passed to constructor

        Parameters:
        -----------
            None
        """
        print("Trying to connect: " + str(self.user) + "@"+ str(self.host) + ":" + str(self.port))
        try:
            self.client = paramiko.SSHClient()
            self.client.load_system_host_keys()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Try usual SSH connection
            if self.gss_auth != True:
                try:
                    self.client.connect(
                        hostname=self.host,
                        port=self.port,
                        username=self.user,
                        password=self.password,
                        timeout=self.timeout)
                except Exception as e:
                    self.log.error("Could not connect via SSH: " + str(e.__class__) + ": " + str(e))
                    raise SSHConnectorError("Could not connect via SSH: " + str(e.__class__) + ": " + str(e))
            # Try to connect with gssapi and Kerberos ticket authentication
            else:
                try:
                    self.client.connect(
                        hostname=self.host,
                        port=self.port,
                        username=self.user,
                        password=self.password,
                        timeout=self.timeout,
                        gss_auth=self.gss_auth,
                        gss_kex=self.gss_kex)
                except Exception as e:
                    self.log.error("Could not connect via Kerberos; Error: " + str(e.__class__) + ": " + str(e))
                    raise SSHConnectorError("Could not connect via Kerberos; Error: " + str(e.__class__) + ": " + str(e))
            print("Connected!")
        except Exception as e:
                self.log.error("connect() failed with: "  + str(e.__class__) + ": " + str(e))
                raise SSHConnectorError("connect() failed with: "  + str(e.__class__) + ": " + str(e))

        self.sftp = 0
        self.shell = 0

    def download(self, src_file, dst_file):
        """
        Description:
        ------------
            Uses the SFTP client to download one file.

        Parameters:
        -----------
            :param src_file: String that contains directory + filename of one file on remote system,
                which should be downloaded.
            :param dst_file: String that contains directory + filename of one file on local system.
                If the directory does not exist, it will be created.

        """
        try:
            os.mkdir(dst_dir)
        except Exception as e:
            self.log.error("download() failed with:"  + str(e.__class__) + ": " + str(e))
            raise SSHConnectorError("download() failed with:"  + str(e.__class__) + ": " + str(e))
        self.sftp.get(src_file, dst_file)

    def upload(self, src_file, dst_file):
        """
        Description:
        ------------
            Uses the SFTP client to upload one file

        Parameters:
        -----------
            :param src_file: String containing directory + filename of one file on local system,
                which should be uploaded.
            :param dst_file: String containing directory + filename of one file on remote system.
                If the directory does not exists the upload will fail.

        """
        self.sftp.put(src_file, dst_file)
    def open_shell(self):
        try:
            self.shell = self.client.invoke_shell()
        except Exception as e:
            self.log.error("open_shell() failed with:"  + str(e.__class__) + ": " + str(e))
            raise SSHConnectorError("open_shell() failed with:"  + str(e.__class__) + ": " + str(e))
        self.shell_active = True

    def open_sftp(self):
        try:
            self.sftp = self.client.open_sftp()
        except Exception as e:
            self.log.error("open_sftp() failed with:"  + str(e.__class__) + ": " + str(e))
            raise SSHConnectorError("open_sftp() failed with:"  + str(e.__class__) + ": " + str(e))
        self.sftp_active = True
    def execute(self, cmd, expected_succes_string, expected_fail_string):
        if not self.shell:
            return
        self.shell.sendall(cmd + "\r\n")
        timeout_cnt = 0
        while True:
            try:
                data = self.shell.recv(1 << 15)
            except socket.timeout:
                self.log.error("execute() client did not respond: timeout")
                raise SSHConnectorError("execute() client did not respond: timeout" + cmd)
            output = data
            if output.find(expected_succes_string) != -1:
                self.log.info(output)
                return output
            if output.find(expected_fail_string) != -1:
                self.log.error("execute() failed with command: " + cmd + "Erro msg: " + output)
                return "failed"
                # raise SSHConnectorError("execute() failed with command: " + cmd)


            # time.sleep(1)
    def download_files(self, src_dir, dst_dir, fnames):
        """
        Description:
        ------------
            Downloads a set of files passed as a list.

        Parameters:
        -----------
            :param src_dir: String containing the directory where the files to be downloaded are located (remote).
            :param dst_dir: String containing the directory where the files should be stored (local).
            :param fnames: List of files
        Returns:
        --------
            A list of all downloaded files
        """
        if src_dir[-1] != "/":
            src_dir += "/"
        stdin, stdout, stderr = self.client.exec_command("cd " + src_dir + " && ls " + fnames)
        file_list = [f.replace('\n', '') for f in stdout.readlines()]
        if not len(file_list):
            print("Error: No file found in source directory " + src_dir + " that matches: " + fnames +" ...")
            sys.exit(1)

        print("Found " + str(len(file_list)) + " files in directory " + src_dir)
        for f_idx, f_name in enumerate(file_list):
            print("Downloading file " + str(f_idx) + ": " + f_name + "...")
            self.download(src_dir + f_name, dst_dir + f_name)
        return file_list

    def upload_files(self, src_dir, dst_dir, fnames):
        """
        Description:
        ------------
            Uploads a one or serveral files that is passed as a list.

        Parameters:
        -----------
            :param src_dir: String containing the directory where the files to be downloaded are located (remote).
            :param dst_dir: String containing the directory where the files should be stored (local).
            :param fnames: List or string containing files
        """
        if src_dir[-1] != "/":
            src_dir += "/"
        if type(fnames) is not list:
            file_list = [fnames]
        else:
            file_list = fnames
        for f_idx, f_name in enumerate(file_list):
            if path.exists(src_dir + f_name):
                print("Uploading file " + str(f_idx) + ": " + dst_dir + f_name + " ...")
                self.upload(src_dir + f_name, dst_dir + f_name)
            else:
                print("File " + src_dir + f_name + " does not exist!")


    def close(self):
        """
        Description:
        ------------
            Closes the connection of opened clients.

        Parameters:
        -----------
            None
        """
        print("Closing connection")
        if self.sftp_active:
            try:
                self.sftp.close()
            except:
                raise SSHConnectorError("close() failed with:"  + str(e.__class__) + ": " + str(e))
        if self.shell_active:
            try:
                self.shell.close()
            except:
                raise SSHConnectorError("close() failed with:"  + str(e.__class__) + ": " + str(e))
        try:
            self.client.close()
        except:
            raise SSHConnectorError("close() failed with:"  + str(e.__class__) + ": " + str(e))
