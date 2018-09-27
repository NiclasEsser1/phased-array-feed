#!/usr/bin/env python

import glob
import numpy as np
import threading
import os

def keyword_value(data_file, nline_header, key_word):
    data_file.seek(0)  # Go to the beginning of DADA file
    for iline in range(nline_header):
        line = data_file.readline()
        if key_word in line and line[0] != '#':
            return line.split()[1]
    print "Can not find the keyword \"{:s}\" in header ...".format(key_word)
    exit(1)

def append_baseband(fileNames, packetShift, utcStart, mjdStart, picoSeconds, directory, iDirectory, pulsarName, packetSize, dadaHeaderSize, numberChunk):
    foutName = "{:s}/{:s}-baseband{:d}.dada".format(directory, pulsarName, iDirectory)
    fout     = open(foutName, "w")
    first    = True

    bytesShift = packetShift * numberChunk * packetSize
    bytesRead  = packetSize * numberChunk
    for fileName in fileNames:
        fin = open(fileName, "r")
        if first:
            fout.write(fin.read(dadaHeaderSize))
            fin.seek(bytesShift, os.SEEK_CUR)
            while True:
                dt = fin.read(bytesRead)
                if dt == '':
                    break
                else:
                    fout.write(dt)                        
            first = False
        else:
            fin.seek(dadaHeaderSize, os.SEEK_CUR)
            while True:
                dt = fin.read(bytesRead)
                if dt == '':
                    break
                else:
                    fout.write(dt)
        fin.close()
    
    fout.close()
    os.system("dada_install_header {:s} -o -p UTC_START={:s}".format(foutName, utcStart))
    os.system("dada_install_header {:s} -o -p MJD_START={:.10f}".format(foutName, mjdStart))
    os.system("dada_install_header {:s} -o -p PICOSECONDS={:d}".format(foutName, picoSeconds))
    
baseDirectory = "/beegfs/DENG/AUG/baseband"
#pulsarName    = "J0332+5434"
pulsarName    = "J1939+2134"
#pulsarName    = "J1713+0747"
#pulsarName    = "J1819-1458"

numberDirectory  = 6
packetTime       = 108.0E-6
numberChunk      = 8
nline_header     = 50
packetSize       = 7232
packetHeaderSize = 64
dadaHeaderSize   = 4096

# Get file names and reference information
fileNames        = []
referenceSeconds = []
utc_starts       = []
picosecondss     = []
mjd_starts       = [] 
for iDirectory in range(numberDirectory):
    directory = "{:s}/{:s}/{:d}".format(baseDirectory, pulsarName, iDirectory)
    globKey   = "{:s}/*.000000.dada".format(directory)
    fnames    = sorted(glob.glob(globKey))
    fileNames.append(fnames)

    print fnames[0]
    f = open(fnames[0], "r")
    mjd_start    = float(keyword_value(f, nline_header, "MJD_START"))
    utc_start    = keyword_value(f, nline_header, "UTC_START")
    picoseconds  = int(keyword_value(f, nline_header, "PICOSECONDS"))
    minutes      = int(utc_start.split(':')[1])
    seconds      = int(utc_start.split(':')[2])
    f.close()

    referenceSeconds.append(minutes * 60.0 + seconds + picoseconds / 1.0E12)
    mjd_starts.append(mjd_start)
    utc_starts.append(utc_start)
    picosecondss.append(picoseconds)
    
referenceSeconds = np.array(referenceSeconds)
packetShift      = (max(referenceSeconds) - referenceSeconds) / packetTime
packetShift      = map(int, packetShift + 0.5)

maximumIndex     = np.argmax(referenceSeconds)
mjdStart         = mjd_starts[maximumIndex]
utcStart         = utc_starts[maximumIndex]
picoSeconds      = picosecondss[maximumIndex]

print utc_starts
print picosecondss
print referenceSeconds
print packetShift
print utcStart
print picoSeconds
print mjdStart
print mjd_starts
print np.argmin(referenceSeconds)

# Combine all files in the same directory into a single file, also align files in different directories
threads = []
for iDirectory in range(numberDirectory):
    directory = "{:s}/{:s}/{:d}".format(baseDirectory, pulsarName, iDirectory)
    threads.append(threading.Thread(target = append_baseband, args=(fileNames[iDirectory], packetShift[iDirectory], utcStart, mjdStart, picoSeconds, directory, iDirectory, pulsarName, packetSize, dadaHeaderSize, numberChunk)))

for thread in threads:
    thread.start()
    
for thread in threads:
    thread.join()

# Combine files in different directories into a single file
fin       = []
foutName  = "{:s}/{:s}/{:s}-baseband.dada".format(baseDirectory, pulsarName, pulsarName)
fout      = open(foutName, "w")
bytesRead = packetSize * numberChunk

for iDirectory in range(numberDirectory):
    finName = "{:s}/{:s}/{:d}/{:s}-baseband{:d}.dada".format(baseDirectory, pulsarName, iDirectory, pulsarName, iDirectory)
    fin.append(open(finName, "r"))

fout.write(fin[0].read(dadaHeaderSize))
for iDirectory in range(1, numberDirectory):
    fin[iDirectory].seek(dadaHeaderSize, os.SEEK_CUR)

while True:
    dt = fin[0].read(bytesRead)
    if dt == '':
        break
    else:
        fout.write(dt)
        
    for iDirectory in range(numberDirectory):
        if not (iDirectory == 0):
            dt = fin[iDirectory].read(bytesRead)
            if dt == '':
                break
            else:
                fout.write(dt)

for iDirectory in range(numberDirectory):
    fin[iDirectory].close()
fout.close()
os.system("dada_install_header {:s} -o -p BW=336".format(foutName))
os.system("dada_install_header {:s} -o -p NCHAN=336".format(foutName))
os.system("dada_install_header {:s} -o -p FREQ=1340.5".format(foutName))

# Remove packet header
finName  = "{:s}/{:s}/{:s}-baseband.dada".format(baseDirectory, pulsarName, pulsarName)
foutName = "{:s}/{:s}/{:s}.dada".format(baseDirectory, pulsarName, pulsarName)

fin      = open(finName, "r")
fout     = open(foutName, "w")

fout.write(fin.read(dadaHeaderSize))
while True:
    fin.seek(packetHeaderSize, os.SEEK_CUR)
    dt = fin.read(packetSize - packetHeaderSize)
    if dt == '':
        break
    else:
        fout.write(dt)
fin.close()
fout.close()
