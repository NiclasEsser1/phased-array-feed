# Copyright (C) 2016 by Ewan Barr
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM ubuntu:16.04

MAINTAINER Xinping Deng "xinping.deng@gmail.com"

# To get rid of "(TERM is not set, so the dialog frontend is not usable.)"
ARG DEBIAN_FRONTEND=noninteractive
# To use bash during build
SHELL ["/bin/bash", "-c"]          

# Create space for ssh daemon and update the system
RUN apt-get -y check && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install apt-utils software-properties-common && \
    apt-get -y update --fix-missing && \
    apt-get -y upgrade && \
    apt-get -y install gcc

RUN apt-get --no-install-recommends -y install \
    python \
    python-dev \
    python-pip \
    && apt-get -y clean


RUN pip install --upgrade pip -U && \
    hash -d pip && \
    pip install setuptools -U && \
    pip install wheel -U && \
    pip install tornado -U && \
    pip install cmd2 -U && \
    pip install pyparsing -U && \
    pip install katcp -U 
    
# Create pulsar user, psr group and change the ownership
RUN groupadd -g 50000 psr && \
    useradd -u 50000 -g 50000 pulsar 