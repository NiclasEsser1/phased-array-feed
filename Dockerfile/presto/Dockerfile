# Copyright (C) 2018 by Xinping Deng
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM ubuntu:xenial-20160923.1

MAINTAINER Xinping Deng "xinping.deng@gmail.com"

# To get rid of "(TERM is not set, so the dialog frontend is not usable.)"
ARG DEBIAN_FRONTEND=noninteractive
# To use bash during build
SHELL ["/bin/bash", "-c"]  

# Create space for ssh daemon and update the system
RUN echo 'deb http://us.archive.ubuntu.com/ubuntu trusty main multiverse' >> /etc/apt/sources.list && \
    #mkdir /var/run/sshd && \
    apt-get -y check && \
    apt-get -y update && \
    apt-get install -y apt-utils apt-transport-https software-properties-common python-software-properties && \
    apt-get -y update --fix-missing && \
    apt-get -y upgrade 

# Install dependencies
RUN apt-get --no-install-recommends -y install \
    build-essential \
    autoconf \
    autotools-dev \
    automake \
    pkg-config \
    csh \
    gcc \
    gfortran \
    wget \
    git \
    libcfitsio-dev \
    pgplot5 \
    swig2.0 \    
    python \
    python-dev \
    python-pip \
    libfftw3-3 \
    libfftw3-bin \
    libfftw3-dev \
    libfftw3-single3 \
    libx11-dev \
    libpng12-dev \
    libpng3 \
    libpnglite-dev \   
    libglib2.0-0 \
    libglib2.0-dev \
    #openssh-server \
    #docker.io \
    python-tk \	
    xorg \
    openbox \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -y clean

# Install python packages
RUN pip install pip -U && \
    hash pip && \
    pip install setuptools -U && \
    pip install numpy -U && \
    pip install scipy -U && \
    pip install matplotlib -U   &&\
    pip install astropy -U

# PGPLOT
ENV PGPLOT_DIR /usr/lib/pgplot5
ENV PGPLOT_FONT /usr/lib/pgplot5/grfont.dat
ENV PGPLOT_INCLUDES /usr/include
ENV PGPLOT_BACKGROUND white
ENV PGPLOT_FOREGROUND black
ENV PGPLOT_DEV /xs

# Define directory for source code, OSTYPE and installation directory
ENV SOURCE           /root
ENV PREFIX           /usr/local
ENV INSTALL_BIN      $PREFIX/bin
ENV INSTALL_INCLUDES $PREFIX/include
ENV INSTALL_LIB      $PREFIX/lib
ENV INSTALL_SHARE    $PREFIX/share
ENV OSTYPE           linux-gnu
ENV LOGIN_ARCH	     linux
ENV CUDA_DIR         /usr/local/cuda
ENV CUDA_HOME        $CUDA_DIR
ENV CUDA_LIB         $CUDA_DIR/lib64
ENV CUDA_INCLUDE     $CUDA_DIR/include
ENV C_INCLUDE_PATH   $INSTALL_INCLUDES:$C_INCLUDE_PATH
ENV LD_LIBRARY_PATH  $INSTALL_LIB:$LD_LIBRARY_PATH    

# Install psrcat
WORKDIR $SOURCE
RUN wget http://www.atnf.csiro.au/people/pulsar/psrcat/downloads/psrcat_pkg.tar.gz && \
    tar xvzf psrcat_pkg.tar.gz
WORKDIR $SOURCE/psrcat_tar
RUN /bin/bash makeit && \
    cp psrcat    $INSTALL_BIN && \
    cp *.h       $INSTALL_INCLUDES && \
    cp *.a       $INSTALL_LIB && \
    cp psrcat.db $INSTALL_SHARE
ENV PSRCAT_FILE  $INSTALL_SHARE/psrcat.db
WORKDIR $SOURCE
RUN rm -rf psrcat_tar psrcat_pkg.tar.gz

# Tempo
ENV TEMPO $PREFIX
WORKDIR $SOURCE
RUN git clone git://git.code.sf.net/p/tempo/tempo
WORKDIR $SOURCE/tempo
RUN ./prepare && \
    ./configure --prefix=$PREFIX && \
    make && \
    make install && \
    sed -i 's/tempo/local/g' tempo.cfg	&& \
    sed -i 's/root/usr/g' tempo.cfg && \
    cp tempo.hlp obsys.dat tempo.cfg $TEMPO && \
    cp -r clock tzpar ephem $TEMPO
WORKDIR $SOURCE
RUN rm -rf tempo

# pyslalib
WORKDIR $SOURCE
RUN git clone https://github.com/scottransom/pyslalib.git 
WORKDIR $SOURCE/pyslalib
ENV PYTHONPATH $PREFIX:$PYTHONPATH
RUN python setup.py install --record list.txt && \
    python setup.py clean --all && \
    rm -rf .git
WORKDIR $SOURCE	
RUN rm -r pyslalib

# Presto
ENV SOURCE $PREFIX
# Pulsar can not access /root
WORKDIR $SOURCE
ENV PRESTO $SOURCE/presto
RUN git clone https://github.com/scottransom/presto.git
ENV PATH $PRESTO/bin:$PRESTO/lib/python:$PATH
ENV LD_LIBRARY_PATH $PRESTO/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH $PRESTO/lib/python:$PYTHONPATH
WORKDIR $PRESTO/src
RUN rm -rf ../.git && \
    make prep && \ 
    make makewisdom && \
    make
WORKDIR $PRESTO/python
RUN make 

# Create pulsar user, psr group and change the ownership
RUN groupadd -g 50000 psr && \
    useradd -u 50000 -g 50000 pulsar 

# Define home and create the directory
ARG HOME=/home/pulsar
ARG PATH=$PATH
ARG LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ARG PYTHONPATH=$PYTHONPATH
RUN mkdir -p $HOME && \
    chown -R pulsar:psr $HOME
    
# Back to root
USER root
WORKDIR /