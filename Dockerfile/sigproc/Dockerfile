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
    xorg \
    openbox \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -y clean

# Install python packages
RUN pip install pip -U && \
    hash pip	&& \
    pip install setuptools -U && \
    pip install numpy -U && \
    pip install scipy -U && \
    pip install matplotlib -U    

RUN apt-get update -y && \
    apt-get --no-install-recommends -y install \
    autogen \
    libtool \
    libltdl-dev	

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

## Pull all repos
#WORKDIR $SOURCE
#RUN git clone https://github.com/SixByNine/sigproc.git 

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

# Install calceph
WORKDIR $SOURCE
RUN wget https://www.imcce.fr/content/medias/recherche/equipes/asd/calceph/calceph-3.1.0.tar.gz && \
    tar xvzf calceph-3.1.0.tar.gz
WORKDIR $SOURCE/calceph-3.1.0
RUN ./configure --enable-shared CFLAGS=-fPIC FFLAGS=-fPIC --prefix=$PREFIX && \
    make -j $(nproc) && \
    make install
WORKDIR $SOURCE
RUN rm -rf calceph-3.1.0 calceph-3.1.0.tar.gz

# Install tempo2
WORKDIR $SOURCE
ENV TEMPO2 $INSTALL_SHARE
RUN git clone https://bitbucket.org/psrsoft/tempo2.git 
WORKDIR $SOURCE/tempo2
RUN cp -r T2runtime/* $TEMPO2 && \
    ./bootstrap && \
    ./configure --prefix=$PREFIX --with-calceph=$INSTALL_LIB --enable-shared CFLAGS=-fPIC FFLAGS=-fPIC --enable-static --with-pic F77=gfortran --x-libraries=/usr/lib/x86_64-linux-gnu && \
    make -j $(nproc) && \
    make install && \
    make plugins-install -j $(nproc)
WORKDIR $SOURCE
RUN rm -rf tempo2

# SIGPROC
WORKDIR $SOURCE
RUN git clone https://github.com/SixByNine/sigproc.git
WORKDIR $SOURCE/sigproc
RUN ./bootstrap && \
    ./configure --prefix=$PREFIX && \
    make && \
    make install 
WORKDIR $SOURCE
RUN rm -rf sigproc

# Create pulsar user, psr group and change the ownership
RUN groupadd -g 50000 psr && \
    useradd -u 50000 -g 50000 pulsar 

# More setup for psrchive on pulsar home directory
ARG HOME=/home/pulsar
RUN mkdir $HOME && \
    chown -R pulsar:psr $HOME

# Back to root
USER root
WORKDIR /
