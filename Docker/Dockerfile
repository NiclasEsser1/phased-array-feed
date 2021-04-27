# Copyright (C) 2016 by Ewan Barr
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM eddinfra0:5000/eddbase:latest

MAINTAINER Niclas Esser "nesser@mpifr-bonn.mpg.com"

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update -qq

RUN git clone https://github.com/NiclasEsser1/phased-array-feed.git && \
  cd phased-array-feed && git checkout bmf_bypassing \
  && cd paf_capture && make && make install \
