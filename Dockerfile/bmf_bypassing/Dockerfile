# Copyright (C) 2016 by Ewan Barr
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM xinpingdeng/paf-base

MAINTAINER Niclas Esser "nesser@mpifr-bonn.mpg.com"

# User pulsar
#USER pulsar

COPY python_requirements.txt ./

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update -qq && \
  pip install --no-cache-dir -r python_requirements.txt

RUN git clone https://github.com/NiclasEsser1/phased-array-feed.git && \
  cd phased-array-feed && git checkout bmf_bypassing
