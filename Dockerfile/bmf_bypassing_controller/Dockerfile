FROM python:2

MAINTAINER Niclas Esser "nesser@mpifr-bonn.mpg.de"

ENV OSTYPE linux



COPY python_requirements.txt ./

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update -qq && \
  apt-get -qqy install \
    heimdal-clients && \
  apt-get --reinstall install python-pyasn1 python-pyasn1-modules && \
  pip install --no-cache-dir -r python_requirements.txt

RUN git clone https://github.com/NiclasEsser1/bmf_bypass_controlling.git

COPY / ./
