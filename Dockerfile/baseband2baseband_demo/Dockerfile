# Copyright (C) 2016 by Ewan Barr
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM xinpingdeng/paf-base
MAINTAINER Xinping Deng "xinping.deng@gmail.com"

# User pulsar
USER pulsar
ENV HOME /home/pulsar
WORKDIR $HOME/xinping/phased-array-feed/script

# Entry point
ENTRYPOINT ["./baseband2baseband_demo_entry.py"]