# phased-array-feed

The package includes all my existing phased-array-feed (PAF) related software . Major update has to be done to these software, to make them much easier to use and more configurable. 

baseband2baseband is the pre-processing software for DSPSR based pulsar fold mode. It downsamples the PAF baseband data and re-digitizes it to get 8 bits critical sampled baseband data. 

baseband2filterbank is the pre-processing software for FRB pipeline. It downsamples the PAF baseband data, re-channelizes and detect it to get 8 bits filterbank data. 

baseband2flux converts baseband to power with raw channels. 

flux2udp puts flux data with other information to form UDP packets for fits-writer. 

Pipeline includes all related pipelines. 
