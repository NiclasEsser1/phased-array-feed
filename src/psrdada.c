#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>

#include "ipcbuf.h"
#include "psrdada.h"
#include "dada_def.h"
#include "ascii_header.h"
#include "futils.h"


int read_dada_header_work(char *hdrbuf, dada_header_t *dada_header)
{  
  if (ascii_header_get(hdrbuf, "UTC_START", "%s", dada_header->utc_start) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting UTC_START, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "PICOSECONDS", "%"SCNu64"", &(dada_header->picoseconds)) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting PICOSECONDS, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "MJD_START", "%lf", &dada_header->mjd_start) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting MJD_START, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "FREQ", "%lf", &(dada_header->freq)) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting FREQ, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }  
  
  if (ascii_header_get(hdrbuf, "TSAMP", "%lf", &dada_header->tsamp) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting TSAMP, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "BW", "%lf", &dada_header->bw) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting BW, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "NCHAN", "%d", &dada_header->nchan) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting NCHAN, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "NBIT", "%d", &dada_header->nbit) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting NBIT, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "NPOL", "%d", &dada_header->npol) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting NPOL, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_get(hdrbuf, "NDIM", "%d", &dada_header->ndim) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting NDIM, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_get(hdrbuf, "FILE_SIZE", "%"SCNu64"", &dada_header->file_size) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting FILE_SIZE, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }   
  
  if (ascii_header_get(hdrbuf, "BYTES_PER_SECOND", "%"SCNu64"", &dada_header->bytes_per_second) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting BYTES_PER_SECOND, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_get(hdrbuf, "RECEIVER", "%s", dada_header->receiver) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting RECEIVER, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_get(hdrbuf, "OBS_ID", "%s", dada_header->obs_id) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting OBS_ID, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_get(hdrbuf, "RA", "%s", dada_header->ra) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting RA, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_get(hdrbuf, "DEC", "%s", dada_header->dec) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error getting DEC, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}

int read_dada_header_from_buffer(ipcbuf_t *hdr, dada_header_t *dada_header)
{
  char *hdrbuf = NULL;
  uint64_t bufsz;

  hdrbuf = ipcbuf_get_next_read(hdr, &bufsz);
  if ((hdrbuf == NULL) || (bufsz != DADA_DEFAULT_HEADER_SIZE))
    {
      fprintf(stderr, "PSRDADA_ERROR: Can not get header buffer to read, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(read_dada_header_work(hdrbuf, dada_header))
    {
      fprintf(stderr, "PSRDADA_ERROR: read_dada_header_work wrong, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  ipcbuf_mark_cleared(hdr);
  
  return EXIT_SUCCESS;
}

int read_dada_header_from_file(char *fname, dada_header_t *dada_header)
{
  char hdrbuf[DADA_DEFAULT_HEADER_SIZE];
  
  if(fileread(fname, hdrbuf, DADA_DEFAULT_HEADER_SIZE) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Can not open %s to read, ", fname);
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(read_dada_header_work(hdrbuf, dada_header))
    {
      fprintf(stderr, "PSRDADA_ERROR: read_dada_header_work wrong, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
    
  return EXIT_SUCCESS;
}

int write_dada_header(ipcbuf_t *hdr, dada_header_t dada_header)
{
  char *hdrbuf = NULL;

  hdrbuf = ipcbuf_get_next_write(hdr);

  if (hdrbuf == NULL)
    {
      fprintf(stderr, "PSRDADA_ERROR: Can not get header buffer to write, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_set(hdrbuf, "UTC_START", "%s", dada_header.utc_start) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting UTC_START, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64"", &(dada_header.picoseconds)) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting PICOSECONDS, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "MJD_START", "%f", &dada_header.mjd_start) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting MJD_START, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "FREQ", "%f", &(dada_header.freq)) < 0)
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting FREQ, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }  
  
  if (ascii_header_set(hdrbuf, "TSAMP", "%f", &dada_header.tsamp) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting TSAMP, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "BW", "%f", &dada_header.bw) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting BW, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "NCHAN", "%d", &dada_header.nchan) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting NCHAN, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "NBIT", "%d", &dada_header.nbit) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting NBIT, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "NPOL", "%d", &dada_header.npol) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting NPOL, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "NDIM", "%d", &dada_header.ndim) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting NDIM, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_set(hdrbuf, "FILE_SIZE", "%"PRIu64"", &dada_header.file_size) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting FILE_SIZE, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }   
  
  if (ascii_header_set(hdrbuf, "BYTES_PER_SECOND", "%"PRIu64"", &dada_header.bytes_per_second) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting BYTES_PER_SECOND, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_set(hdrbuf, "RECEIVER", "%s", dada_header.receiver) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting RECEIVER, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_set(hdrbuf, "OBS_ID", "%s", dada_header.obs_id) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting OBS_ID, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_set(hdrbuf, "RA", "%s", dada_header.ra) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting RA, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (ascii_header_set(hdrbuf, "DEC", "%s", dada_header.dec) < 0)  
    {
      fprintf(stderr, "PSRDADA_ERROR: Error setting DEC, ");
      fprintf(stderr, "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  ipcbuf_mark_filled(hdr, DADA_DEFAULT_HEADER_SIZE);
  
  return EXIT_SUCCESS;
}

/*
  Wrapper of dada_hdu routines 
  write = 0, create/delete hdu as reader
  write = 1, create/delete hdu as writer
  write = 2, create hdu as writer and disable_sod
*/
int dada_hdu(dada_hdu_t *hdu, key_t key, int create, int write, int dbregister)
{
  if (create == 1)
    {
      hdu = dada_hdu_create(NULL);
      dada_hdu_set_key(hdu, key);  
      if(dada_hdu_connect(hdu) < 0)
	{ 
	  fprintf(stderr, "PSRDADA_ERROR: Can not connect to hdu, ");
	  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;    
	}
      
      if (dbregister == 1)
	{
	  if(dada_cuda_dbregister(hdu) < 0)  // registers the existing host memory range for use by CUDA
	    {
	      fprintf(stderr, "PSRDADA_ERROR: Error dbregistering HDU, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      return EXIT_FAILURE;	      
	    }
	}
      if (write == 1 || write == 2)
	{
	  if(dada_hdu_lock_write(hdu) < 0) // make ourselves the write client
	    {
	      fprintf(stderr, "PSRDADA_ERROR: Error locking HDU write, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  if(write == 2) // disable sod as required
	    {
	      if(ipcbuf_disable_sod((ipcbuf_t *)hdu->data_block) < 0)
		{
		  fprintf(stderr, "PSRDADA_ERROR: Can not write data before start, ");
		  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  return EXIT_FAILURE;
		}
	    }
	}
      else    
	{
	  if(dada_hdu_lock_read(hdu) < 0) // make ourselves the write client
	    {
	      fprintf(stderr, "PSRDADA_ERROR: Error locking HDU read, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	}
    }
  else
    {
      if(hdu->data_block)
	{
	  if (write == 1 || write == 2)
	    {
	      if(write == 1)
		{
		  if(ipcbuf_enable_eod((ipcbuf_t *)hdu->data_block) < 0)
		    {
		      fprintf(stderr, "PSRDADA_ERROR: Error enable_eod , ");
		      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		      return EXIT_FAILURE;
		    }
		}
	      if(dada_hdu_unlock_write(hdu) <0)
		{
		  fprintf(stderr, "PSRDADA_ERROR: Error unlocking HDU write, ");
		  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  return EXIT_FAILURE;
		}
	    }
	  else
	    {
	      if(dada_hdu_unlock_read(hdu) < 0)
		{
		  fprintf(stderr, "PSRDADA_ERROR: Error unlocking HDU read, ");
		  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  return EXIT_FAILURE;
		}	    
	    }
	  
	  if (dbregister == 1)
	    {
	      if(dada_cuda_dbunregister(hdu) < 0)
		{
		  fprintf(stderr, "PSRDADA_ERROR: Error dbunregistering HDU, ");
		  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  return EXIT_FAILURE;
		}
	    }
	  dada_hdu_destroy(hdu);
	}	
    }
  
  return EXIT_SUCCESS;
}
