#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define MSTR_LEN      1024
#define DADA_HDRSZ    4096
#define PKTSZ         1048576 // 1GB
void usage()
{
  fprintf(stdout,
	  "dada2filterbank_main - Convert dada format data into filterbank format\n"
	  "\n"
	  "Usage: dada2filterbank_main [options]\n"
	  " -a input dada file name \n"
	  " -b output filterbank file name \n"
	   );
}

int main(int argc, char **argv)
{
  int i, arg;
  char d_fname[MSTR_LEN], f_fname[MSTR_LEN];
  char hdr_buf[DADA_HDRSZ];
  FILE *d_fp = NULL, *f_fp = NULL;
  char utc_start[MSTR_LEN], field[MSTR_LEN], source_name[MSTR_LEN];
  double mjd_start, picoseconds, freq, tsamp, bw, fch1, foff, rdm = 0.0;
  int nread = 0, nchan, length, nbit, npol, ndim, nifs;
  int telescope_id = 8, data_type = 1, machine_id = 0, nbeams = 1, ibeam = 0;
  char buf[PKTSZ];
  size_t rsz, wsz;
  
  /* Get input parameters */
  while((arg=getopt(argc,argv,"a:hb:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':
	  sscanf(optarg, "%s", d_fname);
	  break;

	case 'b':
	  sscanf(optarg, "%s", f_fname);
	  break;	  
	}
    }

  /* Open files to play */
  d_fp = fopen(d_fname, "r");
  f_fp = fopen(f_fname, "w");

  /* Setup filterbank header parameters */
  while(strstr(hdr_buf, "end of header") == NULL)
    {
      fgets(hdr_buf, sizeof(hdr_buf), d_fp);
      if(strstr(hdr_buf, "UTC_START"))
	{
	  sscanf(hdr_buf, "%*s %s", utc_start);
	  fprintf(stdout, "%s\n", utc_start);
	  nread++;
	}
      if(strstr(hdr_buf, "SOURCE"))
	{
	  sscanf(hdr_buf, "%*s %s", source_name);
	  fprintf(stdout, "%s\n", source_name);
	  nread++;
	}
      if(strstr(hdr_buf, "MJD_START"))
	{
	  sscanf(hdr_buf, "%*s %lf", &mjd_start);
	  fprintf(stdout, "%f\n", mjd_start);
	  nread++;
	}      
      if(strstr(hdr_buf, "PICOSECONDS"))
	{
	  sscanf(hdr_buf, "%*s %lf", &picoseconds);
	  fprintf(stdout, "%f\n", picoseconds);
	  nread++;
	}
      if(strstr(hdr_buf, "FREQ"))
	{
	  sscanf(hdr_buf, "%*s %lf", &freq);
	  fprintf(stdout, "%f\n", freq);
	  nread++;
	}  
      if(strstr(hdr_buf, "NCHAN"))
	{
	  sscanf(hdr_buf, "%*s %d", &nchan);
	  fprintf(stdout, "%d\n", nchan);
	  nread++;
	}   
      if(strstr(hdr_buf, "NBIT"))
	{
	  sscanf(hdr_buf, "%*s %d", &nbit);
	  fprintf(stdout, "%d\n", nbit);
	  nread++;
	}   
      if(strstr(hdr_buf, "NPOL"))
	{
	  sscanf(hdr_buf, "%*s %d", &npol);
	  fprintf(stdout, "%d\n", npol);
	  nread++;
	}     
      if(strstr(hdr_buf, "NDIM"))
	{
	  sscanf(hdr_buf, "%*s %d", &ndim);
	  fprintf(stdout, "%d\n", ndim);
	  nread++;
	}   
      if(strstr(hdr_buf, "TSAMP"))
	{
	  sscanf(hdr_buf, "%*s %lf", &tsamp);
	  fprintf(stdout, "%f\n", tsamp);
	  nread++;
	}   
      if(strstr(hdr_buf, "BW"))
	{
	  sscanf(hdr_buf, "%*s %lf", &bw);
	  fprintf(stdout, "%f\n", bw);
	  nread++;
	}      
    }
  mjd_start = mjd_start + picoseconds / 86400.0E12;
  bw        = -256.0 * 32.0 / 27.0;
  fch1      = freq - 0.5 * bw / nchan * (nchan - 1);
  foff      = bw / nchan;
  tsamp     = tsamp / 1.0E6;
  fprintf(stdout, "%.10f\t%.10f\t%.10f\t%.10f\n", mjd_start, bw, fch1, foff);

  /* Write filterbank header */
  length = 12;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "HEADER_START");
  fwrite(field, sizeof(char), length, f_fp);

  length = 12;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "telescope_id");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&telescope_id, sizeof(int), 1, f_fp);

  length = 9;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "data_type");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&data_type, sizeof(int), 1, f_fp);

  length = 5;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "tsamp");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&tsamp, sizeof(double), 1, f_fp);
  
  length = 6;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "tstart");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&mjd_start, sizeof(double), 1, f_fp);

  length = 5;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "nbits");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&nbit, sizeof(int), 1, f_fp);

  length = 4;
  nifs = npol * ndim;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "nifs");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&nifs, sizeof(int), 1, f_fp);
  
  length = 4;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "fch1");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&fch1, sizeof(double), 1, f_fp);
  
  length = 4;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "foff");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&foff, sizeof(double), 1, f_fp);

  length = 6;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "nchans");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&nchan, sizeof(int), 1, f_fp);

  length = 11;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "source_name");
  fwrite(field, sizeof(char), length, f_fp);
  length = strlen(source_name);
  strncpy(field, source_name, length);
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  fwrite(field, sizeof(char), length, f_fp);
  
  length = 10;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "machine_id");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&machine_id, sizeof(int), 1, f_fp);
  
  length = 6;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "nbeams");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&nbeams, sizeof(int), 1, f_fp);
  
  length = 5;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "refdm");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&rdm, sizeof(double), 1, f_fp);

  length = 5;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "ibeam");
  fwrite(field, sizeof(char), length, f_fp);
  fwrite((char*)&ibeam, sizeof(int), 1, f_fp);
  
  length = 10;
  fwrite((char*)&length, sizeof(int), 1, f_fp);
  strcpy(field, "HEADER_END");
  fwrite(field, sizeof(char), length, f_fp);
  
  //length = 11;
  //fwrite((char*)&length, sizeof(int), 1, f_fp);
  //strcpy(field, "rawdatafile");
  //fwrite(field, sizeof(char), length, f_fp);
  //length = raw_file.size();
  //fwrite((char*)&length, sizeof(int), 1, f_fp);
  //strcpy(field, raw_file.c_str());
  //fwrite(field, sizeof(char), length, f_fp);
  //
  //length = 8;
  //fwrite((char*)&length, sizeof(int), 1, f_fp);
  //strcpy(field, "az_start");
  //fwrite(field, sizeof(char), length, f_fp);
  //fwrite((char*)&az, sizeof(double), 1, f_fp);
  //fwrite((char*)&length, sizeof(int), 1, f_fp);
  //strcpy(field, "za_start");
  //fwrite(field, sizeof(char), length, f_fp);
  //fwrite((char*)&za, sizeof(double), 1, f_fp);
  //
  //length = 7;
  //fwrite((char*)&length, sizeof(int), 1, f_fp);
  //strcpy(field, "src_raj");
  //fwrite(field, sizeof(char), length, f_fp);
  //fwrite((char*)&ra, sizeof(double), 1, f_fp);
  //fwrite((char*)&length, sizeof(int), 1, f_fp);
  //strcpy(field, "src_dej");
  //fwrite(field, sizeof(char), length, f_fp);
  //fwrite((char*)&dec, sizeof(double), 1, f_fp);

  /* Copy data from DADA file to filterbank file */
  fseek(d_fp, DADA_HDRSZ, SEEK_SET);
  while(!feof(d_fp))
    {
      rsz = fread(buf, sizeof(char), PKTSZ, d_fp);
      wsz = fwrite(buf, sizeof(char), rsz, f_fp);
    }
  
  fclose(d_fp);
  fclose(f_fp);
  
  return EXIT_SUCCESS;
}
