#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <inttypes.h>

//gcc -o idf idf.c  -lm

int acquire_idf(uint64_t idf, uint64_t sec, uint64_t idf_ref, uint64_t sec_ref, double df_res, int64_t *idf_buf);

int main(int argc, char **argv)
{
  uint64_t idf = 250000 - 10101;
  uint64_t sec = 732175;
  uint64_t idf_ref = 250000 - 1000;
  uint64_t sec_ref = 732175 + 27;
  
  int64_t idf_buf;
  double df_res = 1.08E-4;

  acquire_idf(idf, sec, idf_ref, sec_ref, df_res, &idf_buf);
  fprintf(stdout, "%"PRId64"\n", idf_buf);
  return EXIT_SUCCESS;
}

int acquire_idf(uint64_t idf, uint64_t sec, uint64_t idf_ref, uint64_t sec_ref, double df_res, int64_t *idf_buf)
{

  *idf_buf = (int64_t)(idf - idf_ref) + ((double)sec - (double)sec_ref) / df_res;

  //fprintf(stdout, "%f\n", (double)((double)sec - (double)sec_ref) / df_res);
  //fprintf(stdout, "%"PRId64"\t%E\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", *idf_buf, df_res, idf, idf_ref, sec, sec_ref);
  
  return EXIT_SUCCESS;
}
