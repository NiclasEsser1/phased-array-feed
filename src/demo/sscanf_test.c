#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

// gcc -o sscanf_test sscanf_test.c
#define MSTR_LEN  1024

int main(int argc, char **argv)
{
  char buf[MSTR_LEN];
  char command[MSTR_LEN];
  char src_name[MSTR_LEN];
  char ra[MSTR_LEN], dec[MSTR_LEN];
  //char start_buf[MSTR_LEN], start_byte[MSTR_LEN];
  uint64_t start_buf, start_byte;
  
  strcpy(buf, "START-OF-DATA:PSR J1939+2134:06 05 56.34:+23 23 40.0:10000:10001");
  //strcpy(buf, "START-OF-DATA:PSR J1939+2134:06:05:56.34:+23:23:40.0:10000:10001");
  sscanf(buf, "%[^:]:%[^:]:%[^:]:%[^:]:%"SCNu64":%"SCNu64"", command, src_name, ra, dec, &start_buf, &start_byte);
  //sscanf(buf, "%[^:]:%[^:]:%*[^:]:%*[^:]:%u:%*[^:]:%*[^:]:%u:%"SCNu64":%"SCNu64"", command, src_name, ra, dec, &start_buf, &start_byte);

  
     
  fprintf(stdout, "%s\n", buf);
  fprintf(stdout, "%s\t%s\t%s\t%s\t%"PRIu64"\t%"PRIu64"\n", command, src_name, ra, dec, start_buf, start_byte);
  fprintf(stdout, "%s\n", command);
  fprintf(stdout, "%s\n", src_name);
  fprintf(stdout, "%s\n", ra);
  fprintf(stdout, "%s\n", dec);
  fprintf(stdout, "%"PRIu64"\n", start_buf);
  fprintf(stdout, "%"PRIu64"\n", start_byte);
  
  return EXIT_SUCCESS;
}
