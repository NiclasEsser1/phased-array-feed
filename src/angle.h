#ifdef __cplusplus
extern "C" {
#endif
#ifndef __ANGLE_H
#define __ANGLE_H

  /* 
     function to convert between degree and hms/dms
     modified from functions in TEMPO2
  */
  int degree_hms(double degree, char *hms);
  int degree_dms(double degree, char *dms);
  double hms_degree(char *hms);
  double dms_degree(char *dms);
  
#endif

#ifdef __cplusplus
} 
#endif
