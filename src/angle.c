#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "angle.h"

int degree_hms(double degree, char *hms)
{
  /* Converts double degree to string " hh:mm:ss.ssss" */
  
  int hh, mm, isec;
  double sec;
  
  hh = (int)(degree/15.);
  mm = (int)((degree/15.-hh)*60.);
  sec = ((degree/15.-hh)*60.-mm)*60.;
  isec = (int)((sec*10000. +0.5)/10000);

  if(isec==60)
    {
      sec=0.;
      mm=mm+1;
      if(mm==60)
	{
	  mm=0;
	  hh=hh+1;
	  if(hh==24)
	    hh=0;
	}
    }
  
  sprintf(hms," %02d:%02d:%010.7f",hh,mm,sec);
  return EXIT_SUCCESS;
}

int degree_dms(double degree, char *dms)
{
  /* Converts double degree to string "sddd:mm:ss.sss" */
  
  int dd, mm, isec;
  double trn, sec;
  char sign;
  
  sign=' ';
  if (degree < 0.)
    {
      sign = '-';
      degree = -degree;
    }
  else
    {
      sign = '+';
      degree = degree;
    }
  dd = (int)(degree);
  mm = (int)((degree-dd)*60.);
  sec = ((degree-dd)*60.-mm)*60.;
  isec = (int)((sec*1000. +0.5)/1000);
  if(isec==60)
    {
      sec=0.;
      mm=mm+1;
      if(mm==60)
	{
	  mm=0;
	  dd=dd+1;
        }
    }
  sprintf(dms,"%c%02d:%02d:%08.5f",sign,dd,mm,sec);
  
  return EXIT_SUCCESS;
}

double hms_degree(char *hms)
{
  /* Converts string " hh:mm:ss.ss" or " hh mm ss.ss" to double degree */
  
  int i;
  double hr, min, sec, turn=0.0;
  
  /* Get rid of ":" */
  for(i=0; *(hms+i) != '\0'; i++)
    if(*(hms+i) == ':')
      *(hms+i) = ' ';
  
  i = sscanf(hms,"%lf %lf %lf", &hr, &min, &sec);
  
  if(i > 0)
    {
      turn = hr/24.;
      if(i > 1)turn += min/1440.;
      if(i > 2)turn += sec/86400.;
    }
  if(i == 0 || i > 3)
    turn = 1.0;
  
  return (turn * 360.0);
}

double dms_degree(char *dms)
{
  /* Converts string "-dd:mm:ss.ss" or " -dd mm ss.ss" to double degree */
  
  int i;
  char *ic, ln[40];
  double deg, min, sec, sign, turn=0.0;
  
  /* Copy dms to internal string */
  strcpy(ln,dms);
  
  /* Get rid of ":" */
  for(i=0; *(ln+i) != '\0'; i++)
    if(*(ln+i) == ':')
      *(ln+i) = ' ';
  
  /* Get sign */
  if((ic = strchr(ln,'-')) == NULL)
    sign = 1.;
  else
    {
      *ic = ' ';
      sign = -1.;
    }
  
  /* Get value */
  i = sscanf(ln,"%lf %lf %lf", &deg, &min, &sec);
  if(i > 0)
    {
      turn = deg/360.;
      if(i > 1)turn += min/21600.;
      if(i > 2)turn += sec/1296000.;
      if(turn >= 1.0)turn = turn - 1.0;
      turn *= sign;
    }
  if(i == 0 || i > 3)
    turn =1.0;
  
  return (turn * 360.0);
}
