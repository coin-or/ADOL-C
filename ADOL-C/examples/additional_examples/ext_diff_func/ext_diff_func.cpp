/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ext_diff_func.cpp
 Revision: $Id$
 Contents: example for external differentiated functions

 Copyright (c) Andrea Walther
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#include <math.h>
#include <adolc/adolc.h>

#define h 0.01
#define steps 100

// time step function
// double version 
int euler_step(int n, double *yin, int m, double *yout);

// adouble version 
void euler_step_act(int n, adouble *yin, int m, adouble *yout);


// versions for usage as external differentiated function
ADOLC_ext_fct zos_for_euler_step;
ADOLC_ext_fct_fos_reverse fos_rev_euler_step;

ext_diff_fct *edf;
int tag_full, tag_part, tag_ext_fct;


int main()
{
  // time interval
  double t0, tf;

  // state, double and adouble version
  adouble y[2];
  adouble ynew[2];
  int n, m;

  // control, double and adouble version
  adouble con[2];
  double conp[2];

  // target value;
  double f;

  //variables for derivative caluclation
  double yp[2], ynewp[2];
  double u[2], z[2];
  double grad[2];


  int i,j;
  
  // tape identifiers
  tag_full = 1;
  tag_part = 2;
  tag_ext_fct = 3;

  // two input variables for external differentiated function
  n = 2;
  // two output variables for external differentiated function
  m = 2;

  // time interval
  t0 = 0.0;
  tf = 1.0;

  //control
  conp[0] = 1.0;
  conp[1] = 1.0;

  trace_on(tag_full);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];
 
    for(i=0;i<steps;i++)
      {
	euler_step_act(n,y,m,ynew);
	for(j=0;j<2;j++)
	  y[j] = ynew[j];
      }
    y[0] + y[1] >>= f;
  trace_off(1);

  gradient(tag_full,2,conp,grad);
  
  printf(" full taping:\n gradient=( %f, %f)\n\n",grad[0],grad[1]);

  // Now using external function facilities

  // tape external differentiated function

  trace_on(tag_ext_fct);
    y[0] <<= conp[0];
    y[1] <<= conp[1];
  
    euler_step_act(2,y,2,ynew);
    ynew[0] >>= f;
    ynew[1] >>= f;
  trace_off(1);


  // register external function
  edf = reg_ext_fct(euler_step);

  // information for Zero-Order-Scalar (=zos) forward
//   yp = new double[2];
//   ynewp = new double[2];
  edf->zos_forward = zos_for_euler_step;
  edf->dp_x = yp;
  edf->dp_y = ynewp;
  // information for First-Order-Scalar (=fos) reverse
  edf->fos_reverse = fos_rev_euler_step;
  edf->dp_U = u;
  edf->dp_Z = z;

  trace_on(tag_part);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];
  
    for(i=0;i<steps;i++)
      {
	call_ext_fct(edf, 2, y, 2, ynew);
	for(j=0;j<2;j++)
	  y[j] = ynew[j];
      }
    y[0] + y[1] >>= f;
  trace_off(1);
  gradient(tag_part,2,conp,grad);
  
  printf(" taping with external function facility:\n gradient=( %f, %f)\n\n",grad[0],grad[1]);

  return 0;
}

void euler_step_act(int n, adouble *yin, int m, adouble *yout)
{

 // Euler step, adouble version
 
 yout[0] = yin[0]+h*yin[0];
 yout[1] = yin[1]+h*2*yin[1];
}

int euler_step(int n, double *yin, int m, double *yout)
{

 // Euler step, double version
 yout[0] = yin[0]+h*yin[0];
 yout[1] = yin[1]+h*2*yin[1];

 return 1;
}

int zos_for_euler_step(int n, double *yin, int m, double *yout)
{
  int rc;

  set_nested_ctx(tag_ext_fct,true);
  rc = zos_forward(tag_ext_fct, 2, 2, 0, yin, yout);
  set_nested_ctx(tag_ext_fct,false);

  return rc;
}

int fos_rev_euler_step(int n, double *u, int m, double *z, double */* unused */, double */*unused*/)
{
  int rc;

  set_nested_ctx(tag_ext_fct,true);
  zos_forward(tag_ext_fct, 2, 2, 1, edf->dp_x, edf->dp_y);
  rc = fos_reverse(tag_ext_fct, 2, 2, u, z);
  set_nested_ctx(tag_ext_fct,false);

  return rc;
}
