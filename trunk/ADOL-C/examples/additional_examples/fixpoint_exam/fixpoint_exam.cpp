/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fixpoint_exam.cpp
 Revision: $Id$
 Contents: example for differentiation of fixpoint iterations

 Copyright (c) Andrea Walther
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#include <math.h>
#include <adolc/adolc.h>

#define imax 100
#define imax_deriv 100

// fixpoint iteration

// double version 
int fixpoint_iter(double *x, double *u, double *y, int dim_x, int dim_u);
double norm(double *x, int dim_x);

// adouble version 
int fixpoint_iter_act(adouble *x, adouble *u, adouble *y, int dim_x, int dim_u);
adouble norm(adouble *x, int dim_x);


int tag_full, tag_part, tag_fixpoint;


int main()
{
  adouble x[1];
  adouble u[1];
  adouble y[1];
  adouble diff[1];
  double eps=1.0e-6;
  
  double up[1];
  double yp[1];
  double grad[1];

  int i;
  
  tag_full = 1;
  tag_part = 2;
  tag_fixpoint = 3;

  trace_on(tag_full);
    i = 0;
    u[0] <<= 0.5;
    y[0] = 1.57079632679;

    do
      {
	i++;
	x[0] = y[0];
        fixpoint_iter_act(x,u,y,1,1);
	printf(" i = %3d y = %12.9f\n",i,y[0].value());
	diff[0] = x[0]-y[0];
      }
    while((norm(diff,1)>eps) && (i<imax));
    y[0] >>= yp[0];
  trace_off(1);

  up[0] = 0.5;
  gradient(tag_full,1,up,grad);

  printf("\n full taping:\n gradient = ( %f )\n",grad[0]);

  printf("\n taping with fixpoint facility:\n\n");
  trace_on(tag_part);
    i = 0;
    u[0] <<= 0.5;
    x[0] = 1.57079632679;

    fp_iteration(tag_fixpoint,fixpoint_iter,fixpoint_iter_act,norm,norm,eps,eps,imax,imax_deriv,x,u,y,1,1);

    y[0] >>= yp[0];
  trace_off(1);

  up[0] = 0.5;
  gradient(tag_part,1,up,grad);

  printf("\n=> gradient = ( %f )\n",grad[0]);
  return 0;
}

int fixpoint_iter(double *x, double *u, double *y, int dim_x, int dim_u)
{  
  y[0] = u[0]*(x[0]+cos(x[0]));
  
  return 0;
}

double norm(double *x, int dim_x)
{
  return fabs(x[0]);
}

int fixpoint_iter_act(adouble *x, adouble *u, adouble *y, int dim_x, int dim_u)
{  
  y[0] = u[0]*(x[0]+cos(x[0]));
  
  return 0;
}

adouble norm(adouble *x, int dim_x)
{
  return fabs(x[0]);
}

