/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     checkpointing.cpp
 Revision: $Id$
 Contents: example for checkpointing

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
int euler_step(int n, double *y);

// adouble version 
int euler_step_act(int n, adouble *y);

int tag_full, tag_part, tag_check;


int main()
{
  // time interval
  double t0, tf;

  // state, double and adouble version
  adouble y[2];
  int n;

  // control, double and adouble version
  adouble con[2];
  double conp[2];

  // target value;
  double f;

  //variables for derivative caluclation
  double grad[2];

  int i;
  
  // tape identifiers
  tag_full = 1;
  tag_part = 2;
  tag_check = 3;

  // two input and output variables for checkpointing function
  n = 2;

  // time interval
  t0 = 0.0;
  tf = 1.0;

  //control
  conp[0] = 1.0;
  conp[1] = 1.0;

  // basis variant: full taping of time step loop

  trace_on(tag_full);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];
 
    for(i=0;i<steps;i++)
      {
	euler_step_act(n,y);
      }
    
    y[0] + y[1] >>= f;
  trace_off(1);

  gradient(tag_full,2,conp,grad);
  
  printf(" full taping:\n gradient=( %f, %f)\n\n",grad[0],grad[1]);

  // Now using checkpointing facilities

  // define checkpointing procedure

  // generate checkpointing context => define active variante of the time step
  CP_Context cpc(euler_step_act);

  // double variante of the time step function
  cpc.setDoubleFct(euler_step);

  // number of time steps to perform
  cpc.setNumberOfSteps(steps);

  // number of checkpoint
  cpc.setNumberOfCheckpoints(5);

  // dimension of input/output
  cpc.setDimensionXY(n);
  // input vector
  cpc.setInput(y);
  // output vector
  cpc.setOutput(y);
  // tape number for checkpointing
  cpc.setTapeNumber(tag_check);
  // always retape or not ?
  cpc.setAlwaysRetaping(false);

  trace_on(tag_part);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];
  
    cpc.checkpointing();

    y[0] + y[1] >>= f;
  trace_off(1);
  gradient(tag_part,2,conp,grad);
  
  printf(" taping with checkpointing facility:\n gradient=( %f, %f)\n\n",grad[0],grad[1]);

  return 0;
}

int euler_step(int n, double *y)
{

 // Euler step, double version
 y[0] = y[0]+h*y[0];
 y[1] = y[1]+h*2*y[1];

 return 1;
}

int euler_step_act(int n, adouble *y)
{

 // Euler step, adouble version
 
 y[0] = y[0]+h*y[0];
 y[1] = y[1]+h*2*y[1];

 return 1;
}

