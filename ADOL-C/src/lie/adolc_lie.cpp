/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     lie/adolc_lie.cpp
 Revision: $Id$
 Contents: Implementation of a C++ Wrapper for adolc_lie C-function calls
 

 Copyright (c) Siquian Wang, Klaus Röbenack, Jan Winkler, Mirko Franke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
  
----------------------------------------------------------------------------*/
#include <adolc/adolc.h>
#include <adolc/lie/drivers.h>


int lie_scalar(short Tape_F, short Tape_H, short n, short m, double* x0,
               short d, double** result) 
{
    return lie_scalarcv(Tape_F, Tape_H, n, m, x0, d, result);
}


int lie_scalar(short Tape_F, short Tape_H, short n, double* x0, short d,
               double* result) 
{
    return lie_scalarc(Tape_F, Tape_H, n, x0, d, result);
}


int lie_gradient(short int Tape_F, short int Tape_H, short int n, short int m, 
                  double* x0, short int d, double*** result)
{     
  return lie_gradientcv(Tape_F, Tape_H, n, m, x0, d, result);
}


int lie_gradient(short int Tape_F, short int Tape_H, short int n, double* x0, 
                  short int d, double** result)
{     
  return lie_gradientc(Tape_F, Tape_H, n, x0, d, result);
}
