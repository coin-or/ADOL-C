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

int lie_scalar(ValueTape &tape_F, ValueTape &tape_H, short n, short m,
               double *x0, short d, double **result) {
  return lie_scalarcv(tape_F, tape_H, n, m, x0, d, result);
}

int lie_scalar(ValueTape &tape_F, ValueTape &tape_H, short n, double *x0,
               short d, double *result) {
  return lie_scalarc(tape_F, tape_H, n, x0, d, result);
}

int lie_gradient(ValueTape &tape_F, ValueTape &tape_H, short int n, short int m,
                 double *x0, short int d, double ***result) {
  return lie_gradientcv(tape_F, tape_H, n, m, x0, d, result);
}

int lie_gradient(ValueTape &tape_F, ValueTape &tape_H, short int n, double *x0,
                 short int d, double **result) {
  return lie_gradientc(tape_F, tape_H, n, x0, d, result);
}
