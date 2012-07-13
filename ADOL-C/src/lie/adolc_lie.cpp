/********************************************************************
 * File: adolc_lie.cpp
 *
 * Implementation of a C++ Wrapper for adolc_lie C-function calls
 *
 * Authors: Siquian Wang, Klaus Röbenack, Jan Winkler
 ********************************************************************/
#include <adolc/lie/adolc_lie.h>
#include <adolc/adolc.h>
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


int lie_covector(short int Tape_F, short int Tape_H, short int n, double* x0, 
                  short int d, double** result)
{
   return lie_covectorv(Tape_F, Tape_H, n, x0, d, result);
}


int lie_bracket(short int Tape_F, short int Tape_G, short int n, double* x0, 
                 short int d, double** result)
{
    return lie_bracketv(Tape_F, Tape_G, n, x0, d, result);
}
