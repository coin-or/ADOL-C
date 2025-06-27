/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/taylor.h
 Revision: $Id$
 Contents: Easy to use drivers for the evaluation of higher order derivative
           tensors and inverse/impicit function differentiation

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_DRIVERS_TAYLOR_H)
#define ADOLC_DRIVERS_TAYLOR_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                       TENSOR EVALUATIONS */

/*--------------------------------------------------------------------------*/
/* tensor_eval(tag,m,n,d,p,x[n],tensor[m][dim],S[n][p])
      with dim = ((p+d) over d) */
ADOLC_API int tensor_eval(short tag, int m, int n, int d, int p, double *x,
                          double **tensor, double **S);

/*--------------------------------------------------------------------------*/
/* inverse_tensor_eval(tag,n,d,p,x,tensor[n][dim],S[n][p])
      with dim = ((p+d) over d) */
ADOLC_API int inverse_tensor_eval(short tag, int n, int d, int p, double *x,
                                  double **tensor, double **S);

/*--------------------------------------------------------------------------*/
/*  inverse_Taylor_prop(tag,n,d,Y[n][d+1],X[n][d+1]) */
ADOLC_API int inverse_Taylor_prop(short tag, int n, int d, double **Y,
                                  double **X);

/****************************************************************************/
/*                                                  ACCESS TO TENSOR VALUES */

/*--------------------------------------------------------------------------*/
/* tensor_value(d,m,y[m],tensori[m][dim],multi[d])
      with dim = ((p+d) over d) */
ADOLC_API void tensor_value(int d, int m, double *y, double **tensor,
                            int *multi);

/*--------------------------------------------------------------------------*/
/* void** tensorsetup(m,p,d,tensorig) */
ADOLC_API void **tensorsetup(int m, int p, int d, double **tensorig);

/*--------------------------------------------------------------------------*/
/* void freetensor(m,p,d,tensor) */
ADOLC_API void freetensor(int m, int p, int d, double **tensor);

/*--------------------------------------------------------------------------*/
/* int tensor_address(d, im[d]) */
ADOLC_API int tensor_address(int d, int *im);

/****************************************************************************/
/*                                                                    UTILS */

/*--------------------------------------------------------------------------*/
/* int binomi(a,b)  ---> binomial coefficient to compute tensor dimension */
ADOLC_API long binomi(int a, int b);

/*--------------------------------------------------------------------------*/
/* jac_solv(tag,n,x,b,mode) */
ADOLC_API int jac_solv(unsigned short tag, int n, const double *x, double *b,
                       unsigned short mode);

END_C_DECLS

/****************************************************************************/
#endif
