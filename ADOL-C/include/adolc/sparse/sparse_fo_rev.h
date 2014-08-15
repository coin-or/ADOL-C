/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparse_fo_rev.h
 Revision: $Id$
 Contents: This file containts some "Easy To Use" interfaces of sparse package.
 
 
 Copyright (c) Andrea Walther, Christo Mitev

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/
#if !defined (ADOLC_SPARSE_SPARSE_H)
#define ADOLC_SPARSE_SPARSE_H 1

#include <adolc/internal/common.h>

#if defined(__cplusplus)
/****************************************************************************/
/*                                           FORWARD MODE, overloaded calls */
/*                                                                          */
/* nBV = number of Boolean Vectors to be packed                             */
/*       (see Chapter Dependence Analysis, ADOL-C Documentation)            */
/* bits_per_long = 8*sizeof(unsigned long int)                              */
/* p = nBV / bits_per_long + ( (nBV % bits_per_long) != 0 )                 */
/*                                                                          */
/* For the full Jacobian matrix set                                         */
/*    p = indep / bits_per_long + ((indep % bits_per_long) != 0)            */
/* and pass a bit pattern version of the identity matrix as an argument     */
/*                                                                          */
/*--------------------------------------------------------------------------*/
/*  Bit pattern propagation call, d = 1, tight version                      */
/*                                                                          */
/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p], mode) : intfov       */

ADOLC_DLL_EXPORT int forward
(short, int, int, int, double*, unsigned long int**,
 double*, unsigned long int**, char =0);

/*--------------------------------------------------------------------------*/
/*  Bit pattern propagation call, d = 1, safe version (no x[] and y[])      */
/*                                                                          */
/* forward(tag, m, n, p, X[n][p], Y[m][p], mode) : intfov                   */

ADOLC_DLL_EXPORT int forward
(short, int, int, int, unsigned long int**, unsigned long int**, char =0);


/****************************************************************************/
/*                                           REVERSE MODE, overloaded calls */
/*                                                                          */
/* nBV = number of Boolean Vectors to be packed                             */
/*       (see Chapter Dependence Analysis, ADOL-C Documentation)            */
/* bits_per_long = 8*sizeof(unsigned long int)                              */
/* q = nBV / bits_per_long + ( (nBV % bits_per_long) != 0 )                 */
/*                                                                          */
/* For the full Jacobian matrix set                                         */
/*      q = depen / bits_per_long + ((depen % bits_per_long) != 0)          */
/* and pass a bit pattern version of the identity matrix as an argument     */
/*                                                                          */
/*--------------------------------------------------------------------------*/
/*                                                                          */
/*  Bit pattern propagation call, d = 0, tight & safe version               */
/*                                                                          */
/* reverse(tag, m, n, q, U[q][m], Z[q][n], mode) : intfov                   */

ADOLC_DLL_EXPORT int reverse
(short, int, int, int, unsigned long int**, unsigned long int**, char =0);

#endif

/****************************************************************************/

#endif
