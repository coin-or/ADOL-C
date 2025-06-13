/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparse_fo_rev.cpp
 Revision: $Id$
 Contents: All "Easy To Use" C++ interfaces of SPARSE package

 Copyright (c) Andrea Walther, Christo Mitev

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h>
#include <adolc/interfaces.h>
#include <adolc/sparse/sparse_fo_rev.h>
#include <math.h>

#if defined(__cplusplus)

/****************************************************************************/
/*                                    Bit pattern propagation; general call */
/*                                                                          */
int forward(short tag, int m, int n, int p, double *x, size_t **X, double *y,
            size_t **Y, char mode)
/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p], mode)                */
{
  int rc = -1;
  if (mode == 1) // tight version
    if (x != NULL)
      rc = int_forward_tight(tag, m, n, p, x, X, y, Y);
    else
      ADOLCError::fail(ADOLCError::ErrorType::SPARSE_NO_BP, CURRENT_LOCATION);

  else if (mode == 0) // safe version
    rc = int_forward_safe(tag, m, n, p, X, Y);
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);

  return (rc);
}

/****************************************************************************/
/*                                    Bit pattern propagation; no basepoint */
/*                                                                          */
int forward(short tag, int m, int n, int p, size_t **X, size_t **Y, char mode)
/* forward(tag, m, n, p, X[n][p], Y[m][p], mode)                            */
{
  if (mode != 0) // not safe
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);

  return int_forward_safe(tag, m, n, p, X, Y);
}

/****************************************************************************/
/*                                                                          */
/*                                    Bit pattern propagation, general call */
/*                                                                          */
int reverse(short tag, int m, int n, int q, size_t **U, size_t **Z, char mode)
/* reverse(tag, m, n, q, U[q][m], Z[q][n]) */
{
  int rc = -1;

  /* ! use better the tight version, the safe version supports no subscripts*/

  if (mode == 0) // safe version
    rc = int_reverse_safe(tag, m, n, q, U, Z);
  else if (mode == 1)
    rc = int_reverse_tight(tag, m, n, q, U, Z);
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);

  return rc;
}

/****************************************************************************/

#endif
