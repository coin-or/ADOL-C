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
#include <vector>

namespace ADOLC::Sparse {

/****************************************************************************/
/*                                    Bit pattern propagation; general call */
/*                                                                          */
int forward(short tag, int m, int n, int p, double *x,
            std::vector<bitword_t *> &X, double *y, std::vector<bitword_t *> &Y,
            char mode)
/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p], mode)                */
{
  int rc = -1;
  if (mode == 1) // tight version
    if (x != NULL)
      rc = int_forward_tight(tag, m, n, p, x, X.data(), y, Y.data());
    else
      ADOLCError::fail(ADOLCError::ErrorType::SPARSE_NO_BP, CURRENT_LOCATION);

  else if (mode == 0) // safe version
    rc = int_forward_safe(tag, m, n, p, X.data(), Y.data());
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);

  return (rc);
}

/****************************************************************************/
/*                                    Bit pattern propagation; no basepoint */
/*                                                                          */
int forward(short tag, int m, int n, int p, std::vector<bitword_t *> &X,
            std::vector<bitword_t *> &Y, char mode)
/* forward(tag, m, n, p, X[n][p], Y[m][p], mode)                            */
{
  if (mode != 0) // not safe
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);

  return int_forward_safe(tag, m, n, p, X.data(), Y.data());
}

/****************************************************************************/
/*                                                                          */
/*                                    Bit pattern propagation, general call */
/*                                                                          */
int reverse(short tag, int m, int n, int q, std::vector<bitword_t *> &U,
            std::vector<bitword_t *> &Z, char mode)
/* reverse(tag, m, n, q, U[q][m], Z[q][n]) */
{
  int rc = -1;

  /* ! use better the tight version, the safe version supports no subscripts*/

  if (mode == 0) // safe version
    rc = int_reverse_safe(tag, m, n, q, U.data(), Z.data());
  else if (mode == 1)
    rc = int_reverse_tight(tag, m, n, q, U.data(), Z.data());
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);

  return rc;
}

/****************************************************************************/

} // namespace ADOLC::Sparse