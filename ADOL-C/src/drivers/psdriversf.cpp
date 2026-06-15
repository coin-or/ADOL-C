/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.cpp
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran
            callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege, Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduct ion, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/drivers/psdrivers.h>
#include <adolc/fortutils.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */
/* abs_normal(tag,m,n,s,x[n],sig[s],y[m],z[s],cz[s],cy[m],                  */
/*            J[m][s],Y[m][n],Z[s][n],L[s][s])                              */
fint abs_normal_(fint *ftag, fint *fdepen, fint *findep, fint *fswchk,
                 fdouble *fx, fdouble *fy, fdouble *fz, fdouble *fcz,
                 fdouble *fcy, fdouble *fJ, fdouble *fY, fdouble *fZ,
                 fdouble *fL) {
  int rc = -1;
  short tag = (short)*ftag;
  int m = (int)*fdepen, n = (int)*findep, s = (int)*fswchk;
  double *x;

  ADOLC::AbsNormalForm anf(
      {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(s)});

  x = myalloc1(n);
  spread1(n, fx, x);
  rc = ADOLC::abs_normal(tag, x, anf);
  pack1(m, anf.y.data(), fy);
  pack1(s, anf.z.data(), fz);
  pack1(s, anf.cz.data(), fcz);
  pack1(m, anf.cy.data(), fcy);

  pack2(m, n, anf.Y.data(), fY);
  pack2(m, s, anf.J.data(), fJ);
  pack2(s, n, anf.Z.data(), fZ);
  pack2(s, s, anf.L.data(), fL);
  myfree1(x);
  return rc;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
