/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     interfacesf.cpp
 Revision: $Id$
 Contents: Genuine Fortran callable C Interfaces to ADOL-C forward
           & reverse calls.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/fortutils.h>
#include <adolc/interfaces.h>

BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
fint hos_forward_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fd, const fint *fk, const fdouble *fbase,
                  const fdouble *fx, fdouble *fvalue, fdouble *fy) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), k = static_cast<int>(*fk);
  double *base = myalloc1(n);
  double *value = myalloc1(m);
  double **X = myalloc2(n, d);
  double **Y = myalloc2(m, d);
  spread1(n, fbase, base);
  spread2(n, d, fx, X);
  rc = hos_forward(tape, m, n, d, k, base, X, value, Y);
  pack2(m, d, Y, fy);
  pack1(m, value, fvalue);
  myfree2(X);
  myfree2(Y);
  myfree1(base);
  myfree1(value);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint zos_forward_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fk, const fdouble *fbase, fdouble *fvalue) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      k = static_cast<int>(*fk);
  double *base = myalloc1(n);
  double *value = myalloc1(m);
  spread1(n, fbase, base);
  rc = zos_forward(tape, m, n, k, base, value);
  pack1(m, value, fvalue);
  myfree1(base);
  myfree1(value);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_forward_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fd, const fint *fp, const fdouble *fbase,
                  const fdouble *fx, fdouble *fvalue, fdouble *fy) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), p = static_cast<int>(*fp);
  double *base = myalloc1(n);
  double *value = myalloc1(m);
  double ***X = myalloc3(n, p, d);
  double ***Y = myalloc3(m, p, d);
  spread1(n, fbase, base);
  spread3(n, p, d, fx, X);
  rc = hov_forward(tape, m, n, d, p, base, X, value, Y);
  pack3(m, p, d, Y, fy);
  pack1(m, value, fvalue);
  myfree3(X);
  myfree3(Y);
  myfree1(base);
  myfree1(value);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint fov_forward_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fp, const fdouble *fbase, const fdouble *fx,
                  fdouble *fvalue, fdouble *fy) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      p = static_cast<int>(*fp);
  double *base = myalloc1(n);
  double *value = myalloc1(m);
  double **X = myalloc2(n, p);
  double **Y = myalloc2(m, p);
  spread1(n, fbase, base);
  spread2(n, p, fx, X);
  rc = fov_forward(tape, m, n, p, base, X, value, Y);
  pack2(m, p, Y, fy);
  pack1(m, value, fvalue);
  myfree2(X);
  myfree2(Y);
  myfree1(base);
  myfree1(value);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hos_reverse_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fd, const fdouble *fu, fdouble *fz) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd);
  double **Z = myalloc2(n, d + 1);
  double *u = myalloc1(m);
  spread1(m, fu, u);
  rc = hos_reverse(tape, m, n, d, u, Z);
  pack2(n, d + 1, Z, fz);
  myfree2(Z);
  myfree1(u);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hos_ti_reverse_(ValueTape &tape, const fint *fm, const fint *fn,
                     const fint *fd, const fdouble *fu, fdouble *fz) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd);
  double **Z = myalloc2(n, d + 1);
  double **U = myalloc2(m, d + 1);
  spread2(m, d + 1, fu, U);
  rc = hos_ti_reverse(tape, m, n, d, U, Z);
  pack2(n, d + 1, Z, fz);
  myfree2(Z);
  myfree2(U);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint fos_reverse_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fdouble *fu, fdouble *fz) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn);
  double *u = myalloc1(m);
  double *Z = myalloc1(n);
  spread1(m, fu, u);
  rc = fos_reverse(tape, m, n, u, Z);
  pack1(n, Z, fz);
  myfree1(Z);
  myfree1(u);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_reverse_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fd, const fint *fq, const fdouble *fu,
                  fdouble *fz) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), q = static_cast<int>(*fq);
  double **U = myalloc2(q, m);
  double ***Z = myalloc3(q, n, d + 1);
  short **nop = 0;
  spread2(q, m, fu, U);
  rc = hov_reverse(tape, m, n, d, q, U, Z, nop);
  pack3(q, n, d + 1, Z, fz);
  myfree3(Z);
  myfree2(U);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_ti_reverse_(ValueTape &tape, const fint *fm, const fint *fn,
                     const fint *fd, const fint *fq, const fdouble *fu,
                     fdouble *fz) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), q = static_cast<int>(*fq);
  double ***U = myalloc3(q, m, d + 1);
  double ***Z = myalloc3(q, n, d + 1);
  short **nop = 0;
  spread3(q, m, d + 1, fu, U);
  rc = hov_ti_reverse(tape, m, n, d, q, U, Z, nop);
  pack3(q, n, d + 1, Z, fz);
  myfree3(Z);
  myfree3(U);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint fov_reverse_(ValueTape &tape, const fint *fm, const fint *fn,
                  const fint *fq, const fdouble *fu, fdouble *fz) {
  int rc = -1;
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      q = static_cast<int>(*fq);
  double **U = myalloc2(q, m);
  double **Z = myalloc2(q, n);
  spread2(q, m, fu, U);
  rc = fov_reverse(tape, m, n, q, U, Z);
  pack2(q, n, Z, fz);
  myfree2(Z);
  myfree2(U);
  return rc;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
