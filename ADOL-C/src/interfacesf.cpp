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
#include <vector>

BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
fint hos_forward_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fd, const fint *fk, const fdouble *fbase,
                  const fdouble *fx, fdouble *fvalue, fdouble *fy) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), k = static_cast<int>(*fk);
  std::vector<double> base(n);
  std::vector<double> value(m);
  Matrix<double> X{static_cast<size_t>(n), static_cast<size_t>(d)};
  Matrix<double> Y{static_cast<size_t>(m), static_cast<size_t>(d)};
  spread1(n, fbase, base.data());
  spread2(n, d, fx, X.data());
  rc = hos_forward(tag, m, n, d, k, base, X, value, Y);
  pack2(m, d, Y.data(), fy);
  pack1(m, value.data(), fvalue);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint zos_forward_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fk, const fdouble *fbase, fdouble *fvalue) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      k = static_cast<int>(*fk);
  std::vector<double> base(n);
  std::vector<double> value(m);
  spread1(n, fbase, base.data());
  rc = zos_forward(tag, m, n, k, base.data(), value.data());
  pack1(m, value.data(), fvalue);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_forward_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fd, const fint *fp, const fdouble *fbase,
                  const fdouble *fx, fdouble *fvalue, fdouble *fy) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), p = static_cast<int>(*fp);
  std::vector<double> base(n);
  std::vector<double> value(m);
  Tensor<double> X{static_cast<size_t>(n), static_cast<size_t>(p),
                   static_cast<size_t>(d)};
  Tensor<double> Y{static_cast<size_t>(m), static_cast<size_t>(p),
                   static_cast<size_t>(d)};
  spread1(n, fbase, base.data());
  spread3(n, p, d, fx, X.data());
  rc = hov_forward(tag, m, n, d, p, base, X, value, Y);
  pack3(m, p, d, Y.data(), fy);
  pack1(m, value.data(), fvalue);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint fov_forward_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fp, const fdouble *fbase, const fdouble *fx,
                  fdouble *fvalue, fdouble *fy) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      p = static_cast<int>(*fp);
  std::vector<double> base(n);
  std::vector<double> value(m);
  Matrix<double> X{static_cast<size_t>(n), static_cast<size_t>(p)};
  Matrix<double> Y{static_cast<size_t>(m), static_cast<size_t>(p)};
  spread1(n, fbase, base.data());
  spread2(n, p, fx, X.data());
  rc = fov_forward(tag, m, n, p, base, X, value, Y);
  pack2(m, p, Y.data(), fy);
  pack1(m, value.data(), fvalue);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hos_reverse_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fd, const fdouble *fu, fdouble *fz) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd);
  Matrix<double> Z{static_cast<size_t>(n), static_cast<size_t>(d + 1)};
  std::vector<double> u(m);
  spread1(m, fu, u.data());
  rc = hos_reverse(tag, m, n, d, u, Z);
  pack2(n, d + 1, Z.data(), fz);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hos_ti_reverse_(const fint *ftag, const fint *fm, const fint *fn,
                     const fint *fd, const fdouble *fu, fdouble *fz) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd);
  Matrix<double> Z{static_cast<size_t>(n), static_cast<size_t>(d + 1)};
  Matrix<double> U{static_cast<size_t>(m), static_cast<size_t>(d + 1)};
  spread2(m, d + 1, fu, U.data());
  rc = hos_ti_reverse(tag, m, n, d, U.data(), Z.data());
  pack2(n, d + 1, Z.data(), fz);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint fos_reverse_(const fint *ftag, const fint *fm, const fint *fn,
                  const fdouble *fu, fdouble *fz) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn);
  std::vector<double> u(m);
  std::vector<double> Z(n);
  spread1(m, fu, u.data());
  rc = fos_reverse(tag, m, n, u, Z);
  pack1(n, Z.data(), fz);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_reverse_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fd, const fint *fq, const fdouble *fu,
                  fdouble *fz) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), q = static_cast<int>(*fq);
  Tensor<double> Z{static_cast<size_t>(q), static_cast<size_t>(n),
                   static_cast<size_t>(d + 1)};
  Matrix<double> U{static_cast<size_t>(q), static_cast<size_t>(m)};
  short **nop = 0;
  spread2(q, m, fu, U.data());
  rc = hov_reverse(tag, m, n, d, q, U.data(), Z.data(), nop);
  pack3(q, n, d + 1, Z.data(), fz);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_ti_reverse_(const fint *ftag, const fint *fm, const fint *fn,
                     const fint *fd, const fint *fq, const fdouble *fu,
                     fdouble *fz) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      d = static_cast<int>(*fd), q = static_cast<int>(*fq);
  Tensor<double> Z{static_cast<size_t>(q), static_cast<size_t>(n),
                   static_cast<size_t>(d + 1)};
  Tensor<double> U{static_cast<size_t>(q), static_cast<size_t>(m),
                   static_cast<size_t>(d + 1)};
  short **nop = 0;
  spread3(q, m, d + 1, fu, U.data());
  rc = hov_ti_reverse(tag, m, n, d, q, U.data(), Z.data(), nop);
  pack3(q, n, d + 1, Z.data(), fz);
  return rc;
}

/*--------------------------------------------------------------------------*/
fint fov_reverse_(const fint *ftag, const fint *fm, const fint *fn,
                  const fint *fq, const fdouble *fu, fdouble *fz) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      q = static_cast<int>(*fq);
  Matrix<double> U{static_cast<size_t>(q), static_cast<size_t>(m)};
  Matrix<double> Z{static_cast<size_t>(q), static_cast<size_t>(n)};
  spread2(q, m, fu, U.data());
  rc = fov_reverse(tag, m, n, q, U, Z);
  pack2(q, n, Z.data(), fz);
  return rc;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
