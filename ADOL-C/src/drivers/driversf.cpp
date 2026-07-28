/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/driversf.cpp
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (Implementation of the Fortran callable interfaces).

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adalloc.h>
#include <adolc/drivers/drivers.h>
#include <adolc/fortutils.h>
#include <adolc/interfaces.h>

#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
fint function_(fint *ftag, fint *fm, fint *fn, fdouble *fargument,
               fdouble *fresult) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn);
  std::vector<double> argument(n);
  std::vector<double> result(m);
  spread1(n, fargument, argument.data());
  rc = function(tag, m, n, argument, result);
  pack1(m, result.data(), fresult);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
fint gradient_(fint *ftag, fint *fn, fdouble *fargument, fdouble *fresult) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int n = static_cast<int>(*fn);
  std::vector<double> argument(n);
  std::vector<double> result(n);
  spread1(n, fargument, argument.data());
  rc = gradient(tag, n, argument, result);
  pack1(n, result.data(), fresult);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
fint vec_jac_(fint *ftag, fint *fm, fint *fn, fint *frepeat, fdouble *fargument,
              const fdouble *flagrange, fdouble *frow) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn),
      repeat = static_cast<int>(*frepeat);
  std::vector<double> argument(n);
  std::vector<double> lagrange(m);
  std::vector<double> row(n);
  spread1(m, flagrange, lagrange.data());
  spread1(n, fargument, argument.data());
  rc = vec_jac(tag, m, n, repeat, argument, lagrange, row);
  pack1(n, row.data(), frow);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */
fint jacobian_(fint *ftag, fint *fdepen, fint *findep, fdouble *fargument,
               fdouble *fjac) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int depen = static_cast<int>(*fdepen), indep = static_cast<int>(*findep);
  Matrix<double> Jac{static_cast<size_t>(depen), static_cast<size_t>(indep)};
  std::vector<double> argument(indep);
  spread1(indep, fargument, argument.data());
  rc = jacobian(tag, depen, indep, argument, Jac);
  pack2(depen, indep, Jac.data(), fjac);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
fint jac_vec_(fint *ftag, fint *fm, fint *fn, fdouble *fargument,
              fdouble *ftangent, fdouble *fcolumn) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn);
  std::vector<double> argument(n);
  std::vector<double> tangent(n);
  std::vector<double> column(m);
  spread1(n, ftangent, tangent.data());
  spread1(n, fargument, argument.data());
  rc = jac_vec(tag, m, n, argument.data(), tangent.data(), column.data());
  pack1(m, column.data(), fcolumn);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
fint hess_vec_(fint *ftag, fint *fn, fdouble *fargument, fdouble *ftangent,
               fdouble *fresult) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int n = static_cast<int>(*fn);
  std::vector<double> argument(n);
  std::vector<double> tangent(n);
  std::vector<double> result(n);
  spread1(n, fargument, argument.data());
  spread1(n, ftangent, tangent.data());
  rc = hess_vec(tag, n, argument, tangent, result);
  pack1(n, result.data(), fresult);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
fint hessian_(fint *ftag, fint *fn, fdouble *fx,
              fdouble *fh) /* length of h should be n*n but the
                            upper half of this matrix remains unchanged */
{
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int n = static_cast<int>(*fn);
  Matrix<double> H{static_cast<size_t>(n)};
  std::vector<double> x(n);
  spread1(n, fx, x.data());
  rc = hessian(tag, n, x, H);
  pack2(n, n, H.data(), fh);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
fint lagra_hess_vec_(fint *ftag, fint *fm, fint *fn, fdouble *fargument,
                     fdouble *ftangent, fdouble *flagrange, fdouble *fresult) {
  int rc = -1;
  short tag = static_cast<short>(*ftag);
  int m = static_cast<int>(*fm), n = static_cast<int>(*fn);
  std::vector<double> argument(n);
  std::vector<double> tangent(n);
  std::vector<double> lagrange(m);
  std::vector<double> result(n);
  spread1(n, fargument, argument.data());
  spread1(n, ftangent, tangent.data());
  spread1(m, flagrange, lagrange.data());
  rc = lagra_hess_vec(tag, m, n, argument.data(), tangent.data(),
                      lagrange.data(), result.data());
  pack1(n, result.data(), fresult);
  return rc;
}

END_C_DECLS
