/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/drivers.cpp
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (Implementation of the C/C++ callable interfaces).

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adalloc.h>
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>
#include <vector>

#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
int function(short tag, int m, int n, const double *argument, double *result) {
  int rc = -1;

  rc = zos_forward(tag, m, n, 0, argument, result);

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
int gradient(short tag, int n, const double *argument, double *result) {
  int rc = -1;
  double one = 1.0;

  rc = zos_forward(tag, 1, n, 1, argument, result);
  if (rc < 0)
    return rc;
  MINDEC(rc, fos_reverse(tag, 1, n, &one, result));
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
int vec_jac(short tag, int m, int n, int repeat, const double *argument,
            const double *lagrange, double *row) {
  int rc = -1;

  if (!repeat) {
    std::vector<double> y(m);
    rc = zos_forward(tag, m, n, 1, argument, y.data());
    if (rc < 0)
      return rc;
  }
  MINDEC(rc, fos_reverse(tag, m, n, lagrange, row));
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */

int jacobian(short tag, int depen, int indep, const double *argument,
             double **jacobian) {
  int rc;

  std::vector<double> result(depen);
  Matrix<double> I;

  if (indep / 2 < depen) {
    I = unitMatrix<double>(indep);
    rc = fov_forward(tag, depen, indep, indep, argument, I.data(),
                     result.data(), jacobian);
  } else {
    I = unitMatrix<double>(depen);
    rc = zos_forward(tag, depen, indep, 1, argument, result.data());
    if (rc < 0)
      return rc;
    MINDEC(rc, fov_reverse(tag, depen, indep, depen, I.data(), jacobian));
  }

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                           large_jacobian */
/* large_jacobian(tag, m, n, k, x[n], y[m], J[m][n])                        */

int large_jacobian(short tag, int depen, int indep, int runns, double *argument,
                   double *result, double **jacobian) {
  int rc, dirs, i;

  auto I = unitMatrix<double>(indep);

  if (runns > indep)
    runns = indep;
  if (runns < 1)
    runns = 1;
  dirs = indep / runns;
  if (indep % runns)
    ++dirs;
  for (i = 0; i < runns - 1; ++i) {
    rc = fov_offset_forward(tag, depen, indep, dirs, i * dirs, argument,
                            I.data(), result, jacobian);
  }
  dirs = indep - (runns - 1) * dirs;
  rc = fov_offset_forward(tag, depen, indep, dirs, indep - dirs, argument,
                          I.data(), result, jacobian);
  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  jac_vec */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
int jac_vec(short tag, int m, int n, const double *argument,
            const double *tangent, double *column) {
  int rc = -1;

  std::vector<double> y(m);

  rc = fos_forward(tag, m, n, 0, argument, tangent, y.data(), column);

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 hess_vec */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
int hess_vec(short tag, int n, const double *argument, const double *tangent,
             double *result) {
  double one = 1.0;
  return lagra_hess_vec(tag, 1, n, argument, tangent, &one, result);
}

/*--------------------------------------------------------------------------*/
/*                                                                 hess_mat */
/* hess_mat(tag, n, q, x[n], V[n][q], W[n][q])                              */
int hess_mat(short tag, int n, int q, const double *argument,
             const double *const *tangent, double **result) {
  int rc;
  int i, j;
  double y;

  Tensor<double> Xppp{static_cast<size_t>(n), static_cast<size_t>(q),
                      1}; /* matrix on right-hand side  */
  Tensor<double> Yppp{1, static_cast<size_t>(q),
                      1}; /* results of hos_wk_forward  */
  Tensor<double> Zppp{static_cast<size_t>(q), static_cast<size_t>(n),
                      2};   /* result of Up x H x XPPP */
  Matrix<double> Upp{1, 2}; /* vector on left-hand side */

  for (i = 0; i < n; ++i)
    for (j = 0; j < q; ++j)
      Xppp[i][j][0] = tangent[i][j];

  Upp[0][0] = 1;
  Upp[0][1] = 0;

  rc = hov_wk_forward(tag, 1, n, 1, 2, q, argument, Xppp.data(), &y,
                      Yppp.data());
  MINDEC(rc, hos_ov_reverse(tag, 1, n, 1, q, Upp.data(), Zppp.data()));

  for (i = 0; i < q; ++i)
    for (j = 0; j < n; ++j)
      result[j][i] = Zppp[i][j][1];

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
/* uses Hessian-vector product                                              */
int hessian(short tag, int n, const double *argument, double **hess) {
  int rc = 3;
  int i, j;
  std::vector<double> v(n);
  std::vector<double> w(n);
  for (i = 0; i < n; i++)
    v[i] = 0;
  for (i = 0; i < n; i++) {
    v[i] = 1;
    MINDEC(rc, hess_vec(tag, n, argument, v.data(), w.data()));
    if (rc < 0) {
      /* free((char *)v);
      free((char *)w); */
      return rc;
    }
    for (j = 0; j <= i; j++)
      hess[i][j] = w[j];
    v[i] = 0;
  }

  /* free((char *)v);
  free((char *)w); */
  return rc;
  /* Note that only the lower triangle of hess is filled */
}

/*--------------------------------------------------------------------------*/
/*                                                                 hessian2 */
/* hessian2(tag, n, x[n], lower triangle of H[n][n])                        */
/* uses Hessian-matrix product                                              */
int hessian2(short tag, int n, double *argument, double **hess) {
  int rc;
  int i, j;

  Tensor<double> Xppp{static_cast<size_t>(n), static_cast<size_t>(n),
                      1};   /* matrix on right-hand side  */
  std::vector<double> y(1); /* results of function evaluation */
  Tensor<double> Yppp{1, static_cast<size_t>(n),
                      1}; /* results of hov_wk_forward  */
  Tensor<double> Zppp{static_cast<size_t>(n), static_cast<size_t>(n),
                      2};   /* result of Up x H x XPPP */
  Matrix<double> Upp{1, 2}; /* vector on left-hand side */

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      Xppp[i][j][0] = 0;
    Xppp[i][i][0] = 1;
  }

  Upp[0][0] = 1;
  Upp[0][1] = 0;

  rc = hov_wk_forward(tag, 1, n, 1, 2, n, argument, Xppp.data(), y.data(),
                      Yppp.data());
  MINDEC(rc, hos_ov_reverse(tag, 1, n, 1, n, Upp.data(), Zppp.data()));

  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++)
      hess[i][j] = Zppp[i][j][1];

  return rc;
  /* Note that only the lower triangle of hess is filled */
}

/*--------------------------------------------------------------------------*/
/*                                                           lagra_hess_vec */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
int lagra_hess_vec(short tag, int m, int n, const double *argument,
                   const double *tangent, const double *lagrange,
                   double *result) {
  int rc = -1;
  int i;
  int degree = 1;
  int keep = degree + 1;

  Matrix<double> X{static_cast<size_t>(n), 2};
  std::vector<double> y(m);
  std::vector<double> y_tangent(m);

  rc = fos_forward(tag, m, n, keep, argument, tangent, y.data(),
                   y_tangent.data());

  if (rc < 0)
    return rc;

  MINDEC(rc, hos_reverse(tag, m, n, degree, lagrange, X.data()));

  for (i = 0; i < n; ++i)
    result[i] = X[i][1];

  return rc;
}

END_C_DECLS
