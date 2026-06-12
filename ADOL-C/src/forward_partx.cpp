/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     forward_partx.cpp
 Revision: $Id$
 Contents:

 Copyright (c) Andrea Walther

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/interfaces.h>
#include <vector>

BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
/*                                                                ZOS_PARTX */
/* zos_forward_partx(tag, m, mdim[n], n, x[n][d], y[m])                     */
/* (based on zos_forward)                                                   */

int zos_forward_partx(short tag, int m, int n, const int *ndim,
                      const double *const *x, double *y) {
  /* double *x0; */ /* base point */
  int i, j, ind, sum_n, rc;

  sum_n = 0;
  for (i = 0; i < n; i++)
    sum_n += ndim[i];

  /* x0 = myalloc1(sum_n); */
  std::vector<double> x0(sum_n); /* base point */

  ind = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < ndim[i]; j++) {
      x0[ind] = x[i][j];
      ind++;
    }

  rc = zos_forward(tag, m, sum_n, 0, x0.data(), y);

  /* myfree1(x0); */

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                FOS_PARTX */
/* fos_forward_partx(tag, m, n, ndim[n], x[n][][2], y[m][2])                */
/* (based on fos_forward)                                                   */

int fos_forward_partx(short tag, int m, int n, const int *ndim,
                      const double *const *const *x, double **y) {
  // double *x0;   /* base point */
  // double *xtay; /* Taylor coefficients */
  // double *y0;   /* result */
  // double *ytay; /* derivatives */
  int i, j, ind, sum_n, rc;

  sum_n = 0;
  for (i = 0; i < n; i++)
    sum_n += ndim[i];

  /* x0 = myalloc1(sum_n);
  xtay = myalloc1(sum_n);
  y0 = myalloc1(m);
  ytay = myalloc1(m); */
  std::vector<double> x0(sum_n);   /* base point */
  std::vector<double> xtay(sum_n); /* Taylor coefficients */
  std::vector<double> y0(m);       /* result */
  std::vector<double> ytay(m);     /* derivatives */

  ind = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < ndim[i]; j++) {
      x0[ind] = x[i][j][0];
      xtay[ind] = x[i][j][1];
      ind++;
    }

  rc = fos_forward(tag, m, sum_n, 0, x0, xtay, y0, ytay);

  for (i = 0; i < m; i++) {
    y[i][0] = y0[i];
    y[i][1] = ytay[i];
  }

  /* myfree1(x0);
  myfree1(xtay);
  myfree1(y0);
  myfree1(ytay); */

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                HOS_PARTX */
/* hos_forward_partx(tag, m, n, ndim[n], d, x[n][][d+1], y[m][d+1])         */
/* (based on hos_forward)                                                   */

int hos_forward_partx(short tag, int m, int n, const int *ndim, int d,
                      const double *const *const *x, double **y) {
  // double *x0;    /* base point */
  // double **xtay; /* Taylor coefficients */
  // double *y0;    /* result */
  // double **ytay; /* derivatives */
  int i, j, k, ind, sum_n, rc;

  sum_n = 0;
  for (i = 0; i < n; i++)
    sum_n += ndim[i];

  /* x0 = myalloc1(sum_n);
  xtay = myalloc2(sum_n, d);
  y0 = myalloc1(m);
  ytay = myalloc2(m, d); */
  std::vector<double> x0(sum_n); /* base point */
  Matrix<double> xtay{static_cast<size_t>(sum_n),
                      static_cast<size_t>(d)}; /* Taylor coefficients */
  std::vector<double> y0(m);                   /* result */
  Matrix<double> ytay{static_cast<size_t>(m),
                      static_cast<size_t>(d)}; /* derivatives */

  ind = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < ndim[i]; j++) {
      x0[ind] = x[i][j][0];
      for (k = 0; k < d; k++)
        xtay[ind][k] = x[i][j][k + 1];
      ind++;
    }

  rc = hos_forward(tag, m, sum_n, d, 0, x0, xtay, y0, ytay);

  for (i = 0; i < m; i++) {
    y[i][0] = y0[i];
    for (j = 0; j < d; j++)
      y[i][j + 1] = ytay[i][j];
  }

  /* myfree1(x0);
  myfree2(xtay);
  myfree1(y0);
  myfree2(ytay); */

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                FOV_PARTX */
/* fov_forward_partx(tag, m, n, ndim[n], p, x[n][], X[n][][p],
                     y[m], Y[m][p]) */
/* (based on fov_forward)                                                   */

int fov_forward_partx(short tag, int m, int n, const int *ndim, int p,
                      const double *const *x, const double *const *const *Xppp,
                      double *y, double **Ypp) {
  // double *x0; /* base point */
  // double **X; /* Taylor coefficients */
  int i, j, k, ind, sum_n, rc;

  sum_n = 0;
  for (i = 0; i < n; i++)
    sum_n += ndim[i];

  /* x0 = myalloc1(sum_n);
  X = myalloc2(sum_n, p); */
  std::vector<double> x0(sum_n); /* base point */
  Matrix<double> X{static_cast<size_t>(sum_n),
                   static_cast<size_t>(p)}; /* Taylor coefficients */

  ind = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < ndim[i]; j++) {
      x0[ind] = x[i][j];
      for (k = 0; k < p; k++)
        X[ind][k] = Xppp[i][j][k];
      ind++;
    }

  rc = fov_forward(tag, m, sum_n, p, x0.data(), X.data(), y, Ypp);

  /* myfree1(x0);
  myfree2(X); */

  return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                HOV_PARTX */
/* hov_forward_partx(tag, m, n, ndim[n], p, x[n][], X[n][][p][d],
                     y[m], Y[m][p][d]) */
/* (based on hov_forward)                                                   */

int hov_forward_partx(short tag, int m, int n, const int *ndim, int d, int p,
                      const double *const *x,
                      const double *const *const *const *Xpppp, double *y,
                      double ***Yppp) {
  // double *x0;  /* base point */
  // double ***X; /* Taylor coefficients */
  int i, j, k, l, ind, sum_n, rc;

  sum_n = 0;
  for (i = 0; i < n; i++)
    sum_n += ndim[i];

  /* x0 = myalloc1(sum_n);
  X = myalloc3(sum_n, p, d); */
  std::vector<double> x0(sum_n); /* base point */
  Tensor<double> X{static_cast<size_t>(sum_n), static_cast<size_t>(p),
                   static_cast<size_t>(d)}; /* Taylor coefficients */

  ind = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < ndim[i]; j++) {
      x0[ind] = x[i][j];
      for (k = 0; k < p; k++)
        for (l = 0; l < d; l++)
          X[ind][k][l] = Xpppp[i][j][k][l];
      ind++;
    }

  rc = hov_forward(tag, m, sum_n, d, p, x0.data(), X.data(), y, Yppp);

  /* myfree1(x0);
  myfree3(X); */

  return rc;
}

/****************************************************************************/

END_C_DECLS
