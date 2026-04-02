/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.cpp
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran
            callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduct ion, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/drivers/psdrivers.h>
#include <adolc/dvlparms.h>
#include <adolc/interfaces.h>
#include <adolc/internal/common.h>
#include <math.h>
#include <vector>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */

int abs_normal(short tag,       /* tape identifier */
               int m,           /* number od dependents   */
               int n,           /* number of independents */
               int swchk,       /* number of switches (check) */
               const double *x, /* base point */
               double *y,       /* function value */
               double *z,       /* switching variables */
               double *cz,      /* first constant */
               double *cy,      /* second constant */
               double **Y,      /* m times n */
               double **J,      /* m times s */
               double **Z,      /* s times n */
               double **L)      /* s times s (lowtri) */
{

  const size_t s = get_num_switches(tag);
  /* This check is required because the user is probably allocating his
   * arrays sigma, cz, Z, L, Y, J according to swchk */
  if (s != to_size_t(swchk))
    ADOLCError::fail(
        ADOLCError::ErrorType::SWITCHES_MISMATCH, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info1 = tag, .info3 = swchk, .info6 = s});

  zos_pl_forward(tag, m, n, 1, x, y, z);

  // Build an identity matrix of size (m+s)x(m+s) for the weights and fill in
  // the row pointers provided by the user to the respective results.
  // Notice that the weight row vectors are ordered [dep vars, switch vars].
  // Therefore the first m rows of the results/resultsSwitch are for Y and J,
  // while the following s rows are for Z and L.
  std::vector<double> weights_mem((m + s) * (m + s), 0.0);
  std::vector<double *> weights(m + s);
  std::vector<double *> results(m + s);
  std::vector<double *> resultsSwitch(m + s);
  for (size_t depRow = 0; depRow < static_cast<size_t>(m); depRow++) {
    weights[depRow] = weights_mem.data() + depRow * (m + s);
    weights[depRow][depRow] = 1.0;
    results[depRow] = Y[depRow];
    resultsSwitch[depRow] = J[depRow];
  }
  for (size_t switchRow = 0; switchRow < s; switchRow++) {
    weights[m + switchRow] = weights_mem.data() + (m + switchRow) * (m + s);
    weights[m + switchRow][m + switchRow] = 1.0;
    results[m + switchRow] = Z[switchRow];
    resultsSwitch[m + switchRow] = L[switchRow];
  }

  // vectorized reverse mode for all (m+s) rows at onces
  fov_pl_reverse(tag, m, n, static_cast<int>(s), m + static_cast<int>(s),
                 weights.data(), results.data(), resultsSwitch.data());

  // compute cy = y - J|z| and cz = z - L|z|.
  for (size_t depRow = 0; depRow < static_cast<size_t>(m); depRow++) {
    cy[depRow] = 0.0;
    for (size_t col = 0; col < s; col++) {
      cy[depRow] = cy[depRow] - J[depRow][col] * fabs(z[col]);
    }
  }
  for (size_t switchRow = 0; switchRow < s; switchRow++) {
    cz[switchRow] = 0.0;
    for (size_t col = 0; col < s; col++) {
      if (col < switchRow) {
        cz[switchRow] = cz[switchRow] - L[switchRow][col] * fabs(z[col]);
      }
    }
  }
  return 0;
}

/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient
 */
/*                                                                          */
int directional_active_gradient(short tag,       /* trace identifier */
                                int n,           /* number of independents */
                                const double *x, /* value of independents */
                                const double *d, /* direction */
                                double *g,     /* directional active gradient */
                                short *sigma_g /* sigma of g */
) {
  int max_dk, keep;
  double max_entry, y, by;
  double *z;
  double **E, **grad, **gradu;

  keep = 1;
  by = 1;

  const size_t s = get_num_switches(tag);

  z = myalloc1(s);

  grad = (double **)myalloc2(1, n);
  gradu = (double **)myalloc2(s, n);
  E = (double **)myalloc2(n, n);

  max_dk = 0;
  max_entry = -1;
  for (int i = 0; i < n; i++) {
    E[i][0] = d[i];
    if (max_entry < fabs(d[i])) {
      max_dk = i;
      max_entry = fabs(d[i]);
    }
  }

  int k = 1;
  bool done = 0;
  int j = 0;

  while ((k < 6) && (done == 0)) {
    fov_pl_forward(tag, 1, n, k, x, E, &y, grad, z, gradu, sigma_g);

    size_t sum = 0;
    for (size_t i = 0; i < s; i++) {
      sum += abs(sigma_g[i]);
    }

    if (sum == s) {
      zos_pl_forward(tag, 1, n, keep, x, &y, z);
      // the cast is necessary since the type signature uses "int". Its now
      // explicit.
      fos_pl_sig_reverse(tag, 1, n, static_cast<int>(s), sigma_g, &by, g);
      done = 1;
    } else {
      if (j == max_dk)
        j++;
      E[j][k] = 1;
      j++;
      k++;
    }
  }

  myfree1(z);
  myfree2(E);
  myfree2(grad);
  myfree2(gradu);

  if (done == 0)
    ADOLCError::fail(ADOLCError::ErrorType::DIRGRAD_NOT_ENOUGH_DIRS,
                     CURRENT_LOCATION);
  return 0;
}

END_C_DECLS
