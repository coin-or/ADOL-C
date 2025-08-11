/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse_hessian.cpp
 Revision: $Id$
 Contents: example for computation of sparse hessians

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <adolc/adolc.h>

/***************************************************************************/

double feval(double *x) {
  double res;

  res = 0.5 * (x[0] - 1) * (x[0] - 1) + 0.8 * (x[1] - 2) * (x[1] - 2) +
        0.9 * (x[2] - 3) * (x[2] - 3);
  res += 5 * x[0] * x[1];
  res += cos(x[3]);
  res += sin(x[4]) * pow(x[1], 2);
  res += exp(x[5]) * x[2];
  res += sin(x[4] * x[5]);

  return res;
}

/***************************************************************************/

adouble feval_ad(adouble *x) {
  adouble res;

  res = 0.5 * (x[0] - 1) * (x[0] - 1) + 0.8 * (x[1] - 2) * (x[1] - 2) +
        0.9 * (x[2] - 2) * (x[2] - 2);
  res += 5 * x[0] * x[1];
  res += cos(x[3]);
  res += sin(x[4]) * x[1] * x[1];
  res += exp(x[5]) * x[2];
  res += sin(x[4] * x[5]);

  return res;
}

/***************************************************************************/

void printmat(const char *name, int m, int n, double **M) {
  int i, j;

  printf("%s \n", name);
  for (i = 0; i < m; i++) {
    printf("\n %d: ", i);
    for (j = 0; j < n; j++)
      printf(" %10.4f ", M[i][j]);
  }
  printf("\n");
}

int main() {

  const short tapeId = 1;
  createNewTape(tapeId);
  constexpr size_t dim = 6;
  double f, x[dim];
  adouble fad, xad[dim];

  int i, j;

  /****************************************************************************/
  /*******                function evaluation                   ***************/
  /****************************************************************************/

  for (i = 0; i < dim; i++)
    x[i] = log(1.0 + i);

  /* Tracing of function f(x) */

  trace_on(tapeId);
  for (i = 0; i < dim; i++)
    xad[i] <<= x[i];

  fad = feval_ad(xad);

  fad >>= f;
  trace_off();

  printf("\n f = %e\n\n\n", f);

  /****************************************************************************/
  /********           For comparisons: Full Hessian                    ********/
  /****************************************************************************/

  double **H;
  H = myalloc2(dim, dim);

  hessian(tapeId, dim, x, H);

  printmat(" H", dim, dim, H);
  printf("\n");

  /****************************************************************************/
  /*******       sparse Hessians, complete driver              ***************/
  /****************************************************************************/

  /* coordinate format for Hessian */
  unsigned int *rind = NULL;
  unsigned int *cind = NULL;
  double *values = NULL;
  int nnz;
  int options[2];

  options[0] = 0; /*                               safe mode (default) */
  options[1] = 0; /*                       indirect recovery (default) */

  ADOLC::Sparse::sparse_hess(tapeId, dim, 0, x, &nnz, &rind, &cind, &values,
                             options);

  printf("In sparse format:\n");
  for (i = 0; i < nnz; i++)
    printf("%2d %2d %10.6f\n\n", rind[i], cind[i], values[i]);

  free(rind);
  rind = NULL;
  free(cind);
  cind = NULL;
  free(values);
  values = NULL;

  options[0] = 0; /*                               safe mode (default) */
  options[1] = 1; /*                                   direct recovery */

  ADOLC::Sparse::sparse_hess(tapeId, dim, 0, x, &nnz, &rind, &cind, &values,
                             options);

  printf("In sparse format:\n");
  for (i = 0; i < nnz; i++)
    printf("%2d %2d %10.6f\n\n", rind[i], cind[i], values[i]);

  free(rind);
  rind = NULL;
  free(cind);
  cind = NULL;
  free(values);
  values = NULL;

  /*--------------------------------------------------------------------------*/
  /*  change value of x, but not the sparsity pattern                         */
  /*--------------------------------------------------------------------------*/

  for (i = 0; i < dim; i++)
    x[i] = 2.0 * i;

  /*  For comparisons: Full Hessian:                                         */

  hessian(tapeId, dim, x, H);

  printmat(" H", dim, dim, H);
  printf("\n");

  /*  repeated call of sparse_hess with same sparsity pattern => repeat = 1 */

  ADOLC::Sparse::sparse_hess(tapeId, dim, 0, x, &nnz, &rind, &cind, &values,
                             options);

  printf("In sparse format:\n");
  for (i = 0; i < nnz; i++)
    printf("%2d %2d %10.6f\n\n", rind[i], cind[i], values[i]);

  free(rind);
  rind = NULL;
  free(cind);
  cind = NULL;
  free(values);
  values = NULL;

  /****************************************************************************/
  /*******        sparse Hessians, separate drivers             ***************/
  /****************************************************************************/

  /*--------------------------------------------------------------------------*/
  /*                                                 sparsity pattern Hessian */
  /*--------------------------------------------------------------------------*/

  unsigned int **HP = NULL; /* compressed block row storage */
  int ctrl;

  HP = (unsigned int **)malloc(dim * sizeof(unsigned int *));
  ctrl = 0;

  ADOLC::Sparse::hess_pat(tapeId, dim, x, HP, ctrl);

  printf("\n");
  printf("Sparsity pattern of Hessian: \n");
  for (i = 0; i < dim; i++) {
    printf(" %d: ", i);
    for (j = 1; j <= (int)HP[i][0]; j++)
      printf(" %d ", HP[i][j]);
    printf("\n");
  }
  printf("\n");

  /*--------------------------------------------------------------------------*/
  /*                                                              seed matrix */
  /*--------------------------------------------------------------------------*/

  double **Seed;
  int p;
  int option = 1;

  /* option = 0  indirect recovery (default),
     option = 1  direct recovery                       */

  ADOLC::Sparse::generate_seed_hess(dim, HP, &Seed, &p, option);

  printmat(" Seed matrix", dim, p, Seed);
  printf("\n");

  /*--------------------------------------------------------------------------*/
  /*                                                       compressed Hessian */
  /*--------------------------------------------------------------------------*/

  double **Hcomp;
  Hcomp = myalloc2(dim, p);

  hess_mat(tapeId, dim, p, x, Seed, Hcomp);

  printmat("compressed H:", dim, p, Hcomp);
  printf("\n");

  /*--------------------------------------------------------------------------*/
  /*  change value of x, but not the sparsity pattern                         */
  /*--------------------------------------------------------------------------*/

  for (i = 0; i < dim; i++)
    x[i] = 2.0 * i;

  /*  For comparisons: Full Hessian                                           */

  hessian(tapeId, dim, x, H);

  printmat(" H", dim, dim, H);
  printf("\n");

  hess_mat(tapeId, dim, p, x, Seed, Hcomp);

  printmat("compressed H:", dim, p, Hcomp);
  printf("\n");

  for (i = 0; i < dim; i++)
    free(HP[i]);
  free(HP);

  myfree2(H);
  myfree2(Hcomp);

  for (i = 0; i < dim; i++)
    delete[] Seed[i];
  delete[] Seed;
}
