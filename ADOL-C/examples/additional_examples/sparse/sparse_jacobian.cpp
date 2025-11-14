/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse_jacobian.cpp
 Revision: $Id$
 Contents: example for computation of sparse jacobians

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#include <adolc/sparse/sparsedrivers.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <adolc/adolc.h>

/***************************************************************************/

void ceval(double *x, double *c) {
  c[0] = 2 * x[0] + x[1] - 2.0;
  c[1] = x[2] * x[2] + x[3] * x[3] - 2.0;
  c[2] = 3 * x[4] * x[5] - 3.0;
}

/***************************************************************************/

void ceval_ad(adouble *x, adouble *c) {
  c[0] = 2 * x[0] + x[1] - 2.0;
  c[0] += cos(x[3]) * sin(x[4]);
  c[1] = x[2] * x[2] + x[3] * x[3] - 2.0;
  c[2] = 3 * x[4] * x[5] - 3.0 + sin(x[4] * x[5]);
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
  constexpr size_t dimIn = 6;
  constexpr size_t dimOut = 3;
  double x[dimIn], c[dimOut];
  adouble xad[dimIn], cad[dimOut];

  int i, j;

  /****************************************************************************/
  /*******                function evaluation                   ***************/
  /****************************************************************************/

  for (i = 0; i < dimIn; i++)
    x[i] = log(1.0 + i);

  /* Tracing of function c(x) */

  trace_on(tapeId);
  for (i = 0; i < dimIn; i++)
    xad[i] <<= x[i];

  ceval_ad(xad, cad);

  for (i = 0; i < dimOut; i++)
    cad[i] >>= c[i];
  trace_off();

  printf("\n c =  ");
  for (j = 0; j < dimOut; j++)
    printf(" %e ", c[j]);
  printf("\n");

  /****************************************************************************/
  /********           For comparisons: Full Jacobian                   ********/
  /****************************************************************************/

  double **J;
  J = myalloc2(dimOut, dimIn);

  jacobian(tapeId, dimOut, dimIn, x, J);

  printmat(" J", dimOut, dimIn, J);
  printf("\n");

  /****************************************************************************/
  /*******       sparse Jacobians, complete driver              ***************/
  /****************************************************************************/

  /* coordinate format for Jacobian */
  unsigned int *rind = NULL; /* row indices    */
  unsigned int *cind = NULL; /* column indices */
  double *values = NULL;     /* values         */
  int nnz;

  ADOLC::Sparse::sparse_jac(tapeId, dimOut, dimIn, 0, x, &nnz, &rind, &cind,
                            &values);

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
  /*  same approach but using row compression                                 */
  /*--------------------------------------------------------------------------*/

  ADOLC::Sparse::sparse_jac<ADOLC::Sparse::SparseMethod::IndexDomains,
                            ADOLC::Sparse::CompressionMode::Row,
                            ADOLC::Sparse::ControlFlowMode::Safe>(
      tapeId, dimOut, dimIn, 0, x, &nnz, &rind, &cind, &values);

  printf("In sparse format (using row compression): \n");
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

  for (i = 0; i < dimIn; i++)
    x[i] = 2.0 * i;

  /*  For comparisons: Full Jacobian                                          */

  jacobian(tapeId, dimOut, dimIn, x, J);

  printmat(" J", dimOut, dimIn, J);
  printf("\n");

  /*  repeated call of sparse_jac with same sparsity pattern => repeat = 1 */

  ADOLC::Sparse::sparse_jac<ADOLC::Sparse::SparseMethod::IndexDomains,
                            ADOLC::Sparse::CompressionMode::Row,
                            ADOLC::Sparse::ControlFlowMode::Safe>(
      tapeId, dimOut, dimIn, 1, x, &nnz, &rind, &cind, &values);

  printf("In sparse format:\n");
  for (i = 0; i < nnz; i++)
    printf("%2d %2d %10.6f\n\n", rind[i], cind[i], values[i]);

  /*--------------------------------------------------------------------------*/
  /*  same approach but using row compression                                 */
  /*--------------------------------------------------------------------------*/

  ADOLC::Sparse::sparse_jac<ADOLC::Sparse::SparseMethod::IndexDomains,
                            ADOLC::Sparse::CompressionMode::Row,
                            ADOLC::Sparse::ControlFlowMode::Safe>(
      tapeId, dimOut, dimIn, 0, x, &nnz, &rind, &cind, &values);

  printf("In sparse format (using row compression): \n");
  for (i = 0; i < nnz; i++)
    printf("%2d %2d %10.6f\n\n", rind[i], cind[i], values[i]);

  constexpr int dimOutCT = 3;
  constexpr int dimInCT = 6;
  ADOLC::Sparse::sparse_jac<ADOLC::Sparse::SparseMethod::IndexDomains,
                            ADOLC::Sparse::CompressionMode::Row,
                            ADOLC::Sparse::ControlFlowMode::Safe>(
      tapeId, dimOutCT, dimInCT, 0, x, &nnz, &rind, &cind, &values);

  printf("In sparse format (using row compression): \n");
  for (i = 0; i < nnz; i++)
    printf("%2d %2d %10.6f\n\n", rind[i], cind[i], values[i]);

  free(rind);
  rind = NULL;
  free(cind);
  cind = NULL;
  free(values);
  values = NULL;
  /****************************************************************************/
  /*******       sparse Jacobians, separate drivers             ***************/
  /****************************************************************************/

  /*--------------------------------------------------------------------------*/
  /*                                                sparsity pattern Jacobian */
  /*--------------------------------------------------------------------------*/

  std::vector<uint *> JP(dimOut); /* compressed block row storage */
  std::span<uint *> JP_(JP);
  ADOLC::Sparse::jac_pat<ADOLC::Sparse::SparseMethod::IndexDomains,
                         ADOLC::Sparse::ControlFlowMode::Safe>(tapeId, dimOut,
                                                               dimIn, x, JP_);

  printf("\n");
  printf("Sparsity pattern of Jacobian: \n");
  for (i = 0; i < dimOut; i++) {
    printf(" %d: ", i);
    for (j = 1; j <= (int)JP[i][0]; j++)
      printf(" %d ", JP[i][j]);
    printf("\n");
  }
  printf("\n");

  /*--------------------------------------------------------------------------*/
  /*                                                              seed matrix */
  /*--------------------------------------------------------------------------*/

  double **Seed;
  int p;
  ADOLC::Sparse::generate_seed_jac<ADOLC::Sparse::CompressionMode::Column>(
      dimOut, dimIn, JP_, &Seed, &p);

  printf(" p_J = %d \n", p);
  printmat(" Seed matrix", dimIn, p, Seed);
  printf("\n");

  /*--------------------------------------------------------------------------*/
  /*                                                      compressed Jacobian */
  /*--------------------------------------------------------------------------*/

  double **Jcomp;
  Jcomp = myalloc2(dimOut, p);

  fov_forward(tapeId, dimOut, dimIn, p, x, Seed, c, Jcomp);
  printmat("compressed J:", dimOut, p, Jcomp);
  printf("\n");

  /*--------------------------------------------------------------------------*/
  /*  change value of x, but not the sparsity pattern                         */
  /*--------------------------------------------------------------------------*/

  for (i = 0; i < dimIn; i++)
    x[i] = 2.0 * i;

  /*  For comparisons: Full Jacobian                                          */

  jacobian(tapeId, dimOut, dimIn, x, J);

  printmat(" J", dimOut, dimIn, J);
  printf("\n");

  fov_forward(tapeId, dimOut, dimIn, p, x, Seed, c, Jcomp);
  printmat("compressed J:", dimOut, p, Jcomp);
  printf("\n");

  for (i = 0; i < dimOut; i++)
    delete[] JP[i];
  myfree2(J);

  for (i = 0; i < dimIn; i++)
    delete[] Seed[i];
  delete[] Seed;

  myfree2(Jcomp);
}
