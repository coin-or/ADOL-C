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

#include <adolc/adolc.h>
#include <array>
#include <vector>

namespace {
struct ADProblem {
  short tapeId;
  int dim;
};

struct SparseHessData {
  unsigned int *rind{nullptr};
  unsigned int *cind{nullptr};
  double *values{nullptr};
  int nnz;

  void reset() {
    delete[] rind;
    rind = nullptr;
    delete[] cind;
    cind = nullptr;
    delete[] values;
    values = nullptr;
  }
};

template <ADProblem problem> struct CompressedHessian {
  std::vector<uint *> HP{problem.dim};
  double **Seed{nullptr};
  int p;
  std::vector<double *> Hcomp;

  ~CompressedHessian() {
    for (auto &hp : HP)
      delete[] hp;
    for (auto &hcomp : Hcomp)
      delete[] hcomp;

    for (int i = 0; i < problem.dim; i++)
      delete[] Seed[i];
    delete[] Seed;
  }
};

/***************************************************************************/
template <typename T> T feval(const std::array<T, 6> x) {
  T res = 0.5 * (x[0] - 1) * (x[0] - 1) + 0.8 * (x[1] - 2) * (x[1] - 2) +
          0.9 * (x[2] - 3) * (x[2] - 3);
  res += 5 * x[0] * x[1];
  res += cos(x[3]);
  res += sin(x[4]) * pow(x[1], 2);
  res += exp(x[5]) * x[2];
  res += sin(x[4] * x[5]);
  return res;
}

void printmat(const char *name, int m, int n, double **M) {
  printf("%s \n", name);
  for (int i = 0; i < m; i++) {
    printf("\n %d: ", i);
    for (int j = 0; j < n; j++)
      printf(" %10.4f ", M[i][j]);
  }
  printf("\n");
}
template <ADProblem problem> void taping(std::span<double, problem.dim> x) {
  trace_on(problem.tapeId);
  std::array<adouble, problem.dim> xad;

  for (auto i = 0; i < problem.dim; i++) {
    xad[i] <<= x[i];
  }

  auto fad = feval(xad);

  double f;
  fad >>= f;
  trace_off();
}

template <ADProblem problem>
void computeHess(std::span<double, problem.dim> x) {
  std::array<double *, problem.dim> H;
  for (auto &h : H)
    h = new double[problem.dim];

  hessian(problem.tapeId, problem.dim, x.data(), H.data());
  printmat("Hessian (non-sparse)", problem.dim, problem.dim, H.data());
}

template <ADOLC::Sparse::RecoveryMethod RM>
void printSparse(SparseHessData &hess) {
  if constexpr (RM == ADOLC::Sparse::RecoveryMethod::Direct)
    std::cout << "Direct recovery in sparse format:" << std::endl;

  else if constexpr (RM == ADOLC::Sparse::RecoveryMethod::Indirect)
    std::cout << "Indirect recovery in sparse format:" << std::endl;

  for (int i = 0; i < hess.nnz; i++)
    std::cout << "(" << hess.rind[i] << "," << hess.cind[i]
              << "): " << hess.values[i] << std::endl;
  std::cout << std::endl;
}

template <ADProblem problem, ADOLC::Sparse::RecoveryMethod RM>
void computeSparseHess(std::span<double, problem.dim> x) {
  auto hess = SparseHessData{};
  ADOLC::Sparse::sparse_hess<ADOLC::Sparse::ControlFlowMode::Safe, RM>(
      problem.tapeId, problem.dim, 0, x.data(), &hess.nnz, &hess.rind,
      &hess.cind, &hess.values);

  printSparse<RM>(hess);
  hess.reset();
}

template <ADProblem problem>
CompressedHessian<problem>
computeSparsityPattern(std::span<double, problem.dim> x) {
  auto cHess = CompressedHessian<problem>{};
  std::span<uint *> HP_(cHess.HP);
  ADOLC::Sparse::hess_pat<ADOLC::Sparse::ControlFlowMode::Safe>(
      problem.tapeId, problem.dim, x.data(), HP_);

  std::cout << std::endl;
  std::cout << "Sparsity pattern of Hessian: \n";
  for (int i = 0; i < problem.dim; i++) {
    std::cout << i << ": ";
    for (uint j = 1; j <= cHess.HP[i][0]; j++)
      std::cout << cHess.HP[i][j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
  return cHess;
}
template <ADProblem problem>
void computeCompressedHessian(std::span<double, problem.dim> x,
                              CompressedHessian<problem> &cHess) {
  std::span<uint *> HP_(cHess.HP);
  ADOLC::Sparse::generate_seed_hess<ADOLC::Sparse::RecoveryMethod::Direct>(
      problem.dim, HP_, &cHess.Seed, &cHess.p);

  printmat(" Seed matrix", problem.dim, cHess.p, cHess.Seed);
  std::cout << std::endl;

  cHess.Hcomp.resize(problem.dim);
  for (auto &hcomp : cHess.Hcomp)
    hcomp = new double[cHess.p];

  hess_mat(problem.tapeId, problem.dim, cHess.p, x.data(), cHess.Seed,
           cHess.Hcomp.data());

  printmat("compressed H:", problem.dim, cHess.p, cHess.Hcomp.data());
}
} // namespace

int main() {

  constexpr auto problem = ADProblem{1, 6};
  createNewTape(problem.tapeId);
  std::array<double, problem.dim> x;
  for (auto i = 0; i < problem.dim; i++) {
    x[i] = log(1.0 + i);
  }

  /* Tracing of function f(x) */
  taping<problem>(x);
  computeHess<problem>(x);
  computeSparseHess<problem, ADOLC::Sparse::RecoveryMethod::Indirect>(x);
  computeSparseHess<problem, ADOLC::Sparse::RecoveryMethod::Direct>(x);
  auto cHess = computeSparsityPattern<problem>(x);
  computeCompressedHessian<problem>(x, cHess);
}
