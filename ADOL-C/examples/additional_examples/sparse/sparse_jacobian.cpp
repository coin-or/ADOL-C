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
#include <adolc/adolc.h>
#include <array>
#include <vector>

namespace {
struct ADProblem {
  short tapeId;
  int dimOut;
  int dimIn;
};

struct SparseJacData {
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

template <ADProblem problem> struct CompressedJacobian {
  std::vector<uint *> JP{problem.dimOut};
  double **Seed{nullptr};
  int p;
  std::vector<double *> Jcomp;

  ~CompressedJacobian() {
    for (auto &jp : JP)
      delete[] jp;
    for (auto &jcomp : Jcomp)
      delete[] jcomp;

    for (int i = 0; i < problem.dimIn; i++)
      delete[] Seed[i];
    delete[] Seed;
  }
};

template <typename T>
void ceval(const std::array<T, 6> &x, std::array<T, 3> &out) {
  out[0] = 2.0 * x[0] + x[1] - 2.0;
  out[0] += cos(x[3]) * sin(x[4]);
  out[1] = x[2] * x[2] + x[3] * x[3] - 2.0;
  out[2] = 3 * x[4] * x[5] - 3.0 + sin(x[4] * x[5]);
}

void printmat(const char *name, int m, int n, double **M) {
  std::cout << name << std::endl;
  for (int i = 0; i < m; i++) {
    std::cout << "\n" << i << ": ";
    for (int j = 0; j < n; j++)
      printf(" %10.4f ", M[i][j]);
  }
  std::cout << "\n";
}

template <ADProblem problem> void taping(std::span<double, problem.dimIn> x) {

  trace_on(problem.tapeId);
  std::array<adouble, problem.dimIn> xad;
  for (int i = 0; i < problem.dimIn; i++)
    xad[i] <<= x[i];

  std::array<adouble, problem.dimOut> out;
  ceval(xad, out);

  std::array<double, problem.dimOut> dummyOut{};
  for (int i = 0; i < problem.dimOut; i++)
    out[i] >>= dummyOut[i];

  trace_off();
}

template <ADProblem problem>
void computeJac(std::span<double, problem.dimIn> x) {
  std::array<double *, problem.dimOut> J;
  for (auto &j : J)
    j = new double[problem.dimIn];

  jacobian(problem.tapeId, problem.dimOut, problem.dimIn, x.data(), J.data());

  printmat(" J", problem.dimOut, problem.dimIn, J.data());
  std::cout << "\n";

  for (auto &j : J)
    delete[] j;
}

template <ADOLC::Sparse::CompressionMode CM>
void printSparseJac(SparseJacData &jac) {
  if constexpr (CM == ADOLC::Sparse::CompressionMode::Row)
    std::cout << "In sparse format (row compression):\n";
  else if constexpr (CM == ADOLC::Sparse::CompressionMode::Column)
    std::cout << "In sparse format (using row compression): \n";

  for (int i = 0; i < jac.nnz; i++)
    printf("%2d %2d %10.6f\n\n", jac.rind[i], jac.cind[i], jac.values[i]);
}

template <ADProblem problem, ADOLC::Sparse::CompressionMode CM>
void computeSparseJac(std::span<double, problem.dimIn> x) {
  auto jac = SparseJacData{};
  ADOLC::Sparse::sparse_jac<
      ADOLC::Sparse::SparseMethod::IndexDomains, CM,
      ADOLC::Sparse::ControlFlowMode::Safe,
      ADOLC::Sparse::BitPatternPropagationDirection::Auto>(
      problem.tapeId, problem.dimOut, problem.dimIn, 0, x.data(), &jac.nnz,
      &jac.rind, &jac.cind, &jac.values);
  printSparseJac<CM>(jac);
  jac.reset();
}

template <ADProblem problem>
CompressedJacobian<problem>
computeSparsityPattern(std::span<double, problem.dimIn> x) {
  auto cJac = CompressedJacobian<problem>{};
  std::span<uint *> JP_(cJac.JP);
  ADOLC::Sparse::jac_pat<ADOLC::Sparse::SparseMethod::IndexDomains,
                         ADOLC::Sparse::ControlFlowMode::Safe>(
      problem.tapeId, problem.dimOut, problem.dimIn, x.data(), JP_);

  std::cout << "\n";
  std::cout << "Sparsity pattern of Jacobian: \n";
  for (int i = 0; i < problem.dimOut; i++) {
    std::cout << i << ": ";
    for (int j = 1; j <= cJac.JP[i][0]; j++)
      std::cout << cJac.JP[i][j] << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
  return cJac;
}

template <ADProblem problem>
void computeCompressedJacobian(std::span<double, problem.dimIn> x,
                               CompressedJacobian<problem> &cJac) {

  std::span<uint *> JP_(cJac.JP);
  ADOLC::Sparse::generate_seed_jac<ADOLC::Sparse::CompressionMode::Column>(
      problem.dimOut, problem.dimIn, JP_, &cJac.Seed, &cJac.p);

  std::cout << " p_J = " << cJac.p << std::endl;
  printmat(" Seed matrix", problem.dimIn, cJac.p, cJac.Seed);
  std::cout << "\n";

  cJac.Jcomp.resize(problem.dimOut);
  for (auto &jcomp : cJac.Jcomp)
    jcomp = new double[cJac.p];

  std::array<double, problem.dimOut> out;
  fov_forward(problem.tapeId, problem.dimOut, problem.dimIn, cJac.p, x.data(),
              cJac.Seed, out.data(), cJac.Jcomp.data());
  printmat("compressed J:", problem.dimOut, cJac.p, cJac.Jcomp.data());
  std::cout << "\n";
}

} // namespace
/***************************************************************************/

int main() {
  constexpr auto problem = ADProblem{.tapeId = 1, .dimOut = 3, .dimIn = 6};
  createNewTape(problem.tapeId);

  std::array<double, problem.dimIn> x;
  for (int i = 0; i < problem.dimIn; i++)
    x[i] = log(1.0 + i);

  taping<problem>(x);
  computeJac<problem>(x);
  computeSparseJac<problem, ADOLC::Sparse::CompressionMode::Column>(x);
  computeSparseJac<problem, ADOLC::Sparse::CompressionMode::Row>(x);
  auto cJac = computeSparsityPattern<problem>(x);
  computeCompressedJacobian(x, cJac);
}
