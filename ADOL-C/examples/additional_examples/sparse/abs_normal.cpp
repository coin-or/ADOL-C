/*
This file defines an abs-smooth example function f and computes its sparse
extended Jacobian in abs-normal form.

dimIn = 2, dimOut = 1
Input x = (x0, x1)

Let z = |x1|. Then

    f8(x) = max( -100,  3x0 + 2z,  2x0 + 5z ).

Recovered sparse entries:
(0, 0) = 1.75
(0, 2) = 3
(0, 3) = 0.25
(0, 4) = 0.5
(1, 1) = 1
(2, 0) = 3
(2, 2) = 2
(3, 0) = 0.5
(3, 2) = 4
(3, 3) = -0.5
*/

#include <adolc/adolc.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

struct ExampleProblem {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> in = {2.0, 2.0};
  static constexpr size_t dimOut = 1;
  double out = 0.0;
};

void taping(ExampleProblem &problem) {
  std::array<adouble, ExampleProblem::dimIn> x;
  adouble y;
  x[0] <<= problem.in[0];
  x[1] <<= problem.in[1];
  adouble z1 = fabs(x[1]);
  y = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
  y >>= problem.out;
}

namespace {

using ADOLC::Sparse::SparseMatrix;
using ADOLC::Sparse::detail::SparseANFBlock;
std::unordered_map<SparseANFBlock, std::string> blockString{
    {SparseANFBlock::Y, "Y"},
    {SparseANFBlock::Z, "Z"},
    {SparseANFBlock::J, "J"},
    {SparseANFBlock::L, "L"}};

void printBlock(SparseANFBlock block, const SparseMatrix &mat) {
  std::cout << blockString[block] << ": \n";
  for (const auto &entry : mat.entries()) {
    std::cout << "(" << entry.rowIndex() << ", " << entry.colIndex()
              << "): " << entry.value() << std::endl;
  }
}
} // namespace
void problem() {
  using ADOLC::Sparse::SparseANF;
  ExampleProblem example{};
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(example);
  trace_off();

  int numSwitchingVars = get_num_switches(tapeId);
  SparseANF sparseANF{};
  ADOLC::Sparse::sparse_jac(tapeId, ExampleProblem::dimOut,
                            ExampleProblem::dimIn, numSwitchingVars, 0,
                            ExampleProblem::in.data(), sparseANF);
  printBlock(SparseANFBlock::Y, sparseANF.Y);
  printBlock(SparseANFBlock::Z, sparseANF.Z);
  printBlock(SparseANFBlock::J, sparseANF.J);
  printBlock(SparseANFBlock::L, sparseANF.L);
}
//******************************************************************

int main() {
  problem();
  return 0;
}
