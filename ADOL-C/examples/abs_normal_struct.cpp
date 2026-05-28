#include <adolc/adolc.h>
#include <adolc/drivers/psdrivers.h>
#include <iostream>
#include <array>
#include <vector>

struct ADProblem {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::array<double, dimIn> x = {1.0, -2.0};
  std::array<double, dimOut> y{};

  size_t numSwitches{0};

  ADProblem() : tapeId(createNewTape()) {}
};

void taping(ADProblem &problem) {
  currentTape().enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  std::vector<adouble> ax(ADProblem::dimIn);
  std::vector<adouble> ay(ADProblem::dimOut);

  for (int i = 0; i < ADProblem::dimIn; i++)
    ax[i] <<= problem.x[i];

  ay[0] = ax[0] + ax[1] - fabs(ax[0]) - fabs(ax[1]);

  ay[0] >>= problem.y[0];
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  std::cout << "s = " << problem.numSwitches << "\n";
}

void printMatrix(std::string_view description, double *const *matrix,
                 size_t dimx, size_t dimy) {
  std::cout << description << " \n";
  for (size_t i = 0; i < dimx; ++i) {
    for (size_t j = 0; j < dimy; ++j) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}

void computeAbsNormal(ADProblem &problem) {
  // Use the new struct-based API
  ADOLC::DenseAbsNormalForm anf =
      ADOLC::DenseAbsNormalForm::fromTape(problem.tapeId);

  int rc = ADOLC::abs_normal(problem.tapeId, problem.x.data(), anf);

  std::cout << "rc = " << rc << "\n";

  printMatrix("L (s x s):", anf.L.data(), anf.s, anf.s);
  printMatrix("Z (s x n):", anf.Z.data(), anf.s, anf.n);
  printMatrix("Y (m x n):", anf.Y.data(), anf.m, anf.n);
  printMatrix("J (m x s):", anf.J.data(), anf.m, anf.s);
}

int main() {
  ADProblem problem{};
  taping(problem);
  computeAbsNormal(problem);
  return 0;
}
