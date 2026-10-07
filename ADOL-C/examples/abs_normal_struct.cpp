#include <adolc/adolc.h>
#include <adolc/drivers/psdrivers.h>
#include <array>
#include <cmath>
#include <iostream>
#include <string_view>
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

//! [abs-normal-recording]
void taping(ADProblem &problem) {
  findTape(problem.tapeId).enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  {
    std::vector<adouble> ax(ADProblem::dimIn);
    std::vector<adouble> ay(ADProblem::dimOut);

    for (size_t i = 0; i < ADProblem::dimIn; ++i)
      ax[i] <<= problem.x[i];

    // Record the switches in a fixed order on every compiler.
    adouble abs0 = fabs(ax[0]);
    adouble abs1 = fabs(ax[1]);
    ay[0] = ax[0] + ax[1] - abs0 - abs1;
    ay[0] >>= problem.y[0];
  }
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  std::cout << "s = " << problem.numSwitches << "\n";
}
//! [abs-normal-recording]

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

bool almostEqual(double lhs, double rhs) {
  return std::fabs(lhs - rhs) <= 1.0e-12;
}

bool computeAbsNormal(ADProblem &problem) {
  //! [abs-normal-struct]
  ADOLC::AbsNormalForm anf = ADOLC::AbsNormalForm::fromTape(problem.tapeId);

  const int rc = ADOLC::abs_normal(problem.tapeId, problem.x, anf);

  std::cout << "rc = " << rc << "\n";

  printMatrix("L (s x s):", anf.L.data(), anf.shape.s, anf.shape.s);
  printMatrix("Z (s x n):", anf.Z.data(), anf.shape.s, anf.shape.n);
  printMatrix("Y (m x n):", anf.Y.data(), anf.shape.m, anf.shape.n);
  printMatrix("J (m x s):", anf.J.data(), anf.shape.m, anf.shape.s);
  //! [abs-normal-struct]

  const bool dimensionsAreCorrect =
      anf.shape.m == 1 && anf.shape.n == 2 && anf.shape.s == 2;
  if (rc != 0 || problem.numSwitches != 2 || !dimensionsAreCorrect)
    return false;

  const bool blocksAreCorrect =
      almostEqual(anf.L[0][0], 0.0) && almostEqual(anf.L[0][1], 0.0) &&
      almostEqual(anf.L[1][0], 0.0) && almostEqual(anf.L[1][1], 0.0) &&
      almostEqual(anf.Z[0][0], 1.0) && almostEqual(anf.Z[0][1], 0.0) &&
      almostEqual(anf.Z[1][0], 0.0) && almostEqual(anf.Z[1][1], 1.0) &&
      almostEqual(anf.Y[0][0], 1.0) && almostEqual(anf.Y[0][1], 1.0) &&
      almostEqual(anf.J[0][0], -1.0) && almostEqual(anf.J[0][1], -1.0);

  return blocksAreCorrect;
}

int main() {
  ADProblem problem{};
  taping(problem);
  if (!computeAbsNormal(problem)) {
    std::cerr << "abs-normal form validation failed\n";
    return 1;
  }

  std::cout << "validation passed\n";
}
