#include <adolc/adolc.h>
#include <iostream>

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

void printMatrix(std::string_view description, std::vector<double *> matrix,
                 size_t dimx, size_t dimy) {

  std::cout << description << " \n";
  for (int i = 0; i < dimx; ++i) {
    for (int j = 0; j < dimy; ++j) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}
void computerAbsNormal(ADProblem &problem) {
  std::vector<double> z(problem.numSwitches);
  std::vector<double> cz(problem.numSwitches);
  std::vector<double> cy(problem.numSwitches);

  std::vector<double> Y_storage(ADProblem::dimOut * ADProblem::dimIn);
  std::vector<double *> Y(ADProblem::dimOut);
  for (int i = 0; i < ADProblem::dimOut; ++i) {
    Y[i] = Y_storage.data() + i * ADProblem::dimIn;
  }
  std::vector<double> J_storage(ADProblem::dimOut * problem.numSwitches);
  std::vector<double *> J(ADProblem::dimOut);
  for (int i = 0; i < ADProblem::dimOut; ++i) {
    J[i] = J_storage.data() + i * problem.numSwitches;
  }

  std::vector<double> Z_storage(problem.numSwitches * ADProblem::dimIn);
  std::vector<double *> Z(problem.numSwitches);
  for (int i = 0; i < problem.numSwitches; ++i) {
    Z[i] = Z_storage.data() + i * ADProblem::dimIn;
  }

  std::vector<double> L_storage(problem.numSwitches * problem.numSwitches);
  std::vector<double *> L(problem.numSwitches);
  for (int i = 0; i < problem.numSwitches; ++i) {
    L[i] = L_storage.data() + i * problem.numSwitches;
  }

  int rc = abs_normal(problem.tapeId, ADProblem::dimOut, ADProblem::dimIn,
                      problem.numSwitches, problem.x.data(), problem.y.data(),
                      z.data(), cz.data(), cy.data(), Y.data(), J.data(),
                      Z.data(), L.data());

  std::cout << "rc = " << rc << "\n";

  printMatrix("L (s x s):", L, problem.numSwitches, problem.numSwitches);
  printMatrix("Z (s x n):", Z, problem.numSwitches, ADProblem::dimIn);
  printMatrix("Y (m x n):", L, ADProblem::dimOut, ADProblem::dimIn);
  printMatrix("J (m x s):", J, ADProblem::dimOut, problem.numSwitches);
}
int main() {
  ADProblem problem{};
  taping(problem);
  computerAbsNormal(problem);
}