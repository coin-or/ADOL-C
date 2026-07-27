#include <adolc/adolc.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  //! [higher-order]
  const short tapeId = createNewTape();

  trace_on(tapeId);
  {
    adouble x;
    double value = 0.0;
    x <<= 2.0;
    adouble y = x * x * x * x;
    y >>= value;
  }
  trace_off();

  constexpr int degree = 3;
  const std::array<double, 1> point{2.0};

  // X stores the coefficients of x(t) = 2 + t.
  const std::array<double, degree> inputTaylor{1.0, 0.0, 0.0};
  const double *inputRows[]{inputTaylor.data()};
  std::array<double, 1> value{};
  std::array<double, degree> outputTaylor{};
  double *outputRows[]{outputTaylor.data()};

  // keep=degree saves the Taylor data needed by the following reverse sweep.
  const int forwardStatus =
      hos_forward(tapeId, 1, 1, degree, degree, point.data(), inputRows,
                  value.data(), outputRows);

  // outputTaylor[k-1] = f^(k)(2) / k! for x(t) = 2 + t.
  const std::array<double, degree> forwardDerivatives{
      outputTaylor[0], 2.0 * outputTaylor[1], 6.0 * outputTaylor[2]};

  const std::array<double, 1> weight{1.0};
  constexpr int reverseDegree = degree - 1;
  std::array<double, reverseDegree + 1> gradientTaylor{};
  double *gradientRows[]{gradientTaylor.data()};
  const int reverseStatus =
      hos_reverse(tapeId, 1, 1, reverseDegree, weight.data(), gradientRows);

  // Z[k] is f^(k+1)(x) / k! for this scalar weighted output.
  const std::array<double, degree> reverseDerivatives{
      gradientTaylor[0], gradientTaylor[1], 2.0 * gradientTaylor[2]};
  //! [higher-order]

  const std::array<double, degree> expected{32.0, 48.0, 48.0};
  if (forwardStatus < 0 || reverseStatus < 0 ||
      !(std::abs(value[0] - 16.0) <= 1e-12))
    return 1;
  for (int order = 0; order < degree; ++order)
    if (!(std::abs(forwardDerivatives[order] - expected[order]) <= 1e-12) ||
        !(std::abs(reverseDerivatives[order] - expected[order]) <= 1e-12))
      return 1;

  std::cout << std::fixed << std::setprecision(6)
            << "hos_forward derivatives = [" << forwardDerivatives[0] << ", "
            << forwardDerivatives[1] << ", " << forwardDerivatives[2] << "]\n"
            << "hos_reverse derivatives = [" << reverseDerivatives[0] << ", "
            << reverseDerivatives[1] << ", " << reverseDerivatives[2] << "]\n";
}
