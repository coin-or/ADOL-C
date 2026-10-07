#include <adolc/adolc.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  //! [taped-jacobian]
  constexpr int inputs = 2;
  constexpr int outputs = 2;
  const short tapeId = createNewTape();
  std::array<double, inputs> point{2.0, 3.0};

  trace_on(tapeId);
  {
    std::array<adouble, inputs> x;
    std::array<double, outputs> values{};
    x <<= point;
    std::array<adouble, outputs> y{x[0] * x[1], sin(x[0]) + x[1] * x[1]};
    y >>= values;
  }
  trace_off();

  std::array<std::array<double, inputs>, outputs> storage{};
  std::array<double *, outputs> rows{storage[0].data(), storage[1].data()};
  const int status =
      jacobian(tapeId, outputs, inputs, point.data(), rows.data());
  //! [taped-jacobian]

  const std::array<std::array<double, inputs>, outputs> expected{
      std::array<double, inputs>{3.0, 2.0},
      std::array<double, inputs>{std::cos(2.0), 6.0}};
  if (status < 0)
    return 1;
  for (int row = 0; row < outputs; ++row)
    for (int column = 0; column < inputs; ++column)
      if (!(std::abs(storage[row][column] - expected[row][column]) <= 1e-12))
        return 1;

  std::cout << std::fixed << std::setprecision(6) << "Jacobian row 0 = ["
            << storage[0][0] << ", " << storage[0][1] << "]\n"
            << "Jacobian row 1 = [" << storage[1][0] << ", " << storage[1][1]
            << "]\n";
}
