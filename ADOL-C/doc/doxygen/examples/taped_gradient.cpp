#include <adolc/adolc.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  //! [taped-gradient]
  const short tapeId = createNewTape();
  const std::array<double, 2> point{2.0, 3.0};

  trace_on(tapeId);
  {
    std::array<adouble, 2> x;
    double value = 0.0;
    x[0] <<= point[0];
    x[1] <<= point[1];
    adouble y = x[0] * x[1] + sin(x[0]);
    y >>= value;
  }
  trace_off();

  std::array<double, 2> gradientValue{};
  const int status = gradient(tapeId, static_cast<int>(point.size()),
                              point.data(), gradientValue.data());
  //! [taped-gradient]

  const std::array<double, 2> expected{3.0 + std::cos(2.0), 2.0};
  if (status < 0 || !(std::abs(gradientValue[0] - expected[0]) <= 1e-12) ||
      !(std::abs(gradientValue[1] - expected[1]) <= 1e-12))
    return 1;

  std::cout << std::fixed << std::setprecision(6) << "gradient = ["
            << gradientValue[0] << ", " << gradientValue[1] << "]\n";
}
