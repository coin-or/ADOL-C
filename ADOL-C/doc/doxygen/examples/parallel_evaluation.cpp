#include <adolc/adolc.h>

#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <tuple>

int main() {
  //! [parallel-evaluation]
  const short tapeId = createNewTape();

  trace_on(tapeId);
  {
    adouble x;
    double value = 0.0;
    x <<= 0.0;
    adouble y = x * x + sin(x);
    y >>= value;
  }
  trace_off();

  findTape(tapeId).setSharedMode();

  auto derivativeAt = [tapeId](double point) {
    double direction = 1.0;
    double value = 0.0;
    double derivative = 0.0;
    const int status =
        fos_forward(tapeId, 1, 1, 0, &point, &direction, &value, &derivative);
    return std::tuple{status, value, derivative};
  };

  auto first = std::async(std::launch::async, derivativeAt, 2.0);
  auto second = std::async(std::launch::async, derivativeAt, 3.0);
  const auto [firstStatus, valueAt2, derivativeAt2] = first.get();
  const auto [secondStatus, valueAt3, derivativeAt3] = second.get();
  //! [parallel-evaluation]

  if (firstStatus < 0 || secondStatus < 0 ||
      !(std::abs(valueAt2 - (4.0 + std::sin(2.0))) <= 1e-12) ||
      !(std::abs(valueAt3 - (9.0 + std::sin(3.0))) <= 1e-12) ||
      !(std::abs(derivativeAt2 - (4.0 + std::cos(2.0))) <= 1e-12) ||
      !(std::abs(derivativeAt3 - (6.0 + std::cos(3.0))) <= 1e-12))
    return 1;

  std::cout << std::fixed << std::setprecision(6) << "f'(2) = " << derivativeAt2
            << "\n"
            << "f'(3) = " << derivativeAt3 << "\n";
}
