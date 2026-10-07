#include <adolc/adtl.h>

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  //! [tapeless-forward]
  adtl::setNumDir(1);

  adtl::adouble x = 2.0;
  const double direction = 1.0;
  x.setADValue(&direction);

  const adtl::adouble y = x * x + sin(x);
  const double value = y.getValue();
  const double derivative = *y.getADValue();
  //! [tapeless-forward]

  if (!(std::abs(value - (4.0 + std::sin(2.0))) <= 1e-12) ||
      !(std::abs(derivative - (4.0 + std::cos(2.0))) <= 1e-12))
    return 1;

  std::cout << std::fixed << std::setprecision(6) << "value = " << value
            << "\nderivative = " << derivative << "\n";
}
