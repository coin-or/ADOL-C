#include <array>
#include <cassert>

#include <adolc/adolc.h>

const int m = 1;
const int n = 1;
const int num_points = 3;

std::array<double, num_points> xvals = {0., 1., 2.};
std::array<double, num_points> yvals = {3., 2., 4.};

int find_index(double xval) {
  if (xval < xvals[0]) {
    return -1;
  }

  if (xval >= xvals[num_points - 1]) {
    return -1;
  }

  for (int i = 0; i < num_points - 1; ++i) {
    if (xval >= xvals[i] && xval <= xvals[i + 1]) {
      return i;
    }
  }

  return -1;
}

int interpolator(short tapeId, size_t n, double *x, size_t m, double *y) {
  std::cout << "Calling interpolator()" << std::endl;
  assert(n == 1);
  assert(m == 1);

  double xval = x[0];

  const int i = find_index(xval);

  double dydx = (yvals[i + 1] - yvals[i]) / (xvals[i + 1] - xvals[i]);

  *y = (yvals[i] + dydx * (xval - xvals[i]));
  return 0;
}

int interpolator_fos_forward(short tapeId, size_t n, double *x, double *X,
                             size_t m, double *y, double *Y) {
  std::cout << "Calling interpolator_fos_forward()" << std::endl;

  assert(n == 1);
  assert(m == 1);

  double xval = x[0];

  const int i = find_index(xval);

  double dydx = (yvals[i + 1] - yvals[i]) / (xvals[i + 1] - xvals[i]);

  *y = (xvals[i] + dydx * (xval - xvals[i]));

  Y[0] = dydx * X[0];

  return 0;
}

int interpolator_fov_forward(short tapeId, size_t n, double *x, size_t p,
                             double **X, size_t m, double *y, double **Y) {
  std::cout << "Calling interpolator_fov_forward()" << std::endl;
  assert(n == 1);
  assert(m == 1);

  double xval = x[0];

  const int i = find_index(xval);

  double dydx = (yvals[i + 1] - yvals[i]) / (xvals[i + 1] - xvals[i]);

  *y = (xvals[i] + dydx * (xval - xvals[i]));

  for (int l = 0; l < p; ++l) {
    Y[0][l] = dydx * X[0][l];
  }

  return 0;
}

int main() {
  const int num_points = 2;
  short default_tag = 0;

  createNewTape(default_tag);
  setCurrentTape(default_tag);

  adouble x[m];
  adouble f[n];
  // double gradval[1];

  double xval[num_points] = {0.5, 1.25};

  double exp_fval[num_points] = {};
  double exp_jacs[num_points] = {-1., 2.};

  double fval[n];

  double jacvals[m * n] = {};
  double *jac[] = {jacvals};

  // Interpolator interpolator(default_tag, num_points, xvals, yvals);

  // Initial taping

  ext_diff_fct *edf = reg_ext_fct(0, 0, interpolator);
  edf->fos_forward = interpolator_fos_forward;
  edf->fov_forward = interpolator_fov_forward;
  trace_on(default_tag);

  x[0] <<= xval[0];

  call_ext_fct(edf, n, x, m, f);

  f[0] >>= fval[0];

  std::cout << "Interpolator(" << xval[0] << ") = " << fval[0] << std::endl;

  trace_off(default_tag);

  for (int k = 0; k < num_points; ++k) {
    std::cout << "Evaluating derivative at " << xval[k] << ": " << std::endl;

    jacobian(default_tag, m, n, xval + k, jac);

    double act_jacval = jac[0][0];
    double exp_jacval = exp_jacs[k];
    double jac_error = abs(act_jacval - exp_jacval);

    std::cout << "Expected = " << exp_jacval << ", actual = " << act_jacval
              << ", error = " << jac_error << std::endl;

    if (jac_error >= 1e-10) {
      return 1;
    }
  }
  return 0;
}
