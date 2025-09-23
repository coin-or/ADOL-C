

#include <adolc/adolc.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

double analytic_f(const std::vector<double> &x) {
  // f(x0, x1) = x0^2 + sin(x1)
  return x[0] * x[0] + std::sin(x[1]);
}

std::vector<double> analytic_grad(const std::vector<double> &x) {
  return {2.0 * x[0], std::cos(x[1])};
}

void record_test_function(int tid) {
  const int N = 2;
  double dummy_vals[N] = {0.0, 0.0};
  adouble X[N];
  trace_on(tid);
  for (int i = 0; i < N; ++i) {
    X[i] <<= dummy_vals[i];
  }
  adouble Y = X[0] * X[0] + sin(X[1]);
  double out_dummy;
  Y >>= out_dummy;
  trace_off();
}

int main() {
  int nthreads = 4;
  omp_set_num_threads(nthreads);

  // 1) parallel recording -- one tape per thread
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    createNewTape(tid);
    setCurrentTape(tid);
    record_test_function(tid);
  }

  // 2) prepare testpoints
  std::vector<std::vector<double>> test_points = {
      {0.0, 0.0},  {1.0, M_PI / 2.0}, {-2.5, 0.3},
      {0.7, -1.2}, {3.14, 0.5},       {-1.1, -2.2}};
  const int N = 2;
  const double tol_rel = 1e-8;

  // 3) parallel grad evaluation
#pragma omp parallel for schedule(static)
  for (int idx = 0; idx < static_cast<int>(test_points.size()); ++idx) {
    int tid = omp_get_thread_num();
    const auto &pt = test_points[idx];
    double x_vals[2] = {pt[0], pt[1]};
    double grad_out[2] = {0.0, 0.0};

    int ret = gradient(tid, N, x_vals, grad_out);

    // analytic grad
    auto grad_true = analytic_grad(pt);

    // compare results
    for (int d = 0; d < N; ++d)
      assert(std::abs(grad_true[d] - grad_out[d]) <= 1.e-12);
  }
}
