#define BOOST_TEST_DYN_LINK
#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <array>
#include <vector>

#include <adolc/adolc.h>

/*
 * Tests for externally differentiated functions
 */

/**
 * Inner test function: y = f(x0, x1, x2) = x0^4 + x1^4 + x2^4 + x0 x1
 *
 * This is the function to be treated as externally differentiable.
 */
template <typename T> static int f(int n, T *x, int m, T *y) {
  assert(n == 3 && m == 1);

  y[0] = 0;
  for (int i = 0; i < n; ++i) {
    const T z = x[i] * x[i];
    y[0] += z * z;
  }
  y[0] += x[0] * x[1];

  return 0;
}

const int f_tape = 1;

static int f_nested(int n, double *x, int m, double *y) {
  trace_on(f_tape);

  std::vector<adouble> xa(n), ya(m);
  for (int i = 0; i < n; ++i)
    xa[i] <<= x[i];

  const int ret = f(xa.size(), xa.data(), ya.size(), ya.data());

  for (int i = 0; i < m; ++i)
    ya[i] >>= y[i];

  trace_off();

  return ret;
}

/**
 * Implement first derivatives by calling fos_forward
 */
static int f_fos_forward(int n, double *x, double *dx, int m, double *y,
                         double *dy) {
  const int keep = 2;
  return fos_forward(f_tape, m, n, keep, x, dx, y, dy);
}

/**
 * Implement first derivatives by hand
 */
static int f_fos_forward_manual(int n, double *x, double *dx, int m, double *y,
                                double *dy) {
  assert(n == 3 && m == 1);
  y[0] = x[0] * x[0] * x[0] * x[0] + x[1] * x[1] * x[1] * x[1] +
         x[2] * x[2] * x[2] * x[2] + x[0] * x[1];
  dy[0] = 4.0 * x[0] * x[0] * x[0] * dx[0] + 4.0 * x[1] * x[1] * x[1] * dx[1] +
          4.0 * x[2] * x[2] * x[2] * dx[2] + x[1] * dx[0] + x[0] * dx[1];
  return 0;
}

/**
 * Implement derivatives calling hos_ti_reverse
 */
static int f_hos_ti_reverse(int m, double **u, int n, int d, double **z,
                            double **x, double **y) {
  int const ret = hos_ti_reverse(f_tape, m, n, d, u, z);

  return ret;
}

/**
 * Hand-code the expected outcome of hos_ti_reverse for the particular test
 * function
 */
static int f_hos_ti_reverse_manual(int m, double **u, int n, int d, double **z,
                                   double **x, double **y) {
  assert(n == 3 && m == 1 && d == 1);

  // First column z0 of Z: z0^T = u0^T F'(x0)
  z[0][0] = u[0][0] * (4 * x[0][0] * x[0][0] * x[0][0] + x[1][0]);
  z[1][0] = u[0][0] * (4 * x[1][0] * x[1][0] * x[1][0] + x[0][0]);
  z[2][0] = u[0][0] * 4 * x[2][0] * x[2][0] * x[2][0];

  // Second column z1 of Z: u0^T (Hessian(F) x1) + u1^T F'(x0)?
  // Hessian(F) = [12 x0^2, 1, 0; 1, 12 x1^2, 0; 0, 0, 12 x2^2]
  z[0][1] = u[0][0] * (12 * x[0][0] * x[0][0] * x[0][1] + x[1][1]) +
            u[0][1] * (4 * x[0][0] * x[0][0] * x[0][0] + x[1][0]);
  z[1][1] = u[0][0] * (12 * x[1][0] * x[1][0] * x[1][1] + x[0][1]) +
            u[0][1] * (4 * x[1][0] * x[1][0] * x[1][0] + x[0][0]);
  z[2][1] = u[0][0] * 12 * x[2][0] * x[2][0] * x[2][1] +
            u[0][1] * 4 * x[2][0] * x[2][0] * x[2][0];

  return 0;
}

/* Compute y = g(x0, x1, x2) = f(x0^2, x1, x2)^2 using given implementation of f
 *
 * This is the full test function.  It adds a few operations before and after
 * the call to f, to make sure handing data to and from externally
 * differentiated functions works as expected.
 */
template <typename F> static double g(int n, const double *x, F f) {
  assert(n == 3);

  std::vector<adouble> xa(n), ya(1);
  for (int i = 0; i < xa.size(); ++i)
    xa[i] <<= x[i];

  xa[0] = xa[0] * xa[0];
  f(xa.size(), xa.data(), ya.size(), ya.data());
  ya[0] = ya[0] * ya[0];

  double y;
  ya[0] >>= y;

  return y;
}

/*
 * The Hessian of g, implemented by hand
 */
static std::array<std::array<double, 3>, 3> g_hessian(int n, const double *x) {
  assert(n == 3);

  using std::pow;
  std::array<std::array<double, 3>, 3> hessian = {
      {{240. * pow(x[0], 14) + 180. * pow(x[0], 8) * x[1] +
            112. * pow(x[0], 6) * pow(x[1], 4) +
            112. * pow(x[0], 6) * pow(x[2], 4) +
            12. * x[0] * x[0] * x[1] * x[1] + 4. * pow(x[1], 5) +
            4. * x[1] * pow(x[2], 4),
        20. * pow(x[0], 9) + 64. * pow(x[0], 7) * pow(x[1], 3) +
            8. * pow(x[0], 3) * x[1] + 20. * x[0] * pow(x[1], 4) +
            4. * x[0] * pow(x[2], 4),
        64. * pow(x[0], 7) * pow(x[2], 3) + 16. * x[0] * x[1] * pow(x[2], 3)},
       {0.,
        24. * pow(x[0], 8) * x[1] * x[1] + 2. * pow(x[0], 4) +
            40. * x[0] * x[0] * pow(x[1], 3) + 56. * pow(x[1], 6) +
            24. * x[1] * x[1] * pow(x[2], 4),
        8. * x[0] * x[0] * pow(x[2], 3) + 32. * pow(x[1], 3) * pow(x[2], 3)},
       {0., 0.,
        24. * pow(x[0], 8) * x[2] * x[2] +
            24. * x[0] * x[0] * x[1] * x[2] * x[2] +
            24. * pow(x[1], 4) * x[2] * x[2] + 56. * pow(x[2], 6)}}};
  hessian[1][0] = hessian[0][1];
  hessian[2][0] = hessian[0][2];
  hessian[2][1] = hessian[1][2];

  return hessian;
}

/* Manage test suite fixtures */
struct Fixture {
  Fixture() {
    f_ext = reg_ext_fct(f_nested);
    f_ext->zos_forward = f<double>;
    f_ext->fos_forward = f_fos_forward;
    f_ext->hos_ti_reverse = f_hos_ti_reverse;
    f_ext->nestedAdolc = 1;

    f_ext_manual = reg_ext_fct(f<double>);
    f_ext_manual->fos_forward = f_fos_forward_manual;
    f_ext_manual->hos_ti_reverse = f_hos_ti_reverse_manual;

    call_f_direct = f<adouble>;
    call_f_nested = [this](int n, adouble *xa, int m, adouble *ya) {
      return call_ext_fct(f_ext, n, xa, m, ya);
    };
    call_f_manual = [this](int n, adouble *xa, int m, adouble *ya) {
      return call_ext_fct(f_ext_manual, n, xa, m, ya);
    };
  }

  ext_diff_fct *f_ext;
  ext_diff_fct *f_ext_manual;
  std::function<int(int, adouble *, int, adouble *)> call_f_direct,
      call_f_manual, call_f_nested;
};

BOOST_FIXTURE_TEST_SUITE(trace_ext_diff, Fixture)

template <typename F> static void g_hess_vec(F f) {
  const int g_tape = 0;

  trace_on(g_tape);
  std::array<double, 3> x{3.0, 4.0, 5.0};
  const double y = g(x.size(), x.data(), f);
  trace_off();

  BOOST_TEST(y == 55920484.0);

  const auto hessian = g_hessian(x.size(), x.data());

  for (int i = 0; i < x.size(); ++i) {
    std::array<double, 3> dx{0.0, 0.0, 0.0};
    dx[i] = 1.0;
    std::array<double, 3> dy;
    const int ret = hess_vec(g_tape, x.size(), x.data(), dx.data(), dy.data());
    BOOST_TEST(ret >= 0);
    BOOST_TEST(dy == hessian[i], tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE(g_hess_vec_direct) { g_hess_vec(call_f_direct); }

BOOST_AUTO_TEST_CASE(g_hess_vec_nested) { g_hess_vec(call_f_nested); }

BOOST_AUTO_TEST_CASE(g_hess_vec_manual) { g_hess_vec(call_f_manual); }

/**
 * \tparam H The ADOL-C method to compute the Hessian (either 'hessian' or
 * 'hessian2')
 */
template <typename H, typename F> static void test_g_hessian(H hessian, F f) {
  const int g_tape = 0;

  trace_on(g_tape);
  std::array<double, 3> x{3.0, 4.0, 5.0};
  const double y = g(x.size(), x.data(), f);
  trace_off();

  BOOST_TEST(y == 55920484.0);

  const auto hess = g_hessian(x.size(), x.data());

  std::array<double, 6> h_;
  std::array<double *, 3> h{h_.data(), h_.data() + 1, h_.data() + 3};

  const int ret = hessian(g_tape, x.size(), x.data(), h.data());
  BOOST_TEST(ret >= 0);

  /* hessian() only fills lower left half, so only compare those */
  BOOST_TEST(h[0][0] == hess[0][0]);
  BOOST_TEST(h[1][0] == hess[1][0]);
  BOOST_TEST(h[1][1] == hess[1][1]);
  BOOST_TEST(h[2][0] == hess[2][0]);
  BOOST_TEST(h[2][1] == hess[2][1]);
  BOOST_TEST(h[2][2] == hess[2][2]);
}

BOOST_AUTO_TEST_CASE(test_g_hessian_direct) {
  test_g_hessian(hessian, call_f_direct);
}

BOOST_AUTO_TEST_CASE(test_g_hessian_nested) {
  test_g_hessian(hessian, call_f_nested);
}

BOOST_AUTO_TEST_CASE(test_g_hessian_manual) {
  test_g_hessian(hessian, call_f_manual);
}

BOOST_AUTO_TEST_SUITE_END()
