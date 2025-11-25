
#include "../const.h"
#include <adolc/adolc.h>
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

const short tapeId = 590;
struct TapeInitializer {
  TapeInitializer() { createNewTape(tapeId); }
};

BOOST_GLOBAL_FIXTURE(TapeInitializer);

BOOST_AUTO_TEST_SUITE(SpeelpenningExample)

BOOST_AUTO_TEST_CASE(CHECK_OUT_GRADIENTS_HESSIANS) {
  setCurrentTape(tapeId);
  constexpr size_t n = 7;

  double *xp = new double[n];
  double yp = 0.0;
  adouble *x = new adouble[n];
  adouble y = 1;

  for (auto i = 0; i < n; i++)
    xp[i] = (i + 1.0) / (2.0 + i); // some initialization

  trace_on(tapeId); // tag = 1, keep = 0 by default
  for (auto i = 0; i < n; i++) {
    x[i] <<= xp[i]; // or  x <<= xp outside the loop
    y *= x[i];
  } // end for
  y >>= yp;
  delete[] x;
  trace_off();

  auto tape_stats = tapestats(tapeId); // reading of tape statistics
  BOOST_TEST(tape_stats[TapeInfos::NUM_MAX_LIVES] == 16);
  // ..... print other tape stats

  double *g = new double[n];
  gradient(tapeId, n, xp, g); // gradient evaluation

  double **H = new double *[n];
  for (auto i = 0; i < n; i++)
    H[i] = new double[i + 1];

  hessian(tapeId, n, xp, H); // H equals (n-1)g since g is
  double errg = 0;           // homogeneous of degree n-1.
  double errh = 0;

  for (auto i = 0; i < n; i++)
    errg += fabs(g[i] - yp / xp[i]); // vanishes analytically.
  for (auto i = 0; i < n; i++) {
    for (auto j = 0; j < n; j++) {
      if (i > j) // lower half of hessian
        errh += fabs(H[i][j] - g[i] / xp[j]);
    } // end for
  } // end for

  BOOST_TEST((yp - 1 / (1.0 + n)) == 0.0, tt::tolerance(tol));
  BOOST_TEST(errg == 0.0, tt::tolerance(tol));
  BOOST_TEST(errh == 0.0, tt::tolerance(tol));

  for (auto i = 0; i < n; i++)
    delete[] H[i];
  delete[] H;
  delete[] g;
  delete[] xp;
}

BOOST_AUTO_TEST_SUITE_END()
