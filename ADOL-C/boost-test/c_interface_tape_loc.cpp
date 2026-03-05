#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "const.h"
#include <adolc/adolc.h>
#include <adolc/tape_interface.h>
#include <cmath>

extern "C" {
#include "ADOLC_TB_interface.h"
}

BOOST_AUTO_TEST_SUITE(c_interface_tape_loc)

BOOST_AUTO_TEST_CASE(LeftScalarAndOwnershipSmoke) {
  const short tape_id = createNewTape();
  setCurrentTape(tape_id);

  adolc_trace_on(tape_id, 0);

  tape_loc x = adolc_tb_new(0.0);
  tape_loc y = adolc_tb_new(0.0);
  adolc_tb_independent(x, 1.0);
  adolc_tb_independent(y, 3.0);

  tape_loc z1 = adolc_tb_d_sub(5.0, x); // x - a style coverage
  tape_loc z2 = adolc_tb_d_mul(2.0, y);
  tape_loc z = adolc_tb_add(z1, z2);

  double out = 0.0;
  adolc_tb_dependent(z, &out);
  adolc_trace_off(0);

  BOOST_TEST(out == 10.0, tt::tolerance(tol));
  BOOST_TEST(adolc_num_independent(tape_id) == 2u);
  BOOST_TEST(adolc_num_dependent(tape_id) == 1u);

  double *xv = myalloc1(2);
  double *yv = myalloc1(1);
  xv[0] = 1.0;
  xv[1] = 3.0;
  zos_forward(tape_id, 1, 2, 0, xv, yv);
  BOOST_TEST(yv[0] == 10.0, tt::tolerance(tol));

  double *xd = myalloc1(2);
  double *yd = myalloc1(1);

  xd[0] = 1.0;
  xd[1] = 0.0;
  fos_forward(tape_id, 1, 2, 0, xv, xd, yv, yd);
  BOOST_TEST(yd[0] == -1.0, tt::tolerance(tol));

  xd[0] = 0.0;
  xd[1] = 1.0;
  fos_forward(tape_id, 1, 2, 0, xv, xd, yv, yd);
  BOOST_TEST(yd[0] == 2.0, tt::tolerance(tol));

  myfree1(xv);
  myfree1(yv);
  myfree1(xd);
  myfree1(yd);

  // Freeing locations is tape-context dependent.
  setCurrentTape(tape_id);

  // Explicitly release all locations to validate ownership transfer path.
  adolc_tb_free(z);
  adolc_tb_free(z2);
  adolc_tb_free(z1);
  adolc_tb_free(y);
  adolc_tb_free(x);
}

BOOST_AUTO_TEST_SUITE_END()
