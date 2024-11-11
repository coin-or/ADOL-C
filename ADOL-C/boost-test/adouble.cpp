#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <adolc/adolc.h>

// Check that the specialization of std::numeric_limits for adouble exists
// and does something reasonable.
BOOST_AUTO_TEST_CASE(std_numeric_limits_adouble) {
  BOOST_TEST(bool(std::numeric_limits<adouble>::is_specialized));
  BOOST_TEST(std::numeric_limits<adouble>::max() ==
             std::numeric_limits<double>::max());
}
