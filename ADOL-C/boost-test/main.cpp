#define BOOST_TEST_MODULE boost-adolc-test
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

namespace utf = boost::unit_test;

// entry point:
int main(int argc, char* argv[], char* envp[])
{
  return utf::unit_test_main( &init_unit_test, argc, argv );
}

