ADOL-C offers a unit-testing capability developed using the Boost.Test library.
There are more than 400 tests to verify the basic functionality of ADOL-C, including both traceless and trace-based adouble variants.

The minimum required version of BOOST library is 1.59.0. Any older version will cause compile-time errors.
Building the ADOL-C teste requires the BOOST libraries `unit_test_framework` and `system`.
In case you are compiling BOOST on your own, be sure to add the flags `--with-test` and `--with-system`
to activate the corresponding modules.

Instructions for compiling and running the test suite with cmake:

1) Create the `build` directory inside the boost-test directory (it is ignored by git).

2) In `boost-test/build` type: `cmake ..  -DADOLC_BASE=[location of installed ADOL-C]` or `cmake-gui ..`.
   Remember to specify `ADOLC_BASE` manually when using `cmake-gui`.

3) Cmake will search for the system installed version of BOOST. If the minimum required version is not satisfied, please enter the path where an appropriate BOOST version is installed in `3RDPARTY_BOOST_DIR`.

4) Notice that ADOL-C has to be compiled with the same version of BOOST defined in 3). When using a different BOOST version than the one provided by the operating system, ADOL-C can be configured with `--with-boost` flag before compiling the ADOL-C sources.

5) After the Cmake configuration was successful, compile the test (e.g. using `make`).

6) Run the executable `./boost-test-adolc`.
