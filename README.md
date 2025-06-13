# ADOL-C

[![Build Status](https://github.com/coin-or/ADOL-C/actions/workflows/ci.yml/badge.svg)](https://github.com/coin-or/ADOL-C/actions?query=branch%3Amaster)
[![codecov](https://codecov.io/github/coin-or/ADOL-C/graph/badge.svg?token=4FSN87ZXCZ)](https://codecov.io/github/coin-or/ADOL-C)

> [!WARNING]  
> We are in the process of modernizing ADOL-C. The master branch is unstable. Please use the latest release!

## Pre-release Examples
We're modernizing ADOL-C's tape internals, which also introduces a new user interface.
### Single Tape
If you rely on a single tape, the required changes are minimal. Here's a complete example demonstrating forward tracing and gradient evaluation:
```cpp
#include <adolc/adolc.h>
#include <array>
#include <iostream>
#include <numeric>

// Define your function using templated types
template <typename T, size_t N> T your_function(const std::array<T, N> &indep) {
  return std::accumulate(indep.begin(), indep.end(), T(0));
}

int main() {
  constexpr size_t dim = 2;
  const short tapeId = 0;

  // 1. Prepare input data
  std::array<double, dim> inputs;
  inputs.fill(2.0);
  std::array<double, 1> out;

  // 2. Explicitly create a new tape before using any adouble variables
  createNewTape(tapeId);

  // 3. Declare active variables after tape creation to avoid segmentation faults
  std::array<adouble, dim> indeps;

  // 4. Start tracing the operation sequence
  trace_on(tapeId);
  {
    for (size_t i = 0; i < dim; ++i) {
      indeps[i] <<= inputs[i]; // declare independent variable
    }

    adouble result = your_function(indeps);

    result >>= out[0]; // declare dependent variable for differentiation
  }
  trace_off(); // stop tracing

  // 5. Evaluate the gradient (∂output / ∂inputs)
  std::array<double, dim> grad;
  gradient(tapeId, dim, inputs.data(), grad.data());

  // 6. Print the resulting gradient
  std::cout << "Gradient of sum: ";
  for (double g : grad)
    std::cout << g << " ";
  std::cout << "\n";

  return 0;
}
```
## Multiple Tapes


If your application requires multiple differentiated functions, you can 
manage separate tapes using unique `tapeId`'s. 
When working with multiple tapes it is crucial to ensure that the correct tape context is active both when allocating/constructing and when deallocating/destroying `adouble`'s. 
Failing to do so can lead to undefined behavior, memory corruption, or crashes, because each `adouble` interacts with the currently selected tape.


1. **Create and select the tape before allocation**  
   Before you allocate a `adouble`'s, make sure you have created the tape and called `setCurrentTape(tapeId)`. 
   This ensures that each `adouble` constructor correctly registers itself with the intended tape.

2. **Use the array within the correct tape context**  
   While tracing or using the array in derivative computations, always have the correct tape selected via `setCurrentTape(tapeId)` 
   before any ADOL-C operations (`trace_on`, marking independents/dependents, derivative drivers, etc.).

3. **Select the tape before deallocation**  
   Before calling `delete[]` on a `adouble*`, call `setCurrentTape(tapeId)` again. 
   This ensures that the destructor for each `adouble` runs with the correct tape active, so resources are freed appropriately.

We recommend using scopes for each tape like this:
```cpp
#include <adolc/adolc.h>
#include <array>
#include <iostream>
#include <numeric>

// Function 1: sum of inputs
template <typename T, size_t N> T sum_function(const std::array<T, N> &x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

// Function 2: product of inputs
template <typename T, size_t N> T product_function(const std::array<T, N> &x) {
  T result = 1.0;
  for (const auto &xi : x)
    result *= xi;
  return result;
}

int main() {
  constexpr size_t dim = 3;
  const short sumTapeId = 1;
  const short prodTapeId = 2;

  std::array<double, dim> inputs = {1.0, 2.0, 3.0};
  std::array<double, 1> output;

  // --- Taping sum function ---
  createNewTape(sumTapeId);
  setCurrentTape(sumTapeId); // IMPORTANT
  {
    std::array<adouble, dim> x;
    trace_on(sumTapeId);
    for (size_t i = 0; i < dim; ++i)
      x[i] <<= inputs[i];

    adouble y = sum_function(x);
    y >>= output[0];
    trace_off();
  }

  // --- Taping product function ---
  createNewTape(prodTapeId);
  setCurrentTape(prodTapeId); // IMPORTANT
  {
    std::array<adouble, dim> x;
    trace_on(prodTapeId);
    for (size_t i = 0; i < dim; ++i)
      x[i] <<= inputs[i];

    adouble y = product_function(x);
    y >>= output[0];
    trace_off();
  }

  // --- Evaluate gradients ---
  std::array<double, dim> grad_sum, grad_prod;

  gradient(sumTapeId, dim, inputs.data(), grad_sum.data());
  gradient(prodTapeId, dim, inputs.data(), grad_prod.data());

  // --- Output results ---
  std::cout << "Gradient of sum: ";
  for (double g : grad_sum)
    std::cout << g << " ";
  std::cout << "\n";

  std::cout << "Gradient of product: ";
  for (double g : grad_prod)
    std::cout << g << " ";
  std::cout << "\n";

  return 0;
}
```

## Local installation using CMake

1. Create a build directory somewhere, and move into that directory

2. Call CMake:

     `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/you/want/to/install/in path/to/adolc/sources`

3. Build and install:

     `make`  
     `make install`  



## Local installation using the AutoTools

1. Type `autoreconf -fi`

2. Run configure with possibly using one of these options:

    `--prefix=PREFIX`           install library and header files in PREFIX
                              (default: `${HOME}/adolc_base`)  
 
    `--enable-sparse`           build sparse drivers [default=no]  

    `--with-openmp-flag=FLAG`   use FLAG to enable OpenMP at compile time
                              [default=none]  

    `--enable-docexa`           build documented examples [default=no]  
    `--enable-addexa`           build additional examples [default=no]  
    `--enable-parexa`           build parallel example [default=no], if yes
                              `-with-openmp-flag=FLAG` required  

    `--with-cflags=FLAGS`       use `CFLAGS=FLAGS` (default=`-g -02`)  
    `--with-cxxflags=FLAGS`     use `CXXFLAGS=FLAGS` (default=`-g -02 -std=c++11`)  

    `--with-boost=BOOST_PATH`   path to the compiled boost library, otherwise
                              the system one is chosen by default (if exists)  

3. Type `make`

4. Type `make install`

   By default, `make install` will install all the files in `${PREFIX}/lib` and
   `${PREFIX}/include`. You can specify another installation directory by using
   the `--prefix-option` in the configure call.

This procedure provides all makefiles required in the appropriate directories.
Execute `configure --help` for more details on other available option.



## Nonlocal installation

As mentioned in INSTALL one can configure the adolc package to be installed
in a different directory than `${HOME}/adolc_base` by using the `--prefix=PATH`
configure option. This is typically used for global installations. Common PATHs
are `/usr` and `/usr/local/`, and others are known to be used. Fine control
over the installation directories can be gained by supplying additional
configure options. See `./configure --help` for details.

Completing the installation by executing `make install` requires write
permissions for all target directories. Make sure to have them or the result
may be surprising otherwise.

A global installation can be helpful if many users need the library. By adding
the library's path to `/etc/ld.so.conf` the usage of `LD_LIBRARY_PATH` and the
`-L` link switch becomes unnecessary. In many cases, for instance for
`PATH=/usr/local`, the use of the `-I` directive for compiling sources becomes
unnecessary too.



## Examples

Examples must be configured to build by using the configure switches
   `--enable-docexa` or `--enable-addexa`.
They will never be installed by make install but can be found in the
appropriate example subdirectory.


## Windows Compilation with MINGW

Please refer to `INSTALL`

## Windows Compilation with Visual Studio

Please refer to the file `MSMSVisualStudio/v14/Readme_VC++.txt` for building the library and
`ADOL-C/examples/Readme_VC++.txt` for the documented examples.



## Unit tests

ADOL-C provides more than 500 unit tests to verify its basic functionality including both traceless and trace-based adouble variants. The tests are based on BOOST (version >= 1.59.0). Building the ADOL-C teste requires the BOOST libraries `unit_test_framework` and `system`. In case you are compiling BOOST on your own, be sure to add the flags `--with-test` and `--with-system` to activate the corresponding modules.  
You can build and run them as follows:  
`mkdir build && cd build`  
`cmake -S .. -B . -DBUILD_TESTS=ON`  
`make`  
`./ADOL-C/boost-test/boost-test-adolc`  

Cmake will search for the system installed version of BOOST. If the minimum required version is not satisfied, please enter the path where an appropriate BOOST version is installed in `3RDPARTY_BOOST_DIR` in the `CMakelists.txt` inside the `boost-test` folder. Notice that ADOL-C has to be compiled with the same version of BOOST as used here. When using a different BOOST version than the one provided by the operating system, ADOL-C can be configured with `--with-boost` flag before compiling the ADOL-C sources.


