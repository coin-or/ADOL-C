# ADOL-C

[![Build Status](https://github.com/coin-or/ADOL-C/actions/workflows/ci.yml/badge.svg)](https://github.com/coin-or/ADOL-C/actions?query=branch%3Amaster)
[![codecov](https://codecov.io/github/coin-or/ADOL-C/graph/badge.svg?token=4FSN87ZXCZ)](https://codecov.io/github/coin-or/ADOL-C)

> [!WARNING]  
> We are in the process of modernizing ADOL-C. The master branch is unstable. Please use the latest release or help us by reporting bugs.



## Installation
Minimal compiler version: clang-13, gcc-11 or MSVC 19.31.

1. Create a build directory somewhere, and move into that directory

2. Call CMake:

     `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/you/want/to/install/in -S path/to/ADOL-C/ -B .`

3. Build and install:

     `make -j install` 


Customize the build with the following options
  `-DENABLE_MEDIPACK=1` Build ADOL-C with MeDiPack (MPI) support (default=False)

  `-DENABLE_ADVANCE_BRANCHING=1` Enable advanced branching operations to reduce retaping (default=False)

  `-DENABLE_TRACELESS_REFCOUNTING=1`   Enable reference counting for tapeless numbers (default=False)

  `-DENABLE_ACTIVITY_TRACKING=1` Enable activity tracking to reduce trace size but increased tracing time (default=False)

  `-DENABLE_HARDDEBUG=1` Enable ADOL-C hard debug mode (default=False)

  `-DENABLE_STDCZERO=0` adouble default constructor does not initialize the value to zero (default=True, use option to disable)

  `-DENABLE_DOCEXA=1` build documented examples (default=False)

  `-DENABLE_ADDEXA=1` build additional examples (default=False)

  `-DENABLE_PAREXA=1` build parallel example (requires OpenMP!) (default=False)

  `-DENABLE_TAPEDOC_VALUES=1` should the tape_doc routine compute the values as it interprets and prints the tape contents (default=False)

  `-DENABLE_BOOST_POOL=1` Enable the use of boost pool (default=False)

  `-DENABLE_SPARSE=1` Build with Colpack to enable sparse AD (default=False)

  `-DBUILD_INTERFACE=1` Build the c interface (default=False)

  `-DBUILD_TESTS=1` Build the tests (default=False)

  `-DBUILD_SHARED_LIBS=0` Build as shared library (default=True)



## Examples


### Single Tape


Here's a complete example demonstrating forward tracing and gradient evaluation:
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
