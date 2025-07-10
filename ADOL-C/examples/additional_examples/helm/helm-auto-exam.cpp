/**
 * @file helm-auto-exam.cpp
 * @brief ADOL-C example: Automatic differentiation of Helmholtz free energy.
 *
 * This example demonstrates the use of
 * [ADOL-C](https://github.com/coin-or/ADOL-C) for automatic differentiation
 * (AD) of a Helmholtz energy function. It compares reverse-mode AD with
 * finite-difference approximation.
 *
 *
 * \copyright ADOL-C contributors
 */

/**
 * @defgroup HelmholtzADOLC ADOL-C Helmholtz Example
 * @brief Example that describes ADOL-C's usage
 */

#include <adolc/adolc.h>
#include <array>
#include <cmath>
#include <iomanip> // for std::setw, std::setprecision
#include <iostream>

/**
 * @brief Parameter pack for the Helmholtz energy problem.
 *
 * This structure holds all physical and numerical parameters needed
 * for evaluating and differentiating the Helmholtz energy.
 *
 * @param TE_     Temperature scaling factor
 * @param R_      Compressibility factor
 * @param coeff_  Scaling coefficient of the energy function
 * @param r_      Initial condition scaling
 * @param dimIn_  Number of input dimensions (variables)
 * @param dimOut_ Output dimension (usually 1 for scalar energy)
 */
struct HelmholtzParameters {
  double TE_;
  double R_;
  double coeff_;
  double r_;
  size_t dimIn_;
  size_t dimOut_;

  constexpr HelmholtzParameters(double TE, double R, double coeff, double r,
                                size_t dimIn, size_t dimOut)
      : TE_(TE), R_(R), r_(r), coeff_(coeff), dimIn_(dimIn), dimOut_(dimOut) {};
};

/**
 * @brief Bundles the problem input variables for energy evaluation.
 *
 * This templated structure holds both the bulk variables (`bv_`)
 * and the independent variables (`x_`), supporting both `double`
 * and `adouble` types.
 *
 * @tparam T    Either `double` or `adouble`
 * @tparam dim  Problem dimensionality
 */
template <typename T, size_t dim> struct ProblemInput {
  std::array<double, dim> bv_;
  std::array<T, dim> x_;
};

/**
 * @brief Computes the Helmholtz free energy.
 *
 * Evaluates a nonlinear Helmholtz energy function based on input `x` and
 * `bv`. Supports both AD (via `adouble`) and standard floating-point types.
 *
 * @tparam T      Type of independent variables (e.g. `double`, `adouble`)
 * @tparam params Compile-time constant parameters
 * @param x       Vector of independent variables
 * @param bv      Vector of bulk values
 * @return Energy value of type `T`
 *
 */
template <typename T, HelmholtzParameters params>
T energy(const std::array<T, params.dimIn_> &x,
         const std::array<double, params.dimIn_> &bv) {
  T he, xax, bx, tem;
  xax = 0;
  bx = 0;
  he = 0;
  for (auto i = 0; i < params.dimIn_; ++i) {
    he += x[i] * log(x[i]);
    bx += bv[i] * x[i];
    tem = (2.0 / (1.0 + i + i)) * x[i];
    for (auto j = 0; j < i; ++j)
      tem += (1.0 / (1.0 + i + j)) * x[j];
    xax += x[i] * tem;
  }
  xax *= 0.5;
  he = params.coeff_ * (he - params.TE_ * log(1.0 - bx));
  he = he -
       log((1 + bx * (1 + params.R_)) / (1 + bx * (1 - params.R_))) * xax / bx;
  return he;
}

/**
 * @brief Prepares the input vectors for energy evaluation and taping.
 *
 * Fills the `bv_` vector with problem-specific bulk data and initializes
 * the `x_` vector using the square-root scaling formula.
 *
 * If `T = adouble`, the independent variables are properly traced using `<<=`.
 *
 * @tparam T      Type of independent variables
 * @tparam params Problem parameters
 * @return Initialized `ProblemInput` structure
 */
template <typename T, HelmholtzParameters params>
ProblemInput<T, params.dimIn_> prepareInput() {
  ProblemInput<T, params.dimIn_> problemInput = {
      .bv_ = std::array<double, params.dimIn_>(),
      .x_ = std::array<T, params.dimIn_>()};

  for (auto j = 0; j < params.dimIn_; ++j)
    problemInput.bv_[j] = 0.02 * (1.0 + fabs(sin(static_cast<double>(j))));

  for (auto j = 0; j < params.dimIn_; ++j)
    if constexpr (std::is_same_v<adouble, T>)
      problemInput.x_[j] <<= params.r_ * sqrt(1.0 + j);
    else
      problemInput.x_[j] = params.r_ * sqrt(1.0 + j);

  return problemInput;
}

/**
 * @brief Creates a tape for the Helmholtz energy.
 *
 * This function marks the independent variables, computes the energy,
 * and traces the computational graph using `ADOL-C`.
 *
 * @param tapeId Identifier for the ADOL-C tape
 * @return Scalar energy result from the traced computation
 */
template <HelmholtzParameters params> double prepareTape(short tapeId) {
  trace_on(tapeId, 1);
  std::array<adouble, params.dimIn_> x;
  auto problemInput = prepareInput<adouble, params>();
  adouble he = energy<adouble, params>(problemInput.x_, problemInput.bv_);
  double result;
  he >>= result;
  trace_off();
  return result;
}

/**
 * @brief Evaluates the gradient of the Helmholtz energy using AD.
 *
 * Uses reverse mode  (`reverse`) on the previously created tape.
 *
 * @param tapeId Identifier for the ADOL-C tape
 * @return Gradient as a fixed-size array
 */
template <HelmholtzParameters params>
std::array<double, params.dimIn_> evaluateTape(short tapeId) {
  std::array<double, params.dimIn_> grad;
  const double weight = 1.0;
  reverse(tapeId, params.dimOut_, params.dimIn_, 0, weight, grad.data());
  return grad;
}

/**
 * @brief Computes the gradient using finite differences.
 *
 * Perturbs each variable by `delta` and uses forward finite differences
 * to approximate the gradient.
 *
 * @param delta Small perturbation step size
 * @return Approximated gradient
 */
template <HelmholtzParameters params>
std::array<double, params.dimIn_> evaluateFiniteDiff(double delta) {
  auto problemInput = prepareInput<double, params>();
  std::array<double, params.dimIn_> finiteDiffGrad;

  const auto baseResult =
      energy<double, params>(problemInput.x_, problemInput.bv_);

  for (auto i = 0; i < params.dimIn_; ++i) {
    problemInput.x_[i] += delta;
    auto pertubatedResult =
        energy<double, params>(problemInput.x_, problemInput.bv_);
    problemInput.x_[i] -= delta;
    finiteDiffGrad[i] = (pertubatedResult - baseResult) / delta;
  }

  return finiteDiffGrad;
}

/**
 * @brief Outputs the energy and gradients in a human-readable format.
 *
 * Displays AD gradient vs. finite-difference gradient side by side for
 * visual comparison.
 *
 * @param result          The scalar Helmholtz energy
 * @param ADGrad          Gradient computed via ADOL-C
 * @param finiteDiffGrad  Gradient computed via finite differences
 */
template <size_t dim>
void printResult(double result, const std::array<double, dim> &ADGrad,
                 const std::array<double, dim> &finiteDiffGrad) {
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Helmholtz energy: " << result << std::endl;
  std::cout << "--------------------------------------------------------\n";

  // Print header with column names
  std::cout << std::setw(6) << "Index" << std::setw(20) << "AD Gradient"
            << std::setw(30) << "Finite Difference Grad\n";
  std::cout << "--------------------------------------------------------\n";

  // Print each gradient value with its index
  for (size_t i = 0; i < dim; ++i) {
    std::cout << std::setw(6) << i << std::setw(20) << std::fixed
              << std::setprecision(6) << ADGrad[i] << std::setw(20)
              << std::fixed << std::setprecision(6) << finiteDiffGrad[i]
              << std::endl;
  }

  std::cout << "--------------------------------------------------------\n";
}

/**
 * @brief Main function to compare ADOL-C and finite difference gradients.
 *
 * This program initializes the problem, creates an AD tape,
 * evaluates the AD gradient and a finite-difference approximation,
 * and prints both side by side.
 *
 * @return Exit status (0 = success)
 */
int main() {

  constexpr size_t dimIn = 5;
  constexpr double delta = 0.000001;
  constexpr HelmholtzParameters params(0.01, 1.41421356237 /* sqrt(2.0)*/,
                                       1.3625E-3, 1.0 / dimIn, dimIn, 1);

  const short tapeId = 1;
  createNewTape(tapeId);
  printResult(prepareTape<params>(tapeId), evaluateTape<params>(tapeId),
              evaluateFiniteDiff<params>(delta));

  return 0;
}
