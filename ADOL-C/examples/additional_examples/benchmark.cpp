/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     helm-auto-exam.cpp
 Revision: $Id$
 Contents: example for  Helmholtz energy example
           Computes gradient using AD driver reverse(..)

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/
#include <adolc/adolc.h>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip> // for std::setw, std::setprecision
#include <iostream>

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

template <typename T, size_t dim> struct ProblemInput {
  std::array<double, dim> bv_;
  std::array<T, dim> x_;
};

struct HelmHoltzBenchmark {
  double timeTaping;
  double timeEvaluateTape;
  double timeEvaluateFiniteDiff;
};

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

template <typename T, HelmholtzParameters params>
ProblemInput<T, params.dimIn_> prepareInput() {
  ProblemInput<T, params.dimIn_> problemInput = {
      .bv_ = std::array<double, params.dimIn_>(),
      .x_ = std::array<T, params.dimIn_>()};

  for (auto j = 0; j < params.dimIn_; ++j)
    problemInput.bv_[j] = 0.02 * (1.0 + fabs(sin(static_cast<double>(j))));

  // mark independents if adouble
  for (auto j = 0; j < params.dimIn_; ++j)
    if constexpr (std::is_same_v<adouble, T>)
      problemInput.x_[j] <<= params.r_ * sqrt(1.0 + j);
    else
      problemInput.x_[j] = params.r_ * sqrt(1.0 + j);

  return problemInput;
}

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

template <HelmholtzParameters params>
std::array<double, params.dimIn_> evaluateTape(short tapeId) {
  std::array<double, params.dimIn_> grad;
  const double weight = 1.0;
  reverse(tapeId, params.dimOut_, params.dimIn_, 0, weight, grad.data());
  return grad;
}

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

template <size_t dim>
void printResult(double result, const std::array<double, dim> &ADGrad,
                 const std::array<double, dim> &finiteDiffGrad,
                 const HelmHoltzBenchmark &benchmarkResults) {
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Helmholtz energy: " << result << std::endl;
  std::cout << "--------------------------------------------------------\n";

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

  std::cout << std::setw(6) << "Times" << std::setw(20) << "Taping"
            << std::setw(30) << "Evaluate Tape" << std::setw(30)
            << "Evaluate Finite Diff \n";

  std::cout << std::setw(6) << std::setw(20) << std::fixed
            << std::setprecision(6) << benchmarkResults.timeTaping
            << std::setw(20) << std::fixed << std::setprecision(6)
            << benchmarkResults.timeEvaluateTape << std::setw(20) << std::fixed
            << std::setprecision(6) << benchmarkResults.timeEvaluateFiniteDiff
            << std::endl;

  std::cout << "--------------------------------------------------------\n";
}

template <typename Func> double measureTime(Func &&f) {
  const auto start = std::chrono::steady_clock::now();
  f();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename Func> double benchmarkAverage(Func &&f, int repeats = 10) {
  double total = 0.0;
  for (int i = 0; i < repeats; ++i)
    total += timeInMilliseconds(f);
  return total / repeats;
}

int main() {

  constexpr size_t dimIn = 100;
  constexpr double delta = 0.000001;
  constexpr HelmholtzParameters params(0.01, 1.41421356237 /* sqrt(2.0)*/,
                                       1.3625E-3, 1.0 / dimIn, dimIn, 1);

  const short tapeId = 1;
  createNewTape(tapeId);
  const auto benchmarkResult = HelmHoltzBenchmark{
      .timeTaping = measureTime(
          [tapeId, params]() { auto result = prepareTape<params>(tapeId); }),
      .timeEvaluateTape = measureTime(
          [tapeId, params]() { auto result = evaluateTape<params>(tapeId); }),
      .timeEvaluateFiniteDiff = measureTime([delta, params]() {
        auto grad = evaluateFiniteDiff<params>(delta);
      })};

  printResult(prepareTape<params>(tapeId), evaluateTape<params>(tapeId),
              evaluateFiniteDiff<params>(delta), benchmarkResult);

  return 0;
}
