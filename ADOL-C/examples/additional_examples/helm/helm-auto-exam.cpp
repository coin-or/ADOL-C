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

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include <array>
#include <cmath>

/****************************************************************************/
/*                                                    CONSTANTS & VARIABLES */
constexpr double TE = 0.01; /* originally 0.0 */
const double R = std::sqrt(2.0);

/****************************************************************************/
/*                                                         HELMHOLTZ ENERGY */
template <size_t dim>
adouble energy(const std::array<adouble, dim> &x,
               const std::array<double, dim> &bv) {
  adouble he, xax, bx, tem;
  xax = 0;
  bx = 0;
  he = 0;
  for (auto i = 0; i < dim; ++i) {
    he += x[i] * log(x[i]);
    bx += bv[i] * x[i];
    tem = (2.0 / (1.0 + i + i)) * x[i];
    for (auto j = 0; j < i; ++j)
      tem += (1.0 / (1.0 + i + j)) * x[j];
    xax += x[i] * tem;
  }
  xax *= 0.5;
  he = 1.3625E-3 * (he - TE * log(1.0 - bx));
  he = he - log((1 + bx * (1 + R)) / (1 + bx * (1 - R))) * xax / bx;
  return he;
}

template <size_t dim, double r, short tapeId> double prepareTape() {
  std::array<double, dim> bv;
  for (auto j = 0; j < dim; ++j)
    bv[j] = 0.02 * (1.0 + fabs(sin(static_cast<double>(j))));

  trace_on(tapeId, 1);
  std::array<adouble, dim> x;
  adouble he;
  // mark independents
  for (auto j = 0; j < dim; ++j)
    x[j] <<= r * sqrt(1.0 + j);
  he = energy(x, bv);

  double result;
  he >>= result;
  trace_off();
  return result;
}
/****************************************************************************/
/*                                                                     MAIN */
/* This program computes first order directional derivatives
   for the helmholtz energy function */
int main() {
  constexpr size_t nf = 10;
  constexpr size_t dimIn = 10 * nf;
  constexpr size_t dimOut = 1;
  constexpr double r = 1.0 / dimOut;

  const short tapeId = 1;
  createNewTape(tapeId);
  const double result = prepareTape<dimIn, r, tapeId>();

  fprintf(stdout, "%14.6E -- energy\n", result);

  /*--------------------------------------------------------------------------*/
  /* reverse computation of gradient */
  std::array<double, dimOut> grad;
  const double weight = 1.0;
  reverse(tapeId, dimOut, dimIn, 0, weight, grad.data());

  /*--------------------------------------------------------------------------*/
  for (auto l = 0; l < dimIn; l++) /* results */
    fprintf(stdout, "%3d: %14.6E,  \n", l, grad[l]);
  fprintf(stdout, "%14.6E -- energy\n", result);

  return 1;
}

/****************************************************************************/
/*                                                               THAT'S ALL */
