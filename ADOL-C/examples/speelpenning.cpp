/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     speelpenning.cpp
 Revision: $Id$
 Contents: speelpennings example, described in the manual

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adalloc.h> // For Matrix
#include <adolc/adolc.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <ostream>

/****************************************************************************/
/*                                                             EXAMPLE CODE */

/* @brief ADOL-C problem
 *
 * Contains the necessary data and methods for handling an ADOL-C optimization
 * problem.
 */
struct ADProblem {
  static constexpr size_t dimIn = 7;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::array<double, dimIn> inputs;
  std::array<double, dimOut> out;
  std::array<double, dimIn> gradient;
  Matrix<double, dimIn> hessian;

  ADProblem() : tapeId(createNewTape()) {
    // Prepare input
    for (size_t i = 0; i < dimIn; i++) {
      inputs[i] =
          (static_cast<double>(i) + 1.0) / (2.0 + static_cast<double>(i));
    }
  }
};

/* @brief Tape the function.
 */
void taping(ADProblem &problem) {
  trace_on(problem.tapeId);
  {
    std::array<adouble, ADProblem::dimIn> indeps;
    for (size_t i = 0; i < ADProblem::dimIn; i++) {
      indeps[i] <<= problem.inputs[i];
    }
    adouble result = 1.0;
    for (size_t i = 0; i < ADProblem::dimIn; i++) {
      result *= indeps[i];
    }
    result >>= problem.out[0];
  }
  trace_off();
}

void printTapeStats(ADProblem &problem) {
  auto tape_stats = tapestats(problem.tapeId); // reading of tape statistics
  std::cout << "Number of maxlives: " << tape_stats[TapeInfos::NUM_MAX_LIVES]
            << std::endl;
  std::cout << "Number of operations: " << tape_stats[TapeInfos::NUM_OPERATIONS]
            << std::endl;
  std::cout << "Number of Taylor values: " << tape_stats[TapeInfos::NUM_TAYS]
            << std::endl;
  // ..... print other tape stats
}

void computeDerivatives(ADProblem &problem) {
  gradient(problem.tapeId, problem.dimIn, problem.inputs,
           problem.gradient); // gradient evaluation
  hessian(
      problem.tapeId, problem.dimIn, problem.inputs,
      problem.hessian); // H equals (n-1)g since g is homogeneous of degree n-1.
}

/* @brief Error calculation for hessian and gradient
 */
void calculateErrors(ADProblem &problem) {
  double grad_err = 0;
  double hess_err = 0;

  // Calculate Gradient Error
  for (size_t i = 0; i < problem.dimIn; i++)
    grad_err +=
        fabs(problem.gradient[i] -
             problem.out[0] / problem.inputs[i]); // vanishes analytically.

  // Calculate Hessian Error
  for (size_t i = 0; i < problem.dimIn; i++) {
    for (size_t j = 0; j < problem.dimIn; j++) {
      if (i > j) // lower half of hessian
        hess_err += fabs(problem.hessian[i][j] -
                         problem.gradient[i] / problem.inputs[j]);
    }
  }

  std::cout << "Error in function: "
            << problem.out[0] - 1 / (1.0 + static_cast<double>(problem.dimIn))
            << std::endl;
  std::cout << "Error in gradient: " << grad_err << std::endl;
  std::cout << "Consistency check: " << hess_err << std::endl;
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
  ADProblem problem;
  taping(problem);
  printTapeStats(problem);
  computeDerivatives(problem);
  calculateErrors(problem);
  return 0;
}
