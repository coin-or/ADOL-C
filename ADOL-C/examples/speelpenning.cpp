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
#include <adolc/adolc.h>
#include <iostream>
using namespace std;

#include <cstdlib>
#include <math.h>

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
  const short tapeId = 1;
  createNewTape(tapeId);
  constexpr size_t n = 7;

  double *xp = new double[n];
  double yp = 0.0;
  adouble *x = new adouble[n];
  adouble y = 1;

  for (auto i = 0; i < n; i++)
    xp[i] = (i + 1.0) / (2.0 + i); // some initialization

  trace_on(tapeId); // tag = 1, keep = 0 by default
  for (auto i = 0; i < n; i++) {
    x[i] <<= xp[i]; // or  x <<= xp outside the loop
    y *= x[i];
  } // end for
  y >>= yp;
  delete[] x;
  trace_off();

  auto tape_stats = tapestats(tapeId); // reading of tape statistics
  cout << "maxlive " << tape_stats[TapeInfos::NUM_MAX_LIVES] << "\n";
  // ..... print other tape stats

  double *g = new double[n];
  gradient(tapeId, n, xp, g); // gradient evaluation

  double **H = new double *[n];
  for (auto i = 0; i < n; i++)
    H[i] = new double[i + 1];
  hessian(tapeId, n, xp, H); // H equals (n-1)g since g is
  double errg = 0;           // homogeneous of degree n-1.
  double errh = 0;
  for (auto i = 0; i < n; i++)
    errg += fabs(g[i] - yp / xp[i]); // vanishes analytically.
  for (auto i = 0; i < n; i++) {
    for (auto j = 0; j < n; j++) {
      if (i > j) // lower half of hessian
        errh += fabs(H[i][j] - g[i] / xp[j]);
    } // end for
  } // end for
  cout << yp - 1 / (1.0 + n) << " error in function \n";
  cout << errg << " error in gradient \n";
  cout << errh << " consistency check \n";

  for (auto i = 1; i < n; i++)
    delete[] H[i];

  delete[] H;
  return 0;
} // end main
