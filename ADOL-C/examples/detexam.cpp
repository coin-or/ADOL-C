/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     detexam.cpp
 Revision: $Id$
 Contents: computation of determinants, described in the manual

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

template <typename T>
adouble det(const T &A, size_t row,
            size_t col) // k <= n is the order of the submatrix
{
  if (col == 0)
    return 1.0;
  else {
    adouble t = 0;
    int p = 1;
    int sign;
    if (row % 2)
      sign = 1;
    else
      sign = -1;

    for (auto i = 0; i < A.size(); i++) {
      int p1 = 2 * p;
      if (col % p1 >= p) {
        if (col == p) {
          if (sign > 0)
            t += A[row - 1][i];
          else
            t -= A[row - 1][i];
        } else {
          if (sign > 0)
            t += A[row - 1][i] *
                 det(A, row - 1, col - p); // recursive call to det
          else
            t -= A[row - 1][i] *
                 det(A, row - 1, col - p); // recursive call to det
        }
        sign = -sign;
      }
      p = p1;
    }
    return t;
  }
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {
  const short tapeId = 0;
  createNewTape(tapeId);

  const int keep = 1;
  constexpr size_t n = 7;
  int m = 1;

  std::array<std::array<adouble, n>, n> A;

  trace_on(tapeId, keep); // tapeId=1=keep
  double detout = 0.0;
  double diag = 1.0;           // here keep the intermediates for
  for (auto i = 0; i < n; i++) // the subsequent call to reverse
  {
    m *= 2;
    for (auto j = 0; j < n; j++)
      A[i][j] <<= j / (1.0 + i); // make all elements of A independent

    diag += A[i][i].value(); // value(adouble) converts to double
    A[i][i] += 1.0;
  }
  adouble ad = det(A, n, m - 1); // actual function call.

  ad >>= detout;
  printf("\n %f - %f = %f  (should be 0)\n", detout, diag, detout - diag);
  trace_off();

  std::array<double, 1> u;
  u[0] = 1.0;
  std::array<double, n * n> B;

  reverse(tapeId, 1, n * n, 0, u.data(),
          B.data()); // call reverse to calculate the gradient

  std::cout << " \n first base? : ";
  for (auto i = 0; i < n; i++) {
    adouble sum = 0;
    for (auto j = 0; j < n; j++)     // the matrix A times the first n
      sum += A[i][j] * B[j];         // components of the gradient B
    std::cout << sum.value() << " "; // must be a Cartesian basis vector
  }
  std::cout << "\n";
  return 1;
}
