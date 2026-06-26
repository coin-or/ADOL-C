/*
This file defines eight abs-smooth test functions f_V, V = 1,...,8,
each represented by a templated taping routine `taping<Version>()`.
All functions use ADOL-C's ABS-normal form for studying piecewise
differentiability, switching structures, and Jacobian sparsity.

--------------------------------------------------------------------
VERSION 1
--------------------------------------------------------------------
dimIn  = 2,  dimOut = 2
Input  x = (x0, x1)

f1(x) = ( f1_0(x), f1_1(x) ) where
    f1_0(x) = x0 + |x0 - x1| + |x0 - |x1||,
    f1_1(x) = x1.

sparsity pattern (crs):
[3, 0, 2, 4]
[1, 1]
[2, 0, 1]
[1, 1]
[3, 0, 3]

--------------------------------------------------------------------
VERSION 2
--------------------------------------------------------------------
dimIn  = 2,  dimOut = 2
Input  x = (x0, x1)

f2_0(x) = x0 + |x0 - x1| + |x0 - |x1||,
f2_1(x) = |x1 - 5|.

sparsity pattern (crs):
[3, 0, 2, 4]
[1, 5]
[2, 0, 1]
[1, 1]
[2, 0, 3]
[1, 1]

--------------------------------------------------------------------
VERSION 3
--------------------------------------------------------------------
dimIn  = 2, dimOut = 1
Input  x = (x0, x1)

f3(x) = max( 0,  x1² - max(0, x0) ).

sparsity pattern (crs):
[4, 0, 1, 2, 3]
[1, 0]
[3, 0, 1, 2]

--------------------------------------------------------------------
VERSION 4
--------------------------------------------------------------------
dimIn = 5, dimOut = 1
Input x = (x0, …, x4)

Let
    z1 = | sum_{j=0..4} x_j / (j+1) |.

For each i = 1..4 define
    z2(i) = | sum_{j=0..4} x_j / (i + j + 1) |.

Then
    f4(x) = max( z1, z2(1), z2(2), z2(3), z2(4) ).

sparsity pattern (crs):
[9, 5, 6, 7, 8, 9, 10, 11, 12, 13]
[5, 0, 1, 2, 3, 4]
[5, 0, 1, 2, 3, 4]
[2, 5, 6]
[5, 0, 1, 2, 3, 4]
[4, 5, 6, 7, 8]
[5, 0, 1, 2, 3, 4]
[7, 5, 6, 7, 8, 9, 10]
[5, 0, 1, 2, 3, 4]
[8, 5, 6, 7, 8, 9, 10, 11, 12]
--------------------------------------------------------------------
VERSION 5
--------------------------------------------------------------------
dimIn = 2, dimOut = 1
Input x = (x0, x1)

f5(x) = max
    0, 2x0 - 5x1, 3x0 + 2x1, 2x0 + 5x1, 3x0 - 2x1, -100).


sparsity patter (crs):
[7, 0, 1, 2, 3, 4, 5, 6]
[2, 0, 1]
[3, 0, 1, 2]
[4, 0, 1, 2, 3]
[5, 0, 1, 2, 3, 4]
[6, 0, 1, 2, 3, 4, 5]
--------------------------------------------------------------------
VERSION 6
--------------------------------------------------------------------
dimIn = 2, dimOut = 1
Input x = (x0, x1)

  f6(x) = ( z1 + |x1² - x0 - 1 - (x0 + |-x0|)/2| ) / 2 .

sparsity pattern (crs):
[4, 0, 1, 2, 3]
[1, 0]
[3, 0, 1, 2]
--------------------------------------------------------------------
VERSION 7
--------------------------------------------------------------------
dimIn = 2, dimOut = 2
Input x = (x0, x1)

Let z = |x1|. Then

    f7_0(x) = max( -100,  3x0 + 2z,  2x0 + 5z ),
    f7_1(x) = max( -100,  3x0 + 2z,  2x0 + 5z ).

sparsity patter (crs):
[4, 0, 2, 3, 4]
[4, 0, 2, 5, 6]
[1, 1]
[2, 0, 2]
[3, 0, 2, 3]
[2, 0, 2]
[3, 0, 2, 5]


--------------------------------------------------------------------
VERSION 8
--------------------------------------------------------------------
dimIn = 2, dimOut = 1
Input x = (x0, x1)

Let z = |x1|. Then

    f8(x) = max( -100,  3x0 + 2z,  2x0 + 5z ).

sparsity pattern (crs):
[3, 0, 2, 3, 4]
[1, 1]
[2, 0, 2]
[3, 0, 2, 3]
--------------------------------------------------------------------
*/
#define BOOST_TEST_DYN_LINK
#include "../const.h"
#include <adolc/adolc.h>
#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(test_sparse_abs_normal)

static_assert(ADOLC::AbsNormalFormType<ADOLC::Sparse::SparseANF>);

namespace {
using ADOLC::Sparse::CoordinateFormatTripled;
using ADOLC::Sparse::SparseANF;
using ADOLC::Sparse::SparseMatrix;
template <size_t Version> struct version_trait {};

std::vector<double> updateConstant(const SparseMatrix &sparseEntries,
                                   const std::vector<double> &vals,
                                   const std::vector<double> &z) {
  std::vector<double> res(vals);
  for (int i = 0; i < sparseEntries.size(); i++) {
    const auto entry = sparseEntries[i];
    res[entry.rowIndex()] -= entry.value() * std::fabs(z[entry.colIndex()]);
  }
  return res;
}

template <> struct version_trait<1> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {3.0, 2.0};
  static constexpr size_t dimOut = 2;
  static constexpr std::array<std::array<size_t, 4>, 5> crs = {
      {{3, 0, 2, 4}, {1, 1, 0, 0}, {2, 0, 1, 0}, {1, 1, 0, 0}, {2, 0, 3, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 9> sparse = {
      {{0, 0, 1.0},
       {0, 2, 1.0},
       {0, 4, 1.0},
       {1, 1, 1.0},
       {2, 0, 1.0},
       {2, 1, -1.0},
       {3, 1, 1.0},
       {4, 0, 1.0},
       {4, 3, -1.0}}};
};

template <> struct version_trait<2> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {3.0, 2.0};
  static constexpr size_t dimOut = 2;
  static constexpr std::array<std::array<size_t, 4>, 6> crs = {{{3, 0, 2, 4},
                                                                {1, 5, 0, 0},
                                                                {2, 0, 1, 0},
                                                                {1, 1, 0, 0},
                                                                {2, 0, 3, 0},
                                                                {1, 1, 0, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 10> sparse = {
      {{0, 0, 1.0},
       {0, 2, 1.0},
       {0, 4, 1.0},
       {1, 5, 1.0},
       {2, 0, 1.0},
       {2, 1, -1.0},
       {3, 1, 1.0},
       {4, 0, 1.0},
       {4, 3, -1.0},
       {5, 1, 1.0}}};
};

template <> struct version_trait<3> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {1.0, 1.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 3> crs = {
      {{4, 0, 1, 2, 3}, {1, 0, 0, 0, 0}, {3, 0, 1, 2, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 8> sparse = {
      {{0, 0, -0.25},
       {0, 1, 1.0},
       {0, 2, -0.25},
       {0, 3, 0.5},
       {1, 0, 1.0},
       {2, 0, -0.5},
       {2, 1, 2.0},
       {2, 2, -0.5}}};
};

template <> struct version_trait<4> {
  static constexpr size_t dimIn = 5;
  static constexpr std::array<double, dimIn> init = {1.0, 1.0, 1.0, 1.0, 1.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 10>, 10> crs = {
      {{9, 5, 6, 7, 8, 9, 10, 11, 12, 13},
       {5, 0, 1, 2, 3, 4, 0, 0, 0, 0},
       {5, 0, 1, 2, 3, 4, 0, 0, 0, 0},
       {2, 5, 6, 0, 0, 0, 0, 0, 0, 0},
       {5, 0, 1, 2, 3, 4, 0, 0, 0, 0},
       {4, 5, 6, 7, 8, 0, 0, 0, 0, 0},
       {5, 0, 1, 2, 3, 4, 0, 0, 0, 0},
       {6, 5, 6, 7, 8, 9, 10, 0, 0, 0},
       {5, 0, 1, 2, 3, 4, 0, 0, 0, 0},
       {8, 5, 6, 7, 8, 9, 10, 11, 12, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 54> sparse = {{
      {0, 5, 1.0 / 16.0},  {0, 6, 1.0 / 16.0},  {0, 7, 1.0 / 16.0},
      {0, 8, 1.0 / 8.0},   {0, 9, 1.0 / 8.0},   {0, 10, 1.0 / 4.0},
      {0, 11, 1.0 / 4.0},  {0, 12, 1.0 / 2.0},  {0, 13, 1.0 / 2.0},
      {1, 0, 1.0},         {1, 1, 1.0 / 2.0},   {1, 2, 1.0 / 3.0},
      {1, 3, 1.0 / 4.0},   {1, 4, 1.0 / 5.0},   {2, 0, 1.0 / 2.0},
      {2, 1, 1.0 / 3.0},   {2, 2, 1.0 / 4.0},   {2, 3, 1.0 / 5.0},
      {2, 4, 1.0 / 6.0},   {3, 5, -1.0},        {3, 6, 1.0},
      {4, 0, 1.0 / 3.0},   {4, 1, 1.0 / 4.0},   {4, 2, 1.0 / 5.0},
      {4, 3, 1.0 / 6.0},   {4, 4, 1.0 / 7.0},   {5, 5, -1.0 / 2.0},
      {5, 6, -1.0 / 2.0},  {5, 7, -1.0 / 2.0},  {5, 8, 1.0},
      {6, 0, 1.0 / 4.0},   {6, 1, 1.0 / 5.0},   {6, 2, 1.0 / 6.0},
      {6, 3, 1.0 / 7.0},   {6, 4, 1.0 / 8.0},   {7, 5, -1.0 / 4.0},
      {7, 6, -1.0 / 4.0},  {7, 7, -1.0 / 4.0},  {7, 8, -1.0 / 2.0},
      {7, 9, -1.0 / 2.0},  {7, 10, 1.0},        {8, 0, 1.0 / 5.0},
      {8, 1, 1.0 / 6.0},   {8, 2, 1.0 / 7.0},   {8, 3, 1.0 / 8.0},
      {8, 4, 1.0 / 9.0},   {9, 5, -1.0 / 8.0},  {9, 6, -1.0 / 8.0},
      {9, 7, -1.0 / 8.0},  {9, 8, -1.0 / 4.0},  {9, 9, -1.0 / 4.0},
      {9, 10, -1.0 / 2.0}, {9, 11, -1.0 / 2.0}, {9, 12, 1.0},
  }};
};

template <> struct version_trait<5> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {0.0, 0.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 8>, 6> crs = {
      {{7, 0, 1, 2, 3, 4, 5, 6},
       {2, 0, 1, 0, 0, 0, 0, 0},
       {3, 0, 1, 2, 0, 0, 0, 0},
       {4, 0, 1, 2, 3, 0, 0, 0},
       {5, 0, 1, 2, 3, 4, 0, 0},
       {6, 0, 1, 2, 3, 4, 5, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 27> sparse = {{
      {0, 0, 1.25},   {0, 1, 0.09375}, {0, 2, 0.03125}, {0, 3, 0.0625},
      {0, 4, 0.125},  {0, 5, 0.25},    {0, 6, 0.5},     {1, 0, 2.0},
      {1, 1, -5.0},   {2, 0, 2.0},     {2, 1, 4.5},     {2, 2, -0.5},
      {3, 0, 0.0},    {3, 1, 5.25},    {3, 2, -0.25},   {3, 3, -0.5},
      {4, 0, 1.0},    {4, 1, -4.375},  {4, 2, -0.125},  {4, 3, -0.25},
      {4, 4, -0.5},   {5, 0, -2.5},    {5, 1, -0.1875}, {5, 2, -0.0625},
      {5, 3, -0.125}, {5, 4, -0.25},   {5, 5, -0.5},
  }};
};

template <> struct version_trait<6> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {0.0, -1.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 3> crs = {
      {{4, 0, 1, 2, 3}, {1, 0, 0, 0, 0}, {3, 0, 1, 2, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 8> sparse = {
      {{0, 0, -0.75},
       {0, 1, -1.0},
       {0, 2, -0.25},
       {0, 3, 0.5},
       {1, 0, -1.0},
       {2, 0, 1.5},
       {2, 1, 2.0},
       {2, 2, 0.5}}};
};

template <> struct version_trait<7> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {2.0, 2.0};
  static constexpr size_t dimOut = 2;
  static constexpr std::array<std::array<size_t, 5>, 7> crs = {
      {{4, 0, 2, 3, 4},
       {4, 0, 2, 5, 6},
       {1, 1, 0, 0, 0},
       {2, 0, 2, 0, 0},
       {3, 0, 2, 3, 0},
       {2, 0, 2, 0, 0},
       {3, 0, 2, 5, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 19> sparse = {{
      {0, 0, 1.75}, {0, 2, 3.0},  {0, 3, 0.25}, {0, 4, 0.5},  {1, 0, 1.75},
      {1, 2, 3.0},  {1, 5, 0.25}, {1, 6, 0.5},  {2, 1, 1.0},  {3, 0, 3.0},
      {3, 2, 2.0},  {4, 0, 0.5},  {4, 2, 4.0},  {4, 3, -0.5}, {5, 0, 3.0},
      {5, 2, 2.0},  {6, 0, 0.5},  {6, 2, 4.0},  {6, 5, -0.5},
  }};
};

template <> struct version_trait<8> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {2.0, 2.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 4> crs = {
      {{4, 0, 2, 3, 4}, {1, 1, 0, 0, 0}, {2, 0, 2, 0, 0}, {3, 0, 2, 3, 0}}};
  static constexpr std::array<CoordinateFormatTripled, 10> sparse = {
      {{0, 0, 1.75},
       {0, 2, 3.0},
       {0, 3, 0.25},
       {0, 4, 0.5},
       {1, 1, 1.0},
       {2, 0, 3.0},
       {2, 2, 2.0},
       {3, 0, 0.5},
       {3, 2, 4.0},
       {3, 3, -0.5}}};
};

struct UnAllocated {};
struct Allocated {};

template <size_t Version, typename State = UnAllocated> struct ANFProblem;

template <size_t Version> struct ANFProblem<Version, UnAllocated> {
  static constexpr auto init = version_trait<Version>::init;
  static constexpr size_t dimIn = version_trait<Version>::dimIn;
  static constexpr size_t dimOut = version_trait<Version>::dimOut;

  std::array<double, dimIn> in{};
  std::array<double, dimOut> out{};

  constexpr ANFProblem() : in(init) {}
  ANFProblem<Version, Allocated> allocateBuffers(short numSwitchingVars) {
    return ANFProblem<Version, Allocated>(numSwitchingVars, *this);
  }
};

template <size_t Version> struct ANFProblem<Version, Allocated> {
  static constexpr auto init = version_trait<Version>::init;
  static constexpr size_t dimIn = version_trait<Version>::dimIn;
  static constexpr size_t dimOut = version_trait<Version>::dimOut;

  static constexpr auto crs = version_trait<Version>::crs;

  std::array<double, dimIn> in{};
  std::array<double, dimOut> out{};

  int numSwitchingVars;
  std::vector<double> z;
  std::vector<double> cz;
  std::vector<double> cy;

  /* double **J, **Y;
  double **Z, **L; */
  Matrix<double> J;
  Matrix<double> Y;
  Matrix<double> Z;
  Matrix<double> L;

  std::vector<double> d;
  std::vector<double> g;
  /* double **gradz; */
  Matrix<double> gradz;

  std::vector<short> sigma_x;
  std::vector<short> sigma_g;

  ANFProblem(const ANFProblem &anfProblem) = default;
  ANFProblem(ANFProblem &&anfProblem) = default;

  ANFProblem &operator=(ANFProblem &anfProblem) = default;
  ANFProblem &operator=(ANFProblem &&anfProblem) = default;

  ~ANFProblem() = default;
  constexpr ANFProblem(short numSVars,
                       const ANFProblem<Version, UnAllocated> &base)
      : in(base.init), out(base.out) {

    numSwitchingVars = numSVars;
    sigma_x.resize(numSwitchingVars);
    sigma_g.resize(numSwitchingVars);
    z.resize(numSwitchingVars);

    cz.resize(numSwitchingVars);
    cy.resize(dimOut);

    Z = Matrix<double>(numSwitchingVars, dimIn);
    L = Matrix<double>(numSwitchingVars, numSwitchingVars);
    J = Matrix<double>(dimOut, numSwitchingVars);
    Y = Matrix<double>(dimOut, dimIn);

    d.resize(dimIn);
    g.resize(dimIn);
    gradz = Matrix<double>(numSwitchingVars, dimIn);
  }
};

template <size_t Version>
void computeANF(ANFProblem<Version, Allocated> &anfProblemAlloc, short tapeId) {

  zos_pl_forward(tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn, 1,
                 anfProblemAlloc.in, anfProblemAlloc.out, anfProblemAlloc.z);

  abs_normal(tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn,
             anfProblemAlloc.numSwitchingVars, anfProblemAlloc.in.data(),
             anfProblemAlloc.out.data(), anfProblemAlloc.z.data(),
             anfProblemAlloc.Y.data(), anfProblemAlloc.J.data(),
             anfProblemAlloc.Z.data(), anfProblemAlloc.L.data());
}

template <size_t Version>
void taping(ANFProblem<Version, UnAllocated> &anfProblem) {
  std::vector<adouble> x(anfProblem.dimIn);
  std::vector<adouble> y(anfProblem.dimOut);
  if constexpr (Version == 1) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = x[0] + fabs(x[0] - x[1]);
    y[0] += fabs(x[0] - fabs(x[1]));
    y[1] = x[1];
    y[0] >>= anfProblem.out[0];
    y[1] >>= anfProblem.out[1];
  } else if constexpr (Version == 2) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = x[0] + fabs(x[0] - x[1]);
    y[0] += fabs(x[0] - fabs(x[1]));
    y[1] = fabs(x[1] - 5);
    y[0] >>= anfProblem.out[0];
    y[1] >>= anfProblem.out[1];
  } else if constexpr (Version == 3) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = fmax(0, x[1] * x[1] - fmax(0, x[0]));
    y[0] >>= anfProblem.out[0];
  } else if constexpr (Version == 4) {
    {
      for (int i = 0; i < anfProblem.dimIn; i++) {
        x[i] <<= anfProblem.in[i];
      }
      int j = 0;
      adouble z1 =
          fabs(std::accumulate(x.begin(), x.end(), adouble{0.0},
                               [&](adouble &&res, adouble val) -> adouble {
                                 res += val / (j + 1);
                                 j++;
                                 return res;
                               }));
      for (int i = 1; i < anfProblem.dimIn; i++) {
        adouble z2 = 0;
        for (int j = 0; j < anfProblem.dimIn; j++) {
          z2 += x[j] / (i + j + 1);
        }
        z2 = fabs(z2);
        z1 = fmax(z1, z2);
      }
      y[0] = z1;
      y[0] >>= anfProblem.out[0];
    }
  } else if constexpr (Version == 5) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = 0;
    y[0] = fmax(y[0], 2 * x[0] - 5 * x[1]);
    y[0] = fmax(y[0], 3 * x[0] + 2 * x[1]);
    y[0] = fmax(y[0], 2 * x[0] + 5 * x[1]);
    y[0] = fmax(y[0], 3 * x[0] - 2 * x[1]);
    y[0] = fmax(y[0], static_cast<adouble>(-100));
    y[0] >>= anfProblem.out[0];
  } else if constexpr (Version == 6) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    adouble z1 = x[1] * x[1] - x[0] - 1.0 - (x[0] + fabs(-x[0])) / 2.0;
    y[0] = (z1 + fabs(-z1)) / 2.0;
    y[0] >>= anfProblem.out[0];
  } else if constexpr (Version == 7) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    adouble z1 = fabs(x[1]);
    y[0] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
    y[1] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
    y[0] >>= anfProblem.out[0];
    y[1] >>= anfProblem.out[1];
  } else if constexpr (Version == 8) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    adouble z1 = fabs(x[1]);
    y[0] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
    y[0] >>= anfProblem.out[0];
  }
}

void sortSparseEntries(SparseMatrix &mat) {
  std::sort(mat.entries().begin(), mat.entries().end(),
            [](const CoordinateFormatTripled &lhs,
               const CoordinateFormatTripled &rhs) {
              if (lhs.rowIndex() != rhs.rowIndex())
                return lhs.rowIndex() < rhs.rowIndex();
              if (lhs.colIndex() != rhs.colIndex())
                return lhs.colIndex() < rhs.colIndex();
              return lhs.value() < rhs.value();
            });
}

void checkEntries(const SparseMatrix &actualEntries,
                  const SparseMatrix &expectedEntries, size_t version,
                  const char *label) {
  auto actual = actualEntries;
  auto expected = expectedEntries;
  sortSparseEntries(actual);
  sortSparseEntries(expected);

  BOOST_TEST(actual.size() == expected.size());
  if (actual.size() != expected.size()) {
    return;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_TEST_CONTEXT("problem: " << version << " block=" << label
                                   << " entry=" << i) {
      BOOST_TEST(actual[i].rowIndex() == expected[i].rowIndex());
      BOOST_TEST(actual[i].colIndex() == expected[i].colIndex());
      BOOST_TEST(actual[i].value() == expected[i].value(), tt::tolerance(tol));
    }
  }
}

void checkVector(const std::vector<double> &actual,
                 const std::vector<double> &expected, size_t version,
                 const char *label) {
  BOOST_TEST_CONTEXT("problem: " << version << " vector=" << label) {
    BOOST_TEST(actual.size() == expected.size());
    if (actual.size() != expected.size()) {
      return;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
      BOOST_TEST(actual[i] == expected[i], tt::tolerance(tol));
    }
  }
}

template <size_t Version> SparseANF splitExpectedSparseANFEntries() {
  using ADOLC::Sparse::detail::classifySparseANFBlock;
  using ADOLC::Sparse::detail::SparseANFBlock;
  SparseANF expected;
  for (const auto &entry : version_trait<Version>::sparse) {
    switch (classifySparseANFBlock({entry.rowIndex(), entry.colIndex()},
                                   version_trait<Version>::dimOut,
                                   version_trait<Version>::dimIn)) {
    case SparseANFBlock::Y:
      expected.Y.push_back(entry);
      break;

    case SparseANFBlock::J:
      expected.J.push_back(
          {entry.rowIndex(),
           static_cast<unsigned int>(entry.colIndex() -
                                     version_trait<Version>::dimIn),
           entry.value()});
      break;

    case SparseANFBlock::Z:
      expected.Z.push_back(
          {static_cast<unsigned int>(entry.rowIndex() -
                                     version_trait<Version>::dimOut),
           entry.colIndex(), entry.value()});
      break;
    case SparseANFBlock::L:
      expected.L.push_back(
          {static_cast<unsigned int>(entry.rowIndex() -
                                     version_trait<Version>::dimOut),
           static_cast<unsigned int>(entry.colIndex() -
                                     version_trait<Version>::dimIn),
           entry.value()});
      break;
    }
  }
  return expected;
}

template <size_t Version>
void checkSparseANFEntries(const ADOLC::Sparse::SparseANF &sparseANF) {
  const auto expected = splitExpectedSparseANFEntries<Version>();
  checkEntries(sparseANF.Y, expected.Y, Version, "Y");
  checkEntries(sparseANF.J, expected.J, Version, "J");
  checkEntries(sparseANF.Z, expected.Z, Version, "Z");
  checkEntries(sparseANF.L, expected.L, Version, "L");
}

template <size_t Version>
void checkSparseANF(const ADOLC::Sparse::SparseANF &sparseANF,
                    const ANFProblem<Version, Allocated> &expectedDenseANF) {
  checkSparseANFEntries<Version>(sparseANF);
  BOOST_TEST(sparseANF.cy.size() == to_size_t(expectedDenseANF.dimOut));
  BOOST_TEST(sparseANF.cz.size() ==
             to_size_t(expectedDenseANF.numSwitchingVars));

  checkVector(sparseANF.cy,
              updateConstant(sparseANF.J, sparseANF.y, sparseANF.z), Version,
              "cy");
  checkVector(sparseANF.cz,
              updateConstant(sparseANF.L, sparseANF.z, sparseANF.z), Version,
              "cz");
}

template <size_t Version>
void runsparseAnfAndCheck(short tapeId,
                          const ANFProblem<Version, UnAllocated> &problem,
                          int numSwitchingVars, int repeat,
                          ADOLC::Sparse::SparseANF &sparseAnf) {
  using ADOLC::Sparse::MemoryHandler;
  using ADOLC::Sparse::PiecewiseLinear;

  const bool hasUserMemory = !sparseAnf.empty();
  const int ret =
      hasUserMemory
          ? ADOLC::Sparse::sparse_jac<PiecewiseLinear{}, MemoryHandler::Manual>(
                tapeId, problem.dimOut, problem.dimIn, numSwitchingVars, repeat,
                problem.in.data(), sparseAnf)
          : ADOLC::Sparse::sparse_jac<PiecewiseLinear{}, MemoryHandler::Auto>(
                tapeId, problem.dimOut, problem.dimIn, numSwitchingVars, repeat,
                problem.in.data(), sparseAnf);
  BOOST_TEST(ret >= 0);
  checkSparseANFEntries<Version>(sparseAnf);
}

template <size_t Version, ADOLC::Sparse::MemoryHandler MH>
void runSparseANFAndCheck(
    short tapeId, const ANFProblem<Version, UnAllocated> &problem,
    const ANFProblem<Version, Allocated> &expectedDenseANF,
    int numSwitchingVars, int repeat, ADOLC::Sparse::SparseANF &sparseANF) {
  using ADOLC::Sparse::PiecewiseLinear;

  const int ret = ADOLC::Sparse::sparse_jac<PiecewiseLinear{}, MH>(
      tapeId, problem.dimOut, problem.dimIn, numSwitchingVars, repeat,
      problem.in.data(), sparseANF);
  BOOST_TEST(ret >= 0);
  checkSparseANF<Version>(sparseANF, expectedDenseANF);
}

template <size_t Version> void problem() {
  ANFProblem<Version, UnAllocated> anfProblem{};
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(anfProblem);
  trace_off();

  int numSwitchingVars = get_num_switches(tapeId);
  auto anfProblemAlloc = anfProblem.allocateBuffers(numSwitchingVars);
  computeANF(anfProblemAlloc, tapeId);

  std::vector<uint *> crs(anfProblemAlloc.dimOut +
                          anfProblemAlloc.numSwitchingVars);
  std::span<uint *> crsSpan(crs);
  ADOLC::Sparse::jac_pat<ADOLC::Sparse::PiecewiseLinear{}>(
      tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn,
      anfProblemAlloc.numSwitchingVars, anfProblemAlloc.in.data(), crsSpan);

  for (int row{0}; row < anfProblemAlloc.crs.size(); row++) {
    for (int col = 0; col <= anfProblemAlloc.crs[row][0]; col++) {
      BOOST_TEST_CONTEXT("problem: " << Version << " row=" << row
                                     << " col=" << col) {
        BOOST_TEST(crs[row][col] == anfProblemAlloc.crs[row][col]);
      }
    }
  };
  for (auto &c : crs)
    delete[] c;
}

template <size_t Version> void problemsparseAnf() {
  ANFProblem<Version, UnAllocated> anfProblem{};
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(anfProblem);
  trace_off();

  int numSwitchingVars = get_num_switches(tapeId);
  ADOLC::Sparse::SparseANF sparseAnf;

  runsparseAnfAndCheck<Version>(tapeId, anfProblem, numSwitchingVars, 0,
                                sparseAnf);
}

template <size_t Version> void problemSparseANF() {
  using ADOLC::Sparse::SparseShape;
  ANFProblem<Version, UnAllocated> anfProblem{};
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(anfProblem);
  trace_off();

  const int numSwitchingVars = get_num_switches(tapeId);
  auto anfProblemAlloc = anfProblem.allocateBuffers(numSwitchingVars);
  computeANF(anfProblemAlloc, tapeId);

  ADOLC::Sparse::SparseANF sparseANF;
  runSparseANFAndCheck<Version, ADOLC::Sparse::MemoryHandler::Auto>(
      tapeId, anfProblem, anfProblemAlloc, numSwitchingVars, 0, sparseANF);

  const auto expected = splitExpectedSparseANFEntries<Version>();
  ADOLC::Sparse::SparseANF sparseANF2(
      SparseShape{expected.Y.size(), expected.J.size(), expected.Z.size(),
                  expected.L.size()});
  runSparseANFAndCheck<Version, ADOLC::Sparse::MemoryHandler::Manual>(
      tapeId, anfProblem, anfProblemAlloc, numSwitchingVars, 0, sparseANF2);

  ADOLC::Sparse::SparseANF sparseANF3(
      SparseShape{expected.Y.size(), expected.J.size(), expected.Z.size(),
                  expected.L.size()});
  runSparseANFAndCheck<Version, ADOLC::Sparse::MemoryHandler::Manual>(
      tapeId, anfProblem, anfProblemAlloc, numSwitchingVars, 1, sparseANF3);
}

template <size_t Version> void problemsparseAnfBranches() {
  ANFProblem<Version, UnAllocated> anfProblem{};
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(anfProblem);
  trace_off();

  const int numSwitchingVars = get_num_switches(tapeId);
  const auto expected = splitExpectedSparseANFEntries<Version>();

  ADOLC::Sparse::SparseANF sparseAnf;
  runsparseAnfAndCheck<Version>(tapeId, anfProblem, numSwitchingVars, 0,
                                sparseAnf);
  sparseAnf.clear();
  runsparseAnfAndCheck<Version>(tapeId, anfProblem, numSwitchingVars, 0,
                                sparseAnf);

  sparseAnf.clear();
  runsparseAnfAndCheck<Version>(tapeId, anfProblem, numSwitchingVars, 1,
                                sparseAnf);

  ADOLC::Sparse::SparseANF sparseAnf2;
  sparseAnf2.Y = ADOLC::Sparse::SparseMatrix(expected.Y.size());
  sparseAnf2.J = ADOLC::Sparse::SparseMatrix(expected.J.size());
  sparseAnf2.Z = ADOLC::Sparse::SparseMatrix(expected.Z.size());
  sparseAnf2.L = ADOLC::Sparse::SparseMatrix(expected.L.size());
  runsparseAnfAndCheck<Version>(tapeId, anfProblem, numSwitchingVars, 1,
                                sparseAnf2);
}

template <size_t Version> void problemSparseANFManualSizeMismatch() {
  using ADOLC::Sparse::MemoryHandler;
  using ADOLC::Sparse::PiecewiseLinear;

  ANFProblem<Version, UnAllocated> anfProblem{};
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(anfProblem);
  trace_off();

  const int numSwitchingVars = get_num_switches(tapeId);
  const auto expected = splitExpectedSparseANFEntries<Version>();

  ADOLC::Sparse::SparseANF wrongSized;
  wrongSized.Y = ADOLC::Sparse::SparseMatrix(expected.Y.size());
  wrongSized.J = ADOLC::Sparse::SparseMatrix(expected.J.size());
  wrongSized.Z = ADOLC::Sparse::SparseMatrix(expected.Z.size());
  wrongSized.L = ADOLC::Sparse::SparseMatrix(expected.L.size() + 1);

  int ret = ADOLC::Sparse::sparse_jac<PiecewiseLinear{}, MemoryHandler::Manual>(
      tapeId, anfProblem.dimOut, anfProblem.dimIn, numSwitchingVars, 0,
      anfProblem.in.data(), wrongSized);
  BOOST_TEST(ret == -3);

  ADOLC::Sparse::SparseANF sparseANF;
  ret = ADOLC::Sparse::sparse_jac<PiecewiseLinear{}, MemoryHandler::Auto>(
      tapeId, anfProblem.dimOut, anfProblem.dimIn, numSwitchingVars, 0,
      anfProblem.in.data(), sparseANF);
  BOOST_TEST(ret >= 0);

  ret = ADOLC::Sparse::sparse_jac<PiecewiseLinear{}, MemoryHandler::Manual>(
      tapeId, anfProblem.dimOut, anfProblem.dimIn, numSwitchingVars, 1,
      anfProblem.in.data(), wrongSized);
  BOOST_TEST(ret == -3);
}
} // namespace
BOOST_AUTO_TEST_CASE(PiecewiseLinearJacPatMatchesExpectedPattern) {
  constexpr std::array<void (*)(), 8> problems = {
      problem<1>, problem<2>, problem<3>, problem<4>,
      problem<5>, problem<6>, problem<7>, problem<8>};
  for (auto problem : problems) {
    problem();
  }
}

BOOST_AUTO_TEST_CASE(PiecewiseLinearsparseAnfMatchesExpectedTriplets) {
  constexpr std::array<void (*)(), 8> problems = {
      problemsparseAnf<1>, problemsparseAnf<2>, problemsparseAnf<3>,
      problemsparseAnf<4>, problemsparseAnf<5>, problemsparseAnf<6>,
      problemsparseAnf<7>, problemsparseAnf<8>};
  for (auto problem : problems) {
    problem();
  }
}

BOOST_AUTO_TEST_CASE(PiecewiseLinearSparseANFMatchesExpectedBlocks) {
  constexpr std::array<void (*)(), 8> problems = {
      problemSparseANF<1>, problemSparseANF<2>, problemSparseANF<3>,
      problemSparseANF<4>, problemSparseANF<5>, problemSparseANF<6>,
      problemSparseANF<7>, problemSparseANF<8>};
  for (auto problem : problems) {
    problem();
  }
}

BOOST_AUTO_TEST_CASE(PiecewiseLinearsparseAnfCoversDriverBranches) {
  problemsparseAnfBranches<8>();
}

BOOST_AUTO_TEST_CASE(PiecewiseLinearSparseANFManualSizeMismatch) {
  problemSparseANFManualSizeMismatch<8>();
}

BOOST_AUTO_TEST_SUITE_END()
