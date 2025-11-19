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
#include <numeric>
#define BOOST_TEST_DYN_LINK
#include <adolc/adolc.h>
#include <boost/test/unit_test.hpp>
#include <cstdlib>

BOOST_AUTO_TEST_SUITE(test_sparse_abs_normal)

template <size_t Version> struct version_trait {};

template <> struct version_trait<1> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {3.0, 2.0};
  static constexpr size_t dimOut = 2;
  static constexpr std::array<std::array<size_t, 4>, 5> crs = {
      {{3, 0, 2, 4}, {1, 1, 0, 0}, {2, 0, 1, 0}, {1, 1, 0, 0}, {2, 0, 3, 0}}};
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
};

template <> struct version_trait<3> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {1.0, 1.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 3> crs = {
      {{4, 0, 1, 2, 3}, {1, 0, 0, 0, 0}, {3, 0, 1, 2, 0}}};
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
};

template <> struct version_trait<6> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {0.0, -1.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 3> crs = {
      {{4, 0, 1, 2, 3}, {1, 0, 0, 0, 0}, {3, 0, 1, 2, 0}}};
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
};

template <> struct version_trait<8> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> init = {2.0, 2.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 4> crs = {
      {{4, 0, 2, 3, 4}, {1, 1, 0, 0, 0}, {2, 0, 2, 0, 0}, {3, 0, 2, 3, 0}}};
};

struct UnAllocated {};
struct Allocated {};

template <size_t Version, typename State = UnAllocated> struct ANFProblem;

template <size_t Version> struct ANFProblem<Version, UnAllocated> {
  static constexpr auto init = version_trait<Version>::init;
  static constexpr size_t dimIn = version_trait<Version>::dimIn;
  static constexpr size_t dimOut = version_trait<Version>::dimOut;

  std::array<double, dimIn> in;
  std::array<double, dimOut> out;

  static constexpr short tapeId = 716 + Version;

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

  std::array<double, dimIn> in;
  std::array<double, dimOut> out;

  static constexpr short tapeId = 716 + Version;

  int numSwitchingVars;
  std::vector<double> z;
  std::vector<double> cz;
  std::vector<double> cy;

  double **J, **Y;
  double **Z, **L;

  std::vector<double> d;
  std::vector<double> g;
  double **gradz;

  std::vector<short> sigma_x;
  std::vector<short> sigma_g;

  ANFProblem(ANFProblem &anfProblem) = delete;
  ANFProblem(ANFProblem &&anfProblem) = delete;

  ANFProblem operator=(ANFProblem &anfProblem) = delete;
  ANFProblem operator=(ANFProblem &&anfProblem) = delete;

  ~ANFProblem() {
    myfree2(Z);
    myfree2(L);
    myfree2(J);
    myfree2(Y);
    myfree2(gradz);
  }
  constexpr ANFProblem(short numSVars,
                       const ANFProblem<Version, UnAllocated> &base)
      : in(base.init), out(base.out) {

    numSwitchingVars = numSVars;
    sigma_x.resize(numSwitchingVars);
    sigma_g.resize(numSwitchingVars);
    z.resize(numSwitchingVars);

    cz.resize(numSwitchingVars);
    cy.resize(dimOut);
    Z = myalloc2(numSwitchingVars, dimIn);
    L = myalloc2(numSwitchingVars, numSwitchingVars);
    J = myalloc2(dimOut, numSwitchingVars);
    Y = myalloc2(dimOut, dimIn);

    d.resize(dimIn);
    g.resize(dimIn);
    gradz = myalloc2(numSwitchingVars, dimIn);
  }
};

template <size_t Version>
static void computeANF(ANFProblem<Version, Allocated> &anfProblemAlloc) {

  zos_pl_forward(anfProblemAlloc.tapeId, anfProblemAlloc.dimOut,
                 anfProblemAlloc.dimIn, 1, anfProblemAlloc.in.data(),
                 anfProblemAlloc.out.data(), anfProblemAlloc.z.data());

  abs_normal(anfProblemAlloc.tapeId, anfProblemAlloc.dimOut,
             anfProblemAlloc.dimIn, anfProblemAlloc.numSwitchingVars,
             anfProblemAlloc.in.data(), anfProblemAlloc.out.data(),
             anfProblemAlloc.z.data(), anfProblemAlloc.cz.data(),
             anfProblemAlloc.cy.data(), anfProblemAlloc.Y, anfProblemAlloc.J,
             anfProblemAlloc.Z, anfProblemAlloc.L);
}

template <size_t Version>
static void taping(ANFProblem<Version, UnAllocated> &anfProblem) {
  std::vector<adouble> x(anfProblem.dimIn);
  std::vector<adouble> y(anfProblem.dimOut);
  if constexpr (Version == 1) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = x[0] + fabs(x[0] - x[1]) + fabs(x[0] - fabs(x[1]));
    y[1] = x[1];
    y[0] >>= anfProblem.out[0];
    y[1] >>= anfProblem.out[1];
  } else if (Version == 2) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = x[0] + fabs(x[0] - x[1]) + fabs(x[0] - fabs(x[1]));
    y[1] = fabs(x[1] - 5);
    y[0] >>= anfProblem.out[0];
    y[1] >>= anfProblem.out[1];
  } else if (Version == 3) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = fmax(0, x[1] * x[1] - fmax(0, x[0]));
    y[0] >>= anfProblem.out[0];
  } else if (Version == 4) {
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
  } else if (Version == 5) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    y[0] = 0;
    y[0] = fmax(y[0], 2 * x[0] - 5 * x[1]);
    y[0] = fmax(y[0], 3 * x[0] + 2 * x[1]);
    y[0] = fmax(y[0], 2 * x[0] + 5 * x[1]);
    y[0] = fmax(y[0], 3 * x[0] - 2 * x[1]);
    y[0] = fmax(y[0], static_cast<adouble>(-100));
    y[0] >>= anfProblem.out[0];
  } else if (Version == 6) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    adouble z1 = x[1] * x[1] - x[0] - 1.0 - (x[0] + fabs(-x[0])) / 2.0;
    y[0] = (z1 + fabs(-z1)) / 2.0;
    y[0] >>= anfProblem.out[0];
  } else if (Version == 7) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    adouble z1 = fabs(x[1]);
    y[0] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
    y[1] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
    y[0] >>= anfProblem.out[0];
    y[1] >>= anfProblem.out[1];
  } else if (Version == 8) {
    x[0] <<= anfProblem.in[0];
    x[1] <<= anfProblem.in[1];
    adouble z1 = fabs(x[1]);
    y[0] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
    y[0] >>= anfProblem.out[0];
  }
}
template <size_t Version> static void problem() {
  ANFProblem<Version, UnAllocated> anfProblem{};
  createNewTape(anfProblem.tapeId);
  setCurrentTape(anfProblem.tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(anfProblem.tapeId);
  taping(anfProblem);
  trace_off();

  int numSwitchingVars = get_num_switches(anfProblem.tapeId);
  auto anfProblemAlloc = anfProblem.allocateBuffers(numSwitchingVars);
  computeANF(anfProblemAlloc);

  std::vector<uint *> crs(anfProblemAlloc.dimOut +
                          anfProblemAlloc.numSwitchingVars);
  std::span<uint *> crsSpan(crs);
  ADOLC::Sparse::absnormal_jac_pat(
      anfProblemAlloc.tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn,
      anfProblemAlloc.numSwitchingVars, anfProblemAlloc.in.data(), crsSpan);

  for (int row{0}; row < anfProblemAlloc.crs.size(); row++) {
    BOOST_TEST(crs[row][0] == anfProblemAlloc.crs[row][0]);
    for (int col{1}; col < anfProblemAlloc.crs[row][0]; col++) {
      BOOST_TEST(crs[row][col] == anfProblemAlloc.crs[row][col]);
    }
  };
}

BOOST_AUTO_TEST_CASE(SparseANFPatTest) {
  constexpr std::array<void (*)(), 8> problems = {
      problem<1>, problem<2>, problem<3>, problem<4>,
      problem<5>, problem<6>, problem<7>, problem<8>};
  for (auto problem : problems) {
    problem();
  }
}

BOOST_AUTO_TEST_SUITE_END()
