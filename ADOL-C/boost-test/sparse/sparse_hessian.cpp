#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <cmath>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
namespace tt = boost::test_tools;

#include "../const.h"
#include <adolc/adolc.h>
#include <cstdlib>

BOOST_AUTO_TEST_SUITE(test_sparse_hessian)

using ADOLC::Sparse::ControlFlowMode;
using ADOLC::Sparse::RecoveryMethod;

/*
 This test constructs a simple function whose Hessian is strictly diagonal.
 It is designed to verify that ADOL-C’s sparse Hessian driver correctly
 identifies and recovers only the diagonal nonzero entries.

 The function is:
   f(x) = sum_{i=0}^{4} [ sin(x_i) + (x_{i+10})^2 ]

 Analytical Hessian (expected):
   For i = 0,... , 4:
     (i,i)       = -sin(x_i)
   For i = 10,... , 14:
     (i,i)       = 2
   All off-diagonal entries are zero.

 Therefore, the Hessian is diagonal with 10 nonzero entries in total,
 all located on the diagonal. The sparse driver should only report those
 diagonal positions (rowIndices[i] == columnIndices[i]).

 Total unique non-zero positions: 10 (purely diagonal).
*/

template <ControlFlowMode CFM, RecoveryMethod RCM>
static void testSparseHessWithDiagonal(short tapeId) {
  createNewTape(tapeId);
  constexpr int dimOut = 5;
  constexpr int dimIn = 20;

  std::array<double, dimIn> in;
  int i = 0;
  std::for_each(in.begin(), in.end(), [&i](double &v) {
    i++;
    v = i * 1.0 / (i + 2.0);
  });

  trace_on(tapeId);
  {
    std::array<adouble, dimIn> indeps;
    for (int i{0}; i < indeps.size(); ++i)
      indeps[i] <<= in[i];

    adouble deps;
    for (int i{0}; i < dimOut; ++i)
      deps += sin(indeps[i]) + indeps[i + 10] * indeps[i + 10];

    double out;
    deps >>= out;
  }
  trace_off();

  std::array<double *, dimIn> hess;
  for (auto &h : hess) {
    h = new double[dimIn];
    for (int j{0}; j < dimIn; j++) {
      h[j] = 0.0;
    }
  }
  hessian(tapeId, dimIn, in.data(), hess.data());
  unsigned int *rowIndices = nullptr;    /* row indices    */
  unsigned int *columnIndices = nullptr; /* column indices */
  int numberOfNonzeros = 0;
  double *sparseValues = nullptr;

  ADOLC::Sparse::sparse_hess<CFM, RCM>(tapeId, dimIn, 0, in.data(),
                                       &numberOfNonzeros, &rowIndices,
                                       &columnIndices, &sparseValues);
  std::array<double, 10> trueSolution = {-std::sin(in[0]),
                                         -std::sin(in[1]),
                                         -std::sin(in[2]),
                                         -std::sin(in[3]),
                                         -std::sin(in[4]),
                                         2,
                                         2,
                                         2,
                                         2,
                                         2};

  BOOST_TEST(numberOfNonzeros == trueSolution.size());

  for (int i{0}; i < numberOfNonzeros; i++) {
    // test how it should be
    BOOST_TEST(sparseValues[i] == trueSolution[i]);

    // sanity check with non-sparse hess
    if (i < 5)
      BOOST_TEST(hess[i][i] == sparseValues[i]);
    else if (i >= 5)
      BOOST_TEST(hess[i + 5][i + 5] == sparseValues[i]);

    // we only have diagonal elements
    BOOST_TEST(rowIndices[i] == columnIndices[i]);
  }
  // test with rowIndices, columnIndices and sparseValues != nullptr
  ADOLC::Sparse::sparse_hess<CFM, RCM>(tapeId, dimIn, 0, in.data(),
                                       &numberOfNonzeros, &rowIndices,
                                       &columnIndices, &sparseValues);

  BOOST_TEST(numberOfNonzeros == trueSolution.size());

  for (int i{0}; i < numberOfNonzeros; i++) {
    // test how it should be
    BOOST_TEST(sparseValues[i] == trueSolution[i]);

    // sanity check with non-sparse hess
    if (i < 5)
      BOOST_TEST(hess[i][i] == sparseValues[i]);
    else if (i >= 5)
      BOOST_TEST(hess[i + 5][i + 5] == sparseValues[i]);

    // we only have diagonal elements
    BOOST_TEST(rowIndices[i] == columnIndices[i]);
  }
  // test with repeat == 1
  ADOLC::Sparse::sparse_hess<CFM, RCM>(tapeId, dimIn, 1, in.data(),
                                       &numberOfNonzeros, &rowIndices,
                                       &columnIndices, &sparseValues);

  BOOST_TEST(numberOfNonzeros == trueSolution.size());

  for (int i{0}; i < numberOfNonzeros; i++) {
    // test how it should be
    BOOST_TEST(sparseValues[i] == trueSolution[i]);

    // sanity check with non-sparse hess
    if (i < 5)
      BOOST_TEST(hess[i][i] == sparseValues[i]);
    else if (i >= 5)
      BOOST_TEST(hess[i + 5][i + 5] == sparseValues[i]);

    // we only have diagonal elements
    BOOST_TEST(rowIndices[i] == columnIndices[i]);
  }
  delete[] sparseValues;
  delete[] columnIndices;
  delete[] rowIndices;
  for (auto &h : hess)
    delete[] h;
}

BOOST_AUTO_TEST_CASE(SparseHessSafeDirect) {
  const short tapeId = 632;
  testSparseHessWithDiagonal<ControlFlowMode::Safe, RecoveryMethod::Direct>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessSafeIndirect) {
  const short tapeId = 633;
  testSparseHessWithDiagonal<ControlFlowMode::Safe, RecoveryMethod::Indirect>(
      tapeId);
}
BOOST_AUTO_TEST_CASE(SparseHessTightDirect) {
  const short tapeId = 634;
  testSparseHessWithDiagonal<ControlFlowMode::Tight, RecoveryMethod::Direct>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessTightIndirect) {
  const short tapeId = 635;
  testSparseHessWithDiagonal<ControlFlowMode::Tight, RecoveryMethod::Indirect>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessOldSafeDirect) {
  const short tapeId = 636;
  testSparseHessWithDiagonal<ControlFlowMode::OldSafe, RecoveryMethod::Direct>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessOldSafeIndirect) {
  const short tapeId = 637;
  testSparseHessWithDiagonal<ControlFlowMode::OldSafe,
                             RecoveryMethod::Indirect>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessOldTightDirect) {
  const short tapeId = 638;
  testSparseHessWithDiagonal<ControlFlowMode::OldTight, RecoveryMethod::Direct>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessOldTightIndirect) {
  const short tapeId = 639;
  testSparseHessWithDiagonal<ControlFlowMode::OldTight,
                             RecoveryMethod::Indirect>(tapeId);
}
/*
 This test constructs a function with mixed-product terms so that the Hessian
 contains off-diagonal nonzeros.

 f(x) = sin(x0) + sin(x1)
      + x2*x3
      + 0.5*x0*x2
      + x4*x5
      + x4*x4

 Analytical Hessian nonzeros (expected):
  (0,0) = -sin(x0)
  (1,1) = -sin(x1)
  (2,3) = 1
  (3,2) = 1   // symmetric of (2,3)
  (0,2) = 0.5
  (2,0) = 0.5 // symmetric of (0,2)
  (4,5) = 1
  (5,4) = 1   // symmetric of (4,5)
  (4,4) = 2
 Total unique non-zero positions: 9 (but sparse_hess return only lower triangle
 like hessian driver).
*/

template <ControlFlowMode CFM, RecoveryMethod RCM>
static void testSparseHessWithOffDiagonals(short tapeId) {
  createNewTape(tapeId);
  constexpr int dimIn = 6;

  std::array<double, dimIn> in;
  int i = 0;
  std::for_each(in.begin(), in.end(), [&i](double &v) {
    i++;
    v = 0.1 * (i + 1);
  });

  trace_on(tapeId);
  {
    std::array<adouble, dimIn> indeps;
    for (int i{0}; i < indeps.size(); ++i)
      indeps[i] <<= in[i];

    adouble deps = 0.0;
    deps += sin(indeps[0]);              // contributes (0,0) -> -sin(x0)
    deps += sin(indeps[1]);              // contributes (1,1) -> -sin(x1)
    deps += indeps[2] * indeps[3];       // contributes (2,3) and (3,2) -> 1
    deps += 0.5 * indeps[0] * indeps[2]; // contributes (0,2) and (2,0) -> 0.5
    deps += indeps[4] * indeps[5];       // contributes (4,5) and (5,4) -> 1
    deps += indeps[4] * indeps[4];       // contributes (4,4) -> 2

    double out;
    deps >>= out;
  }
  trace_off();

  // compute dense hessian with ADOL-C for sanity comparison
  std::array<double *, dimIn> hess;
  for (auto &h : hess) {
    h = new double[dimIn];
    for (int j{0}; j < dimIn; j++) {
      h[j] = 0.0;
    }
  }
  hessian(tapeId, dimIn, in.data(), hess.data());

  // call sparse_hess
  unsigned int *rowIndices = nullptr;    /* row indices    */
  unsigned int *columnIndices = nullptr; /* column indices */
  double *sparseValues = nullptr;        /* values         */
  int numberOfNonzeros = 0;

  ADOLC::Sparse::sparse_hess<CFM, RCM>(tapeId, dimIn, 0, in.data(),
                                       &numberOfNonzeros, &rowIndices,
                                       &columnIndices, &sparseValues);

  // build the expected set of nonzeros (list of triples)
  struct SparseResult {
    unsigned r, c;
    double v;
  };
  std::vector<SparseResult> expected{{0, 0, -std::sin(in[0])},
                                     {1, 1, -std::sin(in[1])},
                                     {3, 2, 1.0},
                                     {2, 0, 0.5},
                                     {5, 4, 1.0},
                                     {4, 4, 2.0}};

  BOOST_TEST(numberOfNonzeros == expected.size());
  for (int i{0}; i < numberOfNonzeros; i++) {
    bool found = false;

    auto row = std::max(rowIndices[i], columnIndices[i]);
    auto col = std::min(rowIndices[i], columnIndices[i]);

    // search if the computed row and column indices are present
    for (auto &expect : expected) {
      if (expect.r == row && expect.c == col) {
        found = true;
        BOOST_TEST(expect.v == sparseValues[i]);
        BOOST_TEST(hess[row][col] == sparseValues[i], tt::tolerance(tol));
      }
    }
    if (!found) {
      auto s = "Non-zero at (" + std::to_string(rowIndices[i]) + "," +
               std::to_string(columnIndices[i]) + ") not expected!";
      BOOST_CHECK(found && s.c_str());
    }
  }

  // cleanup
  delete[] rowIndices;
  delete[] columnIndices;
  delete[] sparseValues;
  for (auto &h: hess)
    delete[] h;
}
BOOST_AUTO_TEST_CASE(SparseHessNonDiagSafeDirect) {
  const short tapeId = 700;
  testSparseHessWithOffDiagonals<ControlFlowMode::Safe, RecoveryMethod::Direct>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessNonDiagSafeIndirect) {
  const short tapeId = 701;
  testSparseHessWithOffDiagonals<ControlFlowMode::Safe,
                                 RecoveryMethod::Indirect>(tapeId);
}
BOOST_AUTO_TEST_CASE(SparseHessNonDiagTightDirect) {
  const short tapeId = 702;
  testSparseHessWithOffDiagonals<ControlFlowMode::Tight,
                                 RecoveryMethod::Direct>(tapeId);
}
BOOST_AUTO_TEST_CASE(SparseHessNonDiagTightIndirect) {
  const short tapeId = 703;
  testSparseHessWithOffDiagonals<ControlFlowMode::Tight,
                                 RecoveryMethod::Indirect>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessNonDiagOldSafeDirect) {
  const short tapeId = 704;
  testSparseHessWithOffDiagonals<ControlFlowMode::OldSafe,
                                 RecoveryMethod::Direct>(tapeId);
}
BOOST_AUTO_TEST_CASE(SparseHessNonDiagOldSafeIndirect) {
  const short tapeId = 705;
  testSparseHessWithOffDiagonals<ControlFlowMode::OldSafe,
                                 RecoveryMethod::Indirect>(tapeId);
}
BOOST_AUTO_TEST_CASE(SparseHessNonDiagOldTightDirect) {
  const short tapeId = 706;
  testSparseHessWithOffDiagonals<ControlFlowMode::OldTight,
                                 RecoveryMethod::Direct>(tapeId);
}
BOOST_AUTO_TEST_CASE(SparseHessNonDiagOldTightIndirect) {
  const short tapeId = 707;
  testSparseHessWithOffDiagonals<ControlFlowMode::OldTight,
                                 RecoveryMethod::Indirect>(tapeId);
}

template <ControlFlowMode CFM>
static void testSparseHessPatWithDiagonal(short tapeId) {
  createNewTape(tapeId);
  constexpr int dimOut = 5;
  constexpr int dimIn = 20;

  std::array<double, dimIn> in;
  int i = 0;
  std::for_each(in.begin(), in.end(), [&i](double &v) {
    i++;
    v = i * 1.0 / (i + 2.0);
  });

  trace_on(tapeId);
  {
    std::array<adouble, dimIn> indeps;
    for (int i{0}; i < indeps.size(); ++i)
      indeps[i] <<= in[i];

    adouble deps;
    for (int i{0}; i < dimOut; ++i)
      deps += sin(indeps[i]) + indeps[i + 10] * indeps[i + 10];

    double out;
    deps >>= out;
  }
  trace_off();
  std::array<uint *, dimIn> compressedRowStorage;
  std::span<uint *> compressedRowStorageSpan(compressedRowStorage);
  ADOLC::Sparse::hess_pat<CFM>(tapeId, dimIn, in.data(),
                               compressedRowStorageSpan);

  // first hessian row only has first entry
  BOOST_TEST(compressedRowStorage[0][0] == 1);
  BOOST_TEST(compressedRowStorage[0][1] == 0);

  // etc...
  BOOST_TEST(compressedRowStorage[1][0] == 1);
  BOOST_TEST(compressedRowStorage[1][1] == 1);

  BOOST_TEST(compressedRowStorage[2][0] == 1);
  BOOST_TEST(compressedRowStorage[2][1] == 2);

  BOOST_TEST(compressedRowStorage[3][0] == 1);
  BOOST_TEST(compressedRowStorage[3][1] == 3);

  BOOST_TEST(compressedRowStorage[4][0] == 1);
  BOOST_TEST(compressedRowStorage[4][1] == 4);

  BOOST_TEST(compressedRowStorage[5][0] == 0);
  BOOST_TEST(compressedRowStorage[6][0] == 0);
  BOOST_TEST(compressedRowStorage[7][0] == 0);
  BOOST_TEST(compressedRowStorage[8][0] == 0);
  BOOST_TEST(compressedRowStorage[9][0] == 0);

  BOOST_TEST(compressedRowStorage[10][0] == 1);
  BOOST_TEST(compressedRowStorage[10][1] == 10);

  BOOST_TEST(compressedRowStorage[11][0] == 1);
  BOOST_TEST(compressedRowStorage[11][1] == 11);

  BOOST_TEST(compressedRowStorage[12][0] == 1);
  BOOST_TEST(compressedRowStorage[12][1] == 12);

  BOOST_TEST(compressedRowStorage[13][0] == 1);
  BOOST_TEST(compressedRowStorage[13][1] == 13);

  BOOST_TEST(compressedRowStorage[14][0] == 1);
  BOOST_TEST(compressedRowStorage[14][1] == 14);

  BOOST_TEST(compressedRowStorage[15][0] == 0);
  BOOST_TEST(compressedRowStorage[16][0] == 0);
  BOOST_TEST(compressedRowStorage[17][0] == 0);
  BOOST_TEST(compressedRowStorage[18][0] == 0);
  BOOST_TEST(compressedRowStorage[19][0] == 0);

  for (auto& crs: compressedRowStorage)
    delete[] crs;
}

BOOST_AUTO_TEST_CASE(SparseHessPatDiagSafe) {
  const short tapeId = 708;
  testSparseHessPatWithDiagonal<ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessPatDiagTight) {
  const short tapeId = 709;
  testSparseHessPatWithDiagonal<ControlFlowMode::Tight>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessPatDiagOldSafe) {
  const short tapeId = 710;
  testSparseHessPatWithDiagonal<ControlFlowMode::OldSafe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessPatDiagOldTight) {
  const short tapeId = 711;
  testSparseHessPatWithDiagonal<ControlFlowMode::OldTight>(tapeId);
}

template <ControlFlowMode CFM>
static void testSparseHessPatWithOffDiagonal(short tapeId) {
  createNewTape(tapeId);
  constexpr int dimIn = 6;

  std::array<double, dimIn> in;
  int i = 0;
  std::for_each(in.begin(), in.end(), [&i](double &v) {
    i++;
    v = 0.1 * (i + 1);
  });

  trace_on(tapeId);
  {
    std::array<adouble, dimIn> indeps;
    for (int i{0}; i < indeps.size(); ++i)
      indeps[i] <<= in[i];

    adouble deps = 0.0;
    deps += sin(indeps[0]);              // contributes (0,0) -> -sin(x0)
    deps += sin(indeps[1]);              // contributes (1,1) -> -sin(x1)
    deps += indeps[2] * indeps[3];       // contributes (2,3) and (3,2) -> 1
    deps += 0.5 * indeps[0] * indeps[2]; // contributes (0,2) and (2,0) -> 0.5
    deps += indeps[4] * indeps[5];       // contributes (4,5) and (5,4) -> 1
    deps += indeps[4] * indeps[4];       // contributes (4,4) -> 2

    double out;
    deps >>= out;
  }
  trace_off();

  std::array<uint *, dimIn> compressedRowStorage;
  std::span<uint *> compressedRowStorageSpan(compressedRowStorage);
  ADOLC::Sparse::hess_pat<CFM>(tapeId, dimIn, in.data(),
                               compressedRowStorageSpan);

  // first hessian row has first entry and third
  BOOST_TEST(compressedRowStorage[0][0] == 2);
  BOOST_TEST(compressedRowStorage[0][1] == 0);
  BOOST_TEST(compressedRowStorage[0][2] == 2);

  // second hessian row has second entry
  BOOST_TEST(compressedRowStorage[1][0] == 1);
  BOOST_TEST(compressedRowStorage[1][1] == 1);

  // third hessian row has first and fourth
  BOOST_TEST(compressedRowStorage[2][0] == 2);
  BOOST_TEST(compressedRowStorage[2][1] == 0);
  BOOST_TEST(compressedRowStorage[2][2] == 3);

  // fourth hessian row has third
  BOOST_TEST(compressedRowStorage[3][0] == 1);
  BOOST_TEST(compressedRowStorage[3][1] == 2);

  // fifth hessian row has fifth and sixth
  BOOST_TEST(compressedRowStorage[4][0] == 2);
  BOOST_TEST(compressedRowStorage[4][1] == 4);
  BOOST_TEST(compressedRowStorage[4][2] == 5);

  // sixth hessian row has fifth
  BOOST_TEST(compressedRowStorage[5][0] == 1);
  BOOST_TEST(compressedRowStorage[5][1] == 4);

  for (auto& crs: compressedRowStorage)
    delete[] crs;
}

BOOST_AUTO_TEST_CASE(SparseHessPatOffDiagSafe) {
  const short tapeId = 712;
  testSparseHessPatWithOffDiagonal<ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessPatOffDiagTight) {
  const short tapeId = 713;
  testSparseHessPatWithOffDiagonal<ControlFlowMode::Tight>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessPatOffDiagOldSafe) {
  const short tapeId = 714;
  testSparseHessPatWithOffDiagonal<ControlFlowMode::OldSafe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseHessPatOffDiagOldTight) {
  const short tapeId = 715;
  testSparseHessPatWithOffDiagonal<ControlFlowMode::OldTight>(tapeId);
}
BOOST_AUTO_TEST_SUITE_END()
