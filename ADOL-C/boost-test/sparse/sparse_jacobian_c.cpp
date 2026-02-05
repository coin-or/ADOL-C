#include "adolc/sparse/sparse_options.h"
#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
namespace tt = boost::test_tools;

#include <adolc/adolc.h>
#include <cstdlib>

#include "../const.h"

BOOST_AUTO_TEST_SUITE(test_sparse_jacobian_c)

using ADOLC::Sparse::BitPatternPropagationDirection;
using ADOLC::Sparse::CompressionMode;
using ADOLC::Sparse::ControlFlowMode;
using ADOLC::Sparse::SparseMethod;

template <SparseMethod SM, CompressionMode CM, ControlFlowMode CFM,
          BitPatternPropagationDirection BPPD>
static int *mapOptions() {
  int *options = new int[4];
  options[0] = (SM == SparseMethod::IndexDomains) ? 0 : 1;
  options[1] = (CFM == ControlFlowMode::Safe) ? 0 : 1;
  options[2] =
      (BPPD == BitPatternPropagationDirection::Auto)
          ? 0
          : ((BPPD == BitPatternPropagationDirection::Forward) ? 1 : 2);
  options[3] = (CM == CompressionMode::Column) ? 0 : 1;
  return options;
}

template <
    SparseMethod SM, CompressionMode CM, ControlFlowMode CFM,
    BitPatternPropagationDirection BPPD = BitPatternPropagationDirection::Auto>
static void testSparseJac(short tapeId) {
  createNewTape(tapeId);
  constexpr int dimOut = 10;
  constexpr int dimIn = 20;

  std::array<double, dimIn> in;
  in.fill(std::rand());
  trace_on(tapeId);
  {
    std::array<adouble, dimIn> indeps;
    for (int i{0}; i < indeps.size(); ++i)
      indeps[i] <<= in[i];

    std::array<adouble, dimOut> deps;
    for (int i{0}; i < deps.size(); ++i)
      deps[i] = sin(indeps[i]) + indeps[i + 10] * indeps[i + 10];

    std::array<double, dimOut> out;
    for (int i{0}; i < deps.size(); ++i)
      deps[i] >>= out[i];
  }
  trace_off();
  std::array<double *, dimOut> jac;
  for (auto &j : jac)
    j = new double[dimIn];
  jacobian(tapeId, dimOut, dimIn, in.data(), jac.data());

  unsigned int *rowIndices = nullptr;    /* row indices    */
  unsigned int *columnIndices = nullptr; /* column indices */
  double *nonzeroValues = nullptr;       /* values         */
  int numberOfNonzeros = 0;

  int *options = mapOptions<SM, CM, CFM, BPPD>();
  sparse_jac(tapeId, dimOut, dimIn, 0, in.data(), &numberOfNonzeros,
             &rowIndices, &columnIndices, &nonzeroValues, options);
  BOOST_TEST(numberOfNonzeros == 20);
  // go through all entries: either 0 or == sparse_jac
  int nonzeroCounter = 0;
  for (int row{0}; row < dimOut; ++row) {
    for (int col{0}; col < dimIn; ++col) {
      if (jac[row][col] != 0) {
        BOOST_TEST(jac[row][col] == nonzeroValues[nonzeroCounter++],
                   tt::tolerance(tol));
      }
    }
  }
  sparse_jac(tapeId, dimOut, dimIn, 0, in.data(), &numberOfNonzeros,
             &rowIndices, &columnIndices, &nonzeroValues, options);
  BOOST_TEST(numberOfNonzeros == 20);
  nonzeroCounter = 0;
  // go through all entries: either 0 or == sparse_jac
  for (int row{0}; row < dimOut; ++row) {
    for (int col{0}; col < dimIn; ++col) {
      if (jac[row][col] != 0) {
        BOOST_TEST(jac[row][col] == nonzeroValues[nonzeroCounter++],
                   tt::tolerance(tol));
      }
    }
  }
  sparse_jac(tapeId, dimOut, dimIn, 1, in.data(), &numberOfNonzeros,
             &rowIndices, &columnIndices, &nonzeroValues, options);
  BOOST_TEST(numberOfNonzeros == 20);
  nonzeroCounter = 0;
  // go through all entries: either 0 or == sparse_jac
  for (int row{0}; row < dimOut; ++row) {
    for (int col{0}; col < dimIn; ++col) {
      if (jac[row][col] != 0) {
        BOOST_TEST(jac[row][col] == nonzeroValues[nonzeroCounter++],
                   tt::tolerance(tol));
      }
    }
  }
  delete[] rowIndices;
  delete[] columnIndices;
  delete[] nonzeroValues;
  for (auto &j : jac)
    delete[] j;

  delete[] options;
}

BOOST_AUTO_TEST_CASE(SparseJacIndexColSafe_c) {
  const short tapeId = 1019;
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Column,
                ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacIndexColTight_c) {
  const short tapeId = 1020;
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Column,
                ControlFlowMode::Tight>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacIndexRowSafe_c) {
  const short tapeId = 1021;
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Row,
                ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacIndexRowTight_c) {
  const short tapeId = 1022;
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Row,
                ControlFlowMode::Tight>(tapeId);
}
BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropTight_c) {
  const short tapeId = 1023;
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Tight>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropSafe_c) {
  const short tapeId = 1024;
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropForwardTight_c) {
  const short tapeId = 1025;
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Tight,
                BitPatternPropagationDirection::Forward>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropForwardSafe_c) {
  const short tapeId = 1026;
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Safe, BitPatternPropagationDirection::Forward>(
      tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropReverseTight_c) {
  const short tapeId = 1027;
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Tight,
                BitPatternPropagationDirection::Reverse>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropReverseSafe_c) {
  const short tapeId = 1028;
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Safe, BitPatternPropagationDirection::Reverse>(
      tapeId);
}

template <SparseMethod SM, ControlFlowMode CFM>
static void testSparseJacPat(short tapeId) {
  createNewTape(tapeId);
  constexpr int dimIn = 7;
  constexpr int dimOut = 2;

  std::array<double, dimIn> in;
  int i = 0;
  std::for_each(in.begin(), in.end(), [&i](double &v) { v = (i++ * 2.3); });

  trace_on(tapeId);
  {
    std::array<adouble, dimIn> indeps;
    int i = 0;
    std::for_each(indeps.begin(), indeps.end(),
                  [&i, &in](auto &&v) { v <<= in[i++]; });
    adouble out1 = pow(indeps[0], 3);
    out1 += indeps[1] * indeps[4];

    double dummyOut = 0.0;
    out1 >>= dummyOut;

    adouble out2 = exp(indeps[6] + indeps[5]);
    out2 -= indeps[2];
    out2 >>= dummyOut;
  }
  trace_off();

  std::array<uint *, dimOut> compressedRowStorage;
  auto options = mapOptions<SM, CompressionMode::Row, CFM,
                            BitPatternPropagationDirection::Forward>();
  jac_pat(tapeId, dimOut, dimIn, in.data(), compressedRowStorage.data(),
          options);

  // first output depends on 0, 1 and 4
  BOOST_TEST(compressedRowStorage[0][0] == 3);
  BOOST_TEST(compressedRowStorage[0][1] == 0);
  BOOST_TEST(compressedRowStorage[0][2] == 1);
  BOOST_TEST(compressedRowStorage[0][3] == 4);

  // second output depends on 2, 5 and 6
  BOOST_TEST(compressedRowStorage[1][0] == 3);
  BOOST_TEST(compressedRowStorage[1][1] == 2);
  BOOST_TEST(compressedRowStorage[1][2] == 5);
  BOOST_TEST(compressedRowStorage[1][3] == 6);

  for (auto &crs : compressedRowStorage)
    delete[] crs;

  delete[] options;
}

BOOST_AUTO_TEST_CASE(SparseJacPatIndexSafe_c) {
  short tapeId = 1029;
  testSparseJacPat<SparseMethod::IndexDomains, ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacPatBitPatternSafe_c) {
  short tapeId = 1030;
  testSparseJacPat<SparseMethod::BitPattern, ControlFlowMode::Safe>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacPatIndexTight_c) {
  short tapeId = 1031;
  testSparseJacPat<SparseMethod::IndexDomains, ControlFlowMode::Tight>(tapeId);
}

BOOST_AUTO_TEST_CASE(SparseJacPatBitPatternTight_c) {
  short tapeId = 1032;
  testSparseJacPat<SparseMethod::BitPattern, ControlFlowMode::Tight>(tapeId);
}
BOOST_AUTO_TEST_SUITE_END()
