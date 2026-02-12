#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
namespace tt = boost::test_tools;

#include <adolc/adolc.h>
#include <cstdlib>

#include "../const.h"

BOOST_AUTO_TEST_SUITE(test_sparse_jacobian)

using ADOLC::Sparse::BitPatternPropagationDirection;
using ADOLC::Sparse::CompressionMode;
using ADOLC::Sparse::ControlFlowMode;
using ADOLC::Sparse::SparseMethod;

template <
    SparseMethod SM, CompressionMode CM, ControlFlowMode CFM,
    BitPatternPropagationDirection BPPD = BitPatternPropagationDirection::Auto>
static void testSparseJac() {
  const auto tapeId = createNewTape();
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

  ADOLC::Sparse::sparse_jac<SM, CM, CFM, BPPD>(
      tapeId, dimOut, dimIn, 0, in.data(), &numberOfNonzeros, &rowIndices,
      &columnIndices, &nonzeroValues);
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
  ADOLC::Sparse::sparse_jac<SM, CM, CFM, BPPD>(
      tapeId, dimOut, dimIn, 0, in.data(), &numberOfNonzeros, &rowIndices,
      &columnIndices, &nonzeroValues);
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
  ADOLC::Sparse::sparse_jac<SM, CM, CFM, BPPD>(
      tapeId, dimOut, dimIn, 1, in.data(), &numberOfNonzeros, &rowIndices,
      &columnIndices, &nonzeroValues);
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
}

BOOST_AUTO_TEST_CASE(SparseJacIndexColSafe) {
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Column,
                ControlFlowMode::Safe>();
}

BOOST_AUTO_TEST_CASE(SparseJacIndexColTight) {
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Column,
                ControlFlowMode::Tight>();
}

BOOST_AUTO_TEST_CASE(SparseJacIndexRowSafe) {
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Row,
                ControlFlowMode::Safe>();
}

BOOST_AUTO_TEST_CASE(SparseJacIndexRowTight) {
  testSparseJac<SparseMethod::IndexDomains, CompressionMode::Row,
                ControlFlowMode::Tight>();
}
BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropTight) {
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Tight>();
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropSafe) {
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Safe>();
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropForwardTight) {
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Tight,
                BitPatternPropagationDirection::Forward>();
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropForwardSafe) {
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Safe,
                BitPatternPropagationDirection::Forward>();
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropReverseTight) {
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Tight,
                BitPatternPropagationDirection::Reverse>();
}

BOOST_AUTO_TEST_CASE(SparseJacBitPatterPropReverseSafe) {
  testSparseJac<SparseMethod::BitPattern, CompressionMode::Row,
                ControlFlowMode::Safe,
                BitPatternPropagationDirection::Reverse>();
}

template <SparseMethod SM, ControlFlowMode CFM> static void testSparseJacPat() {
  const auto tapeId = createNewTape();
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
  std::span<uint *> compressedRowStorageSpan(compressedRowStorage);
  ADOLC::Sparse::jac_pat<SM, CFM>(tapeId, dimOut, dimIn, in.data(),
                                  compressedRowStorageSpan);

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
}

BOOST_AUTO_TEST_CASE(SparseJacPatIndexSafe) {
  testSparseJacPat<SparseMethod::IndexDomains, ControlFlowMode::Safe>();
}

BOOST_AUTO_TEST_CASE(SparseJacPatBitPatternSafe) {
  testSparseJacPat<SparseMethod::BitPattern, ControlFlowMode::Safe>();
}

BOOST_AUTO_TEST_CASE(SparseJacPatIndexTight) {
  testSparseJacPat<SparseMethod::IndexDomains, ControlFlowMode::Tight>();
}

BOOST_AUTO_TEST_CASE(SparseJacPatBitPatternTight) {
  testSparseJacPat<SparseMethod::BitPattern, ControlFlowMode::Tight>();
}
BOOST_AUTO_TEST_SUITE_END()
