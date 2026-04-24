#define BOOST_TEST_DYN_LINK
#include "../const.h"
#include <adolc/adolc.h>
#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cstdio>

BOOST_AUTO_TEST_SUITE(Test_TapeInfos)

BOOST_AUTO_TEST_CASE(TestConstructorTapeId) {
  TapeInfos ti(3);

  BOOST_CHECK(ti.tapeId_ == 3);
}

BOOST_AUTO_TEST_CASE(TestMoveConstructor) {
  TapeInfos tp(3);
  tp.numInds = 10;
  tp.numDeps = 11;

  tp.keepTaylors = 1;
  tp.stats[2] = 5;

  auto opBuffer = new unsigned char[10];
  tp.opBuffer_ = ADOLC::detail::OpBuffer(opBuffer, 10);
  tp.opBuffer_.openFile("test_move_constr_op.txt", "w");
  tp.opBuffer_.position(3);
  tp.opBuffer_.numOnTape(10);

  tp.num_eq_prod = 4;

  auto valBuffer = new double[123];
  tp.valBuffer_ = ADOLC::detail::ValBuffer(valBuffer, 123);
  tp.valBuffer_.openFile("test_move_constr_val.txt", "w");
  tp.valBuffer_.position(15);
  tp.valBuffer_.numOnTape(123);

  auto locBuffer = new size_t[14];
  tp.locBuffer_ = ADOLC::detail::LocBuffer(locBuffer, 14);
  tp.locBuffer_.openFile("test_move_constr_loc.txt", "w");
  tp.locBuffer_.position(3);
  tp.locBuffer_.numOnTape(13);

  auto tayBuffer = new double[14];
  tp.tayBuffer_ = ADOLC::detail::TayBuffer(tayBuffer, 14);
  tp.tayBuffer_.openFile("test_move_constr_tay.txt", "w");
  tp.tayBuffer_.position(4);
  tp.tayBuffer_.numOnTape(13);

  tp.nextBufferNumber = 4;

  tp.lastTayBlockInCore = 1;

  tp.deg_save = 1;
  tp.tay_numInds = 3;
  tp.tay_numDeps = 5;

  tp.workMode = TapeInfos::NO_MODE;

  tp.ext_diff_fct_index = 5;
  tp.nestedReverseEval = true;
  tp.numSwitches = 6;

  auto l = new double[23];
  tp.signature = l;

  //// ======= MOVE CONSTRUCTOR =========
  //======================================
  TapeInfos tp2(std::move(tp));
  BOOST_CHECK_EQUAL(tp2.tapeId_, 3);
  BOOST_CHECK_EQUAL(tp2.numInds, 10);
  BOOST_CHECK_EQUAL(tp2.numDeps, 11);
  BOOST_CHECK_EQUAL(tp2.keepTaylors, 1);
  BOOST_CHECK_EQUAL(tp2.stats[2], 5);
  BOOST_CHECK_EQUAL(tp2.opBuffer_.numOnTape(), 10);
  BOOST_CHECK_EQUAL(tp2.valBuffer_.numOnTape(), 123);
  BOOST_CHECK_EQUAL(tp2.num_eq_prod, 4);

  BOOST_CHECK_EQUAL(tp2.locBuffer_.numOnTape(), 13);
  BOOST_CHECK_EQUAL(tp2.tayBuffer_.numOnTape(), 13);
  BOOST_CHECK_EQUAL(tp2.nextBufferNumber, 4);
  BOOST_CHECK_EQUAL(tp2.lastTayBlockInCore, 1);
  BOOST_CHECK_EQUAL(tp2.deg_save, 1);
  BOOST_CHECK_EQUAL(tp2.tay_numInds, 3);
  BOOST_CHECK_EQUAL(tp2.tay_numDeps, 5);
  BOOST_CHECK_EQUAL(tp2.workMode, TapeInfos::NO_MODE);
  BOOST_CHECK_EQUAL(tp2.ext_diff_fct_index, 5);
  BOOST_CHECK_EQUAL(tp2.nestedReverseEval, true);
  BOOST_CHECK_EQUAL(tp2.numSwitches, 6);

  // === Validate: pointers moved ===
  BOOST_CHECK_EQUAL(tp2.opBuffer_.file() != nullptr, true);
  BOOST_CHECK_EQUAL(tp2.opBuffer_.begin(), opBuffer);
  BOOST_CHECK_EQUAL(tp2.opBuffer_.position(), 3);
  BOOST_CHECK_EQUAL(tp2.valBuffer_.file() != nullptr, true);
  BOOST_CHECK_EQUAL(tp2.valBuffer_.begin(), valBuffer);
  BOOST_CHECK_EQUAL(tp2.valBuffer_.position(), 15);

  BOOST_CHECK_EQUAL(tp2.locBuffer_.file() != nullptr, true);
  BOOST_CHECK_EQUAL(tp2.locBuffer_.begin(), locBuffer);
  BOOST_CHECK_EQUAL(tp2.locBuffer_.position(), 3);

  BOOST_CHECK_EQUAL(tp2.tayBuffer_.file() != nullptr, true);
  BOOST_CHECK_EQUAL(tp2.tayBuffer_.begin(), tayBuffer);
  BOOST_CHECK_EQUAL(tp2.tayBuffer_.position(), 4);
  BOOST_CHECK_EQUAL(tp2.signature, l);
}

BOOST_AUTO_TEST_SUITE_END()
