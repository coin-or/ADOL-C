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
  tp.inUse = 1;
  tp.numInds = 10;
  tp.numDeps = 11;

  tp.keepTaylors = 1;
  tp.stats[2] = 5;
  tp.traceFlag = 8;
  tp.tapingComplete = 1;

  auto fileH = fopen("test_move_constr_op.txt", "w");
  tp.op_file = fileH;
  auto opBuffer = new unsigned char[10];
  tp.opBuffer = opBuffer;
  tp.currOp = tp.opBuffer + 3;
  tp.lastOpP1 = tp.opBuffer + 11;

  tp.numOps_Tape = 10;
  tp.num_eq_prod = 4;

  auto fileH2 = fopen("test_move_constr_val.txt", "w");
  tp.val_file = fileH2;
  auto valBuffer = new double[12];
  tp.valBuffer = valBuffer;
  tp.currVal = tp.valBuffer + 3;
  tp.lastValP1 = tp.valBuffer + 11;

  tp.numVals_Tape = 70;

  auto fileH3 = fopen("test_move_constr_loc.txt", "w");
  tp.loc_file = fileH3;
  auto locBuffer = new size_t[14];

  tp.locBuffer = locBuffer;
  tp.currLoc = tp.locBuffer + 3;
  tp.lastLocP1 = tp.locBuffer + 13;

  tp.numLocs_Tape = 13;

  auto fileH4 = fopen("test_move_constr_tay.txt", "w");
  tp.tay_file = fileH4;
  auto tayBuffer = new double[14];
  tp.tayBuffer = tayBuffer;
  tp.currTay = tp.tayBuffer + 4;
  tp.lastTayP1 = tp.tayBuffer + 13;

  tp.numTays_Tape = 13;
  tp.numTBuffersInUse = 1;

  tp.nextBufferNumber = 4;

  tp.lastTayBlockInCore = 1;

  auto a = new double *[10];
  tp.T_for = a;
  tp.deg_save = 1;
  tp.tay_numInds = 3;
  tp.tay_numDeps = 5;

  tp.lowestXLoc_for = 10;
  tp.lowestYLoc_for = 17;
  tp.lowestXLoc_rev = 20;
  tp.lowestYLoc_rev = 21;
  tp.cpIndex = 12;
  tp.numDirs_rev = 4;

  auto b = new size_t[4];
  tp.lowestXLoc_ext_v2 = b;
  auto c = new size_t[6];
  tp.lowestYLoc_ext_v2 = c;

  auto d = new double[3];
  tp.dp_T0 = d;

  tp.gDegree = 31;
  tp.numTay = 45;
  tp.workMode = TapeInfos::TAPING;

  auto e = new double *[10];
  tp.dpp_T = e;
  auto f = new double[11];
  tp.rp_T = f;
  auto g = new double *[4];
  tp.rpp_T = g;
  auto h = new double[2];
  tp.rp_A = h;
  auto i = new double *[22];
  tp.rpp_A = i;
  auto j = new size_t *[3];
  tp.upp_A = j;

  tp.ext_diff_fct_index = 5;
  tp.in_nested_ctx = 4;
  tp.numSwitches = 6;

  auto k = new size_t[3];
  tp.switchlocs = k;
  auto l = new double[23];
  tp.signature = l;

  //// ======= MOVE CONSTRUCTOR =========
  //======================================
  TapeInfos tp2(std::move(tp));
  BOOST_CHECK_EQUAL(tp2.tapeId_, 3);
  BOOST_CHECK_EQUAL(tp2.inUse, 1);
  BOOST_CHECK_EQUAL(tp2.numInds, 10);
  BOOST_CHECK_EQUAL(tp2.numDeps, 11);
  BOOST_CHECK_EQUAL(tp2.keepTaylors, 1);
  BOOST_CHECK_EQUAL(tp2.stats[2], 5);
  BOOST_CHECK_EQUAL(tp2.traceFlag, 8);
  BOOST_CHECK_EQUAL(tp2.tapingComplete, 1);
  BOOST_CHECK_EQUAL(tp2.numOps_Tape, 10);
  BOOST_CHECK_EQUAL(tp2.num_eq_prod, 4);
  BOOST_CHECK_EQUAL(tp2.numVals_Tape, 70);
  BOOST_CHECK_EQUAL(tp2.numLocs_Tape, 13);
  BOOST_CHECK_EQUAL(tp2.numTays_Tape, 13);
  BOOST_CHECK_EQUAL(tp2.numTBuffersInUse, 1);
  BOOST_CHECK_EQUAL(tp2.nextBufferNumber, 4);
  BOOST_CHECK_EQUAL(tp2.lastTayBlockInCore, 1);
  BOOST_CHECK_EQUAL(tp2.deg_save, 1);
  BOOST_CHECK_EQUAL(tp2.tay_numInds, 3);
  BOOST_CHECK_EQUAL(tp2.tay_numDeps, 5);
  BOOST_CHECK_EQUAL(tp2.lowestXLoc_for, 10);
  BOOST_CHECK_EQUAL(tp2.lowestYLoc_for, 17);
  BOOST_CHECK_EQUAL(tp2.lowestXLoc_rev, 20);
  BOOST_CHECK_EQUAL(tp2.lowestYLoc_rev, 21);
  BOOST_CHECK_EQUAL(tp2.cpIndex, 12);
  BOOST_CHECK_EQUAL(tp2.numDirs_rev, 4);
  BOOST_CHECK_EQUAL(tp2.gDegree, 31);
  BOOST_CHECK_EQUAL(tp2.numTay, 45);
  BOOST_CHECK_EQUAL(tp2.workMode, TapeInfos::TAPING);
  BOOST_CHECK_EQUAL(tp2.ext_diff_fct_index, 5);
  BOOST_CHECK_EQUAL(tp2.in_nested_ctx, 4);
  BOOST_CHECK_EQUAL(tp2.numSwitches, 6);

  // === Validate: pointers moved ===
  BOOST_CHECK_EQUAL(tp2.op_file, fileH);
  BOOST_CHECK_EQUAL(tp2.opBuffer, opBuffer);
  BOOST_CHECK_EQUAL(tp2.currOp, opBuffer + 3);
  BOOST_CHECK_EQUAL(tp2.lastOpP1, opBuffer + 11);

  BOOST_CHECK_EQUAL(tp2.val_file, fileH2);
  BOOST_CHECK_EQUAL(tp2.valBuffer, valBuffer);
  BOOST_CHECK_EQUAL(tp2.currVal, valBuffer + 3);
  BOOST_CHECK_EQUAL(tp2.lastValP1, valBuffer + 11);

  BOOST_CHECK_EQUAL(tp2.loc_file, fileH3);
  BOOST_CHECK_EQUAL(tp2.locBuffer, locBuffer);
  BOOST_CHECK_EQUAL(tp2.currLoc, locBuffer + 3);
  BOOST_CHECK_EQUAL(tp2.lastLocP1, locBuffer + 13);

  BOOST_CHECK_EQUAL(tp2.tay_file, fileH4);
  BOOST_CHECK_EQUAL(tp2.tayBuffer, tayBuffer);
  BOOST_CHECK_EQUAL(tp2.currTay, tayBuffer + 4);
  BOOST_CHECK_EQUAL(tp2.lastTayP1, tayBuffer + 13);

  BOOST_CHECK_EQUAL(tp2.T_for, a);
  BOOST_CHECK_EQUAL(tp2.lowestXLoc_ext_v2, b);
  BOOST_CHECK_EQUAL(tp2.lowestYLoc_ext_v2, c);
  BOOST_CHECK_EQUAL(tp2.dp_T0, d);
  BOOST_CHECK_EQUAL(tp2.dpp_T, e);
  BOOST_CHECK_EQUAL(tp2.rp_T, f);
  BOOST_CHECK_EQUAL(tp2.rpp_T, g);
  BOOST_CHECK_EQUAL(tp2.rp_A, h);
  BOOST_CHECK_EQUAL(tp2.rpp_A, i);
  BOOST_CHECK_EQUAL(tp2.upp_A, j);
  BOOST_CHECK_EQUAL(tp2.switchlocs, k);
  BOOST_CHECK_EQUAL(tp2.signature, l);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
  delete[] g;
  delete[] h;
  delete[] i;
  delete[] j;
}

BOOST_AUTO_TEST_SUITE_END()