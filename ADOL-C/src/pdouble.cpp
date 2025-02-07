/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     pdouble.cpp
 Revision: $Id$
 Contents: This file specifies the definitions for pdouble class and
           corresponding arithmetics

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adtb_types.h>
#include <adolc/oplate.h>
#include <cassert>

pdouble::pdouble(const double pval) {
#include <limits>
  using std::numeric_limits;

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    tape_loc_ =
        tape_location{ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc()};
    ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc_.loc_] = pval;
  } else
    tape_loc_ = tape_location{numeric_limits<size_t>::max()};
}

pdouble::pdouble(tape_location tape_loc) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (tape_loc.loc_ < ADOLC_GLOBAL_TAPE_VARS.numparam) {
    tape_loc_ = tape_loc;
  } else {
    fprintf(DIAG_OUT,
            "ADOL-C error: Parameter index %zu out of bounds, "
            "# existing parameters = %zu\n",
            tape_loc.loc_, ADOLC_GLOBAL_TAPE_VARS.numparam);
    adolc_exit(-1, "", __func__, __FILE__, __LINE__);
  }
}

pdouble pdouble::mkparam(const double pval) {
#include <limits>
  using std::numeric_limits;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    tape_location tape_loc{ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc()};
    ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc.loc_] = pval;
    return pdouble(tape_loc);
  }
  return pdouble(tape_location{numeric_limits<size_t>::max()});
}

pdouble::operator adouble() const {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(assign_p);
    ADOLC_PUT_LOCINT(tape_loc_.loc_);
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(this->value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif

  return ret_adouble;
}
