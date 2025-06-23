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

#include <adolc/adolcerror.h>
#include <adolc/adtb_types.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>

pdouble::pdouble(const double pval) { value(pval); }

pdouble::operator adouble() const {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  if (tape.traceFlag()) {
    tape.put_op(assign_p);
    tape.put_loc(loc());
    tape.put_loc(ret_adouble.loc());

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(value());

#if defined(ADOLC_TRACK_ACTIVITY)
  tape->globalTapeVars_.actStore[ret_adouble.loc()] = true;
#endif

  return ret_adouble;
}
