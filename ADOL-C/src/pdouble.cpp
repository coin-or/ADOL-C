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
  }
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

adouble operator-(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(neg_sign_p);
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(-p.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif

  return ret_adouble;
}

adouble operator+(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif

    put_op(plus_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(a.value() + p.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif

    put_op(plus_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(a.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(a.value() + p.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] = true;
#endif
  return a;
}

adouble operator-(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif

    put_op(min_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(a.value() - p.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif

    put_op(min_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(a.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(a.value() - p.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] = true;
#endif
  return a;
}

adouble operator*(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif

    put_op(mult_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(a.value() * p.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif
  return ret_adouble;
}

adouble operator*(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif

    put_op(mult_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(a.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(a.value() * p.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] = true;
#endif
  return a;
}

adouble recipr(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(recipr_p);
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(1.0 / p.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.value()] = true;
#endif

  return ret_adouble;
}

adouble operator/(const pdouble &p, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.getValue);
    }
#endif

    put_op(div_p_a);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)

      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(p.value() / a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif

  return ret_adouble;
}

adouble operator/(const pdouble &p, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.getValue);
    }
#endif

    put_op(div_p_a);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(a.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)

      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(p.value() / a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] = true;
#endif

  return a;
}

adouble pow(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
    put_op(pow_op_p);
    ADOLC_PUT_LOCINT(a.loc());           // = arg
    ADOLC_PUT_LOCINT(p.loc());           // = coval
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(ADOLC_MATH_NSP::pow(a.value(), p.value()));

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] = true;
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble pow(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (a.value() == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (a.value() == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(a.value());
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
    put_op(pow_op_p);
    ADOLC_PUT_LOCINT(a.loc()); // = arg
    ADOLC_PUT_LOCINT(p.loc()); // = coval
    ADOLC_PUT_LOCINT(a.loc()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(ADOLC_MATH_NSP::pow(a.value(), p.value()));

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] = true;
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble pow(const pdouble &p, const adouble &a) {
  assert((p.value() < 0) && "\nADOL-C message: negative basis deactivated\n ");
  assert(p.value() == 0 && "\nADOL-C message: zero basis deactivated\n ");

  adouble a1, a2, ret;
  condassign(a1, -a, adouble(ADOLC_MATH_NSP::pow(p.value(), a.value())),
             pow(p, a.value()));
  condassign(a2, fabs(p), pow(p, a.value()), a1);
  condassign(ret, adouble(p), exp(a * log(p)), a2);
  return ret;
}

#ifdef ADOLC_ADVANCED_BRANCHING
adouble operator!=(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double p_coval = p.value();
  const double res = static_cast<double>(a_coval != p_coval);
  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(neq_a_p);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(p.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator!=(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() != p.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(neq_a_p);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(p.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator==(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() == p.value());
  ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_a_p);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(p.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);

  return ret_adouble;
}

adouble operator==(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() == p.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_a_p);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(p.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }
  a.value(res);

  return a;
}

adouble operator<=(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() <= p.value());
  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(le_a_p);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(p.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<=(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() <= p.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(le_a_p);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(p.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>=(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.loc() >= p.loc());
  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ge_a_p);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(p.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>=(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.loc() >= p.loc());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ge_a_p);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(p.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }
  a.value(res);
  return a;
}

adouble operator>(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() > p.value());
  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(gt_a_p);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(p.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() > p.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(gt_a_p);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(p.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }
  a.value(res);
  return a;
}

adouble operator<(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() < p.value());
  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(lt_a_p);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(p.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.value() < p.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(lt_a_p);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(p.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }
  a.value(res);
  return a;
}
#endif
