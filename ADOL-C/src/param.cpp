/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     param.cpp
 Revision: $Id$
 Contents: class for parameter dependent functions

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include "dvlparms.h"
#include "oplate.h"
#include "taping_p.h"
#include <adolc/adouble.h>

#include <limits>

pdouble::pdouble(double pval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  _val = pval;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    _idx = ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc();
    ADOLC_GLOBAL_TAPE_VARS.pStore[_idx] = _val;
  } else {
    _idx = std::numeric_limits<locint>::max();
  }
}

pdouble::pdouble(locint idx) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (idx < ADOLC_GLOBAL_TAPE_VARS.numparam) {
    _val = ADOLC_GLOBAL_TAPE_VARS.pStore[idx];
    _idx = idx;
  } else {
    fprintf(DIAG_OUT,
            "ADOL-C error: Parameter index %zu out of bounds, "
            "# existing parameters = %zu\n",
            idx, ADOLC_GLOBAL_TAPE_VARS.numparam);
    adolc_exit(-1, "", __func__, __FILE__, __LINE__);
  }
}

pdouble::operator pdouble *() const {
  pdouble *ret = new pdouble(_idx);
  return ret;
}

pdouble padouble::mkparam(double pval) {
  locint _idx;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    _idx = ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc();
    ADOLC_GLOBAL_TAPE_VARS.pStore[_idx] = pval;
  } else {
    return pval;
  }
  return _idx;
}

pdouble getparam(locint index) { return index; }

locint mkparam_idx(double pval) {
  locint _idx;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    _idx = ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc();
    ADOLC_GLOBAL_TAPE_VARS.pStore[_idx] = pval;
  } else {
    fprintf(DIAG_OUT, "ADOL-C error: cannot define indexed parameter "
                      "while tracing is turned off!\n");
    adolc_exit(-1, "", __func__, __FILE__, __LINE__);
  }
  return _idx;
}

pdouble::operator adub() const {
  locint location;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  location = next_loc();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(assign_p);
    ADOLC_PUT_LOCINT(_idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[location] = _val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[location] = true;
#endif
  return location;
}

badouble &badouble::operator=(const pdouble &p) {
  locint loc;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  loc = this->loc();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(assign_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(loc);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[loc] = p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[loc] = true;
#endif
  return *this;
}

adouble &adouble::operator=(const pdouble &p) {
  this->loc();
  (*this).badouble::operator=(p);
  return (*this);
}

adubref &adubref::operator=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ref_assign_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[refloc] = p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif
  return *this;
}

double pdouble::getValue() const {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  return ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc_.loc_];
}

badouble &badouble::operator+=(const pdouble &p) {
  locint loc;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  loc = this->loc();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(eq_plus_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(loc);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[loc] += p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[loc] = true;
#endif
  return *this;
}

adubref &adubref::operator+=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ref_eq_plus_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] += p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif
  return *this;
}

badouble &badouble::operator-=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_d_same_arg(eq_min_d,loc(),coval);
    put_op(eq_min_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[loc()] -= p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = true;
#endif
  return *this;
}

adubref &adubref::operator-=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_d_same_arg(eq_min_d,location,coval);
    put_op(ref_eq_min_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] -= p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif
  return *this;
}

badouble &badouble::operator*=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_d_same_arg(eq_mult_d,loc(),coval);
    put_op(eq_mult_p);
    ADOLC_PUT_LOCINT(p._idx); // = coval
    ADOLC_PUT_LOCINT(loc());  // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[loc()] *= p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = true;
#endif
  return *this;
}

adubref &adubref::operator*=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_d_same_arg(eq_mult_d,location,coval);
    put_op(ref_eq_mult_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] *= p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif
  return *this;
}

adub operator+(const badouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  locint locat = next_loc();

  /* olvo 980708 test coval to be zero */
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_args_d_a(plus_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      locint tmploc = a.loc();
      double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tmploc);
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tmploc);
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tmploc);
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    put_op(plus_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(locat);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[locat] =
      ADOLC_GLOBAL_TAPE_VARS.store[a.loc()] + p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = true;
#endif
  return locat;
}

adub operator-(const badouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  locint locat = next_loc();

  /* olvo 980708 test coval to be zero */
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_args_d_a(plus_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      locint tmploc = a.loc();
      double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tmploc);
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tmploc);
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tmploc);
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    put_op(min_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(locat);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[locat] =
      ADOLC_GLOBAL_TAPE_VARS.store[a.loc()] - p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = true;
#endif
  return locat;
}

adub operator*(const badouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  locint locat = next_loc();

  /* olvo 980708 test coval to be zero */
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_args_d_a(plus_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      locint tmploc = a.loc();
      double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tmploc);
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tmploc);
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tmploc);
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    put_op(mult_a_p);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(locat);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[locat] =
      ADOLC_GLOBAL_TAPE_VARS.store[a.loc()] * p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = true;
#endif
  return locat;
}

adub operator/(const pdouble &p, const badouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  locint locat = next_loc();

  /* olvo 980708 test coval to be zero */
  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_args_d_a(plus_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      locint tmploc = a.loc();
      double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tmploc);
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tmploc);
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tmploc);
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    put_op(div_p_a);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(locat);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[locat] =
      p._val / ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = true;
#endif
  return locat;
}

adub pow(const badouble &x, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  locint locat = next_loc();

  if (ADOLC_CURRENT_TAPE_INFOS
          .traceFlag) { // old: write_args_d_a(pow_op,locat,cocval,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
    if (!ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
      locint tmploc = x.loc();
      double temp = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tmploc);
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tmploc);
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tmploc);
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    put_op(pow_op_p);
    ADOLC_PUT_LOCINT(x.loc()); // = arg
    ADOLC_PUT_LOCINT(p._idx);  // = coval
    ADOLC_PUT_LOCINT(locat);   // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[locat] =
      ADOLC_MATH_NSP::pow(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()], p._val);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = true;
#endif
  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return locat;
}
adub recipr(const pdouble &p) {
  locint location;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  location = next_loc();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(recipr_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[location] = 1.0 / p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[location] = true;
#endif
  return location;
}

adub operator-(const pdouble &p) {
  locint location;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  location = next_loc();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(neg_sign_p);
    ADOLC_PUT_LOCINT(p._idx);
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[location] = -p._val;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[location] = true;
#endif
  return location;
}

adouble pow(const pdouble &p, const badouble &y) {
  adouble a1, a2, ret;
  double vx = p._val;
  double vy = y.getValue();
  if (!(vx > 0)) {
    if (vx < 0 || vy >= 0)
      fprintf(
          DIAG_OUT,
          "\nADOL-C message: exponent of zero/negative basis deactivated\n");
    else
      fprintf(
          DIAG_OUT,
          "\nADOL-C message: negative exponent and zero basis deactivated\n");
  }
  condassign(a1, -y, (adouble)ADOLC_MATH_NSP::pow(vx, vy), pow(p, vy));
  condassign(a2, fabs(adub(p)), pow(p, vy), a1);
  condassign(ret, p, exp(y * log(adub(p))), a2);

  return ret;
}

#ifdef ADOLC_ADVANCED_BRANCHING
adouble operator!=(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.getValue();
  const double p_coval = p.getValue();
  const double res = static_cast<double>(a_coval != p_coval);
  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(neq_a_p);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(p.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator!=(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() != p.getValue());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(neq_a_p);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(p.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator!=(const pdouble &p, const adouble &a) { return (a != p); }
adouble operator!=(const pdouble &p, adouble &&a) {
  return (std::move(a) != p);
}

adouble operator==(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() == p.getValue());
  ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_a_p);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(p.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;

  return ret_adouble;
}

adouble operator==(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() == p.getValue());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_a_p);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(p.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;

  return a;
}

adouble operator==(const pdouble &p, const adouble &a) { return (a == p); }
adouble operator==(const pdouble &p, adouble &&a) {
  return (std::move(a) == p);
}

adouble operator<=(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() <= p.getValue());
  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(le_a_p);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(p.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator<=(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() <= p.getValue());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(le_a_p);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(p.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator<=(const pdouble &p, const adouble &a) { return (a >= p); }
adouble operator<=(const pdouble &p, adouble &&a) {
  return (std::move(a) >= p);
}

adouble operator>=(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getLoc() >= p.getLoc());
  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ge_a_p);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(p.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator>=(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getLoc() >= p.getLoc());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ge_a_p);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(p.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}
adouble operator>=(const pdouble &p, const adouble &a) { return (a <= p); }
adouble operator>=(const pdouble &p, adouble &&a) {
  return (std::move(a) <= p);
}

adouble operator>(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() > p.getValue());
  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(gt_a_p);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(p.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator>(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() > p.getValue());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(gt_a_p);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(p.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator>(const pdouble &p, const adouble &a) { return (a < p); }
adouble operator>(const pdouble &p, adouble &&a) { return (std::move(a) < p); }

adouble operator<(const adouble &a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() < p.getValue());
  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(lt_a_p);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(p.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator<(adouble &&a, const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double res = static_cast<double>(a.getValue() < p.getValue());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(lt_a_p);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(p.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}
adouble operator<(const pdouble &p, const adouble &a) { return (a > p); }
adouble operator<(const pdouble &p, adouble &&a) { return (std::move(a) > p); }
#endif
