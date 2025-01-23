/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: advector.cpp contains a vector<adouble> implementation
           that is able to trace subscripting operations.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#include <adolc/ad_types.h>
#include <adolc/advector.h>
#include <adolc/dvlparms.h>
#include <adolc/oplate.h>
#include <adolc/taping_p.h>
#include <cmath>
#include <limits>

adubref::adubref(size_t lo, size_t ref) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  location = lo;
  refloc =
      static_cast<size_t>(trunc(fabs(ADOLC_GLOBAL_TAPE_VARS.store[location])));

  if (ref != refloc) {
    fprintf(DIAG_OUT,
            "ADOL-C error: strange construction of an active"
            " vector subscript reference\n(passed ref = %zu, stored refloc = "
            "%zu)\n",
            ref, refloc);
    adolc_exit(-2, "", __func__, __FILE__, __LINE__);
  }
}

adubref::~adubref() { free_loc(location); }

adubref &adubref::operator=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if (coval == 0) {

      put_op(ref_assign_d_zero);
      ADOLC_PUT_LOCINT(location); // = res
    } else if (coval == 1.0) {
      put_op(ref_assign_d_one);
      ADOLC_PUT_LOCINT(location); // = res
    } else {
      put_op(ref_assign_d);
      ADOLC_PUT_LOCINT(location); // = res
      ADOLC_PUT_VAL(coval);       // = coval
    }

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] = coval;
  return *this;
}

adubref &adubref::operator=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
  if (location != a.loc()) {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

      put_op(ref_assign_a);
      ADOLC_PUT_LOCINT(a.loc());  // = arg
      ADOLC_PUT_LOCINT(location); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] = a.value();
  }
  return *this;
}

adubref &adubref::operator=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_assign_p);
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] = p.value();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif

  return *this;
}

adubref::operator adouble() const {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_copyout);
    ADOLC_PUT_LOCINT(location);          // = arg
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  return ret_adouble;
}

adouble adubref::operator++(int) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ref_copyout);
    ADOLC_PUT_LOCINT(location);          // = arg
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_incr_a);
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc]++;
  return ret_adouble;
}

adouble adubref::operator--(int) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ref_copyout);
    ADOLC_PUT_LOCINT(location);          // = arg
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_decr_a);
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc]--;
  return ret_adouble;
}

adubref &adubref::operator++() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_incr_a);
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc]++;
  return *this;
}

adubref &adubref::operator--() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_decr_a);
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc]--;
  return *this;
}

adubref &adubref::operator+=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_plus_d);
    ADOLC_PUT_LOCINT(location); // = res
    ADOLC_PUT_VAL(coval);       // = coval

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] += coval;
  return *this;
}

adubref &adubref::operator+=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ref_eq_plus_a);
    ADOLC_PUT_LOCINT(a.loc());  // = arg
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] += a.value();
  return *this;
}

adubref &adubref::operator+=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_plus_p);
    ADOLC_PUT_LOCINT(p.value());
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] += p.value();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif

  return *this;
}

adubref &adubref::operator-=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_min_d);
    ADOLC_PUT_LOCINT(location); // = res
    ADOLC_PUT_VAL(coval);       // = coval

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] -= coval;
  return *this;
}

adubref &adubref::operator-=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_min_a);
    ADOLC_PUT_LOCINT(a.loc());  // = arg
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] -= a.value();
  return *this;
}

adubref &adubref::operator-=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_min_p);
    ADOLC_PUT_LOCINT(p.value());
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] -= p.value();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif

  return *this;
}

adubref &adubref::operator*=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_mult_d);
    ADOLC_PUT_LOCINT(location); // = res
    ADOLC_PUT_VAL(coval);       // = coval

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] *= coval;
  return *this;
}

adubref &adubref::operator*=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_mult_a);
    ADOLC_PUT_LOCINT(a.loc());  // = arg
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] *= a.value();
  return *this;
}

adubref &adubref::operator*=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_eq_mult_p);
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(location);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] *= p.value();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[refloc] = true;
#endif

  return *this;
}

adubref &adubref::operator<<=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ADOLC_CURRENT_TAPE_INFOS.numInds++;

    put_op(ref_assign_ind);
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[refloc] = coval;

  return *this;
}

void adubref::declareIndependent() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ADOLC_CURRENT_TAPE_INFOS.numInds++;

    put_op(ref_assign_ind);
    ADOLC_PUT_LOCINT(location); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
  }
}

adubref &adubref::operator>>=(double &coval) {
  adouble(*this) >>= coval;
  return *this;
}

void adubref::declareDependent() { adouble(*this).declareDependent(); }

void condassign(adubref &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ref_cond_assign);
    ADOLC_PUT_LOCINT(cond.loc()); // = arg
    ADOLC_PUT_VAL(cond.value());
    ADOLC_PUT_LOCINT(arg1.loc());        // = arg1
    ADOLC_PUT_LOCINT(arg2.loc());        // = arg2
    ADOLC_PUT_LOCINT(res.getLocation()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(res.value());
  }

  if (cond.value() > 0)
    res.value(arg1.value());
  else
    res.value(arg2.value());
}

void condassign(adubref &res, const adouble &cond, const adouble &arg) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_cond_assign_s);
    ADOLC_PUT_LOCINT(cond.loc()); // = arg
    ADOLC_PUT_VAL(cond.value());
    ADOLC_PUT_LOCINT(arg.loc());         // = arg1
    ADOLC_PUT_LOCINT(res.getLocation()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(res.value());
  }

  if (cond.value() > 0)
    res.value(arg.value());
}

void condeqassign(adubref &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_cond_eq_assign);
    ADOLC_PUT_LOCINT(cond.loc()); // = arg
    ADOLC_PUT_VAL(cond.value());
    ADOLC_PUT_LOCINT(arg1.loc());        // = arg1
    ADOLC_PUT_LOCINT(arg2.loc());        // = arg2
    ADOLC_PUT_LOCINT(res.getLocation()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(res.value());
  }

  if (cond.value() >= 0)
    res.value(arg1.value());
  else
    res.value(arg2.value());
}

void condeqassign(adubref &res, const adouble &cond, const adouble &arg) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(ref_cond_eq_assign_s);
    ADOLC_PUT_LOCINT(cond.loc()); // = arg
    ADOLC_PUT_VAL(cond.value());
    ADOLC_PUT_LOCINT(arg.loc());         // = arg1
    ADOLC_PUT_LOCINT(res.getLocation()); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(res.value());
  }

  if (cond.value() >= 0)
    res.value(arg.value());
}

bool advector::nondecreasing() const {
  bool ret = true;
  double last = -ADOLC_MATH_NSP::numeric_limits<double>::infinity();
  for (const auto &a : data)
    ret = ret && (a.value() >= last);
  return ret;
}

adouble advector::operator[](const adouble &index) const {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const size_t idx = (size_t)trunc(fabs(index.value()));
  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(subscript);
    ADOLC_PUT_LOCINT(index.loc());
    ADOLC_PUT_VAL(size());
    ADOLC_PUT_LOCINT(data[0].loc());
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  if (idx >= size())
    fprintf(DIAG_OUT,
            "ADOL-C warning: index out of bounds while subscripting n=%zu, "
            "idx=%zu\n",
            size(), idx);

  ret_adouble.value(data[idx].value());
  return ret_adouble;
}

adubref advector::operator[](const adouble &index) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const size_t idx = (size_t)trunc(fabs(index.value()));
  size_t locat = next_loc();
  size_t n = data.size();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(subscript_ref);
    ADOLC_PUT_LOCINT(index.loc());
    ADOLC_PUT_VAL(n);
    ADOLC_PUT_LOCINT(data[0].loc());
    ADOLC_PUT_LOCINT(locat);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
  }

  if (idx >= n)
    fprintf(DIAG_OUT,
            "ADOL-C warning: index out of bounds while subscripting (ref) "
            "n=%zu, idx=%zu\n",
            n, idx);

  ADOLC_GLOBAL_TAPE_VARS.store[locat] = data[idx].loc();
  return adubref(locat, data[idx].loc());
}

adouble advector::lookupindex(const adouble &a, const adouble &b) const {
  if (!nondecreasing()) {
    fprintf(DIAG_OUT, "ADOL-C error: can only call lookup index if advector "
                      "ist nondecreasing\n");
    adolc_exit(-2, "", __func__, __FILE__, __LINE__);
  }
  if (b.value() < 0) {
    fprintf(DIAG_OUT,
            "ADOL-C error: index lookup needs a nonnegative denominator\n");
    adolc_exit(-2, "", __func__, __FILE__, __LINE__);
  }
  adouble r = 0;
  for (size_t i = 0; i < size(); ++i)
    condassign(r, a - data[i] * b, adouble(i + 1));
  return r;
}
