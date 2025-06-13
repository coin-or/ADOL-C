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

#include <adolc/adolcerror.h>
#include <adolc/adtb_types.h>
#include <adolc/advector.h>
#include <adolc/dvlparms.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <cmath>
#include <limits>

adubref::adubref(size_t lo, size_t ref) {
  ValueTape &tape = currentTape();
  loc_ = lo;
  refloc_ = static_cast<size_t>(trunc(fabs(tape.get_ad_value(loc_))));

  if (ref != refloc_) {
    ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_CONSTRUCTOR,
                     CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info5 = ref, .info6 = refloc_});
  }
}

adubref::~adubref() { currentTape().free_loc(loc_); }

adubref &adubref::operator=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    if (coval == 0) {

      tape.put_op(ref_assign_d_zero);
      tape.put_loc(loc_); // = res
    } else if (coval == 1.0) {
      tape.put_op(ref_assign_d_one);
      tape.put_loc(loc_); // = res
    } else {
      tape.put_op(ref_assign_d);
      tape.put_loc(loc_);  // = res
      tape.put_val(coval); // = coval
    }

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(coval);
  return *this;
}

adubref &adubref::operator=(const adouble &a) {
  ValueTape &tape = currentTape();
  /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
  if (loc_ != a.loc()) {
    if (tape.traceFlag()) {

      tape.put_op(ref_assign_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(loc_);    // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(refloc_value());
    }

    refloc_value(a.value());
  }
  return *this;
}

adubref::operator adouble() const {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  if (tape.traceFlag()) {

    tape.put_op(ref_copyout);
    tape.put_loc(loc_);              // = arg
    tape.put_loc(ret_adouble.loc()); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(refloc_value());
  return ret_adouble;
}

adouble adubref::operator++(int) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  if (tape.traceFlag()) {
    tape.put_op(ref_copyout);
    tape.put_loc(loc_);              // = arg
    tape.put_loc(ret_adouble.loc()); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(refloc_value());

  if (tape.traceFlag()) {

    tape.put_op(ref_incr_a);
    tape.put_loc(loc_); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() + 1);
  return ret_adouble;
}

adouble adubref::operator--(int) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  if (tape.traceFlag()) {
    tape.put_op(ref_copyout);
    tape.put_loc(loc_);              // = arg
    tape.put_loc(ret_adouble.loc()); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(refloc_value());

  if (tape.traceFlag()) {

    tape.put_op(ref_decr_a);
    tape.put_loc(loc_); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() - 1);
  return ret_adouble;
}

adubref &adubref::operator++() {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_incr_a);
    tape.put_loc(loc_); // = res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() + 1);
  return *this;
}

adubref &adubref::operator--() {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_decr_a);
    tape.put_loc(loc_); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() - 1);
  return *this;
}

adubref &adubref::operator+=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_eq_plus_d);
    tape.put_loc(loc_);  // = res
    tape.put_val(coval); // = coval

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() + coval);
  return *this;
}

adubref &adubref::operator+=(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    tape.put_op(ref_eq_plus_a);
    tape.put_loc(a.loc()); // = arg
    tape.put_loc(loc_);    // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() + a.value());
  return *this;
}

adubref &adubref::operator-=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_eq_min_d);
    tape.put_loc(loc_);  // = res
    tape.put_val(coval); // = coval

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() - coval);
  return *this;
}

adubref &adubref::operator-=(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_eq_min_a);
    tape.put_loc(a.loc()); // = arg
    tape.put_loc(loc_);    // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() - a.value());
  return *this;
}

adubref &adubref::operator*=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_eq_mult_d);
    tape.put_loc(loc_);  // = res
    tape.put_val(coval); // = coval

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() * coval);
  return *this;
}

adubref &adubref::operator*=(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_eq_mult_a);
    tape.put_loc(a.loc()); // = arg
    tape.put_loc(loc_);    // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(refloc_value() * a.value());
  return *this;
}

adubref &adubref::operator<<=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    tape.increment_numInds();

    tape.put_op(ref_assign_ind);
    tape.put_loc(loc_); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }

  refloc_value(coval);

  return *this;
}

void adubref::declareIndependent() {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    tape.increment_numInds();

    tape.put_op(ref_assign_ind);
    tape.put_loc(loc_); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(refloc_value());
  }
}

adubref &adubref::operator>>=(double &coval) {
  adouble(*this) >>= coval;
  return *this;
}

void adubref::declareDependent() { adouble(*this).declareDependent(); }

void condassign(adubref &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    tape.put_op(ref_cond_assign);
    tape.put_loc(cond.loc()); // = arg
    tape.put_val(cond.value());
    tape.put_loc(arg1.loc()); // = arg1
    tape.put_loc(arg2.loc()); // = arg2
    tape.put_loc(res.loc());  // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(res.refloc_value());
  }

  if (cond.value() > 0)
    res.refloc_value(arg1.value());
  else
    res.refloc_value(arg2.value());
}

void condassign(adubref &res, const adouble &cond, const adouble &arg) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_cond_assign_s);
    tape.put_loc(cond.loc()); // = arg
    tape.put_val(cond.value());
    tape.put_loc(arg.loc()); // = arg1
    tape.put_loc(res.loc()); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(res.refloc_value());
  }

  if (cond.value() > 0)
    res.refloc_value(arg.value());
}

void condeqassign(adubref &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_cond_eq_assign);
    tape.put_loc(cond.loc()); // = arg
    tape.put_val(cond.value());
    tape.put_loc(arg1.loc()); // = arg1
    tape.put_loc(arg2.loc()); // = arg2
    tape.put_loc(res.loc());  // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(res.refloc_value());
  }

  if (cond.value() >= 0)
    res.refloc_value(arg1.value());
  else
    res.refloc_value(arg2.value());
}

void condeqassign(adubref &res, const adouble &cond, const adouble &arg) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(ref_cond_eq_assign_s);
    tape.put_loc(cond.loc()); // = arg
    tape.put_val(cond.value());
    tape.put_loc(arg.loc()); // = arg1
    tape.put_loc(res.loc()); // = res

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(res.refloc_value());
  }

  if (cond.value() >= 0)
    res.refloc_value(arg.value());
}

bool advector::nondecreasing() const {
  bool ret = true;
  double last = -ADOLC_MATH_NSP::numeric_limits<double>::infinity();
  for (const auto &a : data_)
    ret = ret && (a.value() >= last);
  return ret;
}

adouble advector::operator[](const adouble &index) const {
  ValueTape &tape = currentTape();
  const size_t idx = (size_t)trunc(fabs(index.value()));
  adouble ret_adouble;

  if (tape.traceFlag()) {
    tape.put_op(subscript);
    tape.put_loc(index.loc());
    tape.put_val(size());
    tape.put_loc(data_[0].loc());
    tape.put_loc(ret_adouble.loc());

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  if (idx >= size())
    ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_OOB, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info5 = size(), .info6 = idx});

  ret_adouble.value(data_[idx].value());
  return ret_adouble;
}

adubref advector::operator[](const adouble &index) {
  ValueTape &tape = currentTape();
  const size_t idx = (size_t)trunc(fabs(index.value()));
  size_t locat = tape.next_loc();
  size_t n = data_.size();
  if (tape.traceFlag()) {
    tape.put_op(subscript_ref);
    tape.put_loc(index.loc());
    tape.put_val(n);
    tape.put_loc(data_[0].loc());
    tape.put_loc(locat);

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(tape.get_ad_value(locat));
  }

  if (idx >= n)
    ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_OOB, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info5 = n, .info6 = idx});

  tape.set_ad_value(locat, data_[idx].loc());
  return adubref(locat, data_[idx].loc());
}

adouble advector::lookupindex(const adouble &a, const adouble &b) const {
  if (!nondecreasing())
    ADOLCError::fail(ADOLCError::ErrorType::ADVECTOR_NON_DECREASING,
                     CURRENT_LOCATION);

  if (b.value() < 0)
    ADOLCError::fail(ADOLCError::ErrorType::ADVECTOR_NON_NEGATIVE,
                     CURRENT_LOCATION);
  adouble r = 0.0;
  for (size_t i = 0; i < size(); ++i)
    condassign(r, a - data_[i] * b, adouble(i + 1));
  return r;
}
