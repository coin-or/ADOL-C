/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: advector.h contains a vector<adouble> implementation
           that is able to trace subscripting operations.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#ifndef ADOLC_ADVECTOR_H
#define ADOLC_ADVECTOR_H

#include <adolc/adolcexport.h>
#include <adolc/adtb_types.h>
#include <adolc/valuetape/valuetape.h>
#include <memory>
#include <vector>

/****************************************************************************/
/*                                           THIS IS ONLY FOR TAPED VERSION
 */
#if !defined(TAPELESS)

class advector;
class adubref;

class ADOLC_API adubref {
  /* This class is supposed to be used only when an advector subscript
   * occurs as an lvalue somewhere. What we need to do is read the location
   * of the referenced adouble out of store[location] and perform
   * operations with this refloc. This means that the tape needs new
   * opcodes (ref_assign_* /ref_eq_* / ref_{incr,decr}_a) for each of
   * these operations, most of the code  will simply be copied from
   * adouble class, since the operation is really the same except for
   * the part where the refloc is read from store[location].
   * Reverse mode is also straightforward the same way.
   *
   * Convert to a new adub as soon as used as rvalue, this is why adubref
   * is not a child of badouble, since it should never occur as rvalue.
   */

  size_t loc_;
  size_t refloc_;

public:
  explicit adubref(size_t lo, size_t ref);
  adubref(void) = delete;
  adubref(double) = delete;
  adubref(const adubref &) = delete;
  adubref(adubref &&) = delete;
  adubref(const adouble &) = delete;
  adubref(adouble &&) = delete;
  ~adubref();

  adubref &operator=(const double coval);
  adubref &operator=(const adouble &a);
  adubref &operator=(const pdouble &p) { return *this = adouble(p); }

  size_t loc() const { return loc_; }
  size_t refloc() const { return refloc_; }
  double refloc_value() const { return currentTape().get_ad_value(refloc_); }
  double loc_value() const { return currentTape().get_ad_value(loc_); }
  void refloc_value(const double coval) {
    currentTape().set_ad_value(refloc_, coval);
  }
  operator adouble() const;

  adouble operator++(int);
  adouble operator--(int);

  adubref &operator++();
  adubref &operator--();

  adubref &operator+=(const double coval);
  adubref &operator+=(const adouble &a);
  inline adubref &operator+=(const pdouble &p) { return *this += adouble(p); }

  adubref &operator-=(const double coval);
  adubref &operator-=(const adouble &a);
  inline adubref &operator-=(const pdouble &p) { return *this -= adouble(p); }

  adubref &operator*=(const double coval);
  adubref &operator*=(const adouble &a);
  inline adubref &operator*=(const pdouble &p) { return *this *= adouble(p); }

  inline adubref &operator/=(const double coval) {
    return *this *= (1.0 / coval);
  }
  inline adubref &operator/=(const adouble &a) { return *this *= (1.0 / a); }
  inline adubref &operator/=(const pdouble &p) { return *this /= adouble(p); }

  adubref &operator<<=(const double coval);
  void declareIndependent();
  adubref &operator>>=(double &coval);
  void declareDependent();
};

void ADOLC_API condassign(adubref &res, const adouble &cond,
                          const adouble &arg1, const adouble &arg2);
void ADOLC_API condassign(adubref &res, const adouble &cond,
                          const adouble &arg);

void ADOLC_API condeqassign(adubref &res, const adouble &cond,
                            const adouble &arg1, const adouble &arg2);
void ADOLC_API condeqassign(adubref &res, const adouble &cond,
                            const adouble &arg);

class ADOLC_API advector {
  std::vector<adouble> data_;

public:
  advector() = default;
  explicit advector(size_t n)
      : data_(currentTape().ensureContiguousLocations_(n)) {}

  advector(const advector &ad_vec)
      : data_(currentTape().ensureContiguousLocations_(ad_vec.size())) {
    adolc_vec_copy(data_.data(), ad_vec.data_.data(), ad_vec.size());
  }

  // given adouble vector might not have contiguous locations
  advector(const std::vector<adouble> &v) : data_(v) {};

  ~advector() {}

  operator const std::vector<adouble> &() const { return data_; }
  operator std::vector<adouble> &() { return data_; }
  operator adouble *() { return data_.data(); }
  adouble operator[](const adouble &index) const;
  adubref operator[](const adouble &index);
  adouble &operator[](size_t i) { return data_[i]; }
  const adouble &operator[](size_t i) const { return data_[i]; }
  adouble lookupindex(const adouble &a, const adouble &b) const;

  size_t size() const { return data_.size(); }
  bool nondecreasing() const;
};

#endif /* TAPELESS */
#endif // ADOLC_ADVECTOR_H
