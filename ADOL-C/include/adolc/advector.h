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

#include <adolc/ad_types.h>

/****************************************************************************/
/*                                                         THIS FILE IS C++ */
#include <vector>

/****************************************************************************/
/*                                           THIS IS ONLY FOR TAPED VERSION
 */
#if !defined(TAPELESS)

class advector;
class adubref;

class ADOLC_DLL_EXPORT adubref {
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
  adubref &operator=(const pdouble &);

  inline size_t getLocation() const { return location; }
  inline size_t getRefloc() const { return refloc; }
  inline size_t getValue() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[refloc];
  }

  inline void setValue(const double coval) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.store[refloc] = coval;
  }
  operator adouble() const;

  adouble operator++(int);
  adouble operator--(int);

  adubref &operator++();
  adubref &operator--();

  adubref &operator+=(const double coval);
  adubref &operator+=(const adouble &a);
  adubref &operator+=(const pdouble &p);

  adubref &operator-=(const double coval);
  adubref &operator-=(const adouble &a);
  adubref &operator-=(const pdouble &p);

  adubref &operator*=(const double coval);
  adubref &operator*=(const adouble &a);
  adubref &operator*=(const pdouble &p);

  inline adubref &operator/=(const double coval) {
    return *this *= (1.0 / coval);
  }

  inline adubref &operator/=(const adouble &a) { return *this *= (1.0 / a); }

  inline adubref &operator/=(const pdouble &p) { return *this *= recipr(p); }

  adubref &operator<<=(const double coval);
  void declareIndependent();
  adubref &operator>>=(double &coval);
  void declareDependent();

private:
  size_t location;
  size_t refloc;
};

void condassign(adubref &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2);
void condassign(adubref &res, const adouble &cond, const adouble &arg);

void condeqassign(adubref &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2);
void condeqassign(adubref &res, const adouble &cond, const adouble &arg);

class advector {
public:
  ADOLC_DLL_EXPORT advector() = default;
  ADOLC_DLL_EXPORT explicit advector(size_t n)
      : data(ensureContiguousLocations_(n)) {}

  advector(const advector &ad_vec)
      : data(ensureContiguousLocations_(ad_vec.size())) {
    adolc_vec_copy(data.data(), ad_vec.data.data(), ad_vec.size());
  }

  // given adouble vector might not have contiguous locations
  advector(const std::vector<adouble> &v) : data(v) {};
  ADOLC_DLL_EXPORT ~advector() {}

  operator const std::vector<adouble> &() const { return data; }
  ADOLC_DLL_EXPORT operator std::vector<adouble> &() { return data; }
  ADOLC_DLL_EXPORT operator adouble *() { return data.data(); }
  ADOLC_DLL_EXPORT adouble operator[](const adouble &index) const;
  ADOLC_DLL_EXPORT adubref operator[](const adouble &index);
  ADOLC_DLL_EXPORT adouble &operator[](size_t i) { return data[i]; }
  ADOLC_DLL_EXPORT const adouble &operator[](size_t i) const { return data[i]; }
  ADOLC_DLL_EXPORT adouble lookupindex(const adouble &a,
                                       const adouble &b) const;

  ADOLC_DLL_EXPORT size_t size() const { return data.size(); }
  ADOLC_DLL_EXPORT bool nondecreasing() const;

private:
  std::vector<adouble> data;
};

#endif /* TAPELESS */
#endif // ADOLC_ADVECTOR_H
