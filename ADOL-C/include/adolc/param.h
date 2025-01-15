/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     param.h
 Revision: $Id$
 Contents: class for parameter dependent functions

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_PARAM_H)
#define ADOLC_PARAM_H 1
#include "adouble.h"
#include "internal/common.h"
#if defined(__cplusplus)

#include <cstdio>
#include <stdexcept>

class ADOLC_DLL_EXPORT pdouble {

public:
  ~pdouble() {}
  pdouble(const pdouble &) = delete;
  pdouble(void) = delete;

  double getValue() const;

  static padouble mkparam(double pval);
  inline pdouble getparam(size_t loc_) { return padouble{loc_}; };
  explicit operator pdouble *() const;
  friend locint mkparam_idx(double pval);
  operator adouble() const;

  ADOLC_DLL_EXPORT pdouble getparam(locint index);
  ADOLC_DLL_EXPORT locint mkparam_idx(double pval);

private:
  double val_;
  tape_location tape_loc_;

  pdouble(double pval);
  pdouble(locint index);
};

#ifdef ADOLC_ADVANCED_BRANCHING
adouble operator!=(const adouble &a, const pdouble &p);
adouble operator!=(adouble &&a, const pdouble &p);
adouble operator!=(const pdouble &p, const adouble &b);
adouble operator!=(const pdouble &p, adouble &&b);

adouble operator==(const adouble &a, const pdouble &p);
adouble operator==(adouble &&a, const pdouble &p);
adouble operator==(const pdouble &p, const adouble &b);
adouble operator==(const pdouble &p, adouble &&b);

adouble operator<=(const adouble &a, const pdouble &p);
adouble operator<=(adouble &&a, const pdouble &p);
adouble operator<=(const pdouble &p, const adouble &b);
adouble operator<=(const pdouble &p, adouble &&b);

adouble operator>=(const adouble &a, const pdouble &p);
adouble operator>=(adouble &&a, const pdouble &p);
adouble operator>=(const pdouble &p, const adouble &b);
adouble operator>=(const pdouble &p, adouble &&b);

adouble operator<(const adouble &a, const pdouble &p);
adouble operator<(adouble &&a, const pdouble &p);
adouble operator<(const pdouble &p, const adouble &b);
adouble operator<(const pdouble &p, adouble &&b);

adouble operator>(const adouble &a, const pdouble &p);
adouble operator>(adouble &&a, const pdouble &p);
adouble operator>(const pdouble &p, const adouble &b);
adouble operator>(const pdouble &p, adouble &&b);

#else  // ADOLC_ADVANCED_BRANCHING

inline int operator!=(const pdouble &a, const adouble &b) { return (b != a); }
inline int operator==(const pdouble &a, const adouble &b) { return (b == a); }
inline int operator<=(const pdouble &a, const adouble &b) { return (b >= a); }
inline int operator>=(const pdouble &a, const adouble &b) { return (b <= a); }
inline int operator>(const pdouble &a, const adouble &b) { return (b < a); }
inline int operator<(const pdouble &a, const adouble &b) { return (b > a); }

inline int operator!=(const adouble &a, const pdouble &b) {
  return ((a - b) != 0);
}
inline int operator==(const adouble &a, const pdouble &b) {
  return ((a - b) == 0);
}
inline int operator<=(const adouble &a, const pdouble &b) {
  return ((a - b) <= 0);
}
inline int operator>=(const adouble &a, const pdouble &b) {
  return ((a - b) >= 0);
}
inline int operator>(const adouble &a, const pdouble &b) {
  return ((a - b) > 0);
}
inline int operator<(const adouble &a, const pdouble &b) {
  return ((a - b) < 0);
}
#endif // ADOLC_ADVANCED_BRANCHING

inline friend adouble operator+(const pdouble &, const adouble &);
inline friend adouble operator+(const pdouble &, double);
inline friend adouble operator+(double, const pdouble &);
friend ADOLC_DLL_EXPORT adouble operator+(const adouble &, const pdouble &);
friend ADOLC_DLL_EXPORT adouble operator-(const pdouble &);
friend ADOLC_DLL_EXPORT adouble operator-(const adouble &, const pdouble &);
inline friend adouble operator-(const pdouble &, double);
inline friend adouble operator-(double, const pdouble &);
inline friend adouble operator-(const pdouble &, const adouble &);
friend ADOLC_DLL_EXPORT adouble operator*(const adouble &, const pdouble &);
inline friend adouble operator*(const pdouble &, const adouble &);
inline friend adouble operator*(const pdouble &, double);
inline friend adouble operator*(double, const pdouble &);
friend ADOLC_DLL_EXPORT adouble recipr(const pdouble &);
inline friend adouble operator/(const adouble &, const pdouble &);
inline friend adouble operator/(double, const pdouble &);
inline friend adouble operator/(const pdouble &, double);
friend ADOLC_DLL_EXPORT adouble operator/(const pdouble &, const adouble &);
friend ADOLC_DLL_EXPORT adouble pow(const adouble &, const pdouble &);
inline friend adouble fmax(const pdouble &, const adouble &);
inline friend adouble fmax(const adouble &, const pdouble &);
inline friend adouble fmin(const pdouble &, const adouble &);
inline friend adouble fmin(const adouble &, const pdouble &);
/*--------------------------------------------------------------------------*/
/* unary operators (friends) */
inline friend ADOLC_DLL_EXPORT adouble exp(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble log(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble sqrt(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble cbrt(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble sin(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble cos(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble tan(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble asin(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble acos(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble atan(const pdouble &);

/*--------------------------------------------------------------------------*/
/* special operators (friends) */
/* no internal use of condassign: */
inline friend ADOLC_DLL_EXPORT adouble pow(const pdouble &, double);
inline friend ADOLC_DLL_EXPORT adouble log10(const pdouble &);

/* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
inline friend ADOLC_DLL_EXPORT adouble sinh(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble cosh(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble tanh(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble asinh(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble acosh(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble atanh(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble erf(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble erfc(const pdouble &);

inline friend ADOLC_DLL_EXPORT adouble fabs(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble ceil(const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble floor(const pdouble &);

inline friend ADOLC_DLL_EXPORT adouble fmax(const pdouble &, const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble fmax(double, const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble fmax(const pdouble &, double);
inline friend ADOLC_DLL_EXPORT adouble fmin(const pdouble &, const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble fmin(double, const pdouble &);
inline friend ADOLC_DLL_EXPORT adouble fmin(const pdouble &, double);

inline friend ADOLC_DLL_EXPORT adouble ldexp(const pdouble &, int);
inline friend ADOLC_DLL_EXPORT adouble frexp(const pdouble &, int *);

/*--------------------------------------------------------------------------*/

friend ADOLC_DLL_EXPORT adouble pow(const pdouble &, const adouble &);
#endif
#endif

#if !defined(SWIGPRE)
inline adouble operator+(const pdouble &a, const adouble &b) { return (b + a); }

inline adouble operator+(const pdouble &a, double b) {
  return (b + adouble(a));
}

inline adouble operator+(double a, const pdouble &b) {
  return (a + adouble(b));
}

inline adouble operator-(const pdouble &a, const adouble &b) {
  return ((-b) + a);
}

inline adouble operator-(const pdouble &a, double b) {
  return (adouble(a) - b);
}

inline adouble operator-(double a, const pdouble &b) { return (a + (-b)); }

inline adouble operator*(const pdouble &a, const adouble &b) { return (b * a); }

inline adouble operator*(const pdouble &a, double b) {
  return (b * adouble(a));
}

inline adouble operator*(double a, const pdouble &b) {
  return (a * adouble(b));
}

inline adouble operator/(const adouble &a, const pdouble &b) {
  return (a * recipr(b));
}

inline adouble operator/(double a, const pdouble &b) { return (a * recipr(b)); }

inline adouble operator/(const pdouble &a, double b) {
  return (adouble(a) / b);
}

inline adouble fmax(const adouble &y, const pdouble &d) {
  return (-fmin(-d, -y));
}

inline adouble fmax(const pdouble &a, const adouble &b) { return fmax(b, a); }

inline adouble fmin(const pdouble &a, const adouble &b) { return fmin(b, a); }

inline adouble fmin(const adouble &a, const pdouble &b) {
  return fmin(a, adouble(b));
}

/* unary operators (friends) */
inline adouble exp(const pdouble &p) { return exp(adouble(p)); }
inline adouble log(const pdouble &p) { return log(adouble(p)); }
inline adouble sqrt(const pdouble &p) { return sqrt(adouble(p)); }
inline adouble cbrt(const pdouble &p) { return cbrt(adouble(p)); }
inline adouble sin(const pdouble &p) { return sin(adouble(p)); }
inline adouble cos(const pdouble &p) { return cos(adouble(p)); }
inline adouble tan(const pdouble &p) { return tan(adouble(p)); }
inline adouble asin(const pdouble &p) { return asin(adouble(p)); }
inline adouble acos(const pdouble &p) { return acos(adouble(p)); }
inline adouble atan(const pdouble &p) { return atan(adouble(p)); }

/*--------------------------------------------------------------------------*/
/* special operators (friends) */
/* no internal use of condassign: */
inline adouble pow(const pdouble &p, double q) { return pow(adouble(p), q); }
inline adouble log10(const pdouble &p) { return log10(adouble(p)); }

/* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
inline adouble sinh(const pdouble &p) { return sinh(adouble(p)); }
inline adouble cosh(const pdouble &p) { return cosh(adouble(p)); }
inline adouble tanh(const pdouble &p) { return tanh(adouble(p)); }
inline adouble asinh(const pdouble &p) { return asinh(adouble(p)); }
inline adouble acosh(const pdouble &p) { return acosh(adouble(p)); }
inline adouble atanh(const pdouble &p) { return atanh(adouble(p)); }
inline adouble erf(const pdouble &p) { return erf(adouble(p)); }
inline adouble erfc(const pdouble &p) { return erfc(adouble(p)); }
inline adouble fabs(const pdouble &p) { return fabs(adouble(p)); }
inline adouble ceil(const pdouble &p) { return ceil(adouble(p)); }
inline adouble floor(const pdouble &p) { return floor(adouble(p)); }

inline adouble fmax(const pdouble &p, const pdouble &q) {
  return fmax(adouble(p), adouble(q));
}
inline adouble fmax(double p, const pdouble &q) { return fmax(p, adouble(q)); }
inline adouble fmax(const pdouble &p, double q) { return fmax(adouble(p), q); }
inline adouble fmin(const pdouble &p, const pdouble &q) {
  return fmin(adouble(p), adouble(q));
}
inline adouble fmin(double p, const pdouble &q) { return fmin(p, adouble(q)); }
inline adouble fmin(const pdouble &p, double q) { return fmin(adouble(p), q); }

inline adouble ldexp(const pdouble &p, int n) { return ldexp(adouble(p), n); }
inline adouble frexp(const pdouble &p, int *n) { return frexp(adouble(p), n); }
#endif

/*--------------------------------------------------------------------------*/
#endif

BEGIN_C_DECLS

/****************************************************************************/
/* Returns the number of parameters recorded on tape                        */
/****************************************************************************/
ADOLC_DLL_EXPORT size_t get_num_param(short tag);

/****************************************************************************/
/* Overrides the parameters for the next evaluations. This will invalidate  */
/* the taylor stack, so next reverse call will fail, if not preceded by a   */
/* forward call after setting the parameters.                               */
/****************************************************************************/
ADOLC_DLL_EXPORT void set_param_vec(short tag, size_t numparam,
                                    revreal *paramvec);

END_C_DECLS
#endif
