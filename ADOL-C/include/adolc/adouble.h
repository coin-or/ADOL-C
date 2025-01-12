/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: adouble.h contains the basis for the class of adouble
           included here are all the possible functions defined on
           the adouble class.  Notice that, as opposed to earlier versions,
           both the class adub and the class adouble are derived from a base
           class (badouble).  See below for further explanation.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#if !defined(ADOLC_ADOUBLE_H)
#define ADOLC_ADOUBLE_H 1

/****************************************************************************/
/*                                                         THIS FILE IS C++ */
#ifdef __cplusplus

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#if !defined(SWIGPRE)
using std::cerr;
using std::cin;
using std::cout;
using std::istream;
using std::logic_error;
using std::ostream;
#endif

#include <adolc/internal/common.h>

/* NOTICE: There are automatic includes at the end of this file! */

/* The intent of the struct is to define a type-safe location on a tape. This
 * type should be leverage by all type-based types, like adouble or padouble.
 *
 * @param loc_ Location on the tape
 */
struct tape_location {
  size_t loc_;
};
/****************************************************************************/
/*                                             FORWARD DECLARATIONS (TAPES) */

/*--------------------------------------------------------------------------*/
class adouble;
class adub;
class badouble;
class pdouble;

/*--------------------------------------------------------------------------*/
void ADOLC_DLL_EXPORT condassign(double &res, const double &cond,
                                 const double &arg1, const double &arg2);
void ADOLC_DLL_EXPORT condassign(double &res, const double &cond,
                                 const double &arg);

void ADOLC_DLL_EXPORT condeqassign(double &res, const double &cond,
                                   const double &arg1, const double &arg2);
void ADOLC_DLL_EXPORT condeqassign(double &res, const double &cond,
                                   const double &arg);

/*--------------------------------------------------------------------------*/
/* Conditionals */
friend ADOLC_DLL_EXPORT void condassign(adouble &res, const badouble &cond,
                                        const badouble &arg1,
                                        const badouble &arg2);
friend ADOLC_DLL_EXPORT void condassign(adouble &res, const badouble &cond,
                                        const badouble &arg);
friend ADOLC_DLL_EXPORT void condeqassign(adouble &res, const badouble &cond,
                                          const badouble &arg1,
                                          const badouble &arg2);
friend ADOLC_DLL_EXPORT void condeqassign(adouble &res, const badouble &cond,
                                          const badouble &arg);

/****************************************************************************/
/*                                                               CLASS ADUB
 */

/*
   The class Adub
   ---- Basically used as a temporary result.  The address for an
        adub is usually generated within an operation.  That address
        is "freed" when the adub goes out of scope (at destruction time).
   ---- operates just like a badouble, but it has a destructor defined for
   it.
*/
#if !defined(SWIGPRE)
/* s = adolc_vec_dot(x,y,size); <=> s = <x,y>_2 */
ADOLC_DLL_EXPORT adub adolc_vec_dot(const adouble *const, const adouble *const,
                                    locint);
#endif

class ADOLC_DLL_EXPORT adub : public badouble {
  friend ADOLC_DLL_EXPORT class adouble;
  friend ADOLC_DLL_EXPORT class advector;
  friend ADOLC_DLL_EXPORT class adubref;
  friend ADOLC_DLL_EXPORT class pdouble;

protected:
  /* this is the only logically legal constructor, which can be called by
   * friend classes and functions
   */
  adub(locint lo) : badouble(lo) {}

public:
/*--------------------------------------------------------------------------*/
#if !defined(SWIGPRE)
  /* s = adolc_vec_dot(x,y,size); <=> s = <x,y>_2 */
  friend adub adolc_vec_dot(const adouble *const, const adouble *const, locint);
#endif
/* Functions friends with both badouble and adub */
#define _IN_CLASS_ 1
#define _IN_ADUB_ 1
#include <adolc/internal/adubfunc.h>
#undef _IN_ADUB_
#undef _IN_CLASS_

/*--------------------------------------------------------------------------*/
/* Parameter dependent functions (friends) */
#define _IN_CLASS_ 1
#define _IN_ADUB_ 1
#include <adolc/internal/paramfunc.h>
#undef _IN_ADUB_
#undef _IN_CLASS_
};

BEGIN_C_DECLS
ADOLC_DLL_EXPORT void ensureContiguousLocations(size_t n);
END_C_DECLS

/****************************************************************************/
/*                                                            CLASS ADOUBLE */
/*
  The class adouble.
  ---Derived from badouble.  Contains the standard constructors/destructors.
  ---At construction, it is given a new address, and at destruction, that
     address is freed.
*/
class ADOLC_DLL_EXPORT adouble {
  friend ADOLC_DLL_EXPORT class advector;
  friend ADOLC_DLL_EXPORT class pdouble;

protected:
  void initInternal(void); // Init for late initialization
public:
  adouble();
  adouble(double coval);
  explicit adouble(const tape_location &tape_loc);
  adouble(const adouble &a);
  adouble(adouble &&a) noexcept;
  ~adouble();

  adouble &operator=(double coval);
  adouble &operator=(const adouble &a);
  adouble &operator=(adouble &&a);

  adouble &operator<<=(double in);
  adouble &operator>>=(double &out);
  void declareIndependent();
  void declareDependent();

  double getValue() const;
  double value() const;
  void setValue(const double in);

  size_t getLoc() const;

  explicit operator double() const;
  explicit operator const double &() const;

  adouble &operator+=(const double coval);
  adouble &operator+=(const adouble &a);

  adouble &operator-=(const double coval);
  adouble &operator-=(const adouble &a);

  adouble &operator*=(const double coval);
  adouble &operator*=(const adouble &a);

  adouble &operator/=(const double coval);
  adouble &operator/=(const adouble &a);

  /*--------------------------------------------------------------------------*/
  badouble &operator=(const pdouble &p);
  badouble &operator+=(const pdouble &p);
  badouble &operator-=(const pdouble &p);
  badouble &operator*=(const pdouble &p);
  badouble &operator/=(const pdouble &p);

  adouble operator++(int);
  adouble operator--(int);

  adouble &operator++();
  adouble &operator--();

/* Comparison (friends) */
#ifdef ADOLC_ADVANCED_BRANCHING

  friend ADOLC_DLL_EXPORT adouble operator!=(const adouble &a,
                                             const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator!=(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator!=(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator==(const adouble &a,
                                             const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator==(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator==(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator<=(const adouble &a,
                                             const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator<=(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator<=(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator>=(const adouble &a,
                                             const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator>=(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator>=(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator<(const adouble &a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator<(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator<(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator>(const adouble &a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator>(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator>(const adouble &a, adouble &&b);

#else // ADOLC_ADVANCED_BRANCHING

  friend bool operator!=(const adouble &a, const adouble &b);
  friend bool operator==(const adouble &a, const adouble &b);
  friend bool operator<=(const adouble &a, const adouble &b);
  friend bool operator>=(const adouble &a, const adouble &b);
  friend bool operator>(const adouble &a, const adouble &b);
  friend bool operator<(const adouble &a, const adouble &b);

#endif // ADOLC_ADVANCED_BRANCHING

  friend ADOLC_DLL_EXPORT bool operator!=(const adouble &a, const double coval);
  friend bool operator!=(const double coval, const adouble &a);

  friend bool operator==(const double coval, const adouble &a);
  friend ADOLC_DLL_EXPORT bool operator==(const adouble &a, const double coval);

  friend bool operator<=(const double coval, const adouble &a);
  friend ADOLC_DLL_EXPORT bool operator<=(const adouble &a, const double coval);

  friend bool operator>=(const double coval, const adouble &a);
  friend ADOLC_DLL_EXPORT bool operator>=(const adouble &a, const double coval);

  friend bool operator<(const double coval, const adouble &a);
  friend ADOLC_DLL_EXPORT bool operator<(const adouble &a, const double coval);

  friend bool operator>(const double coval, const adouble &a);
  friend ADOLC_DLL_EXPORT bool operator>(const adouble &a, const double coval);

  /*--------------------------------------------------------------------------*/
  /* sign operators (friends) */
  friend ADOLC_DLL_EXPORT adouble operator+(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble operator+(adouble &&a);

  friend ADOLC_DLL_EXPORT adouble operator-(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble operator-(adouble &&a);

  /*--------------------------------------------------------------------------*/
  /* binary operators */
  friend ADOLC_DLL_EXPORT adouble operator+(const adouble &a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator+(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator+(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator+(const double coval,
                                            const adouble &a);
  friend ADOLC_DLL_EXPORT adouble operator+(const double coval, adouble &&a);

  friend adouble operator+(const adouble &a, const double coval);
  friend adouble operator+(adouble &&a, const double coval);

  friend ADOLC_DLL_EXPORT adouble operator-(const adouble &a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator-(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator-(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator-(const double coval,
                                            const adouble &a);
  friend ADOLC_DLL_EXPORT adouble operator-(const double coval, adouble &&a);

  friend adouble operator-(const adouble &a, const double coval);
  friend adouble operator-(adouble &&a, const double coval);

  friend ADOLC_DLL_EXPORT adouble operator*(const adouble &a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator*(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator*(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator*(const double coval,
                                            const adouble &a);
  friend ADOLC_DLL_EXPORT adouble operator*(const double coval, adouble &&a);

  friend adouble operator*(const adouble &a, const double coval);
  friend adouble operator*(adouble &&a, const double coval);

  friend ADOLC_DLL_EXPORT adouble operator/(const adouble &a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator/(adouble &&a, const adouble &b);
  friend ADOLC_DLL_EXPORT adouble operator/(const adouble &a, adouble &&b);

  friend ADOLC_DLL_EXPORT adouble operator/(const double coval,
                                            const adouble &a);
  friend ADOLC_DLL_EXPORT adouble operator/(const double coval, adouble &&a);

  friend adouble operator/(const adouble &a, const double coval);
  friend adouble operator/(adouble &&a, const double coval);

  /*--------------------------------------------------------------------------*/
  /* unary operators */
  friend ADOLC_DLL_EXPORT adouble exp(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble exp(adouble &&a);

  friend ADOLC_DLL_EXPORT adouble log(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble log(adouble &&a);

  friend ADOLC_DLL_EXPORT adouble sqrt(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble sqrt(adouble &&a);

  friend ADOLC_DLL_EXPORT adouble cbrt(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble cbrt(adouble &&a);

  friend ADOLC_DLL_EXPORT adouble sin(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble sin(adouble &&a);

  friend ADOLC_DLL_EXPORT adouble cos(const adouble &a);
  friend ADOLC_DLL_EXPORT adouble cos(adouble &&a);

  friend ADOLC_DLL_EXPORT adub tan(const badouble &);
  friend ADOLC_DLL_EXPORT adub asin(const badouble &);
  friend ADOLC_DLL_EXPORT adub acos(const badouble &);
  friend ADOLC_DLL_EXPORT adub atan(const badouble &);

  /*--------------------------------------------------------------------------*/
  /* special operators (friends) */
  /* no internal use of condassign: */
  friend ADOLC_DLL_EXPORT adub pow(const badouble &, double);
  friend ADOLC_DLL_EXPORT adub log10(const badouble &);

  /* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
  friend ADOLC_DLL_EXPORT adub sinh(const badouble &);
  friend ADOLC_DLL_EXPORT adub cosh(const badouble &);
  friend ADOLC_DLL_EXPORT adub tanh(const badouble &);
  friend ADOLC_DLL_EXPORT adub asinh(const badouble &);
  friend ADOLC_DLL_EXPORT adub acosh(const badouble &);
  friend ADOLC_DLL_EXPORT adub atanh(const badouble &);
  friend ADOLC_DLL_EXPORT adub erf(const badouble &);
  friend ADOLC_DLL_EXPORT adub erfc(const badouble &);

  friend ADOLC_DLL_EXPORT adub fabs(const badouble &);
  friend ADOLC_DLL_EXPORT adub ceil(const badouble &);
  friend ADOLC_DLL_EXPORT adub floor(const badouble &);

  friend ADOLC_DLL_EXPORT adub fmax(const badouble &, const badouble &);
  friend ADOLC_DLL_EXPORT adub fmax(double, const badouble &);
  friend ADOLC_DLL_EXPORT adub fmax(const badouble &, double);
  friend ADOLC_DLL_EXPORT adub fmin(const badouble &, const badouble &);
  friend ADOLC_DLL_EXPORT adub fmin(double, const badouble &);
  friend ADOLC_DLL_EXPORT adub fmin(const badouble &, double);

  friend ADOLC_DLL_EXPORT adub ldexp(const badouble &, int);
  friend ADOLC_DLL_EXPORT adub frexp(const badouble &, int *);

  /*--------------------------------------------------------------------------*/
  /* special operators (friends) */
  friend ADOLC_DLL_EXPORT adouble atan2(const badouble &, const badouble &);
  /* uses condassign internally */
  friend ADOLC_DLL_EXPORT adouble pow(const badouble &, const badouble &);
  friend ADOLC_DLL_EXPORT adouble pow(double, const badouble &);
  /* User defined version of logarithm to test extend_quad macro */
  friend ADOLC_DLL_EXPORT adouble myquad(const badouble &);

#if defined(ADOLC_DEFAULT_CONTIG_LOC)
  void *operator new[](size_t sz) {
    void *p = ::new char[sz];
    size_t n = (sz - sizeof(size_t)) / sizeof(adouble);
    ensureContiguousLocations(n);
    return p;
  }
  void operator delete[](void *p) {
    ::delete[] (char *)p;
    adouble
#endif
        const p /
        a /
        stores the location of the adouble on adouble tape
            tape_location const tape /
        a / all constructors ensure a valid adouble,
        this state changes only when
        // moving an adouble
        int valid{1};
  };

  ADOLC_DLL_EXPORT std::ostream &operator<<(std::ostream &, const adouble &);
  ADOLC_DLL_EXPORT std::istream &operator>>(std::istream &, const adouble &);

#endif /* __cplusplus */

#include <adolc/advector.h>
#include <adolc/param.h>

#ifdef __cplusplus
  /****************************************************************************/
  /*                                                       INLINE DEFINITIONS */

  /*--------------------------------------------------------------------------*/
  inline locint badouble::loc(void) const { return location; }

  inline locint adouble::loc(void) const {
    const_cast<adouble *>(this)->initInternal();
    return location;
  }

  /*--------------------------------------------------------------------------*/
  /* Comparison */

#if !defined(SWIGPRE)

  /* Subtract a floating point from an adouble  */
  inline adub operator-(const badouble &x, double coval) {
    return (-coval) + x;
  }

  /*--------------------------------------------------------------------------*/
  /* Multiply an adouble by a floating point */
  inline adub operator*(const badouble &x, double coval) { return coval * x; }

  /*--------------------------------------------------------------------------*/
  /* Divide an adouble by a floating point */
  inline adub operator/(const badouble &x, double coval) {
    return (1.0 / coval) * x;
  }
#endif

  inline badouble &badouble::operator/=(const pdouble &p) {
    *this *= recipr(p);
    return *this;
  }

  /* numeric_limits<adouble> specialization
   *
   * All methods return double instead of adouble, because these values
   * never depend on the independent variables.
   */
  template <> struct std::numeric_limits<adouble> {
    static constexpr bool is_specialized = true;

    static constexpr double min() noexcept {
      return std::numeric_limits<double>::min();
    }

    static constexpr double max() noexcept {
      return std::numeric_limits<double>::max();
    }

    static constexpr double lowest() noexcept {
      return std::numeric_limits<double>::lowest();
    }

    static constexpr int digits = std::numeric_limits<double>::digits;
    static constexpr int digits10 = std::numeric_limits<double>::digits10;
    static constexpr int max_digits10 =
        std::numeric_limits<double>::max_digits10;
    static constexpr bool is_signed = std::numeric_limits<double>::is_signed;
    static constexpr bool is_integer = std::numeric_limits<double>::is_integer;
    static constexpr bool is_exact = std::numeric_limits<double>::is_exact;
    static constexpr int radix = std::numeric_limits<double>::radix;

    static constexpr double epsilon() noexcept {
      return std::numeric_limits<double>::epsilon();
    }

    static constexpr double round_error() noexcept {
      return std::numeric_limits<double>::round_error();
    }

    static constexpr int min_exponent =
        std::numeric_limits<double>::min_exponent;
    static constexpr int min_exponent10 =
        std::numeric_limits<double>::min_exponent10;
    static constexpr int max_exponent =
        std::numeric_limits<double>::max_exponent;
    static constexpr int max_exponent10 =
        std::numeric_limits<double>::max_exponent10;

    static constexpr bool has_infinity =
        std::numeric_limits<double>::has_infinity;
    static constexpr bool has_quiet_NaN =
        std::numeric_limits<double>::has_quiet_NaN;
    static constexpr bool has_signaling_NaN =
        std::numeric_limits<double>::has_signaling_NaN;
    static constexpr float_denorm_style has_denorm =
        std::numeric_limits<double>::has_denorm;
    static constexpr bool has_denorm_loss =
        std::numeric_limits<double>::has_denorm_loss;

    static constexpr double infinity() noexcept {
      return std::numeric_limits<double>::infinity();
    }

    static constexpr double quiet_NaN() noexcept {
      return std::numeric_limits<double>::quiet_NaN();
    }

    static constexpr double signaling_NaN() noexcept {
      return std::numeric_limits<double>::signaling_NaN();
    }

    static constexpr double denorm_min() noexcept {
      return std::numeric_limits<double>::denorm_min();
    }

    static constexpr bool is_iec559 = std::numeric_limits<double>::is_iec559;
    static constexpr bool is_bounded = std::numeric_limits<double>::is_bounded;
    static constexpr bool is_modulo = std::numeric_limits<double>::is_modulo;

    static constexpr bool traps = std::numeric_limits<double>::traps;
    static constexpr bool tinyness_before =
        std::numeric_limits<double>::tinyness_before;
    static constexpr float_round_style round_style =
        std::numeric_limits<double>::round_style;
  };

/****************************************************************************/
/*                                                                THAT'S ALL*/
#endif /* __cplusplus */
#endif /* ADOLC_ADOUBLE_H */
