/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: adouble.h contains the basis for the class of adouble
           included here are all the possible functions defined on
           the adouble class.  See below for further explanation.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#ifndef ADOLC_ADOUBLE_H
#define ADOLC_ADOUBLE_H

#include <adolc/internal/common.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

/* The intent of the struct is to define a type-safe location on a tape. This
 * type should be leverage by all type-based types, like adouble or padouble.
 *
 * @param loc_ Location on the tape
 */
struct tape_location {
  size_t loc_;
};

/****************************************************************************/
/*                                                            CLASS ADOUBLE */
/*
  The class adouble.
  ---Derived from badouble.  Contains the standard constructors/destructors.
  ---At construction, it is given a new address, and at destruction, that
     address is freed.
*/
class ADOLC_DLL_EXPORT adouble {
public:
  adouble();
  adouble(double coval);
  explicit adouble(const tape_location &tape_loc);
  adouble(const adouble &a);
  adouble(adouble &&a) noexcept;
  ~adouble();

  adouble &operator=(const double coval);
  adouble &operator=(const adouble &a);
  adouble &operator=(adouble &&a);

  double getValue() const;
  inline double value() const { return getValue(); };
  inline size_t getLoc() const { return tape_loc_.loc_; };

  void setValue(const double coval);

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

  badouble &operator=(const pdouble &p);
  badouble &operator+=(const pdouble &p);
  badouble &operator-=(const pdouble &p);
  badouble &operator*=(const pdouble &p);
  badouble &operator/=(const pdouble &p);

  inline adouble &adouble::operator/=(const padouble &p) {
    *this *= recipr(p);
    return *this;
  }

  adouble operator++(int);
  adouble operator--(int);

  adouble &operator++();
  adouble &operator--();

  adouble &operator<<=(double in);
  adouble &operator>>=(double &out);
  void declareIndependent();
  void declareDependent();

private:
  // stores the location of the adouble on tape
  tape_location tape_loc_;
  // all constructors ensure a valid (valid=1) adouble, this state changes only
  // when moving an adouble
  int valid{1};
};

std::ostream &operator<<(std::ostream &, const adouble &);
std::istream &operator>>(std::istream &, const adouble &);

/* Comparison  */
#ifdef ADOLC_ADVANCED_BRANCHING

ADOLC_DLL_EXPORT adouble operator!=(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator!=(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator!=(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator==(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator==(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator==(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator<=(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator<=(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator<=(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator>=(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator>=(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator>=(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator<(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator<(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator<(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator>(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator>(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator>(const adouble &a, adouble &&b);

#else // ADOLC_ADVANCED_BRANCHING

bool operator!=(const adouble &a, const adouble &b);
bool operator==(const adouble &a, const adouble &b);
bool operator<=(const adouble &a, const adouble &b);
bool operator>=(const adouble &a, const adouble &b);
bool operator>(const adouble &a, const adouble &b);
bool operator<(const adouble &a, const adouble &b);

#endif // ADOLC_ADVANCED_BRANCHING

bool operator!=(const adouble &a, const double coval);
bool operator!=(const double coval, const adouble &a);

bool operator==(const double coval, const adouble &a);
bool operator==(const adouble &a, const double coval);

bool operator<=(const double coval, const adouble &a);
bool operator<=(const adouble &a, const double coval);

bool operator>=(const double coval, const adouble &a);
bool operator>=(const adouble &a, const double coval);

bool operator<(const double coval, const adouble &a);
bool operator<(const adouble &a, const double coval);

bool operator>(const double coval, const adouble &a);
bool operator>(const adouble &a, const double coval);

/*--------------------------------------------------------------------------*/
/* sign operators */
ADOLC_DLL_EXPORT adouble operator+(const adouble &a);
ADOLC_DLL_EXPORT adouble operator+(adouble &&a);

ADOLC_DLL_EXPORT adouble operator-(const adouble &a);
ADOLC_DLL_EXPORT adouble operator-(adouble &&a);

/*--------------------------------------------------------------------------*/
/* binary operators */
ADOLC_DLL_EXPORT adouble operator+(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator+(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator+(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator+(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator+(const double coval, adouble &&a);

adouble operator+(const adouble &a, const double coval);
adouble operator+(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble operator-(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator-(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator-(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator-(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator-(const double coval, adouble &&a);

adouble operator-(const adouble &a, const double coval);
adouble operator-(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble operator*(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator*(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator*(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator*(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator*(const double coval, adouble &&a);

adouble operator*(const adouble &a, const double coval);
adouble operator*(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble operator/(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator/(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator/(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble operator/(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator/(const double coval, adouble &&a);

adouble operator/(const adouble &a, const double coval);
adouble operator/(adouble &&a, const double coval);

/*--------------------------------------------------------------------------*/
/* unary operators */
ADOLC_DLL_EXPORT adouble exp(const adouble &a);
ADOLC_DLL_EXPORT adouble exp(adouble &&a);

ADOLC_DLL_EXPORT adouble log(const adouble &a);
ADOLC_DLL_EXPORT adouble log(adouble &&a);

ADOLC_DLL_EXPORT adouble log10(const adouble &a);
ADOLC_DLL_EXPORT adouble log10(adouble &&a);

ADOLC_DLL_EXPORT adouble sqrt(const adouble &a);
ADOLC_DLL_EXPORT adouble sqrt(adouble &&a);

ADOLC_DLL_EXPORT adouble cbrt(const adouble &a);
ADOLC_DLL_EXPORT adouble cbrt(adouble &&a);

ADOLC_DLL_EXPORT adouble sin(const adouble &a);
ADOLC_DLL_EXPORT adouble sin(adouble &&a);

ADOLC_DLL_EXPORT adouble cos(const adouble &a);
ADOLC_DLL_EXPORT adouble cos(adouble &&a);

ADOLC_DLL_EXPORT adouble tan(const adouble &a);
ADOLC_DLL_EXPORT adouble tan(adouble &&a);

ADOLC_DLL_EXPORT adouble asin(const adouble &a);
ADOLC_DLL_EXPORT adouble asin(adouble &&a);

ADOLC_DLL_EXPORT adouble acos(const adouble &a);
ADOLC_DLL_EXPORT adouble acos(adouble &&a);

ADOLC_DLL_EXPORT adouble atan(const adouble &a);
ADOLC_DLL_EXPORT adouble atan(adouble &&a);

ADOLC_DLL_EXPORT adouble sinh(const adouble &a);
ADOLC_DLL_EXPORT adouble sinh(adouble &&a);

ADOLC_DLL_EXPORT adouble cosh(const adouble &a);
ADOLC_DLL_EXPORT adouble cosh(adouble &&a);

ADOLC_DLL_EXPORT adouble tanh(const adouble &a);
ADOLC_DLL_EXPORT adouble tanh(adouble &&a);

ADOLC_DLL_EXPORT adouble asinh(const adouble &a);
ADOLC_DLL_EXPORT adouble asinh(adouble &&a);

ADOLC_DLL_EXPORT adouble acosh(const adouble &a);
ADOLC_DLL_EXPORT adouble acosh(adouble &&a);

ADOLC_DLL_EXPORT adouble atanh(const adouble &a);
ADOLC_DLL_EXPORT adouble atanh(adouble &&a);

ADOLC_DLL_EXPORT adouble erf(const adouble &a);
ADOLC_DLL_EXPORT adouble erf(adouble &&a);

ADOLC_DLL_EXPORT adouble erfc(const adouble &a);
ADOLC_DLL_EXPORT adouble erfc(adouble &&a);

ADOLC_DLL_EXPORT adouble ceil(const adouble &a);
ADOLC_DLL_EXPORT adouble ceil(adouble &&a);

ADOLC_DLL_EXPORT adouble floor(const adouble &a);
ADOLC_DLL_EXPORT adouble floor(adouble &&a);

ADOLC_DLL_EXPORT adouble fabs(const adouble &a);
ADOLC_DLL_EXPORT adouble fabs(adouble &&a);

ADOLC_DLL_EXPORT adouble fmin(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmin(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmin(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble fmin(const adouble &a, const double coval);
ADOLC_DLL_EXPORT adouble fmin(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble fmin(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble fmin(const double coval, adouble &&a);

ADOLC_DLL_EXPORT adouble fmax(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmax(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmax(const adouble &a, adouble &&b);

ADOLC_DLL_EXPORT adouble fmax(const adouble &a, const double coval);
ADOLC_DLL_EXPORT adouble fmax(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble fmax(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble fmax(const double coval, adouble &&a);

ADOLC_DLL_EXPORT adouble ldexp(const adouble &a, const int exp);
ADOLC_DLL_EXPORT adouble ldexp(adouble &&a, const int exp);

ADOLC_DLL_EXPORT adouble frexp(const adouble &a, int *exp);
ADOLC_DLL_EXPORT adouble frexp(adouble &&a, int *exp);

ADOLC_DLL_EXPORT adouble atan2(const adouble &a, const adouble &b);

ADOLC_DLL_EXPORT adouble pow(const adouble &a, const adouble &b);

ADOLC_DLL_EXPORT adouble pow(const adouble &a, const double exp);
ADOLC_DLL_EXPORT adouble pow(adouble &&a, const double exp);

ADOLC_DLL_EXPORT adouble pow(const double base, const adouble &a);

//-----------------------------------------------------------
/* User defined version of logarithm to test extend_quad macro */
ADOLC_DLL_EXPORT adouble myquad(double x);

/*---------------------------------------------------------------*/
/*                                Conditional assignments */
void condassign(adouble &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2);
void condassign(adouble &res, const adouble &cond, const adouble &arg);
void condeqassign(adouble &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2);
void condeqassign(adouble &res, const adouble &cond, const adouble &arg);

void ADOLC_DLL_EXPORT condassign(double &res, const double &cond,
                                 const double &arg1, const double &arg2);
void ADOLC_DLL_EXPORT condassign(double &res, const double &cond,
                                 const double &arg);
void ADOLC_DLL_EXPORT condeqassign(double &res, const double &cond,
                                   const double &arg1, const double &arg2);
void ADOLC_DLL_EXPORT condeqassign(double &res, const double &cond,
                                   const double &arg);

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
  static constexpr int max_digits10 = std::numeric_limits<double>::max_digits10;
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

  static constexpr int min_exponent = std::numeric_limits<double>::min_exponent;
  static constexpr int min_exponent10 =
      std::numeric_limits<double>::min_exponent10;
  static constexpr int max_exponent = std::numeric_limits<double>::max_exponent;
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

#endif /* ADOLC_ADOUBLE_H */
