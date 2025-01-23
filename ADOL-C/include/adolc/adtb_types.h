/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adtb_types.h
 Revision: $Id$
 Contents: adtb_types.h contains the declarations of tape_location, adouble and
pdouble.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert, Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#ifndef ADOLC_AD_TYPES_H
#define ADOLC_AD_TYPES_H

#include <adolc/taping_p.h>
#include <iostream>

/* The intent of the struct is to define a type-safe location on a tape. This
 * type should be leverage by all type-based types, like adouble or padouble.
 *
 * @param loc_ Location on the tape
 */
struct tape_location {
  size_t loc_;
};

class pdouble;

/**
 * @brief The `adouble` class is leveraged to compute tape-based derivatives. It
 * is represented by a location on the tape and the value that is stored on the
 * tape at the location
 *
 * Its interface acts in principle as `double`. However, internally, whenever it
 * participates in an arithmetic operation, the `adouble` registers locations
 * and the type of the operation on the tape.
 */
class ADOLC_DLL_EXPORT adouble {
public:
  /** @brief Default constructor. Initializes an `adouble` with next location
   * from the tape. If ADOLC_ADOUBLE_STDCZERO is set, 0 is written on the tape
   * at the new location.
   */
  adouble();

  /**
   * @brief Constructor initializing an `adouble` with a new location and puts
   * the given value `coval` on the tape.
   * @param coval Value that is stored on the tape at the new location.
   */
  adouble(double coval);

  /**
   * @brief Constructor initializing an `adouble` with a specific tape location.
   * @param tape_loc The tape location to associate with this `adouble`.
   */
  explicit adouble(const tape_location &tape_loc);

  /**
   * @brief Copy constructor. Creates a new `adouble` with a new location and
   * puts assignment operation onto the tape`
   * @param a The `adouble` to copy.
   */
  adouble(const adouble &a);

  /**
   * @brief Move constructor. Creates new `adouble` with location of `a`. The
   * state of `a` is set to invalid. Therefore, the destructor will not free the
   * location of `a`.
   * @param a The `adouble` to move.
   */
  adouble(adouble &&a) noexcept;

  /** @brief Destructor. Releases the location from the tape. */
  ~adouble();

  // Assignment Operators

  /**
   * @brief Records the assingment of a value to the `adouble` on the tape at
   * the location of the `adouble`.
   * @param coval The value to assign.
   * @return Reference to `this`.
   */
  adouble &operator=(const double coval);

  /**
   * @brief Registers an assignment on the tape with location of `this` and `a`.
   * @param a The `adouble` to assign.
   * @return Reference to `this`.
   */
  adouble &operator=(const adouble &a);

  /**
   * @brief Moves the state from another `adouble`. The location is overwritten
   * by the locaiton of `a`. Afterwards the old location is removed from the
   * tape and `a` is invalid.
   * @param a The `adouble` to move.
   * @return Reference to the updated `adouble`.
   */
  adouble &operator=(adouble &&a) noexcept;

  /**
   * @brief Registers the assignment of the `pdouble` to the `adouble`.
   * @param p The `pdouble` to assign.
   * @return Reference to `this`.
   */
  adouble &operator=(const pdouble &p);

  // Accessors

  /**
   * @brief Retrieves the current value stored at the tape location.
   * @return The value of the `adouble` from the tape.
   */
  inline double value() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
  }

  /**
   * @brief Retrieves the tape location of the `adouble`.
   * @return The location on the tape.
   */
  inline size_t loc() const { return tape_loc_.loc_; }

  // Mutators

  /**
   * @brief Updates the value stored at the tape location.
   * @param coval the new value to assign.
   */
  inline void value(const double coval) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = coval;
  }

  // Type Conversions

  /**
   * @brief Converts the `adouble` to a `double` by returning its corresponding
   * `double` value from the tape.
   * @return The value of the `adouble` at the tape location.
   */
  inline explicit operator double() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
  }

  /**
   * @brief Provides a reference to the internal `double` value from the tape.
   * @return A reference to the `double` value at the tape location.
   */
  inline explicit operator const double &() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
  }

  // Arithmetic assignment Operators

  adouble &operator+=(const double coval);
  adouble &operator+=(const adouble &a);
  adouble &operator+=(const pdouble &p);

  adouble &operator-=(const double coval);
  adouble &operator-=(const adouble &a);
  adouble &operator-=(const pdouble &p);

  adouble &operator*=(const double coval);
  adouble &operator*=(const adouble &a);
  adouble &operator*=(const pdouble &p);

  adouble &operator/=(const double coval);
  adouble &operator/=(const adouble &a);
  adouble &operator/=(const pdouble &p);

  // Postfix and Prefix Increment and Decrement Operators

  adouble operator++(int);
  adouble operator--(int);

  adouble &operator++();
  adouble &operator--();

  // Independent and Dependent Markers

  /**
   * @brief Registers the assignment of the input value to the `adouble` and
   * mark the `adouble` as independent variable.
   * @param in Value that is assigned to the `adouble`
   * @return A reference to the `adouble`.
   */
  adouble &operator<<=(double in);

  /**
   * @brief Stores the value of `adouble` in the provided reference and mark
   * `adouble` as dependent variable.
   * @param out Value that will get the value of `adouble`
   * @return A reference to the `adouble`.
   */
  adouble &operator>>=(double &out);

  /** @brief Declares the `adouble` as an independent variable on the tape */
  void declareIndependent();

  /** @brief Declares the `adouble` as a dependent variable on the tape */
  void declareDependent();

private:
  /** @brief Stores the location of the `adouble` on the tape. */
  tape_location tape_loc_;

  /**
   * @brief Indicates whether the `adouble` is valid.
   *
   * All constructors ensure validity (`valid=1`). The validity changes only
   * during move operations.
   */
  int valid{1};
};

/**
 * @brief The `pdouble` class represents parameter values in tape-based
 * computations.
 */
class ADOLC_DLL_EXPORT pdouble {
public:
  /** @brief Destructor. */
  ~pdouble() = default;

  /** @brief Deleted copy constructor. */
  pdouble(const pdouble &) = delete;

  /** @brief Deleted default constructor. */
  pdouble() = delete;

  /** @brief Deleted move constructor. */
  pdouble(pdouble &&) = delete;

  /** @brief Deleted move assignment operator. */
  pdouble &operator=(pdouble &&) = delete;

  /**
   * @brief Constructor initializing a `pdouble` with a constant value.
   * @param pval The initial value for the `pdouble`.
   */
  pdouble(const double pval);

  /**
   * @brief Constructor initializing a `pdouble` with a specific tape location.
   * @param tape_loc The tape location to associate with this `pdouble`.
   */
  explicit pdouble(tape_location tape_loc);

  /**
   * @brief Creates a parameterized `pdouble` with the given value.
   * @param pval The value to assign.
   * @return A `pdouble` instance representing the value.
   */
  static pdouble mkparam(const double pval);

  /**
   * @brief Converts the `pdouble` to an `adouble`.
   * @return An `adouble` representing the `pdouble`.
   */
  explicit operator adouble() const;

  /**
   * @brief Retrieves the tape location of the `pdouble`.
   * @return The tape location.
   */
  inline size_t loc() const { return tape_loc_.loc_; }

  /**
   * @brief Retrieves the current value stored at the tape location.
   * @return The value of the `pdouble`.
   */
  inline double value() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc_.loc_];
  }

  /**
   * @brief Updates the value stored at the tape location.
   * @param pval The new value to assign.
   */
  inline void value(const double pval) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc_.loc_] = pval;
  }

private:
  /** @brief Stores the location of the `pdouble` on the tape. */
  tape_location tape_loc_;
};

std::ostream &operator<<(std::ostream &, const adouble &);
std::istream &operator>>(std::istream &, const adouble &);

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
inline adouble operator+(adouble &&a, adouble &&b) { return std::move(a) + b; }

ADOLC_DLL_EXPORT adouble operator+(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator+(const double coval, adouble &&a);

adouble operator+(const adouble &a, const double coval);
adouble operator+(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble operator-(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator-(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator-(const adouble &a, adouble &&b);
inline adouble operator-(adouble &&a, adouble &&b) { return std::move(a) - b; }

ADOLC_DLL_EXPORT adouble operator-(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator-(const double coval, adouble &&a);

adouble operator-(const adouble &a, const double coval);
adouble operator-(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble operator*(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator*(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator*(const adouble &a, adouble &&b);
inline adouble operator*(adouble &&a, adouble &&b) { return std::move(a) * b; }

ADOLC_DLL_EXPORT adouble operator*(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator*(const double coval, adouble &&a);

adouble operator*(const adouble &a, const double coval);
adouble operator*(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble operator/(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator/(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator/(const adouble &a, adouble &&b);
inline adouble operator/(adouble &&a, adouble &&b) { return std::move(a) / b; }

ADOLC_DLL_EXPORT adouble operator/(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator/(const double coval, adouble &&a);

adouble operator/(const adouble &a, const double coval);
adouble operator/(adouble &&a, const double coval);

/* Comparison  */

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

inline bool operator!=(const adouble &a, const adouble &b) {
  return (a - b != 0.0);
}
inline bool operator==(const adouble &a, const adouble &b) {
  return (a - b == 0.0);
}
inline bool operator<=(const adouble &a, const adouble &b) {
  return (a - b <= 0.0);
}
inline bool operator>=(const adouble &a, const adouble &b) {
  return (a - b >= 0.0);
}
inline bool operator>(const adouble &a, const adouble &b) {
  return (a - b > 0.0);
}
inline bool operator<(const adouble &a, const adouble &b) {
  return (a - b < 0.0);
}

#endif // ADOLC_ADVANCED_BRANCHING

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
inline adouble fmin(adouble &&a, adouble &&b) { return fmin(std::move(a), b); }

ADOLC_DLL_EXPORT adouble fmin(const adouble &a, const double coval);
ADOLC_DLL_EXPORT adouble fmin(adouble &&a, const double coval);

ADOLC_DLL_EXPORT adouble fmin(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble fmin(const double coval, adouble &&a);

ADOLC_DLL_EXPORT adouble fmax(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmax(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmax(const adouble &a, adouble &&b);
inline adouble fmax(adouble &&a, adouble &&b) { return fmax(std::move(a), b); }

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
ADOLC_DLL_EXPORT double myquad(double x);

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

// copies src to dest; both should be of size "size"
void ADOLC_DLL_EXPORT adolc_vec_copy(adouble *const dest,
                                     const adouble *const src, size_t size);

/* computes canonical scalarproduct, src and dest must have size "size" */
void ADOLC_DLL_EXPORT adolc_vec_dot(adouble *const dest,
                                    const adouble *const src, size_t size);

/* res = a*vec_a + vec_b, vec_a and vec_b must have same size "size" */
void ADOLC_DLL_EXPORT adolc_vec_axpy(adouble *const res, const adouble &a,
                                     const adouble *const vec_a,
                                     const adouble *const vec_b, size_t size);

adouble operator-(const pdouble &p);
adouble operator+(const adouble &a, const pdouble &p);
adouble operator+(adouble &&a, const pdouble &p);
inline adouble operator+(const pdouble &p, const adouble &a) { return a + p; };
inline adouble operator+(const pdouble &p, adouble &&a) {
  return std::move(a) + p;
}
inline adouble operator+(const pdouble &p, const double coval) {
  return coval + adouble(p);
}
inline adouble operator+(const double coval, const pdouble &p) {
  return coval + adouble(p);
}

adouble operator-(const adouble &a, const pdouble &p);
adouble operator-(adouble &&a, const pdouble &p);
inline adouble operator-(const pdouble &p, const adouble &a) {
  return (-a) + p;
}
inline adouble operator-(const pdouble &p, adouble &&a) {
  return (-std::move(a)) + p;
}
inline adouble operator-(const pdouble &p, const double coval) {
  return adouble(p) - coval;
}
inline adouble operator-(const double coval, const pdouble &p) {
  return coval - adouble(p);
}

adouble operator*(const adouble &a, const pdouble &p);
adouble operator*(adouble &&a, const pdouble &p);
inline adouble operator*(const pdouble &p, const adouble &a) { return a * p; };
inline adouble operator*(const pdouble &p, adouble &&a) {
  return std::move(a) * p;
}
inline adouble operator*(const pdouble &p, const double coval) {
  return coval * adouble(p);
}
inline adouble operator*(const double coval, const pdouble &p) {
  return coval * adouble(p);
}

adouble recipr(const pdouble &p);
adouble operator/(const pdouble &p, const adouble &a);
adouble operator/(const pdouble &p, adouble &&a);
inline adouble operator/(const adouble &a, const pdouble &p) {
  return a * recipr(p);
}
inline adouble operator/(const pdouble &p, const double coval) {
  return adouble(p) / coval;
}
inline adouble operator/(const double coval, const pdouble &p) {
  return coval * recipr(p);
}

#ifdef ADOLC_ADVANCED_BRANCHING
adouble operator!=(const adouble &a, const pdouble &p);
adouble operator!=(adouble &&a, const pdouble &p);
inline adouble operator!=(const pdouble &p, const adouble &a) {
  return (a != p);
}
inline adouble operator!=(const pdouble &p, adouble &&a) {
  return (std::move(a) != p);
}

adouble operator==(const adouble &a, const pdouble &p);
adouble operator==(adouble &&a, const pdouble &p);
inline adouble operator==(const pdouble &p, const adouble &a) {
  return (a == p);
}
inline adouble operator==(const pdouble &p, adouble &&a) {
  return (std::move(a) == p);
}

adouble operator<=(const adouble &a, const pdouble &p);
adouble operator<=(adouble &&a, const pdouble &p);
inline adouble operator<=(const pdouble &p, const adouble &a) {
  return (a >= p);
}
inline adouble operator<=(const pdouble &p, adouble &&a) {
  return (std::move(a) >= p);
}

adouble operator>=(const adouble &a, const pdouble &p);
adouble operator>=(adouble &&a, const pdouble &p);
inline adouble operator>=(const pdouble &p, const adouble &a) {
  return (a <= p);
}
inline adouble operator>=(const pdouble &p, adouble &&a) {
  return (std::move(a) <= p);
}

adouble operator>(const adouble &a, const pdouble &p);
adouble operator>(adouble &&a, const pdouble &p);
inline adouble operator>(const pdouble &p, const adouble &a) { return (a < p); }
inline adouble operator>(const pdouble &p, adouble &&a) {
  return (std::move(a) < p);
}

adouble operator<(const adouble &a, const pdouble &p);
adouble operator<(adouble &&a, const pdouble &p);
inline adouble operator<(const pdouble &p, const adouble &a) { return (a > p); }
inline adouble operator<(const pdouble &p, adouble &&a) {
  return (std::move(a) > p);
}

#else  // ADOLC_ADVANCED_BRANCHING

inline bool operator!=(const adouble &a, const pdouble &p) {
  return ((a - p) != 0);
}
inline bool operator!=(const pdouble &p, const adouble &a) { return (a != p); }
inline bool operator!=(const pdouble &p, adouble &&a) {
  return (std::move(a) != p);
}

inline bool operator==(const adouble &a, const pdouble &p) {
  return ((a - p) == 0);
}
inline bool operator==(const pdouble &p, const adouble &a) { return (a == p); }
inline bool operator==(const pdouble &p, adouble &&a) {
  return (std::move(a) == p);
}

inline bool operator<=(const adouble &a, const pdouble &p) {
  return ((a - p) <= 0);
}
inline bool operator>=(const adouble &a, const pdouble &p) {
  return ((a - p) >= 0);
}
inline bool operator<=(const pdouble &p, const adouble &a) { return (a >= p); }
inline bool operator<=(const pdouble &p, adouble &&a) {
  return (std::move(a) >= p);
}
inline bool operator>=(const pdouble &p, const adouble &a) { return (a <= p); }
inline bool operator>=(const pdouble &p, adouble &&a) {
  return (std::move(a) <= p);
}

inline bool operator>(const adouble &a, const pdouble &p) {
  return ((a - p) > 0);
}
inline bool operator<(const adouble &a, const pdouble &p) {
  return ((a - p) < 0);
}

inline bool operator>(const pdouble &p, const adouble &a) { return (a < p); }
inline bool operator>(const pdouble &p, adouble &&a) {
  return (std::move(a) < p);
}

inline bool operator<(const pdouble &p, const adouble &a) { return (a > p); }
inline bool operator<(const pdouble &p, adouble &&a) {
  return (std::move(a) > p);
}
#endif // ADOLC_ADVANCED_BRANCHING

/* unary operators */
inline adouble exp(const pdouble &p) { return exp(adouble(p)); }
inline adouble log(const pdouble &p) { return log(adouble(p)); }
inline adouble log10(const pdouble &p) { return log10(adouble(p)); }
inline adouble sqrt(const pdouble &p) { return sqrt(adouble(p)); }
inline adouble cbrt(const pdouble &p) { return cbrt(adouble(p)); }
inline adouble sin(const pdouble &p) { return sin(adouble(p)); }
inline adouble cos(const pdouble &p) { return cos(adouble(p)); }
inline adouble tan(const pdouble &p) { return tan(adouble(p)); }
inline adouble asin(const pdouble &p) { return asin(adouble(p)); }
inline adouble acos(const pdouble &p) { return acos(adouble(p)); }
inline adouble atan(const pdouble &p) { return atan(adouble(p)); }
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

inline adouble fmin(const adouble &a, const pdouble &p) {
  return fmin(a, adouble(p));
}
inline adouble fmin(const pdouble &p, const adouble &a) { return fmin(p, a); }
inline adouble fmin(const pdouble &p, const pdouble &q) {
  return fmin(p, adouble(q));
}
inline adouble fmin(const double coval, const pdouble &p) {
  return fmin(coval, adouble(p));
}
inline adouble fmin(const pdouble &p, const double coval) {
  return fmin(adouble(p), coval);
}

inline adouble fmax(const adouble &a, const pdouble &p) {
  return fmax(a, adouble(p));
}
inline adouble fmax(const pdouble &p, const adouble &a) { return fmax(a, p); }
inline adouble fmax(const pdouble &p, const pdouble &q) {
  return fmax(p, adouble(q));
}
inline adouble fmax(const double coval, const pdouble &p) {
  return fmax(coval, adouble(p));
}
inline adouble fmax(const pdouble &p, const double coval) {
  return fmax(adouble(p), coval);
}

inline adouble ldexp(const pdouble &p, const int exp) {
  return ldexp(adouble(p), exp);
}
inline adouble frexp(const pdouble &p, int *exp) {
  return frexp(adouble(p), exp);
}

adouble pow(const adouble &a, const pdouble &p);
adouble pow(adouble &&a, const pdouble &p);
adouble pow(const pdouble &p, const adouble &a);
inline adouble pow(const pdouble &p, const double coval) {
  return pow(adouble(p), coval);
}
inline adouble pow(const double coval, const pdouble &p) {
  return pow(coval, adouble(p));
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

#endif // ADOLC_AD_TYPES_H
