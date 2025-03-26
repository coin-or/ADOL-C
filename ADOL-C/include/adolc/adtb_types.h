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

#include <adolc/adolcerror.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

// Forward declarations for concept
class adouble;
class pdouble;

/**
 * @brief Concept to check if a type is either `adouble` or `pdouble`.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept adouble_or_pdouble =
    std::is_same_v<T, adouble> || std::is_same_v<T, pdouble>;

/**
 * @class tape_location
 * @brief Represents a location on a tape.
 *
 * This type should be leveraged by all type-based types, like `adouble` or
 * `pdouble`. `tape_location` stores a location `loc_` on a tape `tape_` and
 * handles the creation and deletion of the location. Thus, it is not
 * recommended to use calls of `free_loc` or `next_loc` outside of
 * `tape_location`. Getters of the `tape_` and `loc_` are provided.
 *
 * @tparam T Either `adouble` or `pdouble`.
 * @param loc_ Location on the tape
 * @param valid_ Specifies whether `tape_location` was moved. Used to decide if
 * `loc_` can be free'd or not.
 */

template <adouble_or_pdouble T> class tape_location {
  size_t loc_{0}; ///< Location on the tape.

  /**
   * @brief Indicates whether the `tape_location` is valid.
   *
   * All instances are constructed in a valid state (`valid_ = 1`). The validity
   * changes only during move operations.
   */
  int valid_{1};

  /**
   * @brief Determines the next available location on the tape.
   *
   * @param tape The tape to retrieve the next location from.
   * @return The next location index.
   */
  size_t next_loc() {
    if constexpr (std::is_same_v<T, adouble>)
      return currentTape().next_loc();
    else
      return currentTape().p_next_loc();
  }

  /**
   * @brief Frees the allocated location on the tape.
   *
   * Ensures that the location is properly deallocated. At the moment only
   * `adouble` support `free_loc`.
   */
  void free_loc() {
    assert(currentTapePtr() != nullptr &&
           "Tape was deleted before all adoubles are destroyed!");
    if constexpr (std::is_same_v<T, adouble>)
      currentTape().free_loc(loc());

    // currently freeing the location of pdouble leads to errors
    // need to revisit the issue
    // currentTape().p_free_loc(loc());
  }

public:
  /**
   * @brief Destructor. Releases the location from the tape.
   *
   * The destructor removes the location only if `tape_location` ownes `loc_`,
   * i.e., if `valid_=1`.
   */
  ~tape_location() {
    if (valid_) {
      free_loc();
    }
  }

  /**
   * @brief Constructs a `tape_location` with the next available location on the
   * tape.
   */
  tape_location() : loc_(next_loc()) {}

  /** @brief Deleted copy constructor. */
  tape_location(const tape_location &) = delete;

  /** @brief Deleted copy assignment operator. */
  tape_location &operator=(const tape_location &) = delete;

  /**
   * @brief Move constructor. Transfers ownership of the location.
   *
   * @param other The `tape_location` to move from.
   */
  tape_location(tape_location &&other) noexcept : loc_(other.loc_) {
    other.valid_ = 0;
  };

  /**
   * @brief Move assignment operator. Transfers the location and frees the old
   * location.
   *
   * @param other The `tape_location` to move from.
   * @return Reference to the updated `tape_location`.
   */
  tape_location &operator=(tape_location &&other) noexcept {
    if (this == &other)
      return *this;

    // free location of *this
    free_loc();
    loc_ = other.loc();
    other.valid_ = 0;
    return *this;
  };

  /**
   * @brief Retrieves the tape location.
   *
   * @return The index of the location on the tape.
   */
  size_t loc() const { return loc_; }
};

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
   * @brief Constructor initializing an `adouble` with a new location on the
   * DefaultTape and puts the given value `coval` on the tape.
   *
   * @param coval Value that is stored on the tape at the new location.
   */
  adouble(double coval);

  /**
   * @brief Copy constructor. Creates a new `adouble` with a new location and
   * puts assignment operation onto the tape`
   * @param a The `adouble` to copy.
   */
  adouble(const adouble &a);

  /**
   * @brief Moving a's tape_location to the tape_location of this and set a's
   * state to invalid (valid = 0). This will tell the destruction that "a" does
   * not own its location. Thus, the location is not removed from the tape.
   * @param a The `adouble` to move.
   */
  adouble(adouble &&a) noexcept : tape_loc_{std::move(a.tape_loc_)} {
    a.valid = 0;
  }

  /** @brief Destructor. Releases the location from the tape.
   * The destructor is used to remove unused locations (tape_loc_.loc_) from the
   * tape. A location is only removed (free_loc), if the destructed adouble owns
   * the location. The adouble does not own its location if it is in an invalid
   * state (valid = 0). The state is only invalid, if the adouble was moved to a
   * new adouble. The location is reused for the new adouble in this case and
   * must remain on the tape.
   */
  ~adouble() {
    if (valid) {
      free_loc(tape_loc_.loc_);
    }
  }

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
  adouble &operator=(adouble &&a) noexcept {
    if (this == &a) {
      return *this;
    }
    // remove location of this from tape to ensure it can be reused
    free_loc(tape_loc_.loc_);

    tape_loc_ = tape_location{a.loc()};
    a.valid = 0;

    return *this;
  }
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
  double value() const { return currentTape().get_ad_value(loc()); }

  /**
   * @brief Retrieves the tape location of the `adouble`.
   * @return The location on the tape.
   */
  size_t loc() const { return tape_loc_.loc(); }

  // Mutators

  /**
   * @brief Updates the value stored at the tape location.
   * @param coval the new value to assign.
   */
  void value(const double coval) { currentTape().set_ad_value(loc(), coval); }

  // Type Conversions

  /**
   * @brief Converts the `adouble` to a `double` by returning its corresponding
   * `double` value from the tape.
   * @return The value of the `adouble` at the tape location.
   */
  explicit operator double() const { return value(); }

  // Arithmetic assignment Operators

  adouble &operator+=(const double coval);
  adouble &operator+=(const adouble &a);
  adouble &operator+=(adouble &&a);
  inline adouble &operator+=(const pdouble &p);

  adouble &operator-=(const double coval);
  adouble &operator-=(const adouble &a);
  adouble &operator-=(adouble &&a);
  inline adouble &operator-=(const pdouble &p);

  adouble &operator*=(const double coval);
  adouble &operator*=(const adouble &a);
  // adouble &operator*=(adouble &&a);
  inline adouble &operator*=(const pdouble &p);

  inline adouble &operator/=(const double coval);
  inline adouble &operator/=(const adouble &a);
  inline adouble &operator/=(adouble &&a);
  inline adouble &operator/=(const pdouble &p);

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
 * @brief The `pdouble` class represents a non-differentiable type, which acts
 * like a `double` on the tape. The main application of `pdouble` is the
 * modification of parameters on the tape without re-taping. For example:

adouble f(const pdouble& p, const adouble& x){
  return p * x;
}

int main() {
adouble indep;
pdouble p = 3.0;
double out[1];

trace_on(1);
indep <<= 2.0;
adouble out = f(p, x);
dep >> out[0];

double grad[1]

// compute d/dx p*x
gradient(1, 1, 2.0, grad);
std::cout << grad[0] << std::endl;

// change that is stored at `p`s location on tape 1 to 1.0
// one have to hand in the number of tracked `pdoubles` on the tape
set_param_vec(1, 1, 1.0);

// compute d/dx p*x
gradient(1, 1, 2.0, grad);
std::cout << grad[0] << std::endl;
}
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
   * @brief Constructor initializing a `pdouble` with a tape location and puts
   * the input `double` to this location on the parameter tape.
   * @param pval The initial value for the `pdouble` on the parameter tape.
   */
  explicit pdouble(const double pval);

  /**
   * @brief Converts the `pdouble` to an `adouble` by creating a new location
   * for the `adouble`, storing the assignment of the `pdouble` to th `adouble`
   * on the operations tape and storing the value of the `pdouble` at the
   * location of the `adouble`.
   * @return An `adouble` with associated value of the `pdouble` on the tape.
   */
  explicit operator adouble() const;

  /**
   * @brief Retrieves the location on the tape of the `pdouble`.
   * @return The location on the tape.
   */
  size_t loc() const { return tape_loc_.loc(); }

  /**
   * @brief Retrieves the current value stored at the tape location.
   * @return The value of the `pdouble`.
   */
  double value() const { return currentTape().get_pd_value(loc()); }

  /**
   * @brief Updates the value stored at the tape location.
   * @param pval The new value to assign.
   */
  void value(const double pval) { currentTape().set_pd_value(loc(), pval); }
};

std::ostream &operator<<(std::ostream &, const adouble &);
std::istream &operator>>(std::istream &, const adouble &);

/**
 * ADOUBLE ARTIHEMTICS
 */

/*--------------------------------------------------------------------------*/
/* sign operators */
ADOLC_DLL_EXPORT adouble operator+(const adouble &a);
ADOLC_DLL_EXPORT adouble operator+(adouble &&a);

ADOLC_DLL_EXPORT adouble operator-(const adouble &a);
ADOLC_DLL_EXPORT adouble operator-(adouble &&a);

/*--------------------------------------------------------------------------*/
/* binary operators */

/*--------------------------------------------------------------------------*/
ADOLC_DLL_EXPORT adouble operator+(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator+(adouble &&a, const adouble &b);
inline adouble operator+(const adouble &a, adouble &&b) {
  return std::move(b) + a;
}
inline adouble operator+(adouble &&a, adouble &&b) { return std::move(a) + b; }
ADOLC_DLL_EXPORT adouble operator+(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator+(const double coval, adouble &&a);
inline adouble operator+(const adouble &a, const double coval) {
  return coval + a;
}
inline adouble operator+(adouble &&a, const double coval) {
  return coval + std::move(a);
}
/*--------------------------------------------------------------------------*/

ADOLC_DLL_EXPORT adouble operator-(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator-(adouble &&a, const adouble &b);
inline adouble operator-(const adouble &a, adouble &&b) {
  return -(std::move(b)) + a;
}
inline adouble operator-(adouble &&a, adouble &&b) { return std::move(a) - b; }
ADOLC_DLL_EXPORT adouble operator-(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator-(const double coval, adouble &&a);
inline adouble operator-(const adouble &a, const double coval) {
  return (-coval) + a;
}
inline adouble operator-(adouble &&a, const double coval) {
  return (-coval) + std::move(a);
}

/*--------------------------------------------------------------------------*/

ADOLC_DLL_EXPORT adouble operator*(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator*(adouble &&a, const adouble &b);
inline adouble operator*(const adouble &a, adouble &&b) {
  return std::move(b) * a;
}
inline adouble operator*(adouble &&a, adouble &&b) { return std::move(a) * b; }
ADOLC_DLL_EXPORT adouble operator*(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator*(const double coval, adouble &&a);
inline adouble operator*(const adouble &a, const double coval) {
  return coval * a;
}
inline adouble operator*(adouble &&a, const double coval) {
  return coval * std::move(a);
}

/*--------------------------------------------------------------------------*/

ADOLC_DLL_EXPORT adouble operator/(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator/(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble operator/(const adouble &a, adouble &&b);
inline adouble operator/(adouble &&a, adouble &&b) { return std::move(a) / b; }
ADOLC_DLL_EXPORT adouble operator/(const double coval, const adouble &a);
ADOLC_DLL_EXPORT adouble operator/(const double coval, adouble &&a);
inline adouble operator/(const adouble &a, const double coval) {
  return a * (1.0 / coval);
}
inline adouble operator/(adouble &&a, const double coval) {
  return std::move(a) * (1.0 / coval);
}

/*--------------------------------------------------------------------------*/
/* Inlined Arithmetic assignments  */
inline adouble &adouble::operator/=(const double coval) {
  *this *= 1.0 / coval;
  return *this;
}
inline adouble &adouble::operator/=(const adouble &a) {
  *this *= 1.0 / a;
  return *this;
}
inline adouble &adouble::operator/=(adouble &&a) {
  *this *= 1.0 / std::move(a);
  return *this;
}
/*--------------------------------------------------------------------------*/
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

inline adouble log10(const adouble &a) {
  return log(a) / ADOLC_MATH_NSP::log(10.0);
}
inline adouble log10(adouble &&a) {
  return log(std::move(a)) / ADOLC_MATH_NSP::log(10.0);
}

ADOLC_DLL_EXPORT adouble sqrt(const adouble &a);
ADOLC_DLL_EXPORT adouble sqrt(adouble &&a);

ADOLC_DLL_EXPORT adouble cbrt(const adouble &a);
ADOLC_DLL_EXPORT adouble cbrt(adouble &&a);

ADOLC_DLL_EXPORT adouble sin(const adouble &a);
ADOLC_DLL_EXPORT adouble sin(adouble &&a);

ADOLC_DLL_EXPORT adouble cos(const adouble &a);
ADOLC_DLL_EXPORT adouble cos(adouble &&a);

inline adouble tan(const adouble &a) { return sin(a) / cos(a); }
inline adouble tan(adouble &&a) { return sin(a) / cos(a); }

ADOLC_DLL_EXPORT adouble asin(const adouble &a);
ADOLC_DLL_EXPORT adouble asin(adouble &&a);

ADOLC_DLL_EXPORT adouble acos(const adouble &a);
ADOLC_DLL_EXPORT adouble acos(adouble &&a);

ADOLC_DLL_EXPORT adouble atan(const adouble &a);
ADOLC_DLL_EXPORT adouble atan(adouble &&a);

/*--------------------------------------------------------------------------*/
inline adouble sinh(const adouble &a) {
  if (a.value() < 0.0) {
    adouble temp = exp(a);
    return 0.5 * (temp - 1.0 / temp);
  } else {
    adouble temp = exp(-a);
    return 0.5 * (1.0 / temp - temp);
  }
}
inline adouble sinh(adouble &&a) {
  if (a.value() < 0.0) {
    adouble temp = exp(std::move(a));
    return 0.5 * (temp - 1.0 / temp);
  } else {
    adouble temp = exp(-std::move(a));
    return 0.5 * (1.0 / temp - temp);
  }
}
/*--------------------------------------------------------------------------*/

inline adouble cosh(const adouble &a) {
  adouble temp = (a.value() < 0.0) ? exp(a) : exp(-a);
  return 0.5 * (temp + 1.0 / temp);
}
inline adouble cosh(adouble &&a) {
  adouble temp = (a.value() < 0.0) ? exp(std::move(a)) : exp(-std::move(a));
  return 0.5 * (temp + 1.0 / temp);
}
/*--------------------------------------------------------------------------*/

inline adouble tanh(const adouble &a) {
  if (a.value() < 0.0) {
    adouble temp = exp(2.0 * a);
    return (temp - 1.0) / (temp + 1.0);
  } else {
    adouble temp = exp((-2.0) * a);
    return (1.0 - temp) / (temp + 1.0);
  }
}
inline adouble tanh(adouble &&a) {
  if (a.value() < 0.0) {
    adouble temp = exp(2.0 * std::move(a));
    return (temp - 1.0) / (temp + 1.0);
  } else {
    adouble temp = exp((-2.0) * std::move(a));
    return (1.0 - temp) / (temp + 1.0);
  }
}
/*--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------*/
/* Min and Max */

ADOLC_DLL_EXPORT adouble fmin(const adouble &a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmin(adouble &&a, const adouble &b);
ADOLC_DLL_EXPORT adouble fmin(const adouble &a, adouble &&b);
inline adouble fmin(adouble &&a, adouble &&b) { return fmin(std::move(a), b); }
inline adouble fmin(const double coval, const adouble &a) {
  return fmin(adouble(coval), a);
  return fmin(adouble(coval), a);
}
inline adouble fmin(const double coval, adouble &&a) {
  return fmin(adouble(coval), std::move(a));
  return fmin(adouble(coval), std::move(a));
}
inline adouble fmin(const adouble &a, const double coval) {
  return (fmin(a, adouble(coval)));
  return (fmin(a, adouble(coval)));
}
inline adouble fmin(adouble &&a, const double coval) {
  return (fmin(std::move(a), adouble(coval)));
  return (fmin(std::move(a), adouble(coval)));
}

/*--------------------------------------------------------------------------*/

inline adouble fmax(const adouble &a, const adouble &b) {
  return (-fmin(-a, -b));
}
inline adouble fmax(adouble &&a, const adouble &b) {
  return (-fmin(-std::move(a), -b));
}
inline adouble fmax(const adouble &a, adouble &&b) {
  return (-fmin(-a, -std::move(b)));
}
inline adouble fmax(adouble &&a, adouble &&b) { return fmax(std::move(a), b); }
inline adouble fmax(const double coval, const adouble &a) {
  return (-fmin(-coval, -a));
}
inline adouble fmax(const double coval, adouble &&a) {
  return (-fmin(-coval, -std::move(a)));
}
inline adouble fmax(const adouble &a, const double coval) {
  return (-fmin(-a, -coval));
}
inline adouble fmax(adouble &&a, const double coval) {
  return (-fmin(-std::move(a), -coval));
}

/*--------------------------------------------------------------------------*/
/* ldexp and frexp */

inline adouble ldexp(const adouble &a, const int exp) {
  return a * ldexp(1.0, exp);
}
inline adouble ldexp(adouble &&a, const int exp) { return a * ldexp(1.0, exp); }

/*--------------------------------------------------------------------------*/

inline adouble frexp(const adouble &a, int *exp) {
  double coval = std::frexp(a.value(), exp);
  return adouble(coval);
  return adouble(coval);
}
inline adouble frexp(adouble &&a, int *exp) {
  const double coval = std::frexp(a.value(), exp);
  a.value(coval);
  return a;
}

/*---------------------------------------------------------------*/
/* Conditional assignments */

void condassign(adouble &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2);
inline void condassign(double &res, const double &cond, const double &arg1,
                       const double &arg2) {
  res = cond > 0 ? arg1 : arg2;
}
/*---------------------------------------------------------------*/
void condassign(adouble &res, const adouble &cond, const adouble &arg);
inline void condassign(double &res, const double &cond, const double &arg) {
  res = cond > 0 ? arg : res;
}
/*---------------------------------------------------------------*/
void condeqassign(adouble &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2);
inline void condeqassign(double &res, const double &cond, const double &arg1,
                         const double &arg2) {
  res = cond >= 0 ? arg1 : arg2;
}
/*---------------------------------------------------------------*/
inline void condeqassign(double &res, const double &cond, const double &arg) {
  res = cond >= 0 ? arg : res;
}
void condeqassign(adouble &res, const adouble &cond, const adouble &arg);

/*---------------------------------------------------------------*/
/* power function */
ADOLC_DLL_EXPORT adouble pow(const adouble &a, const double exp);
ADOLC_DLL_EXPORT adouble pow(adouble &&a, const double exp);
inline adouble pow(double coval, const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret;
  if (coval <= 0)
    fail(ADOLC_ERRORS::ADOLC_NONPOSITIVE_BASIS,
         std::source_location::current());

  condassign(ret, adouble{coval}, exp(a * ADOLC_MATH_NSP::log(coval)),
             adouble{ADOLC_MATH_NSP::pow(coval, a.value())});

  return ret;
}
inline adouble pow(const adouble &a, const adouble &b) {
  assert((a.value() >= 0) && "\nADOL-C message: negative basis deactivated\n ");
  assert(a.value() != 0 && "\nADOL-C message: zero basis deactivated\n ");

  adouble a1, a2, ret;
  adouble a1, a2, ret;

  condassign(a1, -b, adouble{ADOLC_MATH_NSP::pow(a.value(), b.value())},
  condassign(a1, -b, adouble{ADOLC_MATH_NSP::pow(a.value(), b.value())},
             pow(a, b.value()));
  condassign(a2, fabs(a), pow(a, b.value()), a1);
  condassign(ret, a, exp(b * log(a)), a2);
  return ret;
}

/*---------------------------------------------------------------*/
/* atan(x, y) */
inline adouble atan2(const adouble &a, const adouble &b) {
  adouble a1, a2, ret, sy;
  adouble a1, a2, ret, sy;
  const double pihalf = ADOLC_MATH_NSP::asin(1.0);
  condassign(sy, a, adouble{1.0}, adouble{-1.0});
  condassign(sy, a, adouble{1.0}, adouble{-1.0});
  condassign(a1, b, atan(a / b), atan(a / b) + sy * 2 * pihalf);
  condassign(a2, fabs(a), sy * pihalf - atan(b / a), adouble{0.0});
  condassign(a2, fabs(a), sy * pihalf - atan(b / a), adouble{0.0});
  condassign(ret, fabs(b) - fabs(a), a1, a2);
  return ret;
}

/*---------------------------------------------------------------*/
/* Advector functions  */
// copies src to dest; both should be of size "size"
void ADOLC_DLL_EXPORT adolc_vec_copy(adouble *const dest,
                                     const adouble *const src, size_t size);

// computes canonical scalarproduct, src and dest must have size "size"
void ADOLC_DLL_EXPORT adolc_vec_dot(adouble *const dest,
                                    const adouble *const src, size_t size);

// res = a*vec_a + vec_b, vec_a and vec_b must have same size "size"
void ADOLC_DLL_EXPORT adolc_vec_axpy(adouble *const res, const adouble &a,
                                     const adouble *const vec_a,
                                     const adouble *const vec_b, size_t size);

/**
 * PDOUBLE ARTIHEMTICS
 */

/*--------------------------------------------------------------------------*/
/* Arithmetic assignments */
inline adouble &adouble::operator+=(const pdouble &p) {
  return *this += adouble(p);
}
inline adouble &adouble::operator-=(const pdouble &p) {
  return *this -= adouble(p);
}
inline adouble &adouble::operator*=(const pdouble &p) {
  return *this *= adouble(p);
}
inline adouble &adouble::operator/=(const pdouble &p) {
  return *this /= adouble(p);
}

/*--------------------------------------------------------------------------*/
/* sign operators */
inline adouble operator-(const pdouble &p) { return -adouble(p); }
inline adouble operator+(const pdouble &p) { return adouble(p); };

/*--------------------------------------------------------------------------*/
/* Binary Operations */
inline adouble operator+(const pdouble &p, const adouble &a) {
  return a + adouble(p);
};
inline adouble operator+(const adouble &a, const pdouble &p) {
  return a + adouble(p);
};
inline adouble operator+(const pdouble &p, const double coval) {
  return coval + adouble(p);
}
inline adouble operator+(const double coval, const pdouble &p) {
  return coval + adouble(p);
}
/*--------------------------------------------------------------------------*/
inline adouble operator-(const adouble &a, const pdouble &p) {
  return a - adouble(p);
}
inline adouble operator-(const pdouble &p, const adouble &a) {
  return (-a) + adouble(p);
}
inline adouble operator-(const pdouble &p, const double coval) {
  return adouble(p) - coval;
}
inline adouble operator-(const double coval, const pdouble &p) {
  return coval - adouble(p);
}
/*--------------------------------------------------------------------------*/
inline adouble operator*(const adouble &a, const pdouble &p) {
  return a * adouble(p);
}
inline adouble operator*(const pdouble &p, const adouble &a) {
  return a * adouble(p);
}
inline adouble operator*(const pdouble &p, const double coval) {
  return coval * adouble(p);
}
inline adouble operator*(const double coval, const pdouble &p) {
  return coval * adouble(p);
}
/*--------------------------------------------------------------------------*/
inline adouble operator/(const pdouble &p, const adouble &a) {
  return adouble(p) / a;
}
inline adouble operator/(const adouble &a, const pdouble &p) {
  return a / adouble(p);
}
inline adouble operator/(const pdouble &p, const double coval) {
  return adouble(p) / coval;
}
inline adouble operator/(const double coval, const pdouble &p) {
  return coval / adouble(p);
}

/*--------------------------------------------------------------------------*/
/* Comparions */
#ifdef ADOLC_ADVANCED_BRANCHING
inline adouble operator!=(const adouble &a, const pdouble &p) {
  return a != adouble(p);
}
inline adouble operator!=(const pdouble &p, const adouble &a) {
  return adouble(p) != a;
}

inline adouble operator==(const adouble &a, const pdouble &p) {
  return a == adouble(p);
}
inline adouble operator==(const pdouble &p, const adouble &a) {
  return adouble(p) == a;
}

inline adouble operator<=(const adouble &a, const pdouble &p) {
  return a <= adouble(p);
}
inline adouble operator<=(const pdouble &p, const adouble &a) {
  return adouble(p) <= a;
}

inline adouble operator>=(const adouble &a, const pdouble &p) {
  return a >= adouble(p);
}
inline adouble operator>=(const pdouble &p, const adouble &a) {
  return adouble(p) >= a;
}

inline adouble operator>(const adouble &a, const pdouble &p) {
  return a > adouble(p);
}
inline adouble operator>(const pdouble &p, const adouble &a) {
  return adouble(p) > a;
}
inline adouble operator<(const adouble &a, const pdouble &p) {
  return a < adouble(p);
}
inline adouble operator<(const pdouble &p, const adouble &a) {
  return adouble(p) < a;
}
#else // ADOLC_ADVANCED_BRANCHING

inline bool operator!=(const adouble &a, const pdouble &p) {
  return a != adouble(p);
}
inline bool operator!=(const pdouble &p, const adouble &a) {
  return adouble(p) != a;
}

inline bool operator==(const adouble &a, const pdouble &p) {
  return a == adouble(p);
}
inline bool operator==(const pdouble &p, const adouble &a) {
  return adouble(p) == a;
}

inline bool operator<=(const adouble &a, const pdouble &p) {
  return a <= adouble(p);
}
inline bool operator<=(const pdouble &p, const adouble &a) {
  return adouble(p) <= a;
}

inline bool operator>=(const adouble &a, const pdouble &p) {
  return a >= adouble(p);
}
inline bool operator>=(const pdouble &p, const adouble &a) {
  return adouble(p) >= a;
}

inline bool operator>(const adouble &a, const pdouble &p) {
  return a > adouble(p);
}
inline bool operator>(const pdouble &p, const adouble &a) {
  return adouble(p) > a;
}

inline bool operator<(const adouble &a, const pdouble &p) {
  return a < adouble(p);
}
inline bool operator<(const pdouble &p, const adouble &a) {
  return adouble(p) < a;
}

#endif // ADOLC_ADVANCED_BRANCHING

/*--------------------------------------------------------------------------*/
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

/*--------------------------------------------------------------------------*/
/* Min and Max */
inline adouble fmin(const adouble &a, const pdouble &p) {
  return fmin(a, adouble(p));
}
inline adouble fmin(const pdouble &p, const adouble &a) {
  return fmin(adouble(p), a);
}
inline adouble fmin(const pdouble &p, const pdouble &q) {
  return fmin(adouble(p), adouble(q));
}
inline adouble fmin(const double coval, const pdouble &p) {
  return fmin(coval, adouble(p));
}
inline adouble fmin(const pdouble &p, const double coval) {
  return fmin(adouble(p), coval);
}
/*--------------------------------------------------------------------------*/
inline adouble fmax(const adouble &a, const pdouble &p) {
  return fmax(a, adouble(p));
}
inline adouble fmax(const pdouble &p, const adouble &a) {
  return fmax(adouble(p), a);
}
inline adouble fmax(const pdouble &p, const pdouble &q) {
  return fmax(adouble(p), adouble(q));
}
inline adouble fmax(const double coval, const pdouble &p) {
  return fmax(coval, adouble(p));
}
inline adouble fmax(const pdouble &p, const double coval) {
  return fmax(adouble(p), coval);
}

/*--------------------------------------------------------------------------*/
/* ldexp and frexp */
inline adouble ldexp(const pdouble &p, const int exp) {
  return ldexp(adouble(p), exp);
}
inline adouble frexp(const pdouble &p, int *exp) {
  return frexp(adouble(p), exp);
}
/*--------------------------------------------------------------------------*/
/* power function */
inline adouble pow(const adouble &a, const pdouble &p) {
  return pow(a, adouble(p));
}
inline adouble pow(const pdouble &p, const adouble &a) {
  return pow(adouble(p), a);
}
inline adouble pow(const pdouble &p, const double coval) {
  return pow(adouble(p), coval);
}

/*--------------------------------------------------------------------------*/
/* atan(x, y) */
inline adouble atan2(const pdouble &p, const adouble &a) {
  return atan2(adouble(p), a);
}
inline adouble atan2(const adouble &a, const pdouble &p) {
  return atan2(a, adouble(p));
}

//-----------------------------------------------------------
/* User defined version of logarithm to test extend_quad macro */
ADOLC_DLL_EXPORT double myquad(double x);

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
