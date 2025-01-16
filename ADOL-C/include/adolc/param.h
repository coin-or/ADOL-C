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

#ifndef ADOLC_PARAM_H
#define ADOLC_PARAM_H
#include "adouble.h"
#include "internal/common.h"

class ADOLC_DLL_EXPORT pdouble {

public:
  ~pdouble() = default;
  pdouble(const pdouble &) = delete;
  pdouble() = delete;
  pdouble(pdouble &&) = delete;
  pdouble &operator=(pdouble &&) = delete;

  pdouble(const double pval);
  explicit pdouble(tape_location tape_loc);

  static pdouble mkparam(const double pval);
  static inline pdouble getparam(size_t loc_) { return pdouble{loc_}; };
  explicit operator adouble() const;

  inline size_t getLoc() const { return tape_loc_.loc_; }
  inline double getValue() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc_.loc_];
  }
  inline void setValue(const double pval) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.pStore[tape_loc_.loc_] = pval;
  }

private:
  tape_location tape_loc_;
};

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
inline bool operator<=(const pdouble &p, const adouble &a) { return (a >= p); }
inline bool operator<=(const pdouble &p, adouble &&a) {
  return (std::move(a) >= p);
}

inline bool operator>=(const adouble &a, const pdouble &p) {
  return ((a - p) >= 0);
}
inline bool operator>=(const pdouble &p, const adouble &a) { return (a <= p); }
inline bool operator>=(const pdouble &p, adouble &&a) {
  return (std::move(a) <= p);
}

inline bool operator>(const adouble &a, const pdouble &p) {
  return ((a - p) > 0);
}
inline bool operator>(const pdouble &p, const adouble &a) { return (a < p); }
inline bool operator>(const pdouble &p, adouble &&a) {
  return (std::move(a) < p);
}

inline bool operator<(const adouble &a, const pdouble &p) {
  return ((a - p) < 0);
}
inline bool operator<(const pdouble &p, const adouble &a) { return (a > p); }
inline bool operator<(const pdouble &p, adouble &&a) {
  return (std::move(a) > p);
}
#endif // ADOLC_ADVANCED_BRANCHING

adouble operator-(const pdouble &p);
adouble operator+(const adouble &a, const pdouble &p);
adouble operator+(adouble &&a, const pdouble &p);
inline adouble operator+(const pdouble &p, const adouble &a) { return a + p; };
inline adouble operator+(const pdouble &p, adouble &&a) {
  return std::move(a) + p;
};
inline adouble operator+(const pdouble &p, const double coval) {
  return coval + adouble(p)
};
inline adouble operator+(const double coval, const pdouble &p) {
  return coval + adouble(p)
};

adouble operator-(const adouble &a, const pdouble &p);
adouble operator-(adouble &&a, const pdouble &p);
inline adouble operator-(const pdouble &p, const adouble &a) {
  return (-a) + p
};
inline adouble operator-(const pdouble &p, adouble &&a) {
  return (-std::move(a)) + p
};
inline adouble operator-(const pdouble &p, const double coval) {
  return adouble(p) - coval
};
inline adouble operator-(const double coval, const pdouble &p) {
  return coval - adouble(p)
};

adouble operator*(const adouble &a, const pdouble &p);
adouble operator*(adouble &&a, const pdouble &p);
inline adouble operator*(const pdouble &p, const adouble &a) { return a * p; };
inline adouble operator*(const pdouble &p, adouble &&a) {
  return std::move(a) * p;
};
inline adouble operator*(const pdouble &p, const double coval) {
  return coval * adouble(p);
};
inline adouble operator*(const double coval, const pdouble &p) {
  return coval * adouble(p)
};

adouble recipr(const pdouble &);

adouble operator/(const pdouble &p, const adouble &a);
adouble operator/(const pdouble &p, adouble &&a);
inline adouble operator/(const adouble &a, const pdouble &p) {
  return a * recipr(p);
};
inline adouble operator/(const pdouble &p, const double coval) {
  return adouble(p) / coval;
};
inline adouble operator/(const double coval, const pdouble &p) {
  return coval * recipr(p);
};

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

#endif // ADOLC_PARAM_H