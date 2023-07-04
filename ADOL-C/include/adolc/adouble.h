/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: adouble.h contains the basis for the class of adouble
           included here are all the possible functions defined on
           the adouble class.  Notice that, as opposed to ealier versions,
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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <cmath>
#include <stdexcept>

#if !defined(SWIGPRE)
using std::cout;
using std::cin;
using std::cerr;
using std::ostream;
using std::istream;
using std::logic_error;
#endif

#include <adolc/internal/common.h>

/* NOTICE: There are automatic includes at the end of this file! */

/****************************************************************************/
/*                                             FORWARD DECLARATIONS (TAPES) */

/*--------------------------------------------------------------------------*/
class adouble;
class adub;
class badouble;
class pdouble;

/*--------------------------------------------------------------------------*/
void ADOLC_DLL_EXPORT condassign( double &res, const double &cond,
                                  const double &arg1, const double &arg2 );
void ADOLC_DLL_EXPORT condassign( double &res, const double &cond,
                                  const double &arg );

void ADOLC_DLL_EXPORT condeqassign( double &res, const double &cond,
                                    const double &arg1, const double &arg2 );
void ADOLC_DLL_EXPORT condeqassign( double &res, const double &cond,
                                    const double &arg );

/****************************************************************************/
/*                                                           CLASS BADOUBLE */

/**
   The class badouble contains the basic definitions for 
   the arithmetic operations, comparisons, etc. 
   This is a basic class from which the adub and adouble are 
   derived.  Notice that the constructors/destructors for 
   the class badouble are of the trivial variety.  This is the
   main difference among badoubles, adubs, and adoubles.
*/
class ADOLC_DLL_EXPORT badouble {
    friend ADOLC_DLL_EXPORT class pdouble;
protected:
    locint location;
    badouble( void ) {};
    // Copy constructor:
    // must be public when using gcc >= 3.4 and gcc <= 4.3.0
    // (see GCC 3.4 Release Series - Changes, New Features, and Fixes)
    // so we make it protected for newer compilers again.
    badouble( const badouble& a ) {};           /* ctor */
    explicit badouble( locint lo ) {
        location = lo;
        isInit = true;
    };

    bool isInit;  // marker if the badouble is properly initialized

public:

    ~badouble();

    /*--------------------------------------------------------------------------*/    
    inline locint loc( void ) const;                         /* Helpful stuff */

    /*------------------------------------------------------------------------*/
    badouble& operator >>= ( double& );                        /* Assignments */
    badouble& operator <<= ( double );
    void declareIndependent ();
    void declareDependent ();
    badouble& operator = ( double );
    badouble& operator = ( const badouble& );
    badouble& operator = ( const adub& );
    double getValue() const;
    inline double value() const {
        return getValue();
    }
    explicit operator double();
    explicit operator double const&() const;
    explicit operator double&&();
    void setValue ( const double );
    /* badouble& operator = ( const adouble& );
       !!! olvo 991210: was the same as badouble-assignment */

    /*--------------------------------------------------------------------------*/
    friend ADOLC_DLL_EXPORT std::ostream& operator << ( std::ostream&, const badouble& );  /* IO friends */
    friend ADOLC_DLL_EXPORT std::istream& operator >> ( std::istream&, const badouble& );

    /*------------------------------------------------------------------------*/
    badouble& operator += ( double );               /* Operation + Assignment */
    badouble& operator += ( const badouble& );
    badouble& operator -= ( double y );
    badouble& operator -= ( const badouble& );
    badouble& operator *= ( double );
    badouble& operator *= ( const badouble& );
    badouble& operator /= ( double );
    badouble& operator /= ( const badouble& );
    /* olvo 991122 n2l: new special op_codes */
    badouble& operator += ( const adub& );
    badouble& operator -= ( const adub& );

    /*--------------------------------------------------------------------------*/
    badouble& operator = (const pdouble&);
    badouble& operator += (const pdouble&);
    badouble& operator -= (const pdouble&);
    badouble& operator *= (const pdouble&);
    inline badouble& operator /= (const pdouble&);
    /*--------------------------------------------------------------------------*/
    /* Comparison (friends) */
#if !defined(ADOLC_ADVANCED_BRANCHING)
    inline friend bool operator != ( const badouble&, const badouble& );
    inline friend bool operator == ( const badouble&, const badouble& );
    inline friend bool operator <= ( const badouble&, const badouble& );
    inline friend bool operator >= ( const badouble&, const badouble& );
    inline friend bool operator >  ( const badouble&, const badouble& );
    inline friend bool operator <  ( const badouble&, const badouble& );
#endif
    inline friend bool operator != ( double, const badouble& );
    friend ADOLC_DLL_EXPORT bool operator != ( const badouble&, double );
    inline friend bool operator == ( double, const badouble& );
    friend ADOLC_DLL_EXPORT bool operator == ( const badouble&, double );
    inline friend bool operator <= ( double, const badouble& );
    friend ADOLC_DLL_EXPORT bool operator <= ( const badouble&, double );
    inline friend bool operator >= ( double, const badouble& );
    friend ADOLC_DLL_EXPORT bool operator >= ( const badouble&, double );
    inline friend bool operator >  ( double, const badouble& );
    friend ADOLC_DLL_EXPORT bool operator >  ( const badouble&, double );
    inline friend bool operator <  ( double, const badouble& );
    friend ADOLC_DLL_EXPORT bool operator <  ( const badouble&, double );


    /*--------------------------------------------------------------------------*/
    /* Functions friends with both badouble and adub */
#define _IN_CLASS_ 1
#define _IN_BADOUBLE_ 1
#include <adolc/internal/adubfunc.h>
#undef _IN_BADOUBLE_
#undef _IN_CLASS_

    /*--------------------------------------------------------------------------*/
    /* special operators (friends) */
    friend ADOLC_DLL_EXPORT adouble atan2 ( const badouble&, const badouble& );
    /* uses condassign internally */
    friend ADOLC_DLL_EXPORT adouble pow   ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adouble pow   ( double, const badouble& );
    /* User defined version of logarithm to test extend_quad macro */
    friend ADOLC_DLL_EXPORT adouble myquad( const badouble& );

    /*--------------------------------------------------------------------------*/
    /* Conditionals */
    friend ADOLC_DLL_EXPORT void condassign( adouble &res, const badouble &cond,
            const badouble &arg1, const badouble &arg2 );
    friend ADOLC_DLL_EXPORT void condassign( adouble &res, const badouble &cond,
            const badouble &arg );
    friend ADOLC_DLL_EXPORT void condeqassign( adouble &res, const badouble &cond,
            const badouble &arg1, const badouble &arg2 );
    friend ADOLC_DLL_EXPORT void condeqassign( adouble &res, const badouble &cond,
            const badouble &arg );

#define _IN_CLASS_ 1
#define _IN_BADOUBLE_ 1
#include <adolc/internal/paramfunc.h>
#undef _IN_BADOUBLE_
#undef _IN_CLASS_

};



/****************************************************************************/
/*                                                               CLASS ADUB */

/*
   The class Adub
   ---- Basically used as a temporary result.  The address for an
        adub is usually generated within an operation.  That address
        is "freed" when the adub goes out of scope (at destruction time).
   ---- operates just like a badouble, but it has a destructor defined for it.
*/
#if !defined(SWIGPRE)
/* s = adolc_vec_dot(x,y,size); <=> s = <x,y>_2 */
ADOLC_DLL_EXPORT adub adolc_vec_dot(const adouble*const, const adouble*const, locint);
#endif

class ADOLC_DLL_EXPORT adub:public badouble {
    friend ADOLC_DLL_EXPORT class adouble;
    friend ADOLC_DLL_EXPORT class advector;
    friend ADOLC_DLL_EXPORT class adubref;
    friend ADOLC_DLL_EXPORT class pdouble;

protected:
   /* this is the only logically legal constructor, which can be called by 
    * friend classes and functions 
    */
   adub( locint lo ) : badouble(lo) {} 

public:
    /*--------------------------------------------------------------------------*/
#if !defined(SWIGPRE)
    /* s = adolc_vec_dot(x,y,size); <=> s = <x,y>_2 */
    friend adub adolc_vec_dot(const adouble*const, const adouble*const, locint);
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
class ADOLC_DLL_EXPORT adouble:public badouble {
    friend ADOLC_DLL_EXPORT class advector;
    friend ADOLC_DLL_EXPORT class pdouble;
protected:
    void initInternal(void); // Init for late initialization
public:
    adouble( const adub& );
    adouble( const adouble& );
    adouble( void );
    adouble( double );
    /* adub prevents postfix operators to occur on the left
       side of an assignment which would not work  */
#if !defined(SWIGPRE)
    adub operator++( int );
    adub operator--( int );
#else
    adub* operator++( int );
    adub* operator--( int );
#endif
    badouble& operator++( void );
    badouble& operator--( void );
    /*   inline double value(); */

    adouble& operator = ( double );
    adouble& operator = ( const badouble& );
    adouble& operator = ( const adouble& );
    adouble& operator = ( const adub& );
    adouble& operator = (const pdouble&);
    
    inline locint loc(void) const;

#if defined(ADOLC_DEFAULT_CONTIG_LOC)
    void *operator new[](size_t sz) {
        void *p = ::new char[sz];
        size_t n = (sz - sizeof(size_t))/sizeof(adouble);
        ensureContiguousLocations(n);
        return p;
    }
    void operator delete[](void* p) {
        ::delete[] (char*)p;
    }
#endif
};

#endif /* __cplusplus */

#include <adolc/param.h>
#include <adolc/advector.h>

#ifdef __cplusplus
/****************************************************************************/
/*                                                       INLINE DEFINITIONS */

/*--------------------------------------------------------------------------*/
inline locint badouble::loc( void ) const {
    return location;
}

inline locint adouble::loc( void ) const {
    const_cast<adouble*>(this)->initInternal();
    return location;
}

/*--------------------------------------------------------------------------*/
/* Comparison */

#if !defined(ADOLC_ADVANCED_BRANCHING)
inline bool operator != ( const badouble& u, const badouble& v ) {
    return (u-v != 0);
}

inline bool operator == ( const badouble& u, const badouble& v ) {
    return (u-v == 0);
}

inline bool operator <= ( const badouble& u, const badouble& v ) {
    return (u-v <= 0);
}

inline bool operator >= ( const badouble& u, const badouble& v ) {
    return (u-v >= 0);
}

inline bool operator > ( const badouble& u, const badouble& v ) {
    return (u-v > 0);
}

inline bool operator < ( const badouble& u, const badouble& v ) {
    return (u-v < 0);
}
#endif

inline bool operator != ( double coval, const badouble& v) {
    if (coval)
        return (-coval+v != 0);
    else
        return (v != 0);
}

inline bool operator == ( double coval, const badouble& v) {
    if (coval)
        return (-coval+v == 0);
    else
        return (v == 0);
}

inline bool operator <= ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v >= 0);
    else
        return (v >= 0);
}

inline bool operator >= ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v <= 0);
    else
        return (v <= 0);
}

inline bool operator > ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v < 0);
    else
        return (v < 0);
}

inline bool operator < ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v > 0);
    else
        return (v > 0);
}

#if !defined(SWIGPRE)
/*--------------------------------------------------------------------------*/
/* Adding a floating point from an adouble  */
inline adub operator + ( const badouble& x , double coval ) {
    return coval + x;
}

/* Subtract a floating point from an adouble  */
inline adub operator - ( const badouble& x , double coval ) {
    return (-coval) + x;
}

/*--------------------------------------------------------------------------*/
/* Multiply an adouble by a floating point */
inline adub operator * (const badouble& x, double coval) {
    return coval * x;
}

/*--------------------------------------------------------------------------*/
/* Divide an adouble by a floating point */
inline adub operator / (const badouble& x, double coval) {
    return (1.0/coval) * x;
}
#endif


inline badouble& badouble::operator /= (const pdouble& p) {
    *this *= recipr(p);
    return *this;
}

/* numeric_limits<adouble> specialization
 *
 * All methods return double instead of adouble, because these values
 * never depend on the independent variables.
 */
template<>
struct std::numeric_limits<adouble>
{
    static constexpr bool is_specialized = true;

    static constexpr double
    min() noexcept { return std::numeric_limits<double>::min(); }

    static constexpr double
    max() noexcept { return std::numeric_limits<double>::max(); }

    static constexpr double
    lowest() noexcept { return std::numeric_limits<double>::lowest(); }

    static constexpr int digits = std::numeric_limits<double>::digits;
    static constexpr int digits10 = std::numeric_limits<double>::digits10;
    static constexpr int max_digits10 = std::numeric_limits<double>::max_digits10;
    static constexpr bool is_signed = std::numeric_limits<double>::is_signed;
    static constexpr bool is_integer = std::numeric_limits<double>::is_integer;
    static constexpr bool is_exact = std::numeric_limits<double>::is_exact;
    static constexpr int radix = std::numeric_limits<double>::radix;

    static constexpr double
    epsilon() noexcept { return std::numeric_limits<double>::epsilon(); }

    static constexpr double
    round_error() noexcept { return std::numeric_limits<double>::round_error(); }

    static constexpr int min_exponent = std::numeric_limits<double>::min_exponent;
    static constexpr int min_exponent10 = std::numeric_limits<double>::min_exponent10;
    static constexpr int max_exponent = std::numeric_limits<double>::max_exponent;
    static constexpr int max_exponent10 = std::numeric_limits<double>::max_exponent10;

    static constexpr bool has_infinity = std::numeric_limits<double>::has_infinity;
    static constexpr bool has_quiet_NaN = std::numeric_limits<double>::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = std::numeric_limits<double>::has_signaling_NaN;
    static constexpr float_denorm_style has_denorm = std::numeric_limits<double>::has_denorm;
    static constexpr bool has_denorm_loss = std::numeric_limits<double>::has_denorm_loss;

    static constexpr double
    infinity() noexcept { return std::numeric_limits<double>::infinity(); }

    static constexpr double
    quiet_NaN() noexcept { return std::numeric_limits<double>::quiet_NaN(); }

    static constexpr double
    signaling_NaN() noexcept { return std::numeric_limits<double>::signaling_NaN(); }

    static constexpr double
    denorm_min() noexcept { return std::numeric_limits<double>::denorm_min(); }

    static constexpr bool is_iec559 = std::numeric_limits<double>::is_iec559;
    static constexpr bool is_bounded = std::numeric_limits<double>::is_bounded;
    static constexpr bool is_modulo = std::numeric_limits<double>::is_modulo;

    static constexpr bool traps = std::numeric_limits<double>::traps;
    static constexpr bool tinyness_before = std::numeric_limits<double>::tinyness_before;
    static constexpr float_round_style round_style = std::numeric_limits<double>::round_style;
};

/****************************************************************************/
/*                                                                THAT'S ALL*/
#endif /* __cplusplus */
#endif /* ADOLC_ADOUBLE_H */
