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
               Benjamin Letschert

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
#include <cmath>
using std::cout;
using std::cin;
using std::cerr;
using std::ostream;
using std::istream;

#include <adolc/common.h>

/* NOTICE: There are automatic includes at the end of this file! */

#undef TAPELESS
#undef SAFE
#if defined(ADOLC_TAPELESS)
#  define TAPELESS
#  undef SAFE
#endif

#if defined(SAFE_ADOLC_TAPELESS)
#  define TAPELESS
#  define SAFE
#endif

#if !defined(TAPELESS)

/****************************************************************************/
/*                                             FORWARD DECLARATIONS (TAPES) */

/*--------------------------------------------------------------------------*/
class adouble;
class adub;
class badouble;
class adubv;
/* class doublev;  that's history */

/*--------------------------------------------------------------------------*/
void ADOLC_DLL_EXPORT condassign( double &res, const double &cond,
                                  const double &arg1, const double &arg2 );
void ADOLC_DLL_EXPORT condassign( double &res, const double &cond,
                                  const double &arg );

#if !defined(_ISOC99_SOURCE) && !defined(__USE_ISOC99) && !defined(__APPLE_CC__)
double ADOLC_DLL_EXPORT fmin( const double &x, const double &y );
double ADOLC_DLL_EXPORT fmax( const double &x, const double &y );
#endif


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
    friend ADOLC_DLL_EXPORT class badoublev;
protected:
    locint location;
    badouble( void ) {};
    // must be public when using gcc >= 3.4 ( problems with value() )
    // (see GCC 3.4 Release Series - Changes, New Features, and Fixes)
    //
    // badouble( const badouble& a ) {location = a.location;};
    badouble( locint lo ) {
        location = lo;
    };

public:
    /*--------------------------------------------------------------------------*/
    badouble( const badouble& a ) {
        location = a.location;
    }
    ;           /* ctor */

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
    inline double value() {
        return getValue();
    }
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
    /* Comparison (friends) */
    inline friend int operator != ( const badouble&, const badouble& );
    inline friend int operator != ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator != ( const badouble&, double );
    inline friend int operator == ( const badouble&, const badouble& );
    inline friend int operator == ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator == ( const badouble&, double );
    inline friend int operator <= ( const badouble&, const badouble& );
    inline friend int operator <= ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator <= ( const badouble&, double );
    inline friend int operator >= ( const badouble&, const badouble& );
    inline friend int operator >= ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator >= ( const badouble&, double );
    inline friend int operator >  ( const badouble&, const badouble& );
    inline friend int operator >  ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator >  ( const badouble&, double );
    inline friend int operator <  ( const badouble&, const badouble& );
    inline friend int operator <  ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator <  ( const badouble&, double );


    /*--------------------------------------------------------------------------*/
    /* sign operators (friends) */
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble& x );
    friend ADOLC_DLL_EXPORT adub operator - ( const badouble& x );

    /*--------------------------------------------------------------------------*/
    /* binary operators (friends) */
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator + ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub operator - ( const badouble&, const badouble& );
    inline friend adub operator - ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub operator - ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator * ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator * ( double, const badouble& );
    inline friend adub operator * ( const badouble&, double );
    inline friend adub operator / ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub operator / ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator / ( double, const badouble& );

    /*--------------------------------------------------------------------------*/
    /* unary operators (friends) */
    friend ADOLC_DLL_EXPORT adub exp  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub log  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub sqrt ( const badouble& );
    friend ADOLC_DLL_EXPORT adub sin  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub cos  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub tan  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub asin ( const badouble& );
    friend ADOLC_DLL_EXPORT adub acos ( const badouble& );
    friend ADOLC_DLL_EXPORT adub atan ( const badouble& );

    /*--------------------------------------------------------------------------*/
    /* special operators (friends) */
    friend ADOLC_DLL_EXPORT adouble atan2 ( const badouble&, const badouble& );
    /* no internal use of condassign: */
    friend ADOLC_DLL_EXPORT adub    pow   ( const badouble&, double );
    /* uses condassign internally */
    friend ADOLC_DLL_EXPORT adouble pow   ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adouble pow   ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub    log10 ( const badouble& );
    /* User defined version of logarithm to test extend_quad macro */
    friend ADOLC_DLL_EXPORT adouble myquad( const badouble& );

    /*--------------------------------------------------------------------------*/
    /* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
    friend ADOLC_DLL_EXPORT adub sinh  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub cosh  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub tanh  ( const badouble& );
#if defined(ATRIG_ERF)
    friend ADOLC_DLL_EXPORT adub asinh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub acosh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub atanh ( const badouble& );
#endif

    friend ADOLC_DLL_EXPORT adub fabs  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub ceil  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub floor ( const badouble& );

    friend ADOLC_DLL_EXPORT adub fmax ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmax ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmax ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub fmin ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmin ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmin ( const badouble&, double );

    friend ADOLC_DLL_EXPORT adub ldexp ( const badouble&, int );
    friend ADOLC_DLL_EXPORT adub frexp ( const badouble&, int* );
    friend ADOLC_DLL_EXPORT adub erf   ( const badouble& );

    /*--------------------------------------------------------------------------*/
    /* Conditionals */
    friend ADOLC_DLL_EXPORT void condassign( adouble &res, const adouble &cond,
            const adouble &arg1, const adouble &arg2 );
    friend ADOLC_DLL_EXPORT void condassign( adouble &res, const adouble &cond,
            const adouble &arg );
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

class ADOLC_DLL_EXPORT adub:public badouble {
    friend ADOLC_DLL_EXPORT class adouble;
#if GCC_VERSION >= 4003
    adub( adub const &) {}
#endif
protected:
    adub( locint lo ):badouble(lo) {};
    adub( void ):badouble(0) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal default construction of adub"
                " variable\n");
        exit(-2);
    };
    adub( double ):badouble(0) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal  construction of adub variable"
                " from double\n");
        exit(-2);
    };

public:

    /*--------------------------------------------------------------------------*/
    /* sign operators (friends) */
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble& x );
    friend ADOLC_DLL_EXPORT adub operator - ( const badouble& x );

    /*--------------------------------------------------------------------------*/
    /* binary operators (friends) */
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator + ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub operator - ( const badouble&, const badouble& );
    inline friend adub operator - ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub operator - ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator * ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator * ( double, const badouble& );
    inline friend adub operator * ( const badouble&, double );
    inline friend adub operator / ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub operator / ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator / ( double, const badouble& );

    /*--------------------------------------------------------------------------*/
    /* unary operators (friends) */
    friend ADOLC_DLL_EXPORT adub exp  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub log  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub sqrt ( const badouble& );
    friend ADOLC_DLL_EXPORT adub sin  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub cos  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub tan  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub asin ( const badouble& );
    friend ADOLC_DLL_EXPORT adub acos ( const badouble& );
    friend ADOLC_DLL_EXPORT adub atan ( const badouble& );

    /*--------------------------------------------------------------------------*/
    /* special operators (friends) */
    /* no internal use of condassign: */
    friend ADOLC_DLL_EXPORT adub    pow   ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub    log10 ( const badouble& );

    /*--------------------------------------------------------------------------*/
    /* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
    friend ADOLC_DLL_EXPORT adub sinh  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub cosh  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub tanh  ( const badouble& );
#if defined(ATRIG_ERF)
    friend ADOLC_DLL_EXPORT adub asinh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub acosh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub atanh ( const badouble& );
#endif

    friend ADOLC_DLL_EXPORT adub fabs  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub ceil  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub floor ( const badouble& );

    friend ADOLC_DLL_EXPORT adub fmax ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmax ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmax ( const badouble&, double );
    friend ADOLC_DLL_EXPORT adub fmin ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmin ( double, const badouble& );
    friend ADOLC_DLL_EXPORT adub fmin ( const badouble&, double );

    friend ADOLC_DLL_EXPORT adub ldexp ( const badouble&, int );
    friend ADOLC_DLL_EXPORT adub frexp ( const badouble&, int* );
    friend ADOLC_DLL_EXPORT adub erf   ( const badouble& );

    ~adub();
};


/****************************************************************************/
/*                                                            CLASS ADOUBLE */
/*
  The class adouble.
  ---Derived from badouble.  Contains the standard constructors/destructors.
  ---At construction, it is given a new address, and at destruction, that
     address is freed.
*/
class ADOLC_DLL_EXPORT adouble:public badouble {
public:
    adouble( const adub& );
    adouble( const adouble& );
    adouble( void );
    adouble( double );
    /* adub prevents postfix operators to occur on the left
       side of an assignment which would not work  */
    adub operator++( int );
    adub operator--( int );
    badouble& operator++( void );
    badouble& operator--( void );
    /*   inline double value(); */
    ~adouble();

    adouble& operator = ( double );
    adouble& operator = ( const badouble& );
    /* adouble& operator = ( const adouble& );
       !!! olvo 991210 was the same as badouble-assignment */
    adouble& operator = ( const adub& );
};


/****************************************************************************/
/*                                                       INLINE DEFINITIONS */

/*--------------------------------------------------------------------------*/
inline locint badouble::loc( void ) const {
    return location;
}

/*--------------------------------------------------------------------------*/
/* Comparison */
inline int operator != ( const badouble& u, const badouble& v ) {
    return (u-v != 0);
}

inline int operator != ( double coval, const badouble& v) {
    if (coval)
        return (-coval+v != 0);
    else
        return (v != 0);
}

inline int operator == ( const badouble& u, const badouble& v ) {
    return (u-v == 0);
}

inline int operator == ( double coval, const badouble& v) {
    if (coval)
        return (-coval+v == 0);
    else
        return (v == 0);
}

inline int operator <= ( const badouble& u, const badouble& v ) {
    return (u-v <= 0);
}

inline int operator <= ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v >= 0);
    else
        return (v >= 0);
}

inline int operator >= ( const badouble& u, const badouble& v ) {
    return (u-v >= 0);
}

inline int operator >= ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v <= 0);
    else
        return (v <= 0);
}

inline int operator > ( const badouble& u, const badouble& v ) {
    return (u-v > 0);
}

inline int operator > ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v < 0);
    else
        return (v < 0);
}

inline int operator < ( const badouble& u, const badouble& v ) {
    return (u-v < 0);
}

inline int operator < ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v > 0);
    else
        return (v > 0);
}

/*--------------------------------------------------------------------------*/
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

/****************************************************************************/
/* tapeless implementation                                                  */
/****************************************************************************/
#else

#include <limits>

namespace adtl {

#if defined(NUMBER_DIRECTIONS)
extern int ADOLC_numDir;
#define ADOLC_TAPELESS_UNIQUE_INTERNALS int adtl::ADOLC_numDir = NUMBER_DIRECTIONS;
#if !defined(DYNAMIC_DIRECTIONS)
#  define ADVAL                adval[NUMBER_DIRECTIONS]
#else
#  define ADVAL                *adval;
#endif
#  define ADVAL_TYPE           const double *
#  define FOR_I_EQ_0_LT_NUMDIR for (int _i=0; _i < ADOLC_numDir; ++_i)
#  define ADVAL_I              adval[_i]
#  define ADV_I                adv[_i]
#  define V_I                  v[_i]
#else
#  define ADVAL                adval
#  define ADVAL_TYPE           double
#  define FOR_I_EQ_0_LT_NUMDIR
#  define ADVAL_I              adval
#  define ADV_I                adv
#  define V_I                  v
#endif

#if !defined(_ISOC99_SOURCE) && !defined(__USE_ISOC99)
inline double fmin( const double &x, const double &y ) {
    if (x<y) return x;
    else return y;
}

inline double fmax( const double &x, const double &y ) {
    if (x>y) return x;
    else return y;
}
#endif

inline double makeNaN() {
    return ADOLC_MATH_NSP::numeric_limits<double>::quiet_NaN();
}

inline double makeInf() {
    return ADOLC_MATH_NSP::numeric_limits<double>::infinity();
}

class adouble {
public:
    // ctors
    inline adouble();
    inline adouble(const double v);
    inline adouble(const double v, ADVAL_TYPE adv);
    inline adouble(const adouble& a);
#if (defined( DYNAMIC_DIRECTIONS) | defined(ADOLC_TRACELESS_SPARSE_PATTERN) )
    inline ~adouble();
#endif

    /*******************  temporary results  ******************************/
    // sign
    inline adouble operator - () const;
    inline adouble operator + () const;

    // addition
    inline adouble operator + (const double v) const;
    inline adouble operator + (const adouble& a) const;
    inline friend
    adouble operator + (const double v, const adouble& a);

    // substraction
    inline adouble operator - (const double v) const;
    inline adouble operator - (const adouble& a) const;
    inline friend
    adouble operator - (const double v, const adouble& a);

    // multiplication
    inline adouble operator * (const double v) const;
    inline adouble operator * (const adouble& a) const;
    inline friend
    adouble operator * (const double v, const adouble& a);

    // division
    inline adouble operator / (const double v) const;
    inline adouble operator / (const adouble& a) const;
    inline friend
    adouble operator / (const double v, const adouble& a);

    // inc/dec
    inline adouble operator ++ ();
    inline adouble operator ++ (int);
    inline adouble operator -- ();
    inline adouble operator -- (int);

    // functions
    inline friend adouble tan(const adouble &a);
    inline friend adouble exp(const adouble &a);
    inline friend adouble log(const adouble &a);
    inline friend adouble sqrt(const adouble &a);
    inline friend adouble sin(const adouble &a);
    inline friend adouble cos(const adouble &a);
    inline friend adouble asin(const adouble &a);
    inline friend adouble acos(const adouble &a);
    inline friend adouble atan(const adouble &a);

    inline friend adouble atan2(const adouble &a, const adouble &b);
    inline friend adouble pow(const adouble &a, double v);
    inline friend adouble pow(const adouble &a, const adouble &b);
    inline friend adouble pow(double v, const adouble &a);
    inline friend adouble log10(const adouble &a);

    inline friend adouble sinh (const adouble &a);
    inline friend adouble cosh (const adouble &a);
    inline friend adouble tanh (const adouble &a);
#if defined(ATRIG_ERF)
    inline friend adouble asinh (const adouble &a);
    inline friend adouble acosh (const adouble &a);
    inline friend adouble atanh (const adouble &a);
#endif
    inline friend adouble fabs (const adouble &a);
    inline friend adouble ceil (const adouble &a);
    inline friend adouble floor (const adouble &a);
    inline friend adouble fmax (const adouble &a, const adouble &b);
    inline friend adouble fmax (double v, const adouble &a);
    inline friend adouble fmax (const adouble &a, double v);
    inline friend adouble fmin (const adouble &a, const adouble &b);
    inline friend adouble fmin (double v, const adouble &a);
    inline friend adouble fmin (const adouble &a, double v);
    inline friend adouble ldexp (const adouble &a, const adouble &b);
    inline friend adouble ldexp (const adouble &a, const double v);
    inline friend adouble ldexp (const double v, const adouble &a);
    inline friend double frexp (const adouble &a, int* v);
#if defined(ATRIG_ERF)
    inline friend adouble erf (const adouble &a);
#endif


    /*******************  nontemporary results  ***************************/
    // assignment
    inline void operator = (const double v);
    inline void operator = (const adouble& a);

    // addition
    inline void operator += (const double v);
    inline void operator += (const adouble& a);

    // substraction
    inline void operator -= (const double v);
    inline void operator -= (const adouble& a);

    // multiplication
    inline void operator *= (const double v);
    inline void operator *= (const adouble& a);

    // division
    inline void operator /= (const double v);
    inline void operator /= (const adouble& a);

    // not
    inline int operator ! () const;

    // comparision
    inline int operator != (const adouble&) const;
    inline int operator != (const double) const;
    inline friend int operator != (const double, const adouble&);

    inline int operator == (const adouble&) const;
    inline int operator == (const double) const;
    inline friend int operator == (const double, const adouble&);

    inline int operator <= (const adouble&) const;
    inline int operator <= (const double) const;
    inline friend int operator <= (const double, const adouble&);

    inline int operator >= (const adouble&) const;
    inline int operator >= (const double) const;
    inline friend int operator >= (const double, const adouble&);

    inline int operator >  (const adouble&) const;
    inline int operator >  (const double) const;
    inline friend int operator >  (const double, const adouble&);

    inline int operator <  (const adouble&) const;
    inline int operator <  (const double) const;
    inline friend int operator <  (const double, const adouble&);

    /*******************  getter / setter  ********************************/
    inline double getValue() const;
    inline void setValue(const double v);
    inline ADVAL_TYPE getADValue() const;
    inline void setADValue(ADVAL_TYPE v);
#if defined(NUMBER_DIRECTIONS)
    inline double getADValue(const unsigned int p) const;
    inline void setADValue(const unsigned int p, const double v);
#endif
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    inline unsigned int* get_pattern() const;
    inline void add_to_pattern(const unsigned int* v);
    inline void delete_pattern();
#endif

    /*******************  i/o operations  *********************************/
    inline friend ostream& operator << ( ostream&, const adouble& );
    inline friend istream& operator >> ( istream&, adouble& );


private:
    // internal variables
    double val;
    double ADVAL;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    unsigned int *pattern;
#endif
};

/*******************************  ctors  ************************************/
adouble::adouble() {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 22 );
    pattern[1]=20;
    pattern[0]=0;
#endif
}

adouble::adouble(const double v) : val(v) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I = 0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 22 );
    pattern[1]=20;
    pattern[0]=0;
#endif
}

adouble::adouble(const double v, ADVAL_TYPE adv) : val(v) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=ADV_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 22 );
    pattern[1]=20;
    pattern[0]=0;
#endif
}

adouble::adouble(const adouble& a) : val(a.val) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 22 );
    pattern[1]=20;
    pattern[0]=0;
    add_to_pattern(a.get_pattern());
#endif
}

/*******************************  dtors  ************************************/
#if ( defined(DYNAMIC_DIRECTIONS) | defined(ADOLC_TRACELESS_SPARSE_PATTERN) )
adouble::~adouble() {
#if defined(DYNAMIC_DIRECTIONS)
     delete[] adval;
#endif
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     free(pattern);
#endif
}
#endif
/*************************  temporary results  ******************************/
// sign
adouble adouble::operator - () const {
    adouble tmp;
    tmp.val=-val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=-ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     tmp.add_to_pattern( get_pattern() ) ;
#endif
    return tmp;
}

adouble adouble::operator + () const {
    return *this;
}

// addition
adouble adouble::operator + (const double v) const {
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    adouble tmp;
    tmp.val=val+v;
    FOR_I_EQ_0_LT_NUMDIR
      tmp.ADVAL_I=ADVAL_I;
    tmp.add_to_pattern( get_pattern() ) ;
    return tmp;
#else
     return adouble(val+v, adval);
#endif
}

adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
    tmp.val=val+a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I+a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     tmp.add_to_pattern( get_pattern()  ) ;
     tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble operator + (const double v, const adouble& a) {
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    adouble tmp;
    tmp.val=a.val+v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I;
    tmp.add_to_pattern( a.get_pattern() ) ;
    return tmp;
#else
    return adouble(v+a.val, a.adval);
#endif
}

// subtraction
adouble adouble::operator - (const double v) const {
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    adouble tmp;
    tmp.val=val-v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I;
    tmp.add_to_pattern( get_pattern() ) ;
    return tmp;
#else
    return adouble(val-v, adval);
#endif
}

adouble adouble::operator - (const adouble& a) const {
    adouble tmp;
    tmp.val=val-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I-a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     tmp.add_to_pattern( get_pattern() ) ;
     tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble operator - (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=-a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

// multiplication
adouble adouble::operator * (const double v) const {
    adouble tmp;
    tmp.val=val*v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I*v;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     tmp.add_to_pattern( get_pattern() ) ;
#endif
    return tmp;
}

adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
    tmp.val=val*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     tmp.add_to_pattern(   get_pattern() ) ;
     tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble operator * (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=v*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

// division
adouble adouble::operator / (const double v) const {
    adouble tmp;
    tmp.val=val/v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I/v;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( get_pattern() ) ;
#endif
    return tmp;
}

adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
    tmp.val=val/a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern(   get_pattern() ) ;
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble operator / (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v/a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=(-v*a.ADVAL_I)/(a.val*a.val);
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

// inc/dec
adouble adouble::operator ++ () {
    ++val;
    return *this;
}

adouble adouble::operator ++ (int) {
    adouble tmp;
    tmp.val=val++;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( get_pattern() ) ;
#endif
    return tmp;
}

adouble adouble::operator -- () {
    --val;
    return *this;
}

adouble adouble::operator -- (int) {
    adouble tmp;
    tmp.val=val--;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( get_pattern() ) ;
#endif
    return tmp;
}

// functions
adouble tan(const adouble& a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::tan(a.val);
    tmp2=ADOLC_MATH_NSP::cos(a.val);
    tmp2*=tmp2;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble exp(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::exp(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp.val*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble log(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::log(a.val);
    FOR_I_EQ_0_LT_NUMDIR
	if (a.val>0) tmp.ADVAL_I=a.ADVAL_I/a.val;
	else if (a.val==0 && a.ADVAL_I != 0.0) {
	    int sign = (a.ADVAL_I < 0)  ? -1 : 1;
	    tmp.ADVAL_I=sign*makeInf();
	} else tmp.ADVAL_I=makeNaN();
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble sqrt(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::sqrt(a.val);
    FOR_I_EQ_0_LT_NUMDIR
	if (a.val>0) tmp.ADVAL_I=a.ADVAL_I/(tmp.val*2);
        else if (a.val==0.0 && a.ADVAL_I != 0.0) {
	    int sign = (a.ADVAL_I < 0) ? -1 : 1;
	    tmp.ADVAL_I=sign * makeInf();
	} else tmp.ADVAL_I=makeNaN();
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble sin(const adouble &a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::sin(a.val);
    tmp2=ADOLC_MATH_NSP::cos(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble cos(const adouble &a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::cos(a.val);
    tmp2=-ADOLC_MATH_NSP::sin(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble asin(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::asin(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble acos(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::acos(a.val);
    double tmp2=-ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble atan(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::atan(a.val);
    double tmp2=1+a.val*a.val;
    tmp2=1/tmp2;
    if (tmp2!=0)
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=a.ADVAL_I*tmp2;
    else
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble atan2(const adouble &a, const adouble &b) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::atan2(a.val, b.val);
    double tmp2=a.val*a.val;
    double tmp3=b.val*b.val;
    double tmp4=tmp3/(tmp2+tmp3);
    if (tmp4!=0)
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=(a.ADVAL_I*b.val-a.val*b.ADVAL_I)/tmp3*tmp4;
    else
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
    tmp.add_to_pattern( b.get_pattern() ) ;
#endif
    return tmp;
}

adouble pow(const adouble &a, double v) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(a.val, v);
    double tmp2=v*ADOLC_MATH_NSP::pow(a.val, v-1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(a.val, b.val);
    double tmp2=b.val*ADOLC_MATH_NSP::pow(a.val, b.val-1);
    double tmp3=ADOLC_MATH_NSP::log(a.val)*tmp.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I+tmp3*b.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
    tmp.add_to_pattern( b.get_pattern() ) ;
#endif
    return tmp;
}

adouble pow(double v, const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(v, a.val);
    double tmp2=tmp.val*ADOLC_MATH_NSP::log(v);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble log10(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::log10(a.val);
    double tmp2=ADOLC_MATH_NSP::log((double)10)*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble sinh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::sinh(a.val);
    double tmp2=ADOLC_MATH_NSP::cosh(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I*tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble cosh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::cosh(a.val);
    double tmp2=ADOLC_MATH_NSP::sinh(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I*tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble tanh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::tanh(a.val);
    double tmp2=ADOLC_MATH_NSP::cosh(a.val);
    tmp2*=tmp2;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

#if defined(ATRIG_ERF)
adouble asinh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::asinh(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val+1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble acosh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::acosh(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val-1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble atanh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::atanh(a.val);
    double tmp2=1-a.val*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}
#endif

adouble fabs (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::fabs(a.val);
    int as=0;
    if (a.val>0) as=1;
    if (a.val<0) as=-1;
    if (as!=0)
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=a.ADVAL_I*as;
    else
        FOR_I_EQ_0_LT_NUMDIR {
            as=0;
            if (a.ADVAL_I>0) as=1;
            if (a.ADVAL_I<0) as=-1;
                tmp.ADVAL_I=a.ADVAL_I*as;
            }
 #if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
           return tmp;
}

adouble ceil (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::ceil(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble floor (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::floor(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}

adouble fmax (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.val-b.val;
    if (tmp2<0) {
        tmp.val=b.val;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=b.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( b.get_pattern() ) ;
#endif
    } else {
        tmp.val=a.val;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
            tmp.add_to_pattern( a.get_pattern() ) ;
#endif
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=b.ADVAL_I;
                else tmp.ADVAL_I=a.ADVAL_I;
                }
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
    tmp.add_to_pattern( b.get_pattern() ) ;
#endif
            }
}
return tmp;
}

adouble fmax (double v, const adouble &a) {
    adouble tmp;
    double tmp2=v-a.val;
    if (tmp2<0) {
        tmp.val=a.val;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    } else {
        tmp.val=v;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=0.0;
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
                }
            }
}
return tmp;
}

adouble fmax (const adouble &a, double v) {
    adouble tmp;
    double tmp2=a.val-v;
    if (tmp2<0) {
        tmp.val=v;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=0.0;
    } else {
        tmp.val=a.val;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=0.0;
                }
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
            }
}
return tmp;
}

adouble fmin (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.val-b.val;
    if (tmp2<0) {
        tmp.val=a.val;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    } else {
        tmp.val=b.val;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=b.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( b.get_pattern() ) ;
#endif
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=b.ADVAL_I;
                }
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
    tmp.add_to_pattern( b.get_pattern() ) ;
#endif
            }
}
return tmp;
}

adouble fmin (double v, const adouble &a) {
    adouble tmp;
    double tmp2=v-a.val;
    if (tmp2<0) {
        tmp.val=v;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=0.0;
    } else {
        tmp.val=a.val;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=0.0;
                }
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
            }
}
return tmp;
}

adouble fmin (const adouble &a, double v) {
    adouble tmp;
    double tmp2=a.val-v;
    if (tmp2<0) {
        tmp.val=a.val;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    } else {
        tmp.val=v;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=0.0;
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=0.0;
                }
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
            }
}
return tmp;
}

adouble ldexp (const adouble &a, const adouble &b) {
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    adouble tmp = a*pow(2.,b);
    tmp.add_to_pattern( a.get_pattern() ) ;
    tmp.add_to_pattern( b.get_pattern() ) ;
    return tmp;
#else
    return a*pow(2.,b);
#endif
}

adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

adouble ldexp (const double v, const adouble &a) {
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    adouble tmp = v*pow(2.,a);
    tmp.add_to_pattern( a.get_pattern() ) ;
    return tmp;
#else
    return v*pow(2.,a);
#endif
}

double frexp (const adouble &a, int* v) {
    return ADOLC_MATH_NSP::frexp(a.val, v);
}

#if defined(ATRIG_ERF)
adouble erf (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::erf(a.val);
    double tmp2 = 2.0 /
        ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) *
        ADOLC_MATH_NSP_ERF::exp(-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    tmp.add_to_pattern( a.get_pattern() ) ;
#endif
    return tmp;
}
#endif


/*******************  nontemporary results  *********************************/
void adouble::operator = (const double v) {
    val=v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=0.0;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    pattern[0]=0;
#endif
}

void adouble::operator = (const adouble& a) {
    val=a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
     pattern[0] = 0;
     add_to_pattern( a.get_pattern() );
#endif
}

void adouble::operator += (const double v) {
    val+=v;
}

void adouble::operator += (const adouble& a) {
    val=val+a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I+=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    add_to_pattern( a.get_pattern() ) ;
#endif
}

void adouble::operator -= (const double v) {
    val-=v;
}

void adouble::operator -= (const adouble& a) {
    val=val-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I-=a.ADVAL_I;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    add_to_pattern( a.get_pattern() ) ;
#endif
}

void adouble::operator *= (const double v) {
    val=val*v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I*=v;
}

void adouble::operator *= (const adouble& a) {
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    val*=a.val;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    add_to_pattern( a.get_pattern() ) ;
#endif
}

void adouble::operator /= (const double v) {
    val/=v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I/=v;
}

void adouble::operator /= (const adouble& a) {
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    val=val/a.val;
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
    add_to_pattern( a.get_pattern() ) ;
#endif
}

// not
int adouble::operator ! () const {
    return val==0.0;
}

// comparision
int adouble::operator != (const adouble &a) const {
    return val!=a.val;
}

int adouble::operator != (const double v) const {
    return val!=v;
}

int operator != (const double v, const adouble &a) {
    return v!=a.val;
}

int adouble::operator == (const adouble &a) const {
    return val==a.val;
}

int adouble::operator == (const double v) const {
    return val==v;
}

int operator == (const double v, const adouble &a) {
    return v==a.val;
}

int adouble::operator <= (const adouble &a) const {
    return val<=a.val;
}

int adouble::operator <= (const double v) const {
    return val<=v;
}

int operator <= (const double v, const adouble &a) {
    return v<=a.val;
}

int adouble::operator >= (const adouble &a) const {
    return val>=a.val;
}

int adouble::operator >= (const double v) const {
    return val>=v;
}

int operator >= (const double v, const adouble &a) {
    return v>=a.val;
}

int adouble::operator >  (const adouble &a) const {
    return val>a.val;
}

int adouble::operator >  (const double v) const {
    return val>v;
}

int operator >  (const double v, const adouble &a) {
    return v>a.val;
}

int adouble::operator <  (const adouble &a) const {
    return val<a.val;
}

int adouble::operator <  (const double v) const {
    return val<v;
}

int operator <  (const double v, const adouble &a) {
    return v<a.val;
}

/*******************  getter / setter  **************************************/
double adouble::getValue() const {
    return val;
}

void adouble::setValue(const double v) {
    val=v;
}

ADVAL_TYPE adouble::getADValue() const {
    return adval;
}

void adouble::setADValue(ADVAL_TYPE v) {
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=V_I;
}

#  if defined(NUMBER_DIRECTIONS)
double adouble::getADValue(const unsigned int p) const {
    if (p>=NUMBER_DIRECTIONS) {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"getADValue(...)\"!!!\n");
        exit(-1);
    }
    return adval[p];
}

void adouble::setADValue(const unsigned int p, const double v) {
    if (p>=NUMBER_DIRECTIONS) {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"setADValue(...)\"!!!\n");
        exit(-1);
    }
    adval[p]=v;
}
#  endif

#if defined(NUMBER_DIRECTIONS)
static void setNumDir(const unsigned int p) {
#if !defined(DYNAMIC_DIRECTIONS)
    if (p>NUMBER_DIRECTIONS) ADOLC_numDir=NUMBER_DIRECTIONS;
    else ADOLC_numDir=p;
#else
    ADOLC_numDir = p;
#endif
}
#endif

/*******************  i/o operations  ***************************************/
ostream& operator << ( ostream& out, const adouble& a) {
    out << "Value: " << a.val;
#if !defined(NUMBER_DIRECTIONS)
    out << " ADValue: ";
#else
    out << " ADValues (" << ADOLC_numDir << "): ";
#endif
    FOR_I_EQ_0_LT_NUMDIR
    out << a.ADVAL_I << " ";
    out << "(a)";
    return out;
}

istream& operator >> ( istream& in, adouble& a) {
    char c;
    do {
        in >> c;
    } while (c!=':' && !in.eof());
    in >> a.val;
#if !defined(NUMBER_DIRECTIONS)
    do in >> c;
    while (c!=':' && !in.eof());
#else
unsigned int num;
do in >> c;
while (c!='(' && !in.eof());
in >> num;
if (num>NUMBER_DIRECTIONS) {
    cout << "ADOL-C error: to many directions in input\n";
    exit(-1);
}
do in >> c;
while (c!=')' && !in.eof());
#endif
    FOR_I_EQ_0_LT_NUMDIR
    in >> a.ADVAL_I;
    do in >> c;
    while (c!=')' && !in.eof());
    return in;
}

/**************** ADOLC_TRACELESS_SPARSE_PATTERN ****************************/
#if defined(ADOLC_TRACELESS_SPARSE_PATTERN)
unsigned int* adouble::get_pattern() const {
    return pattern;
}

void adouble::add_to_pattern(const unsigned int* v) {
  if( pattern != v ){
    int num,num1,num2, i,j,k,l;
    unsigned int *temp_array;
    if (pattern == NULL){
        if ( v[0] < 11 ){
            pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 22 );
            pattern[1]=20;
        } else{
            pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 2*(v[0]+1));
            pattern[1]=2*v[0];
        }
        for(i=0; i< v[0] ; i++)
            pattern[i+2] = v[i+2];
        pattern[0] = v[0];
    } else {
    if (pattern[0] == 0){ // Copy_Index_Domain
        if (pattern[1] < v[0]){
            free(pattern);
            pattern = (unsigned int*)  malloc(sizeof(unsigned int) * 2*(v[0]+1));
            pattern[1] = 2*v[0];
        }
        for(i=0; i< v[0] ; i++)
            pattern[i+2] = v[i+2];
        pattern[0] = v[0];
    }
    else
    {
       num  = pattern[0];
       num1 = v[0];
       num2 = pattern[1];
       if (num2 < num1+num)
           num2 = num1+num;
       temp_array = (unsigned int*)  malloc(sizeof(unsigned int)* (num2+2));
       temp_array[1] = num2;
       i = 2;
       j = 2;
       k = 2;
       num += 2;
       num1 += 2;
       while ((i< num) && (j < num1)){
           if (pattern[i] < v[j]) {
               temp_array[k] = pattern[i];
               i++; k++;
           } else {
               if (pattern[i] == v[j]) {
                   temp_array[k] = v[j];
                   i++;j++;k++;
               } else {
                   temp_array[k] = v[j];
                   j++;k++;
               }
           }
       }
       for(l = i;l<num;l++) {
          temp_array[k] = pattern[l];
          k++;
       }
       for(l = j;l<num1;l++) {
           temp_array[k] = v[l];
           k++;
       }
       temp_array[0] = k-2;
       free(pattern);
       pattern=temp_array;
    }
}
}
}

void adouble::delete_pattern(){
     free(pattern);
}

void ADOLC_Init_sparse_pattern(adouble *a, int n ){
     for(unsigned int i=0; i < n; i++){
         unsigned int *v = new unsigned int[4];
         v[0] = 1;
         v[1] = 2;
         v[2] = i;
         v[3] = 0;
         a[i].add_to_pattern(v);
         delete[] v;
     }
}

unsigned int** ADOLC_get_sparse_pattern(const adouble *b,const int m ){
     unsigned int ** patt = (unsigned int**) malloc(m*sizeof(unsigned int*));
     for( int i=0; i < m ; i++){
         unsigned int *tmp = b[i].get_pattern();
         if (tmp[0] != 0) {
            patt[i] = (unsigned int*) malloc(sizeof(unsigned int) * (tmp[0]+1));
            patt[i][0] = tmp[0];
            for(int l=1;l<=tmp[0];l++)
                patt[i][l] = tmp[l+1];
         } else {
            patt[i] = (unsigned int*) malloc(sizeof(unsigned int));
            patt[i][0] =0;
         }
       free(tmp);
     }
     return patt;
}

#endif /* ADOLC_TRACELESS_SPARSE_PATTERN */

} // END adtl namespace
/****************************************************************************/
#endif /* ADOLC_TAPELESS */

/****************************************************************************/
/*                                                                THAT'S ALL*/
#endif /* __cplusplus */
#endif /* ADOLC_ADOUBLE_H */
