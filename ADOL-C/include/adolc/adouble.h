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
#include <cmath>
#include <stdexcept>

using std::cout;
using std::cin;
using std::cerr;
using std::ostream;
using std::istream;
using std::logic_error;

#include <adolc/common.h>

/* NOTICE: There are automatic includes at the end of this file! */

/****************************************************************************/
/*                                             FORWARD DECLARATIONS (TAPES) */

/*--------------------------------------------------------------------------*/
class adouble;
class adub;
class badouble;

/*--------------------------------------------------------------------------*/
void ADOLC_DLL_EXPORT condassign( double &res, const double &cond,
                                  const double &arg1, const double &arg2 );
void ADOLC_DLL_EXPORT condassign( double &res, const double &cond,
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
protected:
    locint location;
    badouble( void ) {};
    // must be public when using gcc >= 3.4 ( problems with value() )
    // (see GCC 3.4 Release Series - Changes, New Features, and Fixes)
    //
    // badouble( const badouble& a ) {location = a.location;};
    explicit badouble( locint lo ) {
        location = lo;
        isInit = true;
    };

    bool isInit;  // marker if the badouble is properly initialized

public:
    /*--------------------------------------------------------------------------*/
    badouble( const badouble& a ) {};           /* ctor */

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
#if defined(ADOLC_ADVANCED_BRANCHING)
    friend ADOLC_DLL_EXPORT adub operator != ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator == ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator <= ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator >= ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator >  ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator <  ( const badouble&, const badouble& );
#else
    inline friend int operator != ( const badouble&, const badouble& );
    inline friend int operator == ( const badouble&, const badouble& );
    inline friend int operator <= ( const badouble&, const badouble& );
    inline friend int operator >= ( const badouble&, const badouble& );
    inline friend int operator >  ( const badouble&, const badouble& );
    inline friend int operator <  ( const badouble&, const badouble& );
#endif
    inline friend int operator != ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator != ( const badouble&, double );
    inline friend int operator == ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator == ( const badouble&, double );
    inline friend int operator <= ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator <= ( const badouble&, double );
    inline friend int operator >= ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator >= ( const badouble&, double );
    inline friend int operator >  ( double, const badouble& );
    friend ADOLC_DLL_EXPORT int operator >  ( const badouble&, double );
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
    friend ADOLC_DLL_EXPORT void condassign( adouble &res, const badouble &cond,
            const badouble &arg1, const badouble &arg2 );
    friend ADOLC_DLL_EXPORT void condassign( adouble &res, const badouble &cond,
            const badouble &arg );
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
    friend ADOLC_DLL_EXPORT class advector;
    friend ADOLC_DLL_EXPORT class adubref;
    friend ADOLC_DLL_EXPORT adub* adubp_from_adub(const adub&);
    adub( adub const &) {
	isInit = false;
        fprintf(DIAG_OUT,"ADOL-C error: illegal copy construction of adub"
		" variable\n          ... adub objects must never be copied\n");
        throw logic_error("illegal constructor call, errorcode=-2");
    }
    adub( void ) {
	isInit = false;
        fprintf(DIAG_OUT,"ADOL-C error: illegal default construction of adub"
                " variable\n");
        throw logic_error("illegal constructor call, errorcode=-2");
    }
    explicit adub( double ) {
	isInit = false;
        fprintf(DIAG_OUT,"ADOL-C error: illegal  construction of adub variable"
                " from double\n");
        throw logic_error("illegal constructor call, errorcode=-2");
    }
protected:
   /* this is the only logically legal constructor, which can be called by 
    * friend classes and functions 
    */
   adub( locint lo ) : badouble(lo) {} 

public:

    /*--------------------------------------------------------------------------*/
    /* Comparison (friends) */
#if defined(ADOLC_ADVANCED_BRANCHING)
    friend ADOLC_DLL_EXPORT adub operator != ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator == ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator <= ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator >= ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator < ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator > ( const badouble&, const badouble& );
#endif
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
    friend ADOLC_DLL_EXPORT class advector;
protected:
    void initInternal(void); // Init for late initialization
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
    adouble& operator = ( const adouble& );
    adouble& operator = ( const adub& );
    
    inline locint loc(void) const;
};


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
inline int operator != ( const badouble& u, const badouble& v ) {
    return (u-v != 0);
}

inline int operator == ( const badouble& u, const badouble& v ) {
    return (u-v == 0);
}

inline int operator <= ( const badouble& u, const badouble& v ) {
    return (u-v <= 0);
}

inline int operator >= ( const badouble& u, const badouble& v ) {
    return (u-v >= 0);
}

inline int operator > ( const badouble& u, const badouble& v ) {
    return (u-v > 0);
}

inline int operator < ( const badouble& u, const badouble& v ) {
    return (u-v < 0);
}
#endif

inline int operator != ( double coval, const badouble& v) {
    if (coval)
        return (-coval+v != 0);
    else
        return (v != 0);
}

inline int operator == ( double coval, const badouble& v) {
    if (coval)
        return (-coval+v == 0);
    else
        return (v == 0);
}

inline int operator <= ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v >= 0);
    else
        return (v >= 0);
}

inline int operator >= ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v <= 0);
    else
        return (v <= 0);
}

inline int operator > ( double coval, const badouble& v ) {
    if (coval)
        return (-coval+v < 0);
    else
        return (v < 0);
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
/*                                                                THAT'S ALL*/
#endif /* __cplusplus */
#endif /* ADOLC_ADOUBLE_H */
