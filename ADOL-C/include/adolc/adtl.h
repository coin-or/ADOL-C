/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adtl.h contains that declaratins of procedures used to
           define various tapeless adouble operations.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert, Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_ADTL_H
#define ADOLC_ADTL_H

#include <ostream>
#include <adolc/internal/common.h>
#include <list>
#include <stdexcept>

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define COMPILER_HAS_CXX11
#else
#error "please use -std=c++11 compiler flag with a C++11 compliant compiler"
#endif

#if USE_BOOST_POOL
#include <boost/pool/pool_alloc.hpp>
#endif

using std::ostream;
using std::istream;
using std::list;
using std::logic_error;

namespace adtl {

double makeNaN();
double makeInf();

#ifdef USE_ADTL_REFCOUNTING
class adouble;

class refcounter {
private:
    ADOLC_DLL_EXPIMP static size_t refcnt;
    ADOLC_DLL_EXPORT friend void setNumDir(const size_t p);
    friend class adouble;
public:
    refcounter() { ++refcnt; }
    ~refcounter() { --refcnt; }
    inline static size_t getNumLiveVar() {return refcnt;}
};
#endif


//class func_ad {
//public:
//    virtual int operator() (int n, adouble *x, int m, adouble *y) = 0;
//};

class adouble {
public:
    inline adouble();
    inline adouble(const double v);
    inline adouble(const double v, const double* adv);
    inline adouble(const adouble& a);
    inline ~adouble();

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

    inline friend void condassign( adouble &res, const adouble &cond,
            const adouble &arg1, const adouble &arg2 );
    inline friend void condassign( adouble &res, const adouble &cond,
            const adouble &arg );
    inline friend void condeqassign( adouble &res, const adouble &cond,
            const adouble &arg1, const adouble &arg2 );
    inline friend void condeqassign( adouble &res, const adouble &cond,
            const adouble &arg );

    /*******************  nontemporary results  ***************************/
    // assignment
    inline adouble& operator = (const double v);
    inline adouble& operator = (const adouble& a);

    // addition
    inline adouble& operator += (const double v);
    inline adouble& operator += (const adouble& a);

    // substraction
    inline adouble& operator -= (const double v);
    inline adouble& operator -= (const adouble& a);

    // multiplication
    inline adouble& operator *= (const double v);
    inline adouble& operator *= (const adouble& a);

    // division
    inline adouble& operator /= (const double v);
    inline adouble& operator /= (const adouble& a);

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
    inline const double* const getADValue() const;
    inline void setADValue(const double* v);

    inline double getADValue(const unsigned int p) const;
    inline void setADValue(const unsigned int p, const double v);
    inline explicit operator double const&() const;
    inline explicit operator double&&();
    inline explicit operator double();

public:
    /*******************  i/o operations  *********************************/
    ADOLC_DLL_EXPORT friend ostream& operator << ( ostream&, const adouble& );
    ADOLC_DLL_EXPORT friend istream& operator >> ( istream&, adouble& );

private:
#if USE_BOOST_POOL
    static boost::pool<boost::default_user_allocator_new_delete>* advalpool;
#endif
    double *adval;
#ifdef USE_ADTL_REFCOUNTING
    refcounter __rcnt;
#endif
    ADOLC_DLL_EXPIMP static size_t numDir;
    inline friend void setNumDir(const size_t p);
    inline friend size_t getNumDir();
};

}

#include <cmath>
#include <iostream>
#include <limits>

namespace adtl {

inline void setNumDir(const size_t p) {
#ifdef USE_ADTL_REFCOUNTING
  if (refcounter::refcnt > 0) {
    fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting numDir will not change the number of\n directional derivative in existing adoubles and may lead to erronious results\n or memory corruption\n Number of currently existing adoubles = %zu\n", refcounter::refcnt);
  }
#else
  fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting numDir could change memory allocation of\n derivatives in existing adoubles and may lead to erronious results\n or memory corruption\n");
#endif

    if (p < 1) {
	fprintf(DIAG_OUT, "ADOL-C Error: Tapeless: You are being a moron now.\n");
	abort();
    }
    adouble::numDir = p;
#if USE_BOOST_POOL
    if (adouble::advalpool != NULL) {
        delete adouble::advalpool;
        adouble::advalpool = NULL;
    }
    adouble::advalpool = new boost::pool<boost::default_user_allocator_new_delete>((adouble::numDir+1)*sizeof(double));
#endif
}

inline size_t getNumDir() {return adouble::numDir;}

inline double makeNaN() {
    return ADOLC_MATH_NSP::numeric_limits<double>::quiet_NaN();
}

inline double makeInf() {
    return ADOLC_MATH_NSP::numeric_limits<double>::infinity();
}

#define FOR_I_EQ_0_LTEQ_NUMDIR  for (size_t _i=0; _i <= adouble::numDir; ++_i)
#define FOR_I_EQ_1_LTEQ_NUMDIR  for (size_t _i=1; _i <= adouble::numDir; ++_i)
#define ADVAL_I                 adval[_i]
#define PRIMAL_VALUE            adval[0]

/*******************************  ctors  ************************************/
inline adouble::adouble() : adval(NULL) {
#if USE_BOOST_POOL
    adval = reinterpret_cast<double*>(advalpool->malloc());
#else
    adval = new double[adouble::numDir+1];
#endif
    PRIMAL_VALUE = 0.;
}

inline adouble::adouble(const double v) : adval(NULL) {
#if USE_BOOST_POOL
    adval = reinterpret_cast<double*>(advalpool->malloc());
#else
	adval = new double[adouble::numDir+1];
#endif
    PRIMAL_VALUE = v;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    ADVAL_I = 0.0;
}

inline adouble::adouble(const double v, const double* adv) :  adval(NULL) {
#if USE_BOOST_POOL
    adval = reinterpret_cast<double*>(advalpool->malloc());
#else
	adval = new double[adouble::numDir+1];
#endif
    PRIMAL_VALUE = v;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    ADVAL_I=adv[_i-1];
}

inline adouble::adouble(const adouble& a) : adval(NULL) {
#if USE_BOOST_POOL
    adval = reinterpret_cast<double*>(advalpool->malloc());
#else
	adval = new double[adouble::numDir+1];
#endif
    FOR_I_EQ_0_LTEQ_NUMDIR
        ADVAL_I=a.ADVAL_I;
}

/*******************************  dtors  ************************************/
inline adouble::~adouble() {
    if (adval != NULL)
#if USE_BOOST_POOL
        advalpool->free(adval);
#else
	delete[] adval;
#endif
}

/*************************  temporary results  ******************************/
// sign
inline adouble adouble::operator - () const {
    adouble tmp;
	FOR_I_EQ_0_LTEQ_NUMDIR
	    tmp.ADVAL_I=-ADVAL_I;
    return tmp;
}

inline adouble adouble::operator + () const {
    return *this;
}

// addition
inline adouble adouble::operator + (const double v) const {
    adouble tmp(PRIMAL_VALUE+v, adval+1);
    return tmp;
}

inline adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
	FOR_I_EQ_0_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I+a.ADVAL_I;
    return tmp;
}

inline adouble operator + (const double v, const adouble& a) {
    adouble tmp(v+a.PRIMAL_VALUE, a.adval+1);
    return tmp;
}

// subtraction
inline adouble adouble::operator - (const double v) const {
    adouble tmp(PRIMAL_VALUE-v, adval+1);
    return tmp;
}

inline adouble adouble::operator - (const adouble& a) const {
    adouble tmp;
	FOR_I_EQ_0_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I-a.ADVAL_I;
    return tmp;
}

inline adouble operator - (const double v, const adouble& a) {
    adouble tmp;
    tmp.PRIMAL_VALUE=v-a.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=-a.ADVAL_I;
    return tmp;
}

// multiplication
inline adouble adouble::operator * (const double v) const {
    adouble tmp;
	FOR_I_EQ_0_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I*v;
    return tmp;
}

inline adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
	tmp.PRIMAL_VALUE=PRIMAL_VALUE*a.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I*a.PRIMAL_VALUE+PRIMAL_VALUE*a.ADVAL_I;
    return tmp;
}

inline adouble operator * (const double v, const adouble& a) {
    adouble tmp;
	FOR_I_EQ_0_LTEQ_NUMDIR
	    tmp.ADVAL_I=v*a.ADVAL_I;
    return tmp;
}

// division
inline adouble adouble::operator / (const double v) const {
    adouble tmp;
	FOR_I_EQ_0_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I/v;
    return tmp;
}

inline adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
	tmp.PRIMAL_VALUE=PRIMAL_VALUE/a.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=(ADVAL_I*a.PRIMAL_VALUE-PRIMAL_VALUE*a.ADVAL_I)/(a.PRIMAL_VALUE*a.PRIMAL_VALUE);
    return tmp;
}

inline adouble operator / (const double v, const adouble& a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=v/a.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=(-v*a.ADVAL_I)/(a.PRIMAL_VALUE*a.PRIMAL_VALUE);
    return tmp;
}

// inc/dec
inline adouble adouble::operator ++ () {
	++PRIMAL_VALUE;
    return *this;
}

inline adouble adouble::operator ++ (int) {
    adouble tmp;
	tmp.PRIMAL_VALUE=PRIMAL_VALUE++;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I;
    return tmp;
}

inline adouble adouble::operator -- () {
	--PRIMAL_VALUE;
    return *this;
}

inline adouble adouble::operator -- (int) {
    adouble tmp;
	tmp.PRIMAL_VALUE=PRIMAL_VALUE--;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=ADVAL_I;
    return tmp;
}

// functions
inline adouble tan(const adouble& a) {
    adouble tmp;
    double tmp2;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::tan(a.PRIMAL_VALUE);
	tmp2=ADOLC_MATH_NSP::cos(a.PRIMAL_VALUE);
	tmp2*=tmp2;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

inline adouble exp(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::exp(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp.PRIMAL_VALUE*a.ADVAL_I;
    return tmp;
}

inline adouble log(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::log(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    if (a.PRIMAL_VALUE>0) tmp.ADVAL_I=a.ADVAL_I/a.PRIMAL_VALUE;
	    else if (a.PRIMAL_VALUE==0 && a.ADVAL_I != 0.0) {
		int sign = (a.ADVAL_I < 0)  ? -1 : 1;
		tmp.ADVAL_I=sign*makeInf();
	    } else tmp.ADVAL_I=makeNaN();
    return tmp;
}

inline adouble sqrt(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::sqrt(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    if (a.PRIMAL_VALUE>0) tmp.ADVAL_I=a.ADVAL_I/(tmp.PRIMAL_VALUE*2);
	    else if (a.PRIMAL_VALUE==0.0 && a.ADVAL_I != 0.0) {
		int sign = (a.ADVAL_I < 0) ? -1 : 1;
		tmp.ADVAL_I=sign * makeInf();
	    } else tmp.ADVAL_I=makeNaN();
    return tmp;
}

inline adouble sin(const adouble &a) {
    adouble tmp;
    double tmp2;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::sin(a.PRIMAL_VALUE);
	tmp2=ADOLC_MATH_NSP::cos(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

inline adouble cos(const adouble &a) {
    adouble tmp;
    double tmp2;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::cos(a.PRIMAL_VALUE);
	tmp2=-ADOLC_MATH_NSP::sin(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

inline adouble asin(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::asin(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::sqrt(1-a.PRIMAL_VALUE*a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

inline adouble acos(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::acos(a.PRIMAL_VALUE);
	double tmp2=-ADOLC_MATH_NSP::sqrt(1-a.PRIMAL_VALUE*a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

inline adouble atan(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::atan(a.PRIMAL_VALUE);
	double tmp2=1+a.PRIMAL_VALUE*a.PRIMAL_VALUE;
	tmp2=1/tmp2;
	if (tmp2!=0)
	    FOR_I_EQ_1_LTEQ_NUMDIR
		tmp.ADVAL_I=a.ADVAL_I*tmp2;
	else
	    FOR_I_EQ_1_LTEQ_NUMDIR
		tmp.ADVAL_I=0.0;
    return tmp;
}

inline adouble atan2(const adouble &a, const adouble &b) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::atan2(a.PRIMAL_VALUE, b.PRIMAL_VALUE);
	double tmp2=a.PRIMAL_VALUE*a.PRIMAL_VALUE;
	double tmp3=b.PRIMAL_VALUE*b.PRIMAL_VALUE;
	double tmp4=tmp3/(tmp2+tmp3);
	if (tmp4!=0)
	    FOR_I_EQ_1_LTEQ_NUMDIR
		tmp.ADVAL_I=(a.ADVAL_I*b.PRIMAL_VALUE-a.PRIMAL_VALUE*b.ADVAL_I)/tmp3*tmp4;
	else
	    FOR_I_EQ_1_LTEQ_NUMDIR
		tmp.ADVAL_I=0.0;
    return tmp;
}

inline adouble pow(const adouble &a, double v) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::pow(a.PRIMAL_VALUE, v);
	double tmp2=v*ADOLC_MATH_NSP::pow(a.PRIMAL_VALUE, v-1);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

inline adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::pow(a.PRIMAL_VALUE, b.PRIMAL_VALUE);
	double tmp2=b.PRIMAL_VALUE*ADOLC_MATH_NSP::pow(a.PRIMAL_VALUE, b.PRIMAL_VALUE-1);
	double tmp3=ADOLC_MATH_NSP::log(a.PRIMAL_VALUE)*tmp.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I+tmp3*b.ADVAL_I;
    return tmp;
}

inline adouble pow(double v, const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::pow(v, a.PRIMAL_VALUE);
	double tmp2=tmp.PRIMAL_VALUE*ADOLC_MATH_NSP::log(v);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

inline adouble log10(const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::log10(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::log((double)10)*a.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

inline adouble sinh (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::sinh(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::cosh(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    return tmp;
}

inline adouble cosh (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::cosh(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::sinh(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    return tmp;
}

inline adouble tanh (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::tanh(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::cosh(a.PRIMAL_VALUE);
	tmp2*=tmp2;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

#if defined(ATRIG_ERF)
inline adouble asinh (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP_ERF::asinh(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::sqrt(a.PRIMAL_VALUE*a.PRIMAL_VALUE+1);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

inline adouble acosh (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP_ERF::acosh(a.PRIMAL_VALUE);
	double tmp2=ADOLC_MATH_NSP::sqrt(a.PRIMAL_VALUE*a.PRIMAL_VALUE-1);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

inline adouble atanh (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP_ERF::atanh(a.PRIMAL_VALUE);
	double tmp2=1-a.PRIMAL_VALUE*a.PRIMAL_VALUE;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}
#endif

inline adouble fabs (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::fabs(a.PRIMAL_VALUE);
	int as=0;
	if (a.PRIMAL_VALUE>0) as=1;
	if (a.PRIMAL_VALUE<0) as=-1;
	if (as!=0)
	    FOR_I_EQ_1_LTEQ_NUMDIR
		tmp.ADVAL_I=a.ADVAL_I*as;
	else
	    FOR_I_EQ_1_LTEQ_NUMDIR {
		as=0;
		if (a.ADVAL_I>0) as=1;
		if (a.ADVAL_I<0) as=-1;
                tmp.ADVAL_I=a.ADVAL_I*as;
            }
    return tmp;
}

inline adouble ceil (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::ceil(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=0.0;
    return tmp;
}

inline adouble floor (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP::floor(a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=0.0;
    return tmp;
}

inline adouble fmax (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.PRIMAL_VALUE-b.PRIMAL_VALUE;
    if (tmp2<0) {
	    tmp.PRIMAL_VALUE=b.PRIMAL_VALUE;
	    FOR_I_EQ_1_LTEQ_NUMDIR
		    tmp.ADVAL_I=b.ADVAL_I;
    } else {
	    tmp.PRIMAL_VALUE=a.PRIMAL_VALUE;
        if (tmp2>0) {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		        tmp.ADVAL_I=a.ADVAL_I;
        } else {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		    {
		        if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=b.ADVAL_I;
		        else tmp.ADVAL_I=a.ADVAL_I;
            }
	    }
    }
    return tmp;
}

inline adouble fmax (double v, const adouble &a) {
    adouble tmp;
    double tmp2=v-a.PRIMAL_VALUE;
    if (tmp2<0) {
	    tmp.PRIMAL_VALUE=a.PRIMAL_VALUE;
	    FOR_I_EQ_1_LTEQ_NUMDIR
		    tmp.ADVAL_I=a.ADVAL_I;
    } else {
	    tmp.PRIMAL_VALUE=v;
        if (tmp2>0) {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		        tmp.ADVAL_I=0.0;
        } else {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		    {
		        if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
		        else tmp.ADVAL_I=0.0;
            }
	    }
    }
    return tmp;
}

inline adouble fmax (const adouble &a, double v) {
    adouble tmp;
    double tmp2=a.PRIMAL_VALUE-v;
    if (tmp2<0) {
	    tmp.PRIMAL_VALUE=v;
	    FOR_I_EQ_1_LTEQ_NUMDIR
		    tmp.ADVAL_I=0.0;
    } else {
	    tmp.PRIMAL_VALUE=a.PRIMAL_VALUE;
        if (tmp2>0) {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		        tmp.ADVAL_I=a.ADVAL_I;
        } else {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		    {
		        if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
		        else tmp.ADVAL_I=0.0;
            }
	    }
    }
    return tmp;
}

inline adouble fmin (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.PRIMAL_VALUE-b.PRIMAL_VALUE;
    if (tmp2<0) {
	    tmp.PRIMAL_VALUE=a.PRIMAL_VALUE;
	    FOR_I_EQ_1_LTEQ_NUMDIR
		    tmp.ADVAL_I=a.ADVAL_I;
    } else {
	    tmp.PRIMAL_VALUE=b.PRIMAL_VALUE;
        if (tmp2>0) {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		        tmp.ADVAL_I=b.ADVAL_I;
        } else {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		    {
		        if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=a.ADVAL_I;
		        else tmp.ADVAL_I=b.ADVAL_I;
            }
	    }
    }
    return tmp;
}

inline adouble fmin (double v, const adouble &a) {
    adouble tmp;
    double tmp2=v-a.PRIMAL_VALUE;
    if (tmp2<0) {
	    tmp.PRIMAL_VALUE=v;
	    FOR_I_EQ_1_LTEQ_NUMDIR
		    tmp.ADVAL_I=0.0;
    } else {
	    tmp.PRIMAL_VALUE=a.PRIMAL_VALUE;
        if (tmp2>0) {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		        tmp.ADVAL_I=a.ADVAL_I;
        } else {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		    {
		        if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
		        else tmp.ADVAL_I=0.0;
            }
	    }
    }
    return tmp;
}

inline adouble fmin (const adouble &a, double v) {
    adouble tmp;
    double tmp2=a.PRIMAL_VALUE-v;
    if (tmp2<0) {
	    tmp.PRIMAL_VALUE=a.PRIMAL_VALUE;
	    FOR_I_EQ_1_LTEQ_NUMDIR
		    tmp.ADVAL_I=a.ADVAL_I;
    } else {
	    tmp.PRIMAL_VALUE=v;
        if (tmp2>0) {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		        tmp.ADVAL_I=0.0;
        } else {
		    FOR_I_EQ_1_LTEQ_NUMDIR
		    {
		        if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
		        else tmp.ADVAL_I=0.0;
            }
	    }
    }
    return tmp;
}

inline adouble ldexp (const adouble &a, const adouble &b) {
    adouble tmp = a*pow(2.,b);
    return tmp;
}

inline adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

inline adouble ldexp (const double v, const adouble &a) {
    adouble tmp = v*pow(2.,a);
    return tmp;
}

inline double frexp (const adouble &a, int* v) {
    return ADOLC_MATH_NSP::frexp(a.PRIMAL_VALUE, v);
}

#if defined(ATRIG_ERF)
inline adouble erf (const adouble &a) {
    adouble tmp;
	tmp.PRIMAL_VALUE=ADOLC_MATH_NSP_ERF::erf(a.PRIMAL_VALUE);
	double tmp2 = 2.0 /
	    ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) *
	    ADOLC_MATH_NSP_ERF::exp(-a.PRIMAL_VALUE*a.PRIMAL_VALUE);
	FOR_I_EQ_1_LTEQ_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}
#endif

inline void condassign( adouble &res, const adouble &cond,
			const adouble &arg1, const adouble &arg2 ) {
	if (cond.getValue() > 0) 
	    res = arg1;
	else
	    res = arg2;
}

inline void condassign( adouble &res, const adouble &cond,
			const adouble &arg ) {
	if (cond.getValue() > 0) 
	    res = arg;
}

inline void condeqassign( adouble &res, const adouble &cond,
                          const adouble &arg1, const adouble &arg2 ) {
	if (cond.getValue() >= 0) 
	    res = arg1;
	else
	    res = arg2;
}

inline void condeqassign( adouble &res, const adouble &cond,
                          const adouble &arg ) {
	if (cond.getValue() >= 0) 
	    res = arg;
}



/*******************  nontemporary results  *********************************/
inline adouble& adouble::operator = (const double v) {
	PRIMAL_VALUE=v;
	FOR_I_EQ_1_LTEQ_NUMDIR
	    ADVAL_I=0.0;
    return *this;
}

inline adouble& adouble::operator = (const adouble& a) {
    FOR_I_EQ_0_LTEQ_NUMDIR
        ADVAL_I=a.ADVAL_I;
    return *this;
}

inline adouble& adouble::operator += (const double v) {
	PRIMAL_VALUE+=v;
    return *this;
}

inline adouble& adouble::operator += (const adouble& a) {
    FOR_I_EQ_0_LTEQ_NUMDIR
        ADVAL_I+=a.ADVAL_I;
    return *this;
}

inline adouble& adouble::operator -= (const double v) {
	PRIMAL_VALUE-=v;
    return *this;
}

inline adouble& adouble::operator -= (const adouble& a) {
	FOR_I_EQ_0_LTEQ_NUMDIR
	    ADVAL_I-=a.ADVAL_I;
    return *this;
}

inline adouble& adouble::operator *= (const double v) {
	FOR_I_EQ_0_LTEQ_NUMDIR
	    ADVAL_I*=v;
    return *this;
}

inline adouble& adouble::operator *= (const adouble& a) {
	FOR_I_EQ_1_LTEQ_NUMDIR
	    ADVAL_I=ADVAL_I*a.PRIMAL_VALUE+PRIMAL_VALUE*a.ADVAL_I;
	PRIMAL_VALUE*=a.PRIMAL_VALUE;
    return *this;
}

inline adouble& adouble::operator /= (const double v) {
	FOR_I_EQ_0_LTEQ_NUMDIR
	    ADVAL_I/=v;
    return *this;
}

inline adouble& adouble::operator /= (const adouble& a) {
	FOR_I_EQ_1_LTEQ_NUMDIR
	    ADVAL_I=(ADVAL_I*a.PRIMAL_VALUE-PRIMAL_VALUE*a.ADVAL_I)/(a.PRIMAL_VALUE*a.PRIMAL_VALUE);
	PRIMAL_VALUE=PRIMAL_VALUE/a.PRIMAL_VALUE;
    return *this;
}

// not
inline int adouble::operator ! () const {
    return PRIMAL_VALUE==0.0;
}

// comparision
inline int adouble::operator != (const adouble &a) const {
    return PRIMAL_VALUE!=a.PRIMAL_VALUE;
}

inline int adouble::operator != (const double v) const {
    return PRIMAL_VALUE!=v;
}

inline int operator != (const double v, const adouble &a) {
    return v!=a.PRIMAL_VALUE;
}

inline int adouble::operator == (const adouble &a) const {
    return PRIMAL_VALUE==a.PRIMAL_VALUE;
}

inline int adouble::operator == (const double v) const {
    return PRIMAL_VALUE==v;
}

inline int operator == (const double v, const adouble &a) {
    return v==a.PRIMAL_VALUE;
}

inline int adouble::operator <= (const adouble &a) const {
    return PRIMAL_VALUE<=a.PRIMAL_VALUE;
}

inline int adouble::operator <= (const double v) const {
    return PRIMAL_VALUE<=v;
}

inline int operator <= (const double v, const adouble &a) {
    return v<=a.PRIMAL_VALUE;
}

inline int adouble::operator >= (const adouble &a) const {
    return PRIMAL_VALUE>=a.PRIMAL_VALUE;
}

inline int adouble::operator >= (const double v) const {
    return PRIMAL_VALUE>=v;
}

inline int operator >= (const double v, const adouble &a) {
    return v>=a.PRIMAL_VALUE;
}

inline int adouble::operator >  (const adouble &a) const {
    return PRIMAL_VALUE>a.PRIMAL_VALUE;
}

inline int adouble::operator >  (const double v) const {
    return PRIMAL_VALUE>v;
}

inline int operator >  (const double v, const adouble &a) {
    return v>a.PRIMAL_VALUE;
}

inline int adouble::operator <  (const adouble &a) const {
    return PRIMAL_VALUE<a.PRIMAL_VALUE;
}

inline int adouble::operator <  (const double v) const {
    return PRIMAL_VALUE<v;
}

inline int operator <  (const double v, const adouble &a) {
    return v<a.PRIMAL_VALUE;
}

/*******************  getter / setter  **************************************/
inline adouble::operator double const & () const {
    return PRIMAL_VALUE;
}

inline adouble::operator double && () {
    return (double&&)PRIMAL_VALUE;
}

inline adouble::operator double() {
    return PRIMAL_VALUE;
}


inline double adouble::getValue() const {
    return PRIMAL_VALUE;
}

inline void adouble::setValue(const double v) {
    PRIMAL_VALUE=v;
}

inline const double *const adouble::getADValue() const {
    return (adval+1);
}

inline void adouble::setADValue(const double *const v) {
    FOR_I_EQ_1_LTEQ_NUMDIR
    ADVAL_I=v[_i-1];
}

inline double adouble::getADValue(const unsigned int p) const {
    if (p>=adouble::numDir) 
    {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"getADValue(...)\"!!!\n");
        throw logic_error("incorrect function call, errorcode=-1");
    }
    return adval[p+1];
}

inline void adouble::setADValue(const unsigned int p, const double v) {
    if (p>=adouble::numDir) 
    {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"setADValue(...)\"!!!\n");
        throw logic_error("incorrect function call, errorcode=-1");
    }
    adval[p+1]=v;
}

}
#endif
