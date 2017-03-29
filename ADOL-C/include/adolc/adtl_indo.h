/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adtl_indo.h contains that declaratins of procedures used
           for sparsity pattern recognition.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert, Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_ADTL_INDO_H
#define ADOLC_ADTL_INDO_H

#include <ostream>
#include <adolc/internal/common.h>
#include <list>
#include <stdexcept>

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define COMPILER_HAS_CXX11
#else
#error "please use -std=c++11 compiler flag with a C++11 compliant compiler"
#endif

#include <adolc/adtl.h>

using std::ostream;
using std::istream;
using std::list;
using std::logic_error;

template<typename T>
class func_ad {
public:
    virtual int operator() (int n, T *x, int m, T *y) = 0;
};

namespace adtl_indo{
class adouble;
ADOLC_DLL_EXPORT int ADOLC_Init_sparse_pattern(adouble *a, int n,unsigned int start_cnt);
ADOLC_DLL_EXPORT int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat);
}

ADOLC_DLL_EXPORT int ADOLC_get_sparse_jacobian( func_ad<adtl::adouble> *const func, func_ad<adtl_indo::adouble> *const func_indo, int n, int m, int repeat, double* basepoints, int *nnz, unsigned int **rind, unsigned int **cind, double **values);

namespace adtl_indo {

double makeNaN();
double makeInf();

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

    inline explicit operator double const&() const;
    inline explicit operator double&&();
    inline explicit operator double();

protected:
    inline const list<unsigned int>& get_pattern() const;
    inline void add_to_pattern(const list<unsigned int>& v);
    inline size_t get_pattern_size() const;
    inline void delete_pattern();

public:
    friend int ADOLC_Init_sparse_pattern(adouble *a, int n,unsigned int start_cnt);
    friend int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat);
    /*******************  i/o operations  *********************************/
    ADOLC_DLL_EXPORT friend ostream& operator << ( ostream&, const adouble& );
    ADOLC_DLL_EXPORT friend istream& operator >> ( istream&, adouble& );

private:
    double val;
    list<unsigned int> pattern;
};

}

#include <cmath>
#include <iostream>
#include <limits>

namespace adtl_indo {

#if defined(HAVE_BUILTIN_EXPECT) && HAVE_BUILTIN_EXPECT
#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)
#endif

#ifndef likely
#define likely(x) (x)
#endif
#ifndef unlikely
#define unlikely(x) (x)
#endif

inline double makeNaN() {
    return ADOLC_MATH_NSP::numeric_limits<double>::quiet_NaN();
}

inline double makeInf() {
    return ADOLC_MATH_NSP::numeric_limits<double>::infinity();
}

/*******************************  ctors  ************************************/
inline adouble::adouble() : val(0) {
    if (!pattern.empty())
        pattern.clear();
}

inline adouble::adouble(const double v) : val(v) {
    if (!pattern.empty())
        pattern.clear();
}

inline adouble::adouble(const double v, const double* adv) : val(v) {
    if (!pattern.empty())
        pattern.clear();
}

inline adouble::adouble(const adouble& a) : val(a.val) {
    if (!pattern.empty())
        pattern.clear();

    add_to_pattern(a.get_pattern());
}

/*******************************  dtors  ************************************/
inline adouble::~adouble() {
#if 0
    if ( !pattern.empty() )
	pattern.clear();
#endif
}

/*************************  temporary results  ******************************/
// sign
inline adouble adouble::operator - () const {
    adouble tmp;
	tmp.val=-val;
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator + () const {
    return *this;
}

// addition
inline adouble adouble::operator + (const double v) const {
    adouble tmp(val+v);
	tmp.add_to_pattern( get_pattern() ) ;
    return tmp;
}

inline adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
	tmp.val=val+a.val;
	tmp.add_to_pattern( get_pattern()  );
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble operator + (const double v, const adouble& a) {
    adouble tmp(v+a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// subtraction
inline adouble adouble::operator - (const double v) const {
    adouble tmp(val-v);
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator - (const adouble& a) const {
    adouble tmp;
	tmp.val=val-a.val;
	tmp.add_to_pattern( get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble operator - (const double v, const adouble& a) {
    adouble tmp;
	tmp.val=v-a.val;
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// multiplication
inline adouble adouble::operator * (const double v) const {
    adouble tmp;
	tmp.val=val*v;
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
	tmp.val=val*a.val;
	tmp.add_to_pattern(   get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble operator * (const double v, const adouble& a) {
    adouble tmp;
	tmp.val=v*a.val;
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// division
inline adouble adouble::operator / (const double v) const {
    adouble tmp;
	tmp.val=val/v;
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
	tmp.val=val/a.val;
	tmp.add_to_pattern(   get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble operator / (const double v, const adouble& a) {
    adouble tmp;
	tmp.val=v/a.val;
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// inc/dec
inline adouble adouble::operator ++ () {
	++val;
    return *this;
}

inline adouble adouble::operator ++ (int) {
    adouble tmp;
	tmp.val=val++;
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator -- () {
	--val;
    return *this;
}

inline adouble adouble::operator -- (int) {
    adouble tmp;
	tmp.val=val--;
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

// functions
inline adouble tan(const adouble& a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::tan(a.val);    
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble exp(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::exp(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble log(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::log(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble sqrt(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::sqrt(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble sin(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::sin(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble cos(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::cos(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble asin(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::asin(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble acos(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::acos(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble atan(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::atan(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble atan2(const adouble &a, const adouble &b) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::atan2(a.val, b.val);
	tmp.add_to_pattern( a.get_pattern() );
	tmp.add_to_pattern( b.get_pattern() );
    return tmp;
}

inline adouble pow(const adouble &a, double v) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::pow(a.val, v);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::pow(a.val, b.val);
	tmp.add_to_pattern( a.get_pattern() );
	tmp.add_to_pattern( b.get_pattern() );
    return tmp;
}

inline adouble pow(double v, const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::pow(v, a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble log10(const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::log10(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble sinh (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::sinh(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble cosh (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::cosh(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble tanh (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::tanh(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

#if defined(ATRIG_ERF)
inline adouble asinh (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP_ERF::asinh(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble acosh (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP_ERF::acosh(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble atanh (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP_ERF::atanh(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}
#endif

inline adouble fabs (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::fabs(a.val);
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble ceil (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::ceil(a.val);
    return tmp;
}

inline adouble floor (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP::floor(a.val);
    return tmp;
}

inline adouble fmax (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.val-b.val;
    if (tmp2<0) {
	    tmp.val=b.val;
	    tmp.add_to_pattern( b.get_pattern() );
    } else {
	    tmp.val=a.val;
        if (tmp2>0) {
		    tmp.add_to_pattern( a.get_pattern() );
        } else {
		    tmp.add_to_pattern( a.get_pattern() );
		    tmp.add_to_pattern( b.get_pattern() );
	    }
    }
    return tmp;
}

inline adouble fmax (double v, const adouble &a) {
    adouble tmp;
    double tmp2=v-a.val;
    if (tmp2<0) {
	    tmp.val=a.val;
	    tmp.add_to_pattern( a.get_pattern() );
    } else {
	    tmp.val=v;
        if (tmp2>0) {
        } else {
		    tmp.add_to_pattern( a.get_pattern() );
	    }
    }
    return tmp;
}

inline adouble fmax (const adouble &a, double v) {
    adouble tmp;
    double tmp2=a.val-v;
    if (tmp2<0) {
	    tmp.val=v;
    } else {
	    tmp.val=a.val;
		tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

inline adouble fmin (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.val-b.val;
    if (tmp2<0) {
	    tmp.val=a.val;
	    tmp.add_to_pattern( a.get_pattern() );
    } else {
	    tmp.val=b.val;
        if (tmp2>0) {
		    tmp.add_to_pattern( b.get_pattern() );
        } else {
		    tmp.add_to_pattern( a.get_pattern() );
		    tmp.add_to_pattern( b.get_pattern() );
	    }
    }
    return tmp;
}

inline adouble fmin (double v, const adouble &a) {
    adouble tmp;
    double tmp2=v-a.val;
    if (tmp2<0) {
	    tmp.val=v;
    } else {
	    tmp.val=a.val;
		tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

inline adouble fmin (const adouble &a, double v) {
    adouble tmp;
    double tmp2=a.val-v;
    if (tmp2<0) {
	    tmp.val=a.val;
	    tmp.add_to_pattern( a.get_pattern() );
    } else {
	    tmp.val=v;
        if (tmp2>0) {
        } else {
		    tmp.add_to_pattern( a.get_pattern() );
	    }
    }
    return tmp;
}

inline adouble ldexp (const adouble &a, const adouble &b) {
    adouble tmp = a*pow(2.,b);
	tmp.add_to_pattern( a.get_pattern() ) ;
	tmp.add_to_pattern( b.get_pattern() ) ;
    return tmp;
}

inline adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

inline adouble ldexp (const double v, const adouble &a) {
    adouble tmp = v*pow(2.,a);
	tmp.add_to_pattern( a.get_pattern() ) ;
    return tmp;
}

inline double frexp (const adouble &a, int* v) {
    return ADOLC_MATH_NSP::frexp(a.val, v);
}

#if defined(ATRIG_ERF)
inline adouble erf (const adouble &a) {
    adouble tmp;
	tmp.val=ADOLC_MATH_NSP_ERF::erf(a.val);
	tmp.add_to_pattern( a.get_pattern() ) ;
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
	val=v;
	if (!pattern.empty()) pattern.clear();
    return *this;
}

inline adouble& adouble::operator = (const adouble& a) {
	val=a.val;
	if (!pattern.empty()) pattern.clear();
	add_to_pattern( a.get_pattern() );
    return *this;
}

inline adouble& adouble::operator += (const double v) {
	val+=v;
    return *this;
}

inline adouble& adouble::operator += (const adouble& a) {
	val=val+a.val;
	add_to_pattern( a.get_pattern() );
    return *this;
}

inline adouble& adouble::operator -= (const double v) {
	val-=v;
    return *this;
}

inline adouble& adouble::operator -= (const adouble& a) {
	val=val-a.val;
	add_to_pattern( a.get_pattern() ) ;
    return *this;
}

inline adouble& adouble::operator *= (const double v) {
	val=val*v;
    return *this;
}

inline adouble& adouble::operator *= (const adouble& a) {
	val*=a.val;
	add_to_pattern( a.get_pattern() ) ;
    return *this;
}

inline adouble& adouble::operator /= (const double v) {
	val/=v;
    return *this;
}

inline adouble& adouble::operator /= (const adouble& a) {
	val=val/a.val;
	add_to_pattern( a.get_pattern() ) ;
    return *this;
}

// not
inline int adouble::operator ! () const {
    return val==0.0;
}

// comparision
inline int adouble::operator != (const adouble &a) const {
    return val!=a.val;
}

inline int adouble::operator != (const double v) const {
    return val!=v;
}

inline int operator != (const double v, const adouble &a) {
    return v!=a.val;
}

inline int adouble::operator == (const adouble &a) const {
    return val==a.val;
}

inline int adouble::operator == (const double v) const {
    return val==v;
}

inline int operator == (const double v, const adouble &a) {
    return v==a.val;
}

inline int adouble::operator <= (const adouble &a) const {
    return val<=a.val;
}

inline int adouble::operator <= (const double v) const {
    return val<=v;
}

inline int operator <= (const double v, const adouble &a) {
    return v<=a.val;
}

inline int adouble::operator >= (const adouble &a) const {
    return val>=a.val;
}

inline int adouble::operator >= (const double v) const {
    return val>=v;
}

inline int operator >= (const double v, const adouble &a) {
    return v>=a.val;
}

inline int adouble::operator >  (const adouble &a) const {
    return val>a.val;
}

inline int adouble::operator >  (const double v) const {
    return val>v;
}

inline int operator >  (const double v, const adouble &a) {
    return v>a.val;
}

inline int adouble::operator <  (const adouble &a) const {
    return val<a.val;
}

inline int adouble::operator <  (const double v) const {
    return val<v;
}

inline int operator <  (const double v, const adouble &a) {
    return v<a.val;
}

/*******************  getter / setter  **************************************/
inline adouble::operator double const & () const {
    return val;
}

inline adouble::operator double && () {
    return (double&&)val;
}

inline adouble::operator double() {
    return val;
}


inline double adouble::getValue() const {
    return val;
}

inline void adouble::setValue(const double v) {
    val=v;
}

inline const list<unsigned int>& adouble::get_pattern() const {
    return pattern;
}

inline void adouble::delete_pattern() {
    if ( !pattern.empty() )
	pattern.clear();
}

inline void adouble::add_to_pattern(const list<unsigned int>& v) {
    if (likely( pattern != v)) {
	if( !v.empty() ){
	    list<unsigned int> cv = v;
	    //pattern.splice(pattern.end(), cv);
	    pattern.merge(cv);
	    //if (pattern.size() > refcounter::refcnt) {
	    //pattern.sort();
	    pattern.unique();
		//}
	}
    }
}

inline size_t adouble::get_pattern_size() const {
    size_t s=0;
    if( !pattern.empty() )
      s = pattern.size();
    return s;
}

}

#endif
