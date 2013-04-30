/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: adoublecuda.h contains the class of adouble specifically
           suited to be used within CUDA environment

 Copyright (c) Alina Koniaeva, Andrea Walther

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

#if !defined(ADOLC_ADOUBLECUDA_H)
#define ADOLC_ADOUBLECUDA_H 1

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>
using std::cout;
using std::cin;
using std::cerr;
using std::ostream;
using std::istream;

#include <cuda_runtime.h>
#include <math_constants.h>

namespace adtlc {


#  define ADVAL                adval
#  define ADVAL_TYPE           double
#  define FOR_I_EQ_0_LT_NUMDIR
#  define ADVAL_I              adval
#  define ADV_I                adv
#  define V_I                  v


#define ADOLC_MATH_NSP std

inline __device__ double makeNaN() {
    return CUDART_NAN;
}

inline __device__ double makeInf() {
    return CUDART_INF;
}


class adouble {
public:
    // ctors
    __device__  inline adouble();
    __device__  inline adouble(const double v);
    __device__  inline adouble(const double v, ADVAL_TYPE adv);
    __device__  inline adouble(const adouble& a);
#if defined(DYNAMIC_DIRECTIONS)
    inline ~adouble();
#endif
    /*******************  temporary results  ******************************/
    // sign
    __device__  inline adouble operator - () const;
    __device__  inline adouble operator + () const;

    // addition
    __device__  inline adouble operator + (const double v) const;
    __device__  inline adouble operator + (const adouble& a) const;
    __device__  inline friend
    adouble operator + (const double v, const adouble& a);

    // substraction
    __device__  inline adouble operator - (const double v) const;
    __device__  inline adouble operator - (const adouble& a) const;
    __device__  inline friend
    adouble operator - (const double v, const adouble& a);

    // multiplication
    __device__  inline adouble operator * (const double v) const;
    __device__  inline adouble operator * (const adouble& a) const;
    __device__  inline friend
    adouble operator * (const double v, const adouble& a);

    // division
    __device__  inline adouble operator / (const double v) const;
    __device__  inline adouble operator / (const adouble& a) const;
    __device__  inline friend
    adouble operator / (const double v, const adouble& a);

    // inc/dec
    __device__  inline adouble operator ++ ();
    __device__  inline adouble operator ++ (int);
    __device__  inline adouble operator -- ();
    __device__  inline adouble operator -- (int);

    // functions
    __device__  inline friend adouble tan(const adouble &a);
    __device__  inline friend adouble exp(const adouble &a);
    __device__  inline friend adouble log(const adouble &a);
    __device__  inline friend adouble sqrt(const adouble &a);
    __device__  inline friend adouble sin(const adouble &a);
    __device__  inline friend adouble cos(const adouble &a);
    __device__  inline friend adouble asin(const adouble &a);
    __device__  inline friend adouble acos(const adouble &a);
    __device__  inline friend adouble atan(const adouble &a);

    __device__  inline friend adouble atan2(const adouble &a, const adouble &b);
    __device__  inline friend adouble pow(const adouble &a, double v);
    __device__  inline friend adouble pow(const adouble &a, const adouble &b);
    __device__  inline friend adouble pow(double v, const adouble &a);
    __device__  inline friend adouble log10(const adouble &a);

    __device__  inline friend adouble sinh (const adouble &a);
    __device__  inline friend adouble cosh (const adouble &a);
    __device__  inline friend adouble tanh (const adouble &a);
#if defined(ATRIG_ERF)
    __device__  inline friend adouble asinh (const adouble &a);
    __device__  inline friend adouble acosh (const adouble &a);
    __device__  inline friend adouble atanh (const adouble &a);
#endif
    __device__  inline friend adouble fabs (const adouble &a);
    __device__  inline friend adouble ceil (const adouble &a);
    __device__  inline friend adouble floor (const adouble &a);
    __device__  inline friend adouble fmax (const adouble &a, const adouble &b);
    __device__  inline friend adouble fmax (double v, const adouble &a);
    __device__  inline friend adouble fmax (const adouble &a, double v);
    __device__  inline friend adouble fmin (const adouble &a, const adouble &b);
    __device__  inline friend adouble fmin (double v, const adouble &a);
    __device__  inline friend adouble fmin (const adouble &a, double v);
    __device__  inline friend adouble ldexp (const adouble &a, const adouble &b);
    __device__  inline friend adouble ldexp (const adouble &a, const double v);
    __device__  inline friend adouble ldexp (const double v, const adouble &a);
    __device__  inline friend double frexp (const adouble &a, int* v);
#if defined(ATRIG_ERF)
    __device__  inline friend adouble erf (const adouble &a);
#endif


    /*******************  nontemporary results  ***************************/
    // assignment
    __device__  inline void operator = (const double v);
    __device__  inline void operator = (const adouble& a);

    // addition
    __device__  inline void operator += (const double v);
    __device__  inline void operator += (const adouble& a);

    // substraction
    __device__  inline void operator -= (const double v);
    __device__  inline void operator -= (const adouble& a);

    // multiplication
    __device__  inline void operator *= (const double v);
    __device__  inline void operator *= (const adouble& a);

    // division
    __device__  inline void operator /= (const double v);
    __device__  inline void operator /= (const adouble& a);

    // not
    __device__  inline int operator ! () const;

    // comparision
    __device__  inline int operator != (const adouble&) const;
    __device__  inline int operator != (const double) const;
    __device__  inline friend int operator != (const double, const adouble&);

    __device__  inline int operator == (const adouble&) const;
    __device__  inline int operator == (const double) const;
    __device__  inline friend int operator == (const double, const adouble&);

    __device__  inline int operator <= (const adouble&) const;
    __device__  inline int operator <= (const double) const;
    __device__  inline friend int operator <= (const double, const adouble&);

    __device__  inline int operator >= (const adouble&) const;
    __device__  inline int operator >= (const double) const;
    __device__  inline friend int operator >= (const double, const adouble&);

    __device__  inline int operator >  (const adouble&) const;
    __device__  inline int operator >  (const double) const;
    __device__  inline friend int operator >  (const double, const adouble&);

    __device__  inline int operator <  (const adouble&) const;
    __device__  inline int operator <  (const double) const;
    __device__  inline friend int operator <  (const double, const adouble&);

    /*******************  getter / setter  ********************************/
    __device__  inline double getValue() const;
    __device__  inline void setValue(const double v);
    __device__  inline ADVAL_TYPE getADValue() const;
    __device__  inline void setADValue(ADVAL_TYPE v);
#if defined(NUMBER_DIRECTIONS)
    inline double getADValue(const unsigned int p) const;
    inline void setADValue(const unsigned int p, const double v);
#endif

    /*******************  i/o operations  *********************************/
    inline friend ostream& operator << ( ostream&, const adouble& );
    inline friend istream& operator >> ( istream&, adouble& );


private:
    // internal variables
    double val;
    double ADVAL;
};
  
/*******************************  ctors  ************************************/
adouble::adouble() {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
}

adouble::adouble(const double v) : val(v) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I = 0.0;
}

adouble::adouble(const double v, ADVAL_TYPE adv) : val(v) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=ADV_I;
}

adouble::adouble(const adouble& a) : val(a.val) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=a.ADVAL_I;
}

/*******************************  dtors  ************************************/
#if defined(DYNAMIC_DIRECTIONS)
adouble::~adouble() {
    delete[] adval;
}
#endif

/*************************  temporary results  ******************************/
// sign
adouble adouble::operator - () const {
    adouble tmp;
    tmp.val=-val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=-ADVAL_I;
    return tmp;
}

adouble adouble::operator + () const {
    return *this;
}

// addition
adouble adouble::operator + (const double v) const {
    return adouble(val+v, adval);
}

adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
    tmp.val=val+a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I+a.ADVAL_I;
    return tmp;
}

adouble operator + (const double v, const adouble& a) {
    return adouble(v+a.val, a.adval);
}

// subtraction
adouble adouble::operator - (const double v) const {
    return adouble(val-v, adval);
}

adouble adouble::operator - (const adouble& a) const {
    adouble tmp;
    tmp.val=val-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I-a.ADVAL_I;
    return tmp;
}

adouble operator - (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=-a.ADVAL_I;
    return tmp;
}

// multiplication
adouble adouble::operator * (const double v) const {
    adouble tmp;
    tmp.val=val*v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I*v;
    return tmp;
}

adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
    tmp.val=val*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    return tmp;
}

adouble operator * (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=v*a.ADVAL_I;
    return tmp;
}

// division
adouble adouble::operator / (const double v) const {
    adouble tmp;
    tmp.val=val/v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I/v;
    return tmp;
}

adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
    tmp.val=val/a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    return tmp;
}

adouble operator / (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v/a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=(-v*a.ADVAL_I)/(a.val*a.val);
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
    return tmp;
}

adouble exp(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::exp(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp.val*a.ADVAL_I;
    return tmp;
}

adouble log(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::log(a.val);
    FOR_I_EQ_0_LT_NUMDIR
	if (a.val>0) tmp.ADVAL_I=a.ADVAL_I/a.val;
	else if (a.val==0 && a.ADVAL_I != 0.0) {
	    int sign = (a.ADVAL_I < 0)  ? -1 : 1;
	    tmp.ADVAL_I=sign* makeInf();
	    } else tmp.ADVAL_I=makeNaN();
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
	} 
	else tmp.ADVAL_I=makeNaN(); 
    return tmp;
}

adouble sin(const adouble &a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::sin(a.val);
    tmp2=ADOLC_MATH_NSP::cos(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

adouble cos(const adouble &a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::cos(a.val);
    tmp2=-ADOLC_MATH_NSP::sin(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

adouble asin(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::asin(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

adouble acos(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::acos(a.val);
    double tmp2=-ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
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
    return tmp;
}

adouble pow(const adouble &a, double v) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(a.val, v);
    double tmp2=v*ADOLC_MATH_NSP::pow(a.val, v-1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(a.val, b.val);
    double tmp2=b.val*ADOLC_MATH_NSP::pow(a.val, b.val-1);
    double tmp3=ADOLC_MATH_NSP::log(a.val)*tmp.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I+tmp3*b.ADVAL_I;
    return tmp;
}

adouble pow(double v, const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(v, a.val);
    double tmp2=tmp.val*ADOLC_MATH_NSP::log(v);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

adouble log10(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::log10(a.val);
    double tmp2=ADOLC_MATH_NSP::log((double)10)*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

adouble sinh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::sinh(a.val);
    double tmp2=ADOLC_MATH_NSP::cosh(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    return tmp;
}

adouble cosh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::cosh(a.val);
    double tmp2=ADOLC_MATH_NSP::sinh(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    return tmp;
}

adouble tanh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::tanh(a.val);
    double tmp2=ADOLC_MATH_NSP::cosh(a.val);
    tmp2*=tmp2;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

#if defined(ATRIG_ERF)
adouble asinh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::asinh(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val+1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

adouble acosh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::acosh(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val-1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

adouble atanh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::atanh(a.val);
    double tmp2=1-a.val*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
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
            return tmp;
}

adouble ceil (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::ceil(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=0.0;
    return tmp;
}

adouble floor (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::floor(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=0.0;
    return tmp;
}

adouble fmax (const adouble &a, const adouble &b) {
    adouble tmp;
    double tmp2=a.val-b.val;
    if (tmp2<0) {
        tmp.val=b.val;
        FOR_I_EQ_0_LT_NUMDIR
        tmp.ADVAL_I=b.ADVAL_I;
    } else {
        tmp.val=a.val;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=a.ADVAL_I;
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=b.ADVAL_I;
                else tmp.ADVAL_I=a.ADVAL_I;
                }
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
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=0.0;
                }
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
    } else {
        tmp.val=b.val;
        if (tmp2>0) {
            FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=b.ADVAL_I;
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=b.ADVAL_I;
                }
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
        } else {
            FOR_I_EQ_0_LT_NUMDIR
            {
                if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
                else tmp.ADVAL_I=0.0;
                }
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
            }
}
return tmp;
}

adouble ldexp (const adouble &a, const adouble &b) {
    return a*pow(2.,b);
}

adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

adouble ldexp (const double v, const adouble &a) {
    return v*pow(2.,a);
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
    return tmp;
}
#endif


/*******************  nontemporary results  *********************************/
void adouble::operator = (const double v) {
    val=v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=0.0;
}

void adouble::operator = (const adouble& a) {
    val=a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=a.ADVAL_I;
}

void adouble::operator += (const double v) {
    val+=v;
}

void adouble::operator += (const adouble& a) {
    val=val+a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I+=a.ADVAL_I;
}

void adouble::operator -= (const double v) {
    val-=v;
}

void adouble::operator -= (const adouble& a) {
    val=val-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I-=a.ADVAL_I;
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
}
/****************************************************************************/
/* end traceless gpu implementation first order derivatives                      */
/****************************************************************************/
#endif
