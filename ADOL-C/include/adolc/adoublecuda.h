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

#include <cstdlib>
#include <iostream>
#include <cmath>
using std::cout;
using std::ostream;
using std::istream;

#include <cuda_runtime.h>
#include <math_constants.h>

namespace adtlc {


#if defined(NUMBER_DIRECTIONS)
__managed__ size_t ADOLC_numDir = NUMBER_DIRECTIONS;
# if defined(DYNAMIC_DIRECTIONS)
#  define ADVAL_DECL           *adval
#  define ADVAL_TYPE_ADV       const double* adv
# else
#  define ADVAL_DECL           adval[NUMBER_DIRECTIONS]
#  define ADVAL_TYPE_ADV       const double adv[NUMBER_DIRECTIONS]
# endif
#  define ADVAL_TYPE           double*
#  define FOR_I_EQ_0_LT_NUMDIR for (size_t i = 0; i < ADOLC_numDir; i++)
#  define ADVAL_I              adval[i]
#  define ADV_I                adv[i]
#  define V_I                  v[i]
#else
#  define ADVAL_DECL           adval
#  define ADVAL                adval
#  define ADVAL_TYPE_ADV       double adv
#  define ADVAL_TYPE           double
#  define FOR_I_EQ_0_LT_NUMDIR
#  define ADVAL_I              adval
#  define ADV_I                adv
#  define V_I                  v
#endif


#define ADOLC_MATH_NSP std

inline __device__ double makeNaN() {
    return CUDART_NAN;
}

inline __device__ double makeInf() {
    return CUDART_INF;
}


#define CUDADEV __device__ inline
#define CUDAHOST __host__ inline
#define CUDAHOSTDEV __host__ __device__ inline

class adouble {
public:
    // ctors
    CUDADEV adouble();
    CUDADEV adouble(const double v);
    CUDADEV adouble(const double v, ADVAL_TYPE_ADV);
    CUDADEV adouble(const adouble& a);
#if defined(DYNAMIC_DIRECTIONS)
    CUDADEV ~adouble();
#endif
    /*******************  temporary results  ******************************/
    // sign
    CUDADEV adouble operator - () const;
    CUDADEV adouble operator + () const;

    // addition
    CUDADEV adouble operator + (const double v) const;
    CUDADEV adouble operator + (const adouble& a) const;
    CUDADEV friend
    adouble operator + (const double v, const adouble& a);

    // substraction
    CUDADEV adouble operator - (const double v) const;
    CUDADEV adouble operator - (const adouble& a) const;
    CUDADEV friend
    adouble operator - (const double v, const adouble& a);

    // multiplication
    CUDADEV adouble operator * (const double v) const;
    CUDADEV adouble operator * (const adouble& a) const;
    CUDADEV friend
    adouble operator * (const double v, const adouble& a);

    // division
    CUDADEV adouble operator / (const double v) const;
    CUDADEV adouble operator / (const adouble& a) const;
    CUDADEV friend
    adouble operator / (const double v, const adouble& a);

    // inc/dec
    CUDADEV adouble operator ++ ();
    CUDADEV adouble operator ++ (int);
    CUDADEV adouble operator -- ();
    CUDADEV adouble operator -- (int);

    // functions
    CUDADEV friend adouble tan(const adouble &a);
    CUDADEV friend adouble exp(const adouble &a);
    CUDADEV friend adouble log(const adouble &a);
    CUDADEV friend adouble sqrt(const adouble &a);
    CUDADEV friend adouble sin(const adouble &a);
    CUDADEV friend adouble cos(const adouble &a);
    CUDADEV friend adouble asin(const adouble &a);
    CUDADEV friend adouble acos(const adouble &a);
    CUDADEV friend adouble atan(const adouble &a);

    CUDADEV friend adouble atan2(const adouble &a, const adouble &b);
    CUDADEV friend adouble pow(const adouble &a, double v);
    CUDADEV friend adouble pow(const adouble &a, const adouble &b);
    CUDADEV friend adouble pow(double v, const adouble &a);
    CUDADEV friend adouble log10(const adouble &a);

    CUDADEV friend adouble sinh (const adouble &a);
    CUDADEV friend adouble cosh (const adouble &a);
    CUDADEV friend adouble tanh (const adouble &a);
#if defined(ATRIG_ERF)
    CUDADEV friend adouble asinh (const adouble &a);
    CUDADEV friend adouble acosh (const adouble &a);
    CUDADEV friend adouble atanh (const adouble &a);
#endif
    CUDADEV friend adouble fabs (const adouble &a);
    CUDADEV friend adouble ceil (const adouble &a);
    CUDADEV friend adouble floor (const adouble &a);
    CUDADEV friend adouble fmax (const adouble &a, const adouble &b);
    CUDADEV friend adouble fmax (double v, const adouble &a);
    CUDADEV friend adouble fmax (const adouble &a, double v);
    CUDADEV friend adouble fmin (const adouble &a, const adouble &b);
    CUDADEV friend adouble fmin (double v, const adouble &a);
    CUDADEV friend adouble fmin (const adouble &a, double v);
    CUDADEV friend adouble ldexp (const adouble &a, const adouble &b);
    CUDADEV friend adouble ldexp (const adouble &a, const double v);
    CUDADEV friend adouble ldexp (const double v, const adouble &a);
    CUDADEV friend double frexp (const adouble &a, int* v);
#if defined(ATRIG_ERF)
    CUDADEV friend adouble erf (const adouble &a);
#endif


    /*******************  nontemporary results  ***************************/
    // assignment
    CUDADEV void operator = (const double v);
    CUDADEV void operator = (const adouble& a);

    // addition
    CUDADEV void operator += (const double v);
    CUDADEV void operator += (const adouble& a);

    // substraction
    CUDADEV void operator -= (const double v);
    CUDADEV void operator -= (const adouble& a);

    // multiplication
    CUDADEV void operator *= (const double v);
    CUDADEV void operator *= (const adouble& a);

    // division
    CUDADEV void operator /= (const double v);
    CUDADEV void operator /= (const adouble& a);

    // not
    CUDADEV int operator ! () const;

    // comparision
    CUDADEV int operator != (const adouble&) const;
    CUDADEV int operator != (const double) const;
    CUDADEV friend int operator != (const double, const adouble&);

    CUDADEV int operator == (const adouble&) const;
    CUDADEV int operator == (const double) const;
    CUDADEV friend int operator == (const double, const adouble&);

    CUDADEV int operator <= (const adouble&) const;
    CUDADEV int operator <= (const double) const;
    CUDADEV friend int operator <= (const double, const adouble&);

    CUDADEV int operator >= (const adouble&) const;
    CUDADEV int operator >= (const double) const;
    CUDADEV friend int operator >= (const double, const adouble&);

    CUDADEV int operator >  (const adouble&) const;
    CUDADEV int operator >  (const double) const;
    CUDADEV friend int operator >  (const double, const adouble&);

    CUDADEV int operator <  (const adouble&) const;
    CUDADEV int operator <  (const double) const;
    CUDADEV friend int operator <  (const double, const adouble&);

    /*******************  getter / setter  ********************************/
    CUDAHOSTDEV double getValue() const;
    CUDAHOSTDEV void setValue(const double v);
    CUDAHOSTDEV ADVAL_TYPE getADValue() const;
    CUDAHOSTDEV void setADValue(ADVAL_TYPE v);
#if defined(NUMBER_DIRECTIONS)
    CUDAHOSTDEV double getADValue(const unsigned int p) const;
    CUDAHOSTDEV void setADValue(const unsigned int p, const double v);
#endif

    /*******************  i/o operations  *********************************/
    CUDAHOST friend ostream& operator << ( ostream&, const adouble& );
    CUDAHOST friend istream& operator >> ( istream&, adouble& );

private:
    // internal variables
    double val;
    double ADVAL_DECL;
};
  
/*******************************  ctors  ************************************/
CUDADEV adouble::adouble() {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
}

CUDADEV adouble::adouble(const double v) : val(v) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I = 0.0;
}

CUDADEV adouble::adouble(const double v, ADVAL_TYPE_ADV) : val(v) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=ADV_I;
}

CUDADEV adouble::adouble(const adouble& a) : val(a.val) {
#if defined(DYNAMIC_DIRECTIONS)
    adval = new double[ADOLC_numDir];
#endif
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=a.ADVAL_I;
}

/*******************************  dtors  ************************************/
#if defined(DYNAMIC_DIRECTIONS)
CUDADEV adouble::~adouble() {
    delete[] adval;
}
#endif

/*************************  temporary results  ******************************/
// sign
CUDADEV adouble adouble::operator - () const {
    adouble tmp;
    tmp.val=-val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=-ADVAL_I;
    return tmp;
}

CUDADEV adouble adouble::operator + () const {
    return *this;
}

// addition
CUDADEV adouble adouble::operator + (const double v) const {
    return adouble(val+v, adval);
}

CUDADEV adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
    tmp.val=val+a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I+a.ADVAL_I;
    return tmp;
}

CUDADEV adouble operator + (const double v, const adouble& a) {
    return adouble(v+a.val, a.adval);
}

// subtraction
CUDADEV adouble adouble::operator - (const double v) const {
    return adouble(val-v, adval);
}

CUDADEV adouble adouble::operator - (const adouble& a) const {
    adouble tmp;
    tmp.val=val-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I-a.ADVAL_I;
    return tmp;
}

CUDADEV adouble operator - (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=-a.ADVAL_I;
    return tmp;
}

// multiplication
CUDADEV adouble adouble::operator * (const double v) const {
    adouble tmp;
    tmp.val=val*v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I*v;
    return tmp;
}

CUDADEV adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
    tmp.val=val*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    return tmp;
}

CUDADEV adouble operator * (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=v*a.ADVAL_I;
    return tmp;
}

// division
CUDADEV adouble adouble::operator / (const double v) const {
    adouble tmp;
    tmp.val=val/v;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I/v;
    return tmp;
}

CUDADEV adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
    tmp.val=val/a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    return tmp;
}

CUDADEV adouble operator / (const double v, const adouble& a) {
    adouble tmp;
    tmp.val=v/a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=(-v*a.ADVAL_I)/(a.val*a.val);
    return tmp;
}

// inc/dec
CUDADEV adouble adouble::operator ++ () {
    ++val;
    return *this;
}

CUDADEV adouble adouble::operator ++ (int) {
    adouble tmp;
    tmp.val=val++;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I;
    return tmp;
}

CUDADEV adouble adouble::operator -- () {
    --val;
    return *this;
}

CUDADEV adouble adouble::operator -- (int) {
    adouble tmp;
    tmp.val=val--;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=ADVAL_I;
    return tmp;
}

// functions
CUDADEV adouble tan(const adouble& a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::tan(a.val);
    tmp2=ADOLC_MATH_NSP::cos(a.val);
    tmp2*=tmp2;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

CUDADEV adouble exp(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::exp(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp.val*a.ADVAL_I;
    return tmp;
}

CUDADEV adouble log(const adouble &a) {
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

CUDADEV adouble sqrt(const adouble &a) {
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

CUDADEV adouble sin(const adouble &a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::sin(a.val);
    tmp2=ADOLC_MATH_NSP::cos(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

CUDADEV adouble cos(const adouble &a) {
    adouble tmp;
    double tmp2;
    tmp.val=ADOLC_MATH_NSP::cos(a.val);
    tmp2=-ADOLC_MATH_NSP::sin(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

CUDADEV adouble asin(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::asin(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

CUDADEV adouble acos(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::acos(a.val);
    double tmp2=-ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

CUDADEV adouble atan(const adouble &a) {
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

CUDADEV adouble atan2(const adouble &a, const adouble &b) {
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

CUDADEV adouble pow(const adouble &a, double v) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(a.val, v);
    double tmp2=v*ADOLC_MATH_NSP::pow(a.val, v-1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

CUDADEV adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(a.val, b.val);
    double tmp2=b.val*ADOLC_MATH_NSP::pow(a.val, b.val-1);
    double tmp3=ADOLC_MATH_NSP::log(a.val)*tmp.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I+tmp3*b.ADVAL_I;
    return tmp;
}

CUDADEV adouble pow(double v, const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::pow(v, a.val);
    double tmp2=tmp.val*ADOLC_MATH_NSP::log(v);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    return tmp;
}

CUDADEV adouble log10(const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::log10(a.val);
    double tmp2=ADOLC_MATH_NSP::log((double)10)*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

CUDADEV adouble sinh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::sinh(a.val);
    double tmp2=ADOLC_MATH_NSP::cosh(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    return tmp;
}

CUDADEV adouble cosh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::cosh(a.val);
    double tmp2=ADOLC_MATH_NSP::sinh(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    return tmp;
}

CUDADEV adouble tanh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::tanh(a.val);
    double tmp2=ADOLC_MATH_NSP::cosh(a.val);
    tmp2*=tmp2;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

#if defined(ATRIG_ERF)
CUDADEV adouble asinh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::asinh(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val+1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

CUDADEV adouble acosh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::acosh(a.val);
    double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val-1);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}

CUDADEV adouble atanh (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP_ERF::atanh(a.val);
    double tmp2=1-a.val*a.val;
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    return tmp;
}
#endif

CUDADEV adouble fabs (const adouble &a) {
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

CUDADEV adouble ceil (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::ceil(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=0.0;
    return tmp;
}

CUDADEV adouble floor (const adouble &a) {
    adouble tmp;
    tmp.val=ADOLC_MATH_NSP::floor(a.val);
    FOR_I_EQ_0_LT_NUMDIR
    tmp.ADVAL_I=0.0;
    return tmp;
}

CUDADEV adouble fmax (const adouble &a, const adouble &b) {
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

CUDADEV adouble fmax (double v, const adouble &a) {
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

CUDADEV adouble fmax (const adouble &a, double v) {
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

CUDADEV adouble fmin (const adouble &a, const adouble &b) {
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

CUDADEV adouble fmin (double v, const adouble &a) {
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

CUDADEV adouble fmin (const adouble &a, double v) {
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

CUDADEV adouble ldexp (const adouble &a, const adouble &b) {
    return a*pow(2.,b);
}

CUDADEV adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

CUDADEV adouble ldexp (const double v, const adouble &a) {
    return v*pow(2.,a);
}

CUDADEV double frexp (const adouble &a, int* v) {
    return ADOLC_MATH_NSP::frexp(a.val, v);
}

#if defined(ATRIG_ERF)
CUDADEV adouble erf (const adouble &a) {
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
CUDADEV void adouble::operator = (const double v) {
    val=v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=0.0;
}

CUDADEV void adouble::operator = (const adouble& a) {
    val=a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=a.ADVAL_I;
}

CUDADEV void adouble::operator += (const double v) {
    val+=v;
}

CUDADEV void adouble::operator += (const adouble& a) {
    val=val+a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I+=a.ADVAL_I;
}

CUDADEV void adouble::operator -= (const double v) {
    val-=v;
}

CUDADEV void adouble::operator -= (const adouble& a) {
    val=val-a.val;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I-=a.ADVAL_I;
}

CUDADEV void adouble::operator *= (const double v) {
    val=val*v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I*=v;
}

CUDADEV void adouble::operator *= (const adouble& a) {
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    val*=a.val;
}

CUDADEV void adouble::operator /= (const double v) {
    val/=v;
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I/=v;
}

CUDADEV void adouble::operator /= (const adouble& a) {
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    val=val/a.val;
}

// not
CUDADEV int adouble::operator ! () const {
    return val==0.0;
}

// comparision
CUDADEV int adouble::operator != (const adouble &a) const {
    return val!=a.val;
}

CUDADEV int adouble::operator != (const double v) const {
    return val!=v;
}

CUDADEV int operator != (const double v, const adouble &a) {
    return v!=a.val;
}

CUDADEV int adouble::operator == (const adouble &a) const {
    return val==a.val;
}

CUDADEV int adouble::operator == (const double v) const {
    return val==v;
}

CUDADEV int operator == (const double v, const adouble &a) {
    return v==a.val;
}

CUDADEV int adouble::operator <= (const adouble &a) const {
    return val<=a.val;
}

CUDADEV int adouble::operator <= (const double v) const {
    return val<=v;
}

CUDADEV int operator <= (const double v, const adouble &a) {
    return v<=a.val;
}

CUDADEV int adouble::operator >= (const adouble &a) const {
    return val>=a.val;
}

CUDADEV int adouble::operator >= (const double v) const {
    return val>=v;
}

CUDADEV int operator >= (const double v, const adouble &a) {
    return v>=a.val;
}

CUDADEV int adouble::operator >  (const adouble &a) const {
    return val>a.val;
}

CUDADEV int adouble::operator >  (const double v) const {
    return val>v;
}

CUDADEV int operator >  (const double v, const adouble &a) {
    return v>a.val;
}

CUDADEV int adouble::operator <  (const adouble &a) const {
    return val<a.val;
}

CUDADEV int adouble::operator <  (const double v) const {
    return val<v;
}

CUDADEV int operator <  (const double v, const adouble &a) {
    return v<a.val;
}

/*******************  getter / setter  **************************************/
CUDAHOSTDEV double adouble::getValue() const {
    return val;
}

CUDAHOSTDEV void adouble::setValue(const double v) {
    val=v;
}

CUDAHOSTDEV ADVAL_TYPE adouble::getADValue() const {
    return (ADVAL_TYPE)adval;
}

CUDAHOSTDEV void adouble::setADValue(ADVAL_TYPE v) {
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=V_I;
}

#  if defined(NUMBER_DIRECTIONS)
CUDAHOSTDEV double adouble::getADValue(const unsigned int p) const {
    unsigned int locp = p;
    if (locp>=ADOLC_numDir) {
	locp = ADOLC_numDir -1;
    }
    return adval[locp];
}

CUDAHOSTDEV void adouble::setADValue(const unsigned int p, const double v) {
    unsigned int locp = p;
    if (locp>=ADOLC_numDir) {
	locp = ADOLC_numDir - 1;
    }
    adval[locp]=v;
}
#  endif

#if defined(NUMBER_DIRECTIONS)
void setNumDir(const unsigned int p) {
#if !defined(DYNAMIC_DIRECTIONS)
    if (p>NUMBER_DIRECTIONS) ADOLC_numDir=NUMBER_DIRECTIONS;
    else ADOLC_numDir=p;
#else
    ADOLC_numDir = p;
#endif
}
#endif

/*******************  i/o operations  ***************************************/
CUDAHOST ostream& operator << ( ostream& out, const adouble& a) {
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

CUDAHOST istream& operator >> ( istream& in, adouble& a) {
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
