/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adouble_tl.cpp contains that definitions of procedures used to
           define various tapeless adouble operations.
           These operations actually have two purposes.
           The first purpose is to actual compute the function, just as
           the same code written for double precision (single precision -
           complex - interval) arithmetic would.  The second purpose is
           to compute directional derivatives in forward mode of 
	   automatic differentiation.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adtl.h>
#include <cmath>
#include <iostream>
#include <limits>

namespace adtl {

enum ModeMask {
    ADTL_Z_MASK = 0x1,
    ADTL_F_MASK = 0x2,
    ADTL_I_MASK = 0x4
};

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

bool adouble::_do_val() {
    return (forward_mode & ADTL_Z_MASK == ADTL_Z_MASK);
}
#define do_val() likely(adouble::_do_val())
#define no_do_val() unlikely(!adouble::_do_val())

bool adouble::_do_adval() {
    return (forward_mode & ADTL_F_MASK == ADTL_F_MASK);
}
#define do_adval() likely(adouble::_do_adval())
#define no_do_adval() unlikely(!adouble::_do_adval())

bool adouble::_do_indo() {
    return (forward_mode & ADTL_I_MASK == ADTL_I_MASK);
}
#define do_indo() likely(adouble::_do_indo())
#define no_do_indo() unlikely(!adouble::_do_indo())

size_t adouble::numDir = 1;
enum Mode adouble::forward_mode = ADTL_FOV;

size_t refcounter::refcnt = 0;

void setNumDir(const size_t p) {
    if (refcounter::refcnt > 0) {
	fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting numDir will not change the number of\n directional derivative in existing adoubles and may lead to erronious results\n or memory corruption\n Number of currently existing adoubles = %z\n", refcounter::refcnt);
    }
    if (p < 1) {
	fprintf(DIAG_OUT, "ADOL-C Error: Tapeless: You are being a moron now.\n");
	abort();
    }
    adouble::numDir = p;
}

void setMode(enum Mode newmode) {
    if (refcounter::refcnt > 0) {
	fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting mode will the change the mode of\n computation in previously computed variables and may lead to erronious results\n or memory corruption\n Number of currently exisiting adoubles = %z\n", refcounter::refcnt);
    }
    adouble::forward_mode = newmode;
}

double makeNaN() {
    return ADOLC_MATH_NSP::numeric_limits<double>::quiet_NaN();
}

double makeInf() {
    return ADOLC_MATH_NSP::numeric_limits<double>::infinity();
}

#define FOR_I_EQ_0_LT_NUMDIR for (int _i=0; _i < adouble::numDir; ++_i)
#define ADVAL_I              adval[_i]
#define ADV_I                adv[_i]
#define V_I                  v[_i]

/*******************************  ctors  ************************************/
adouble::adouble() : val(0), adval(NULL) {
    if (do_adval())
	adval = new double[adouble::numDir];
    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }
}

adouble::adouble(const double v) : val(v), adval(NULL) {
    if (do_adval()) {
	adval = new double[adouble::numDir];
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I = 0.0;
    }
    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }
}

adouble::adouble(const double v, const double* adv) : val(v), adval(NULL) {
    if (do_adval()) {
	adval = new double[adouble::numDir];
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=ADV_I;
    }
    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }
}

adouble::adouble(const adouble& a) : val(a.val), adval(NULL) {
    if (do_adval()) {
	adval = new double[adouble::numDir];
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=a.ADVAL_I;
    }
    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();

     add_to_pattern(a.get_pattern());
    }
}

/*******************************  dtors  ************************************/
adouble::~adouble() {
    if (adval != NULL)
	delete[] adval;
    if ( !pattern.empty() )
	pattern.clear();
}

/*************************  temporary results  ******************************/
// sign
adouble adouble::operator - () const {
    adouble tmp;
    if (do_val())
	tmp.val=-val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=-ADVAL_I;
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

adouble adouble::operator + () const {
    return *this;
}

// addition
adouble adouble::operator + (const double v) const {
    adouble tmp(val+v, adval);
    if (do_indo())
	tmp.add_to_pattern( get_pattern() ) ;
    return tmp;
}

adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
    if (do_val())
	tmp.val=val+a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I+a.ADVAL_I;
    if (do_indo()) {
	tmp.add_to_pattern( get_pattern()  );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

adouble operator + (const double v, const adouble& a) {
    adouble tmp(v+a.val, a.adval);
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// subtraction
adouble adouble::operator - (const double v) const {
    adouble tmp(val-v, adval);
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

adouble adouble::operator - (const adouble& a) const {
    adouble tmp;
    if (do_val())
	tmp.val=val-a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I-a.ADVAL_I;
    if (do_indo()) {
	tmp.add_to_pattern( get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

adouble operator - (const double v, const adouble& a) {
    adouble tmp;
    if (do_val())
	tmp.val=v-a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=-a.ADVAL_I;
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// multiplication
adouble adouble::operator * (const double v) const {
    adouble tmp;
    if (do_val())
	tmp.val=val*v;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I*v;
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val())
	tmp.val=val*a.val;
    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    if (do_indo()) {
	tmp.add_to_pattern(   get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

adouble operator * (const double v, const adouble& a) {
    adouble tmp;
    if (do_val())
	tmp.val=v*a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=v*a.ADVAL_I;
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// division
adouble adouble::operator / (const double v) const {
    adouble tmp;
    if (do_val())
	tmp.val=val/v;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I/v;
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val())
	tmp.val=val/a.val;
    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    if (do_indo()) {
	tmp.add_to_pattern(   get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

adouble operator / (const double v, const adouble& a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val())
	tmp.val=v/a.val;
    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=(-v*a.ADVAL_I)/(a.val*a.val);
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// inc/dec
adouble adouble::operator ++ () {
    if (do_val())
	++val;
    return *this;
}

adouble adouble::operator ++ (int) {
    adouble tmp;
    if (do_val())
	tmp.val=val++;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I;
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

adouble adouble::operator -- () {
    if (do_val())
	--val;
    return *this;
}

adouble adouble::operator -- (int) {
    adouble tmp;
    if (do_val())
	tmp.val=val--;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I;
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

// functions
adouble tan(const adouble& a) {
    adouble tmp;
    double tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::tan(a.val);    
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	tmp2=ADOLC_MATH_NSP::cos(a.val);
	tmp2*=tmp2;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble exp(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::exp(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) 
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp.val*a.ADVAL_I;
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble log(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::log(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	FOR_I_EQ_0_LT_NUMDIR
	    if (a.val>0) tmp.ADVAL_I=a.ADVAL_I/a.val;
	    else if (a.val==0 && a.ADVAL_I != 0.0) {
		int sign = (a.ADVAL_I < 0)  ? -1 : 1;
		tmp.ADVAL_I=sign*makeInf();
	    } else tmp.ADVAL_I=makeNaN();
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble sqrt(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::sqrt(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	FOR_I_EQ_0_LT_NUMDIR
	    if (a.val>0) tmp.ADVAL_I=a.ADVAL_I/(tmp.val*2);
	    else if (a.val==0.0 && a.ADVAL_I != 0.0) {
		int sign = (a.ADVAL_I < 0) ? -1 : 1;
		tmp.ADVAL_I=sign * makeInf();
	    } else tmp.ADVAL_I=makeNaN();
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble sin(const adouble &a) {
    adouble tmp;
    double tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::sin(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	tmp2=ADOLC_MATH_NSP::cos(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble cos(const adouble &a) {
    adouble tmp;
    double tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::cos(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	tmp2=-ADOLC_MATH_NSP::sin(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble asin(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::asin(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble acos(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::acos(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=-ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble atan(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::atan(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=1+a.val*a.val;
	tmp2=1/tmp2;
	if (tmp2!=0)
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=a.ADVAL_I*tmp2;
	else
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=0.0;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble atan2(const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::atan2(a.val, b.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=a.val*a.val;
	double tmp3=b.val*b.val;
	double tmp4=tmp3/(tmp2+tmp3);
	if (tmp4!=0)
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=(a.ADVAL_I*b.val-a.val*b.ADVAL_I)/tmp3*tmp4;
	else
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=0.0;
    }
    if (do_indo()) {
	tmp.add_to_pattern( a.get_pattern() );
	tmp.add_to_pattern( b.get_pattern() );
    }
    return tmp;
}

adouble pow(const adouble &a, double v) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::pow(a.val, v);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=v*ADOLC_MATH_NSP::pow(a.val, v-1);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::pow(a.val, b.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=b.val*ADOLC_MATH_NSP::pow(a.val, b.val-1);
	double tmp3=ADOLC_MATH_NSP::log(a.val)*tmp.val;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I+tmp3*b.ADVAL_I;
    }
    if (do_indo()) {
	tmp.add_to_pattern( a.get_pattern() );
	tmp.add_to_pattern( b.get_pattern() );
    }
    return tmp;
}

adouble pow(double v, const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::pow(v, a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=tmp.val*ADOLC_MATH_NSP::log(v);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble log10(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::log10(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::log((double)10)*a.val;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble sinh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::sinh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::cosh(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble cosh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::cosh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sinh(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble tanh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::tanh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::cosh(a.val);
	tmp2*=tmp2;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

#if defined(ATRIG_ERF)
adouble asinh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::asinh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val+1);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble acosh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::acosh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val-1);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble atanh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::atanh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=1-a.val*a.val;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}
#endif

adouble fabs (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::fabs(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
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
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

adouble ceil (const adouble &a) {
    adouble tmp;
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::ceil(a.val);
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=0.0;
    return tmp;
}

adouble floor (const adouble &a) {
    adouble tmp;
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::floor(a.val);
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=0.0;
    return tmp;
}

adouble fmax (const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    double tmp2=a.val-b.val;
    if (tmp2<0) {
	if (do_val()) 
	    tmp.val=b.val;
	if (do_adval())
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=b.ADVAL_I;
	if (do_indo()) 
	    tmp.add_to_pattern( b.get_pattern() );
    } else {
	if (do_val()) 
	    tmp.val=a.val;
        if (tmp2>0) {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		    tmp.ADVAL_I=a.ADVAL_I;
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
        } else {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		{
		    if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=b.ADVAL_I;
		    else tmp.ADVAL_I=a.ADVAL_I;
                }
	    if (do_indo()) {
		tmp.add_to_pattern( a.get_pattern() );
		tmp.add_to_pattern( b.get_pattern() );
	    }
	}
    }
    return tmp;
}

adouble fmax (double v, const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    double tmp2=v-a.val;
    if (tmp2<0) {
	if (do_val()) 
	    tmp.val=a.val;
	if (do_adval())
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=a.ADVAL_I;
	if (do_indo()) 
	    tmp.add_to_pattern( a.get_pattern() );
    } else {
	if (do_val()) 
	    tmp.val=v;
        if (tmp2>0) {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		    tmp.ADVAL_I=0.0;
        } else {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		{
		    if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
		    else tmp.ADVAL_I=0.0;
                }
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
	}
    }
    return tmp;
}

adouble fmax (const adouble &a, double v) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    double tmp2=a.val-v;
    if (tmp2<0) {
	if (do_val()) 
	    tmp.val=v;
	if (do_adval())
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=0.0;
    } else {
	if (do_val()) 
	    tmp.val=a.val;
        if (tmp2>0) {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		    tmp.ADVAL_I=a.ADVAL_I;
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
        } else {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		{
		    if (a.ADVAL_I>0) tmp.ADVAL_I=a.ADVAL_I;
		    else tmp.ADVAL_I=0.0;
                }
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
	}
    }
    return tmp;
}

adouble fmin (const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    double tmp2=a.val-b.val;
    if (tmp2<0) {
	if (do_val()) 
	    tmp.val=a.val;
	if (do_adval())
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=a.ADVAL_I;
	if (do_indo()) 
	    tmp.add_to_pattern( a.get_pattern() );
    } else {
	if (do_val()) 
	    tmp.val=b.val;
        if (tmp2>0) {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		    tmp.ADVAL_I=b.ADVAL_I;
	    if (do_indo()) 
		tmp.add_to_pattern( b.get_pattern() );
        } else {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		{
		    if (a.ADVAL_I<b.ADVAL_I) tmp.ADVAL_I=a.ADVAL_I;
		    else tmp.ADVAL_I=b.ADVAL_I;
                }
	    if (do_indo()) {
		tmp.add_to_pattern( a.get_pattern() );
		tmp.add_to_pattern( b.get_pattern() );

	    }
	}
    }
    return tmp;
}

adouble fmin (double v, const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    double tmp2=v-a.val;
    if (tmp2<0) {
	if (do_val()) 
	    tmp.val=v;
	if (do_adval())
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=0.0;
    } else {
	if (do_val()) 
	    tmp.val=a.val;
        if (tmp2>0) {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		    tmp.ADVAL_I=a.ADVAL_I;
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
        } else {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		{
		    if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
		    else tmp.ADVAL_I=0.0;
                }
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
	}
    }
    return tmp;
}

adouble fmin (const adouble &a, double v) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    double tmp2=a.val-v;
    if (tmp2<0) {
	if (do_val()) 
	    tmp.val=a.val;
	if (do_adval())
	    FOR_I_EQ_0_LT_NUMDIR
		tmp.ADVAL_I=a.ADVAL_I;
	if (do_indo()) 
	    tmp.add_to_pattern( a.get_pattern() );
    } else {
	if (do_val()) 
	    tmp.val=v;
        if (tmp2>0) {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		    tmp.ADVAL_I=0.0;
        } else {
	    if (do_adval())
		FOR_I_EQ_0_LT_NUMDIR
		{
		    if (a.ADVAL_I<0) tmp.ADVAL_I=a.ADVAL_I;
		    else tmp.ADVAL_I=0.0;
                }
	    if (do_indo()) 
		tmp.add_to_pattern( a.get_pattern() );
	}
    }
    return tmp;
}

adouble ldexp (const adouble &a, const adouble &b) {
    adouble tmp = a*pow(2.,b);
    if (do_indo()) {
	tmp.add_to_pattern( a.get_pattern() ) ;
	tmp.add_to_pattern( b.get_pattern() ) ;
    }
    return tmp;
}

adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

adouble ldexp (const double v, const adouble &a) {
    adouble tmp = v*pow(2.,a);
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() ) ;
    return tmp;
}

double frexp (const adouble &a, int* v) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }    
    return ADOLC_MATH_NSP::frexp(a.val, v);
}

#if defined(ATRIG_ERF)
adouble erf (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::erf(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2 = 2.0 /
	    ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) *
	    ADOLC_MATH_NSP_ERF::exp(-a.val*a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() ) ;
    return tmp;
}
#endif

void condassign( adouble &res, const adouble &cond,
		 const adouble &arg1, const adouble &arg2 ) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) {
	if (cond.getValue() > 0) 
	    res = arg1;
	else
	    res = arg2;
    }
}

void condassign( adouble &res, const adouble &cond,
		 const adouble &arg ) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (do_val()) {
	if (cond.getValue() > 0) 
	    res = arg;
    }
}



/*******************  nontemporary results  *********************************/
void adouble::operator = (const double v) {
    if (do_val()) 
	val=v;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=0.0;
    if (do_indo())
	if (!pattern.empty()) pattern.clear();
}

void adouble::operator = (const adouble& a) {
    if (do_val()) 
	val=a.val;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=a.ADVAL_I;
    if (do_indo()) {
	if (!pattern.empty()) pattern.clear();
	add_to_pattern( a.get_pattern() );
    }
}

void adouble::operator += (const double v) {
    if (do_val()) 
	val+=v;
}

void adouble::operator += (const adouble& a) {
    if (do_val()) 
	val=val+a.val;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I+=a.ADVAL_I;
    if (do_indo()) 
	add_to_pattern( a.get_pattern() );
}

void adouble::operator -= (const double v) {
    if (do_val()) 
	val-=v;
}

void adouble::operator -= (const adouble& a) {
    if (do_val()) 
	val=val-a.val;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I-=a.ADVAL_I;
    if (do_indo()) 
	add_to_pattern( a.get_pattern() ) ;
}

void adouble::operator *= (const double v) {
    if (do_val()) 
	val=val*v;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I*=v;
}

void adouble::operator *= (const adouble& a) {
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (likely(adouble::_do_adval() && adouble::_do_val())) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    if (do_val()) 
	val*=a.val;
    if (do_indo()) 
	add_to_pattern( a.get_pattern() ) ;
}

void adouble::operator /= (const double v) {
    if (do_val()) 
	val/=v;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I/=v;
}

void adouble::operator /= (const adouble& a) {
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (likely(adouble::_do_adval() && adouble::_do_val())) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    if (do_val()) 
	val=val/a.val;
    if (do_indo()) 
	add_to_pattern( a.get_pattern() ) ;
}

// not
int adouble::operator ! () const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val==0.0;
}

// comparision
int adouble::operator != (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val!=a.val;
}

int adouble::operator != (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val!=v;
}

int operator != (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return v!=a.val;
}

int adouble::operator == (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val==a.val;
}

int adouble::operator == (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val==v;
}

int operator == (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return v==a.val;
}

int adouble::operator <= (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val<=a.val;
}

int adouble::operator <= (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val<=v;
}

int operator <= (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return v<=a.val;
}

int adouble::operator >= (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val>=a.val;
}

int adouble::operator >= (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val>=v;
}

int operator >= (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return v>=a.val;
}

int adouble::operator >  (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val>a.val;
}

int adouble::operator >  (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val>v;
}

int operator >  (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return v>a.val;
}

int adouble::operator <  (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val<a.val;
}

int adouble::operator <  (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val<v;
}

int operator <  (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return v<a.val;
}

/*******************  getter / setter  **************************************/
double adouble::getValue() const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return val;
}

void adouble::setValue(const double v) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    val=v;
}

const double *const adouble::getADValue() const {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return adval;
}

void adouble::setADValue(const double *const v) {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=V_I;
}

double adouble::getADValue(const unsigned int p) const {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (p>=adouble::numDir) 
    {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"getADValue(...)\"!!!\n");
        exit(-1);
    }
    return adval[p];
}

void adouble::setADValue(const unsigned int p, const double v) {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if (p>=adouble::numDir) 
    {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"setADValue(...)\"!!!\n");
        exit(-1);
    }
    adval[p]=v;
}

/*******************  i/o operations  ***************************************/
ostream& operator << ( ostream& out, const adouble& a) {
    if (likely(adouble::_do_val() && adouble::_do_adval())) {
	out << "Value: " << a.val;
	out << " ADValues (" << adouble::numDir << "): ";
	FOR_I_EQ_0_LT_NUMDIR
	    out << a.ADVAL_I << " ";
	out << "(a)";
    }
    return out;
}

istream& operator >> ( istream& in, adouble& a) {
    if(likely(adouble::_do_val() && adouble::_do_adval())) {
	char c;
	do in >> c;
	while (c!=':' && !in.eof());
	in >> a.val;
	unsigned int num;
	do in >> c;
	while (c!='(' && !in.eof());
	in >> num;
	if (num>adouble::numDir)
	{
	    cout << "ADOL-C error: to many directions in input\n";
	    exit(-1);
	}
	do in >> c;
	while (c!=':' && !in.eof());
	FOR_I_EQ_0_LT_NUMDIR
	    in >> a.ADVAL_I;
	do in >> c;
	while (c!=')' && !in.eof());
	return in;
    }
}

/**************** ADOLC_TRACELESS_SPARSE_PATTERN ****************************/
const list<unsigned int>& adouble::get_pattern() const {
    if (no_do_indo()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    return pattern;
}

void adouble::delete_pattern() {
    if (no_do_indo()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
    if ( !pattern.empty() )
	pattern.clear();
}

void adouble::add_to_pattern(const list<unsigned int>& v) {
    if (no_do_indo()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	exit(1);
    }
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

size_t adouble::get_pattern_size() const {
    if (no_do_indo()) {
     fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
     exit(1);
    }
    size_t s=0;
    if( !pattern.empty() )
      s = pattern.size();
    return s;
}


int ADOLC_Init_sparse_pattern(adouble *a, int n, unsigned int start_cnt) {
    for(unsigned int i=0; i < n; i++) {
	a[i].delete_pattern();
	a[i].pattern.push_back( i+start_cnt );
    }
    return 3;
}

int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat) {
    pat = (unsigned int**) malloc(m*sizeof(unsigned int*));
    for( int i=0; i < m ; i++){
	//const_cast<adouble&>(b[i]).pattern.sort();
	//const_cast<adouble&>(b[i]).pattern.unique();
      if ( b[i].get_pattern_size() > 0 ) {
         pat[i] = (unsigned int*) malloc(sizeof(unsigned int) * (b[i].get_pattern_size() +1) );
         pat[i][0] = b[i].get_pattern_size();
         const list<unsigned int>& tmp_set = b[i].get_pattern();
         list<unsigned int>::const_iterator it;
         unsigned int l=1;
         for(it = tmp_set.begin() ; it != tmp_set.end() ; it++,l++)
             pat[i][l] = *it;
       } else {
          pat[i] = (unsigned int*) malloc(sizeof(unsigned int));
          pat[i][0] =0;
       }
    }
    return 3;
}

}
