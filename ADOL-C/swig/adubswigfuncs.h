/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adubswigfuncs.h
 Revision: $Id$
 Contents: Provides all C/C++ interfaces of ADOL-C.
           NOTICE: ALL C/C++ headers will be included DEPENDING ON 
           whether the source code is plain C or C/C++ code. 
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/
#ifndef ADUBSWIGFUNCS_H
#define ADUBSWIGFUNCS_H

#include <adolc/adouble.h>

#if defined(SWIG)

/*--------------------------------------------------------------------------*/
/* Comparison */
#if defined(ADOLC_ADVANCED_BRANCHING)
adub* operator != ( const badouble&, const badouble& );
adub* operator == ( const badouble&, const badouble& );
adub* operator <= ( const badouble&, const badouble& );
adub* operator >= ( const badouble&, const badouble& );
adub* operator >  ( const badouble&, const badouble& );
adub* operator <  ( const badouble&, const badouble& );
adub* operator != ( const pdouble&, const badouble&);
adub* operator != ( const badouble&, const pdouble&);
adub* operator == ( const pdouble&, const badouble&);
adub* operator == ( const badouble&, const pdouble&);
adub* operator <= ( const pdouble&, const badouble&);
adub* operator <= ( const badouble&, const pdouble&);
adub* operator >= ( const pdouble&, const badouble&);
adub* operator >= ( const badouble&, const pdouble&);
adub* operator >  ( const pdouble&, const badouble&);
adub* operator >  ( const badouble&, const pdouble&);
adub* operator <  ( const pdouble&, const badouble&);
adub* operator <  ( const badouble&, const pdouble&);
#endif

/*--------------------------------------------------------------------------*/
/* sign operators  */
adub* operator + ( const badouble& x );
adub* operator - ( const badouble& x );
adub* operator - ( const pdouble&);

/*--------------------------------------------------------------------------*/
/* binary operators */
adub* operator + ( const badouble&, const badouble& );
adub* operator + ( double, const badouble& );
adub* operator + ( const badouble&, double );
adub* operator - ( const badouble&, const badouble& );
adub* operator - ( const badouble&, double );
adub* operator - ( double, const badouble& );
adub* operator * ( const badouble&, const badouble& );
adub* operator * ( double, const badouble& );
adub* operator * ( const badouble&, double );
adub* operator / ( const badouble&, double );
adub* operator / ( const badouble&, const badouble& );
adub* operator / ( double, const badouble& );
adub* operator + ( const pdouble&, const badouble&);
adub* operator + ( const pdouble&, double);
adub* operator + ( double, const pdouble&);
adub* operator + ( const badouble&, const pdouble&);
adub* operator - ( const badouble&, const pdouble&);
adub* operator - ( const pdouble&, double );
adub* operator - ( double, const pdouble& );
adub* operator - ( const pdouble&, const badouble& );
adub* operator * ( const badouble&, const pdouble&);
adub* operator * ( const pdouble&, const badouble& );
adub* operator * ( const pdouble&, double );
adub* operator * ( double, const pdouble& );
adub* recipr( const pdouble& );
adub* operator / ( const badouble&, const pdouble& );
adub* operator / ( double, const pdouble& );
adub* operator / ( const pdouble&, double );
adub* operator / ( const pdouble&, const badouble& );

/*--------------------------------------------------------------------------*/
/* unary operators */
adub* exp  ( const badouble& );
adub* log  ( const badouble& );
adub* sqrt ( const badouble& );
adub* sin  ( const badouble& );
adub* cos  ( const badouble& );
adub* tan  ( const badouble& );
adub* asin ( const badouble& );
adub* acos ( const badouble& );
adub* atan ( const badouble& );

/*--------------------------------------------------------------------------*/
/* special operators (friends) */
/* no internal use of condassign: */
adub*    pow   ( const badouble&, double );
adub*    log10 ( const badouble& );
adub*    pow   ( const badouble&, const pdouble& );

/* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
adub* sinh  ( const badouble& );
adub* cosh  ( const badouble& );
adub* tanh  ( const badouble& );
#if defined(ATRIG_ERF)
adub* asinh ( const badouble& );
adub* acosh ( const badouble& );
adub* atanh ( const badouble& );
adub* erf   ( const badouble& );
#endif

adub* fabs  ( const badouble& );
adub* ceil  ( const badouble& );
adub* floor ( const badouble& );

adub* fmax ( const badouble&, const badouble& );
adub* fmax ( double, const badouble& );
adub* fmax ( const badouble&, double );
adub* fmin ( const badouble&, const badouble& );
adub* fmin ( double, const badouble& );
adub* fmin ( const badouble&, double );
adub* fmax ( const pdouble&, const badouble& );
adub* fmax ( const badouble&, const pdouble& );
adub* fmin ( const pdouble&, const badouble& );
adub* fmin ( const badouble&, const pdouble& );

adub* ldexp ( const badouble&, int );
adub* frexp ( const badouble&, int* );

adub* exp  ( const pdouble& );
adub* log  ( const pdouble& );
adub* sqrt ( const pdouble& );
adub* sin  ( const pdouble& );
adub* cos  ( const pdouble& );
adub* tan  ( const pdouble& );
adub* asin ( const pdouble& );
adub* acos ( const pdouble& );
adub* atan ( const pdouble& );

adub*    pow   ( const pdouble&, double );
adub*    log10 ( const pdouble& );

adub* sinh  ( const pdouble& );
adub* cosh  ( const pdouble& );
adub* tanh  ( const pdouble& );
#if defined(ATRIG_ERF)
adub* asinh ( const pdouble& );
adub* acosh ( const pdouble& );
adub* atanh ( const pdouble& );
adub* erf   ( const pdouble& );
#endif

adub* fabs  ( const pdouble& );
adub* ceil  ( const pdouble& );
adub* floor ( const pdouble& );

adub* fmax ( const pdouble&, const pdouble& );
adub* fmax ( double, const pdouble& );
adub* fmax ( const pdouble&, double );
adub* fmin ( const pdouble&, const pdouble& );
adub* fmin ( double, const pdouble& );
adub* fmin ( const pdouble&, double );

adub* ldexp ( const pdouble&, int );
adub* frexp ( const pdouble&, int* );

/*--------------------------------------------------------------------------*/
adub* adolc_vec_dot(const adouble*const, const adouble*const, locint);

pdouble* mkparam(double pval);
pdouble* getparam(locint index);

#endif
#endif
