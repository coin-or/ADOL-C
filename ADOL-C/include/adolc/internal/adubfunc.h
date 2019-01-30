/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adubfunc.h
 Revision: $Id$
 Contents: operators and functions returning temporaries
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if defined(_IN_CLASS_) && _IN_CLASS_
#if defined(_IN_BADOUBLE_) || defined(_IN_ADUB_)
    /*--------------------------------------------------------------------------*/
    /* Comparison (friends) */
#if defined(ADOLC_ADVANCED_BRANCHING) && !defined(SWIGPRE)
    friend ADOLC_DLL_EXPORT adub operator != ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator == ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator <= ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator >= ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator >  ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator <  ( const badouble&, const badouble& );
#endif

#if !defined(SWIGPRE)
    /*--------------------------------------------------------------------------*/
    /* sign operators (friends) */
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble& x );
    friend ADOLC_DLL_EXPORT adub operator - ( const badouble& x );

    /*--------------------------------------------------------------------------*/
    /* binary operators (friends) */
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator + ( double, const badouble& );
    inline friend adub operator + ( const badouble&, double );
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
    friend ADOLC_DLL_EXPORT adub cbrt ( const badouble& );
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

    /* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
    friend ADOLC_DLL_EXPORT adub sinh  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub cosh  ( const badouble& );
    friend ADOLC_DLL_EXPORT adub tanh  ( const badouble& );
#if defined(ATRIG_ERF)
    friend ADOLC_DLL_EXPORT adub asinh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub acosh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub atanh ( const badouble& );
    friend ADOLC_DLL_EXPORT adub erf   ( const badouble& );
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

    /*--------------------------------------------------------------------------*/
#endif
#endif
#endif
