/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     paramfunc.h
 Revision: $Id$
 Contents: operators for parameter dependent functions
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if defined(_IN_CLASS_) && _IN_CLASS_
#if defined(_IN_BADOUBLE_) || defined(_IN_ADUB_) || defined(_IN_PDOUBLE_)

#if defined(ADOLC_ADVANCED_BRANCHING) && !defined(SWIGPRE)
    inline friend adub operator != ( const pdouble&, const badouble&);
    friend ADOLC_DLL_EXPORT adub operator != ( const badouble&, const pdouble&);
    inline friend adub operator == ( const pdouble&, const badouble&);
    friend ADOLC_DLL_EXPORT adub operator == ( const badouble&, const pdouble&);
    inline friend adub operator <= ( const pdouble&, const badouble&);
    friend ADOLC_DLL_EXPORT adub operator <= ( const badouble&, const pdouble&);
    inline friend adub operator >= ( const pdouble&, const badouble&);
    friend ADOLC_DLL_EXPORT adub operator >= ( const badouble&, const pdouble&);
    inline friend adub operator >  ( const pdouble&, const badouble&);
    friend ADOLC_DLL_EXPORT adub operator >  ( const badouble&, const pdouble&);
    inline friend adub operator <  ( const pdouble&, const badouble&);
    friend ADOLC_DLL_EXPORT adub operator <  ( const badouble&, const pdouble&);
#else
#if defined(_IN_BADOUBLE_) || defined(_IN_PDOUBLE_)
    inline friend int operator != ( const pdouble&, const badouble&);
    inline friend int operator != ( const badouble&, const pdouble&);
    inline friend int operator == ( const pdouble&, const badouble&);
    inline friend int operator == ( const badouble&, const pdouble&);
    inline friend int operator <= ( const pdouble&, const badouble&);
    inline friend int operator <= ( const badouble&, const pdouble&);
    inline friend int operator >= ( const pdouble&, const badouble&);
    inline friend int operator >= ( const badouble&, const pdouble&);
    inline friend int operator >  ( const pdouble&, const badouble&);
    inline friend int operator >  ( const badouble&, const pdouble&);
    inline friend int operator <  ( const pdouble&, const badouble&);
    inline friend int operator <  ( const badouble&, const pdouble&);
#endif
#endif

#if !defined(SWIGPRE)
    inline friend adub operator + ( const pdouble&, const badouble&);
    inline friend adub operator + ( const pdouble&, double);
    inline friend adub operator + ( double, const pdouble&);
    friend ADOLC_DLL_EXPORT adub operator + ( const badouble&, const pdouble&);
    friend ADOLC_DLL_EXPORT adub operator - ( const pdouble&);
    friend ADOLC_DLL_EXPORT adub operator - ( const badouble&, const pdouble&);
    inline friend adub operator - ( const pdouble&, double );
    inline friend adub operator - ( double, const pdouble& );
    inline friend adub operator - ( const pdouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub operator * ( const badouble&, const pdouble&);
    inline friend adub operator * ( const pdouble&, const badouble& );
    inline friend adub operator * ( const pdouble&, double );
    inline friend adub operator * ( double, const pdouble& );
    friend ADOLC_DLL_EXPORT adub recipr( const pdouble& );
    inline friend adub operator / ( const badouble&, const pdouble& );
    inline friend adub operator / ( double, const pdouble& );
    inline friend adub operator / ( const pdouble&, double );
    friend ADOLC_DLL_EXPORT adub operator / ( const pdouble&, const badouble& );
    friend ADOLC_DLL_EXPORT adub    pow   ( const badouble&, const pdouble& );
    inline friend adub fmax ( const pdouble&, const badouble& );
    inline friend adub fmax ( const badouble&, const pdouble& );
    inline friend adub fmin ( const pdouble&, const badouble& );
    inline friend adub fmin ( const badouble&, const pdouble& );
    /*--------------------------------------------------------------------------*/
    /* unary operators (friends) */
    inline friend ADOLC_DLL_EXPORT adub exp  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub log  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub sqrt ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub cbrt ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub sin  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub cos  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub tan  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub asin ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub acos ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub atan ( const pdouble& );

    /*--------------------------------------------------------------------------*/
    /* special operators (friends) */
    /* no internal use of condassign: */
    inline friend ADOLC_DLL_EXPORT adub    pow   ( const pdouble&, double );
    inline friend ADOLC_DLL_EXPORT adub    log10 ( const pdouble& );

    /* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
    inline friend ADOLC_DLL_EXPORT adub sinh  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub cosh  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub tanh  ( const pdouble& );
#if defined(ATRIG_ERF)
    inline friend ADOLC_DLL_EXPORT adub asinh ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub acosh ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub atanh ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub erf   ( const pdouble& );
#endif

    inline friend ADOLC_DLL_EXPORT adub fabs  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub ceil  ( const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub floor ( const pdouble& );

    inline friend ADOLC_DLL_EXPORT adub fmax ( const pdouble&, const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub fmax ( double, const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub fmax ( const pdouble&, double );
    inline friend ADOLC_DLL_EXPORT adub fmin ( const pdouble&, const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub fmin ( double, const pdouble& );
    inline friend ADOLC_DLL_EXPORT adub fmin ( const pdouble&, double );

    inline friend ADOLC_DLL_EXPORT adub ldexp ( const pdouble&, int );
    inline friend ADOLC_DLL_EXPORT adub frexp ( const pdouble&, int* );

    /*--------------------------------------------------------------------------*/

#endif
    friend ADOLC_DLL_EXPORT adouble pow   ( const pdouble&, const badouble& );
#endif
#endif
