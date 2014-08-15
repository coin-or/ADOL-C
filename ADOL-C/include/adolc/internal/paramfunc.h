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

#if defined(ADOLC_ADVANCED_BRANCHING)
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
    friend ADOLC_DLL_EXPORT adouble pow   ( const pdouble&, const badouble& );
    inline friend adub fmax ( const pdouble&, const badouble& );
    inline friend adub fmax ( const badouble&, const pdouble& );
    inline friend adub fmin ( const pdouble&, const badouble& );
    inline friend adub fmin ( const badouble&, const pdouble& );
#endif
#endif
