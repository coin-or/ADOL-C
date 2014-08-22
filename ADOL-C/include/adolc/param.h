/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     param.h
 Revision: $Id$
 Contents: class for parameter dependent functions
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_PARAM_H)
#define ADOLC_PARAM_H 1
#if defined(__cplusplus)

#include <cstdio>
#include <stdexcept>

using std::logic_error;

class ADOLC_DLL_EXPORT pdouble {
    friend ADOLC_DLL_EXPORT class badouble;
    friend ADOLC_DLL_EXPORT class adub;
    friend ADOLC_DLL_EXPORT class adouble;
    friend ADOLC_DLL_EXPORT class adubref;
protected:
    double _val;
    locint _idx;
    pdouble(const pdouble&) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal copy construction of pdouble"
                " variable\n          ... pdouble objects must never be copied\n");
        throw logic_error("illegal constructor call, errorcode=-2");
    }
    pdouble(void) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal default construction of pdouble"
                " variable\n");
        throw logic_error("illegal constructor call, errorcode=-2");
    }
    pdouble(double pval);
    pdouble(locint index);
public:
    friend ADOLC_DLL_EXPORT pdouble mkparam(double pval);
    friend ADOLC_DLL_EXPORT pdouble getparam(locint index);
    friend ADOLC_DLL_EXPORT locint mkparam_idx(double pval);
    operator adub() const;

#define _IN_CLASS_ 1
#define _IN_PDOUBLE_ 1
#include <adolc/internal/paramfunc.h>
#undef _IN_PDOUBLE_
#undef _IN_CLASS_

    ~pdouble() {}
};

#ifdef ADOLC_ADVANCED_BRANCHING
inline adub operator != ( const pdouble& a, const badouble& b)
{ return (b != a); }
inline adub operator == ( const pdouble& a, const badouble& b)
{ return (b == a); }
inline adub operator <= ( const pdouble& a, const badouble& b)
{ return (b >= a); }
inline adub operator <= ( const pdouble& a, const badouble& b)
{ return (b >= a); }
inline adub operator <= ( const pdouble& a, const badouble& b)
{ return (b >= a); }
inline adub operator >= ( const pdouble& a, const badouble& b)
{ return (b <= a); }
inline adub operator >  ( const pdouble& a, const badouble& b)
{ return (b < a); }
inline adub operator <  ( const pdouble& a, const badouble& b)
{ return (b > a); }
#else
inline int operator != ( const badouble& a, const pdouble& b) 
{ return ((a - b) != 0); }
inline int operator == ( const badouble& a, const pdouble& b) 
{ return ((a - b) == 0); }
inline int operator <= ( const badouble& a, const pdouble& b)
{ return ((a - b) <= 0); }
inline int operator >= ( const badouble& a, const pdouble& b)
{ return ((a - b) >= 0); }
inline int operator >  ( const badouble& a, const pdouble& b)
{ return ((a - b) > 0); }
inline int operator <  ( const badouble& a, const pdouble& b)
{ return ((a - b) < 0); }
inline int operator != ( const pdouble& a, const badouble& b)
{ return (b != a); }
inline int operator == ( const pdouble& a, const badouble& b)
{ return (b == a); }
inline int operator <= ( const pdouble& a, const badouble& b)
{ return (b >= a); }
inline int operator >= ( const pdouble& a, const badouble& b)
{ return (b <= a); }
inline int operator >  ( const pdouble& a, const badouble& b)
{ return (b < a); }
inline int operator <  ( const pdouble& a, const badouble& b)
{ return (b > a); }
#endif

inline adub operator + ( const pdouble& a, const badouble& b)
{ return (b + a); }

inline adub operator + ( const pdouble& a, double b)
{ return (b + adub(a)); }

inline adub operator + ( double a, const pdouble& b)
{ return (a + adub(b)); }

inline adub operator - ( const pdouble& a, const badouble& b)
{ return ((-b) + a); }

inline adub operator - ( const pdouble& a, double b)
{ return (adub(a) - b); }

inline adub operator - ( double a, const pdouble& b)
{ return (a + (-b)); }

inline adub operator * ( const pdouble& a, const badouble& b)
{ return (b*a); }

inline adub operator * ( const pdouble& a, double b)
{ return (b * adub(a)); }

inline adub operator * ( double a, const pdouble& b)
{ return (a * adub(b)); }

inline adub operator / ( const badouble& a, const pdouble& b)
{ return (a*recipr(b)); }

inline adub operator / ( double a, const pdouble& b)
{ return (a*recipr(b)); }

inline adub operator / ( const pdouble& a, double b)
{ return (adub(a)/b); }

inline adub fmax ( const badouble& y, const pdouble& d ) {
    return (-fmin(-d,-y));
}

inline adub fmax ( const pdouble& a, const badouble& b)
{ return fmax(b,a); }

inline adub fmin ( const pdouble& a, const badouble& b)
{ return fmin(b,a); }

inline adub fmin( const badouble& a, const pdouble& b)
{ return fmin(a,adub(b)); }

#endif

BEGIN_C_DECLS

/****************************************************************************/
/* Returns the number of parameters recorded on tape                        */
/****************************************************************************/
ADOLC_DLL_EXPORT size_t get_num_param(short tag);

/****************************************************************************/
/* Overrides the parameters for the next evaluations. This will invalidate  */
/* the taylor stack, so next reverse call will fail, if not preceeded by a  */
/* forward call after setting the parameters.                               */
/****************************************************************************/
ADOLC_DLL_EXPORT void set_param_vec(short tag, size_t numparam, revreal* paramvec);

END_C_DECLS
#endif
