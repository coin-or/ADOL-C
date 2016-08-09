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

class pdouble;

ADOLC_DLL_EXPORT pdouble mkparam(double pval);
ADOLC_DLL_EXPORT pdouble getparam(locint index);
ADOLC_DLL_EXPORT locint mkparam_idx(double pval);

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
    friend pdouble mkparam(double pval);
    friend pdouble getparam(locint index);
    friend locint mkparam_idx(double pval);
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

/* unary operators (friends) */
inline adub exp  ( const pdouble& p)  { return exp(adub(p));   }
inline adub log  ( const pdouble& p)  { return log(adub(p));   }
inline adub sqrt ( const pdouble& p)  { return sqrt(adub(p));  }
inline adub sin  ( const pdouble& p)  { return sin(adub(p));   }
inline adub cos  ( const pdouble& p)  { return cos(adub(p));   }
inline adub tan  ( const pdouble& p)  { return tan(adub(p));   }
inline adub asin ( const pdouble& p)  { return asin(adub(p));  }
inline adub acos ( const pdouble& p)  { return acos(adub(p));  }
inline adub atan ( const pdouble& p)  { return atan(adub(p));  }

/*--------------------------------------------------------------------------*/
/* special operators (friends) */
/* no internal use of condassign: */
inline adub    pow   ( const pdouble& p, double q) { return pow(adub(p),q); }
inline adub    log10 ( const pdouble& p) { return log10(adub(p)); }

/* Additional ANSI C standard Math functions Added by DWJ on 8/6/90 */
inline adub sinh  ( const pdouble& p) { return sinh(adub(p)); }
inline adub cosh  ( const pdouble& p) { return cosh(adub(p)); }
inline adub tanh  ( const pdouble& p) { return tanh(adub(p)); }
#if defined(ATRIG_ERF)
inline adub asinh ( const pdouble& p) { return asinh(adub(p)); }
inline adub acosh ( const pdouble& p) { return acosh(adub(p)); }
inline adub atanh ( const pdouble& p) { return atanh(adub(p)); }
inline adub erf   ( const pdouble& p) { return erf(adub(p));   }
#endif

inline adub fabs  ( const pdouble& p) { return fabs(adub(p));  }
inline adub ceil  ( const pdouble& p) { return ceil(adub(p));  }
inline adub floor ( const pdouble& p) { return floor(adub(p)); }

inline adub fmax ( const pdouble& p, const pdouble& q) 
{ return fmax(adub(p),adub(q)); }
inline adub fmax ( double p, const pdouble& q)
{ return fmax(p,adub(q)); }
inline adub fmax ( const pdouble& p, double q)
{ return fmax(adub(p),q); }
inline adub fmin ( const pdouble& p, const pdouble& q)
{ return fmin(adub(p),adub(q)); }
inline adub fmin ( double p, const pdouble& q)
{ return fmin(p,adub(q)); }
inline adub fmin ( const pdouble& p, double q)
{ return fmin(adub(p),q); }

inline adub ldexp ( const pdouble& p, int n)
{ return ldexp(adub(p),n); }
inline adub frexp ( const pdouble& p, int* n)
{ return frexp(adub(p),n); }

/*--------------------------------------------------------------------------*/
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
