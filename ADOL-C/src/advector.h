/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: advector.h contains a vector<adouble> implementation
           that is able to trace subscripting operations.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#if !defined(ADOLC_ADVECTOR_H)
#define ADOLC_ADVECTOR_H 1

/****************************************************************************/
/*                                                         THIS FILE IS C++ */
#ifdef __cplusplus
#include <vector>

#include <adolc/adouble.h>

/****************************************************************************/
/*                                           THIS IS ONLY FOR TAPED VERSION */
#if !defined(TAPELESS)

class advector;
class adubref;

class ADOLC_DLL_EXPORT adubref {
    /* This class is supposed to be used only when an advector subscript
     * occurs as an lvalue somewhere. What we need to do is read the location
     * of the referenced adouble out of store[location] and perform 
     * operations with this refloc. This means that the tape needs new
     * opcodes (ref_assign_* /ref_eq_* / ref_{incr,decr}_a) for each of 
     * these operations, most of the code  will simply be copied from 
     * adouble class, since the operation is really the same except for 
     * the part where the refloc is read from store[location].
     * Reverse mode is also straightforward the same way.
     *
     * Convert to a new adub as soon as used as rvalue, this is why adubref
     * is not a child of badouble, since it should never occur as rvalue.
     */
    friend ADOLC_DLL_EXPORT class adub;
    friend ADOLC_DLL_EXPORT class advector;
protected:
    locint location;
    locint refloc;
    explicit adubref( locint lo, locint ref );
    explicit adubref( void ) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal default construction of adubref"
                " variable\n");
        exit(-2);
    }
    explicit adubref( double ) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal  construction of adubref"
		" variable from double\n");
        exit(-2);
    }
    explicit adubref( const badouble& ) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal  construction of adubref"
		" variable from badouble\n");
        exit(-2);
    }
    explicit adubref( const adub& ) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal  construction of adubref"
		" variable from adub\n");
        exit(-2);
    }
public:
    /* adub prevents postfix operators to occur on the left
       side of an assignment which would not work  */
    adub operator++( int );
    adub operator--( int );
    adubref& operator++( void );
    adubref& operator--( void );
    adubref& operator = ( double );
    adubref& operator = ( const badouble& );
    adubref& operator = ( const adubref& );
    adubref& operator +=  ( double );
    adubref& operator +=  ( const badouble& );
    adubref& operator -=  ( double x );
    adubref& operator -=  ( const badouble& );
    adubref& operator *=  ( double x );
    adubref& operator *=  ( const badouble& );
    adubref& operator /=  ( double x );
    adubref& operator /=  ( const badouble& );
    adubref& operator <<= ( double );
    void declareIndependent();
    adubref& operator >>= ( double& );
    void declareDependent();
    operator adub() const;
    friend ADOLC_DLL_EXPORT void condassign(adubref, const badouble&, const badouble&, const badouble&);
    friend ADOLC_DLL_EXPORT void condassign(adubref, const badouble&, const badouble&);
};

class ADOLC_DLL_EXPORT advector {
private:
    std::vector<adouble> data;
    bool nondecreasing() const;
public:
    advector() : data() {}
    explicit advector(size_t n) : data(n) {}
    ~advector() {}
    advector(const advector& x) : data(x.data) {}
    size_t size() const { return data.size(); }
    operator const vector<adouble>&() const { return data; }
    operator vector<adouble>&() { return data; }
    adub operator[](const badouble& index) const;
    adubref operator[](const badouble& index);
    adouble& operator[](size_t i) { return data[i]; }
    const adouble& operator[](size_t i) const { return data[i]; }
    adouble lookupindex(const badouble& x, const badouble& y) const;
};

#endif /* TAPELESS */
#endif /* __cplusplus */
#endif /* ADOLC_ADVECTOR_H */
