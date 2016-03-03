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
    friend ADOLC_DLL_EXPORT class pdouble;
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
    adubref( const adubref& ) {
        fprintf(DIAG_OUT,"ADOL-C error: illegal copy construction of adubref"
               " variable\n");
        exit(-2);
    }
    bool isInit;  // marker if the badouble is properly initialized
public:
    /* adub prevents postfix operators to occur on the left
       side of an assignment which would not work  */
#if !defined(SWIGPRE)
    adub operator++( int );
    adub operator--( int );
#else
    adub* operator++( int );
    adub* operator--( int );
#endif
    adubref& operator++( void );
    adubref& operator--( void );
    adubref& operator = ( double );
    adubref& operator = ( const badouble& );
    adubref& operator = ( const adubref& );
    adubref& operator = ( const pdouble& );
    adubref& operator +=  ( double );
    adubref& operator +=  ( const badouble& );
    adubref& operator +=  ( const pdouble& );
    adubref& operator -=  ( double x );
    adubref& operator -=  ( const badouble& );
    adubref& operator -=  ( const pdouble& );
    adubref& operator *=  ( double x );
    adubref& operator *=  ( const badouble& );
    adubref& operator *=  ( const pdouble& );
    inline adubref& operator /=  ( double x );
    inline adubref& operator /=  ( const badouble& );
    inline adubref& operator /=  ( const pdouble&);

    adubref& operator <<= ( double );
    void declareIndependent();
    adubref& operator >>= ( double& );
    void declareDependent();
    operator adub() const;
#if !defined(SWIGPRE)
    explicit operator adubref*() const;
#endif
    friend ADOLC_DLL_EXPORT void condassign(adubref&, const badouble&, const badouble&, const badouble&);
    friend ADOLC_DLL_EXPORT void condassign(adubref&, const badouble&, const badouble&);
    friend ADOLC_DLL_EXPORT void condeqassign(adubref&, const badouble&, const badouble&, const badouble&);
    friend ADOLC_DLL_EXPORT void condeqassign(adubref&, const badouble&, const badouble&);
    ~adubref();
};

/* adolc_vec_copy(dest,src,size); */
void ADOLC_DLL_EXPORT adolc_vec_copy(adouble *const, const adouble*const, locint);
/* adolc_vec_axpy(res,a,x,y,size); <=> res = a*x + y  */
void ADOLC_DLL_EXPORT adolc_vec_axpy(adouble *const, const badouble&, const adouble*const, const adouble*const, locint);

class advector {
private:
    struct ADOLC_DLL_EXPORT blocker {
	blocker() {}
	blocker(size_t n);
	~blocker() {}
    } blk;
    std::vector<adouble> data;
    ADOLC_DLL_EXPORT bool nondecreasing() const;
public:
    ADOLC_DLL_EXPORT advector() : blk(), data() {}
    ADOLC_DLL_EXPORT explicit advector(size_t n) : blk(n), data(n) {}
    ADOLC_DLL_EXPORT ~advector() {}
    ADOLC_DLL_EXPORT advector(const advector& x) : blk(x.size()), data(x.size()) {  adolc_vec_copy(data.data(),x.data.data(),x.size()); }
    // in the above copy we are sure of contiguous locations
    // but not so in the one below
    ADOLC_DLL_EXPORT advector(const std::vector<adouble>& v) : blk(v.size()), data(v) {}
    ADOLC_DLL_EXPORT size_t size() const { return data.size(); }
    ADOLC_DLL_EXPORT operator const std::vector<adouble>&() const { return data; }
    ADOLC_DLL_EXPORT operator std::vector<adouble>&() { return data; }
    ADOLC_DLL_EXPORT operator adouble*() { return data.data(); }
#if !defined(SWIGPRE)
    ADOLC_DLL_EXPORT adub operator[](const badouble& index) const;
    ADOLC_DLL_EXPORT adubref operator[](const badouble& index);
#else
    ADOLC_DLL_EXPORT adub* operator[](const badouble& index) const;
    ADOLC_DLL_EXPORT adubref* operator[](const badouble& index);
#endif
    ADOLC_DLL_EXPORT adouble& operator[](size_t i) { return data[i]; }
    ADOLC_DLL_EXPORT const adouble& operator[](size_t i) const { return data[i]; }
    ADOLC_DLL_EXPORT adouble lookupindex(const badouble& x, const badouble& y) const;
};

inline adubref& adubref::operator /= (double y) {
    *this *=  (1.0/y);
    return *this;
}


inline adubref& adubref::operator /= (const badouble& y) {
    *this *=  (1.0/y);
    return *this;
}

inline adubref& adubref::operator /= (const pdouble& p) {
    *this *= recipr(p); 
    return *this;
}
#endif /* TAPELESS */
#endif /* __cplusplus */
#endif /* ADOLC_ADVECTOR_H */
