/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adtl.h contains that declaratins of procedures used to
           define various tapeless adouble operations.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert, Kshitij Kulshreshtha, Benjamin Jurgelucks

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_ADTL_HOV_H
#define ADOLC_ADTL_HOV_H

#include <ostream>
#include <adolc/internal/common.h>
#include <list>
#include <stdexcept>
#include <adolc/adalloc.h>  //for myalloc2

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define COMPILER_HAS_CXX11
#else
#error "please use -std=c++11 compiler flag with a C++11 compliant compiler"
#endif

using std::ostream;
using std::istream;
using std::list;
using std::logic_error;

namespace adtl_hov {

double makeNaN();
double makeInf();

enum Mode {
    ADTL_ZOS = 0x1,
    ADTL_FOV = 0x3,
    ADTL_INDO = 0x5,
    ADTL_FOV_INDO = 0x7,
	ADTL_HOV = 0x9  // = BIN(1 0 0 1)
};
// Mode

class adouble;

#ifdef USE_ADTL_REFCOUNTING

class refcounter {
private:
    ADOLC_DLL_EXPIMP static size_t refcnt;
    ADOLC_DLL_EXPORT friend void setNumDir(const size_t p);
    ADOLC_DLL_EXPORT friend void setDegree(const size_t p);
    ADOLC_DLL_EXPORT friend void setMode(enum Mode newmode);
    friend class adouble;
public:
    refcounter() { ++refcnt; }
    ~refcounter() { --refcnt; }
    inline static size_t getNumLiveVar() {return refcnt;}
};
#endif

class func_ad {
public:
    virtual int operator() (int n, adouble *x, int m, adouble *y) = 0;
};

class adouble {
public:
    inline adouble();
    inline adouble(const double v);
    inline adouble(const double v, const double* adv);
    inline adouble(const double v, const double** hov);
    inline adouble(const adouble& a);
    inline ~adouble();

    // sign
    inline adouble operator - () const;
    inline adouble operator + () const;

    // addition
    inline adouble operator + (const double v) const;
    inline adouble operator + (const adouble& a) const;
    inline friend
    adouble operator + (const double v, const adouble& a);

    // substraction
    inline adouble operator - (const double v) const;
    inline adouble operator - (const adouble& a) const;
    inline friend
    adouble operator - (const double v, const adouble& a);

    // multiplication
    inline adouble operator * (const double v) const;
    inline adouble operator * (const adouble& a) const;
    inline friend
    adouble operator * (const double v, const adouble& a);

    // division
    inline adouble operator / (const double v) const;
    inline adouble operator / (const adouble& a) const;
    inline friend
    adouble operator / (const double v, const adouble& a);

    // inc/dec
    inline adouble operator ++ ();
    inline adouble operator ++ (int);
    inline adouble operator -- ();
    inline adouble operator -- (int);

    // functions
    inline friend adouble tan(const adouble &a);
    inline friend adouble exp(const adouble &a);
    inline friend adouble log(const adouble &a);
    inline friend adouble sqrt(const adouble &a);
    inline friend adouble sin(const adouble &a);
    inline friend adouble cos(const adouble &a);
    inline friend adouble asin(const adouble &a);
    inline friend adouble acos(const adouble &a);
    inline friend adouble atan(const adouble &a);

    inline friend adouble atan2(const adouble &a, const adouble &b);
    inline friend adouble pow(const adouble &a, double v);
    inline friend adouble pow(const adouble &a, const adouble &b);
    inline friend adouble pow(double v, const adouble &a);
    inline friend adouble log10(const adouble &a);

    inline friend adouble sinh (const adouble &a);
    inline friend adouble cosh (const adouble &a);
    inline friend adouble tanh (const adouble &a);
#if defined(ATRIG_ERF)
    inline friend adouble asinh (const adouble &a);
    inline friend adouble acosh (const adouble &a);
    inline friend adouble atanh (const adouble &a);
#endif
    inline friend adouble fabs (const adouble &a);
    inline friend adouble ceil (const adouble &a);
    inline friend adouble floor (const adouble &a);
    inline friend adouble fmax (const adouble &a, const adouble &b);
    inline friend adouble fmax (double v, const adouble &a);
    inline friend adouble fmax (const adouble &a, double v);
    inline friend adouble fmin (const adouble &a, const adouble &b);
    inline friend adouble fmin (double v, const adouble &a);
    inline friend adouble fmin (const adouble &a, double v);
    inline friend adouble ldexp (const adouble &a, const adouble &b);
    inline friend adouble ldexp (const adouble &a, const double v);
    inline friend adouble ldexp (const double v, const adouble &a);
    inline friend double frexp (const adouble &a, int* v);
#if defined(ATRIG_ERF)
    inline friend adouble erf (const adouble &a);
#endif

    inline friend void condassign( adouble &res, const adouble &cond,
            const adouble &arg1, const adouble &arg2 );
    inline friend void condassign( adouble &res, const adouble &cond,
            const adouble &arg );
    inline friend void condeqassign( adouble &res, const adouble &cond,
            const adouble &arg1, const adouble &arg2 );
    inline friend void condeqassign( adouble &res, const adouble &cond,
            const adouble &arg );

    /*******************  nontemporary results  ***************************/
    // assignment
    inline adouble& operator = (const double v);
    inline adouble& operator = (const adouble& a);

    // addition
    inline adouble& operator += (const double v);
    inline adouble& operator += (const adouble& a);

    // substraction
    inline adouble& operator -= (const double v);
    inline adouble& operator -= (const adouble& a);

    // multiplication
    inline adouble& operator *= (const double v);
    inline adouble& operator *= (const adouble& a);

    // division
    inline adouble& operator /= (const double v);
    inline adouble& operator /= (const adouble& a);

    // not
    inline int operator ! () const;

    // comparision
    inline int operator != (const adouble&) const;
    inline int operator != (const double) const;
    inline friend int operator != (const double, const adouble&);

    inline int operator == (const adouble&) const;
    inline int operator == (const double) const;
    inline friend int operator == (const double, const adouble&);

    inline int operator <= (const adouble&) const;
    inline int operator <= (const double) const;
    inline friend int operator <= (const double, const adouble&);

    inline int operator >= (const adouble&) const;
    inline int operator >= (const double) const;
    inline friend int operator >= (const double, const adouble&);

    inline int operator >  (const adouble&) const;
    inline int operator >  (const double) const;
    inline friend int operator >  (const double, const adouble&);

    inline int operator <  (const adouble&) const;
    inline int operator <  (const double) const;
    inline friend int operator <  (const double, const adouble&);

    /*******************  getter / setter  ********************************/
    inline double getValue() const;
    inline void setValue(const double v);
    inline const double* const getADValue() const;
    inline void setADValue(const double* v);

    inline double getADValue(const unsigned int p) const;
    inline void setADValue(const unsigned int p, const double v);

    double* getOneADValue(int i) const;

    void setOneADValue(int i, double* v);


    inline explicit operator double const&();
    inline explicit operator double&&();
    inline explicit operator double();

protected:
    inline const list<unsigned int>& get_pattern() const;
    inline void add_to_pattern(const list<unsigned int>& v);
    inline size_t get_pattern_size() const;
    inline void delete_pattern();

public:
    ADOLC_DLL_EXPORT friend int ADOLC_Init_sparse_pattern(adouble *a, int n,unsigned int start_cnt);
    ADOLC_DLL_EXPORT friend int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat);
    ADOLC_DLL_EXPORT friend int ADOLC_get_sparse_jacobian( func_ad *const func, int n, int m, int repeat, double* basepoints, int *nnz, unsigned int **rind, unsigned int **cind, double **values);
#if 0
    ADOLC_DLL_EXPORT friend int ADOLC_get_sparse_jacobian(int n, int m, adouble *x, int *nnz, unsigned int *rind, unsigned int *cind, double *values);
#endif
    /*******************  i/o operations  *********************************/
    ADOLC_DLL_EXPORT friend ostream& operator << ( ostream&, const adouble& );
    ADOLC_DLL_EXPORT friend istream& operator >> ( istream&, adouble& );

private:
    double val;
    double *adval;
    double **ho_deriv;
    list<unsigned int> pattern;
#ifdef USE_ADTL_REFCOUNTING
    refcounter __rcnt;
#endif
    inline static bool _do_val();
    inline static bool _do_adval();
    inline static bool _do_hoval();
    inline static bool _do_indo();
    ADOLC_DLL_EXPIMP static size_t numDir;
    ADOLC_DLL_EXPIMP static size_t degree;
    ADOLC_DLL_EXPIMP static enum Mode forward_mode;
    inline friend void setNumDir(const size_t p);
    inline friend void setDegree(const size_t p);
    inline friend void setMode(enum Mode newmode);
};

}

#include <cmath>
#include <iostream>
#include <limits>

namespace adtl_hov {

enum ModeMask {
    ADTL_Z_MASK = 0x1,
    ADTL_F_MASK = 0x2,
    ADTL_I_MASK = 0x4,
    ADTL_H_MASK = 0x8
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

inline bool adouble::_do_val() {
return true;
//    return ((forward_mode & ADTL_Z_MASK) == ADTL_Z_MASK);
}
#define do_val() likely(adouble::_do_val())
#define no_do_val() unlikely(!adouble::_do_val())

inline bool adouble::_do_adval() {
return true;
//    return ((forward_mode & ADTL_F_MASK) == ADTL_F_MASK);
}
#define do_adval() likely(adouble::_do_adval())
#define no_do_adval() unlikely(!adouble::_do_adval())

inline bool adouble::_do_indo() {
return true;
//    return ((forward_mode & ADTL_I_MASK) == ADTL_I_MASK);
}
inline bool adouble::_do_hoval(){
return true;
//	return ((forward_mode & ADTL_H_MASK) == ADTL_H_MASK);
}

#define do_indo() unlikely(adouble::_do_indo())
#define no_do_indo() likely(!adouble::_do_indo())
#define do_hoval() unlikely(adouble::_do_hoval())

inline void setNumDir(const size_t p) {
#ifdef USE_ADTL_REFCOUNTING
    if (refcounter::refcnt > 0) {
	fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting numDir will not change the number of\n directional derivative in existing adoubles and may lead to erronious results\n or memory corruption\n Number of currently existing adoubles = %zu\n", refcounter::refcnt);
    }
#else
    fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting numDir could change memory allocation of\n derivatives in existing adoubles and may lead to erronious results\n or memory corruption\n");
#endif
    if (p < 1) {
	fprintf(DIAG_OUT, "ADOL-C Error: Tapeless: You are being a moron now.\n");
	abort();
    }
    adouble::numDir = p;
}
inline void setDegree(const size_t p) {

    if (p < 1) {
	fprintf(DIAG_OUT, "ADOL-C Error: Tapeless: You are being a moron now.\n");
	abort();
    }
    adouble::degree = p;
}

inline void setMode(enum Mode newmode) {
#ifdef USE_ADTL_REFCOUNTING
  if (refcounter::refcnt > 0) {
    fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting mode will the change the mode of\n computation in previously computed variables and may lead to erronious results\n or memory corruption\n Number of currently exisiting adoubles = %zu\n", refcounter::refcnt);
  }
#else
  fprintf(DIAG_OUT, "ADOL-C Warning: Tapeless: Setting mode will the change the mode of\n computation in previously computed variables and may lead to erronious results\n or memory corruption\n");
#endif
    adouble::forward_mode = newmode;
}

inline double makeNaN() {
    return ADOLC_MATH_NSP::numeric_limits<double>::quiet_NaN();
}

inline double makeInf() {
    return ADOLC_MATH_NSP::numeric_limits<double>::infinity();
}

#define FOR_I_EQ_0_LT_NUMDIR for (int _i=0; _i < adouble::numDir; ++_i)
#define FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR for (int _j=0; _j < adouble::degree; ++_j) for (int _i=0; _i < adouble::numDir; ++_i)
#define FOR_I_EQ_0_LT_NUMDIR_FOR_J_EQ_0_LT_DEGREE for (int _i=0; _i < adouble::numDir; ++_i) for (int _j=0; _j < adouble::degree; ++_j)
#define ADVAL_I              adval[_i]
#define ADV_I                adv[_i]
#define V_I                  v[_i]
#define HO_DER_I_J		     ho_deriv[_j][_i]

/*******************************  ctors  ************************************/
inline adouble::adouble() : val(0), adval(NULL) {
    if (do_adval())
	adval = new double[adouble::numDir];

    //     ho_deriv= new double[adouble::degree][ adouble::numDir];
    //double** ho_deriv= new double*[adouble::degree];
//    std::cout << " do_hoval() is :  "<< do_hoval() << std::endl;
    if(do_hoval()) // ADTL_HOV
//    	std::cout << "constucting adtl:   degree: " << adouble::degree << "  numDir: " << adouble::numDir << std::endl;
{
    ho_deriv = myalloc2( adouble::degree, adouble::numDir);
}
/*
    for(int i=0;i< adouble::degree;i++)
    	ho_deriv[i]=new double[adouble::numDir];

*/

    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }

}

inline adouble::adouble(const double v) : val(v), adval(NULL) {
    if (do_adval()) {
	adval = new double[adouble::numDir];
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I = 0.0;
    }
    if(do_hoval()) // ADTL_HOV
    {
    	ho_deriv = myalloc2( adouble::degree, adouble::numDir);
    	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
    	HO_DER_I_J=0.0;
    }
    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }
}

inline adouble::adouble(const double v, const double* adv) : val(v), adval(NULL) {
    if (do_adval()) {
	adval = new double[adouble::numDir];
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=ADV_I;
    }
    if(do_hoval()) // ADTL_HOV
    {
    	fprintf(DIAG_OUT, "Dimension mismatch.\n");
    	abort();
    }


    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }
}
inline adouble::adouble(const double v, const double** hov) : val(v), ho_deriv(NULL)  {
    if(do_hoval()) // ADTL_HOV
    {
    	ho_deriv = myalloc2( adouble::degree, adouble::numDir);
    	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
    	HO_DER_I_J=hov[_j][_i];
    }
	if (do_adval()) {
    	fprintf(DIAG_OUT, "Dimension mismatch.\n");
    	abort();
    }


    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();
    }
}



inline adouble::adouble(const adouble& a) : val(a.val), adval(NULL) {
    if (do_adval()) {
	adval = new double[adouble::numDir];
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=a.ADVAL_I;
    }
    if(do_hoval()) // ADTL_HOV
    {
    	ho_deriv = myalloc2( adouble::degree, adouble::numDir);
    	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
    	HO_DER_I_J=a.HO_DER_I_J;
    }
    if (do_indo()) {
     if (!pattern.empty())
          pattern.clear();

     add_to_pattern(a.get_pattern());
    }
}

/*******************************  dtors  ************************************/
inline adouble::~adouble() {
    if (adval != NULL)
	delete[] adval;
//    if(ho_deriv!=NULL)
//    	delete[][]  ho_deriv;
#if 0
    if ( !pattern.empty() )
	pattern.clear();
#endif
}

/*************************  temporary results  ******************************/
// sign
inline adouble adouble::operator - () const {
    adouble tmp;
    if (do_val())
	tmp.val=-val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=-ADVAL_I;
    if(do_hoval()) // ADTL_HOV
    {
    	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
    	    	tmp.HO_DER_I_J=-HO_DER_I_J;
    }
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator + () const {
    return *this;
}

// addition
inline adouble adouble::operator + (const double v) const {
    adouble tmp(val+v, adval);
    if (do_indo())
	tmp.add_to_pattern( get_pattern() ) ;
    return tmp;
}

inline adouble adouble::operator + (const adouble& a) const {
    adouble tmp;
    if (do_val())
	tmp.val=val+a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I+a.ADVAL_I;
    if(do_hoval()) // ADTL_HOV
       {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=HO_DER_I_J+a.HO_DER_I_J;
       }
    if (do_indo()) {
	tmp.add_to_pattern( get_pattern()  );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

inline adouble operator + (const double v, const adouble& a) {
    adouble tmp;
    if (do_val())
    {
    	adouble tmp(v+a.val, a.adval);
    }
	if( do_hoval() ) // ADTL_HOV  //how can this not be declared? it works fine in the method above
       { // why was "a." necessary here and not after the next line?
	 std::cout << "Not implemented. " << std::endl;
	 throw std::exception();
       	//adouble tmp(v+a.val,a.ho_deriv); //TODO: does not exist yet
       }
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// subtraction
inline adouble adouble::operator - (const double v) const {
    adouble tmp;
    if (do_val())
    {
    	adouble tmp(val-v, adval);
    }
	if( do_hoval() ) // ADTL_HOV  //how can this not be declared? it works fine in the method above
       { // why was "a." necessary here and not after the next line?
	 	 std::cout << "Not implemented. " << std::endl;
	 throw std::exception();
       	//adouble tmp(val-v, adval,ho_deriv); //TODO: does not exist yet
       }
    //adouble tmp(val-v, adval); moved up
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator - (const adouble& a) const {
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
    if( do_hoval() ) // ADTL_HOV
           {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=HO_DER_I_J-a.HO_DER_I_J;
           	//adouble tmp(val-v, adval,ho_deriv); //TODO: does not exist yet
           }
    return tmp;
}

inline adouble operator - (const double v, const adouble& a) {
    adouble tmp;
    if (do_val())
	tmp.val=v-a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=-a.ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
       {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=-a.HO_DER_I_J;
       }
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// multiplication
inline adouble adouble::operator * (const double v) const {
    adouble tmp;
    if (do_val())
	tmp.val=val*v;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I*v;
	if( do_hoval() ) // ADTL_HOV
       {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=v*HO_DER_I_J;
       }
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator * (const adouble& a) const {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val())
	tmp.val=val*a.val;
	if( do_hoval() ) // ADTL_HOV
    {
		int i;
		double sum=0.0;
		for(int l=0;l<numDir;l++)
		{
			for(int k=0;k<degree;k++)
			{
				sum=val*a.ho_deriv[k][l]+ho_deriv[k][l]*a.val;
				i=k-1;
				for(int j=0;j<k;j++)
				{
					sum=sum+ho_deriv[j][l]*a.ho_deriv[i-j][l];
				}
				tmp.ho_deriv[k][l]=sum;
				sum=0.0;
			}
		}
    }
    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    if (do_indo()) {
	tmp.add_to_pattern(   get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

inline adouble operator * (const double v, const adouble& a) {
    adouble tmp;
    if (do_val())
	tmp.val=v*a.val;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=v*a.ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=v*a.HO_DER_I_J;
    }

    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// division
inline adouble adouble::operator / (const double v) const {
    adouble tmp;
    if (do_val())
	tmp.val=val/v;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I/v;
	if( do_hoval() ) // ADTL_HOV
    {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=HO_DER_I_J/v;
    }
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator / (const adouble& a) const {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val())
	tmp.val=val/a.val;

	if( do_hoval() ) // ADTL_HOV
    {
		int i;
		double sum=0.0;
		for(int l=0;l<adouble::numDir;l++)
		{
			for(int k=0;k<adouble::degree;k++)
			{
				sum=tmp.val*a.ho_deriv[k][l];
				i=k-1;
				for(int j=0;j<k;j++)
				{
					sum+= tmp.ho_deriv[j][l]*a.ho_deriv[i-j][l];
				}
				tmp.ho_deriv[k][l]=(1.0/a.val)*(ho_deriv[k][l]-sum);
				sum=0.0;
			}
		}
    }

    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    if (do_indo()) {
	tmp.add_to_pattern(   get_pattern() );
	tmp.add_to_pattern( a.get_pattern() );
    }
    return tmp;
}

inline adouble operator / (const double v, const adouble& a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val())
	tmp.val=v/a.val;

	if( do_hoval() ) // ADTL_HOV
    {
		int i;
		double sum=0.0;
		for(int l=0;l<adouble::numDir;l++)
		{
			for(int k=0;k<adouble::degree;k++)
			{
				sum=tmp.val*a.ho_deriv[k][l];
				i=k-1;
				for(int j=0;j<k;j++)
				{
					sum+= tmp.ho_deriv[j][l]*a.ho_deriv[i-j][l];
				}
				tmp.ho_deriv[k][l]=(1.0/a.val)*(-sum);
				sum=0.0;
			}
		}
    }

    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=(-v*a.ADVAL_I)/(a.val*a.val);
    if (do_indo())
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

// inc/dec
inline adouble adouble::operator ++ () {
    if (do_val())
	++val;
    return *this;
}

inline adouble adouble::operator ++ (int) {
    adouble tmp;
    if (do_val())
	tmp.val=val++;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=HO_DER_I_J;
    }
    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

inline adouble adouble::operator -- () {
    if (do_val())
	--val;
    return *this;
}

inline adouble adouble::operator -- (int) {
    adouble tmp;
    if (do_val())
	tmp.val=val--;
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
       	FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
       	    	tmp.HO_DER_I_J=HO_DER_I_J;
    }

    if (do_indo())
	tmp.add_to_pattern( get_pattern() );
    return tmp;
}

// functions
inline adouble tan(const adouble& a) {
    adouble tmp;
    double tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::tan(a.val);    
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	tmp2=ADOLC_MATH_NSP::cos(a.val);
	tmp2*=tmp2;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for tan have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble exp(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::exp(a.val);

    if (likely(adouble::_do_adval() && adouble::_do_val()))
        FOR_I_EQ_0_LT_NUMDIR
            tmp.ADVAL_I=tmp.val*a.ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
    	double sum=0;
    	int i;
    	FOR_I_EQ_0_LT_NUMDIR
		for(int k=0;k<adouble::degree;k++)
		{
	      sum=tmp.val*(k+1)*a.ho_deriv[k][_i];
	      i=k-1;
	      for(int j=0;j<k;j++)
	    	  sum+=tmp.ho_deriv[i-j][_i]*(j+1.0)*a.ho_deriv[j][_i];
	      tmp.ho_deriv[k][_i]=(1.0/(k+1))*sum;
	      sum=0.0;
	    }
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble log(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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

	if( do_hoval() ) // ADTL_HOV
    {
		int i;
		double sum=0.0;
		for(int l=0;l<adouble::numDir;l++)
		{
			for(int k=0;k<adouble::degree;k++)
			{
//				ho_deriv[i-j][l]
			      i=k-1;
			      for(int j=0;j<k;j++)
			    	  sum+=a.ho_deriv[i-j][l]*(j+1)*tmp.ho_deriv[j][l];
			      tmp.ho_deriv[k][l]=(1.0/(k+1))*(1.0/a.val)*((k+1)*a.ho_deriv[k][l]-sum);
			      sum=0.0;
			}
		}
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble sqrt(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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

	if( do_hoval() ) // ADTL_HOV
    {
		int i;
		double sum=0.0;
		for(int l=0;l<adouble::numDir;l++)
		{
			for(int k=0;k<adouble::degree;k++)
			{
//				ho_deriv[i-j][l]
				i=k-1;
				for(int j=0;j<k;j++)
					sum+=tmp.ho_deriv[j][l]*tmp.ho_deriv[i-j][l];
				tmp.ho_deriv[k][l]=1.0/(2.0*tmp.val)*(a.ho_deriv[k][l]-sum);
				sum=0.0;
			}
		}
    }

    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble sin(const adouble &a) {
    adouble tmp;
    adouble tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::sin(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	tmp2.val=ADOLC_MATH_NSP::cos(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2.val*a.ADVAL_I;
    }


    if( do_hoval() ) // ADTL_HOV
        {
    		int i,m;
    	    double sum1=0.0, sum2=0.0;
    		for(int l=0;l<adouble::numDir;l++)
    		{
    			for(int k=0;k<adouble::degree;k++)
    			{
//    				ho_deriv[k][l]
    				m=k+1;
    				      sum1=tmp2.val*m*a.ho_deriv[k][l];
    				      sum2=-tmp.val*m*a.ho_deriv[k][l];
    				      i=k-1;
    				      for(int j=0; j<k;j++){
    					sum1+=(j+1)*a.ho_deriv[j][l]*tmp2.ho_deriv[i-j][l];
    					sum2+=-(j+1)*a.ho_deriv[j][l]*tmp.ho_deriv[i-j][l];
    				      }
    				      tmp.ho_deriv[k][l]=(1.0/m)*sum1;
    				      tmp2.ho_deriv[k][l]=(1.0/m)*sum2;
    				      sum1=0.0;
    				      sum2=0.0;
    			}
    		}
        }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble cos(const adouble &a) {
    adouble tmp;
    adouble tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::cos(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	tmp2.val=-ADOLC_MATH_NSP::sin(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2.val*a.ADVAL_I;
    }

    if( do_hoval() ) // ADTL_HOV
        {
    		int i,m;
    	    double sum1=0.0, sum2=0.0;
    		for(int l=0;l<adouble::numDir;l++)
    		{
    			for(int k=0;k<adouble::degree;k++)
    			{
//    				ho_deriv[k][l]
    				m=k+1;
    				      sum1=tmp2.val*m*a.ho_deriv[k][l];
    				      sum2=-tmp.val*m*a.ho_deriv[k][l];
    				      i=k-1;
    				      for(int j=0; j<k;j++){
    					sum1+=(j+1)*a.ho_deriv[j][l]*tmp2.ho_deriv[i-j][l];
    					sum2+=-(j+1)*a.ho_deriv[j][l]*tmp.ho_deriv[i-j][l];
    				      }
    				      tmp.ho_deriv[k][l]=(1.0/m)*sum1;
    				      tmp2.ho_deriv[k][l]=(1.0/m)*sum2;
    				      sum1=0.0;
    				      sum2=0.0;
    			}
    		}
        }


    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp2;
}

inline adouble asin(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::asin(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for asin have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble acos(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::acos(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=-ADOLC_MATH_NSP::sqrt(1-a.val*a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for acos have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble atan(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for atan have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble atan2(const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for atan2 have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) {
	tmp.add_to_pattern( a.get_pattern() );
	tmp.add_to_pattern( b.get_pattern() );
    }
    return tmp;
}

inline adouble pow(const adouble &a, double v) {
  
  std::cout << "in adouble pow(const adouble &a, double v)  prepping for debug" << std::endl;
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::pow(a.val, v);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=v*ADOLC_MATH_NSP::pow(a.val, v-1);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }
    if( do_hoval() ) // ADTL_HOV
        {
		if(a.val==0.0)
		{
		    //idea: if v==adouble::degree  (as in: the power of a is equal to order of derivatives to be calculated)
		    //        then the derivative is a factorial of adouble::degree     tgamma(adouble::degree+1)
		    // this is different than in GW08. The formula in the table gives a nan for input zero.
		  double factorial_val=tgamma(v+1.0);
		  for(int k=0;k<adouble::degree;k++)for(int l=0;l<adouble::numDir;l++)
		  {
		    for(int k=0;k<adouble::degree;k++)  //unsure if this is needed here. keep it for the moment TODO: check if tmp.ho_deriv is zero before here
		    {
		      tmp.ho_deriv[k][l]=0;
		    }
		    tmp.ho_deriv[k][l]=(1.0/(k+1))*factorial_val;   // division with k+1 (offset because of zero index [0] ) 
		                                                    // because of distinction between tilde{u_k} and u_k
		  }
		}
		else
		{
			int i,m;
			double sum1=0.0, sum2=0.0;
			for(int l=0;l<adouble::numDir;l++)
			{
				for(int k=0;k<adouble::degree;k++)
				{
	//    				ho_deriv[k][l]
				      sum1=tmp.val*(k+1)*a.ho_deriv[k][l];
				      i=k-1;
				      for(int j=0;j<k;j++)
				      {
					sum1+=tmp.ho_deriv[i-j][l]*(j+1)*a.ho_deriv[j][l];
					sum2+=a.ho_deriv[i-j][l]*(1+j)*tmp.ho_deriv[j][l];
				      }
				      tmp.ho_deriv[k][l]=(1.0/(k+1))*(1.0/a.val)*(v*sum1-sum2);  // if a.val is zero we get a nan!!
				      sum1=0.0;
				      sum2=0.0;
				}
			}
		}
        }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble pow(const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::pow(a.val, b.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=b.val*ADOLC_MATH_NSP::pow(a.val, b.val-1);
	double tmp3=ADOLC_MATH_NSP::log(a.val)*tmp.val;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I+tmp3*b.ADVAL_I;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for pow(adouble,adouble) have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) {
	tmp.add_to_pattern( a.get_pattern() );
	tmp.add_to_pattern( b.get_pattern() );
    }
    return tmp;
}

inline adouble pow(double v, const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::pow(v, a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=tmp.val*ADOLC_MATH_NSP::log(v);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=tmp2*a.ADVAL_I;
    }

    if( do_hoval() ) // ADTL_HOV
        {
    		int i;
    		double sum;
    		for(int l=0;l<adouble::numDir;l++)
    		{
    			sum=0.0;
    			for(int k=0;k<adouble::degree;k++)
    			{
//    				ho_deriv[k][l]
    			      sum=tmp.val*(k+1)*a.ho_deriv[k][l];
    			      i=k-1;
    			      for(int j=0;j<k;j++)
    			    	  sum+=tmp.ho_deriv[i-j][l]*(j+1)*a.ho_deriv[j][l];
    			      tmp.ho_deriv[k][l]=(1.0/(k+1))*sum*ADOLC_MATH_NSP::log(v);
    			      sum=0.0;
    			}
    		}
        }


    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble log10(const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::log10(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::log((double)10)*a.val;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for log10 have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble sinh (const adouble &a) {
    adouble tmp, tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::sinh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::cosh(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    }

    if( do_hoval() ) // ADTL_HOV
        {
    	tmp2=ADOLC_MATH_NSP::cosh(a.val);
    		int i,m;
    		double sum;
    	    double sum1=0.0, sum2=0.0;
    		for(int l=0;l<adouble::numDir;l++)
    		{
    			sum=0.0;
    			for(int k=0;k<adouble::degree;k++)
    			{
//    				ho_deriv[k][l]
    				m=k+1;
    				sum1=tmp2.val*m*a.ho_deriv[k][l];
    				sum2=tmp.val*m*a.ho_deriv[k][l];
    		                i=k-1;
    				for(int j=0; j<k;j++){
    					sum1+=(j+1)*a.ho_deriv[j][l]*tmp2.ho_deriv[i-j][l];
    					sum2+=(j+1)*a.ho_deriv[j][l]*tmp.ho_deriv[i-j][l];
    				}
    				tmp.ho_deriv[k][l]=(1.0/m)*sum1;
    				tmp2.ho_deriv[k][l]=(1.0/m)*sum2;
    				sum1=0.0;
    				sum2=0.0;
    			}
    		}
        }

    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble cosh (const adouble &a) {
    adouble tmp,tmp2;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::cosh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sinh(a.val);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I*tmp2;
    }


    if( do_hoval() ) // ADTL_HOV
        {
    		int i,m;
    		double sum;
    	    double sum1=0.0, sum2=0.0;
    	    tmp2=ADOLC_MATH_NSP::sinh(a.val);
    		for(int l=0;l<adouble::numDir;l++)
    		{
    			sum=0.0;
    			for(int k=0;k<adouble::degree;k++)
    			{
//    				ho_deriv[k][l]
    				m=k+1;
    				sum1=tmp2.val*m*a.ho_deriv[k][l];
    				sum2=tmp.val*m*a.ho_deriv[k][l];
    				i=k-1;
    				for(int j=0; j<k;j++){
    					sum1+=(j+1)*a.ho_deriv[j][l]*tmp2.ho_deriv[i-j][l];
    					sum2+=(j+1)*a.ho_deriv[j][l]*tmp.ho_deriv[i-j][l];
    				}
    				tmp.ho_deriv[k][l]=(1.0/m)*sum1;
    				tmp2.ho_deriv[k][l]=(1.0/m)*sum2;
    				sum1=0.0;
    				sum2=0.0;
    			}
    		}
        }


    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble tanh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::tanh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::cosh(a.val);
	tmp2*=tmp2;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for tanh have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

#if defined(ATRIG_ERF)
inline adouble asinh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::asinh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val+1);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for asinh have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble acosh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::acosh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=ADOLC_MATH_NSP::sqrt(a.val*a.val-1);
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for acosh have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble atanh (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP_ERF::atanh(a.val);
    if (likely(adouble::_do_adval() && adouble::_do_val())) {
	double tmp2=1-a.val*a.val;
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=a.ADVAL_I/tmp2;
    }
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for atanh have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}
#endif

inline adouble fabs (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
    if( do_hoval() ) // ADTL_HOV
            {
    	//find leading Taylor coefficient

    	// TODO: do not forget to check the function value. it is the zero-th taylor coefficient

        		int i,m;
        		double sum;
				int* leading_sgn = (int*)alloca(sizeof(int) * adouble::degree);
        		for(int l=0;l<adouble::numDir;l++)  //Init
        		{
        			leading_sgn[l]=1;
        		}
        	    bool found;

        		for(int l=0;l<adouble::numDir;l++)
        		{
        			for(int k=0;k<adouble::degree;k++)
        			{
        				if (tmp.ho_deriv[k][l]>0 && (!found))
						{
        					leading_sgn[l]=1;
        					found = true;
        					break;
        					//found leading coefficient for this direction. it is positive.
        					// no need to do any further for this direction
						}
        				if (tmp.ho_deriv[k][l]<0 && (!found))
        				{
        					//found leading coefficient for this direction. it is negative.
        					found = true;
        					leading_sgn[l]=-1;
        					break;
        				}
        				/*
        				if(tmp.ho_deriv[k][l]=0 && (!found))
        				{
        					//not the leading Taylor coefficient
        					//continue searching
        				}
        				*/
        			}
        			found = false;
        		}

        		//assume we found all leading Taylor coefficients

        		for(int l=0;l<adouble::numDir;l++)
        		{
        			for(int k=0;k<adouble::degree;k++)
        			{
        				tmp.ho_deriv[k][l]*=leading_sgn[l];
        				//TODO do not forget zero-th Taylor coefficient
        			}
        		}


            }
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() );
    return tmp;
}

inline adouble ceil (const adouble &a) {
    adouble tmp;
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::ceil(a.val);
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=0.0;
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for ceil have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble floor (const adouble &a) {
    adouble tmp;
    if (do_val()) 
	tmp.val=ADOLC_MATH_NSP::floor(a.val);
    if (do_adval())
	FOR_I_EQ_0_LT_NUMDIR
	    tmp.ADVAL_I=0.0;
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for floor have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble fmax (const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for fmax have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble fmax (double v, const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for fmax have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble fmax (const adouble &a, double v) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for fmax have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble fmin (const adouble &a, const adouble &b) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for fmin have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble fmin (double v, const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for fmin have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble fmin (const adouble &a, double v) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && (adouble::_do_adval() || adouble::_do_indo()))) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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
	if( do_hoval() ) // ADTL_HOV
    {
		//TODO
       std::cout <<"error: higher order derivatives for fmin have not been implemented yet!!!"<<std::endl;
       exit(1);
    }
    return tmp;
}

inline adouble ldexp (const adouble &a, const adouble &b) {
    adouble tmp = a*pow(2.,b);
    if (do_indo()) {
	tmp.add_to_pattern( a.get_pattern() ) ;
	tmp.add_to_pattern( b.get_pattern() ) ;
    }
    return tmp;
}

inline adouble ldexp (const adouble &a, const double v) {
    return a*ADOLC_MATH_NSP::pow(2.,v);
}

inline adouble ldexp (const double v, const adouble &a) {
    adouble tmp = v*pow(2.,a);
    if (do_indo()) 
	tmp.add_to_pattern( a.get_pattern() ) ;
    return tmp;
}

inline double frexp (const adouble &a, int* v) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }    
    return ADOLC_MATH_NSP::frexp(a.val, v);
}

#if defined(ATRIG_ERF)
inline adouble erf (const adouble &a) {
    adouble tmp;
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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

inline void condassign( adouble &res, const adouble &cond,
			const adouble &arg1, const adouble &arg2 ) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) {
	if (cond.getValue() > 0) 
	    res = arg1;
	else
	    res = arg2;
    }
}

inline void condassign( adouble &res, const adouble &cond,
			const adouble &arg ) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) {
	if (cond.getValue() > 0) 
	    res = arg;
    }
}

inline void condeqassign( adouble &res, const adouble &cond,
                          const adouble &arg1, const adouble &arg2 ) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) {
	if (cond.getValue() >= 0) 
	    res = arg1;
	else
	    res = arg2;
    }
}

inline void condeqassign( adouble &res, const adouble &cond,
                          const adouble &arg ) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (do_val()) {
	if (cond.getValue() >= 0) 
	    res = arg;
    }
}



/*******************  nontemporary results  *********************************/
inline adouble& adouble::operator = (const double v) {
    if (do_val()) 
	val=v;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=0.0;

	if( do_hoval() ) // ADTL_HOV
    {
		FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
		HO_DER_I_J=0.0;
    }


    if (do_indo())
	if (!pattern.empty()) pattern.clear();
    return *this;
}

inline adouble& adouble::operator = (const adouble& a) {
    if (do_val()) 
	val=a.val;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=a.ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
		FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
		HO_DER_I_J=a.HO_DER_I_J;
    }

    if (do_indo()) {
	if (!pattern.empty()) pattern.clear();
	add_to_pattern( a.get_pattern() );
    }
    return *this;
}

inline adouble& adouble::operator += (const double v) {
    if (do_val()) 
	val+=v;
    return *this;
}

inline adouble& adouble::operator += (const adouble& a) {
    if (do_val()) 
	val=val+a.val;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I+=a.ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
		FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
		HO_DER_I_J+=a.HO_DER_I_J;
    }

    if (do_indo()) 
	add_to_pattern( a.get_pattern() );
    return *this;
}

inline adouble& adouble::operator -= (const double v) {
    if (do_val()) 
	val-=v;
    return *this;
}

inline adouble& adouble::operator -= (const adouble& a) {
    if (do_val()) 
	val=val-a.val;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I-=a.ADVAL_I;

	if( do_hoval() ) // ADTL_HOV
    {
		FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
		HO_DER_I_J-=a.HO_DER_I_J;
    }

    if (do_indo()) 
	add_to_pattern( a.get_pattern() ) ;
    return *this;
}

inline adouble& adouble::operator *= (const double v) {
    if (do_val()) 
	val=val*v;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I*=v;
	if( do_hoval() ) // ADTL_HOV
    {
		FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
		HO_DER_I_J*=v;
    }
    return *this;
}

inline adouble& adouble::operator *= (const adouble& a) {
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (likely(adouble::_do_adval() && adouble::_do_val()))
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=ADVAL_I*a.val+val*a.ADVAL_I;
    if (do_val())
	val*=a.val;

    adouble tmp;
    tmp.val=val*a.val;

    if( do_hoval() ) // ADTL_HOV
            {
        		int i,m;
        		double sum;
        	    double sum1=0.0, sum2=0.0;
        		for(int l=0;l<adouble::numDir;l++)
        		{
        		     for(int k=0;k<degree;k++)
        		     {
						sum=val*a.ho_deriv[k][l]+ho_deriv[k][l]*a.val;
						i=k-1;
						for(int j=0;j<k;j++)
							sum+=ho_deriv[j][l]*a.ho_deriv[i-j][l];
						tmp.ho_deriv[k][l]=sum;
						sum=0.0;
        		     }
        			    val=tmp.val;
        			    for(int k=0; k<degree; k++)
        			    	ho_deriv[k][l]=tmp.ho_deriv[k][l];
                }
            }

    if (do_indo()) 
	add_to_pattern( a.get_pattern() ) ;
    return *this;
}

inline adouble& adouble::operator /= (const double v) {
    if (do_val()) 
	val/=v;
    if (do_adval()) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I/=v;

	if( do_hoval() ) // ADTL_HOV
    {
		FOR_J_EQ_0_LT_DEGREE_FOR_I_EQ_0_LT_NUMDIR
		HO_DER_I_J/=v;
    }

    return *this;
}

inline adouble& adouble::operator /= (const adouble& a) {
    if (unlikely(!adouble::_do_val() && adouble::_do_adval())) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (likely(adouble::_do_adval() && adouble::_do_val())) 
	FOR_I_EQ_0_LT_NUMDIR
	    ADVAL_I=(ADVAL_I*a.val-val*a.ADVAL_I)/(a.val*a.val);
    if (do_val()) 
	val=val/a.val;

    if( do_hoval() ) // ADTL_HOV
            {
    			adouble tmp;
        		int i,m;
        		double sum;
        	    double sum1=0.0, sum2=0.0;
        		for(int l=0;l<adouble::numDir;l++)
        		{
        		    for(int k=0;k<degree;k++){
        		      sum=tmp.val*a.ho_deriv[k][l];
        		      i=k-1;
        		    for(int j=0;j<k;j++)
        		      sum+=tmp.ho_deriv[j][l]*a.ho_deriv[i-j][l];
        		    tmp.ho_deriv[k][l]=(1.0/a.val)*(ho_deriv[k][l]-sum);
        		    sum=0.0;}
        		    val=tmp.val;
        		    for(int k=0; k<degree; k++)
        		    	ho_deriv[k][l]=tmp.ho_deriv[k][l];
                }
            }


    if (do_indo()) 
	add_to_pattern( a.get_pattern() ) ;
    return *this;
}

// not
inline int adouble::operator ! () const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val==0.0;
}

// comparision
inline int adouble::operator != (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val!=a.val;
}

inline int adouble::operator != (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val!=v;
}

inline int operator != (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return v!=a.val;
}

inline int adouble::operator == (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val==a.val;
}

inline int adouble::operator == (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val==v;
}

inline int operator == (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return v==a.val;
}

inline int adouble::operator <= (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val<=a.val;
}

inline int adouble::operator <= (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val<=v;
}

inline int operator <= (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return v<=a.val;
}

inline int adouble::operator >= (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val>=a.val;
}

inline int adouble::operator >= (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val>=v;
}

inline int operator >= (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return v>=a.val;
}

inline int adouble::operator >  (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val>a.val;
}

inline int adouble::operator >  (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val>v;
}

inline int operator >  (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return v>a.val;
}

inline int adouble::operator <  (const adouble &a) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val<a.val;
}

inline int adouble::operator <  (const double v) const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val<v;
}

inline int operator <  (const double v, const adouble &a) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return v<a.val;
}

/*******************  getter / setter  **************************************/
inline adouble::operator double const & () {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val;
}

inline adouble::operator double && () {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return (double&&)val;
}

inline adouble::operator double() {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val;
}


inline double adouble::getValue() const {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return val;
}

inline void adouble::setValue(const double v) {
    if (no_do_val()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    val=v;
}

inline const double *const adouble::getADValue() const {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return adval;
}

inline void adouble::setADValue(const double *const v) {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    FOR_I_EQ_0_LT_NUMDIR
    ADVAL_I=V_I;
}

inline double adouble::getADValue(const unsigned int p) const {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (p>=adouble::numDir) 
    {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"getADValue(...)\"!!!\n");
        throw logic_error("incorrect function call, errorcode=-1");
    }
    return adval[p];
}

inline void adouble::setADValue(const unsigned int p, const double v) {
    if (no_do_adval()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if (p>=adouble::numDir) 
    {
        fprintf(DIAG_OUT, "Derivative array accessed out of bounds"\
                " while \"setADValue(...)\"!!!\n");
        throw logic_error("incorrect function call, errorcode=-1");
    }
    adval[p]=v;
}

double* adouble::getOneADValue(int i) const{
	std::cout << " getOneADValue : i= "<< i<<  std::endl;
  return ho_deriv[i];}

void adouble::setOneADValue(int i, double* v){
  ho_deriv[i]=v;}

inline const list<unsigned int>& adouble::get_pattern() const {
    if (no_do_indo()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    return pattern;
}

inline void adouble::delete_pattern() {
    if (no_do_indo()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
    }
    if ( !pattern.empty() )
	pattern.clear();
}

inline void adouble::add_to_pattern(const list<unsigned int>& v) {
    if (no_do_indo()) {
	fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
	throw logic_error("incorrect function call, errorcode=1");
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

inline size_t adouble::get_pattern_size() const {
    if (no_do_indo()) {
     fprintf(DIAG_OUT, "ADOL-C error: Tapeless: Incorrect mode, call setMode(enum Mode mode)\n");
     throw logic_error("incorrect function call, errorcode=1");
    }
    size_t s=0;
    if( !pattern.empty() )
      s = pattern.size();
    return s;
}

}
#endif
