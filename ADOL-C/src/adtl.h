/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adtl.h contains that declaratins of procedures used to
           define various tapeless adouble operations.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert, Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_ADTL_H
#define ADOLC_ADTL_H

#include <ostream>
#include <adolc/common.h>
#include <list>

using std::ostream;
using std::istream;
using std::list;

namespace adtl {

double makeNaN();
double makeInf();

enum Mode {
    ADTL_ZOS = 0x1,
    ADTL_FOV = 0x3,
    ADTL_INDO = 0x5,
    ADTL_FOV_INDO = 0x7
};

class adouble;

class refcounter {
private:
    static size_t refcnt;
    friend void setNumDir(const size_t p);
    friend void setMode(enum Mode newmode);
    friend class adouble;
public:
    refcounter() { ++refcnt; }
    ~refcounter() { --refcnt; }
};

class func_ad {
public:
    virtual int operator() (int n, adouble *x, int m, adouble *y) = 0;
};

class adouble {
public:
    adouble();
    adouble(const double v);
    adouble(const double v, const double* adv);
    adouble(const adouble& a);
    ~adouble();

    // sign
    adouble operator - () const;
    adouble operator + () const;

    // addition
    adouble operator + (const double v) const;
    adouble operator + (const adouble& a) const;
    friend
    adouble operator + (const double v, const adouble& a);

    // substraction
    adouble operator - (const double v) const;
    adouble operator - (const adouble& a) const;
    friend
    adouble operator - (const double v, const adouble& a);

    // multiplication
    adouble operator * (const double v) const;
    adouble operator * (const adouble& a) const;
    friend
    adouble operator * (const double v, const adouble& a);

    // division
    adouble operator / (const double v) const;
    adouble operator / (const adouble& a) const;
    friend
    adouble operator / (const double v, const adouble& a);

    // inc/dec
    adouble operator ++ ();
    adouble operator ++ (int);
    adouble operator -- ();
    adouble operator -- (int);

    // functions
    friend adouble tan(const adouble &a);
    friend adouble exp(const adouble &a);
    friend adouble log(const adouble &a);
    friend adouble sqrt(const adouble &a);
    friend adouble sin(const adouble &a);
    friend adouble cos(const adouble &a);
    friend adouble asin(const adouble &a);
    friend adouble acos(const adouble &a);
    friend adouble atan(const adouble &a);

    friend adouble atan2(const adouble &a, const adouble &b);
    friend adouble pow(const adouble &a, double v);
    friend adouble pow(const adouble &a, const adouble &b);
    friend adouble pow(double v, const adouble &a);
    friend adouble log10(const adouble &a);

    friend adouble sinh (const adouble &a);
    friend adouble cosh (const adouble &a);
    friend adouble tanh (const adouble &a);
#if defined(ATRIG_ERF)
    friend adouble asinh (const adouble &a);
    friend adouble acosh (const adouble &a);
    friend adouble atanh (const adouble &a);
#endif
    friend adouble fabs (const adouble &a);
    friend adouble ceil (const adouble &a);
    friend adouble floor (const adouble &a);
    friend adouble fmax (const adouble &a, const adouble &b);
    friend adouble fmax (double v, const adouble &a);
    friend adouble fmax (const adouble &a, double v);
    friend adouble fmin (const adouble &a, const adouble &b);
    friend adouble fmin (double v, const adouble &a);
    friend adouble fmin (const adouble &a, double v);
    friend adouble ldexp (const adouble &a, const adouble &b);
    friend adouble ldexp (const adouble &a, const double v);
    friend adouble ldexp (const double v, const adouble &a);
    friend double frexp (const adouble &a, int* v);
#if defined(ATRIG_ERF)
    friend adouble erf (const adouble &a);
#endif

    friend void condassign( adouble &res, const adouble &cond,
            const adouble &arg1, const adouble &arg2 );
    friend void condassign( adouble &res, const adouble &cond,
            const adouble &arg );

    /*******************  nontemporary results  ***************************/
    // assignment
    void operator = (const double v);
    void operator = (const adouble& a);

    // addition
    void operator += (const double v);
    void operator += (const adouble& a);

    // substraction
    void operator -= (const double v);
    void operator -= (const adouble& a);

    // multiplication
    void operator *= (const double v);
    void operator *= (const adouble& a);

    // division
    void operator /= (const double v);
    void operator /= (const adouble& a);

    // not
    int operator ! () const;

    // comparision
    int operator != (const adouble&) const;
    int operator != (const double) const;
    friend int operator != (const double, const adouble&);

    int operator == (const adouble&) const;
    int operator == (const double) const;
    friend int operator == (const double, const adouble&);

    int operator <= (const adouble&) const;
    int operator <= (const double) const;
    friend int operator <= (const double, const adouble&);

    int operator >= (const adouble&) const;
    int operator >= (const double) const;
    friend int operator >= (const double, const adouble&);

    int operator >  (const adouble&) const;
    int operator >  (const double) const;
    friend int operator >  (const double, const adouble&);

    int operator <  (const adouble&) const;
    int operator <  (const double) const;
    friend int operator <  (const double, const adouble&);

    /*******************  getter / setter  ********************************/
    double getValue() const;
    void setValue(const double v);
    const double* const getADValue() const;
    void setADValue(const double* v);

    double getADValue(const unsigned int p) const;
    void setADValue(const unsigned int p, const double v);

protected:
    const list<unsigned int>& get_pattern() const;
    void add_to_pattern(const list<unsigned int>& v);
    size_t get_pattern_size() const;
    void delete_pattern();

public:
    friend int ADOLC_Init_sparse_pattern(adouble *a, int n,unsigned int start_cnt);
    friend int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat);
    friend int ADOLC_get_sparse_jacobian( func_ad *const func, int n, int m, int repeat, double* basepoints, int *nnz, unsigned int **rind, unsigned int **cind, double **values);
#if 0
    friend int ADOLC_get_sparse_jacobian(int n, int m, adouble *x, int *nnz, unsigned int *rind, unsigned int *cind, double *values);
#endif
    /*******************  i/o operations  *********************************/
    friend ostream& operator << ( ostream&, const adouble& );
    friend istream& operator >> ( istream&, adouble& );

private:
    double val;
    double *adval;
    list<unsigned int> pattern;
    refcounter __rcnt;
    static bool _do_val();
    static bool _do_adval();
    static bool _do_indo();
    static size_t numDir;
    static enum Mode forward_mode;
    friend void setNumDir(const size_t p);
    friend void setMode(enum Mode newmode);
};

void setNumDir(const size_t p);
void setMode(enum Mode newmode);

int ADOLC_Init_sparse_pattern(adouble *a, int n, unsigned int start_cnt);
int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat);
int ADOLC_get_sparse_jacobian(func_ad *const func,
			      int n, int m, int repeat, double* basepoints, int *nnz,
			      unsigned int **rind, unsigned int **cind,
			      double **values);
#if 0
int ADOLC_get_sparse_jacobian(int n, int m, adouble *x, int *nnz,
			      unsigned int *rind, unsigned int *cind,
			      double *values);
#endif
}
#endif
