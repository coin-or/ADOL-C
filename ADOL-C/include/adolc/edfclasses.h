/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     edfclasses.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_EDFCLASSES_H)
#define ADOLC_EDFCLASSES_H 1

#include <adolc/externfcts.h>
#include <adolc/externfcts2.h>

class EDFobject {
protected:
    ext_diff_fct *edf;
    void init_edf(EDFobject *ebase);
public:
    EDFobject() { init_edf(this); }
    virtual ~EDFobject() { edf_zero(edf); }
    virtual int function(int n, double *x, int m, double *y) = 0;
    virtual int zos_forward(int n, double *x, int m, double *y) = 0;
    virtual int fos_forward(int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y) = 0;
    virtual int fov_forward(int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y) = 0;
    virtual int hos_forward(int n, double *dp_x, int k, double **dpp_X, int m, double *dp_y, double **dpp_Y);
    virtual int hov_forward(int n, double *dp_x, int k, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y);
    virtual int fos_reverse(int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y) = 0;
    virtual int fov_reverse(int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y) = 0;
    virtual int hos_ti_reverse(int m, int d, double **dpp_U, int n, double **dpp_Z, double *dp_x, double **dpp_X, double *dp_y, double **dpp_Y);
    virtual int hos_ov_reverse(int m, int d, double **dpp_U, int n, int p, double ***dppp_Z, double *dp_x, double ***dppp_X, double *dp_y, double ***dppp_Y);
    virtual int hov_ti_reverse(int m, int d, int p, double ***dppp_U, int n, double ***dppp_Z, double *dp_x, double **dpp_X, double *dp_y, double **dpp_Y);
    virtual int indopro_forward_tight(int n, double *dp_x, int m, unsigned int **ind_dom);
    inline int call(int n, adouble *xa, int m, adouble *ya) {
        return call_ext_fct(edf,n,xa,m,ya);
    }
    inline int call(int n, advector& x, int m, advector& y) {
        return call(n,x.operator adouble*(),m,y.operator adouble*());
    }
    inline locint get_index() {
        return edf->index;
    }
};

class EDFobject_iArr {
protected:
    ext_diff_fct *edf;
    void init_edf(EDFobject_iArr *ebase);
public:
    EDFobject_iArr() { init_edf(this); }
    virtual ~EDFobject_iArr() { edf_zero(edf); }
    virtual int function(int iArrLength, int *iArr, int n, double *x, int m, double *y) = 0;
    virtual int zos_forward(int iArrLength, int *iArr, int n, double *x, int m, double *y) = 0;
    virtual int fos_forward(int iArrLength, int *iArr, int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y) = 0;
    virtual int fov_forward(int iArrLength, int *iArr, int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y) = 0;
    virtual int hos_forward(int iArrLength, int *iArr, int n, double *dp_x, int k, double **dpp_X, int m, double *dp_y, double **dpp_Y);
    virtual int hov_forward(int iArrLength, int *iArr, int n, double *dp_x, int k, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y);
    virtual int fos_reverse(int iArrLength, int *iArr, int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y) = 0;
    virtual int fov_reverse(int iArrLength, int *iArr, int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y) = 0;
    virtual int hos_ti_reverse(int iArrLength, int *iArr, int m, int d, double **dpp_U, int n, double **dpp_Z, double *dp_x, double **dpp_X, double *dp_y, double **dpp_Y);
    virtual int hos_ov_reverse(int iArrlength, int *iArr,int m, int d, double **dpp_U, int n, int p, double ***dppp_Z, double *dp_x, double ***dppp_X, double *dp_y, double ***dppp_Y);
    virtual int hov_ti_reverse(int iArrLength, int *iArr, int m, int d, int p, double ***dppp_U, int n, double ***dppp_Z, double *dp_x, double **dpp_X, double *dp_y, double **dpp_Y);
    virtual int indopro_forward_tight(int iArrLength, int *iArr, int n, double *dp_x, int m, unsigned int **ind_dom);
    inline int call(int iArrLength, int *iArr, int n, adouble *xa, int m, adouble *ya) {
        return call_ext_fct(edf,iArrLength,iArr,n,xa,m,ya);
    }
    inline int call(int iArrLength,int* iArr,int n, advector& x, int m, advector& y) {
        return call(iArrLength,iArr,n,x.operator adouble*(),m,y.operator adouble*());
    }
    inline locint get_index() {
        return edf->index;
    }
};

class EDFobject_v2 {
protected:
    ext_diff_fct_v2 *edf;
    void init_edf(EDFobject_v2 *ebase);
public:
    EDFobject_v2() { init_edf(this); }
    virtual ~EDFobject_v2() { edf_zero(edf); }
    virtual int function(int iArrLen, int *iArr, int nin, int nout, int *insz, double **x, int *outsz, double **y, void* ctx) = 0;
    virtual int zos_forward(int iArrLen, int *iArr, int nin, int nout, int *insz, double **x, int *outsz, double **y, void* ctx) = 0;
    virtual int fos_forward(int iArrLen, int* iArr, int nin, int nout, int *insz, double **x, double **xp, int *outsz, double **y, double **yp, void *ctx) = 0;
    virtual int fov_forward(int iArrLen, int* iArr, int nin, int nout, int *insz, double **x, int ndir, double ***Xp, int *outsz, double **y, double ***Yp, void* ctx) = 0;
    virtual int hos_forward(int iArrLen, int* iArr, int nin, int nout, int *insz, double **x, int deg, double ***Xp, int *outsz, double **y, double ***Yp, void* ctx);
    virtual int hov_forward(int iArrLen, int* iArr, int nin, int nout, int *insz, double **x, int deg, int ndir, double ****Xpp, int *outsz, double **y, double ****Ypp, void* ctx);
    virtual int fos_reverse(int iArrLen, int* iArr, int nout, int nin, int *outsz, double **up, int *insz, double **zp, double **x, double **y, void *ctx) = 0;
    virtual int fov_reverse(int iArrLen, int* iArr, int nout, int nin, int *outsz, int dir, double ***Up, int *insz, double ***Zp, double **x, double **y, void* ctx) = 0;
    virtual int indopro_forward_tight(int iArrLen, int *iArr, int nin, int nout, int *insz, double **x, int *outsz, unsigned int ****ind_dom, void* ctx);
    int call(int iArrLen, int* iArr, int nin, int nout, int *insz, adouble **x, int *outsz, adouble **y) {
        return call_ext_fct(edf,iArrLen,iArr,nin,nout,insz,x,outsz,y);
    }
    inline locint get_index() {
        return edf->index;
    }
    inline void set_opaque_context(void *ctx) {
        edf_set_opaque_context(edf,ctx);
    }
    void allocate_mem(int nin, int nout, int* insz, int* outsz);
};

#endif
