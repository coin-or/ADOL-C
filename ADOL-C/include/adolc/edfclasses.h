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

#include <adolc/adolcexport.h>
#include <adolc/advector.h>
#include <adolc/externfcts.h>
#include <adolc/externfcts2.h>

class ADOLC_API EDFobject {
protected:
  ext_diff_fct *edf;
  void init_edf(EDFobject *ebase);

public:
  EDFobject() { init_edf(this); }
  virtual ~EDFobject() { edf_zero(edf); }
  virtual int function(short tapeId, size_t dim_x, double *x, size_t dim_y,
                       double *y) = 0;
  virtual int zos_forward(short tapeId, size_t dim_x, double *x, size_t dim_y,
                          double *y) = 0;
  virtual int fos_forward(short tapeId, size_t dim_x, double *dp_x,
                          double *dp_X, size_t dim_y, double *dp_y,
                          double *dp_Y) = 0;
  virtual int fov_forward(short tapeId, size_t dim_x, double *dp_x,
                          size_t num_dirs, double **dpp_X, size_t dim_y,
                          double *dp_y, double **dpp_Y) = 0;
  virtual int fos_reverse(short tapeId, size_t dim_y, double *dp_U,
                          size_t dim_x, double *dp_Z, double *dp_x,
                          double *dp_y) = 0;
  virtual int fov_reverse(short tapeId, size_t dim_y, size_t num_weights,
                          double **dpp_U, size_t dim_x, double **dpp_Z,
                          double *dp_x, double *dp_y) = 0;
  int call(size_t dim_x, adouble *xa, size_t dim_y, adouble *ya) {
    return call_ext_fct(edf, dim_x, xa, dim_y, ya);
  }
  int call(size_t dim_x, advector &x, size_t dim_y, advector &y) {
    return call(dim_x, x.operator adouble *(), dim_y, y.operator adouble *());
  }
};

class ADOLC_API EDFobject_iArr {
protected:
  ext_diff_fct *edf;
  void init_edf(EDFobject_iArr *ebase);

public:
  EDFobject_iArr() { init_edf(this); }
  virtual ~EDFobject_iArr() { edf_zero(edf); }
  virtual int function(short tapeId, size_t iArrLength, int *iArr, size_t dim_x,
                       double *x, size_t dim_y, double *y) = 0;
  virtual int zos_forward(short tapeId, size_t iArrLength, int *iArr,
                          size_t dim_x, double *x, size_t dim_y, double *y) = 0;
  virtual int fos_forward(short tapeId, size_t iArrLength, int *iArr,
                          size_t dim_x, double *dp_x, double *dp_X,
                          size_t dim_y, double *dp_y, double *dp_Y) = 0;
  virtual int fov_forward(short tapeId, size_t iArrLength, int *iArr,
                          size_t dim_x, double *dp_x, size_t num_dirs,
                          double **dpp_X, size_t dim_y, double *dp_y,
                          double **dpp_Y) = 0;
  virtual int fos_reverse(short tapeId, size_t iArrLength, int *iArr,
                          size_t dim_y, double *dp_U, size_t dim_x,
                          double *dp_Z, double *dp_x, double *dp_y) = 0;
  virtual int fov_reverse(short tapeId, size_t iArrLength, int *iArr,
                          size_t dim_y, size_t num_weights, double **dpp_U,
                          size_t dim_x, double **dpp_Z, double *dp_x,
                          double *dp_y) = 0;
  int call(size_t iArrLength, int *iArr, size_t dim_x, adouble *xa,
           size_t dim_y, adouble *ya) {
    return call_ext_fct(edf, iArrLength, iArr, dim_x, xa, dim_y, ya);
  }
  int call(size_t iArrLength, int *iArr, size_t dim_x, advector &x,
           size_t dim_y, advector &y) {
    return call(iArrLength, iArr, dim_x, x.operator adouble *(), dim_y,
                y.operator adouble *());
  }
};

class ADOLC_API EDFobject_v2 {
protected:
  ext_diff_fct_v2 *edf;
  void init_edf(EDFobject_v2 *ebase);

public:
  EDFobject_v2() { init_edf(this); }
  virtual ~EDFobject_v2() { edf_zero(edf); }
  virtual int function(short tapeId, size_t iArrLen, int *iArr, size_t dim_in,
                       size_t dim_out, int *insz, double **x, int *outsz,
                       double **y, void *ctx) = 0;
  virtual int zos_forward(short tapeId, size_t iArrLen, int *iArr,
                          size_t dim_in, size_t dim_out, int *insz, double **x,
                          int *outsz, double **y, void *ctx) = 0;
  virtual int fos_forward(short tapeId, size_t iArrLen, int *iArr,
                          size_t dim_in, size_t dim_out, int *insz, double **x,
                          double **xp, int *outsz, double **y, double **yp,
                          void *ctx) = 0;
  virtual int fov_forward(short tapeId, size_t iArrLen, int *iArr,
                          size_t dim_in, size_t dim_out, int *insz, double **x,
                          size_t num_dirs, double ***Xp, int *outsz, double **y,
                          double ***Yp, void *ctx) = 0;
  virtual int fos_reverse(short tapeId, size_t iArrLen, int *iArr,
                          size_t dim_out, size_t dim_in, int *outsz,
                          double **up, int *insz, double **zp, double **x,
                          double **y, void *ctx) = 0;
  virtual int fov_reverse(short tapeId, size_t iArrLen, int *iArr,
                          size_t dim_out, size_t dim_in, int *outsz,
                          size_t num_weights, double ***Up, int *insz,
                          double ***Zp, double **x, double **y, void *ctx) = 0;
  int call(size_t iArrLen, int *iArr, size_t dim_in, size_t dim_out, int *insz,
           adouble **x, int *outsz, adouble **y) {
    return call_ext_fct(edf, iArrLen, iArr, dim_in, dim_out, insz, x, outsz, y);
  }
};

#endif
