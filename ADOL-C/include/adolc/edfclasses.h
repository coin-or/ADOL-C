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
  void init_edf(EDFobject *ebase, short outerTapeId, short extTapeId);

public:
  EDFobject() = delete;
  explicit EDFobject(short outerTapeId, short extTapeId) {
    init_edf(this, outerTapeId, extTapeId);
  }
  virtual ~EDFobject() = default;
  virtual int function(short tapeId, int m, int n, double *x, double *y) = 0;
  virtual int zos_forward(short tapeId, int m, int n, int keep, double *x,
                          double *y) = 0;
  virtual int fos_forward(short tapeId, int m, int n, int keep, double *x,
                          double *X, double *y, double *Y) = 0;
  virtual int fov_forward(short tapeId, int m, int n, int p, double *x,
                          double **X, double *y, double **Y) = 0;
  virtual int hos_forward(short tapeId, int m, int n, int d, int keep,
                          double *x, double **X, double *y, double **Y) = 0;
  virtual int hov_forward(short tapeId, int m, int n, int d, int p, double *x,
                          double ***X, double *y, double ***Y) = 0;
  virtual int fos_reverse(short tapeId, int m, int n, double *u, double *z,
                          double *x, double *y) = 0;
  virtual int fov_reverse(short tapeId, int m, int n, int q, double **Uq,
                          double **Zq, double *x, double *y) = 0;
  virtual int hos_reverse(short tapeId, int m, int n, int d, double *u,
                          double **Zd, double **Xd, double **Yd) = 0;
  virtual int hov_reverse(short tapeId, int m, int n, int d, int q, double **Uq,
                          double ***Zqd, short **nz, double **Xd,
                          double **Yd) = 0;
  int call(size_t dim_x, adouble *xa, size_t dim_y, adouble *ya) {
    return call_ext_fct(edf, static_cast<int>(dim_x), xa,
                        static_cast<int>(dim_y), ya);
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
  virtual ~EDFobject_iArr() = default;
  virtual int function(short tapeId, size_t iArrLength, size_t *iArr, int m,
                       int n, double *x, double *y) = 0;
  virtual int zos_forward(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int keep, double *x, double *y) = 0;
  virtual int fos_forward(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int keep, double *x, double *X, double *y,
                          double *Y) = 0;
  virtual int fov_forward(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int p, double *x, double **X, double *y,
                          double **Y) = 0;
  virtual int hos_forward(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int d, int keep, double *x, double **X,
                          double *y, double **Y) = 0;
  virtual int hov_forward(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int d, int p, double *x, double ***X,
                          double *y, double ***Y) = 0;
  virtual int fos_reverse(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, double *u, double *z, double *x,
                          double *y) = 0;
  virtual int fov_reverse(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int q, double **Uq, double **Zq, double *x,
                          double *y) = 0;
  virtual int hos_reverse(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int d, double *u, double **Zd, double **Xd,
                          double **Yd) = 0;
  virtual int hov_reverse(short tapeId, size_t iArrLength, size_t *iArr, int m,
                          int n, int d, int q, double **Uq, double ***Zqd,
                          short **nz, double **Xd, double **Yd) = 0;
  int call(size_t iArrLength, size_t *iArr, size_t dim_x, adouble *xa,
           size_t dim_y, adouble *ya) {
    return call_ext_fct(edf, iArrLength, iArr, static_cast<int>(dim_x), xa,
                        static_cast<int>(dim_y), ya);
  }
  int call(size_t iArrLength, size_t *iArr, size_t dim_x, advector &x,
           size_t dim_y, advector &y) {
    return call(iArrLength, iArr, dim_x, x.operator adouble *(), dim_y,
                y.operator adouble *());
  }
};

class ADOLC_API EDFobject_v2 {
protected:
  ext_diff_fct_v2 *edf;
  void init_edf(EDFobject_v2 *ebase, short outerTapeId, short extTapeId);

public:
  EDFobject_v2() = delete;
  explicit EDFobject_v2(short outerTapeId, short extTapeId) {
    init_edf(this, outerTapeId, extTapeId);
  }
  virtual ~EDFobject_v2() = default;
  virtual int function(short tapeId, size_t iArrLen, size_t *iArr,
                       size_t dim_in, size_t dim_out, size_t *insz, double **x,
                       size_t *outsz, double **y, void *ctx) = 0;
  virtual int zos_forward(short tapeId, size_t iArrLen, size_t *iArr,
                          size_t dim_in, size_t dim_out, size_t *insz,
                          double **x, size_t *outsz, double **y, void *ctx) = 0;
  virtual int fos_forward(short tapeId, size_t iArrLen, size_t *iArr,
                          size_t dim_in, size_t dim_out, size_t *insz,
                          double **x, double **xp, size_t *outsz, double **y,
                          double **yp, void *ctx) = 0;
  virtual int fov_forward(short tapeId, size_t iArrLen, size_t *iArr,
                          size_t dim_in, size_t dim_out, size_t *insz,
                          double **x, size_t num_dirs, double ***Xp,
                          size_t *outsz, double **y, double ***Yp,
                          void *ctx) = 0;
  virtual int fos_reverse(short tapeId, size_t iArrLen, size_t *iArr,
                          size_t dim_out, size_t dim_in, size_t *outsz,
                          double **up, size_t *insz, double **zp, double **x,
                          double **y, void *ctx) = 0;
  virtual int fov_reverse(short tapeId, size_t iArrLen, size_t *iArr,
                          size_t dim_out, size_t dim_in, size_t *outsz,
                          size_t num_weights, double ***Up, size_t *insz,
                          double ***Zp, double **x, double **y, void *ctx) = 0;
  int call(size_t iArrLen, size_t *iArr, size_t dim_in, size_t dim_out,
           size_t *insz, adouble **x, size_t *outsz, adouble **y) {
    return call_ext_fct(edf, iArrLen, iArr, dim_in, dim_out, insz, x, outsz, y);
  }
};

#endif
