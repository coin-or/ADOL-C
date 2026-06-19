/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     interfaces.h
 Revision: $Id$
 Contents: Declaration of the standard interfaces to ADOL-C forward and
           reverse calls (C++, C and Fortran callable C functions).

           Functions prototyped here are defined in the files
                 uni5_for.cpp for
                                 zos_forward.cpp
                                 fos_forward.cpp
                                 hos_forward.cpp
                                 fov_forward.cpp
                                 hov_forward.cpp
                                 hov_wk_forward.cpp
                 fo_rev.cpp for
                                 fos_reverse.cpp
                                 fov_reverse.cpp
                 ho_rev.cpp for
                                 hos_reverse.cpp
                                 hos_ov_reverse.cpp
                                 hov_reverse.cpp
                                 hos_ti_reverse.cpp
                                 hov_ti_reverse.cpp
                 interfacesc.cpp
                 interfacesf.cpp

           ADOL-C Abbreviations :
                 zos : zero-order-scalar mode
                 fos : first-order-scalar mode
                 hos : higher-order-scalar mode
                 fov : first-order-vector mode
                 hov : higher-order-vector mode
                 wk  : with keep
                 ov  : over vector (forward)
                 ti  : Taylor input

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_INTERFACES_H)
#define ADOLC_INTERFACES_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>
#include <span>

/****************************************************************************/
/****************************************************************************/
/*                                                       Now the C++ THINGS */
#if defined(__cplusplus)

/****************************************************************************/
/*                                           FORWARD MODE, overloaded calls */

/*--------------------------------------------------------------------------*/
/*    General scalar call. For d=0 or d=1 done by specialized code          */
/*                                                                          */
/* forward(tag, m, n, d, keep, X[n][d+1], Y[m][d+1]) : hos || fos || zos    */
ADOLC_API int forward(short, int, int, int, int, double **, double **);

/*--------------------------------------------------------------------------*/
/*    Y can be one dimensional if m=1. d=0 or d=1 done by specialized code  */
/*                                                                          */
/* forward(tag, m, n, d, keep, X[n][d+1], Y[d+1]) : hos || fos || zos       */
ADOLC_API int forward(short, int, int, int, int, double **, double *);

/*--------------------------------------------------------------------------*/
/*    X and Y can be one dimensional if d = 0; done by specialized code     */
/*                                                                          */
/* forward(tag, m, n, d, keep, X[n], Y[m]) : zos                            */
ADOLC_API int forward(short, int, int, int, int, const double *, double *);

/*--------------------------------------------------------------------------*/
/*    X and Y can be one dimensional if d omitted; done by specialized code */
/*                                                                          */
/* forward(tag, m, n, keep, X[n], Y[m]) : zos                               */
ADOLC_API int forward(short, int, int, int, const double *, double *);

/*--------------------------------------------------------------------------*/
/*  General vector call                                                     */
/*                                                                          */
/* forward(tag, m, n, d, p, x[n], X[n][p][d], y[m], Y[m][p][d]) : hov       */
ADOLC_API int forward(short, int, int, int, int, const double *,
                      const double *const *const *, double *, double ***);

/*--------------------------------------------------------------------------*/
/*  d = 1 may be omitted. General vector call, done by specialized code     */
/*                                                                          */
/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) : fov                */
ADOLC_API int forward(short, int, int, int, const double *,
                      const double *const *, double *, double **);
/*--------------------------------------------------------------------------*/
/* forward_wrapper(tag, m, n, d, keep, X[n][d+1], Y[m][d+1]) : hos || fos || zos
 */
ADOLC_API inline int forward(short tnum, int m, int n, int d, int keep,
                             std::span<double *> X, std::span<double *> Y) {
  return forward(tnum, m, n, d, keep, X.data(), Y.data());
};
/*--------------------------------------------------------------------------*/
/*    Y can be one dimensional if m=1. d=0 or d=1 done by specialized code  */
/*                                                                          */
/* forward_wrapper(tag, m, n, d, keep, X[n][d+1], Y[d+1]) : hos || fos || zos */
ADOLC_API inline int forward(short tnum, int m, int n, int d, int keep,
                             std::span<double *> X, std::span<double> Y) {
  return forward(tnum, m, n, d, keep, X.data(), Y.data());
};
/*--------------------------------------------------------------------------*/
/*    X and Y can be one dimensional if d = 0; done by specialized code     */
/*                                                                          */
/* forward_wrapper(tag, m, n, d, keep, X[n], Y[m]) : zos */
ADOLC_API inline int forward(short tnum, int m, int n, int d, int keep,
                             std::span<double> X, std::span<double> Y) {
  return forward(tnum, m, n, d, keep, X.data(), Y.data());
};
/*--------------------------------------------------------------------------*/
/*    X and Y can be one dimensional if d omitted; done by specialized code */
/*                                                                          */
/* forward_wrapper(tag, m, n, keep, X[n], Y[m]) : zos */
ADOLC_API inline int forward(short tnum, int m, int n, int keep,
                             std::span<double> X, std::span<double> Y) {
  return forward(tnum, m, n, keep, X.data(), Y.data());
};
/*--------------------------------------------------------------------------*/
/*  General vector call                                                     */
/*                                                                          */
/* forward_wrapper(tag, m, n, d, p, x[n], X[n][p][d], y[m], Y[m][p][d]) : hov */
ADOLC_API inline int forward(short tnum, int m, int n, int d, int p,
                             std::span<double> x, std::span<double **> X,
                             std::span<double> y, std::span<double **> Y) {
  return forward(tnum, m, n, d, p, x.data(), X.data(), y.data(), Y.data());
};
/*--------------------------------------------------------------------------*/
/*  d = 1 may be omitted. General vector call, done by specialized code     */
/*                                                                          */
/* forward_wrapper(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) : fov */
ADOLC_API inline int forward(short tnum, int m, int n, int p,
                             std::span<double> x, std::span<double *> X,
                             std::span<double> y, std::span<double *> Y) {
  return forward(tnum, m, n, p, x.data(), X.data(), y.data(), Y.data());
};
/****************************************************************************/
/*                                           REVERSE MODE, overloaded calls */

/*--------------------------------------------------------------------------*/
/*  General call                                                            */
/*                                                                          */
/* reverse(tag, m, n, d, u[m], Z[n][d+1]) : hos                             */
ADOLC_API int reverse(short, int, int, int, const double *, double **);

/*--------------------------------------------------------------------------*/
/*    u can be a scalar if m=1                                              */
/*                                                                          */
/* reverse(tag, m, n, d, u, Z[n][d+1]) : hos                                */
ADOLC_API int reverse(short, int, int, int, double, double **);

/*--------------------------------------------------------------------------*/
/*    Z can be vector if d = 0; done by specialized code                    */
/*                                                                          */
/* reverse(tag, m, n, d, u[m], Z[n]) : fos                                  */
ADOLC_API int reverse(short, int, int, int, const double *, double *);

/*--------------------------------------------------------------------------*/
/*    u can be a scalar if m=1 and d=0; done by specialized code            */
/*                                                                          */
/* reverse(tag, m, n, d, u, Z[n]) : fos                                     */
ADOLC_API int reverse(short, int, int, int, double, double *);

/*--------------------------------------------------------------------------*/
/*  General vector call                                                     */
/*                                                                          */
/* reverse(tag, m, n, d, q, U[q][m], Z[q][n][d+1], nz[q][n]) : hov          */
ADOLC_API int reverse(short, int, int, int, int, const double *const *,
                      double ***, short ** = nullptr);

/*--------------------------------------------------------------------------*/
/*    U can be a vector if m=1                                              */
/*                                                                          */
/* reverse(tag, m, n, d, q, U[q], Z[q][n][d+1], nz[q][n]) : hov             */
ADOLC_API int reverse(short, int, int, int, int, double *, double ***,
                      short ** = nullptr);

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    If d=0 then Z may be a matrix, no nz; done by specialized code        */
/*                                                                          */
/* reverse(tag, m, n, d, q, U[q][m], Z[q][n]) : fov                         */
ADOLC_API int reverse(short, int, int, int, int, const double *const *,
                      double **);

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    d=0 may be omitted, Z is a matrix, no nz; done by specialized code    */
/*                                                                          */
/* reverse(tag, m, n, q, U[q][m], Z[q][n]) : fov                            */
ADOLC_API int reverse(short, int, int, int, const double *const *, double **);

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    If m=1 and d=0 then U can be vector and Z a matrix but no nz.         */
/*    Done by specialized code                                              */
/*                                                                          */
/* reverse(tag, m, n, d, q, U[q], Z[q][n]) : fov                            */
ADOLC_API int reverse(short, int, int, int, int, double *, double **);

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    If q and U are omitted they default to m and I so that as above       */
/*                                                                          */
/* reverse(tag, m, n, d, Z[q][n][d+1], nz[q][n]) : hov                      */
ADOLC_API int reverse(short, int, int, int, double ***, short ** = 0);

/*--------------------------------------------------------------------------*/
/*  General call                                                            */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, u[m], Z[n][d+1]) : hos */
ADOLC_API inline int reverse(short tnum, int m, int n, int d,
                             std::span<double> u, std::span<double *> Z) {
  return reverse(tnum, m, n, d, u.data(), Z.data());
};

/*--------------------------------------------------------------------------*/
/*    u can be a scalar if m=1                                              */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, u, Z[n][d+1]) : hos */
ADOLC_API inline int reverse(short tnum, int m, int n, int d, double u,
                             std::span<double *> Z) {
  return reverse(tnum, m, n, d, u, Z.data());
};

/*--------------------------------------------------------------------------*/
/*    Z can be vector if d = 0; done by specialized code                    */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, u[m], Z[n]) : fos */
ADOLC_API inline int reverse(short tnum, int m, int n, int d,
                             std::span<double> U, std::span<double> Z) {
  return reverse(tnum, m, n, d, U.data(), Z.data());
};

/*--------------------------------------------------------------------------*/
/*    u can be a scalar if m=1 and d=0; done by specialized code            */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, u, Z[n]) : fos */
ADOLC_API inline int reverse(short tnum, int m, int n, int d, double u,
                             std::span<double> Z) {
  return reverse(tnum, m, n, d, u, Z.data());
};

/*--------------------------------------------------------------------------*/
/*  General vector call                                                     */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, q, U[q][m], Z[q][n][d+1], nz[q][n]) : hov */
ADOLC_API inline int reverse(short tnum, int m, int n, int d, int q,
                             std::span<double *> U, std::span<double **> Z,
                             std::span<short *> nz = {}) {
  return reverse(tnum, m, n, d, q, U.data(), Z.data(), nz.data());
};

/*--------------------------------------------------------------------------*/
/*    U can be a vector if m=1                                              */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, q, U[q], Z[q][n][d+1], nz[q][n]) : hov */
ADOLC_API inline int reverse(short tnum, int m, int n, int d, int q,
                             std::span<double> U, std::span<double **> Z,
                             std::span<short *> nz = {}) {
  return reverse(tnum, m, n, d, q, U.data(), Z.data(), nz.data());
};

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    If d=0 then Z may be a matrix, no nz; done by specialized code        */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, q, U[q][m], Z[q][n]) : fov */
ADOLC_API inline int reverse(short tnum, int m, int n, int d, int q,
                             std::span<double *> U, std::span<double *> Z) {
  return reverse(tnum, m, n, d, q, U.data(), Z.data());
};

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    d=0 may be omitted, Z is a matrix, no nz; done by specialized code    */
/*                                                                          */
/* reverse_wrapper(tag, m, n, q, U[q][m], Z[q][n]) : fov */
ADOLC_API inline int reverse(short tnum, int m, int n, int q,
                             std::span<double *> U, std::span<double *> Z) {
  return reverse(tnum, m, n, q, U.data(), Z.data());
};

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    If m=1 and d=0 then U can be vector and Z a matrix but no nz.         */
/*    Done by specialized code                                              */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, q, U[q], Z[q][n]) : fov */
ADOLC_API inline int reverse(short tnum, int m, int n, int d, int q,
                             std::span<double> U, std::span<double *> Z) {
  return reverse(tnum, m, n, d, q, U.data(), Z.data());
};

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*    If q and U are omitted they default to m and I so that as above       */
/*                                                                          */
/* reverse_wrapper(tag, m, n, d, Z[q][n][d+1], nz[q][n]) : hov */
ADOLC_API inline int reverse(short tnum, int m, int n, int d,
                             std::span<double **> Z,
                             std::span<short *> nz = {}) {
  return reverse(tnum, m, n, d, Z.data(), nz.data());
};

#endif

/****************************************************************************/
/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS

/****************************************************************************/
/*                                                             FORWARD MODE
 */

/*--------------------------------------------------------------------------*/
/*                                                                      ZOS
 */
/* zos_forward(tag, m, n, keep, x[n], y[m]) */
/* (defined in uni5_for.cpp) */
ADOLC_API int zos_forward(short, int, int, int, const double *, double *);

/* zos_forward_nk(tag, m, n, x[n], y[m])                                    */
/* (no keep, defined in uni5_for.cpp, but not supported in ADOL-C 1.8)      */
ADOLC_API int zos_forward_nk(short, int, int, const double *, double *);

/* zos_forward_partx(tag, m, n, ndim[n], x[n][d], y[m])                     */
/* (based on zos_forward)                                                   */

ADOLC_API int zos_forward_partx(short, int, int, const int *,
                                const double *const *, double *);

/*--------------------------------------------------------------------------*/
/*                                                                      FOS */
/* fos_forward(tag, m, n, keep, x[n], X[n], y[m], Y[m])                     */
/* (defined in uni5_for.cpp)                                                 */
ADOLC_API int fos_forward(short, int, int, int, const double *, const double *,
                          double *, double *);

/* fos_forward_nk(tag,m,n,x[n],X[n],y[m],Y[m])                              */
/* (no keep, defined in uni5_for.cpp, but not supported in ADOL-C 1.8)        */
ADOLC_API int fos_forward_nk(short, int, int, const double *, const double *,
                             double *, double *);

/* fos_forward_partx(tag, m, n, ndim[n], x[n][][2], y[m][2])                */
/* (based on fos_forward)                                                   */
ADOLC_API int fos_forward_partx(short, int, int, const int *,
                                const double *const *const *, double **);

/*--------------------------------------------------------------------------*/
/*                                                                      HOS */
/* hos_forward(tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d])            */
/* (defined in uni5_for.cpp)                                                 */
ADOLC_API int hos_forward(short, int, int, int, int, const double *,
                          const double *const *, double *, double **);

/* hos_forward_nk(tag, m, n, d, x[n], X[n][d], y[m], Y[m][d])               */
/* (no keep, defined in uni5_for.cpp, but not supported in ADOL-C 1.8)        */
ADOLC_API int hos_forward_nk(short, int, int, int, const double *,
                             const double *const *, double *, double **);

/* hos_forward_partx(tag, m, n, ndim[n], d, X[n][d+1], Y[m][d+1])           */
/* (defined in forward_partx.cpp)                                             */
ADOLC_API int hos_forward_partx(short, int, int, const int *, int,
                                const double *const *const *, double **);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint hos_forward_(const fint *, const fint *, const fint *,
                            const fint *, const fint *, const fdouble *,
                            const fdouble *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                      FOV */
/* fov_forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])                  */
/* (defined in uni5_for.cpp)                                                 */
ADOLC_API int fov_forward(short, int, int, int, const double *,
                          const double *const *, double *, double **);
ADOLC_API int fov_offset_forward(short, int, int, int, int, const double *,
                                 const double *const *, double *, double **);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint fov_forward_(const fint *, const fint *, const fint *,
                            const fint *, const fdouble *, const fdouble *,
                            fdouble *, fdouble *);

/*  fov_forward_partx(tag, m, n, ndim[n], p,                                */
/*                    x[n][], X[n][][p],y[m], Y[m][p])                      */
ADOLC_API int fov_forward_partx(short, int, int, const int *, int,
                                const double *const *,
                                const double *const *const *, double *,
                                double **);

/*--------------------------------------------------------------------------*/
/*                                                                      HOV */
/* hov_forward(tag, m, n, d, p, x[n], X[n][p][d], y[m], Y[m][p][d])         */
/* (defined in uni5_for.cpp)                                                 */
ADOLC_API int hov_forward(short, int, int, int, int, const double *,
                          const double *const *const *, double *, double ***);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint hov_forward_(const fint *, const fint *, const fint *,
                            const fint *, const fint *, const fdouble *,
                            const fdouble *, fdouble *, fdouble *);

/*  hov_forward_partx(tag, m, n, ndim[n], d, p,                             */
/*                    x[n][], X[n][][p][d], y[m], Y[m][p][d])               */
ADOLC_API int hov_forward_partx(short, int, int, const int *, int, int,
                                const double *const *,
                                const double *const *const *const *, double *,
                                double ***);

/*--------------------------------------------------------------------------*/
/*                                                                   HOV_WK */
/* hov_wk_forward(tag, m, n, d, keep, p, x[n], X[n][p][d], y[m], Y[m][p][d])  */
/* (defined in uni5_for.cpp)                                                 */
ADOLC_API int hov_wk_forward(short, int, int, int, int, int, const double *,
                             const double *const *const *, double *,
                             double ***);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint hov_wk_forward_(const fint *, const fint *, const fint *,
                               const fint *, const fint *, const fint *,
                               fdouble *, fdouble *, fdouble *, fdouble *);

/****************************************************************************/
/*                                                    BIT PATTERN UTILITIES */
/*--------------------------------------------------------------------------*/
/*                                                            INT_FOR, SAFE */
/* int_forward_safe(tag, m, n, p, X[n][p], Y[m][p])                         */

ADOLC_API int int_forward_safe(short, int, int, int, const bitword_t *const *,
                               bitword_t **);

/*--------------------------------------------------------------------------*/
/*                                                           INT_FOR, TIGHT */
/* int_forward_tight(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])            */

ADOLC_API int int_forward_tight(short, int, int, int, const double *,
                                const bitword_t *const *, double *,
                                bitword_t **);

/****************************************************************************/
/*                                                   INDEX DOMAIN UTILITIES */
/*--------------------------------------------------------------------------*/
/*                                                            INDOPRO, SAFE */
/* indopro_forward_safe(tag, m, n, p, x[n], *Y[m])                          */

ADOLC_API int indopro_forward_safe(short, int, int, const double *, uint **);

/*--------------------------------------------------------------------------*/
/*                                                           INDOPRO, TIGHT */
/* indopro_forward_tight(tag, m, n,  x[n], *Y[m])                           */

ADOLC_API int indopro_forward_tight(short, int, int, const double *, uint **);

/****************************************************************************/
/*                                         NONLINEAR INDEX DOMAIN UTILITIES */
/*--------------------------------------------------------------------------*/
/*                                                            INDOPRO, SAFE */
/* nonl_ind_forward_safe(tag, m, n, p, x[n], *Y[m])                          */

ADOLC_API int nonl_ind_forward_safe(short, int, int, const double *, uint **);

/*--------------------------------------------------------------------------*/
/*                                                           INDOPRO, TIGHT */
/* nonl_ind_forward_tight(tag, m, n,  x[n], *Y[m])                           */

ADOLC_API int nonl_ind_forward_tight(short, int, int, const double *, uint **);
/*--------------------------------------------------------------------------*/
/*                                                            INDOPRO, SAFE */
/* nonl_ind_old_forward_safe(tag, m, n, p, x[n], *Y[m]) */

ADOLC_API int nonl_ind_old_forward_safe(short, int, int, const double *,
                                        uint **);

/*--------------------------------------------------------------------------*/
/*                                                           INDOPRO, TIGHT */
/* nonl_ind_old_forward_tight(tag, m, n,  x[n], *Y[m]) */

ADOLC_API int nonl_ind_old_forward_tight(short, int, int, const double *,
                                         uint **);

/****************************************************************************/
/*                                                             REVERSE MODE */

/*--------------------------------------------------------------------------*/
/*                                                                      FOS */
/* fos_reverse(tag, m, n, u[m], z[n])                                       */
/* (defined in fo_rev.cpp)                                                    */
ADOLC_API int fos_reverse(short, int, int, const double *, double *);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint fos_reverse_(const fint *, const fint *, const fint *,
                            const fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                      HOS */
/*  hos_reverse(tag, m, n, d, u[m], Z[n][d+1])                              */
/* (defined in ho_rev.cpp)                                                    */
ADOLC_API int hos_reverse(short, int, int, int, const double *, double **);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint hos_reverse_(const fint *, const fint *, const fint *,
                            const fint *, const fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                   HOS_TI */
/*  hos_ti_reverse(tag, m, n, d, U[m][d+1], Z[n][d+1])                      */
/* (defined in ho_rev.cpp)                                                    */
ADOLC_API int hos_ti_reverse(short, int, int, int, const double *const *,
                             double **);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint hos_ti_reverse_(const fint *, const fint *, const fint *,
                               const fint *, const fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                   HOS_OV */
/*  hos_ov_reverse(tag, m, n, d, p, U[m][d+1], Z[p][n][d+1])                */
/* (defined in ho_rev.cpp)                                                    */
ADOLC_API int hos_ov_reverse(short, int, int, int, int, const double *const *,
                             double ***);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint hos_ov_reverse_(const fint *, const fint *, const fint *,
                               const fint *, const fint *, const fdouble *,
                               fdouble ***);

/*--------------------------------------------------------------------------*/
/*                                                                      FOV */
/* fov_reverse(tag, m, n, p, U[p][m], Z[p][n])                              */
/* (defined in fo_rev.cpp)                                                    */
ADOLC_API int fov_reverse(short, int, int, int, const double *const *,
                          double **);

/* now pack the arrays into vectors for Fortran calling                     */
ADOLC_API fint fov_reverse_(const fint *, const fint *, const fint *,
                            const fint *, const fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                      HOV */
/* hov_reverse(tag, m, n, d, p, U[p][m], Z[p][n][d+1], nz[p][n])            */
/* (defined in ho_rev.cpp)                                                    */
ADOLC_API int hov_reverse(short, int, int, int, int, const double *const *,
                          double ***, short **);

/* now pack the arrays into vectors for Fortran calling      */
ADOLC_API fint hov_reverse_(const fint *, const fint *, const fint *,
                            const fint *, const fint *, const fdouble *,
                            fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                   HOV_TI */
/* hov_ti_reverse(tag, m, n, d, p, U[p][m][d+1], Z[p][n][d+1], nz[p][n])    */
/* (defined in ho_rev.cpp)                                                   */
ADOLC_API int hov_ti_reverse(short, int, int, int, int,
                             const double *const *const *, double ***,
                             short **);

/* now pack the arrays into vectors for Fortran calling      */
ADOLC_API fint hov_ti_reverse_(const fint *, const fint *, const fint *,
                               const fint *, const fint *, const fdouble *,
                               fdouble *);

/****************************************************************************/
/*                                                    BIT PATTERN UTILITIES */
/*--------------------------------------------------------------------------*/
/*                                                           INT_REV, TIGHT */
/* int_reverse_tight(tag, m, n, q, U[q][m], Z[q][n])                        */

ADOLC_API int int_reverse_tight(short, int, int, int, const bitword_t *const *,
                                bitword_t **);

/*--------------------------------------------------------------------------*/
/*                                                            INT_REV, SAFE */
/* int_reverse_safe(tag, m, n, q, U[q][m], Z[q][n])                         */

ADOLC_API int int_reverse_safe(short, int, int, int, const bitword_t *const *,
                               bitword_t **);

/*--------------------------------------------------------------------------*/
ADOLC_API int zos_pl_forward(short, int, int, int, const double *, double *,
                             double *);
ADOLC_API short firstsign(int, const double *, const double *);
ADOLC_API short ext_firstsign(double, double, int, double *, double *);
ADOLC_API short ext_firstsign2(double, int, double *, double *);
ADOLC_API int fos_pl_forward(short, int, int, const double *, const double *,
                             double *, double *, double *, double *);
ADOLC_API int fov_pl_forward(short, int, int, int, const double *,
                             const double *const *, double *, double **,
                             double *, double **, short *);
ADOLC_API int fos_pl_sig_forward(short, int, int, const double *,
                                 const double *, int, const short *,
                                 const short *, double *, double *, double *,
                                 double *, short *);
ADOLC_API int fov_pl_sig_forward(short, int, int, int, const double *,
                                 const double *const *, int, const short *,
                                 const short *, double *, double **, double *,
                                 double **, short *);
ADOLC_API int indopro_forward_absnormal(short, int, int, int, const double *,
                                        uint **);
/*--------------------------------------------------------------------------*/

/**
 * @brief Computes vector-Jacobian product of the extended Jacobian of the
 * abs-linearization.
 *
 * Using the evaluation point of the previous call of zos_pl_forward this
 * function computes vector-Jacobian products with the extended Jacobian
 * written in row order `[y ; z]` and column order `[x ; |z|]`, i.e.
 * \f[
 *   \left[\begin{array}{cc}
 *     Y & J\\ Z & L
 *   \end{array}\right].
 * \f]
 *
 * \p result = \p lagrange . Y + \p lagrangeSwitch . Z
 *
 * \p resultSwitch = \p lagrange . J + \p lagrangeSwitch . L
 *
 * @param tag            Tape identifier.
 * @param m              Number of dependent variables.
 * @param n              Number of independent variables.
 * @param s              Number of switching variables.
 * @param lagrange       Seedvector for dependents of size \p m.
 * @param lagrangeSwitch Seedvector for switches of size \p s.
 * @param results        Derivatives wrt. indeps of size \p n.
 * @param resultsSwitch  Derivatives wrt. switches of size \p s.
 *
 * @return Zero on success, nonzero on failure.
 */
ADOLC_API int fos_pl_reverse(short tag, int m, int n, int s,
                             const double *lagrange,
                             const double *lagrangeSwitch, double *results,
                             double *resultsSwitch);

/**
 * @brief Computes several vector-Jacobian product of the extended Jacobian of
 * the abs-linearization.
 *
 * This is the vectormode of fos_pl_reverse, check there for more info.
 *
 * @param tag            Tape identifier.
 * @param m              Number of dependent variables.
 * @param n              Number of independent variables.
 * @param s              Number of switching variables.
 * @param nrow           Number of rows.
 * @param lagrange       Seedvectors for dependents of size \p nrow x \p m.
 * @param lagrangeSwitch Seedvectors for switches of size \p nrow x \p s.
 * @param results        Derivatives wrt. indeps of size \p nrow x \p n.
 * @param resultsSwitch  Derivatives wrt. switches of size \p nrow x \p s.
 *
 * @return Zero on success, nonzero on failure.
 */
ADOLC_API int fov_pl_reverse(short tag, int m, int n, int s, int nrow,
                             const double *const *lagrange,
                             const double *const *lagrangeSwitch,
                             double **results, double **resultsSwitch);

ADOLC_API int fos_pl_sig_reverse(short, int, int, int, const short *,
                                 const double *, double *);

END_C_DECLS

/* zos_forward_wrapper(tag, m, n, keep, x[n], y[m])                         */
ADOLC_API inline int zos_forward(short tnum, int depen, int indep, int keep,
                                 std::span<double> argument,
                                 std::span<double> result) {
  return zos_forward(tnum, depen, indep, keep, argument.data(), result.data());
};

/* fos_forward_wrapper(tag, m, n, keep, x[n], X[n], y[m], Y[m])              */
ADOLC_API inline int fos_forward(short tnum, int depen, int indep, int keep,
                                 std::span<double> basepoint,
                                 std::span<double> argument,
                                 std::span<double> valuepoint,
                                 std::span<double> taylors) {
  return fos_forward(tnum, depen, indep, keep, basepoint.data(),
                     argument.data(), valuepoint.data(), taylors.data());
};

/* fov_forward_wrapper(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) */
ADOLC_API inline int fov_forward(short tnum, int depen, int indep, int p,
                                 std::span<double> basepoint,
                                 std::span<double *> argument,
                                 std::span<double> valuepoint,
                                 std::span<double *> taylors) {
  return fov_forward(tnum, depen, indep, p, basepoint.data(), argument.data(),
                     valuepoint.data(), taylors.data());
};
/* ADOLC_API inline int fov_forward(short tnum, int depen, int indep, int p,
                                 std::span<double> basepoint,
                                 const double *const *argument,
                                 std::span<double> valuepoint,
                                 double **taylors) {
  return fov_forward(tnum, depen, indep, p, basepoint.data(), argument,
                     valuepoint.data(), taylors);
}; */

/* fos_reverse_wrapper(tag, m, n, u[m], z[n]) */
ADOLC_API inline int fos_reverse(short tnum, int depen, int indep,
                                 std::span<double> lagrange,
                                 std::span<double> results) {
  return fos_reverse(tnum, depen, indep, lagrange.data(), results.data());
};

/* fov_reverse_wrapper(tag, m, n, p, U[p][m], Z[p][n]) */
ADOLC_API inline int fov_reverse(short tnum, int depend, int indep, int nrows,
                                 std::span<double *> lagrange,
                                 std::span<double *> results) {
  return fov_reverse(tnum, depend, indep, nrows, lagrange.data(),
                     results.data());
};

/* hos_forward_wrapper(tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d]) */
ADOLC_API inline int hos_forward(short tnum, int depen, int indep, int degree,
                                 int keep, std::span<double> basepoint,
                                 std::span<double *> argument,
                                 std::span<double> valuepoint,
                                 std::span<double *> taylors) {
  return hos_forward(tnum, depen, indep, degree, keep, basepoint.data(),
                     argument.data(), valuepoint.data(), taylors.data());
};

/* hov_forward_wrapper(tag, m, n, d, p, x[n], X[n][p][d], y[m], Y[m][p][d]) */
ADOLC_API inline int hov_forward(short tnum, int depen, int indep, int degree,
                                 int p, std::span<double> basepoint,
                                 std::span<double **> argument,
                                 std::span<double> valuepoint,
                                 std::span<double **> taylors) {
  return hov_forward(tnum, depen, indep, degree, p, basepoint.data(),
                     argument.data(), valuepoint.data(), taylors.data());
};

/* hov_wk_forward_wrapper(tag, m, n, d, keep, p, x[n], X[n][p][d], y[m],
 * Y[m][p][d])  */
ADOLC_API inline int
hov_wk_forward(short tnum, int dim_out, int dim_in, int degree, int keep,
               int num_dirs, std::span<double> input, std::span<double **> X,
               std::span<double> result, std::span<double **> Y) {
  return hov_wk_forward(tnum, dim_out, dim_in, degree, keep, num_dirs,
                        input.data(), X.data(), result.data(), Y.data());
};

/*  hos_reverse_wrapper(tag, m, n, d, u[m], Z[n][d+1]) */
ADOLC_API inline int hos_reverse(short tnum, int depen, int indep, int degree,
                                 std::span<double> lagrange,
                                 std::span<double *> results) {
  return hos_reverse(tnum, depen, indep, degree, lagrange.data(),
                     results.data());
};
/* ADOLC_API inline int hos_reverse(short tnum, int depen, int indep, int
degree, std::span<double> lagrange, double **results) { return hos_reverse(tnum,
depen, indep, degree, lagrange.data(), results);
}; */

/*  hos_ov_reverse_wrapper(tag, m, n, d, p, U[m][d+1], Z[p][n][d+1]) */
ADOLC_API inline int hos_ov_reverse(short tnum, int depen, int indep,
                                    int degree, int nrows,
                                    std::span<double *> lagrange,
                                    std::span<double **> results) {
  return hos_ov_reverse(tnum, depen, indep, degree, nrows, lagrange.data(),
                        results.data());
};

/* hov_reverse_wrapper(tag, m, n, d, p, U[p][m], Z[p][n][d+1], nz[p][n]) */
ADOLC_API inline int hov_reverse(short tnum, int depen, int indep, int degree,
                                 int nrows, std::span<double *> lagrange,
                                 std::span<double **> results,
                                 std::span<short *> nonzero) {
  return hov_reverse(tnum, depen, indep, degree, nrows, lagrange.data(),
                     results.data(), nonzero.data());
};

/* zos_pl_forward_wrapper()*/
ADOLC_API inline int zos_pl_forward(short tnum, int depen, int indep, int keep,
                                    std::span<double> x, std::span<double> y,
                                    std::span<double> switching_vec) {
  return zos_pl_forward(tnum, depen, indep, keep, x.data(), y.data(),
                        switching_vec.data());
};

#endif
