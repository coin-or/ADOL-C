/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/drivers.h
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (with C and C++ callable interfaces including Fortran
            callable versions).

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_DRIVERS_DRIVERS_H)
#define ADOLC_DRIVERS_DRIVERS_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
ADOLC_API int function(short, int, int, const double *, double *);
ADOLC_API fint function_(fint *, fint *, fint *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
ADOLC_API int gradient(short, int, const double *, double *);
ADOLC_API fint gradient_(fint *, fint *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */
ADOLC_API int jacobian(short, int, int, const double *, double **);
ADOLC_API fint jacobian_(fint *, fint *, fint *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                           large_jacobian */
/* large_jacobian(tag, m, n, k, x[n], y[m], J[m][n])                        */
ADOLC_API int large_jacobian(short, int, int, int, double *, double *,
                             double **);
ADOLC_API fint large_jacobian_(fint *, fint *, fint *, fint *, fdouble *,
                               fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                         vector_jacobian  */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
ADOLC_API int vec_jac(short, int, int, int, const double *, const double *,
                      double *);
ADOLC_API fint vec_jac_(fint *, fint *, fint *, fint *, fdouble *,
                        const fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                          jacobian_vector */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
ADOLC_API int jac_vec(short, int, int, const double *, const double *,
                      double *);
ADOLC_API fint jac_vec_(fint *, fint *, fint *, fdouble *, fdouble *,
                        fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
/* uses Hessian-vector product                                              */
ADOLC_API int hessian(short, int, const double *, double **);
ADOLC_API fint hessian_(fint *, fint *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                                 hessian2 */
/* hessian2(tag, n, x[n], lower triangle of H[n][n])                        */
/* uses Hessian-matrix product                                              */
ADOLC_API int hessian2(short, int, double *, double **);
ADOLC_API fint hessian2_(fint *, fint *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                           hessian_vector */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
ADOLC_API int hess_vec(short, int, const double *, const double *, double *);
ADOLC_API fint hess_vec_(fint *, fint *, fdouble *, fdouble *, fdouble *);

/*--------------------------------------------------------------------------*/
/*                                                           hessian_matrix */
/* hess_mat(tag, n, q, x[n], V[n][q], W[n][q])                              */
ADOLC_API int hess_mat(short, int, int, const double *, const double *const *,
                       double **);
ADOLC_API fint hess_mat_(fint *, fint *, fint *, fdouble *, fdouble **,
                         fdouble **);

/*--------------------------------------------------------------------------*/
/*                                                  lagrange_hessian_vector */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
ADOLC_API int lagra_hess_vec(short, int, int, const double *, const double *,
                             const double *, double *);
ADOLC_API fint lagra_hess_vec_(fint *, fint *, fint *, fdouble *, fdouble *,
                               fdouble *, fdouble *);

END_C_DECLS

/**
 * @brief Wrapper for function().
 *
 * @param tnum     tape identification
 * @param m        number of dependent variables m
 * @param n        number of independent variables n
 * @param x        independent vector x
 * @param y        dependent vector y = F(x)
 */
ADOLC_API inline int function(short tnum, int m, int n,
                              std::span<const double> x, std::span<double> y) {
  return function(tnum, m, n, x.data(), y.data());
};
/**
 * @brief Wrapper for gradient().
 *
 * @param tnum     tape identification
 * @param n        number of independent variables n (m=1)
 * @param x        independent vector x
 * @param g        resulting gradient of y = F(x)
 */
ADOLC_API inline int gradient(short tnum, int n, std::span<const double> x,
                              std::span<double> g) {
  return gradient(tnum, n, x.data(), g.data());
};
/**
 * @brief Wrapper for jacobian().
 *
 * @param tnum     tape identification
 * @param m        number of dependent variables m
 * @param n        number of independent variables n
 * @param x        independent vector x
 * @param J        resulating jacobian of y = F(x)
 */
ADOLC_API inline int jacobian(short tnum, int m, int n,
                              std::span<const double> x,
                              std::span<double *> J) {
  return jacobian(tnum, m, n, x.data(), J.data());
};
/**
 * @brief Wrapper for large_jacobian().
 *
 * @param tnum     tape identification
 * @param m        number of dependent variables m
 * @param n        number of independent variables n
 * @param k        keep
 * @param x        independent vector x
 * @param y        dependent vector y = F(x)
 * @param J        resulting jacobian
 */
ADOLC_API inline int large_jacobian(short tnum, int m, int n, int k,
                                    std::span<double> x, std::span<double> y,
                                    std::span<double *> J) {
  return large_jacobian(tnum, m, n, k, x.data(), y.data(), J.data());
};
/**
 * @brief Wrapper for function().
 *
 * @param tnum     tape identification
 * @param m        number of dependent variables m
 * @param n        number of independent variables n
 * @param repeat   indicate repeasted call at same argument
 * @param x        independent vector x
 * @param u        range weigth vector u
 * @param z        result z = u^TF'(x)
 */
ADOLC_API inline int vec_jac(short tnum, int m, int n, int repeat,
                             std::span<const double> x,
                             std::span<const double> u, std::span<double> v) {
  return vec_jac(tnum, m, n, repeat, x.data(), u.data(), v.data());
};
/**
 * @brief Wrapper for jav_vec().
 *
 * @param tnum     tape identification
 * @param m        number of dependent variables m
 * @param n        number of independent variables n
 * @param x        independent vector x
 * @param v        tangent vector v
 * @param z        result z = F'(x)v
 */
ADOLC_API inline int jac_vec(short tnum, int m, int n,
                             std::span<const double> x,
                             std::span<const double> v, std::span<double> u) {
  return jac_vec(tnum, m, n, x.data(), v.data(), u.data());
};
/**
 * @brief Wrapper for hessian().
 *
 * @param tnum     tape identification
 * @param n        number of independent variables n (m=1)
 * @param x        independent vector x
 * @param H        resulting hessian matrix of y = F(x)
 */
ADOLC_API inline int hessian(short tnum, int n, std::span<const double> x,
                             std::span<double *> H) {
  return hessian(tnum, n, x.data(), H.data());
};
/**
 * @brief Wrapper for hessian2().
 *
 * @param tnum     tape identification
 * @param n        number of independent variables n (m=1)
 * @param x        independent vector x
 * @param H        resulting hessian matrix of y = F(x) in lower triangular
 */
ADOLC_API inline int hessian2(short tnum, int n, std::span<double> x,
                              std::span<double *> H) {
  return hessian2(tnum, n, x.data(), H.data());
};
/**
 * @brief Wrapper for hess_vec().
 *
 * @param tnum     tape identification
 * @param m        number of dependent variables m
 * @param n        number of independent variables n
 * @param x        independent vector x
 * @param v        tangent vector v
 * @param z        result z = ∇^2F(x)v
 */
ADOLC_API inline int hess_vec(short tnum, int n, std::span<const double> x,
                              std::span<const double> v, std::span<double> w) {
  return hess_vec(tnum, n, x.data(), v.data(), w.data());
};
/**
 * @brief Wrapper for hess_mat().
 *
 * @param tnum     tape identification
 * @param n        number of independent variables n
 * @param q        number of columns in V
 * @param x        independent vector x
 * @param V        tangent matrix V
 * @param W        result W = ∇^2F(x)V
 */
ADOLC_API inline int hess_mat(short tnum, int n, int q,
                              std::span<const double> x,
                              std::span<const double *> V,
                              std::span<double *> W) {
  return hess_mat(tnum, n, q, x.data(), V.data(), W.data());
};

/****************************************************************************/
#endif
