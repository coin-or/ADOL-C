/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.h
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran
           callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_DRIVERS_PSDRIVERS_H)
#define ADOLC_DRIVERS_PSDRIVERS_H 1

#include <adolc/adolcexport.h>
#include <adolc/drivers/absnormalform.h>
#include <adolc/interfaces.h>
#include <adolc/internal/common.h>
#include <span>

namespace ADOLC {
/**
 * @brief Compile-time selector for constant-term updates in
 * `ADOLC::abs_normal`.
 *
 * - `UpdateConsts::True` computes `cy` and `cz` after the abs-normal
 *   form data has been populated.
 * - `UpdateConsts::False` leaves `cy` and `cz` untouched.
 */
ADOLC_API enum class UpdateConsts {
  True,
  False,
};

/**
 * @brief Compute the abs-normal form of a taped function at a point.
 *
 * @param tag           Tape identifier.
 * @param x             Span view of the base point (input values).
 * @param anf           AbsNormalForm object to store results.
 *
 * @tparam uc           Dispatcher for version with or without updating
 *                      the constant terms `cy` and `cz`.
 *
 * @return Zero on success, nonzero on failure.
 */
template <UpdateConsts uc = UpdateConsts::True>
ADOLC_API int abs_normal(short tapeId, std::span<const double> x,
                         AbsNormalForm &anf);

/// @brief Specialization that updates `cy` and `cz` after evaluation.
template <>
ADOLC_API int abs_normal<UpdateConsts::True>(short tapeId,
                                             std::span<const double> x,
                                             AbsNormalForm &anf);
/// @brief Specialization that leaves `cy` and `cz` unchanged.
template <>
ADOLC_API int abs_normal<UpdateConsts::False>(short tapeId,
                                              std::span<const double> x,
                                              AbsNormalForm &anf);

} // namespace ADOLC
BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                             directional_active_gradient_ */
/*                                                                          */
ADOLC_API fint directional_active_gradient_(fint, fint, double *, double *,
                                            double *, double **, short *);
/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient */
/*                                                                          */
ADOLC_API int
directional_active_gradient(short tag,       /* trace identifier */
                            int n,           /* number of independents */
                            const double *x, /* value of independents */
                            const double *d, /* direction */
                            double *g,       /* directional active gradient */
                            short *sigma_g   /* sigma of g */
);

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */
ADOLC_API fint abs_normal_(fint *ftag, fint *fdepen, fint *findep, fint *fswchk,
                           fdouble *fx, fdouble *fy, fdouble *fz, fdouble *fcz,
                           fdouble *fcy, fdouble *fJ, fdouble *fY, fdouble *fZ,
                           fdouble *fL);
/**
 * @brief Compute the ABS-normal form of a taped function.
 *
 * This routine evaluates the abs-normal form in the same ordering used by the
 * driver interfaces: function outputs first and switching variables second,
 * i.e.
 * \f[
 *   \left[\begin{array}{c}
 *     y\\ z
 *   \end{array}\right]
 *   =
 *   \left[\begin{array}{c}
 *     b\\ c
 *   \end{array}\right]
 *   +
 *   \left[\begin{array}{cc}
 *     Y & J\\ Z & L
 *   \end{array}\right]
 *   \left[\begin{array}{c}
 *     x\\ |z|
 *   \end{array}\right]
 * \f]
 * (see Griewank et al.: Solving
 * piecewise linear equations in abs-normal form) associated with a previously
 * recorded tape. It computes function values, switching variables, and the
 * components needed for piecewise-linear / piecewise-smooth analysis.
 *
 *
 * @warning
 * The ordering of switching variables and their corresponding rows in
 * \p J, \p Z, and \p L is **not guaranteed to be stable** across different
 * compilers, optimization levels, or minor changes in the taped operations.
 *
 * For example, for
 * \f[
 *   f(x) = |x_0 - x_1| + |x_2 - x_3|
 * \f]
 * a compiler may process the absolute values in either order.
 * As a consequence, the first switching variable may correspond to
 * \f$|x_0 - x_1|\f$ (so its row depends on \f$x_0, x_1\f$), or instead to
 * \f$|x_2 - x_3|\f$ (so its row depends on \f$x_2, x_3\f$). This yields a
 * different row structure in the abs-normal form even though the mathematical
 * function is identical.
 *
 *
 * @param tag    Tape identifier.
 * @param m      Number of dependent variables.
 * @param n      Number of independent variables.
 * @param swchk  Number of switching variables (as returned by
 * get_num_switches()).
 * @param x      Base point (input values), array of length \p n.
 * @param y      Function values at \p x, array of length \p m.
 * @param z      Switching variable values, array of length \p swchk.
 * @param Y      Matrix of size \p m × \p n.
 * @param J      Matrix of size \p m × \p swchk.
 * @param Z      Matrix of size \p swchk × \p n.
 * @param L      Lower-triangular matrix of size \p swchk × \p swchk.
 *
 * @return Zero on success, nonzero on failure.
 */
ADOLC_API int abs_normal(short tag, int m, int n, int swchk, const double *x,
                         double *y, double *z, double **Y, double **J,
                         double **Z, double **L);

END_C_DECLS

/****************************************************************************/

#endif
