/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparse_fo_rev.h
 Revision: $Id$
 Contents: This file contains some "Easy To Use" interfaces of the sparse
package.


 Copyright (c) Andrea Walther, Christo Mitev

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_SPARSE_FO_REV_H
#define ADOLC_SPARSE_FO_REV_H
#include <adolc/adolcerror.h>
#include <adolc/adolcexport.h>
#include <adolc/dvlparms.h>
#include <adolc/interfaces.h>
#include <adolc/internal/common.h>
#include <adolc/sparse/sparse_fo_rev.h>
#include <adolc/sparse/sparse_options.h>
#include <vector>

namespace ADOLC::Sparse {
/**
 * @brief Forward-mode bit-pattern propagation (safe or tight control-flow).
 *
 * Computes bit-pattern propagation through a taped function in **forward mode**
 * using packed boolean vectors (`bitword_t`).
 *
 * This is a C++ wrapper around the internal ADOL-C drivers:
 * - `int_forward_safe(...)`  (CFM == ControlFlowMode::Safe)
 * - `int_forward_tight(...)` (CFM == ControlFlowMode::Tight)
 *
 * @tparam CFM Control-flow handling mode
 *
 * @param tag Tape identifier.
 * @param m   Number of dependent variables (outputs).
 * @param n   Number of independent variables (inputs).
 * @param p   Number of packed bit-vectors per row/column block.
 *            For the full Jacobian pattern in forward mode use `nBV = n`.
 * @param x   Basepoint array of length @p n. Required when
 *            `CFM == ControlFlowMode::Tight`, ignored in Safe mode.
 * @param[in] X Seed matrix in packed form.
 *            Must contain exactly @p n pointers; each `X[i]` points to an array
 *            of length @p p of `bitword_t`.
 *            The caller owns and manages the memory behind these pointers.
 * @param[out] y Output values array of length @p m.
 *            Only written in Tight mode (because tight propagation actually
 * evaluates at
 *            @p x). May be `nullptr` in Safe mode.
 * @param[out] Y Output bit-pattern matrix in packed form.
 *            Must contain exactly @p m pointers; each `Y[j]` points to an array
 *            of length @p p of `bitword_t`.
 *            On return, `Y` contains the propagated dependency pattern.
 *
 * @return Return code from the underlying ADOL-C driver.
 *
 * @throws (via ADOLCError::fail)
 * - In Tight mode if `x == nullptr` (missing basepoint).
 *
 * @note
 * This function does not allocate or free any memory. It only reads/writes
 * through the provided pointers.
 *
 * @note
 * A legacy interface using a "mode" argument is provided below.
 */
template <ControlFlowMode CFM>
ADOLC_API int forward(short tag, int m, int n, int p, double *x,
                      std::vector<bitword_t *> &X, double *y,
                      std::vector<bitword_t *> &Y) {
  if constexpr (CFM == ControlFlowMode::Tight) {
    if (x != nullptr)
      return int_forward_tight(tag, m, n, p, x, X.data(), y, Y.data());
    else
      ADOLCError::fail(ADOLCError::ErrorType::SPARSE_NO_BP, CURRENT_LOCATION);

  } else if constexpr (CFM == ControlFlowMode::Safe)
    return int_forward_safe(tag, m, n, p, X.data(), Y.data());
  else
    static_assert(is_dependent_v<CFM>,
                  "Mode must be either 'safe' or 'tight'!");
}

/**
 * @brief Forward-mode bit-pattern propagation (safe control-flow only).
 *
 * Safe-mode-only overload of forward bit-pattern propagation that does not
 * require a basepoint @p x and does not produce function values @p y.
 *
 * This is a convenience wrapper around `int_forward_safe(...)`.
 *
 * @tparam CFM Must be `ControlFlowMode::Safe`.
 *
 * @param tag Tape identifier.
 * @param m   Number of dependent variables (outputs).
 * @param n   Number of independent variables (inputs).
 * @param p   Number of packed bit-vectors per row/column block.
 * @param[in] X Seed matrix in packed form.
 *            Must contain exactly @p n pointers; each `X[i]` points to an array
 *            of length @p p of `bitword_t`.
 *            The caller owns and manages the memory behind these pointers.
 * @param[out] Y Output bit-pattern matrix in packed form.
 *            Must contain exactly @p m pointers; each `Y[j]` points to an array
 *            of length @p p of `bitword_t`.
 *            On return, `Y` contains the propagated dependency pattern.
 *
 * @return Return code from the underlying ADOL-C driver.
 *
 * @note
 * Use the 7-argument overload (with @p x and @p y) if you need tight
 * control-flow handling at a basepoint.
 */
template <ControlFlowMode CFM>
ADOLC_API int forward(short tag, int m, int n, int p,
                      std::vector<bitword_t *> &X,
                      std::vector<bitword_t *> &Y) {
  if constexpr (CFM == ControlFlowMode::Safe)
    return int_forward_safe(tag, m, n, p, X.data(), Y.data());
  else
    static_assert(is_dependent_v<CFM>, "Mode has to be 'safe'!");
}

/**
 * @brief Reverse-mode bit-pattern propagation (safe or tight control-flow).
 *
 * Computes bit-pattern propagation through a taped function in **reverse mode**
 * using packed boolean vectors (`bitword_t`).
 *
 * This is a C++ wrapper around the internal ADOL-C drivers:
 * - `int_reverse_safe(...)`  (CFM == ControlFlowMode::Safe)
 * - `int_reverse_tight(...)` (CFM == ControlFlowMode::Tight)
 *
 * @tparam CFM Control-flow handling mode
 *
 * @param tag Tape identifier.
 * @param m   Number of dependent variables (outputs).
 * @param n   Number of independent variables (inputs).
 * @param q   Number of packed bit-vectors (rows of the packed adjoint seed).
 *            For the full Jacobian pattern in reverse mode use `nBV = m`.
 * @param[in] U Reverse seed matrix in packed form.
 *            Must contain exactly @p q pointers; each `U[k]` points to an array
 *            of length @p m of `bitword_t`.
 *            The caller owns and manages the memory behind these pointers.
 * @param[out] Z Output packed bit-pattern matrix.
 *            Must contain exactly @p q pointers; each `Z[k]` points to an array
 *            of length @p n of `bitword_t`.
 *
 * @return Return code from the underlying ADOL-C driver.
 *
 * @note
 * This function does not allocate or free any memory. It only reads/writes
 * through the provided pointers.
 *
 * @note
 * A legacy interface using a "mode" argument is provided below.
 */
template <ControlFlowMode CFM>
ADOLC_API int reverse(short tag, int m, int n, int q,
                      std::vector<bitword_t *> &U,
                      std::vector<bitword_t *> &Z) {
  if constexpr (CFM == ControlFlowMode::Safe)
    return int_reverse_safe(tag, m, n, q, U.data(), Z.data());
  else if constexpr (CFM == ControlFlowMode::Tight)
    return int_reverse_tight(tag, m, n, q, U.data(), Z.data());
  else
    static_assert(is_dependent_v<CFM>,
                  "Mode must be either 'safe' or 'tight'!");
}

/**
 * @brief Legacy forward-mode bit-pattern propagation (mode-selected).
 *
 * Backward-compatible entry point corresponding to the “bit pattern forward”
 * routine documented in the ADOL-C manual (Dependence Analysis section).
 *
 * This overload selects the control-flow handling at runtime via @p mode and
 * dispatches to the typed C++ API:
 * - `mode == 0` → `forward<ControlFlowMode::Safe>(...)`
 * - `mode == 1` → `forward<ControlFlowMode::Tight>(...)`
 *
 * @param tag Tape identifier.
 * @param m   Number of dependent variables (outputs).
 * @param n   Number of independent variables (inputs).
 * @param p   Number of packed bit-vectors per row/column block.
 * @param x   Basepoint array of length @p n. Required when `mode == 1` (tight).
 *            May be `nullptr` when `mode == 0` (safe).
 * @param[in] X Seed dependence structure in packed form.
 *            Must contain exactly @p n pointers; each `X[i]` points to an array
 *            of length @p p of `bitword_t`.
 * @param[out] y Output values array of length @p m. Only required/used for
 *            `mode == 1` (tight).
 * @param[out] Y Output dependence structure in packed form.
 *            Must contain exactly @p m pointers; each `Y[j]` points to an array
 *            of length @p p of `bitword_t`.
 * @param mode Control-flow mode selector:
 *            - `0` : safe mode (default / conservative)
 *            - `1` : tight mode (evaluated at @p x)
 *
 * @return Return code from the underlying ADOL-C driver.
 *
 * @throws (via ADOLCError::fail)
 * - If `mode` is not in `{0,1}`.
 * - If `mode == 1` and `x == nullptr` (missing basepoint).
 *
 * @note
 * This function is retained for backward compatibility. New code should prefer
 * the typed overload `forward<ControlFlowMode::Safe/Tight>(...)`.
 *
 * @note
 * No memory is allocated or freed by this routine; all buffers are managed by
 * the caller.
 *
 * @note
 */
ADOLC_API int forward(short tag, int m, int n, int p, double *x, bitword_t **X,
                      double *y, bitword_t **Y, char mode);

/**
 * @brief Legacy forward-mode bit-pattern propagation, safe mode only.
 *
 * Backward-compatible overload corresponding to the “safe” bit-pattern forward
 * routine in the ADOL-C manual. This variant does not require a basepoint and
 * does not compute function values.
 *
 * Dispatch behavior:
 * - `mode == 0` → `forward<ControlFlowMode::Safe>(tag, m, n, p, X, Y)`
 *
 * @param tag Tape identifier.
 * @param m   Number of dependent variables (outputs).
 * @param n   Number of independent variables (inputs).
 * @param p   Number of packed bit-vectors per row/column block.
 * @param[in] X Seed dependence structure in packed form (`n` pointers, each `p`
 * words).
 * @param[out] Y Output dependence structure in packed form (`m` pointers, each
 * `p` words).
 * @param mode Must be `0` (safe). Any other value is an error.
 *
 * @return Return code from the underlying ADOL-C driver.
 *
 * @throws (via ADOLCError::fail)
 * - If `mode != 0`.
 *
 * @note
 * This function is retained for backward compatibility. New code should prefer
 * `forward<ControlFlowMode::Safe>(tag, m, n, p, X, Y)`.
 */
ADOLC_API int forward(short tag, int m, int n, int p, bitword_t **X,
                      bitword_t **Y, char mode);

/**
 * @brief Legacy reverse-mode bit-pattern propagation (mode-selected).
 *
 * Backward-compatible entry point corresponding to the “bit pattern reverse”
 * routine documented in the ADOL-C manual (Dependence Analysis section).
 *
 * This overload selects the control-flow handling at runtime via @p mode and
 * dispatches to the typed C++ API:
 * - `mode == 0` → `reverse<ControlFlowMode::Safe>(...)`
 * - `mode == 1` → `reverse<ControlFlowMode::Tight>(...)`
 *
 * @param tag Tape identifier.
 * @param m   Number of dependent variables (outputs).
 * @param n   Number of independent variables (inputs).
 * @param q   Number of packed bit-vectors (rows of packed reverse seeds).
 * @param[in] U Seed dependence structure in packed form.
 *            Must contain exactly @p q pointers; each `U[k]` points to an array
 *            of length @p m of `bitword_t`.
 * @param[out] Z Output dependence structure in packed form.
 *            Must contain exactly @p q pointers; each `Z[k]` points to an array
 *            of length @p n of `bitword_t`.
 * @param mode Control-flow mode selector:
 *            - `0` : safe mode (default / conservative)
 *            - `1` : tight mode
 *
 * @return Return code from the underlying ADOL-C driver.
 *
 * @throws (via ADOLCError::fail)
 * - If `mode` is not in `{0,1}`.
 *
 * @note
 * This function is retained for backward compatibility. New code should prefer
 * the typed overload `reverse<ControlFlowMode::Safe/Tight>(...)`.
 *
 * @note
 * No memory is allocated or freed by this routine; all buffers are managed by
 * the caller.
 */
ADOLC_API int reverse(short tag, int m, int n, int q, bitword_t **U,
                      bitword_t **Z, char mode);
} // namespace ADOLC::Sparse
#endif // ADOLC_SPARSE_FO_REV_H