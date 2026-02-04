#ifndef ADOLC_SPARSE_DRIVERS_C_H
#define ADOLC_SPARSE_DRIVERS_C_H

#include <adolc/sparse/sparsedrivers.h>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief High-level C-API to compute a sparse Jacobian (single call or repeat).
 *
 * This function coordinates sparsity detection, seed generation (via ColPack),
 * compressed AD evaluations, and recovery. When @p repeat == 0 the function
 * computes and caches the sparsity/seed information (stored on the tape).
 * Subsequent calls (repeat != 0) reuse cached seeds to compute numerical
 * Jacobian values more efficiently.
 *
 * @param tag        Tape identifier.
 * @param depen      Number of dependent variables (rows).
 * @param indep      Number of independent variables (columns).
 * @param repeat     If 0: compute sparsity and prepare seed (no numeric
 *                   recovery). If >0: perform numeric recovery using cached
 *                   seed.
 * @param basepoint  Array of independent values (required if tight
 *                   control-flow).
 * @param[in,out] nnz
 *                   Input/Output: when repeat==0 set by the routine to the
 *                   number of nonzeros discovered. When repeat!=0 must equal
 *                   the earlier computed number of nonzeros (consistency
 *                   check).
 * @param[out] rind  Pointer-to-pointer receiving row indices (coordinate
 *                   format).
 * @param[out] cind  Pointer-to-pointer receiving column indices (coordinate
 *                   format).
 * @param[out] values Pointer-to-pointer receiving numerical nonzero values.
 *
 * @param[in] options An array defining the sparse options.
 *
 * @return 0 or positive status on success, negative error code on failure.
 *
 * @note Memory ownership semantics for rind/cind/values mirror the underlying
 *       recovery routines: if user provides non-null pointers they will be used
 *       (usermem variants), otherwise unmanaged (allocator) variants are used
 *       and the caller is responsible for freeing the allocated memory.
 *
 * @note This function wraps the C++ API.
 */
int sparse_jac(short tag, int m, int n, int repeat, const double *x, int *nnz,
               unsigned int **rind, unsigned int **cind, double **values,
               int *options);

/**
 * @brief High-level C-API to compute a sparse Hessian.
 *
 * Coordinates Hessian sparsity pattern generation (when `repeat == 0`), seed
 * generation and caching, and the numeric recovery step. For `repeat == 0`
 * the function computes and caches pattern/seed info; for subsequent calls
 * it performs the numeric recovery using cached state.
 *
 * @param tag        ADOL-C tape identifier.
 * @param indep      Number of independent variables (Hessian dimension).
 * @param repeat     If 0: compute pattern and seed (cache them). If >0: perform
 * @param basepoint  Basepoint for tight control-flow (may be nullptr if not
 *                  required).
 * @param[in,out] nnz
 *                   On entry: expected number of nonzeros when repeat != 0.
 *                   On exit: set to the number of nonzeros when repeat == 0.
 * @param[out] rind  Pointer-to-pointer to receive row indices (coordinate
 * format).
 * @param[out] cind  Pointer-to-pointer to receive column indices.
 * @param[out] values Pointer-to-pointer to receive numerical values.
 * @param[in] options Array to set the sparse options.

 * @return
 *   - >= 0 : Success (driver-specific non-negative codes).
 *   - <  0 : Error code (forwarded from pattern/compute routines).
 *
 * @note Memory ownership semantics for rind/cind/values match the underlying
 * recovery calls: if user provided non-null pointers, user memory is used
 * (usermem variants), otherwise the unmanaged variant allocates memory and the
 * caller must free it.
 *
 * @note This function wraps the C++ API.
 */
int sparse_hess(short tag, int n, int repeat, const double *x, int *nnz,
                unsigned int **rind, unsigned int **cind, double **values,
                int *options);

/**
 * @brief C-API to compute Jacobian sparsity pattern using the selected sparse
 * method.
 *
 * Dispatches to either index-domain propagation or bit-vector propagation
 * depending on the template parameter @p SM and control-flow mode @p CFM.
 *
 * @param tag                   Tape identifier (as used by ADOL-C tracing).
 * @param depen                 Number of dependent (output) variables (rows).
 * @param indep                 Number of independent (input) variables (cols).
 * @param basepoint             Pointer to an array of length @p indep. Required
 *                              when @p CFM == ControlFlowMode::Tight; may be
 *                              nullptr in Safe mode.
 * @param[out] compressedRowStorage
 *                              Span of length @p depen where each element is a
 *                              pointer to a Compressed Row Storage (CRS) row:
 *                              - entry [i][0] holds the count of nonzeros in
 *                                row i
 *                              - entry [i][1..] holds the 0-based column
 *                                indices that represent the influencing
 *                                independs
 *
 * @param[in] options An array to set the sparse options.
 *
 * @return  Non-negative: number of nonzeros found; Negative: error code from
 * the underlying propagation routine.
 *
 * @note Ownership: allocated row buffers in compressedRowStorage are
 * transferred to the caller / tape infrastructure and must be freed
 * appropriately.
 *
 * @note This function wraps the C++ API.
 */
int jac_pat(short tag, int m, int n, const double *x, unsigned int **JP,
            int *options);

/**
 * @brief C-API to compute Hessian sparsity pattern (dispatch by control-flow
 * mode).
 *
 * This function computes the sparsity pattern of the Hessian by dispatching to
 * the appropriate internal non-linear independent-index propagation driver
 * based on the compile-time ControlFlowMode template parameter.
 *
 * @param tag                   ADOL-C tape identifier.
 * @param indep                 Number of independent variables (Hessian
 * dimension).
 * @param basepoint             Pointer to an array of length `indep` with the
 *                              evaluation point used when tight control-flow is
 * required.
 * @param[out] compressedRowStorage
 *                              Span of length `indep` where each element is a
 *                              pointer to a Compressed Row Storage (CRS) row
 * for the Hessian pattern. For row i:
 *                              - entry [i][0] is the count of indices stored,
 *                              - entry [i][1..] are the independents the output
 * i depend on.
 *
 * @param[in] options An array to set the sparse options.
 *
 * @return
 *    - >= 0 : Driver-specific non-negative code.
 *    - <  0 : Error code forwarded from the underlying driver.
 *
 * @note
 *  - The routine resets `compressedRowStorage` pointers before use.
 *  - Memory allocated for each row will be owned by the caller/tape.
 *
 * @note This function wraps the C++ API.
 */
int hess_pat(short tag, int n, const double *x, unsigned int **HP, int *option);

/**
 * @brief C-API to generate a seed matrix (coloring) for compressed Jacobian
 * recovery.
 *
 * Uses ColPack's Bipartite graph coloring to produce the seed matrix suitable
 * for recovering the full Jacobian from compressed directional evaluations.
 * This function uses ColPack's unmanaged API and returns pointers via @p Seed
 * and the compressed dimension via @p p.
 *
 * @param m     Number of dependent variables (rows of JP).
 * @param n     Number of independent variables (columns of JP).
 * @param JP    Compressed Row Storage, storing the dependencies on
 *              independents.
 * @param[out] Seed  Output pointer to a 2D array representing the seed matrix.
 *                   Memory is provided by ColPack (unmanaged) and returned to
 *                   caller.
 * @param[out] p     On return: compressed dimension (number of seed columns
 *                   when Row compression, or seed rows when Column
 *                   compression).
 * @param[in] options An array to set the sparse options.
 *
 *
 * @note The caller is responsible for managing the returned @p Seed memory
 *       according to ColPack's unmanaged API semantics.
 *
 * @note This functions wraps the C++ API.
 */
void generate_seed_jac(int m, int n, unsigned int **JP, double ***S, int *p,
                       int *options);

/**
 * @brief C-API to generate a seed matrix for compressed Hessian recovery.
 *
 * Produces a seed matrix for Hessian recovery using ColPack and
 * returns the unmanaged pointer to the seed via @p Seed and the compressed
 * dimension via @p p. The function uses ColPack's unmanaged API, so the
 * returned @p Seed memory is owned by ColPack and will be freed when the
 * ColPack graph object is destroyed (or as per ColPack's unmanaged semantics).
 *
 * @param n     Number of variables (dimension of Hessian).
 * @param HP    Span over CRS row pointers representing Hessian sparsity (HP).
 * @param[out] Seed  Pointer to the returned 2D seed matrix (ColPack-managed).
 * @param[out] p     Compressed dimension (number of seed columns/rows depending
 * on layout).
 * @param[in] options An array to set the sparse options.
 *
 * @note The caller must respect ColPack unmanaged memory semantics for the
 * returned Seed.
 *
 * @note This function wraps the C++ API.
 */
void generate_seed_hess(int n, unsigned int **HP, double ***S, int *p,
                        int *options);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // ADOLC_SPARSE_DRIVERS_C_H