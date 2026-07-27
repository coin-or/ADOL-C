#ifndef ADOLC_SPARSE_DRIVERS_C_H
#define ADOLC_SPARSE_DRIVERS_C_H

#include <adolc/sparse/sparsedrivers.h>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute and recover a sparse Jacobian using ColPack.
 *
 * @param tag Tape identifier.
 * @param m Number of dependent variables (rows).
 * @param n Number of independent variables (columns).
 * @param repeat Zero builds and caches the pattern and seed; nonzero reuses
 * them.
 * @param x Evaluation point, with `n` entries. Required for numeric recovery.
 * @param[in,out] nnz Recovered nonzero count. On entry, the count of any
 * supplied buffers; they are reused only if all three are non-null and the
 * count matches.
 * @param[in,out] rind Zero-based row indices.
 * @param[in,out] cind Zero-based column indices.
 * @param[in,out] values Nonzero values.
 * @param options Four entries: method (0 index domains, 1 bit patterns),
 * control flow (0 safe, 1 tight), bit propagation (0 auto, 1 forward, 2
 * reverse), and compression (0 column, 1 row).
 * @return Nonnegative sweep status on success; negative on failure.
 *
 * @pre `nnz`, `rind`, `cind`, `values`, and `options` are non-null.
 * @note Set the three buffer pointers to null for allocation. If supplied
 * buffers cannot be reused, the wrapper deletes them and allocates replacements
 * with `new[]`. Release returned buffers with `delete[]`.
 */
int sparse_jac(short tag, int m, int n, int repeat, const double *x, int *nnz,
               unsigned int **rind, unsigned int **cind, double **values,
               int *options);

/**
 * @brief Compute and recover a sparse Hessian using ColPack.
 *
 * @param tag Tape identifier for a scalar function.
 * @param n Number of independent variables.
 * @param repeat Zero builds and caches the pattern and seed; nonzero reuses
 * them.
 * @param x Evaluation point, with `n` entries. Required for numeric recovery.
 * @param[in,out] nnz Nonzero count set on the first call; must match on reuse.
 * @param[in,out] rind Zero-based row indices.
 * @param[in,out] cind Zero-based column indices.
 * @param[in,out] values Nonzero values.
 * @param options Two entries: control flow (0 safe, 1 tight, 2 old safe,
 * 3 old tight) and recovery (0 indirect, 1 direct).
 * @return Nonnegative sweep status on success; negative on failure.
 *
 * @pre Pointer arguments other than the three buffer values are non-null.
 * @note Supply all three buffers with capacity `*nnz` to reuse them. Otherwise,
 * the routine deletes any supplied buffers and uses ColPack's unmanaged
 * recovery. Release returned buffers with `delete[]`.
 */
int sparse_hess(short tag, int n, int repeat, const double *x, int *nnz,
                unsigned int **rind, unsigned int **cind, double **values,
                int *options);

/**
 * @brief Compute a Jacobian sparsity pattern.
 *
 * @param tag Tape identifier.
 * @param m Number of dependent variables (rows).
 * @param n Number of independent variables (columns).
 * @param x Evaluation point, with `n` entries; may be null in safe mode.
 * @param[out] JP Array of `m` row pointers. Each allocated row starts with its
 * nonzero count, followed by zero-based column indices.
 * @param options Three entries: method (0 index domains, 1 bit patterns),
 * control flow (0 safe, 1 tight), and bit propagation (0 auto, 1 forward,
 * 2 reverse).
 * @return Nonnegative sweep status on success; negative on failure.
 * @note Initialize row pointers to null. Release each allocated row with
 * `delete[]` before reusing the pointer array; this call resets its entries.
 */
int jac_pat(short tag, int m, int n, const double *x, unsigned int **JP,
            int *options);

/**
 * @brief Compute a scalar function's Hessian sparsity pattern.
 *
 * @param tag Tape identifier.
 * @param n Number of independent variables (rows and columns).
 * @param x Evaluation point, with `n` entries; may be null in safe modes.
 * @param[out] HP Array of `n` row pointers. Each allocated row starts with its
 * nonzero count, followed by zero-based column indices.
 * @param option One entry selecting control flow: 0 safe, 1 tight, 2 old safe,
 * or 3 old tight.
 * @return Nonnegative sweep status on success; negative on failure.
 * @note Initialize row pointers to null. Release each allocated row with
 * `delete[]` before reusing the pointer array; this call resets its entries.
 */
int hess_pat(short tag, int n, const double *x, unsigned int **HP, int *option);

/**
 * @brief Generate a ColPack seed matrix for compressed Jacobian recovery.
 *
 * @param m Number of dependent variables.
 * @param n Number of independent variables.
 * @param JP Jacobian pattern: `m` rows, each with a count and column indices.
 * @param[out] S Seed matrix: `n x p` for column compression or `p x m` for row
 * compression.
 * @param[out] p Compressed dimension.
 * @param options One entry selecting compression: 0 column or 1 row.
 * @note The caller owns the seed. Delete each row with `delete[]`, then delete
 * the row-pointer array with `delete[]`.
 */
void generate_seed_jac(int m, int n, unsigned int **JP, double ***S, int *p,
                       int *options);

/**
 * @brief Generate a ColPack seed matrix for compressed Hessian recovery.
 *
 * @param n Number of independent variables.
 * @param HP Hessian pattern: `n` rows, each with a count and column indices.
 * @param[out] S Seed matrix with `n` rows and `p` columns.
 * @param[out] p Number of seed columns.
 * @param options Two entries; entry 1 selects recovery: 0 indirect or 1 direct.
 * @note The caller owns the seed. Delete each row with `delete[]`, then delete
 * the row-pointer array with `delete[]`.
 */
void generate_seed_hess(int n, unsigned int **HP, double ***S, int *p,
                        int *options);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // ADOLC_SPARSE_DRIVERS_C_H
