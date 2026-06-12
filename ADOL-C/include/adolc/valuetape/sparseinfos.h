#ifndef ADOLC_SPARSE_INFOS_H
#define ADOLC_SPARSE_INFOS_H
#include <adolc/adalloc.h>
#include <adolc/internal/common.h>
#include <adolc/sparse/sparse_options.h>
#include <adolc/sparse/sparsematrix.h>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ADOLC::Sparse {
template <CompressionMode CM>
void generateSeedJac(int, int, const std::span<uint *>, double ***, int *) {};
/**
 * @brief Cached sparse Jacobian recovery data stored on a tape.
 *
 * This helper owns the ColPack graph/coloring objects and all temporary
 * buffers required to recover sparse Jacobians from compressed directional
 * evaluations.
 */
struct ADOLC_API SparseJacInfos {
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
  double *y_{nullptr};

  // Seed is memory managed by ColPack and will be deleted
  double **Seed_{nullptr};
  double **B_{nullptr};

  // type is dictated by ColPack
  std::vector<uint *> JP_;
  int depen_{0};
  int nnzIn_{0};
  int seedClms_{0};
  int seedRows_{0};

  ~SparseJacInfos();
  SparseJacInfos();
  SparseJacInfos(const SparseJacInfos &) = delete;
  SparseJacInfos(SparseJacInfos &&other) noexcept;
  SparseJacInfos &operator=(const SparseJacInfos &) = delete;
  SparseJacInfos &operator=(SparseJacInfos &&other) noexcept;

  /**
   * @brief Replace the cached sparsity pattern.
   *
   * Existing CRS rows are deleted before the new storage is adopted.
   *
   * @param JPIn New CRS sparsity rows.
   */
  void setJP(std::vector<uint *> &&JPIn) {
    // delete JP
    for (auto &j : JP_)
      delete[] j;
    JP_ = std::move(JPIn);
    depen_ = static_cast<int>(JP_.size());
  }
  /** @brief Access the cached CRS Jacobian pattern. */
  std::vector<uint *> &getJP() { return JP_; }
  /** @brief Initialize ColPack coloring state for the given matrix shape. */
  void initColoring(int dimOut, int dimIn);
  /** @brief Generate and cache a Jacobian seed matrix. */
  void generateSeedJac(const std::string &coloringVariant);
  /**
   * @brief Recover row-compressed sparse Jacobian entries into `SparseMatrix`.
   *
   * @note The structure is adapted from ColPack's
   *      `JacobianRecovery1D::RecoverD2Row_CoordinateFormat_usermem` to work
   *       with an array of row-column-value triplets (aka.
   *       `CoordinateFormatTripled`) rather than separate arrays for rows,
   *       columns, and values.
   */
  void recoverRowFormatUserMem(SparseMatrix &sparseJac) const;
  /**
   * @brief Resize and recover a row-compressed sparse Jacobian.
   */
  void recoverRowFormat(SparseMatrix &sparseJac) const;
  /**
   * @brief Recover column-compressed sparse Jacobian entries into
   * `SparseMatrix`.
   *
   * @note The structure is adapted from ColPack's
   *      `JacobianRecovery1D::RecoverD2Cln_CoordinateFormat_usermem` to work
   * with an array of row-column-value triplets (aka. `CoordinateFormatTripled`)
   *       rather than separate arrays for rows, columns, and values.
   */
  void recoverColFormatUserMem(SparseMatrix &sparseJac) const;

  /**
   * @brief Resize and recover a column-compressed sparse Jacobian.
   */
  void recoverColFormat(SparseMatrix &sparseJac) const;
  /**
   * @brief Split the recovered extended Jacobian into caller-sized ANF blocks.
   *
   * `sparseANF.Y`, `sparseANF.J`, `sparseANF.Z`, and `sparseANF.L` must
   * already contain exactly the number of entries required by the cached
   * abs-normal sparsity pattern. Recovered entries use block-local
   * coordinates.
   */
  void recoverANFUserMem(SparseANF &sparseANF, int indep,
                         int numSwitches) const;
  /**
   * @brief Split the recovered extended Jacobian into abs-normal blocks.
   *
   * The recovered matrix is interpreted as `[Y J; Z L]` with rows `[y ; z]`
   * and columns `[x ; |z|]`. The four sparse blocks are resized
   * automatically and filled with block-local coordinates.
   */
  void recoverANF(SparseANF &sparseANF, int indep, int numSwitches) const;
};

/**
 * @brief Generate a Hessian seed matrix using ColPack.
 *
 * @param dimIn             Hessian dimension.
 * @param HP                Hessian sparsity pattern in CRS form.
 * @param[out] Seed         Seed matrix returned by ColPack.
 * @param[out] p            Compressed dimension of the seed matrix.
 * @param coloringVariant   ColPack coloring strategy.
 */
void generateSeedHess(int dimIn, const std::span<uint *> HP, double ***Seed,
                      int *p, const std::string &coloringVariant);

/**
 * @brief Cached sparse Hessian recovery data stored on a tape.
 *
 * This helper owns the ColPack graph/recovery objects and the compressed
 * buffers needed by the sparse Hessian drivers.
 */
struct ADOLC_API SparseHessInfos {
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  double **Hcomp_{nullptr};
  double ***Xppp_{nullptr};
  double ***Yppp_{nullptr};
  double ***Zppp_{nullptr};
  double **Upp_{nullptr};
  Matrix<double> Hcomp_cont{};
  Tensor<double> Xppp_cont{};
  Tensor<double> Yppp_cont{};
  Tensor<double> Zppp_cont{};
  Matrix<double> Upp_cont{};

  // type is dictated by ColPack
  std::vector<uint *> HP_;

  int nnzIn_{0};
  int indep_{0};
  int p_{0};

public:
  ~SparseHessInfos();
  SparseHessInfos();
  SparseHessInfos(const SparseHessInfos &) = delete;
  SparseHessInfos(SparseHessInfos &&other) noexcept;
  SparseHessInfos &operator=(const SparseHessInfos &) = delete;
  SparseHessInfos &operator=(SparseHessInfos &&other) noexcept;

  std::vector<uint *> &getHP() { return HP_; }
  void setHP(int indep, std::vector<uint *> &&HPIn) {
    // delete HP_
    for (auto &h : HP_)
      delete[] h;
    indep_ = indep;
    HP_ = std::move(HPIn);
  }
  void initColoring(int dimIn);
  void generateSeedHess(double ***Seed, const std::string &coloringVariant);
  void directRecoverUserMem(unsigned int **rind, unsigned int **cind,
                            double **values);
  void indirectRecoverUserMem(unsigned int **rind, unsigned int **cind,
                              double **values);
  void directRecover(unsigned int **rind, unsigned int **cind, double **values);
  void indirectRecover(unsigned int **rind, unsigned int **cind,
                       double **values);
};
} // namespace ADOLC::Sparse
#endif // ADOLC_SPARSE_INFOS_H
