#ifndef ADOLC_SPARSE_INFOS_H
#define ADOLC_SPARSE_INFOS_H
#include <adolc/internal/common.h>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ADOLC::Sparse {
void generateSeedJac(int dimOut, int dimIn, const std::span<uint *> JP,
                     double ***Seed, int *p,
                     const std::string &coloringVariant);
// stores everything we need to know to compute the sparse jacobian with
// fov_reverse

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

  void setJP(std::vector<uint *> &&JPIn) {
    // delete JP
    for (auto& j: JP_)
      delete[] j;
    JP_ = std::move(JPIn);
    depen_ = JP_.size();
  }
  std::vector<uint *> &getJP() { return JP_; }
  void initColoring(int dimOut, int dimIn);
  void generateSeedJac(const std::string &coloringVariant);
  void recoverRowFormatUserMem(unsigned int **rind, unsigned int **cind,
                               double **values);
  void recoverColFormatUserMem(unsigned int **rind, unsigned int **cind,
                               double **values);
  void recoverRowFormat(unsigned int **rind, unsigned int **cind,
                        double **values);
  void recoverColFormat(unsigned int **rind, unsigned int **cind,
                        double **values);
};

void generateSeedHess(int dimIn, const std::span<uint *> HP, double ***Seed,
                      int *p, const std::string &coloringVariant);
// stores everything we have to know to compute the sparse hessian via
// reverse-over-forward
struct ADOLC_API SparseHessInfos {
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  double **Hcomp_{nullptr};
  double ***Xppp_{nullptr};
  double ***Yppp_{nullptr};
  double ***Zppp_{nullptr};
  double **Upp_{nullptr};

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
    for (auto& h: HP_)
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
