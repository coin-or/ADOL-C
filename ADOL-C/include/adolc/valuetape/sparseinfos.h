#ifndef ADOLC_SPARSE_INFOS_H
#define ADOLC_SPARSE_INFOS_H
#ifdef SPARSE

#include <ColPack/ColPackHeaders.h>
#include <adolc/internal/common.h>
#include <memory>

namespace ADOLC::Sparse {

// stores everything we need to know to compute the sparse jacobian with
// fov_reverse

struct ADOLC_API SparseJacInfos {
  std::unique_ptr<ColPack::BipartiteGraphPartialColoringInterface> g_{nullptr};
  std::unique_ptr<ColPack::JacobianRecovery1D> jr1d_{nullptr};
  double *y_{nullptr};

  // Seed is memory managed by ColPack and will be deleted
  double **Seed_{nullptr};
  double **B_{nullptr};
  unsigned int **JP_{nullptr};
  int depen_{0};
  int nnzIn_{0};
  int seedClms_{0};
  int seedRows_{0};

  ~SparseJacInfos();
  SparseJacInfos() = default;

  SparseJacInfos(const SparseJacInfos &) = delete;
  SparseJacInfos &operator=(const SparseJacInfos &) = delete;

  SparseJacInfos(SparseJacInfos &&other) noexcept
      : g_(std::move(other.g_)), jr1d_(std::move(other.jr1d_)), y_(other.y_),
        Seed_(other.Seed_), B_(other.B_), JP_(other.JP_), depen_(other.depen_),
        nnzIn_(other.nnzIn_), seedClms_(other.seedClms_),
        seedRows_(other.seedRows_) {

    // Null out source object's pointers to prevent double deletion
    other.g_ = nullptr;
    other.jr1d_ = nullptr;
    other.y_ = nullptr;
    other.Seed_ = nullptr;
    other.B_ = nullptr;
    other.JP_ = nullptr;
  }

  SparseJacInfos &operator=(SparseJacInfos &&other) noexcept;
  void setJP(unsigned int **JPIn) { JP_ = JPIn; }
  unsigned int **getJP() { return JP_; }
};

// stores everything we have to know to compute the sparse hessian via
// reverse-over-forward
struct ADOLC_API SparseHessInfos {

  std::unique_ptr<ColPack::GraphColoringInterface> g_{nullptr};
  std::unique_ptr<ColPack::HessianRecovery> hr_{nullptr};

  double **Hcomp_{nullptr};
  double ***Xppp_{nullptr};
  double ***Yppp_{nullptr};
  double ***Zppp_{nullptr};
  double **Upp_{nullptr};

  unsigned int **HP_{nullptr};

  int nnzIn_{0};
  int indep_{0};
  int p_{0};

public:
  ~SparseHessInfos();
  SparseHessInfos() = default;
  SparseHessInfos(const SparseHessInfos &) = delete;
  SparseHessInfos &operator=(const SparseHessInfos &) = delete;

  SparseHessInfos(SparseHessInfos &&other) noexcept
      : g_(std::move(other.g_)), hr_(std::move(other.hr_)),
        Hcomp_(other.Hcomp_), Xppp_(other.Xppp_), Yppp_(other.Yppp_),
        Zppp_(other.Zppp_), Upp_(other.Upp_), HP_(other.HP_),
        nnzIn_(other.nnzIn_), indep_(other.indep_), p_(other.p_) {

    // Null out moved-from object's pointers
    other.g_ = nullptr;
    other.hr_ = nullptr;
    other.Hcomp_ = nullptr;
    other.Xppp_ = nullptr;
    other.Yppp_ = nullptr;
    other.Zppp_ = nullptr;
    other.Upp_ = nullptr;
    other.HP_ = nullptr;
  }

  SparseHessInfos &operator=(SparseHessInfos &&other) noexcept;
  void setHP(int indep, unsigned int **HP);
  void getHP(unsigned int ***HP);
  static void deepCopyHP(unsigned int ***HPOut, unsigned int **HPIn, int indep);
};

} // namespace ADOLC::Sparse

#endif // SPARSE
#endif // ADOLC_SPARSE_INFOS_H
