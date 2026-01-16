#ifdef ADOLC_SPARSE
#include <ColPack/ColPackHeaders.h>
#include <adolc/adalloc.h>
#include <adolc/valuetape/sparseinfos.h>
#include <span>
namespace ADOLC::Sparse {

void generateSeedJac(int dimOut, int dimIn, const std::span<uint *> JP,
                     double ***Seed, int *p,
                     const std::string &coloringVariant) {
  int dummy = 0;
  ColPack::BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, JP.data(),
                                                  dimOut, dimIn)
      .GenerateSeedJacobian_unmanaged(Seed, p, &dummy, "SMALLEST_LAST",
                                      coloringVariant);
}
struct SparseJacInfos::Impl {
  // unique ptr is used, because colpack does not specify copy assignment or constructor...
  std::unique_ptr<ColPack::BipartiteGraphPartialColoringInterface> g_;
  ColPack::JacobianRecovery1D jr1d_{};

  Impl() : g_(std::make_unique<ColPack::BipartiteGraphPartialColoringInterface>(SRC_WAIT)) {}
};

SparseJacInfos::~SparseJacInfos() {
  if (y_)
    myfree1(y_);
  y_ = nullptr;
  if (B_)
    myfree2(B_);
  B_ = nullptr;

  for (auto& j: JP_){
    delete[] j;
    j = nullptr;
  }
}

SparseJacInfos::SparseJacInfos()
    : pimpl_(std::make_unique<SparseJacInfos::Impl>()) {}

SparseJacInfos::SparseJacInfos(SparseJacInfos &&other) noexcept
    : pimpl_(std::move(other.pimpl_)), y_(other.y_), Seed_(other.Seed_),
      B_(other.B_), JP_(std::move(other.JP_)), depen_(other.depen_),
      nnzIn_(other.nnzIn_), seedClms_(other.seedClms_),
      seedRows_(other.seedRows_) {

  // Null out source object's pointers to prevent double deletion
  other.y_ = nullptr;
  other.Seed_ = nullptr;
  other.B_ = nullptr;
}

SparseJacInfos &SparseJacInfos::operator=(SparseJacInfos &&other) noexcept {
  if (this != &other) {
    // Free existing resources

    myfree1(y_);
    myfree2(B_);
    for (int i = 0; i < depen_; i++) {
      delete[] JP_[i];
      JP_[i] = nullptr;
    }

    // Move resources
    pimpl_ = std::move(other.pimpl_);
    y_ = other.y_;
    Seed_ = other.Seed_;
    B_ = other.B_;
    JP_ = other.JP_;
    depen_ = other.depen_;
    nnzIn_ = other.nnzIn_;
    seedClms_ = other.seedClms_;
    seedRows_ = other.seedRows_;

    // Null out moved-from object’s pointers
    other.y_ = nullptr;
    other.Seed_ = nullptr;
    other.B_ = nullptr;
  }
  return *this;
}

void SparseJacInfos::initColoring(int dimOut, int dimIn) {
  pimpl_->g_ = std::make_unique<ColPack::BipartiteGraphPartialColoringInterface>(
      SRC_MEM_ADOLC, JP_.data(), dimOut, dimIn);
  pimpl_->jr1d_ = ColPack::JacobianRecovery1D();
}

void SparseJacInfos::generateSeedJac(const std::string &coloringVariant) {
  pimpl_->g_->GenerateSeedJacobian(&Seed_, &seedRows_, &seedClms_,
                                  "SMALLEST_LAST", coloringVariant);
}

void SparseJacInfos::recoverRowFormatUserMem(unsigned int **rind,
                                             unsigned int **cind,
                                             double **values) {
  pimpl_->jr1d_.RecoverD2Row_CoordinateFormat_usermem(
      pimpl_->g_.get(), B_, JP_.data(), rind, cind, values);
}

void SparseJacInfos::recoverColFormatUserMem(unsigned int **rind,
                                             unsigned int **cind,
                                             double **values) {
  pimpl_->jr1d_.RecoverD2Cln_CoordinateFormat_usermem(
      pimpl_->g_.get(), B_, JP_.data(), rind, cind, values);
}

void SparseJacInfos::recoverRowFormat(unsigned int **rind, unsigned int **cind,
                                      double **values) {
  pimpl_->jr1d_.RecoverD2Row_CoordinateFormat_unmanaged(
      pimpl_->g_.get(), B_, JP_.data(), rind, cind, values);
}

void SparseJacInfos::recoverColFormat(unsigned int **rind, unsigned int **cind,
                                      double **values) {
  pimpl_->jr1d_.RecoverD2Cln_CoordinateFormat_unmanaged(
      pimpl_->g_.get(), B_, JP_.data(), rind, cind, values);
}

void generateSeedHess(int dimIn, const std::span<uint *> HP, double ***Seed,
                      int *p, const std::string &coloringVariant) {
  int seed_rows = 0;
  ColPack::GraphColoringInterface(SRC_MEM_ADOLC, HP.data(), dimIn)
      .GenerateSeedHessian_unmanaged(Seed, &seed_rows, p, "SMALLEST_LAST",
                                     coloringVariant);
}

struct SparseHessInfos::Impl {
  // unique ptr is used, because colpack does not specify copy assignment or constructor...
  std::unique_ptr<ColPack::GraphColoringInterface> g_;
  ColPack::HessianRecovery hr_;

  Impl() : g_(std::make_unique<ColPack::GraphColoringInterface>(SRC_WAIT)) {}
};

SparseHessInfos::~SparseHessInfos() {
  myfree2(Hcomp_);
  Hcomp_ = nullptr;

  myfree3(Xppp_);
  Xppp_ = nullptr;

  myfree3(Yppp_);
  Yppp_ = nullptr;

  myfree3(Zppp_);
  Zppp_ = nullptr;

  myfree2(Upp_);
  Upp_ = nullptr;

  for(auto& h: HP_){
    delete[] h;
    h = nullptr;
  }
}

SparseHessInfos::SparseHessInfos()
    : pimpl_(std::make_unique<SparseHessInfos::Impl>()) {};

SparseHessInfos::SparseHessInfos(SparseHessInfos &&other) noexcept
    : pimpl_(std::move(other.pimpl_)), Hcomp_(other.Hcomp_), Xppp_(other.Xppp_),
      Yppp_(other.Yppp_), Zppp_(other.Zppp_), Upp_(other.Upp_),
      HP_(std::move(other.HP_)), nnzIn_(other.nnzIn_), indep_(other.indep_),
      p_(other.p_) {

  // Null out moved-from object's pointers
  other.Hcomp_ = nullptr;
  other.Xppp_ = nullptr;
  other.Yppp_ = nullptr;
  other.Zppp_ = nullptr;
  other.Upp_ = nullptr;
}

SparseHessInfos &SparseHessInfos::operator=(SparseHessInfos &&other) noexcept {
  if (this != &other) {
    // Free existing resources
    myfree2(Hcomp_);
    myfree3(Xppp_);
    myfree3(Yppp_);
    myfree3(Zppp_);
    myfree2(Upp_);

    for (int i = 0; i < indep_; i++) {
      delete[] HP_[i];
      HP_[i] = nullptr;
    }

    // Move resources
    pimpl_ = std::move(other.pimpl_);
    Hcomp_ = other.Hcomp_;
    Xppp_ = other.Xppp_;
    Yppp_ = other.Yppp_;
    Zppp_ = other.Zppp_;
    Upp_ = other.Upp_;
    HP_ = other.HP_;
    nnzIn_ = other.nnzIn_;
    indep_ = other.indep_;
    p_ = other.p_;

    // Null out moved-from object's pointers
    other.Hcomp_ = nullptr;
    other.Xppp_ = nullptr;
    other.Yppp_ = nullptr;
    other.Zppp_ = nullptr;
    other.Upp_ = nullptr;
  }
  return *this;
}

void SparseHessInfos::initColoring(int dimIn) {
  pimpl_->g_ =
      std::make_unique<ColPack::GraphColoringInterface>(SRC_MEM_ADOLC, HP_.data(), dimIn);
  pimpl_->hr_ = ColPack::HessianRecovery();
}

void SparseHessInfos::generateSeedHess(double ***Seed,
                                       const std::string &coloringVariant) {
  int dummy = 0;
  pimpl_->g_->GenerateSeedHessian(Seed, &dummy, &p_, "SMALLEST_LAST",
                                 coloringVariant);
}

void SparseHessInfos::indirectRecoverUserMem(unsigned int **rind,
                                             unsigned int **cind,
                                             double **values) {
  pimpl_->hr_.IndirectRecover_CoordinateFormat_usermem(
      pimpl_->g_.get(), Hcomp_, HP_.data(), rind, cind, values);
}

void SparseHessInfos::directRecoverUserMem(unsigned int **rind,
                                           unsigned int **cind,
                                           double **values) {
  pimpl_->hr_.DirectRecover_CoordinateFormat_usermem(
      pimpl_->g_.get(), Hcomp_, HP_.data(), rind, cind, values);
}

void SparseHessInfos::indirectRecover(unsigned int **rind, unsigned int **cind,
                                      double **values) {
  pimpl_->hr_.IndirectRecover_CoordinateFormat_unmanaged(
      pimpl_->g_.get(), Hcomp_, HP_.data(), rind, cind, values);
}

void SparseHessInfos::directRecover(unsigned int **rind, unsigned int **cind,
                                    double **values) {
  pimpl_->hr_.DirectRecover_CoordinateFormat_unmanaged(
      pimpl_->g_.get(), Hcomp_, HP_.data(), rind, cind, values);
}

} // namespace ADOLC::Sparse
#endif // ADOLC_SPARSE
