#include <adolc/adalloc.h>
#include <adolc/valuetape/sparseinfos.h>

#ifdef SPARSE

namespace ADOLC::Sparse {

SparseJacInfos::~SparseJacInfos() {
  if (y_)
    myfree1(y_);
  y_ = nullptr;
  if (B_)
    myfree2(B_);
  B_ = nullptr;

  for (int i = 0; i < depen_; i++) {
    delete[] JP_[i];
    JP_[i] = nullptr;
  }
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
    g_ = std::move(other.g_);
    jr1d_ = std::move(other.jr1d_);
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

  for (int i = 0; i < indep_; i++) {
    delete[] HP_[i];
    HP_[i] = nullptr;
  }
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
    g_ = std::move(other.g_);
    hr_ = std::move(other.hr_);
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

} // namespace ADOLC::Sparse

#endif
