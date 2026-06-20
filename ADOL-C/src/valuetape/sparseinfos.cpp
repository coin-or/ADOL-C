#ifdef ADOLC_SPARSE
#include <ColPack/ColPackHeaders.h>
#include <adolc/adalloc.h>
#include <adolc/sparse/sparsematrix.h>
#include <adolc/valuetape/sparseinfos.h>
#include <span>
namespace ADOLC::Sparse {

template <>
void generateSeedJac<CompressionMode::Column>(int dimOut, int dimIn,
                                              const std::span<uint *> JP,
                                              double ***Seed, int *p) {
  int dummy = 0;
  ColPack::BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, JP.data(),
                                                  dimOut, dimIn)
      .GenerateSeedJacobian_unmanaged(Seed, &dummy, p, "SMALLEST_LAST",
                                      "COLUMN_PARTIAL_DISTANCE_TWO");
}
template <>
void generateSeedJac<CompressionMode::Row>(int dimOut, int dimIn,
                                           const std::span<uint *> JP,
                                           double ***Seed, int *p) {
  int dummy = 0;
  ColPack::BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, JP.data(),
                                                  dimOut, dimIn)
      .GenerateSeedJacobian_unmanaged(Seed, p, &dummy, "SMALLEST_LAST",
                                      "ROW_PARTIAL_DISTANCE_TWO");
}
struct SparseJacInfos::Impl {
  // unique ptr is used, because colpack does not specify copy assignment or
  // constructor...
  std::unique_ptr<ColPack::BipartiteGraphPartialColoringInterface> g_;
  ColPack::JacobianRecovery1D jr1d_{};

  Impl()
      : g_(std::make_unique<ColPack::BipartiteGraphPartialColoringInterface>(
            SRC_WAIT)) {}
};

SparseJacInfos::~SparseJacInfos() {
  y_ = nullptr;
  B_ = nullptr;

  for (auto &j : JP_) {
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

    for (auto &jp : JP_) {
      delete[] jp;
      jp = nullptr;
    }

    // Move resources
    pimpl_ = std::move(other.pimpl_);
    y_ = other.y_;
    Seed_ = other.Seed_;
    B_ = other.B_;
    JP_ = std::move(other.JP_);
    depen_ = other.depen_;
    nnzIn_ = other.nnzIn_;
    seedClms_ = other.seedClms_;
    seedRows_ = other.seedRows_;

    // Null out moved-from object’s pointers
    other.y_ = nullptr;
    other.Seed_ = nullptr;
    other.B_ = nullptr;
    other.depen_ = 0;
    other.nnzIn_ = 0;
    other.seedClms_ = 0;
    other.seedRows_ = 0;
  }
  return *this;
}

void SparseJacInfos::initColoring(int dimOut, int dimIn) {
  pimpl_->g_ =
      std::make_unique<ColPack::BipartiteGraphPartialColoringInterface>(
          SRC_MEM_ADOLC, JP_.data(), dimOut, dimIn);
  pimpl_->jr1d_ = ColPack::JacobianRecovery1D();
}

void SparseJacInfos::generateSeedJac(const std::string &coloringVariant) {
  pimpl_->g_->GenerateSeedJacobian(&Seed_, &seedRows_, &seedClms_,
                                   "SMALLEST_LAST", coloringVariant);
}

void SparseJacInfos::recoverRowFormatUserMem(SparseMatrix &sparseJac) const {
  using coordinate_type = CoordinateFormatTripled::coordinate_type;
  assert(pimpl_->g_.get() != nullptr);
  assert(sparseJac.size() == to_size_t(nnzIn_));

  // The following is ColPack's routine adapted to our CoordinateFormatTriplet
  // layout
  const auto rowCount =
      static_cast<coordinate_type>(pimpl_->g_->GetRowVertexCount());
  std::vector<int> vi_LeftVertexColors;
  pimpl_->g_->GetLeftVertexColors(vi_LeftVertexColors);

  size_t numOfNonZeros_count = 0;
  for (coordinate_type i = 0; i < rowCount; i++) {
    const size_t numOfNonZeros = JP_[i][0];
    for (size_t j = 1; j <= numOfNonZeros; j++) {
      sparseJac[numOfNonZeros_count] = {i, JP_[i][j],
                                        B_[vi_LeftVertexColors[i]][JP_[i][j]]};
      numOfNonZeros_count++;
    }
  }
}

void SparseJacInfos::recoverRowFormat(SparseMatrix &sparseJac) const {
  sparseJac.resize(nnzIn_);
  recoverRowFormatUserMem(sparseJac);
}

void SparseJacInfos::recoverColFormatUserMem(SparseMatrix &sparseJac) const {
  using coordinate_type = CoordinateFormatTripled::coordinate_type;
  assert(pimpl_->g_.get() != nullptr);
  assert(sparseJac.size() == to_size_t(nnzIn_));

  // The following is ColPack's routine adapted to our CoordinateFormatTriplet
  // layout
  const auto rowCount =
      static_cast<coordinate_type>(pimpl_->g_->GetRowVertexCount());
  std::vector<int> vi_RightVertexColors;
  pimpl_->g_->GetRightVertexColors(vi_RightVertexColors);

  size_t numOfNonZeros_count = 0;
  for (coordinate_type i = 0; i < rowCount; i++) {
    const size_t numOfNonZeros = JP_[i][0];
    for (size_t j = 1; j <= numOfNonZeros; j++) {
      sparseJac[numOfNonZeros_count] = {i, JP_[i][j],
                                        B_[i][vi_RightVertexColors[JP_[i][j]]]};
      numOfNonZeros_count++;
    }
  }
}

void SparseJacInfos::recoverColFormat(SparseMatrix &sparseJac) const {
  sparseJac.resize(nnzIn_);
  recoverColFormatUserMem(sparseJac);
}

namespace {
using detail::classifySparseANFBlock;
using detail::SparseANFBlock;
using Coordinates = CoordinateFormatTripled::Coordinates;
using coordinate_type = CoordinateFormatTripled::coordinate_type;

SparseShape countSparseANFEntries(const std::vector<uint *> &JP, int depen,
                                  int indep) {
  SparseShape counts;
  for (coordinate_type row = 0; row < JP.size(); ++row) {
    const auto numOfNonZeros = JP[row][0];
    for (unsigned int j = 1; j <= numOfNonZeros; ++j) {
      switch (classifySparseANFBlock(
          Coordinates{.rowIndex_ = row, .colIndex_ = JP[row][j]}, depen,
          indep)) {
      case SparseANFBlock::Y:
        ++counts.y;
        break;
      case SparseANFBlock::J:
        ++counts.j;
        break;
      case SparseANFBlock::Z:
        ++counts.z;
        break;
      case SparseANFBlock::L:
        ++counts.l;
        break;
      }
    }
  }
  return counts;
}

void fillSparseANF(const std::vector<uint *> JP, double **B,
                   const std::vector<int> &leftVertexColors, int depen,
                   int indep, SparseANF &sparseANF) {

  SparseShape offsets;
  for (coordinate_type row = 0; row < JP.size(); ++row) {
    const auto numOfNonZeros = JP[row][0];
    for (unsigned int j = 1; j <= numOfNonZeros; ++j) {
      const auto col = JP[row][j];
      const auto value = B[leftVertexColors[row]][col];

      Coordinates coords{.rowIndex_ = row, .colIndex_ = col};
      switch (classifySparseANFBlock(coords, depen, indep)) {
      case SparseANFBlock::Y:
        sparseANF.Y[offsets.y++] = CoordinateFormatTripled(coords, value);
        break;
      case SparseANFBlock::J:
        coords = {coords.rowIndex_, coords.colIndex_ - indep};
        sparseANF.J[offsets.j++] = CoordinateFormatTripled(coords, value);
        break;
      case SparseANFBlock::Z:
        coords = {coords.rowIndex_ - depen, col};
        sparseANF.Z[offsets.z++] = CoordinateFormatTripled(coords, value);
        break;
      case SparseANFBlock::L:
        coords = {coords.rowIndex_ - depen, coords.colIndex_ - indep};
        sparseANF.L[offsets.l++] = CoordinateFormatTripled(coords, value);
        break;
      }
    }
  }
}
} // namespace

/**
 * @brief Split the recovered extended Jacobian into caller-provided ANF blocks.
 *
 * @note The recovery loop is based on the same ColPack data used by
 *       `recoverRowFormatUserMem`, but partitions the extended Jacobian into
 *       block-local `Y`, `J`, `Z`, and `L` sparse matrices.
 */
void SparseJacInfos::recoverANFUserMem(SparseANF &sparseANF, int indep,
                                       int numSwitches) const {
  assert(pimpl_->g_.get() != nullptr);
  const int depen = depen_ - numSwitches;
  assert(depen > 0);

  std::vector<int> leftVertexColors;
  pimpl_->g_->GetLeftVertexColors(leftVertexColors);
  fillSparseANF(JP_, B_, leftVertexColors, depen, indep, sparseANF);
}

/**
 * @brief Split the recovered extended Jacobian into the four ANF blocks.
 *
 * @note The recovered entries are partitioned according to the abs-normal
 *       block structure `[Y J; Z L]`, with rows ordered as `[y ; z]` and
 *       columns ordered as `[x ; |z|]`. Entries inside the returned blocks
 *       use block-local coordinates.
 */
void SparseJacInfos::recoverANF(SparseANF &sparseANF, int indep,
                                int numSwitches) const {
  assert(pimpl_->g_.get() != nullptr);
  const int depen = depen_ - numSwitches;
  assert(depen >= 0);
  sparseANF.resize(countSparseANFEntries(JP_, depen, indep));
  recoverANFUserMem(sparseANF, indep, numSwitches);
}

void generateSeedHess(int dimIn, const std::span<uint *> HP, double ***Seed,
                      int *p, const std::string &coloringVariant) {
  int seed_rows = 0;
  ColPack::GraphColoringInterface(SRC_MEM_ADOLC, HP.data(), dimIn)
      .GenerateSeedHessian_unmanaged(Seed, &seed_rows, p, "SMALLEST_LAST",
                                     coloringVariant);
}

struct SparseHessInfos::Impl {
  // unique ptr is used, because colpack does not specify copy assignment or
  // constructor...
  std::unique_ptr<ColPack::GraphColoringInterface> g_;
  ColPack::HessianRecovery hr_;

  Impl() : g_(std::make_unique<ColPack::GraphColoringInterface>(SRC_WAIT)) {}
};

SparseHessInfos::~SparseHessInfos() {
  Hcomp_ = nullptr;

  Xppp_ = nullptr;

  Yppp_ = nullptr;

  Zppp_ = nullptr;

  Upp_ = nullptr;

  for (auto &h : HP_) {
    delete[] h;
    h = nullptr;
  }
}

SparseHessInfos::SparseHessInfos()
    : pimpl_(std::make_unique<SparseHessInfos::Impl>()) {};

SparseHessInfos::SparseHessInfos(SparseHessInfos &&other) noexcept
    : pimpl_(std::move(other.pimpl_)), Hcomp_(nullptr), Xppp_(nullptr),
      Yppp_(nullptr), Zppp_(nullptr), Upp_(nullptr),
      Hcomp_cont(std::move(other.Hcomp_cont)),
      Xppp_cont(std::move(other.Xppp_cont)),
      Yppp_cont(std::move(other.Yppp_cont)),
      Zppp_cont(std::move(other.Zppp_cont)),
      Upp_cont(std::move(other.Upp_cont)), HP_(std::move(other.HP_)),
      nnzIn_(other.nnzIn_), indep_(other.indep_), p_(other.p_) {

  Hcomp_ = Hcomp_cont.data();
  Xppp_ = Xppp_cont.data();
  Yppp_ = Yppp_cont.data();
  Zppp_ = Zppp_cont.data();
  Upp_ = Upp_cont.data();

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

    for (auto &hp : HP_) {
      delete[] hp;
      hp = nullptr;
    }

    // Move resources
    pimpl_ = std::move(other.pimpl_);
    /* Hcomp_ = other.Hcomp_;
    Xppp_ = other.Xppp_;
    Yppp_ = other.Yppp_;
    Zppp_ = other.Zppp_;
    Upp_ = other.Upp_; */
    Hcomp_cont = std::move(other.Hcomp_cont);
    Xppp_cont = std::move(other.Xppp_cont);
    Yppp_cont = std::move(other.Yppp_cont);
    Zppp_cont = std::move(other.Zppp_cont);
    Upp_cont = std::move(other.Upp_cont);
    Hcomp_ = Hcomp_cont.data();
    Xppp_ = Xppp_cont.data();
    Yppp_ = Yppp_cont.data();
    Zppp_ = Zppp_cont.data();
    Upp_ = Upp_cont.data();
    HP_ = std::move(other.HP_);
    nnzIn_ = other.nnzIn_;
    indep_ = other.indep_;
    p_ = other.p_;

    // Null out moved-from object's pointers
    other.Hcomp_ = nullptr;
    other.Xppp_ = nullptr;
    other.Yppp_ = nullptr;
    other.Zppp_ = nullptr;
    other.Upp_ = nullptr;
    other.nnzIn_ = 0;
    other.indep_ = 0;
    other.p_ = 0;
  }
  return *this;
}

void SparseHessInfos::initColoring(int dimIn) {
  pimpl_->g_ = std::make_unique<ColPack::GraphColoringInterface>(
      SRC_MEM_ADOLC, HP_.data(), dimIn);
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
