
#include <adolc/adalloc.h>
#include <adolc/valuetape/persistanttapeinfos.h>

PersistantTapeInfos::~PersistantTapeInfos() {
  free(jacSolv_ci);
  free(jacSolv_ri);
}

PersistantTapeInfos::PersistantTapeInfos(PersistantTapeInfos &&other) noexcept
    : forodec_nax(other.forodec_nax), forodec_dax(other.forodec_dax),
      forodec_y(other.forodec_y), forodec_z(other.forodec_z),
      forodec_Z(other.forodec_Z), jacSolv_J(other.jacSolv_J),
      jacSolv_I(other.jacSolv_I), jacSolv_xold(other.jacSolv_xold),
      jacSolv_ri(other.jacSolv_ri), jacSolv_ci(other.jacSolv_ci),
      jacSolv_nax(other.jacSolv_nax), jacSolv_modeold(other.jacSolv_modeold),
      jacSolv_cgd(other.jacSolv_cgd) {
  other.forodec_y = nullptr;
  other.forodec_z = nullptr;
  other.forodec_Z = nullptr;
  other.jacSolv_J = nullptr;
  other.jacSolv_I = nullptr;
  other.jacSolv_xold = nullptr;
  other.jacSolv_ri = nullptr;
  other.jacSolv_ci = nullptr;
}
PersistantTapeInfos &
PersistantTapeInfos::operator=(PersistantTapeInfos &&other) noexcept {
  if (this != &other) {
    delete[] jacSolv_ri;
    delete[] jacSolv_ci;

    forodec_nax = other.forodec_nax;
    forodec_dax = other.forodec_dax;
    forodec_y = other.forodec_y;
    forodec_z = other.forodec_z;
    forodec_Z = other.forodec_Z;
    jacSolv_J = other.jacSolv_J;
    jacSolv_I = other.jacSolv_I;
    jacSolv_xold = other.jacSolv_xold;
    jacSolv_ri = other.jacSolv_ri;
    jacSolv_ci = other.jacSolv_ci;
    jacSolv_nax = other.jacSolv_nax;
    jacSolv_modeold = other.jacSolv_modeold;
    jacSolv_cgd = other.jacSolv_cgd;

    other.forodec_y = nullptr;
    other.forodec_z = nullptr;
    other.forodec_Z = nullptr;
    other.jacSolv_J = nullptr;
    other.jacSolv_I = nullptr;
    other.jacSolv_xold = nullptr;
    other.jacSolv_ri = nullptr;
    other.jacSolv_ci = nullptr;
  }
  return *this;
}
