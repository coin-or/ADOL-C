
#ifndef ADOLC_PERSISTANT_TAPE_INFOS_H
#define ADOLC_PERSISTANT_TAPE_INFOS_H

#include <adolc/adalloc.h>
/**
 * @brief Reusable scratch storage retained by higher-level tape drivers.
 *
 * The buffers cache allocations used by ODE and Jacobian routines across
 * repeated calls. The object owns the raw arrays and is movable but not
 * copyable.
 */
struct PersistantTapeInfos {

  ~PersistantTapeInfos() = default;
  PersistantTapeInfos() = default;

  PersistantTapeInfos(const PersistantTapeInfos &) = delete;
  PersistantTapeInfos &operator=(const PersistantTapeInfos &) = delete;

  PersistantTapeInfos(PersistantTapeInfos &&other) noexcept = default;
  PersistantTapeInfos &
  operator=(PersistantTapeInfos &&other) noexcept = default;

  struct ForwardODEData {
    int forodec_nax{0};
    int forodec_dax{0};
    std::vector<double> forodec_y;
    std::vector<double> forodec_z;
    Matrix<double> forodec_Z;
  };
  ForwardODEData ODEData_;

  struct JacSolveData {
    int jacSolv_nax{0};
    int jacSolv_modeold{0};
    Matrix<double> jacSolv_J;
    Matrix<double> jacSolv_I;
    std::vector<double> jacSolv_xold;
    std::vector<int> jacSolv_ri;
    std::vector<int> jacSolv_ci;
  };
  JacSolveData jacData_;
};

#endif // ADOLC_PERSISTANT_TAPE_INFOS_H
