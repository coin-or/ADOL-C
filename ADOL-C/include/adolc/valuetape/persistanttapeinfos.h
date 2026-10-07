
#ifndef ADOLC_PERSISTANT_TAPE_INFOS_H
#define ADOLC_PERSISTANT_TAPE_INFOS_H

/**
 * @brief Reusable scratch storage retained by higher-level tape drivers.
 *
 * The buffers cache allocations used by ODE and Jacobian routines across
 * repeated calls. The object owns the raw arrays and is movable but not
 * copyable.
 */
struct PersistantTapeInfos {

  ~PersistantTapeInfos();
  PersistantTapeInfos() = default;

  PersistantTapeInfos(const PersistantTapeInfos &) = delete;
  PersistantTapeInfos &operator=(const PersistantTapeInfos &) = delete;

  PersistantTapeInfos(PersistantTapeInfos &&other) noexcept;
  PersistantTapeInfos &operator=(PersistantTapeInfos &&other) noexcept;

  int forodec_nax{0};
  int forodec_dax{0};
  double *forodec_y{nullptr};
  double *forodec_z{nullptr};
  double **forodec_Z{nullptr};
  double **jacSolv_J{nullptr};
  double **jacSolv_I{nullptr};
  double *jacSolv_xold{nullptr};
  int *jacSolv_ri{nullptr};
  int *jacSolv_ci{nullptr};
  int jacSolv_nax{0};
  int jacSolv_modeold{0};
  int jacSolv_cgd{0};
};

#endif // ADOLC_PERSISTANT_TAPE_INFOS_H
