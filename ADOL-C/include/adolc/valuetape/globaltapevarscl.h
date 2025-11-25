#ifndef ADOLC_GLOBALTAPEVARSCL_H
#define ADOLC_GLOBALTAPEVARSCL_H

#include <adolc/storemanager.h>
#include <adolc/valuetape/tapeinfos.h>
#include <limits>

struct GlobalTapeVarsCL {
  GlobalTapeVarsCL()
      : paramStoreMgrPtr(std::make_unique<StoreManagerLocintBlock>(
            pStore, maxparam, numparam)),
#ifdef ADOLC_TRACK_ACTIVITY
        storeManagerPtr(std::make_unique<StoreManagerLocintBlock>(
            store, actStore, storeSize, numLives))
#else
        storeManagerPtr(std::make_unique<StoreManagerLocintBlock>(
            store, storeSize, numLives))
#endif
  {};

  ~GlobalTapeVarsCL() = default;

  GlobalTapeVarsCL(GlobalTapeVarsCL &&other) noexcept;
  GlobalTapeVarsCL &operator=(GlobalTapeVarsCL &&other) noexcept;

  GlobalTapeVarsCL(const GlobalTapeVarsCL &) = delete;
  GlobalTapeVarsCL &operator=(const GlobalTapeVarsCL &) = delete;

  void reallocStore(unsigned char type);
  void checkInitialStoreSize() {
    if (initialStoreSize > StoreManager::get_initialSize())
      storeManagerPtr->grow(initialStoreSize);
  }

  // double store for calc. while taping
  double *store{nullptr};
#if defined(ADOLC_TRACK_ACTIVITY)
  // activity store for tracking while taping
  char *actStore{nullptr};
#endif
  size_t storeSize{0};
  size_t numLives{0};
  size_t maxLoc{std::numeric_limits<size_t>::max()};

  // Defaults to the value specified in
  // usrparms.h. May be overwritten by values
  // in a local config file .adolcrc.
  size_t operationBufferSize{0};
  size_t locationBufferSize{0};
  size_t valueBufferSize{0};
  size_t taylorBufferSize{0};
  size_t maxNumberTaylorBuffers{0};

  // set to 1 if in an OpenMP parallel region
  char inParallelRegion{0};
  // signals: at least one tape created (0/1)
  char newTape{0};
  char branchSwitchWarning{1};
  uint nominmaxFlag{0};
  size_t numparam{0};
  size_t maxparam{0};
  double *pStore{nullptr};
  size_t initialStoreSize{0};
  std::unique_ptr<StoreManager> paramStoreMgrPtr;
  std::unique_ptr<StoreManager> storeManagerPtr;
};
#endif // ADOLC_GLOBALTAPEVARSCL_H
