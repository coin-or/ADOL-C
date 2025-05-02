#include <adolc/dvlparms.h>
#include <adolc/valuetape/globaltapevarscl.h>
#include <iostream>
#include <string>

GlobalTapeVarsCL::GlobalTapeVarsCL(GlobalTapeVarsCL &&other) noexcept
    : store(other.store),
#if defined(ADOLC_TRACK_ACTIVITY)
      actStore(other.actStore),
#endif
      storeSize(other.storeSize), numLives(other.numLives),
      maxLoc(other.maxLoc),

      operationBufferSize(other.operationBufferSize),
      locationBufferSize(other.locationBufferSize),
      valueBufferSize(other.valueBufferSize),
      taylorBufferSize(other.taylorBufferSize),
      maxNumberTaylorBuffers(other.maxNumberTaylorBuffers),

      inParallelRegion(other.inParallelRegion), newTape(other.newTape),
      branchSwitchWarning(other.branchSwitchWarning),
      nominmaxFlag(other.nominmaxFlag), numparam(other.numparam),
      maxparam(other.maxparam),

      pStore(other.pStore), initialStoreSize(other.initialStoreSize),
      paramStoreMgrPtr(std::move(other.paramStoreMgrPtr)),
      storeManagerPtr(std::move(other.storeManagerPtr)) {
  // Null out pointers in the source object to prevent double free
  other.store = nullptr;
#if defined(ADOLC_TRACK_ACTIVITY)
  other.actStore = nullptr;
#endif
  other.pStore = nullptr;
}

GlobalTapeVarsCL &
GlobalTapeVarsCL::operator=(GlobalTapeVarsCL &&other) noexcept {
  if (this != &other) {
    // Free existing resources to prevent memory leaks
    delete[] store;
#if defined(ADOLC_TRACK_ACTIVITY)
    delete[] actStore;
#endif
    delete[] pStore;

    // Move data
    // move mamangers first, since they hold references to storeSize, etc
    // otherwise storeSize, etc will be overwritten
    paramStoreMgrPtr = std::move(other.paramStoreMgrPtr);
    storeManagerPtr = std::move(other.storeManagerPtr);
    store = other.store;
#if defined(ADOLC_TRACK_ACTIVITY)
    actStore = other.actStore;
#endif
    storeSize = other.storeSize;
    numLives = other.numLives;
    maxLoc = other.maxLoc;
    operationBufferSize = other.operationBufferSize;
    locationBufferSize = other.locationBufferSize;
    valueBufferSize = other.valueBufferSize;
    taylorBufferSize = other.taylorBufferSize;
    maxNumberTaylorBuffers = other.maxNumberTaylorBuffers;
    inParallelRegion = other.inParallelRegion;
    newTape = other.newTape;
    branchSwitchWarning = other.branchSwitchWarning;
    nominmaxFlag = other.nominmaxFlag;
    numparam = other.numparam;
    maxparam = other.maxparam;
    pStore = other.pStore;
    initialStoreSize = other.initialStoreSize;

    // Null out source object to prevent double free
    other.store = nullptr;
#if defined(ADOLC_TRACK_ACTIVITY)
    other.actStore = nullptr;
#endif
    other.pStore = nullptr;
  }
  return *this;
}

void GlobalTapeVarsCL::reallocStore(unsigned char type) {
  delete[] store;
  store = nullptr;
#if defined(ADOLC_TRACK_ACTIVITY)
  actStore = nullptr;
#endif
  storeSize = 0;
  numLives = 0;
  switch (type) {
  case StoreManager::ADOLC_LOCATION_BLOCKS:
#if defined(ADOLC_TRACK_ACTIVITY)
    storeManagerPtr =
        new StoreManagerLocintBlock(store, actStore, storeSize, numLives);
#else
    storeManagerPtr =
        std::make_unique<StoreManagerLocintBlock>(store, storeSize, numLives);
#endif
    break;
  case StoreManager::ADOLC_LOCATION_SINGLETONS:
#if defined(ADOLC_TRACK_ACTIVITY)
    storeManagerPtr =
        new StoreManagerLocint(store, actStore, storeSize, numLives);
#else
    storeManagerPtr =
        std::make_unique<StoreManagerLocint>(store, storeSize, numLives);
#endif
    break;
  }
}
