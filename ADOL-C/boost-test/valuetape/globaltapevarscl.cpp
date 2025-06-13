#define BOOST_TEST_DYN_LINK
#include "../const.h"
#include <adolc/adolc.h>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Test_GlobalTapeVarsCL)

BOOST_AUTO_TEST_CASE(TestDefaultConstructorInitializesPointers) {
  GlobalTapeVarsCL obj;

  // Expect unique_ptrs to be non-null
  BOOST_CHECK(obj.paramStoreMgrPtr != nullptr);
  BOOST_CHECK(obj.paramStoreMgrPtr->maxSize() == obj.maxparam);
  BOOST_CHECK(obj.paramStoreMgrPtr->size() == obj.numparam);

  BOOST_CHECK(obj.storeManagerPtr != nullptr);

  ++obj.storeSize;
  BOOST_CHECK(obj.storeManagerPtr->maxSize() == obj.storeSize);

  ++obj.numLives;
  BOOST_CHECK(obj.storeManagerPtr->size() == obj.numLives);
}

BOOST_AUTO_TEST_CASE(TestMoveConstructorTransfersOwnership) {
  GlobalTapeVarsCL obj1;
  double *a = new double[10];
  double *b = new double[17];
  obj1.store = a;
  obj1.storeSize = 10;
  obj1.numLives = 5;

  obj1.operationBufferSize = 11;
  obj1.locationBufferSize = 12;
  obj1.valueBufferSize = 13;
  obj1.taylorBufferSize = 14;
  obj1.maxNumberTaylorBuffers = 15;

  obj1.inParallelRegion = 1;
  obj1.newTape = 1;
  obj1.branchSwitchWarning = 0;
  obj1.nominmaxFlag = 9;

  obj1.pStore = b;
  obj1.numparam = 16;
  obj1.maxparam = 17;

  obj1.initialStoreSize = 3;

  GlobalTapeVarsCL obj2(std::move(obj1));
  BOOST_CHECK(obj1.paramStoreMgrPtr == nullptr);
  BOOST_CHECK(obj1.storeManagerPtr == nullptr);

  BOOST_CHECK(obj2.store == a);
  BOOST_CHECK(obj2.storeSize == 10);
  BOOST_CHECK(obj2.numLives == 5);

  BOOST_CHECK(obj2.operationBufferSize == 11);
  BOOST_CHECK(obj2.locationBufferSize == 12);
  BOOST_CHECK(obj2.valueBufferSize == 13);
  BOOST_CHECK(obj2.taylorBufferSize == 14);
  BOOST_CHECK(obj2.maxNumberTaylorBuffers == 15);

  BOOST_CHECK(obj2.inParallelRegion == 1);
  BOOST_CHECK(obj2.newTape == 1);
  BOOST_CHECK(obj2.branchSwitchWarning == 0);
  BOOST_CHECK(obj2.nominmaxFlag == 9);

  BOOST_CHECK(obj2.pStore == b);
  BOOST_CHECK(obj2.numparam == 16);
  BOOST_CHECK(obj2.maxparam == 17);

  BOOST_CHECK(obj2.initialStoreSize == 3);

  BOOST_CHECK(obj1.store == nullptr);
  BOOST_CHECK(obj1.pStore == nullptr);

  delete[] a;
  delete[] b;
}

BOOST_AUTO_TEST_CASE(TestMoveAssignmentTransfersOwnership) {
  GlobalTapeVarsCL obj1;
  double *a = new double[10];
  double *b = new double[17];
  obj1.store = a;
  obj1.storeSize = 10;
  obj1.numLives = 5;

  obj1.operationBufferSize = 11;
  obj1.locationBufferSize = 12;
  obj1.valueBufferSize = 13;
  obj1.taylorBufferSize = 14;
  obj1.maxNumberTaylorBuffers = 15;

  obj1.inParallelRegion = 1;
  obj1.newTape = 1;
  obj1.branchSwitchWarning = 0;
  obj1.nominmaxFlag = 9;

  obj1.pStore = b;
  obj1.numparam = 16;
  obj1.maxparam = 17;

  obj1.initialStoreSize = 3;

  GlobalTapeVarsCL obj2;
  obj2 = std::move(obj1);

  BOOST_CHECK(obj2.store == a);
  BOOST_CHECK(obj2.storeSize == 10);
  BOOST_CHECK(obj2.numLives == 5);

  BOOST_CHECK(obj2.operationBufferSize == 11);
  BOOST_CHECK(obj2.locationBufferSize == 12);
  BOOST_CHECK(obj2.valueBufferSize == 13);
  BOOST_CHECK(obj2.taylorBufferSize == 14);
  BOOST_CHECK(obj2.maxNumberTaylorBuffers == 15);

  BOOST_CHECK(obj2.inParallelRegion == 1);
  BOOST_CHECK(obj2.newTape == 1);
  BOOST_CHECK(obj2.branchSwitchWarning == 0);
  BOOST_CHECK(obj2.nominmaxFlag == 9);

  BOOST_CHECK(obj2.pStore == b);
  BOOST_CHECK(obj2.numparam == 16);
  BOOST_CHECK(obj2.maxparam == 17);

  BOOST_CHECK(obj2.initialStoreSize == 3);
  BOOST_CHECK(obj2.storeManagerPtr->maxSize() == obj2.storeSize);
  BOOST_CHECK(obj2.storeManagerPtr->size() == obj2.numLives);

  delete[] a;
  delete[] b;
}

BOOST_AUTO_TEST_CASE(TestReallocStore) {
  GlobalTapeVarsCL obj;

  obj.store = new double[10];
  obj.storeSize = 10;
  obj.numLives = 4;

  obj.reallocStore(StoreManager::ADOLC_LOCATION_BLOCKS);

  BOOST_CHECK(obj.store == nullptr);
  BOOST_CHECK(obj.storeSize == 0);
  BOOST_CHECK(obj.numLives == 0);

  obj.storeManagerPtr->next_loc();
  BOOST_CHECK(obj.store != nullptr);
  BOOST_CHECK(obj.storeSize == obj.storeManagerPtr->get_initialSize());
  BOOST_CHECK(obj.numLives == 1);

  GlobalTapeVarsCL obj2;

  obj2.store = new double[10];
  obj2.storeSize = 10;
  obj2.numLives = 4;

  obj2.reallocStore(StoreManager::ADOLC_LOCATION_SINGLETONS);

  BOOST_CHECK(obj2.store == nullptr);
  BOOST_CHECK(obj2.storeSize == 0);
  BOOST_CHECK(obj2.numLives == 0);
}

BOOST_AUTO_TEST_CASE(TestCheckInitialStoreSize) {
  GlobalTapeVarsCL obj;

  double *a = new double[10];
  obj.store = a;
  obj.storeSize = 10;
  obj.numLives = 4;
  obj.initialStoreSize = 10;

  obj.checkInitialStoreSize();

  BOOST_CHECK(obj.store != a);
  BOOST_CHECK(obj.storeSize == 20);
  BOOST_CHECK(obj.numLives == 4);
}

BOOST_AUTO_TEST_SUITE_END()
