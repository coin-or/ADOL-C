
#include <adolc/adalloc.h>
#include <adolc/valuetape/persistanttapeinfos.h>
#include <iostream>
#include <string>

// handling file names over different threads
#include <atomic>
#include <thread>

PersistantTapeInfos::~PersistantTapeInfos() {
  delete[] paramstore;
  free(jacSolv_ci);
  free(jacSolv_ri);
  myfree1(jacSolv_xold);
  if (jacSolv_nax)
    myfreeI2(jacSolv_nax, jacSolv_I);
  myfree2(jacSolv_J);

  myfree2(forodec_Z);
  myfree1(forodec_z);
  myfree1(forodec_y);

  if (op_fileName) {
    if (!(keepTape && skipFileCleanup))
      remove(op_fileName);
    delete[] op_fileName;
    op_fileName = nullptr;
  }
  if (val_fileName) {
    if (!(keepTape && skipFileCleanup))
      remove(val_fileName);
    delete[] val_fileName;
    val_fileName = nullptr;
  }

  if (loc_fileName) {
    if (!(keepTape && skipFileCleanup))
      remove(loc_fileName);
    delete[] loc_fileName;
    loc_fileName = nullptr;
  }

  if (tay_fileName) {
    if (!(keepTape && skipFileCleanup))
      remove(tay_fileName);
    delete[] tay_fileName;
    tay_fileName = nullptr;
  }
}

PersistantTapeInfos::PersistantTapeInfos(PersistantTapeInfos &&other) noexcept
    : forodec_nax(other.forodec_nax), forodec_dax(other.forodec_dax),
      forodec_y(other.forodec_y), forodec_z(other.forodec_z),
      forodec_Z(other.forodec_Z), jacSolv_J(other.jacSolv_J),
      jacSolv_I(other.jacSolv_I), jacSolv_xold(other.jacSolv_xold),
      jacSolv_ri(other.jacSolv_ri), jacSolv_ci(other.jacSolv_ci),
      jacSolv_nax(other.jacSolv_nax), jacSolv_modeold(other.jacSolv_modeold),
      jacSolv_cgd(other.jacSolv_cgd), op_fileName(other.op_fileName),
      loc_fileName(other.loc_fileName), val_fileName(other.val_fileName),
      tay_fileName(other.tay_fileName), keepTape(other.keepTape),
      skipFileCleanup(other.skipFileCleanup), paramstore(other.paramstore) {

  other.forodec_y = nullptr;
  other.forodec_z = nullptr;
  other.forodec_Z = nullptr;
  other.jacSolv_J = nullptr;
  other.jacSolv_I = nullptr;
  other.jacSolv_xold = nullptr;
  other.jacSolv_ri = nullptr;
  other.jacSolv_ci = nullptr;

  // file names
  other.op_fileName = nullptr;
  other.loc_fileName = nullptr;
  other.val_fileName = nullptr;
  other.tay_fileName = nullptr;

  other.paramstore = nullptr;
}
PersistantTapeInfos &
PersistantTapeInfos::operator=(PersistantTapeInfos &&other) noexcept {
  if (this != &other) {

    myfree1(forodec_y);
    myfree1(forodec_z);
    myfree2(forodec_Z);
    myfree2(jacSolv_J);
    if (jacSolv_nax)
      myfreeI2(jacSolv_nax, jacSolv_I);
    delete[] jacSolv_xold;
    delete[] jacSolv_ri;
    delete[] jacSolv_ci;

    // file names
    delete[] op_fileName;
    delete[] loc_fileName;
    delete[] val_fileName;
    delete[] tay_fileName;
    delete[] paramstore;

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
    op_fileName = other.op_fileName;
    loc_fileName = other.loc_fileName;
    val_fileName = other.val_fileName;
    tay_fileName = other.tay_fileName;
    keepTape = other.keepTape;
    skipFileCleanup = other.skipFileCleanup;
    paramstore = other.paramstore;
  }

  other.forodec_y = nullptr;
  other.forodec_z = nullptr;
  other.forodec_Z = nullptr;
  other.jacSolv_J = nullptr;
  other.jacSolv_I = nullptr;
  other.jacSolv_xold = nullptr;
  other.jacSolv_ri = nullptr;
  other.jacSolv_ci = nullptr;

  // file names
  other.op_fileName = nullptr;
  other.loc_fileName = nullptr;
  other.val_fileName = nullptr;
  other.tay_fileName = nullptr;

  other.paramstore = nullptr;
  return *this;
}

/**
 * @brief Generates an id for the thread within the function is called
 *
 * @return id of the current thread
 */
int getThreadIndex() {
  static std::atomic<int> nextId{0};
  thread_local int id = nextId++;
  return id;
}
/****************************************************************************/
/* Returns the char*: tapeBaseName+thread-threadNumber+tapeId+.tap+\0       */
/* The result string must be freed be the caller!                           */
/****************************************************************************/
char *PersistantTapeInfos::createFileName(short tapeId, int tapeType) {
  std::string fileName(tapeBaseNames_[tapeType]);

  int threadId = getThreadIndex();
  fileName += "thread-" + std::to_string(threadId) + "_";

  fileName += "tape-" + std::to_string(tapeId) + ".tap";

  // don't forget space for null termination
  char *ret_char = new char[fileName.size() + 1];
  std::strcpy(ret_char, fileName.c_str()); // ensures null terminatoin
  return ret_char;
}
