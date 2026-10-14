#ifndef ADOLC_TAPEINFOS_H
#define ADOLC_TAPEINFOS_H

#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/oplate.h>
#include <adolc/valuetape/bufferstate.h>
#include <array>
#include <atomic> /* handling file names over different threads*/
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

using ADOLCError::ErrorType;
/**
 * @brief Identity, statistics, file names, and recording metadata of one tape.
 *
 * This movable metadata object records buffer sizes, independent/dependent
 * counts, file-backed tape names, trace flags, and related configuration.
 * Buffer ownership and evaluation synchronization live in `ValueTape` and its
 * recording/evaluation contexts instead.
 */
struct TapeInfos {

  // named indices of to print out value tape stats
  enum StatEntries {
    NUM_INDEPENDENTS, /* # of independent variables */
    NUM_DEPENDENTS,   /* # of dependent variables */
    NUM_MAX_LIVES,    /* max # of live variables */
    NUM_TAYS,         /* # of values in the taylor (value) stack */
    OP_BUFFER_SIZE,  /* # of operations per buffer == OBUFSIZE   (usrparms.h) */
    NUM_OPERATIONS,  /* overall # of operations */
    OP_FILE_ACCESS,  /* operations file written or not */
    NUM_LOCATIONS,   /* overall # of locations */
    LOC_FILE_ACCESS, /* locations file written or not */
    NUM_VALUES,      /* overall # of values */
    VAL_FILE_ACCESS, /* values file written or not */
    LOC_BUFFER_SIZE, /* # of locations per buffer == LBUFSIZE (usrparms.h) */
    VAL_BUFFER_SIZE, /* # of values per buffer == CBUFSIZE(usrparms.h) */
    TAY_BUFFER_SIZE, /* # of taylors per buffer <= TBUFSIZE (usrparms.h) */
    NUM_EQ_PROD,     /* # of eq_*_prod for sparsity pattern */
    NO_MIN_MAX,   /* no use of min_op, deferred to abs_op for piecewise stuff */
    NUM_SWITCHES, /* # of abs calls that can switch branch */
    NUM_PARAM, /* no of parameters (doubles) interchangeable without retaping */
    STAT_SIZE  /* represents the size of the stats vector */
  };
  // storage order used by the tape I/O layer
  enum FILES { OPERATIONS_FILE, LOCATIONS_FILE, VALUES_FILE, TAYLORS_FILE };

  // tape types => used for file name generation
  enum TAPENAMES { LOCATIONS_TAPE, VALUES_TAPE, OPERATIONS_TAPE, TAYLORS_TAPE };

  ~TapeInfos() {
    for (auto &fileName : fileNames) {
      if (fileName) {
        if (keepTape == 0 || skipFileCleanup == 0)
          remove(fileName);
        delete[] fileName;
        fileName = nullptr;
      }
    }
  };
  TapeInfos() = default;
  explicit TapeInfos(short tapeId) : tapeId_(tapeId) {
    fileNames[OPERATIONS_FILE] = createFileName(tapeId, OPERATIONS_TAPE);
    fileNames[LOCATIONS_FILE] = createFileName(tapeId, LOCATIONS_TAPE);
    fileNames[VALUES_FILE] = createFileName(tapeId, VALUES_TAPE);
    fileNames[TAYLORS_FILE] = createFileName(tapeId, TAYLORS_TAPE);
  };
  TapeInfos(short tapeId, std::array<std::string, 4> &&tapeBaseNames)
      : tapeBaseNames_(std::move(tapeBaseNames)), tapeId_(tapeId) {
    fileNames[OPERATIONS_FILE] = createFileName(tapeId, OPERATIONS_TAPE);
    fileNames[LOCATIONS_FILE] = createFileName(tapeId, LOCATIONS_TAPE);
    fileNames[VALUES_FILE] = createFileName(tapeId, VALUES_TAPE);
    fileNames[TAYLORS_FILE] = createFileName(tapeId, TAYLORS_TAPE);
  };
  TapeInfos(const TapeInfos &) = delete;
  TapeInfos &operator=(const TapeInfos &) = delete;
  TapeInfos(TapeInfos &&other) noexcept
      : stats(other.stats), fileNames(other.fileNames),
        tapeBaseNames_(std::move(other.tapeBaseNames_)), tapeId_(other.tapeId_),
        keepTape(other.keepTape), skipFileCleanup(other.skipFileCleanup) {
    other.fileNames.fill(nullptr);
  }
  TapeInfos &operator=(TapeInfos &&other) noexcept {
    if (this != &other) {
      for (auto &fileName : fileNames) {
        delete[] fileName;
      }
      stats = std::move(other.stats);
      tapeBaseNames_ = std::move(other.tapeBaseNames_);
      fileNames = std::move(other.fileNames);
      tapeId_ = other.tapeId_;
      keepTape = other.keepTape;
      skipFileCleanup = other.skipFileCleanup;
      other.fileNames.fill(nullptr);
    }
    return *this;
  }

  using StatArray = std::array<size_t, STAT_SIZE>;
  StatArray stats{};
  std::array<char *, 4> fileNames{};
  // the base names of every tape type
  std::array<std::string, 4> tapeBaseNames_;
  short tapeId_{-1};

  constexpr static size_t maxLocsPerOp{10}; // used in tape_loc_...

  //  - remember if tapes shall be written out to disk
  // - this information can only be given at taping time and must survive all
  // other actions on the tape
  int keepTape{0};

  // defaults to 0, if 1 skips file removal (when file operations are costly)
  int skipFileCleanup{0};

  /****************************************************************************/
  /* Tries to read a local config file containing, e.g., buffer sizes */
  /****************************************************************************/
  static char *duplicatestr(const char *instr) {
    size_t len = std::strlen(instr);
    char *outstr = new char[len + 1];
    std::strncpy(outstr, instr, len);
    return outstr;
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
  char *createFileName(short tapeId, int tapeType) {
    std::string fileName(tapeBaseNames_[tapeType]);

    int threadId = getThreadIndex();
    fileName += "thread-" + std::to_string(threadId) + "_";

    fileName += "tape-" + std::to_string(tapeId) + ".tap";

    // don't forget space for null termination
    char *ret_char = new char[fileName.size() + 1];
    std::strcpy(ret_char, fileName.c_str()); // ensures null terminatoin
    return ret_char;
  }
};

#endif // ADOLC_TAPEINFOS_H
