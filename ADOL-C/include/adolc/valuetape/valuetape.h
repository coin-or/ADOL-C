
#ifndef ADOLC_VALUETAPE_H
#define ADOLC_VALUETAPE_H

#include <adolc/adolcerror.h>
#include <adolc/adolcexport.h>
#include <adolc/buffer_temp.h>
#include <adolc/checkpointing.h>
#include <adolc/dvlparms.h>
#include <adolc/externfcts.h>
#include <adolc/externfcts2.h>
#include <adolc/storemanager.h>
#include <adolc/valuetape/globaltapevarscl.h>
#include <adolc/valuetape/infotype.h>
#include <adolc/valuetape/persistanttapeinfos.h>
#include <adolc/valuetape/tapeevaluationcontext.h>
#include <adolc/valuetape/tapeinfos.h>
#include <adolc/valuetape/taperecordingcontext.h>
#include <adolc/valuetape/taperegistry.h>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stack>
#include <stdexcept>
#include <type_traits>

// just ignore the missing DLL interface of the class members....
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

#ifdef ADOLC_SPARSE
#include <adolc/valuetape/sparseinfos.h>
#endif

struct ext_diff_fct;
struct ext_diff_fct_v2;

using ADOLC::detail::InfoType;
using ADOLC::detail::LocInfo;
using ADOLC::detail::OpInfo;
using ADOLC::detail::TapeEvaluationContext;
using ADOLC::detail::TapeRecordingContext;
using ADOLC::detail::ValInfo;
using ADOLCError::ErrorType;
using EvalOpInfoT = OpInfo<TapeEvaluationContext, ErrorType>;
using EvalLocInfoT = LocInfo<TapeEvaluationContext, ErrorType>;
using EvalValInfoT = ValInfo<TapeEvaluationContext, ErrorType>;

/**
 * @brief Owns one recorded function and the state required to evaluate it.
 *
 * A `ValueTape` is the central object behind a tape ID. During
 * recording, active `adouble` operations register opcodes, locations,
 * values, and optional Taylor values to the `TapeRecordingContext` of the tape.
 * During a sweep, `init_sweep()` creates a `TapeEvaluationContext` whose
 * cursors read the recorded data in forward or reverse order.
 *
 * Each tape owns value stores, location allocators, statistics, file names,
 * driver scratch data, external-function descriptors, checkpointing state,
 * and sparse recovery data.
 *
 * Use `createNewTape()` to allocate a tape, `findTape()` to look it up, and
 * `currentTape()` to access the thread's selected tape.
 *
 * Exclusive mode is the default and reuses the tape's owned buffers for one
 * evaluation at a time. `setSharedMode()` permits concurrent no-keep sweeps;
 * each sweep initially views the immutable recorded buffers and allocates
 * private storage lazily if it must load or modify a block.
 */
class ADOLC_API ValueTape {
  TapeRecordingContext recordCtx_;
  GlobalTapeVarsCL globalTapeVars_;
  TapeInfos tapeInfos_;
  PersistantTapeInfos perTapeInfos_;
  size_t ext_diff_fct_index_{0};
  std::shared_mutex mutex_;
  std::atomic<bool> statsNeedReload_{false};
  std::mutex statsLoadMutex_;
  std::optional<std::unique_lock<std::shared_mutex>> writeLock_;
  enum class DataAccess { Exclusive, Shared };

  /**
   * @brief Synchronizes changes between exclusive and shared evaluation mode.
   *
   * The mutex prevents the mode from changing while recording or evaluation
   * code holds a corresponding access lock.
   */
  struct DataAccessModeGuard {
    std::shared_mutex mutex_;
    DataAccess mode_{DataAccess::Exclusive};

    explicit DataAccessModeGuard(DataAccess mode) : mode_(mode) {}
  };

  DataAccessModeGuard accessMode_{DataAccess::Exclusive};
  bool containsExtDiff_{false};

#define EDFCTS_BLOCK_SIZE 10
  Buffer<ext_diff_fct, EDFCTS_BLOCK_SIZE> ext_buffer_;
  Buffer<ext_diff_fct_v2, EDFCTS_BLOCK_SIZE> ext2_buffer_;
  Buffer<ADOLC::CP::detail::Infos, EDFCTS_BLOCK_SIZE> cp_buffer_;

#ifdef ADOLC_SPARSE
  ADOLC::Sparse::SparseJacInfos sJInfos_;
  ADOLC::Sparse::SparseHessInfos sHInfos_;

  void initSparse() {
    sJInfos_.~SparseJacInfos();
    new (&sJInfos_) ADOLC::Sparse::SparseJacInfos();

    sHInfos_.~SparseHessInfos();
    new (&sHInfos_) ADOLC::Sparse::SparseHessInfos();
  }
#endif

public:
  ~ValueTape();

  // a tape always need a tapeId,
  ValueTape() = delete;
  explicit ValueTape(short tapeId) : tapeInfos_(tapeId, readConfigFile()) {}

  // copying ValueTape is not allowed!
  ValueTape(const ValueTape &other) = delete;
  ValueTape &operator=(const ValueTape &other) = delete;

  ValueTape(ValueTape &&other) noexcept
      : recordCtx_(std::move(other.recordCtx_)),
        globalTapeVars_(std::move(other.globalTapeVars_)),
        tapeInfos_(std::move(other.tapeInfos_)),
        perTapeInfos_(std::move(other.perTapeInfos_)),
        ext_diff_fct_index_(other.ext_diff_fct_index_),
        statsNeedReload_(
            other.statsNeedReload_.exchange(false, std::memory_order_relaxed)),
        accessMode_(other.accessMode_.mode_),
        containsExtDiff_(other.containsExtDiff_),
        ext_buffer_(std::move(other.ext_buffer_)),
        ext2_buffer_(std::move(other.ext2_buffer_)),
        cp_buffer_(std::move(other.cp_buffer_))
#ifdef ADOLC_SPARSE
        ,
        sJInfos_(std::move(other.sJInfos_)), sHInfos_(std::move(other.sHInfos_))
#endif
  {
  }

  ValueTape &operator=(ValueTape &&other) noexcept {
    if (this != &other) {
      recordCtx_ = std::move(other.recordCtx_);
      tapeInfos_ = std::move(other.tapeInfos_);
      globalTapeVars_ = std::move(other.globalTapeVars_);
      perTapeInfos_ = std::move(other.perTapeInfos_);
      ext_diff_fct_index_ = other.ext_diff_fct_index_;
      statsNeedReload_.store(
          other.statsNeedReload_.exchange(false, std::memory_order_relaxed),
          std::memory_order_release);
      accessMode_.mode_ = other.accessMode_.mode_;
      containsExtDiff_ = other.containsExtDiff_;
      ext_buffer_ = std::move(other.ext_buffer_);
      ext2_buffer_ = std::move(other.ext2_buffer_);
      cp_buffer_ = std::move(other.cp_buffer_);
#ifdef ADOLC_SPARSE
      sJInfos_ = std::move(other.sJInfos_);
      sHInfos_ = std::move(other.sHInfos_);
#endif
    }
    return *this;
  }

  bool isExclusiveNonLocking() {
    return accessMode_.mode_ == DataAccess::Exclusive;
  }
  bool isExclusiveLocking() {
    std::shared_lock<std::shared_mutex> lock(accessMode_.mutex_);
    return accessMode_.mode_ == DataAccess::Exclusive;
  }
  void registerExtDiff() { containsExtDiff_ = true; }
  bool containsExtDiff() const { return containsExtDiff_; }

private:
  void setMode_(DataAccess mode) {
    using ADOLCError::fail;
    using ADOLCError::FailInfo;
    using ADOLCError::ErrorType::TAPING_TAPE_STILL_IN_USE;
    std::unique_lock<std::shared_mutex> dataAccessLock(accessMode_.mutex_);
    if (accessMode_.mode_ == mode) {
      return;
    }
    if (isRecording()) {
      fail(TAPING_TAPE_STILL_IN_USE, CURRENT_LOCATION,
           FailInfo{.info1 = tapeId()});
    }
    accessMode_.mode_ = mode;
  }

public:
  /**
   * @brief Enables concurrent no-keep evaluations of this tape.
   *
   * Existing evaluations and recordings must finish before the mode changes.
   * External differentiated functions are currently unsupported in this mode.
   */
  void setSharedMode() { setMode_(DataAccess::Shared); }

  /// Restores the default single-evaluation mode with reusable owned buffers.
  void setExclusiveMode() { setMode_(DataAccess::Exclusive); }

  void statsNeedReload() {
    statsNeedReload_.store(true, std::memory_order_release);
  }

  std::shared_mutex &accessMutex() noexcept { return mutex_; }

  bool isRecording() const noexcept {
    return currentTapeStack().back().current == this;
  }
  void beginRecording();
  void endRecording();

#ifdef ADOLC_SPARSE
  // updates the tape infos on sparse Jac or Hess for the given ID
  void setTapeInfoJacSparse(ADOLC::Sparse::SparseJacInfos &&sJInfos) {
    sJInfos_ = std::move(sJInfos);
  }
  void setTapeInfoHessSparse(ADOLC::Sparse::SparseHessInfos &&sHInfos) {
    sHInfos_ = std::move(sHInfos);
  }
  ADOLC::Sparse::SparseJacInfos &sJInfos() { return sJInfos_; }
  ADOLC::Sparse::SparseHessInfos &sHInfos() { return sHInfos_; }
  const ADOLC::Sparse::SparseJacInfos &sJInfos() const { return sJInfos_; }
  const ADOLC::Sparse::SparseHessInfos &sHInfos() const { return sHInfos_; }
#endif

  // Interface to PersistentTapeInfos
  void tapeBaseNames(size_t loc, const std::string &baseName) {
    tapeInfos_.tapeBaseNames_[loc] = baseName;
  }
  void skipFileCleanup(int skipFileCleanup) {
    tapeInfos_.skipFileCleanup = skipFileCleanup;
  }
  int skipFileCleanup() const { return tapeInfos_.skipFileCleanup; }

  char *tay_fileName() const {
    return tapeInfos_.fileNames[TapeInfos::TAYLORS_FILE];
  }
  char *op_fileName() const {
    return tapeInfos_.fileNames[TapeInfos::OPERATIONS_FILE];
  }
  char *loc_fileName() const {
    return tapeInfos_.fileNames[TapeInfos::LOCATIONS_FILE];
  }
  char *val_fileName() const {
    return tapeInfos_.fileNames[TapeInfos::VALUES_FILE];
  }
  void tay_fileName(char *name) {
    tapeInfos_.fileNames[TapeInfos::TAYLORS_FILE] = name;
  }
  void op_fileName(char *name) {
    tapeInfos_.fileNames[TapeInfos::OPERATIONS_FILE] = name;
  }
  void loc_fileName(char *name) {
    tapeInfos_.fileNames[TapeInfos::LOCATIONS_FILE] = name;
  }
  void val_fileName(char *name) {
    tapeInfos_.fileNames[TapeInfos::VALUES_FILE] = name;
  }
  int keepTape() const { return tapeInfos_.keepTape; }
  void keepTape(int flag) { tapeInfos_.keepTape = flag; }
  int jacSolv_nax() const { return perTapeInfos_.jacSolv_nax; }
  int *jacSolv_ci() const { return perTapeInfos_.jacSolv_ci; }
  int *jacSolv_ri() const { return perTapeInfos_.jacSolv_ri; }
  double *jacSolv_xold() const { return perTapeInfos_.jacSolv_xold; }
  double **jacSolv_I() const { return perTapeInfos_.jacSolv_I; }
  double **jacSolv_J() const { return perTapeInfos_.jacSolv_J; }
  int jacSolv_modeold() const { return perTapeInfos_.jacSolv_modeold; }
  void jacSolv_nax(int nax) { perTapeInfos_.jacSolv_nax = nax; }
  void jacSolv_I(double **I) { perTapeInfos_.jacSolv_I = I; }
  void jacSolv_J(double **J) { perTapeInfos_.jacSolv_J = J; }
  void jacSolv_xold(double *xold) { perTapeInfos_.jacSolv_xold = xold; }
  void jacSolv_modeold(int mode) { perTapeInfos_.jacSolv_modeold = mode; }
  void jacSolv_ci(int *ci) { perTapeInfos_.jacSolv_ci = ci; }
  void jacSolv_ri(int *ri) { perTapeInfos_.jacSolv_ri = ri; }
  int forodec_nax() const { return perTapeInfos_.forodec_nax; }
  int forodec_dax() const { return perTapeInfos_.forodec_dax; }
  double *forodec_y() const { return perTapeInfos_.forodec_y; }
  double *forodec_z() const { return perTapeInfos_.forodec_z; }
  double **forodec_Z() const { return perTapeInfos_.forodec_Z; }
  void forodec_nax(int nax) { perTapeInfos_.forodec_nax = nax; }
  void forodec_dax(int dax) { perTapeInfos_.forodec_dax = dax; }
  void forodec_y(double *y) { perTapeInfos_.forodec_y = y; }
  void forodec_z(double *z) { perTapeInfos_.forodec_z = z; }
  void forodec_Z(double **Z) { perTapeInfos_.forodec_Z = Z; }

  // Interface to TapeInfos
  void readParams(double *&targetStore);
  void compare_adolc_ids(const ADOLC_ID &id1, const ADOLC_ID &id2);
  void read_tape_stats();
  /****************************************************************************/
  /* Tapestats: */
  /* Returns statistics on the tape tag with following meaning: */
  /* tape_stat[0] = # of independent variables. */
  /* tape_stat[1] = # of dependent variables. */
  /* tape_stat[2] = max # of live variables. */
  /* tape_stat[3] = value stack size. */
  /* tape_stat[4] = buffer size (# of chars, # of doubles, # of size_ts) */
  /* tape_stat[5] = # of operations. */
  /* tape_stat[6] = operation file access flag (1 = file in use, 0
   * otherwise)
   */
  /* tape_stat[7] = # of saved locations. */
  /* tape_stat[8] = location file access flag (1 = file in use, 0 otherwise)
   */
  /* tape_stat[9] = # of saved constant values. */
  /* tape_stat[10]= value file access flag (1 = file in use, 0 otherwise) */
  /****************************************************************************/
  void tapestats(std::array<size_t, TapeInfos::STAT_SIZE> stats) {
    std::copy(tapeInfos_.stats.cbegin(), tapeInfos_.stats.cend(),
              stats.begin());
  }
  void tapestats(size_t *stats) {
    std::copy(tapeInfos_.stats.cbegin(), tapeInfos_.stats.cend(), stats);
  }
  size_t tapestats(size_t stat) const { return tapeInfos_.stats[stat]; };
  void tapestats(size_t stat, size_t val) { tapeInfos_.stats[stat] = val; };
  std::array<size_t, TapeInfos::STAT_SIZE> tapestats() const {
    return tapeInfos_.stats;
  }

  void deg_save(int val) { recordCtx_.deg_save = val; }
  int deg_save() const { return recordCtx_.deg_save; }

  int keepTaylors() const { return recordCtx_.keepTaylors; }
  void keepTaylors(int val) { recordCtx_.keepTaylors = val; }

  size_t numparam() const { return globalTapeVars_.numparam; }

  void increment_numTays_Tape() {
    recordCtx_.tayBuffer_.numOnTape(recordCtx_.tayBuffer_.numOnTape() + 1);
  }
  void add_numTays_Tape(size_t val) {
    recordCtx_.tayBuffer_.numOnTape(recordCtx_.tayBuffer_.numOnTape() + val);
  }

  void lastTayBlockInCore(char val) { recordCtx_.lastTayBlockInCore = val; }
  char lastTayBlockInCore() const { return recordCtx_.lastTayBlockInCore; }

  void decrement_numTays_Tape() {
    recordCtx_.tayBuffer_.numOnTape(recordCtx_.tayBuffer_.numOnTape() - 1);
  }

  size_t num_eq_prod() const { return recordCtx_.num_eq_prod; }
  void num_eq_prod(size_t num) { recordCtx_.num_eq_prod = num; }
  void increment_num_eq_prod() { ++(recordCtx_.num_eq_prod); }
  void add_num_eq_prod(size_t val) { recordCtx_.numDeps += val; }

  void increment_numInds() { ++recordCtx_.numInds; }
  size_t numInds() const { return recordCtx_.numInds; }

  void increment_numDeps() { ++recordCtx_.numDeps; }
  size_t numDeps() const { return recordCtx_.numDeps; }

  void tay_numInds(size_t val) { recordCtx_.tay_numInds = val; }
  size_t tay_numInds() const { return recordCtx_.tay_numInds; }

  void tay_numDeps(size_t val) { recordCtx_.tay_numDeps = val; }
  size_t tay_numDeps() const { return recordCtx_.tay_numDeps; }

  void numSwitches(size_t num) { recordCtx_.numSwitches = num; }
  size_t numSwitches() const { return recordCtx_.numSwitches; }
  void increment_numSwitches() { ++recordCtx_.numSwitches; }

  short tapeId() const { return tapeInfos_.tapeId_; }
  size_t no_min_max() { return tapeInfos_.stats[TapeInfos::NO_MIN_MAX]; }
  size_t ext_diff_fct_index() const { return ext_diff_fct_index_; }
  void ext_diff_fct_index(size_t index) { ext_diff_fct_index_ = index; }

  void nextBufferNumber(size_t num) { recordCtx_.nextBufferNumber = num; }
  size_t nextBufferNumber() const { return recordCtx_.nextBufferNumber; }
  void decrement_nextBufferNumber() { --recordCtx_.nextBufferNumber; }

  constexpr static size_t maxLocsPerOp() { return TapeInfos::maxLocsPerOp; }

  void put_op(OPCODES op, size_t reserveExtraLocations = 0) {
    return recordCtx_.put_op(op, loc_fileName(), op_fileName(), val_fileName(),
                             reserveExtraLocations);
  }

  void put_loc(size_t loc) { return recordCtx_.put_loc(loc); };

  void put_val(const double val) { recordCtx_.valBuffer_.writeAndAdvance(val); }
  /* puts a single constant into the location buffer, no disk access */
  void put_vals_writeBlock(double *reals, size_t numReals) {
    return recordCtx_.put_vals_writeBlock(reals, numReals, op_fileName(),
                                          val_fileName());
  };
  /* fill the constants buffer and write it to disk */
  void put_vals_notWriteBlock(double *reals, size_t numReals) {
    return recordCtx_.put_vals_notWriteBlock(reals, numReals);
  }

  /* reads the previous block of constants into the internal buffer */
  size_t get_val_space() {
    return recordCtx_.get_val_space(op_fileName(), val_fileName());
  };
  /* updates */
  int upd_resloc(size_t temp, size_t lhs) {
    return recordCtx_.upd_resloc(temp, lhs);
  }
  int upd_resloc_check(size_t temp) {
    return recordCtx_.upd_resloc_check(temp);
  }
  int upd_resloc_inc_prod(size_t temp, size_t newlhs, unsigned char newop) {
    return recordCtx_.upd_resloc_inc_prod(temp, newlhs, newop);
  }
  size_t get_num_param() { return tapeInfos_.stats[TapeInfos::NUM_PARAM]; }
  // Marks reverse evaluation as nested so reverse outputs accumulate on the
  // surrounding tape instead of overwriting its adjoints.
  void nestedReverseEval(bool flag) { recordCtx_.nestedReverseEval = flag; }
  // Returns whether reverse evaluation should accumulate into an outer tape.
  bool nestedReverseEval() const { return recordCtx_.nestedReverseEval; }
  double *signature() const { return recordCtx_.signature; }
  void signature(double *buffer) { recordCtx_.signature = buffer; }

  void initTapeInfos_keep();
  // free/allocate memory for buffers, initialize pointers
  template <ADOLC::detail::EvalOrRecordContextType Context>
  void initTapeBuffers(Context &ctx) {
    ctx.opBuffer_.allocIfNull(tapestats(TapeInfos::OP_BUFFER_SIZE));
    ctx.valBuffer_.allocIfNull(tapestats(TapeInfos::VAL_BUFFER_SIZE));
    ctx.locBuffer_.allocIfNull(tapestats(TapeInfos::LOC_BUFFER_SIZE));
  }
  void initTapeBuffers() { initTapeBuffers(recordCtx_); }

  //--------------------------------------------------------------

  // Inteface global tape vars
  size_t operationBufferSize() const {
    return globalTapeVars_.operationBufferSize;
  }
  void operationBufferSize(size_t size) {
    globalTapeVars_.operationBufferSize = size;
  }

  size_t locationBufferSize() const {
    return globalTapeVars_.locationBufferSize;
  }
  void locationBufferSize(size_t size) {
    globalTapeVars_.locationBufferSize = size;
  }

  size_t valueBufferSize() const { return globalTapeVars_.valueBufferSize; }
  void valueBufferSize(size_t size) { globalTapeVars_.valueBufferSize = size; }

  size_t taylorBufferSize() const { return globalTapeVars_.taylorBufferSize; }
  void taylorBufferSize(size_t size) {
    globalTapeVars_.taylorBufferSize = size;
  }

  size_t initalStoreSize() const { return globalTapeVars_.initialStoreSize; }
  void initialStoreSize(size_t size) {
    globalTapeVars_.initialStoreSize = size;
  }
  void checkInitialStoreSize() { globalTapeVars_.checkInitialStoreSize(); }

  unsigned int nominmaxFlag() const { return globalTapeVars_.nominmaxFlag; }
  void enableBranchSwitchWarnings() { globalTapeVars_.branchSwitchWarning = 1; }
  void disableBranchSwitchWarnings() {
    globalTapeVars_.branchSwitchWarning = 0;
  }
  void enableMinMaxUsingAbs() {
    if (isRecording())
      ADOLCError::fail(ADOLCError::ErrorType::ENABLE_MINMAX_USING_ABS,
                       CURRENT_LOCATION);
    std::unique_lock lock(mutex_);
    globalTapeVars_.nominmaxFlag = 1;
  }

  void disableMinMaxUsingAbs() {
    if (isRecording())
      ADOLCError::fail(ADOLCError::ErrorType::DISABLE_MINMAX_USING_ABS,
                       CURRENT_LOCATION);
    std::unique_lock lock(mutex_);
    globalTapeVars_.nominmaxFlag = 0;
  }

  // helper for creating contiguous adouble locations
  void ensureContiguousLocations(size_t n) {
    globalTapeVars_.storeManagerPtr->ensure_block(n);
  }
  size_t ensureContiguousLocations_(size_t n) {
    globalTapeVars_.storeManagerPtr->ensure_block(n);
    return n;
  };
  void setStoreManagerControl(double gcTriggerRatio, size_t gcTriggerMaxSize) {
    globalTapeVars_.storeManagerPtr->setStoreManagerControl(gcTriggerRatio,
                                                            gcTriggerMaxSize);
  }

  void setStoreManagerType(unsigned char type) {
    if (globalTapeVars_.storeManagerPtr->storeType() != type) {
      if (!globalTapeVars_.numLives)
        globalTapeVars_.reallocStore(type);
      else
        ADOLCError::fail(
            ADOLCError::ErrorType::SM_ACTIVE_VARS, CURRENT_LOCATION,
            ADOLCError::FailInfo{.info5 = globalTapeVars_.numLives});
    } else
      ADOLCError::fail(ADOLCError::ErrorType::SM_SAME_TYPE, CURRENT_LOCATION);
  }
  // returns the next free location in "adouble" memory
  size_t next_loc() { return globalTapeVars_.storeManagerPtr->next_loc(); }
  // returns the next free location in "pdouble" memory
  size_t p_next_loc() { return globalTapeVars_.paramStoreMgrPtr->next_loc(); }

  // frees the specified location in "adouble" memory
  void free_loc(size_t loc) const {
    globalTapeVars_.storeManagerPtr->free_loc(loc);
  }

  // frees the specified location in "pdouble" memory
  void p_free_loc(size_t loc) const {
    globalTapeVars_.paramStoreMgrPtr->free_loc(loc);
  }

  double get_ad_value(size_t loc) const { return globalTapeVars_.store[loc]; }
  void set_ad_value(size_t loc, double coval) {
    if (globalTapeVars_.store)
      globalTapeVars_.store[loc] = coval;
  }
  double get_pd_value(size_t loc) const { return globalTapeVars_.pStore[loc]; }
  void set_pd_value(size_t loc, double coval) {
    globalTapeVars_.pStore[loc] = coval;
  }
  size_t numLives() const { return globalTapeVars_.numLives; }
#if defined(ADOLC_TRACK_ACTIVITY)
  char get_active_value(size_t loc) const {
    return globalTapeVars_.actStore[loc];
  }
  void set_active_value(size_t loc, char coval) {
    globalTapeVars_.actStore[loc] = coval;
  }
#endif

  size_t storeSize() const { return globalTapeVars_.storeSize; }
  double *store() const { return globalTapeVars_.store; }
  void store(double *buffer) { globalTapeVars_.store = buffer; }
  double *pStore() const { return globalTapeVars_.pStore; }
  void pStore(double *buffer) { globalTapeVars_.pStore = buffer; }
  double store(size_t idx) { return globalTapeVars_.store[idx]; }
  char branchSwitchWarning() const {
    return globalTapeVars_.branchSwitchWarning;
  }
  void branchSwitchWarning(char val) {
    globalTapeVars_.branchSwitchWarning = val;
  }
  char inParallelRegion() const { return globalTapeVars_.inParallelRegion; }

  // ------------------------------- Buffer utils ---------------------------
  ADOLC::CP::detail::Infos *cp_append() { return cp_buffer_.append(); }
  ADOLC::CP::detail::Infos *cp_getElement(size_t index) {
    return cp_buffer_.getElement(index);
  }

  ext_diff_fct *ext_diff_append() { return ext_buffer_.append(); }
  ext_diff_fct *ext_diff_getElement(size_t index) {
    return ext_buffer_.getElement(index);
  }
  ext_diff_fct_v2 *ext_diff_v2_append() { return ext2_buffer_.append(); }
  ext_diff_fct_v2 *ext_diff_v2_getElement(size_t index) {
    return ext2_buffer_.getElement(index);
  }

  // ------------------------------------------- Combined
  /* tries to read a local config file containing, e.g., buffer sizes */
  std::array<std::string, 4> readConfigFile();

  /// @brief Initialize Taylor-stack recording for the current tape.
  void taylor_begin(int degreeSave);

  // close taylor file if necessary and refill buffer if possible
  void finish_tay_file();
  void taylor_close();

  void write_scaylor(double val) {
    recordCtx_.write_scaylor(val, tay_fileName());
  }

  // deletes the last (single) element (x) of the taylor buffer
  void delete_scaylor(size_t loc) {
    globalTapeVars_.store[loc] = recordCtx_.tayBuffer_.retreatAndRead();
  }

  /**
   * @brief Return the tape file name associated with the given Info adapter.
   *
   * The index is supplied by the Info adapter so the caller does not branch on
   * concrete tape kinds.
   */
  template <typename Info> const char *fileName() const {
    return tapeInfos_.fileNames[Info::fileIndex];
  }

  /** @brief Compile-time list used to prepare tape buffers for forward and
   * reverse sweeps. */
  template <typename... Ts> struct AllTypes {};
  using AllInfoTypes = AllTypes<EvalOpInfoT, EvalLocInfoT, EvalValInfoT>;

  /**
   * @brief Prepare for a forward sweep.
   *
   * If something is stored on disk (tapestats(Info::fileAccess)!=0), this
   * function opens the tape file, preloads up to one block of data into the
   * in-memory buffer, and sets the internal counters/pointers so subsequent
   * reads can continue from the correct position.
   *
   * The logic is:
   *  - optionally read a block into buffer (up to bufferSize)
   *  - set the remaining element count on the selected buffer to what is
   *    still left on disk after the preload
   *  - initialize the current buffer position
   *
   * If nothing was written to disk, we assume all data is already in memory
   * and only initialize the counters/pointers accordingly.
   *
   * Special case:
   *  - Loc tape adjusts the current pointer based on statSpace and may trigger
   *    get_loc_block_f() to align buffer state with statistics bookkeeping.
   */
  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void prepare_for(TapeEvaluationContext &evalCtx) {
    size_t blockSize = 0;
    auto &buffer = Info::getBuffer(evalCtx);
    if (tapestats(Info::fileAccess) == 1) {
      buffer.openFile(fileName<Info>(), "rb");

      // preload at most one block, but never more than total elements on tape
      blockSize = std::min(tapestats(Info::bufferSize), tapestats(Info::num));
      evalCtx.loadBlockIntoBuffer<Info>(blockSize);
      // remaining elements still residing on disk (not yet in buffer)
      blockSize = tapestats(Info::num) - blockSize;
    }
    buffer.numOnTape(blockSize);
    Info::prepareForwardPosition(evalCtx, tapeInfos_.stats[Info::bufferSize]);
  }

  /**
   * @brief Position the tape file at the beginning of the last on-disk block.
   *
   * Computes the offset of the last full block boundary:
   *   floor(num / bufferSize) * bufferSize
   * and seeks the file to that position (in bytes).
   *
   * Used by reverse sweeps to read the final block of the tape first.
   *
   * Preconditions:
   *  - tapestats(Info::num) and tapestats(Info::bufferSize) are initialized.
   */
  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void setFilePosition(TapeEvaluationContext &evalCtx) {
    auto number = (tapestats(Info::num) / tapestats(Info::bufferSize)) *
                  tapestats(Info::bufferSize);
    auto offset = static_cast<long>(number * sizeof(typename Info::value_type));
    auto &buffer = Info::getBuffer(evalCtx);
    buffer.openFile(fileName<Info>(), "rb");
    fseek(buffer.file(), offset, SEEK_SET);
  }

  /**
   * @brief Prepare for reverse sweep.
   *
   * Reverse sweeps start at the end of the tape. If data was written to disk,
   * this function seeks to the last on-disk block and preloads the (possibly
   * partial) final block into the in-memory buffer.
   *
   * After the preload:
   *  - the selected buffer stores the number of elements still remaining on
   *    disk before the loaded block
   *  - the selected buffer position is moved to the logical end of the loaded
   *    region so reverse logic can walk backwards
   *
   * If nothing was written to disk, we assume all data is already in memory
   * and only initialize the counters/pointers accordingly.
   */
  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void prepare_rev(TapeEvaluationContext &evalCtx) {
    size_t blockSize = tapestats(Info::num);
    auto &buffer = Info::getBuffer(evalCtx);
    if (tapestats(Info::fileAccess) == 1) {
      setFilePosition<Info>(evalCtx);

      // size of last (possibly partial) block
      blockSize = tapestats(Info::num) % tapestats(Info::bufferSize);
      evalCtx.loadBlockIntoBuffer<Info>(blockSize);
    }
    buffer.numOnTape(tapestats(Info::num) - blockSize);
    buffer.position(blockSize);
  }

  /**
   * @brief Run prepare_for() for all tape types listed in AllTypes.
   *
   * This is just a compile-time loop (fold expression) over the Info types.
   */
  template <InfoType<TapeEvaluationContext, ErrorType>... Infos>
  void prepare_for_all(TapeEvaluationContext &evalCtx,
                       AllTypes<Infos...> /*unused*/)
    requires(requires { Infos::fileAccess; } && ...)
  {
    (prepare_for<Infos>(evalCtx), ...);
  }

  /**
   * @brief Run prepare_for() for all tape types listed in AllTypes.
   *
   * This is just a compile-time loop (fold expression) over the Info types.
   */
  template <InfoType<TapeEvaluationContext, ErrorType>... Infos>
  void prepare_rev_all(TapeEvaluationContext &evalCtx,
                       AllTypes<Infos...> /*unused*/)
    requires(requires { Infos::fileAccess; } && ...)
  {
    (prepare_rev<Infos>(evalCtx), ...);
  }

  /** @brief Base tag for the direction selected by `init_sweep()`. */
  struct Mode {};
  /** @brief Selects forward-order tape traversal. */
  struct Forward : Mode {};
  /** @brief Selects reverse-order tape traversal. */
  struct Reverse : Mode {};

private:
  template <typename Mode> void prepareSweep(TapeEvaluationContext &evalCtx) {
    using namespace ADOLC::detail;

    initTapeBuffers(evalCtx);
    if (tapestats(TapeInfos::NUM_PARAM) > 0 && evalCtx.paramstore == nullptr)
      readParams(evalCtx.paramstore);
    if constexpr (std::is_same_v<Mode, Forward>) {
      prepare_for_all(evalCtx, AllInfoTypes{});
#ifdef ADOLC_AMPI_SUPPORT
      TAPE_AMPI_resetBottom();
#endif
    } else if constexpr (std::is_same_v<Mode, Reverse>) {
      prepare_rev_all(evalCtx, AllInfoTypes{});
#ifdef ADOLC_AMPI_SUPPORT
      TAPE_AMPI_resetTop();
#endif
    } else {
      static_assert(!std::is_same_v<Mode, Mode>, "Mode not implemented!");
    }
  }

  template <typename Mode>
  TapeEvaluationContext
  initSweepKeep(std::shared_lock<std::shared_mutex> &&dataAccessLock) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    openTape();
    TapeEvaluationContext evalCtx(std::move(recordCtx_), std::move(lock),
                                  std::move(dataAccessLock));
    prepareSweep<Mode>(evalCtx);
    return evalCtx;
  }

  template <typename Mode>
  TapeEvaluationContext
  initSweepNoKeep(std::shared_lock<std::shared_mutex> &&dataAccessLock) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (containsExtDiff_) {
      ADOLCError::fail(ErrorType::EXT_DIFF_SHARED_MODE, CURRENT_LOCATION,
                       ADOLCError::FailInfo{.info1 = tapeId()});
    }
    openTape();
    TapeEvaluationContext evalCtx(recordCtx_, tapeInfos_.stats, std::move(lock),
                                  std::move(dataAccessLock));
    prepareSweep<Mode>(evalCtx);
    return evalCtx;
  }

public:
  /**
   * @brief Initialize a tape sweep.
   *
   * Performs the common setup required before starting either a forward or
   * reverse sweep.
   *
   * Actions performed:
   *  - Reads or refreshes tape statistics.
   *  - Allocates and initializes in-memory tape buffers.
   *  - Prepares all tape types depending on the sweep direction:
   *      * Forward:  open files and preload initial data blocks.
   *      * Reverse:  seek to last on-disk block and preload final data.
   *  - If ADOLC_AMPI_SUPPORT is enabled, resets the AMPI stack bounds
   *    (bottom for forward, top for reverse).
   *
   * @tparam Mode Sweep direction selector. Must be either Forward or Reverse.
   */
  template <typename Mode> TapeEvaluationContext init_sweep(int keep = 0) {
    std::shared_lock<std::shared_mutex> lock(accessMode_.mutex_);
    if (keep || accessMode_.mode_ == DataAccess::Exclusive) {
      return initSweepKeep<Mode>(std::move(lock));
    } else {
      return initSweepNoKeep<Mode>(std::move(lock));
    }
  }
  // finish a forward or reverse sweep
  void end_sweep(TapeEvaluationContext &evalCtx);
  void end_sweep(TapeEvaluationContext &&evalCtx);
  // initialization for the taping process -> buffer allocation, sets files
  // names, and calls appropriate setup routines
  void start_trace();

  // record all existing adoubles on the tape - intended to be used in
  // start_trace only
  void take_stock();

  /* record all remaining live variables on the value stack tape
   * - turns off trace_flag
   * - intended to be used in stop_trace only */
  size_t keep_stock();

  // stop Tracing, clean up, and turn off trace_flag
  void stop_trace(int flag);

  /* initializes a new tape
   * - returns 0 on success
   * - returns 1 in case tapeId is already/still in use */
  int initNewTape();

  // ------------------- Combined methods ------------------------
  // opens an existing tape or creates a new one
  void openTape();

  // updates the tape infos for the given ID - a tapeInfos struct is created
  // and registered if non is found but its state will remain "not in use"
  std::shared_ptr<ValueTape> getTapeInfos(short tapeId);

  // close open tapes, update stats and clean up
  void close_tape(int flag);

  /**
   * @brief Update parameter values used by subsequent evaluations.
   *
   * Replaces the values stored for all pdouble parameters on this tape. The
   * number of supplied values must match the number of parameters recorded on
   * the tape.
   *
   * @param paramvec New parameter values in tape-location order.
   *
   * @note This invalidates saved Taylor coefficients. Any reverse sweep after
   * this call must be preceded by a forward sweep with `keep >= 1`, for example
   * `zos_forward(..., 1, ...)`.
   *
   * @throws ADOLCError::ADOLCError if the tape is currently being written or
   * if the number of supplied values does not match the tape.
   */
  void setParamVec(std::span<const double> paramvec);
  void save_params();
  /****************************************************************************/
  /* Frees parameter indices after taping is complete */
  /****************************************************************************/
  /* Only called during stop_trace() via save_params() */
  void free_all_taping_params() {
    size_t np = tapeInfos_.stats[TapeInfos::NUM_PARAM];
    while (np > 0)
      globalTapeVars_.paramStoreMgrPtr->free_loc(--np);
  }

  /* special IEEE values */
  static double make_nan() {
    return std::numeric_limits<double>::quiet_NaN();
    ;
  }

  static double make_inf() { return std::numeric_limits<double>::infinity(); }

  ADOLC::CP::detail::Infos *get_cp_fct(size_t index) const {
    return cp_buffer_.getElement(index);
  }
};

#endif // ADOLC_VALUETAPE_H
