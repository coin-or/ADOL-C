
#ifndef ADOLC_VALUETAPE_H
#define ADOLC_VALUETAPE_H

#include <adolc/adolcerror.h>
#include <adolc/adolcexport.h>
#include <adolc/buffer_temp.h>
#include <adolc/checkpointing_p.h>
#include <adolc/externfcts.h>
#include <adolc/externfcts2.h>
#include <adolc/storemanager.h>
#include <adolc/valuetape/globaltapevarscl.h>
#include <adolc/valuetape/persistanttapeinfos.h>
#include <adolc/valuetape/tapeinfos.h>
#include <memory>
#include <stack>

#ifdef ADOLC_SPARSE
#include <adolc/valuetape/sparseinfos.h>
#endif

struct ext_diff_fct;
struct ext_diff_fct_v2;

/**
 * class ValueTape
 *
 *
 * Composition of
 * GlobalTapeVars
 * TapeInfos
 * PersistantTapeInfos
 * Buffers for Checkpointing and External Differentiated Functions
 * SparseJacInfos
 * SparseHessInfos
 *
 * A lot of interface utilites that are used in various value tape handling
 * methods
 */

class ADOLC_API ValueTape {
  TapeInfos tapeInfos_;
  GlobalTapeVarsCL globalTapeVars_;
  PersistantTapeInfos perTapeInfos_;

#define EDFCTS_BLOCK_SIZE 10
  Buffer<ext_diff_fct, EDFCTS_BLOCK_SIZE> ext_buffer_;
  Buffer<ext_diff_fct_v2, EDFCTS_BLOCK_SIZE> ext2_buffer_;
  Buffer<CpInfos, EDFCTS_BLOCK_SIZE> cp_buffer_;

  using StackElement = double **;
  std::stack<StackElement> cp_stack_;

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

  // the compiler can not distinguish between the fct ptrs for ext_diff_fct
  // and ext_diff_fct_v2. The wrapper makes the types explicit in the
  // constructor of ValueTape
  template <typename T> static void edf_zero_wrapper(T *arg) { edf_zero(arg); }

public:
  ~ValueTape();

  // a tape always need a tapeId,
  ValueTape() = delete;
  ValueTape(short tapeId)
      : tapeInfos_(tapeId), globalTapeVars_(),
        perTapeInfos_(tapeId, readConfigFile()),
        ext_buffer_(edf_zero_wrapper<ext_diff_fct>),
        ext2_buffer_(edf_zero_wrapper<ext_diff_fct_v2>),
        cp_buffer_(init_CpInfos) {}

  // copying ValueTape is not allowed!
  ValueTape(const ValueTape &other) = delete;
  ValueTape &operator=(const ValueTape &other) = delete;

  ValueTape(ValueTape &&other) noexcept
      : tapeInfos_(std::move(other.tapeInfos_)),
        globalTapeVars_(std::move(other.globalTapeVars_)),
        perTapeInfos_(std::move(other.perTapeInfos_)),
        ext_buffer_(std::move(other.ext_buffer_)),
        ext2_buffer_(std::move(other.ext2_buffer_)),
        cp_buffer_(std::move(other.cp_buffer_)),
        cp_stack_(std::move(other.cp_stack_))
#ifdef ADOLC_SPARSE
        ,
        sJInfos_(std::move(other.sJInfos_)), sHInfos_(std::move(other.sHInfos_))
#endif
  {
  }

  ValueTape &operator=(ValueTape &&other) noexcept {
    if (this != &other) {
      tapeInfos_ = std::move(other.tapeInfos_);
      globalTapeVars_ = std::move(other.globalTapeVars_);
      perTapeInfos_ = std::move(other.perTapeInfos_);
      ext_buffer_ = std::move(other.ext_buffer_);
      ext2_buffer_ = std::move(other.ext2_buffer_);
      cp_buffer_ = std::move(other.cp_buffer_);
      cp_stack_ = std::move(other.cp_stack_);
#ifdef ADOLC_SPARSE
      sJInfos_ = std::move(other.sJInfos_);
      sHInfos_ = std::move(other.sHInfos_);
#endif
    }
    return *this;
  }

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
#endif

  // Interface to PersistentTapeInfos
  void tapeBaseNames(size_t loc, const std::string &baseName) {
    perTapeInfos_.tapeBaseNames_[loc] = baseName;
  }
  void skipFileCleanup(short tnum) { perTapeInfos_.skipFileCleanup = 1; }
  short skipFileCleanup() const { return perTapeInfos_.skipFileCleanup; }

  double *paramstore() const { return perTapeInfos_.paramstore; }
  void paramstore(double *params) { perTapeInfos_.paramstore = params; }

  char *tay_fileName() const { return perTapeInfos_.tay_fileName; }
  char *op_fileName() const { return perTapeInfos_.op_fileName; }
  char *loc_fileName() const { return perTapeInfos_.loc_fileName; }
  char *val_fileName() const { return perTapeInfos_.val_fileName; }
  void tay_fileName(char *name) { perTapeInfos_.tay_fileName = name; }
  void op_fileName(char *name) { perTapeInfos_.op_fileName = name; }
  void loc_fileName(char *name) { perTapeInfos_.loc_fileName = name; }
  void val_fileName(char *name) { perTapeInfos_.val_fileName = name; }
  int keepTape() const { return perTapeInfos_.keepTape; }
  void keepTape(int flag) { perTapeInfos_.keepTape = flag; }
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
  void read_params();
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

  void deg_save(int val) { tapeInfos_.deg_save = val; }
  int deg_save() const { return tapeInfos_.deg_save; }

  int traceFlag() const { return tapeInfos_.traceFlag; }
  void traceFlag(int flag) { tapeInfos_.traceFlag = flag; }
  int keepTaylors() const { return tapeInfos_.keepTaylors; }
  void keepTaylors(int val) { tapeInfos_.keepTaylors = val; }
  void numTay(size_t num) { tapeInfos_.numTay = num; }

  size_t numTBuffersInUse() const { return tapeInfos_.numTBuffersInUse; }
  void increment_numTBuffersInUse() { tapeInfos_.numTBuffersInUse++; }

  size_t numparam() const { return globalTapeVars_.numparam; }
  void gDegree(size_t degree) { tapeInfos_.gDegree = degree; }
  void workMode(TapeInfos::WORKMODES mode) { tapeInfos_.workMode = mode; }
  TapeInfos::WORKMODES workMode() const { return tapeInfos_.workMode; }

  void increment_numTays_Tape() { ++(tapeInfos_.numTays_Tape); }
  void add_numTays_Tape(int val) { tapeInfos_.numTays_Tape += val; }
  size_t numTays_Tape() const { return tapeInfos_.numTays_Tape; }
  void numTays_Tape(int val) { tapeInfos_.numTays_Tape = val; }

  size_t numLocs_Tape() const { return tapeInfos_.numLocs_Tape; }
  void numLocs_Tape(size_t num) { tapeInfos_.numLocs_Tape = num; }

  size_t numOps_Tape() const { return tapeInfos_.numOps_Tape; }
  void numOps_Tape(size_t num) { tapeInfos_.numOps_Tape = num; }

  size_t numVals_Tape() const { return tapeInfos_.numVals_Tape; }
  void numVals_Tape(size_t num) { tapeInfos_.numVals_Tape = num; }

  void lastTayBlockInCore(char val) { tapeInfos_.lastTayBlockInCore = val; }
  char lastTayBlockInCore() const { return tapeInfos_.lastTayBlockInCore; }

  void decrement_numTays_Tape() { --(tapeInfos_.numTays_Tape); }

  size_t num_eq_prod() const { return tapeInfos_.num_eq_prod; }
  void num_eq_prod(size_t num) { tapeInfos_.num_eq_prod = num; }
  void increment_num_eq_prod() { ++(tapeInfos_.num_eq_prod); }
  void add_num_eq_prod(int val) { tapeInfos_.numDeps += val; }

  void increment_numInds() { ++tapeInfos_.numInds; }
  size_t numInds() const { return tapeInfos_.numInds; }

  void increment_numDeps() { ++tapeInfos_.numDeps; }
  size_t numDeps() const { return tapeInfos_.numDeps; }

  void tay_numInds(size_t val) { tapeInfos_.tay_numInds = val; }
  size_t tay_numInds() const { return tapeInfos_.tay_numInds; }

  void tay_numDeps(size_t val) { tapeInfos_.tay_numDeps = val; }
  size_t tay_numDeps() const { return tapeInfos_.tay_numDeps; }

  void numSwitches(size_t num) { tapeInfos_.numSwitches = num; }
  size_t numSwitches() const { return tapeInfos_.numSwitches; }
  void increment_numSwitches() { ++tapeInfos_.numSwitches; }

  short tapeId() const { return tapeInfos_.tapeId_; }
  size_t no_min_max() { return tapeInfos_.stats[TapeInfos::NO_MIN_MAX]; }
  size_t ext_diff_fct_index() const { return tapeInfos_.ext_diff_fct_index; }
  void ext_diff_fct_index(size_t index) {
    tapeInfos_.ext_diff_fct_index = index;
  }
  int cp_index() const { return tapeInfos_.cpIndex; }
  void cp_index(int index) { tapeInfos_.cpIndex = index; }

  unsigned char *currOp() const { return tapeInfos_.currOp; }
  void currOp(unsigned char *op) { tapeInfos_.currOp = op; }

  size_t *currLoc() const { return tapeInfos_.currLoc; }
  void currLoc(size_t *loc) { tapeInfos_.currLoc = loc; }

  double *currVal() const { return tapeInfos_.currVal; }
  void currVal(double *val) { tapeInfos_.currVal = val; }

  size_t lowestXLoc_for() const { return tapeInfos_.lowestXLoc_for; }
  size_t lowestYLoc_for() const { return tapeInfos_.lowestYLoc_for; }
  size_t lowestXLoc_rev() const { return tapeInfos_.lowestXLoc_rev; }
  size_t lowestYLoc_rev() const { return tapeInfos_.lowestYLoc_rev; }
  void lowestXLoc_for(size_t loc) { tapeInfos_.lowestXLoc_for = loc; }
  void lowestYLoc_for(size_t loc) { tapeInfos_.lowestYLoc_for = loc; }
  void lowestXLoc_rev(size_t loc) { tapeInfos_.lowestXLoc_rev = loc; }
  void lowestYLoc_rev(size_t loc) { tapeInfos_.lowestYLoc_rev = loc; }
  size_t *lowestXLoc_ext_v2() const { return tapeInfos_.lowestXLoc_ext_v2; }
  size_t *lowestYLoc_ext_v2() const { return tapeInfos_.lowestYLoc_ext_v2; }
  void lowestXLoc_ext_v2(size_t *locs) { tapeInfos_.lowestXLoc_ext_v2 = locs; }
  void lowestYLoc_ext_v2(size_t *locs) { tapeInfos_.lowestYLoc_ext_v2 = locs; }
  size_t numDirs_rev() const { return tapeInfos_.numDirs_rev; }
  void numDirs_rev(int dir) { tapeInfos_.numDirs_rev = dir; }
  char in_nested_ctx() const { return tapeInfos_.in_nested_ctx; }

  double *currTay() const { return tapeInfos_.currTay; }
  void currTay(double *pos) { tapeInfos_.currTay = pos; }
  void currTay(double val) { *tapeInfos_.currTay = val; }
  void increment_currTay() { ++tapeInfos_.currTay; }
  void decrement_currTay() { --tapeInfos_.currTay; }

  void lastTayP1(double *pos) { tapeInfos_.lastTayP1 = pos; }
  double *lastTayP1() const { return tapeInfos_.lastTayP1; }

  void lastOpP1(unsigned char *lastOp) { tapeInfos_.lastOpP1 = lastOp; }
  void lastLocP1(size_t *lastLoc) { tapeInfos_.lastLocP1 = lastLoc; }

  void lastValP1(double *lastVal) { tapeInfos_.lastValP1 = lastVal; }
  double *lastValP1() const { return tapeInfos_.lastValP1; }

  void nextBufferNumber(int num) { tapeInfos_.nextBufferNumber = num; }
  int nextBufferNumber() const { return tapeInfos_.nextBufferNumber; }
  void decrement_nextBufferNumber() { --tapeInfos_.nextBufferNumber; }

  // Accessors for tapeInfos_ arrays
  double *dp_T0() const { return tapeInfos_.dp_T0; }
  double *rp_T() const { return tapeInfos_.rp_T; }
  double *rp_A() const { return tapeInfos_.rp_A; }
  double **rpp_A() const { return tapeInfos_.rpp_A; }
  // Setter for tapeInfos_ arrays
  void dp_T0(double *T0) { tapeInfos_.dp_T0 = T0; }
  void rp_T(double *T) { tapeInfos_.rp_T = T; }

  void dpp_T(double **T) { tapeInfos_.dpp_T = T; }
  double *dpp_T(size_t loc) { return tapeInfos_.dpp_T[loc]; }

  void rpp_T(double **T) { tapeInfos_.rpp_T = T; }
  double *rpp_T(size_t loc) { return tapeInfos_.rpp_T[loc]; }
  void rp_A(double *A) { tapeInfos_.rp_A = A; }
  void rpp_A(double **A) { tapeInfos_.rpp_A = A; }
  void upp_A(size_t **A) { tapeInfos_.upp_A = A; }

  constexpr static size_t maxLocsPerOp() { return TapeInfos::maxLocsPerOp; }

  void put_op(OPCODES op, size_t reserveExtraLocations = 0) {
    return tapeInfos_.put_op(op, loc_fileName(), op_fileName(), val_fileName(),
                             reserveExtraLocations);
  }
  void put_op_block(unsigned char *lastOpP1) {
    tapeInfos_.put_op_block(lastOpP1, op_fileName());
  };
  bool isTaping() { return tapeInfos_.traceFlag != 0; }

  /* writes a block of operations onto hard disk and handles file creation,
   * removal, ... */
  void get_op_block_f() { tapeInfos_.get_op_block_f(); };
  /* reads the next operations block into the internal buffer */
  void get_op_block_r() { return tapeInfos_.get_op_block_r(); };
  /* reads the previous block of operations into the internal buffer */

  /* puts a single locations into the location buffer, no disk access */
  void put_loc_block(size_t *lastLocP1) {
    return tapeInfos_.put_loc_block(lastLocP1, loc_fileName());
  };
  /* writes a block of locations onto hard disk and handles file creation,
   * removal, ... */
  void get_loc_block_f() { return tapeInfos_.get_loc_block_f(); };
  /* reads the next block of locations into the internal buffer */
  void get_loc_block_r() { return tapeInfos_.get_loc_block_r(); };
  void put_loc(size_t loc) { return tapeInfos_.put_loc(loc); };
#ifndef ADOLC_HARDDEBUG
  char get_op_f() { return *(tapeInfos_.currOp)++; }
  char get_op_r() { return *--(tapeInfos_.currOp); }
  size_t get_locint_f() { return *(tapeInfos_.currLoc)++; }
  size_t get_locint_r() { return *--(tapeInfos_.currLoc); }
  double get_val_f() { return *(tapeInfos_.currVal)++; }
  double get_val_r() { return *--(tapeInfos_.currVal); }
#else
  unsigned char ValueTape::get_op_f() { return tapeInfos_.get_op_f(); }
  unsigned char ValueTape::get_op_r() { return tapeInfos_.get_op_r(); }
  size_t ValueTape::get_size_t_f() { return tapeInfos_.get_size_t_f(); }
  size_t ValueTape::get_size_t_r() { return tapeInfos_.get_size_t_r(); }
  double ValueTape::get_val_f() { return tapeInfos_.get_val_f(); }
  double ValueTape::get_val_r() { return tapeInfos_.get_val_r(); }
#endif // ADOLC_HARDDEBUG
  void put_val(const double val) {
    *tapeInfos_.currVal = val;
    ++tapeInfos_.currVal;
  }
  /* puts a single constant into the location buffer, no disk access */
  void put_vals_writeBlock(double *reals, size_t numReals) {
    return tapeInfos_.put_vals_writeBlock(reals, numReals, op_fileName(),
                                          val_fileName());
  };
  /* fill the constants buffer and write it to disk */
  void put_vals_notWriteBlock(double *reals, size_t numReals) {
    return tapeInfos_.put_vals_notWriteBlock(reals, numReals);
  }
  /* write some constants to the buffer without disk access */
  void put_val_block(double *lastValP1) {
    return tapeInfos_.put_val_block(lastValP1, val_fileName());
  };
  /* writes a block of constants (real) onto hard disk and handles file
   * creation, removal, ... */
  void get_val_block_f() { return tapeInfos_.get_val_block_f(); };
  /* reads the next block of constants into the internal buffer */
  void get_val_block_r() { return tapeInfos_.get_val_block_r(); };
  /* reads the previous block of constants into the internal buffer */
  size_t get_val_space() {
    return tapeInfos_.get_val_space(op_fileName(), val_fileName());
  };
  /* returns the number of free constants in the real tape, ensures that it
   * is at least 5 */
  double *get_val_v_f(size_t size) { return tapeInfos_.get_val_v_f(size); }
  /* return a pointer to the first element of a constants vector
   * -- Forward Mode -- */
  double *get_val_v_r(size_t size) { return tapeInfos_.get_val_v_r(size); }
  /* return a pointer to the first element of a constants vector
   * -- Reverse Mode -- */
  /* suspicious function, maybe for vector class - kept for compatibility */
  void reset_val_r() { return tapeInfos_.reset_val_r(); }
  /* updates */
  int upd_resloc(size_t temp, size_t lhs) {
    return tapeInfos_.upd_resloc(temp, lhs);
  }
  int upd_resloc_check(size_t temp) {
    return tapeInfos_.upd_resloc_check(temp);
  }
  int upd_resloc_inc_prod(size_t temp, size_t newlhs, unsigned char newop) {
    return tapeInfos_.upd_resloc_inc_prod(temp, newlhs, newop);
  }
  size_t get_num_param() { return tapeInfos_.stats[TapeInfos::NUM_PARAM]; }
  void set_nested_ctx(char nested) { tapeInfos_.in_nested_ctx = nested; }
  char currently_nested() { return tapeInfos_.in_nested_ctx; }
  int inUse() const { return tapeInfos_.inUse; }
  void inUse(int val) { tapeInfos_.inUse = val; }
  char tapingComplete() const { return tapeInfos_.tapingComplete; }
  void tapingComplete(char val) { tapeInfos_.tapingComplete = val; }
  FILE *tay_file() const { return tapeInfos_.tay_file; }
  FILE *op_file() const { return tapeInfos_.op_file; }
  FILE *val_file() const { return tapeInfos_.val_file; }
  FILE *loc_file() const { return tapeInfos_.loc_file; }
  void tay_file(FILE *file) { tapeInfos_.tay_file = file; }
  void op_file(FILE *file) { tapeInfos_.op_file = file; }
  void val_file(FILE *file) { tapeInfos_.val_file = file; }
  void loc_file(FILE *file) { tapeInfos_.loc_file = file; }
  unsigned char *opBuffer() const { return tapeInfos_.opBuffer; }
  double *valBuffer() const { return tapeInfos_.valBuffer; }
  size_t *locBuffer() const { return tapeInfos_.locBuffer; }
  double *signature() const { return tapeInfos_.signature; }
  double *tayBuffer() const { return tapeInfos_.tayBuffer; }

  void opBuffer(unsigned char *buffer) { tapeInfos_.opBuffer = buffer; }
  void valBuffer(double *buffer) { tapeInfos_.valBuffer = buffer; }
  void locBuffer(size_t *buffer) { tapeInfos_.locBuffer = buffer; }
  void signature(double *buffer) { tapeInfos_.signature = buffer; }
  void tayBuffer(double *buffer) { tapeInfos_.tayBuffer = buffer; }

  void initTapeInfos_keep();
  // free all resources used by a tape before overwriting the tape
  void freeTapeResources() { tapeInfos_.freeTapeResources(); }
  // free/allocate memory for buffers, initialize pointers
  void initTapeBuffers();

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

  size_t maxNumberTaylorBuffers() const {
    return globalTapeVars_.maxNumberTaylorBuffers;
  }
  void maxNumberTaylorBuffers(size_t num) {
    globalTapeVars_.maxNumberTaylorBuffers = num;
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
    if (!isTaping())
      globalTapeVars_.nominmaxFlag = 1;
    else
      ADOLCError::fail(ADOLCError::ErrorType::ENABLE_MINMAX_USING_ABS,
                       CURRENT_LOCATION);
  }

  void disableMinMaxUsingAbs() {
    if (!isTaping())
      globalTapeVars_.nominmaxFlag = 0;
    else
      ADOLCError::fail(ADOLCError::ErrorType::DISABLE_MINMAX_USING_ABS,
                       CURRENT_LOCATION);
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
  /*--------------------------------------------------------------------------*/
  /* This function sets the flag "newTape" if either a taylor buffer has been */
  /* created or a taping process has been performed. Calling the function is  */
  /* also useful to "convince" the linker of including the cleaner part into  */
  /* the binary when linking statically!                                      */
  /*--------------------------------------------------------------------------*/
  void markNewTape() { globalTapeVars_.newTape = 1; }

  // ------------------------------- Buffer utils ---------------------------
  void cp_clearStack();
  CpInfos *cp_append() { return cp_buffer_.append(); }
  CpInfos *cp_getElement(int index) { return cp_buffer_.getElement(index); }

  ext_diff_fct *ext_diff_append() { return ext_buffer_.append(); }
  ext_diff_fct *ext_diff_getElement(int index) {
    return ext_buffer_.getElement(index);
  }
  ext_diff_fct_v2 *ext_diff_v2_append() { return ext2_buffer_.append(); }
  ext_diff_fct_v2 *ext_diff_v2_getElement(int index) {
    return ext2_buffer_.getElement(index);
  }

  // ------------------------------------------- Combined
  /* tries to read a local config file containing, e.g., buffer sizes */
  std::array<std::string, 4> readConfigFile();
  // ------------------------ set up statics for writing taylor data
  void taylor_begin(uint bufferSize, int degreeSave);

  // close taylor file if necessary and refill buffer if possible
  void taylor_close(bool resetData);
  // initializes a reverse sweep
  void taylor_back(short tag, int *dep, int *ind, int *degree);

  void write_taylor(size_t loc, int keep) {
    return tapeInfos_.write_taylor(loc, keep, tay_fileName());
  }

  // writes the block of size depth of taylor coefficients from point loc to
  // the taylor buffer, if the buffer is filled, then it is written to the
  // taylor tape
  void write_taylors(size_t, int keep, int degree, int numDir);

  void write_scaylor(double val) {
    return tapeInfos_.write_scaylor(val, tay_fileName());
  }

  // write_scaylors writes #size elements from x to the taylor buffer void
  void write_scaylors(double *x, uint size);

  // deletes the last (single) element (x) of the taylor buffer
  void delete_scaylor(size_t loc) {
    --tapeInfos_.currTay;
    globalTapeVars_.store[loc] = *tapeInfos_.currTay;
  }

  // writes the taylor stack buffer onto hard disk
  void put_tay_block(double *lastValP1) {
    return tapeInfos_.put_tay_block(lastValP1, tay_fileName());
  }

  // puts a taylor value from the value stack buffer to the taylor buffer
  void get_taylor(size_t loc) {
    if (tapeInfos_.currTay == tapeInfos_.tayBuffer)
      get_tay_block_r();

    --tapeInfos_.currTay;
    tapeInfos_.rp_T[loc] = *tapeInfos_.currTay;
  }

  // puts a block of taylor coefficients from the value stack buffer to the
  // taylor buffer --- Higher Order Scalar
  void get_taylors(size_t, int degree);

  // puts a block of taylor coefficients from the value stack buffer to the
  // taylor buffer --- Higher Order Vector
  void get_taylors_p(size_t, int degree, int numDir);

  // gets the next (previous block) of the value stack
  void get_tay_block_r() { return tapeInfos_.get_tay_block_r(); }
  // initialize a forward sweep, get stats, open tapes, fill buffers, ...
  void init_for_sweep(short tag);

  // initialize a reverse sweep, get stats, open tapes, fill buffers, ...
  void init_rev_sweep(short tag);

  // finish a forward or reverse sweep
  void end_sweep();
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
  void openTape(char mode);

  // updates the tape infos for the given ID - a tapeInfos struct is created
  // and registered if non is found but its state will remain "not in use"
  std::shared_ptr<ValueTape> getTapeInfos(short tapeId);

  // close open tapes, update stats and clean up
  void close_tape(int flag);

  // release the current tape and give control to the previous one
  // if keepVS is not zero (keep value stack for reverse) => belonging
  // TapeInfos are kept marked as being in use
  void releaseTape();
  /****************************************************************************/
  /* Discards parameters from the end of value tape during reverse mode */
  /****************************************************************************/
  void discard_params_r();
  /****************************************************************************/
  /* Overrides the parameters for the next evaluations. This will invalidate
   */
  /* the taylor stack, so next reverse call will fail, if not preceded by a
   */
  /* forward call after setting the parameters. */
  /****************************************************************************/
  void set_param_vec(short tag, size_t numparam, double *paramvec);
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
    double a, b;
#ifdef inf_num
    a = non_num;
    b = non_den;
#endif
    return a / b;
  }

  static double make_inf() {
    double a, b;
#ifdef inf_num
    a = inf_num;
    b = inf_den;
#endif
    return a / b;
  }

  void cp_takeshot(CpInfos *cpInfos);
  void cp_restore(CpInfos *cpInfos);
  void cp_release(CpInfos *cpInfos);
  CpInfos *get_cp_fct(int index) { return cp_buffer_.getElement(index); }
};

#endif // ADOLC_VALUETAPE_H