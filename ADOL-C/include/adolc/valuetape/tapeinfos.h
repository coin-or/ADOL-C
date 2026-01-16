#ifndef ADOLC_TAPEINFOS_H
#define ADOLC_TAPEINFOS_H

#include <adolc/adalloc.h>
#include <adolc/oplate.h>
#include <array>
#include <memory>

struct TapeInfos {

  // named indices of to print out value tape stats
  enum StatEntries {
    NUM_INDEPENDENTS, /* # of independent variables */
    NUM_DEPENDENTS,   /* # of dependent variables */
    NUM_MAX_LIVES,    /* max # of live variables */
    TAY_STACK_SIZE,   /* # of values in the taylor (value) stack */
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
  // option for removeTape
  enum TapeRemovalType {
    REMOVE_FROM_CORE,
    REMOVE_COMPLETELY // also remove the tape files
  };

  // modes for the tape evaluation; set by functions like "fos_forward"
  enum WORKMODES {
    ADOLC_NO_MODE,

    ZOS_FORWARD,
    FOS_FORWARD,
    FOV_FORWARD,
    HOS_FORWARD,
    HOV_FORWARD,

    FOS_REVERSE,
    FOV_REVERSE,
    HOS_REVERSE,
    HOV_REVERSE,

    TAPING
  };

  ~TapeInfos();
  TapeInfos() = default;
  TapeInfos(short tapeId);
  TapeInfos(const TapeInfos &) = delete;
  TapeInfos &operator=(const TapeInfos &) = delete;
  TapeInfos(TapeInfos &&other) noexcept;
  TapeInfos &operator=(TapeInfos &&other) noexcept;

  short tapeId_{-1};
  int inUse{0};
  size_t numInds{0};
  size_t numDeps{0};
  // 1 - write taylor stack in taping mode
  int keepTaylors{0};
  std::array<size_t, STAT_SIZE> stats{0};
  int traceFlag{0};
  char tapingComplete{0};

  /* ------ operations tape ------- */
  // file descriptor
  FILE *op_file{nullptr};
  // pointer to the current tape buffer
  unsigned char *opBuffer{nullptr};
  // pointer to the current opcode
  unsigned char *currOp{nullptr};
  // pointer to element following the buffer
  unsigned char *lastOpP1{nullptr};
  // overall number of opcodes
  size_t numOps_Tape{0};
  // overall number of eq_*_prod for nlf
  size_t num_eq_prod{0};

  /* --------- values (real) tape ------- */
  FILE *val_file{nullptr};
  double *valBuffer{nullptr};
  double *currVal{nullptr};
  double *lastValP1{nullptr};
  size_t numVals_Tape{0};

  /* ---------- locations tape --------- */
  FILE *loc_file{nullptr};
  size_t *locBuffer{nullptr};
  size_t *currLoc{nullptr};
  size_t *lastLocP1{nullptr};
  size_t numLocs_Tape{0};

  /* taylor stack tape */
  FILE *tay_file{nullptr};
  double *tayBuffer{nullptr};
  double *currTay{nullptr};
  double *lastTayP1{nullptr};
  size_t numTays_Tape{0};
  size_t numTBuffersInUse{0};
  // the next Buffer to read back
  size_t nextBufferNumber{0};
  // == 1 if last taylor buffer is still in
  // in core(first call of reverse)
  char lastTayBlockInCore{0};
  // derivative buffer - forward
  double **T_for{nullptr};
  // degree to save and saved respectively
  int deg_save{0};
  // # of independents for the taylor stack
  size_t tay_numInds{0};
  // # of dependents for the taylor stack
  size_t tay_numDeps{0};

  /* ---------- checkpointing --------- */
  // location of the first ind. - forward mode
  size_t lowestXLoc_for{0};
  // location of the first dep. - forward mode
  size_t lowestYLoc_for{0};
  // location of the first ind. - reverse mode
  size_t lowestXLoc_rev{0};
  // location of the first dep. - reverse mode
  size_t lowestYLoc_rev{0};
  // index of the curr. cp function <- tape
  size_t cpIndex{0};
  // # of directions for **v_reverse (checkpointing)
  int numDirs_rev{0};

  size_t *lowestXLoc_ext_v2{nullptr};
  size_t *lowestYLoc_ext_v2{nullptr};

  /* evaluation forward */
  double *dp_T0{nullptr};
  size_t gDegree{0};
  size_t numTay{0};
  enum WORKMODES workMode;

  double **dpp_T{nullptr};

  /* ---------- evaluation reverse------- */
  double *rp_T{nullptr};
  double **rpp_T{nullptr};
  double *rp_A{nullptr};
  double **rpp_A{nullptr};
  size_t **upp_A{nullptr};

  /* extern diff. fcts */
  size_t ext_diff_fct_index{0}; /* set by forward and reverse (from tape) */
  char in_nested_ctx{0};

  size_t numSwitches{0};
  size_t *switchlocs{nullptr};
  double *signature{nullptr};

  constexpr static size_t maxLocsPerOp{10}; // used in tape_loc_...

  void freeTapeResources();

  // writes the block of size depth of taylor coefficients from point loc to
  // the taylor buffer, if the buffer is filled, then it is written to the
  // taylor tape
  void write_taylor(size_t loc, std::ptrdiff_t keep, const char *tay_fileName);
  // writes a single element (x) to the taylor buffer and writes the buffer
  // to disk if necessary
  void write_scaylor(double val, const char *tay_fileName) {
    if (currTay == lastTayP1)
      put_tay_block(tay_fileName);
    *currTay = val;
    ++currTay;
  }
  void put_tay_block(const char *tay_fileName);
  void get_tay_block_r();

  // functions for handling loc tape
  void put_loc(size_t loc) {
    *currLoc = loc;
    ++currLoc;
  }

  void put_loc_block(const char *loc_fileName);
  void get_loc_block_f();
  void get_loc_block_r();
  // functions for handling op tape

  // puts an operation into the operation buffer, ensures that location
  // buffer and constants buffer are prepared to take the belonging stuff
  void put_op(OPCODES op, const char *loc_fileName, const char *op_fileName,
              const char *val_fileName, size_t reserveExtraLocations = 0);
  void put_op_block(const char *op_fileName);
  void get_op_block_f();
  void get_op_block_r();

  // functions for handling val tape

  /****************************************************************************/
  /* Write some constants to the buffer without disk access                   */
  /****************************************************************************/
  void put_vals_notWriteBlock(double *vals, size_t numVals) {
    for (size_t i = 0; i < numVals; ++i) {
      *currVal = vals[i];
      ++currVal;
    }
  }
  void put_val_block(const char *val_fileName);
  void put_vals_writeBlock(double *vals, size_t numVals,
                           const char *op_fileName, const char *val_fileName);
  void get_val_block_r();
  void get_val_block_f();
  /****************************************************************************/
  /* Returns a pointer to the first element of a values vector and skips the  */
  /* vector. -- Forward Mode --                                               */
  /****************************************************************************/
  double *get_val_v_f(size_t size) {
    double *temp = currVal;
    currVal += size;
    return temp;
  }
  /****************************************************************************/
  /* Returns a pointer to the first element of a values vector and skips the  */
  /* vector. -- Reverse Mode --                                               */
  /****************************************************************************/
  double *get_val_v_r(size_t size) { return currVal -= size; }

  /****************************************************************************/
  /* Not sure what's going on here! -> vector class ?  --- kowarz             */
  /****************************************************************************/
  void reset_val_r(void) {
    if (currVal == valBuffer)
      get_val_block_r();
  }

  /****************************************************************************/
  /* Update locations tape to remove assignments involving temp. variables.   */
  /* e.g.  t = a + b ; y = t  =>  y = a + b                                   */
  /****************************************************************************/
  int upd_resloc(size_t temp, size_t lhs) {
    // LocBuffer points to the first entry of the Locations and CurrLoc-1 to the
    // last placed location in the buffer. Thus, the check ask if there is no
    // element on the tape.
    if (currLoc - locBuffer < 1)
      return 0;
    if (temp == *(currLoc - 1)) {
      *(currLoc - 1) = lhs;
      return 1;
    }
    return 0;
  }

  int upd_resloc_check(const size_t temp) {
    // LocBuffer points to the first entry of the Locations and CurrLoc-1 to the
    // last placed location in the buffer. Thus, the check ask if there is no
    // element on the tape.
    if (currLoc - locBuffer < 1)
      return 0;
    // checks if tape-element represented by "tmp" is the last created.
    if (temp == *(currLoc - 1)) {
      return 1;
    }
    return 0;
  }

  /****************************************************************************/
  /* Update locations and operations tape to remove special operations inv.   */
  /* temporary variables. e.g.  t = a * b ; y += t  =>  y += a * b            */
  /****************************************************************************/
  int upd_resloc_inc_prod(size_t temp, size_t newlhs, unsigned char newop) {
    if (currLoc - locBuffer < 3)
      return 0;
    if (currOp - opBuffer < 1)
      return 0;
    if (temp == *(currLoc - 1) && mult_a_a == *(currOp - 1) &&
        /* skipping recursive case */
        newlhs != *(currLoc - 2) && newlhs != *(currLoc - 3)) {
      *(currLoc - 1) = newlhs;
      *(currOp - 1) = newop;
      return 1;
    }
    return 0;
  }

  /****************************************************************************/
/*                                                          DEBUG FUNCTIONS */
#ifdef ADOLC_HARDDEBUG
  unsigned char get_op_f() {
    unsigned char temp = *currOp;
    ++currOp;
    fprintf(DIAG_OUT, "f_op: %i\n", temp - '\0'); /* why -'\0' ??? kowarz */
    return temp;
  }
  unsigned char get_op_r() {
    --currOp;
    unsigned char temp = *currOp;
    fprintf(DIAG_OUT, "r_op: %i\n", temp - '\0');
    return temp;
  }
  size_t get_size_t_f() {
    size_t temp = *currLoc;
    ++currLoc;
    fprintf(DIAG_OUT, "f_loc: %i\n", temp);
    return temp;
  }
  size_t get_size_t_r() {
    --currLoc;
    unsigned char temp = *currLoc;
    fprintf(DIAG_OUT, "r_loc: %i\n", temp);
    return temp;
  }
  double get_val_f() {
    double temp = *currVal;
    ++currVal;
    fprintf(DIAG_OUT, "f_val: %e\n", temp);
    return temp;
  }
  double get_val_r() {
    --currVal;
    double temp = *currVal;
    fprintf(DIAG_OUT, "r_val: %e\n", temp);
    return temp;
  }
#endif

  size_t get_val_space(const char *op_fileName, const char *val_fileName);
};

#endif // ADOLC_TAPEINFOS_H
