
#ifndef ADOLC_INFO_TYPE_H
#define ADOLC_INFO_TYPE_H
#include <adolc/adolcerror.h>
#include <adolc/valuetape/tapeinfos.h>
#include <concepts>

namespace ADOLC::detail {
template <class T, class Tape>
concept InfoType = requires(Tape &tape) {
  typename T::value;

  { T::num } -> std::same_as<const TapeInfos::StatEntries &>;
  { T::fileAccess } -> std::same_as<const TapeInfos::StatEntries &>;
  { T::bufferSize } -> std::same_as<const TapeInfos::StatEntries &>;
  { T::error } -> std::same_as<const ADOLCError::ErrorType &>;
  { T::chunkSize } -> std::same_as<const size_t &>;

  { T::buffer(tape) } -> std::same_as<typename T::value *>;
  T::setCurr(tape, (typename T::value *)nullptr);
  T::setNum(tape, size_t{});
  T::file(tape);
  T::openFile(tape);
};

template <class Tape> struct OpInfo {
  using value = unsigned char;
  static const TapeInfos::StatEntries num = TapeInfos::NUM_OPERATIONS;
  static const TapeInfos::StatEntries fileAccess = TapeInfos::OP_FILE_ACCESS;
  static const TapeInfos::StatEntries bufferSize = TapeInfos::OP_BUFFER_SIZE;
  static constexpr ADOLCError::ErrorType error =
      ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value);

  static void setNum(Tape &tape, size_t n) { tape.numOps_Tape(n); }
  static void setCurr(Tape &tape, value *p) { tape.currOp(p); }
  static value *buffer(Tape &tape) { return tape.opBuffer(); }
  static FILE *file(Tape &tape) { return tape.op_file(); }
  static void openFile(Tape &tape) {
    tape.op_file(fopen(tape.op_fileName(), "rb"));
  }
};
template <class Tape> struct LocInfo {
  using value = size_t;
  static const TapeInfos::StatEntries num = TapeInfos::NUM_LOCATIONS;
  static const TapeInfos::StatEntries fileAccess = TapeInfos::LOC_FILE_ACCESS;
  static const TapeInfos::StatEntries bufferSize = TapeInfos::LOC_BUFFER_SIZE;
  static constexpr ADOLCError::ErrorType error =
      ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value);

  static void setNum(Tape &tape, size_t n) { tape.numLocs_Tape(n); }
  static void setCurr(Tape &tape, value *p) { tape.currLoc(p); }
  static value *buffer(Tape &tape) { return tape.locBuffer(); }
  static FILE *file(Tape &tape) { return tape.loc_file(); }
  static void openFile(Tape &tape) {
    tape.loc_file(fopen(tape.loc_fileName(), "rb"));
  }
};

template <class Tape> struct ValInfo {
  using value = double;
  static const TapeInfos::StatEntries num = TapeInfos::NUM_VALUES;
  static const TapeInfos::StatEntries fileAccess = TapeInfos::VAL_FILE_ACCESS;
  static const TapeInfos::StatEntries bufferSize = TapeInfos::VAL_BUFFER_SIZE;
  static constexpr ADOLCError::ErrorType error =
      ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED;

  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value);

  static void setNum(Tape &tape, size_t n) { tape.numVals_Tape(n); }
  static void setCurr(Tape &tape, value *p) { tape.currVal(p); }
  static value *buffer(Tape &tape) { return tape.valBuffer(); }
  static FILE *file(Tape &tape) { return tape.val_file(); }
  static void openFile(Tape &tape) {
    tape.val_file(fopen(tape.val_fileName(), "rb"));
  }
};
}; // namespace ADOLC::detail

#endif // ADOLC_INFO_TYPE_H