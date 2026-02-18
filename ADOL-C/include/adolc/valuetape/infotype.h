
#ifndef ADOLC_INFO_TYPE_H
#define ADOLC_INFO_TYPE_H
#include <adolc/internal/usrparms.h>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <string_view>

namespace ADOLC::detail {
template <class T, class TInfos, class ErrorType>
concept InfoType =
    requires(TInfos &tapeInfos, std::string_view fileName, std::string_view) {
      typename T::value;

      { T::num } -> std::same_as<const typename TInfos::StatEntries &>;
      { T::fileAccess } -> std::same_as<const typename TInfos::StatEntries &>;
      { T::bufferSize } -> std::same_as<const typename TInfos::StatEntries &>;
      { T::error } -> std::convertible_to<ErrorType>;
      { T::chunkSize } -> std::convertible_to<size_t>;

      { T::bufferBegin(tapeInfos) } -> std::same_as<typename T::value *>;
      T::setCurr(tapeInfos, (typename T::value *)nullptr);
      T::setNum(tapeInfos, size_t{});
      { T::getNum(tapeInfos) };
      T::file(tapeInfos);
      T::openFile(tapeInfos, fileName);
      { T::removeFile(fileName) } -> std::same_as<int>;
    };

template <class TInfos, class EType> struct OpInfo {
  using value = unsigned char;
  using StatEntries = typename TInfos::StatEntries;
  static const StatEntries num = TInfos::NUM_OPERATIONS;
  static const StatEntries fileAccess = TInfos::OP_FILE_ACCESS;
  static const StatEntries bufferSize = TInfos::OP_BUFFER_SIZE;
  static constexpr EType error = EType::EVAL_OP_TAPE_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value);

  static void setNum(TInfos &tapeInfos, size_t n) { tapeInfos.numOps_Tape = n; }
  static size_t getNum(TInfos &tapeInfos) { return tapeInfos.numOps_Tape; }
  static void setCurr(TInfos &tapeInfos, value *p) { tapeInfos.currOp = p; }
  static value *bufferBegin(TInfos &tapeInfos) { return tapeInfos.opBuffer; }
  static FILE *file(TInfos &tapeInfos) { return tapeInfos.op_file; }
  static void openFile(TInfos &tapeInfos, std::string_view fileName,
                       std::string_view mode = "rb") {
    tapeInfos.op_file = fopen(fileName.data(), mode.data());
  }
  static int removeFile(std::string_view fileName) {
    return remove(fileName.data());
  }
};

template <class TInfos, class EType> struct LocInfo {
  using value = size_t;
  using StatEntries = typename TInfos::StatEntries;
  static const StatEntries num = TInfos::NUM_LOCATIONS;
  static const StatEntries fileAccess = TInfos::LOC_FILE_ACCESS;
  static const StatEntries bufferSize = TInfos::LOC_BUFFER_SIZE;
  static constexpr EType error = EType::EVAL_LOC_TAPE_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value);

  static void setNum(TInfos &tapeInfos, size_t n) {
    tapeInfos.numLocs_Tape = n;
  }
  static size_t getNum(TInfos &tapeInfos) { return tapeInfos.numLocs_Tape; }
  static void setCurr(TInfos &tapeInfos, value *p) { tapeInfos.currLoc = p; }
  static value *bufferBegin(TInfos &tapeInfos) { return tapeInfos.locBuffer; }
  static FILE *file(TInfos &tapeInfos) { return tapeInfos.loc_file; }
  static void openFile(TInfos &tapeInfos, std::string_view fileName,
                       std::string_view mode = "rb") {
    tapeInfos.loc_file = fopen(fileName.data(), mode.data());
  }
  static int removeFile(std::string_view fileName) {
    return remove(fileName.data());
  }
};

template <class TInfos, class EType> struct ValInfo {
  using value = double;
  using StatEntries = typename TInfos::StatEntries;
  static const StatEntries num = TInfos::NUM_VALUES;
  static const StatEntries fileAccess = TInfos::VAL_FILE_ACCESS;
  static const StatEntries bufferSize = TInfos::VAL_BUFFER_SIZE;
  static constexpr EType error = EType::EVAL_VAL_TAPE_READ_FAILED;

  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value);

  static void setNum(TInfos &tapeInfos, size_t n) {
    tapeInfos.numVals_Tape = n;
  }
  static size_t getNum(TInfos &tapeInfos) { return tapeInfos.numVals_Tape; }
  static void setCurr(TInfos &tapeInfos, value *p) { tapeInfos.currVal = p; }
  static value *bufferBegin(TInfos &tapeInfos) { return tapeInfos.valBuffer; }
  static FILE *file(TInfos &tapeInfos) { return tapeInfos.val_file; }
  static void openFile(TInfos &tapeInfos, std::string_view fileName,
                       std::string_view mode = "rb") {
    tapeInfos.val_file = fopen(fileName.data(), mode.data());
  }
  static int removeFile(std::string_view fileName) {
    return remove(fileName.data());
  }
};
}; // namespace ADOLC::detail

#endif // ADOLC_INFO_TYPE_H