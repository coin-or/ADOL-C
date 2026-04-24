
#ifndef ADOLC_INFO_TYPE_H
#define ADOLC_INFO_TYPE_H
#include <adolc/internal/usrparms.h> // ADOLC_IO_CHUNK_SIZE
#include <concepts>
#include <cstddef>
#include <cstdio>

/*
  This header defines a small “policy” interface used by ADOL-C tape I/O code.

  The idea:
    - Different tapes (op/loc/val/tay) store their data in different fields of a
      TInfos object (buffers, file handles, counters, etc.).
    - The tape-handling algorithms want one uniform API to access those fields.
    - The *Info structs* (OpInfo, LocInfo, ValInfo, TayInfo) provide that API.
    - If a check about how many file accesses has to be done InfoType is used,
  otherwise the less restrictive InfoTypeBase

  Usage pattern (typical):
    template<class Info, class TInfos, class Err>
      requires ADOLC::detail::InfoTypeBase<Info, TInfos, Err>
    void do_something(TInfos& ti, ...);

  Each Info struct is a stateless “adapter” that:
    - names the element type stored in the buffer (Info::value_type)
    - exposes a few bookkeeping constants (Info::num, Info::bufferSize, ...)
    - maps generic operations to the concrete fields in TInfos
  (Info::removeFile, ...)
*/

namespace ADOLC::detail {

/**
 * @brief Concept describing the required interface of a tape “Info” adapter.
 *
 * @tparam T         The adapter type (e.g. OpInfo<TInfos, EType>).
 * @tparam TInfos    The tape state/metadata type that stores buffers, counters,
 *                   file handles, etc. (the adapter maps into these fields).
 * @tparam ErrorType The error enum/type used by higher-level code.
 *
 * The concept intentionally checks:
 *   - existence of the nested element type (T::value_type)
 *   - presence of a few static “category” constants (num/bufferSize)
 *   - presence of an error code constant (T::error) convertible to ErrorType
 *   - chunkSize convertible to size_t (bytes per I/O block / element size)
 *   - required static member functions for buffer + file access
 *
 * Note:
 *   - The requires-clause uses expression requirements because these are
 *     “interface adapters” accessed statically (no instances of T).
 *   - We accept convertible_to for error/chunkSize because they are values
 *     (constexpr) and the exact reference category is not important.
 */
template <class T, class TInfos, class ErrorType>
concept InfoTypeBase = requires(TInfos &tapeInfos, const char *fileName) {
  // element type stored in the tape buffer
  typename T::value_type;

  // “statistic entry” identifiers used by ADOL-C bookkeeping
  { T::num } -> std::same_as<const typename TInfos::StatEntries &>;
  { T::bufferSize } -> std::same_as<const typename TInfos::StatEntries &>;

  // error code associated with this tape type (must be usable as ErrorType)
  { T::error } -> std::convertible_to<ErrorType>;

  // number of elements read/written per I/O chunk (must be usable as
  // size_t)
  { T::chunkSize } -> std::convertible_to<size_t>;

  // pointer to the begin of the tape buffer
  { T::bufferBegin(tapeInfos) } -> std::same_as<typename T::value_type *>;

  // map generic “current pointer” and “count” operations to TInfos fields
  T::setCurr(tapeInfos, size_t{});
  T::setNum(tapeInfos, size_t{});
  { T::getNum(tapeInfos) };

  // file handle access and file lifecycle operations
  T::file(tapeInfos);
  T::openFile(tapeInfos, fileName);
  { T::removeFile(fileName) } -> std::same_as<int>;
};

template <class T, class TInfos>
concept HasFileAccessEntry = requires {
  { T::fileAccess } -> std::same_as<const typename TInfos::StatEntries &>;
};

/**
 * @brief Adds to the concept InfoTypeBase the check for file access counters.
 *
 * @tparam T         The adapter type (e.g. OpInfo<TInfos, EType>).
 * @tparam TInfos    The tape state/metadata type that stores buffers, counters,
 *                   file handles, etc. (the adapter maps into these fields).
 * @tparam ErrorType The error enum/type used by higher-level code.
 *
 */
template <class T, class TInfos, class ErrorType>
concept InfoType =
    InfoTypeBase<T, TInfos, ErrorType> && HasFileAccessEntry<T, TInfos>;

/// Wrapper for fread
template <typename TInfos, typename ErrorType,
          InfoTypeBase<TInfos, ErrorType> Info>
static size_t read(TInfos &tapeInfos, size_t chunk, size_t size) {
  return fread(Info::bufferBegin(tapeInfos) + (chunk * Info::chunkSize),
               size * sizeof(typename Info::value_type), 1,
               Info::file(tapeInfos));
}

/// Wrapper of fread
template <typename TInfos, typename ErrorType,
          InfoTypeBase<TInfos, ErrorType> Info>
static size_t write(TInfos &tapeInfos, size_t chunk, size_t size) {
  return fwrite(Info::bufferBegin(tapeInfos) + (chunk * Info::chunkSize),
                size * sizeof(typename Info::value_type), 1,
                Info::file(tapeInfos));
}

/**
 * @brief Adapter for the operations tape (op tape). Fulfills InfoType.
 *
 * Maps generic operations to:
 *   - tapeInfos.opBuffer_
 *
 * The StatEntries constants (num/fileAccess/bufferSize) are used for ADOL-C
 * statistics/counters.
 */
template <class TInfos, class EType> struct OpInfo {
  using value_type = unsigned char;
  using StatEntries = typename TInfos::StatEntries;

  static const StatEntries num = TInfos::NUM_OPERATIONS;
  static const StatEntries fileAccess = TInfos::OP_FILE_ACCESS;
  static const StatEntries bufferSize = TInfos::OP_BUFFER_SIZE;

  static constexpr EType error = EType::EVAL_OP_TAPE_READ_FAILED;

  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static void setNum(TInfos &tapeInfos, size_t n) {
    tapeInfos.opBuffer_.numOnTape(n);
  }
  static size_t getNum(TInfos &tapeInfos) {
    return tapeInfos.opBuffer_.numOnTape();
  }
  static void setCurr(TInfos &tapeInfos, size_t loc) {
    tapeInfos.opBuffer_.position(loc);
  }

  static value_type *bufferBegin(TInfos &tapeInfos) {
    return tapeInfos.opBuffer_.begin();
  }

  static FILE *file(TInfos &tapeInfos) { return tapeInfos.opBuffer_.file(); }
  static void openFile(TInfos &tapeInfos, const char *fileName,
                       const char *mode = "rb") {
    tapeInfos.opBuffer_.openFile(fileName, mode);
  }
  static int removeFile(const char *fileName) { return remove(fileName); }
};

/**
 * @brief Adapter for the locations tape (loc tape). Fulfills InfoType.
 *
 * Maps generic operations to:
 *   - tapeInfos.locBuffer_
 */
template <class TInfos, class EType> struct LocInfo {
  using value_type = size_t;
  using StatEntries = typename TInfos::StatEntries;

  static const StatEntries num = TInfos::NUM_LOCATIONS;
  static const StatEntries fileAccess = TInfos::LOC_FILE_ACCESS;
  static const StatEntries bufferSize = TInfos::LOC_BUFFER_SIZE;

  static constexpr EType error = EType::EVAL_LOC_TAPE_READ_FAILED;

  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static void setNum(TInfos &tapeInfos, size_t n) {
    tapeInfos.locBuffer_.numOnTape(n);
  }
  static size_t getNum(TInfos &tapeInfos) {
    return tapeInfos.locBuffer_.numOnTape();
  }
  static void setCurr(TInfos &tapeInfos, size_t loc) {
    tapeInfos.locBuffer_.position(loc);
  }

  static value_type *bufferBegin(TInfos &tapeInfos) {
    return tapeInfos.locBuffer_.begin();
  }

  static FILE *file(TInfos &tapeInfos) { return tapeInfos.locBuffer_.file(); }
  static void openFile(TInfos &tapeInfos, const char *fileName,
                       const char *mode = "rb") {
    tapeInfos.locBuffer_.openFile(fileName, mode);
  }
  static int removeFile(const char *fileName) { return remove(fileName); }
};

/**
 * @brief Adapter for the values tape (val tape). Fulfills InfoType.
 *
 * Maps generic operations to:
 *   - tapeInfos.valBuffer_
 */
template <class TInfos, class EType> struct ValInfo {
  using value_type = double;
  using StatEntries = typename TInfos::StatEntries;

  static const StatEntries num = TInfos::NUM_VALUES;
  static const StatEntries fileAccess = TInfos::VAL_FILE_ACCESS;

  static const StatEntries bufferSize = TInfos::VAL_BUFFER_SIZE;

  static constexpr EType error = EType::EVAL_VAL_TAPE_READ_FAILED;

  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static void setNum(TInfos &tapeInfos, size_t n) {
    tapeInfos.valBuffer_.numOnTape(n);
  }
  static size_t getNum(TInfos &tapeInfos) {
    return tapeInfos.valBuffer_.numOnTape();
  }
  static void setCurr(TInfos &tapeInfos, size_t loc) {
    tapeInfos.valBuffer_.position(loc);
  }

  static value_type *bufferBegin(TInfos &tapeInfos) {
    return tapeInfos.valBuffer_.begin();
  }

  static FILE *file(TInfos &tapeInfos) { return tapeInfos.valBuffer_.file(); }
  static void openFile(TInfos &tapeInfos, const char *fileName,
                       const char *mode = "rb") {
    tapeInfos.valBuffer_.openFile(fileName, mode);
  }
  static int removeFile(const char *fileName) { return remove(fileName); }
};

/**
 * @brief Adapter for the Taylor tape (tay tape). Fulfills InfoTypeBase.
 *
 * Maps generic operations to:
 *   - tapeInfos.tayBuffer_
 */
template <class TInfos, class EType> struct TayInfo {
  using value_type = double;
  using StatEntries = typename TInfos::StatEntries;

  static const StatEntries num = TInfos::NUM_TAYS;
  static const StatEntries bufferSize = TInfos::TAY_BUFFER_SIZE;

  static constexpr EType error = EType::TAPING_TAYLOR_OPEN_FAILED;

  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static void setNum(TInfos &tapeInfos, size_t n) {
    tapeInfos.tayBuffer_.numOnTape(n);
  }
  static size_t getNum(TInfos &tapeInfos) {
    return tapeInfos.tayBuffer_.numOnTape();
  }
  static void setCurr(TInfos &tapeInfos, size_t loc) {
    tapeInfos.tayBuffer_.position(loc);
  }

  static value_type *bufferBegin(TInfos &tapeInfos) {
    return tapeInfos.tayBuffer_.begin();
  }

  static FILE *file(TInfos &tapeInfos) { return tapeInfos.tayBuffer_.file(); }
  static void openFile(TInfos &tapeInfos, const char *fileName,
                       const char *mode = "rb") {
    tapeInfos.tayBuffer_.openFile(fileName, mode);
  }
  static int removeFile(const char *fileName) { return remove(fileName); }
};
}; // namespace ADOLC::detail

#endif // ADOLC_INFO_TYPE_H
