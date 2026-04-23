#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>

#ifndef ADOLC_BUFFER_STATE
#define ADOLC_BUFFER_STATE

namespace ADOLC::detail {

struct FileDeleter {
  int operator()(FILE *file) { return fclose(file); }
};
/**
 * @brief Owning buffer wrapper used by the ADOL-C tape implementation.
 *
 * BufferState groups the state that is common to the operation, value,
 * location, and Taylor tapes:
 *   - a FILE handle for disk-backed tape blocks,
 *   - an owned heap buffer,
 *   - the current position inside that buffer,
 *   - the buffer capacity,
 *   - the number of elements currently recorded on the complete tape.
 *
 * The buffer behaves like a small pointer abstraction. position() is the
 * index of the current element. Writing usually stores into current() and then
 * advances the buffer; reading in reverse usually retreats first and then
 * consumes the returned element.
 *
 * Ownership:
 *   - allocated buffers are deleted by the destructor,
 *   - FILE handles are closed by the unique_ptr deleter,
 *   - releaseBuffer() and releaseFile() transfer ownership back to the caller.
 *
 * @tparam T Element type stored by this tape buffer.
 */
template <typename T> class BufferState {
  using FilePtr = std::unique_ptr<FILE, FileDeleter>;
  FilePtr file_{nullptr, FileDeleter{}};
  T *buffer_{nullptr};
  size_t currentPos_{0};
  size_t capacity_{0};
  size_t numOnTape_{0};

public:
  /// Deletes the owned buffer and closes the owned file handle, if present.
  ~BufferState() { delete[] buffer_; }

  BufferState() = default;

  /// Takes ownership of an existing heap buffer with the given capacity.
  BufferState(T *buffer, size_t capacity)
      : buffer_(buffer), capacity_(capacity) {};

  /// BufferState owns resources and is therefore move-only.
  BufferState(const BufferState &other) = delete;
  BufferState &operator=(const BufferState &other) = delete;

  /// Moves the file handle, buffer, position, capacity, and tape counter.
  BufferState(BufferState &&other) noexcept {
    file_ = std::move(other.file_);
    buffer_ = other.buffer_;
    other.buffer_ = nullptr;
    capacity_ = other.capacity_;
    currentPos_ = other.currentPos_;
    numOnTape_ = other.numOnTape_;
    other.currentPos_ = 0;
    other.capacity_ = 0;
    other.numOnTape_ = 0;
  }

  /// Releases current resources, then moves all resources from other.
  BufferState &operator=(BufferState &&other) noexcept {
    if (this != &other) {
      file_ = std::move(other.file_);
      other.file_ = nullptr;
      delete[] buffer_;
      buffer_ = other.buffer_;
      other.buffer_ = nullptr;
      capacity_ = other.capacity_;
      currentPos_ = other.currentPos_;
      numOnTape_ = other.numOnTape_;
      other.currentPos_ = 0;
      other.capacity_ = 0;
      other.numOnTape_ = 0;
    }
    return *this;
  }

  /// Takes ownership of file, closing any previously owned file handle.
  void resetFile(FILE *file) { file_.reset(file); }

  /// Returns the owned FILE handle without transferring ownership.
  FILE *file() { return file_.get(); }
  FILE *file() const { return file_.get(); }

  /// Closes the owned FILE handle, if present.
  void closeFile() { file_.reset(); }

  /// Releases the FILE handle without closing it.
  FILE *releaseFile() { return file_.release(); }

  /// Opens fileName with mode and owns the resulting FILE handle.
  void openFile(const char *fileName, const char *mode) {
    file_.reset(fopen(fileName, mode));
  }

  /**
   * @brief Replaces the buffer pointer.
   *
   * The new pointer is treated as owned by BufferState.
   */
  void resetBuffer(T *buffer, size_t capacity) {
    delete[] buffer_;
    buffer_ = buffer;
    capacity_ = capacity;
    currentPos_ = 0;
  }

  /// Returns the beginning of the owned buffer.
  T *begin() { return buffer_; }
  const T *begin() const { return buffer_; }

  /**
   * @brief Releases the owned buffer without deleting it.
   *
   * The capacity and current position are reset because the wrapper no longer
   * has a valid in-memory buffer. numOnTape() is left unchanged.
   */
  T *releaseBuffer() {
    T *buffer = buffer_;
    buffer_ = nullptr;
    capacity_ = 0;
    currentPos_ = 0;
    return buffer;
  }

  /// Allocates a new owned buffer if no buffer is currently present.
  void allocIfNull(size_t capacity) {
    if (buffer_ == nullptr) {
      buffer_ = new T[capacity];
      capacity_ = capacity;
    }
  }

  /// Returns the current buffer index.
  size_t position() const { return currentPos_; }

  /// Sets the current buffer index.
  void position(size_t pos) {
    assert(pos <= capacity_);
    currentPos_ = pos;
  }

  /// Returns how many elements fit between position() and capacity().
  size_t remainingCapacity() const { return capacity_ - currentPos_; }

  /// Returns the capacity of the owned buffer.
  size_t capacity() const { return capacity_; }

  /// Returns the number of elements recorded on the tape.
  size_t numOnTape() const { return numOnTape_; }

  /// Sets the number of elements recorded on the tape.
  void numOnTape(size_t num) { numOnTape_ = num; }

  /// Returns a copy of the element at idx.
  const T &operator[](size_t idx) const {
    assert(idx < capacity_);
    return buffer_[idx];
  }

  /// Returns a mutable reference to the element at idx.
  T &operator[](size_t idx) {
    assert(idx < capacity_);
    return buffer_[idx];
  }

  /// Stores val at position().
  void writeCurrent(T val) {
    assert(currentPos_ < capacity_);
    buffer_[currentPos_] = val;
  }

  /// Stores val at position(), then advances to the next element.
  void writeAndAdvance(T val) {
    writeCurrent(val);
    advance();
  }

  /// Returns a pointer to the element at position().
  T *current() { return buffer_ + currentPos_; }
  const T *current() const { return buffer_ + currentPos_; }

  /// Advances the current position by one element.
  void advance() {
    assert(currentPos_ < capacity_);
    ++currentPos_;
  }

  /// Moves the current position back by one element.
  void retreat() {
    assert(currentPos_ > 0);
    --currentPos_;
  }

  /// Returns the current element, then advances position().
  T readAndAdvance() { return (*this)[currentPos_++]; }

  /// Moves to the previous element, then returns it.
  T retreatAndRead() { return (*this)[--currentPos_]; }
};

/// Operation tape buffer.
using OpBuffer = BufferState<unsigned char>;
using ValBuffer = BufferState<double>;
using LocBuffer = BufferState<size_t>;
using TayBuffer = BufferState<double>;
}; // namespace ADOLC::detail
#endif // ADOLC_BUFFER_STATE
