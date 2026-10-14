#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <utility>

#ifndef ADOLC_BUFFER_STATE
#define ADOLC_BUFFER_STATE

namespace ADOLC::detail {

/**
 * @brief Concept describing the primitive buffer/file interface used by tape
 *        I/O code.
 *
 * This is the contract that generic tape algorithms expect from a concrete
 * buffer wrapper such as BufferState<T>. Higher-level `Info` adapters should
 * select a buffer that satisfies this concept instead of re-expressing these
 * requirements themselves.
 *
 * @tparam T         Buffer wrapper type.
 * @tparam ElementType Element type stored in the buffer.
 */
template <class T, class ElementType>
concept BufferStateType =
    requires(T &buffer, const T &constBuffer, const char *fileName,
             const char *mode, size_t position, size_t numOnTape) {
      { buffer.begin() } -> std::same_as<ElementType *>;
      { constBuffer.begin() } -> std::same_as<const ElementType *>;
      { buffer.file() } -> std::same_as<FILE *>;
      { constBuffer.file() } -> std::same_as<FILE *>;
      { constBuffer.position() } -> std::convertible_to<size_t>;
      buffer.position(position);
      { constBuffer.numOnTape() } -> std::convertible_to<size_t>;
      buffer.numOnTape(numOnTape);
      buffer.openFile(fileName, mode);
      buffer.closeFile();
    };

/** @brief `fclose` deleter for tape files managed by `std::unique_ptr`. */
struct FileDeleter {
  int operator()(FILE *file) { return fclose(file); }
};

inline constexpr auto fileDeleter = FileDeleter{};

/// Selects a non-owning buffer view instead of a deep copy.
struct BufferViewTag {};
inline constexpr BufferViewTag bufferView;

/**
 * @brief Buffer and cursor state used by the ADOL-C tape implementation.
 *
 * BufferState groups the state that is common to the operation, value,
 * location, and Taylor tapes:
 *   - a FILE handle for disk-backed tape blocks,
 *   - an owned heap buffer or a non-owning view of another BufferState,
 *   - the current position inside that buffer,
 *   - the buffer capacity,
 *   - the number of elements currently recorded on the complete tape.
 *
 * The buffer behaves like a small pointer abstraction. position() is the
 * index of the current element. Writing usually stores into current() and then
 * advances the buffer; reading in reverse usually retreats first and then
 * consumes the returned element.
 *
 * A regular copy creates an independent, owning buffer. Constructing with
 * bufferView creates a non-owning view that shares the source's allocation but
 * has independent cursor and tape-count state. allocateAndOwn() replaces a
 * view with a new, uninitialized owned allocation when a tape block must be
 * loaded without modifying the viewed buffer.
 *
 * A view is deliberately lightweight and does not track its source. In
 * particular, BufferState does not invalidate existing views when the owner is
 * reset, moved, released, reallocated, or destroyed. The caller must guarantee
 * that the owner and its allocation outlive every view. Tape evaluations
 * provide this guarantee through the tape's data-access lock.
 *
 * Resource ownership:
 *   - owned buffers are deleted by the destructor,
 *   - viewed buffers are never deleted by the view,
 *   - FILE handles are closed by the unique_ptr deleter,
 *   - file handles are never shared when a buffer view is created,
 *   - releaseBuffer() is valid only for an owning BufferState,
 *   - releaseBuffer() and releaseFile() transfer ownership to the caller.
 *
 * @tparam T Element type stored by this tape buffer.
 */
template <typename T> class BufferState {
  using FilePtr = std::unique_ptr<FILE, FileDeleter>;
  FilePtr file_{nullptr, fileDeleter};
  T *buffer_{nullptr};
  size_t currentPos_{0};
  size_t capacity_{0};
  size_t numOnTape_{0};
  bool owner_{true};

  void copyBuffer(const BufferState &other) {
    T *newBuffer = nullptr;
    if (other.buffer_ != nullptr) {
      newBuffer = new T[other.capacity_];
      std::copy_n(other.buffer_, other.capacity_, newBuffer);
    }

    if (owner_) {
      delete[] buffer_;
    }
    buffer_ = newBuffer;
    currentPos_ = other.currentPos_;
    capacity_ = other.capacity_;
    numOnTape_ = other.numOnTape_;
    owner_ = true;
  }

  void moveData(BufferState &&other) {
    file_ = std::move(other.file_);
    other.file_ = nullptr;
    if (owner_) {
      delete[] buffer_;
    }
    buffer_ = other.buffer_;
    other.buffer_ = nullptr;
    capacity_ = other.capacity_;
    currentPos_ = other.currentPos_;
    numOnTape_ = other.numOnTape_;
    other.currentPos_ = 0;
    other.capacity_ = 0;
    other.numOnTape_ = 0;
    owner_ = std::exchange(other.owner_, false);
  }

public:
  /// Deletes the owned buffer and closes the owned file handle, if present.
  ~BufferState() {
    if (owner_) {
      delete[] buffer_;
    }
  }

  BufferState() = default;

  /// Takes ownership of an existing heap buffer with the given capacity.
  BufferState(T *buffer, size_t capacity)
      : buffer_(buffer), capacity_(capacity) {};

  /// Creates an independent owning copy. File handles are not copied.
  BufferState(const BufferState &other) { copyBuffer(other); }

  /**
   * @brief Creates a non-owning view of other's allocation.
   *
   * The cursor and tape counter are copied by value, while the buffer pointer
   * is shared. The view does not observe later cursor changes and does not keep
   * the source alive. The file handle is not shared.
   */
  BufferState(const BufferState &other, BufferViewTag)
      : buffer_(other.buffer_), currentPos_(other.currentPos_),
        capacity_(other.capacity_), numOnTape_(other.numOnTape_),
        owner_(false) {}

  BufferState &operator=(const BufferState &other) {
    if (this != &other) {
      copyBuffer(other);
      file_.reset();
    }
    return *this;
  }

  /// Moves the file handle, buffer, position, capacity, and tape counter.
  BufferState(BufferState &&other) noexcept { moveData(std::move(other)); }

  /// Releases current resources, then moves all resources from other.
  BufferState &operator=(BufferState &&other) noexcept {
    if (this != &other) {
      moveData(std::move(other));
    }
    return *this;
  }

  /// Returns whether this object owns and will delete its buffer allocation.
  bool isOwner() const { return owner_; }

  /**
   * @brief Replaces a view with an uninitialized owned allocation.
   *
   * Existing elements are intentionally not copied. This operation is used
   * immediately before loading a different tape block into the buffer.
   */
  void allocateAndOwn() {
    if (owner_) {
      return;
    }
    buffer_ = new T[capacity_];
    owner_ = true;
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
    if (owner_) {
      delete[] buffer_;
    }
    buffer_ = buffer;
    capacity_ = capacity;
    currentPos_ = 0;
    owner_ = true;
  }

  /// Returns the beginning of the owned or viewed buffer.
  T *begin() { return buffer_; }
  const T *begin() const { return buffer_; }

  /**
   * @brief Releases the owned buffer without deleting it.
   *
   * The capacity and current position are reset because the wrapper no longer
   * has a valid in-memory buffer. numOnTape() is left unchanged.
   */
  T *releaseBuffer() {
    assert(owner_ && "You can not release a borrowed buffer!");
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
      owner_ = true;
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

  /// Returns the capacity of the owned or viewed buffer.
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
