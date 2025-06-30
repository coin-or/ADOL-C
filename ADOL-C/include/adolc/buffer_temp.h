/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     struct_buf.h
 Revision: $Id$
 Contents: - template class for linked list of Type buffers with constant length
             per buffer
           - intended to be used with structs

 Copyright (c) Andreas Kowarz, Kshitij Kulshreshtha, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_STRUCT_BUF_H)
#define ADOLC_STRUCT_BUF_H 1

#include <adolc/internal/common.h>
#include <array>
#include <cstdlib>

// check whether the type has an allmem storage
template <typename T>
concept AllMemType = requires(T t) {
  { t.allmem } -> std::convertible_to<void *>;
};

template <AllMemType T, size_t buff_size> class Buffer {
  using InitFunctionPointer = void (*)(T *subBufferElement);
  static void zeroAll(T *subBufferElement) { subBufferElement = nullptr; }

  struct SubBuffer {
    std::array<T, buff_size> elements;
    SubBuffer *nextSubBuffer{nullptr};
    SubBuffer() = default;
  };

public:
  Buffer() : initFunction(zeroAll) {}
  Buffer(InitFunctionPointer _initFunction) : initFunction(_initFunction) {}
  inline ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept = default;

  Buffer &operator=(Buffer &&other) noexcept;
  inline void init(InitFunctionPointer _initFunction) {
    initFunction = _initFunction;
  }
  T *append();
  T *getElement(size_t index);

private:
  SubBuffer *firstSubBuffer{nullptr};
  InitFunctionPointer initFunction{nullptr};
  size_t numEntries{0};
};

template <AllMemType T, size_t buff_size>
Buffer<T, buff_size> &
Buffer<T, buff_size>::operator=(Buffer<T, buff_size> &&other) noexcept {
  if (this != &other) {
    // Free current resources
    SubBuffer *next = nullptr;
    while (firstSubBuffer) {
      next = firstSubBuffer->nextSubBuffer;
      for (auto &ele : firstSubBuffer->elements)
        if (ele.allmem)
          free(ele.allmem);
      delete firstSubBuffer;
      firstSubBuffer = next;
    }

    // transfer resources
    firstSubBuffer = other.firstSubBuffer;
    numEntries = other.numEntries;
    initFunction = other.initFunction;

    // Leave other in a valid state
    other.firstSubBuffer = nullptr;
    other.numEntries = 0;
    other.initFunction = nullptr;
  }
  return *this;
}

template <AllMemType T, size_t buff_size> Buffer<T, buff_size>::~Buffer() {
  SubBuffer *next = nullptr;

  // clean-up allmem and subBuffers
  while (firstSubBuffer) {
    next = firstSubBuffer->nextSubBuffer;
    for (auto &ele : firstSubBuffer->elements)
      if (ele.allmem)
        free(ele.allmem);
    delete firstSubBuffer;
    firstSubBuffer = next;
  }
}

template <AllMemType T, size_t buff_size> T *Buffer<T, buff_size>::append() {
  SubBuffer *currentSubBuffer = firstSubBuffer, *previousSubBuffer = nullptr;
  size_t tmp = numEntries;

  while (tmp >= buff_size) {
    previousSubBuffer = currentSubBuffer;
    currentSubBuffer = currentSubBuffer->nextSubBuffer;
    tmp -= buff_size;
  }
  if (currentSubBuffer == nullptr) {
    currentSubBuffer = new SubBuffer;
    if (firstSubBuffer == nullptr)
      firstSubBuffer = currentSubBuffer;
    else
      previousSubBuffer->nextSubBuffer = currentSubBuffer;
    currentSubBuffer->nextSubBuffer = nullptr;
  }
  const size_t index = tmp;

  currentSubBuffer->elements[index].allmem = nullptr;
  if (initFunction != nullptr)
    initFunction(&(currentSubBuffer->elements[index]));

  currentSubBuffer->elements[index].index = numEntries;
  ++numEntries;

  return &currentSubBuffer->elements[index];
}

template <AllMemType T, size_t buff_size>
T *Buffer<T, buff_size>::getElement(size_t index) {
  if (index >= numEntries)
    ADOLCError::fail(ADOLCError::ErrorType::BUFFER_INDEX_TO_LARGE,
                     CURRENT_LOCATION);

  SubBuffer *currentSubBuffer = firstSubBuffer;
  while (index >= buff_size) {
    currentSubBuffer = currentSubBuffer->nextSubBuffer;
    index -= buff_size;
  }
  return &currentSubBuffer->elements[index];
}

#endif /* ADOLC_STRUCT_BUF_H */
