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
#include <cstdlib>

template <class T, size_t buff_size> class Buffer {
  using InitFunctionPointer = void (*)(T *subBufferElement);
  static void zeroAll(T *subBufferElement) { subBufferElement = nullptr; }

  struct SubBuffer {
    T elements[buff_size]{0};
    SubBuffer *nextSubBuffer{nullptr};
    SubBuffer() = default;
  };

public:
  Buffer() : initFunction(zeroAll) {}
  Buffer(InitFunctionPointer _initFunction) : initFunction(_initFunction) {}
  inline ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept
      : firstSubBuffer(other.firstSubBuffer), numEntries(other.numEntries),
        initFunction(other.initFunction) {
    // Leave other in a valid state
    other.firstSubBuffer = nullptr;
    other.numEntries = 0;
    other.initFunction = nullptr;
  }

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

template <class T, size_t buff_size>
Buffer<T, buff_size> &
Buffer<T, buff_size>::operator=(Buffer<T, buff_size> &&other) noexcept {
  if (this != &other) {
    // Free current resources
    SubBuffer *tmpSubBuffer = nullptr;
    while (firstSubBuffer != nullptr) {
      tmpSubBuffer = firstSubBuffer;
      firstSubBuffer = firstSubBuffer->nextSubBuffer;
      for (size_t i = 0; i < buff_size; i++)
        if (tmpSubBuffer->elements[i].allmem != nullptr)
          free(tmpSubBuffer->elements[i].allmem);
      delete tmpSubBuffer;
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

template <class T, size_t buff_size> Buffer<T, buff_size>::~Buffer() {
  SubBuffer *tmpSubBuffer = nullptr;

  while (firstSubBuffer != nullptr) {
    tmpSubBuffer = firstSubBuffer;
    firstSubBuffer = firstSubBuffer->nextSubBuffer;
    for (size_t i = 0; i < buff_size; i++)
      if (tmpSubBuffer->elements[i].allmem != nullptr)
        free(tmpSubBuffer->elements[i].allmem);
    delete tmpSubBuffer;
  }
}

template <class T, size_t buff_size> T *Buffer<T, buff_size>::append() {
  SubBuffer *currentSubBuffer = firstSubBuffer, *previousSubBuffer = nullptr;
  size_t index, tmp = numEntries;

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
  index = tmp;

  currentSubBuffer->elements[index].allmem = nullptr;
  if (initFunction != nullptr)
    initFunction(&(currentSubBuffer->elements[index]));

  currentSubBuffer->elements[index].index = numEntries;
  ++numEntries;

  return &currentSubBuffer->elements[index];
}

template <class T, size_t buff_size>
T *Buffer<T, buff_size>::getElement(size_t index) {
  SubBuffer *currentSubBuffer = firstSubBuffer;
  if (index >= numEntries)
    ADOLCError::fail(ADOLCError::ErrorType::BUFFER_INDEX_TO_LARGE,
                     CURRENT_LOCATION);
  while (index >= buff_size) {
    currentSubBuffer = currentSubBuffer->nextSubBuffer;
    index -= buff_size;
  }
  return &currentSubBuffer->elements[index];
}

#endif /* ADOLC_STRUCT_BUF_H */
