/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     storemanager.cpp
 Revision: $Id$
 Contents: Storage for locations

 Copyright (c) Andreas Kowarz, Andrea Walther, Kshitij Kulshreshtha,
               Benjamin Letschert, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/checkpointing_p.h>
#include <adolc/dvlparms.h>
#include <adolc/revolve.h>
#include <adolc/storemanager.h>
#include <adolc/tape_interface.h>
#include <cassert>
#include <cstring> // For memset
#include <iostream>
#include <limits>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <errno.h>

#if defined(ADOLC_TRACK_ACTIVITY)

char const *const StoreManagerLocint::nowhere = NULL;

StoreManagerLocint::StoreManagerLocint(double *&storePtr, char *&actStorePtr,
                                       size_t &size, size_t &numlives)
    : storePtr(storePtr), activityTracking(1), actStorePtr(actStorePtr),
      indexFree(0), head(0), maxsize(size), currentfill(numlives) {
#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::StoreManagerLocint()\n";
#endif
}

StoreManagerLocint::StoreManagerLocint(const StoreManagerLocint *const stm,
                                       double *&storePtr, char *&actStorePtr,
                                       size_t &size, size_t &numlives)
    : storePtr(storePtr), actStorePtr(actStorePtr), activityTracking(1),
      maxsize(size), currentfill(numlives) {
#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::StoreManagerLocint()\n";
#endif
  head = stm->head;
  indexFree = new size_t[maxsize];
  for (size_t i = 0; i < maxsize; i++)
    indexFree[i] = stm->indexFree[i];
}
#endif

StoreManagerLocint::StoreManagerLocint(double *&storePtr, size_t &size,
                                       size_t &numlives)
    : storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
      activityTracking(0), actStorePtr(const_cast<char *&>(nowhere)),
#endif
      indexFree(0), head(0), maxsize(size), currentfill(numlives) {
#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::StoreManagerLocint()\n";
#endif
}

StoreManagerLocint::~StoreManagerLocint() {
#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::~StoreManagerLocint()\n";
#endif
  if (storePtr) {
    delete[] storePtr;
    storePtr = 0;
  }
  if (indexFree) {
    delete[] indexFree;
    indexFree = 0;
  }
#if defined(ADOLC_TRACK_ACTIVITY)
  if (activityTracking && actStorePtr) {
    delete[] actStorePtr;
  }
#endif
  maxsize = 0;
  currentfill = 0;
  head = 0;
}

StoreManagerLocint::StoreManagerLocint(const StoreManagerLocint *const stm,
                                       double *&storePtr, size_t &size,
                                       size_t &numlives)
    : storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
      activityTracking(0), actStorePtr(const_cast<char *&>(nowhere)),
#endif
      maxsize(size), currentfill(numlives) {
#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::StoreManagerLocint()\n";
#endif
  head = stm->head;
  indexFree = new size_t[maxsize];
  for (size_t i = 0; i < maxsize; i++)
    indexFree[i] = stm->indexFree[i];
}

size_t StoreManagerLocint::next_loc() {
  if (head == 0) {
    grow();
  }
  assert(head);
  size_t const result = head;
  head = indexFree[head];
  ++currentfill;
#ifdef ADOLC_DEBUG
  std::cerr << "next_loc: " << result << " fill: " << size()
            << "max: " << maxSize() << endl;
#endif
  return result;
}

void StoreManagerLocint::free_loc(size_t loc) {
  assert(0 < loc && loc < maxsize);
  indexFree[loc] = head;
  head = loc;
  --currentfill;
#ifdef ADOLC_DEBUG
  std::cerr << "free_loc: " << loc << " fill: " << size()
            << "max: " << maxSize() << endl;
#endif
}

void StoreManagerLocint::ensure_block(size_t n) {
  ADOLCError::fail(ADOLCError::ErrorType::SM_LOCINT_BLOCK, CURRENT_LOCATION);
}
void StoreManagerLocint::grow(size_t mingrow) {
  if (maxsize == 0)
    maxsize += initialSize;
  size_t const oldMaxsize = maxsize;
  maxsize *= 2;
  if (maxsize < mingrow)
    maxsize = mingrow;

  if (maxsize > std::numeric_limits<size_t>::max())
    ADOLCError::fail(
        ADOLCError::ErrorType::SM_MAX_LIVES, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info5 = std::numeric_limits<size_t>::max()});

#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::grow(): increase size from " << oldMaxsize
            << " to " << maxsize << " entries (currently " << size()
            << " entries used)\n";
  assert(oldMaxsize == initialSize or size() == oldMaxsize);
#endif

  double *const oldStore = storePtr;
  size_t *const oldIndex = indexFree;
#if defined(ADOLC_TRACK_ACTIVITY)
  char *oldactStore;
  if (activityTracking) {
    oldactStore = actStorePtr;
  }
#endif

#if defined(ADOLC_DEBUG)
  std::cerr << "StoreManagerLocint::grow(): allocate "
            << maxsize * sizeof(double) << " B doubles "
            << "and " << maxsize * sizeof(size_t) << " B locints\n";
#endif
  storePtr = new double[maxsize];
  indexFree = new size_t[maxsize];
#if defined(ADOLC_TRACK_ACTIVITY)
  if (activityTracking)
    actStorePtr = new char[maxsize];
#endif
  // we use index 0 as end-of-list marker
  size_t i = 1;
  storePtr[0] = std::numeric_limits<double>::quiet_NaN();

  if (oldMaxsize != initialSize) { // not the first time
#if defined(ADOLC_DEBUG)
    std::cerr << "StoreManagerLocint::grow(): copy values\n";
#endif
    for (size_t j = i; j < oldMaxsize; ++j) {
      indexFree[j] = oldIndex[j];
    }
    for (size_t j = i; j < oldMaxsize; ++j) {
      storePtr[j] = oldStore[j];
    }
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking) {
      for (size_t j = i; j < oldMaxsize; ++j) {
        actStorePtr[j] = oldactStore[j];
      }
    }
#endif
    // reset i to start of new slots (upper half)
    i = oldMaxsize;

#if defined(ADOLC_DEBUG)
    std::cerr << "StoreManagerLocint::grow(): free "
              << oldMaxsize * sizeof(double) << " + "
              << oldMaxsize * sizeof(size_t) << " B\n";
#endif
    delete[] oldStore;
    delete[] oldIndex;
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking)
      delete[] oldactStore;
#endif
  }

  head = i;
  // create initial linked list for new slots
  for (; i < maxsize - 1; ++i) {
    indexFree[i] = i + 1;
  }
  indexFree[i] = 0; // end marker
  assert(i == maxsize - 1);
}

#if defined(ADOLC_TRACK_ACTIVITY)

char const *const StoreManagerLocintBlock::nowhere = NULL;

StoreManagerLocintBlock::StoreManagerLocintBlock(double *&storePtr,
                                                 char *&actStorePtr,
                                                 size_t &size, size_t &numlives)
    : storePtr(storePtr), actStorePtr(actStorePtr), activityTracking(1),
      maxsize(size), currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
      ,
      ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
{
  indexFree.clear();
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::StoreManagerLocIntBlock()\n";
#endif
}

StoreManagerLocintBlock::StoreManagerLocintBlock(
    const StoreManagerLocintBlock *const stm, double *&storePtr,
    char *&actStorePtr, size_t &size, size_t &numlives)
    : storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
      actStorePtr(actStorePtr), activityTracking(1),
#endif
      maxsize(size), currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
      ,
      ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
{
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::StoreManagerLocintBlock()\n";
#endif
  indexFree.clear();
  std::forward_list<struct FreeBlock>::const_iterator iter =
      stm->indexFree.begin();
  for (; iter != stm->indexFree.end(); iter++)
    indexFree.emplace_front(*iter);
}
#endif

StoreManagerLocintBlock::StoreManagerLocintBlock(double *&storePtr,
                                                 size_t &size, size_t &numlives)
    : storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
      activityTracking(0), actStorePtr(const_cast<char *&>(nowhere)),
#endif
      maxsize(size), currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
      ,
      ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
{
  indexFree.clear();
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::StoreManagerLocintBlock()\n";
#endif
}

StoreManagerLocintBlock::~StoreManagerLocintBlock() {
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::~StoreManagerLocintBlock()\n";
#endif
  if (storePtr != NULL) {
    delete[] storePtr;
    storePtr = NULL;
  }
  if (!indexFree.empty()) {
    indexFree.clear();
  }
#if defined(ADOLC_TRACK_ACTIVITY)
  if (activityTracking && actStorePtr) {
    delete[] actStorePtr;
  }
#endif
  maxsize = 0;
  currentfill = 0;
}

StoreManagerLocintBlock::StoreManagerLocintBlock(
    const StoreManagerLocintBlock *const stm, double *&storePtr, size_t &size,
    size_t &numlives)
    : storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
      activityTracking(0), actStorePtr(const_cast<char *&>(nowhere)),
#endif
      maxsize(size), currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
      ,
      ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
{
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::StoreManagerLocintBlock()\n";
#endif
  indexFree.clear();
  std::forward_list<struct FreeBlock>::const_iterator iter =
      stm->indexFree.begin();
  for (; iter != stm->indexFree.end(); iter++)
    indexFree.emplace_front(*iter);
}

size_t StoreManagerLocintBlock::next_loc() {
  if (indexFree.empty())
    grow();

  struct FreeBlock &front = indexFree.front();
  size_t const result = front.next;
  if (--front.size == 0) {
    if (next(indexFree.cbegin()) == indexFree.cend()) {
      front.next++;
      grow();
    } else
      indexFree.pop_front();
  } else
    front.next++;

  ++currentfill;
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::next_loc: result: " << result
            << " fill: " << size() << "max: " << maxSize() << endl;
  forward_list<struct FreeBlock>::iterator iter = indexFree.begin();
  for (; iter != indexFree.end(); iter++)
    std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")"
              << endl;
#endif
  return result;
}

void StoreManagerLocintBlock::ensure_block(size_t n) {
  bool found = false;
#ifdef ADOLC_LOCDEBUG
  ++ensure_blockCallsSinceLastConsolidateBlocks;
  std::cerr << "StoreManagerLocintBlock::ensure_block: required " << n
            << " ... ";
  std::cerr << "searching for big enough block " << endl;
#endif
  if (maxSize() - size() > n) {
    if (indexFree.front().size >= n)
      found = true;
    if ((!found) && ((double(maxSize()) / double(size())) > gcTriggerRatio() ||
                     maxSize() > gcTriggerMaxSize())) {
      consolidateBlocks();
#ifdef ADOLC_LOCDEBUG
      std::cerr << "ADOLC: GC called consolidateBlocks because " << maxSize()
                << "/" << size() << ">" << gcTriggerRatio() << " or "
                << maxSize() << ">" << gcTriggerMaxSize() << " after "
                << ensure_blockCallsSinceLastConsolidateBlocks << std::endl;
      ensure_blockCallsSinceLastConsolidateBlocks = 0;
#endif
      std::forward_list<struct FreeBlock>::iterator biter = indexFree
                                                                .before_begin(),
                                                    iter = indexFree.begin();
      for (; iter != indexFree.end(); biter++, iter++) {
        if (iter->size >= n) {
          if (iter != indexFree.begin()) {
            indexFree.emplace_front(*iter);
            indexFree.erase_after(biter);
          }
          found = true;
          break;
        }
      }
    }
  }
  if (!found) {
#ifdef ADOLC_LOCDEBUG
    std::cerr << "no big enough block...growing " << endl;
#endif
    grow(n);
  }

#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::ensure_Block: " << " fill: " << size()
            << "max: " << maxSize() << " ensure_Block (" << n << ")" << endl;
  forward_list<struct FreeBlock>::iterator iter = indexFree.begin();
  for (; iter != indexFree.end(); iter++)
    std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")"
              << endl;
#endif
}

void StoreManagerLocintBlock::grow(size_t minGrow) {
  // first figure out what eventual size we want
  size_t const oldMaxsize = maxsize;

  if (maxsize == 0) {
    maxsize = initialSize;
  } else {
    maxsize *= 2;
  }

  if (minGrow > 0) {
    while (maxsize - oldMaxsize < minGrow) {
      maxsize *= 2;
    }
  }

  if (maxsize > std::numeric_limits<size_t>::max())
    ADOLCError::fail(
        ADOLCError::ErrorType::SM_MAX_LIVES, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info5 = std::numeric_limits<size_t>::max()});

#ifdef ADOLC_LOCDEBUG
  // index 0 is not used, means one slot less
  std::cerr << "StoreManagerLocintBlock::grow(): increase size from "
            << oldMaxsize << " to " << maxsize << " entries (currently "
            << size() << " entries used)\n";
#endif

  double *const oldStore = storePtr;
#if defined(ADOLC_TRACK_ACTIVITY)
  char *oldactStore;
  if (activityTracking)
    oldactStore = actStorePtr;
#endif
#if defined(ADOLC_LOCDEBUG)
  std::cerr << "StoreManagerLocintBlock::grow(): allocate "
            << maxsize * sizeof(double) << " B doubles\n";
#endif
  storePtr = new double[maxsize];
  assert(storePtr);
  memset(storePtr, 0, maxsize * sizeof(double));
#if defined(ADOLC_TRACK_ACTIVITY)
  if (activityTracking) {
    actStorePtr = new char[maxsize];
    memset(actStorePtr, 0, maxsize * sizeof(char));
  }
#endif

  if (oldStore != NULL) { // not the first time
#if defined(ADOLC_LOCDEBUG)
    std::cerr << "StoreManagerLocintBlock::grow(): copy values\n";
#endif

    memcpy(storePtr, oldStore, oldMaxsize * sizeof(double));
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking) {
      memcpy(actStorePtr, oldactStore, oldMaxsize * sizeof(char));
    }
#endif

#if defined(ADOLC_LOCDEBUG)
    std::cerr << "StoreManagerLocintBlock::grow(): free "
              << oldMaxsize * sizeof(double) << "\n";
#endif
    delete[] oldStore;
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking) {
      delete[] oldactStore;
    }
#endif
  }

  bool foundTail = false;
  std::forward_list<struct FreeBlock>::iterator biter =
                                                    indexFree.before_begin(),
                                                iter = indexFree.begin();
  for (; iter != indexFree.end(); biter++, iter++) {
    if (iter->next + iter->size == oldMaxsize) {
      iter->size += (maxsize - oldMaxsize);
      indexFree.emplace_front(*iter);
      indexFree.erase_after(biter);
      foundTail = true;
      break;
    }
  }

  if (!foundTail) {
    indexFree.emplace_front(
#if defined(_MSC_VER) && _MSC_VER <= 1800
        FreeBlock(
#endif
            oldMaxsize, (maxsize - oldMaxsize)
#if defined(_MSC_VER) && _MSC_VER <= 1800
                )
#endif
    );
  }

  biter = indexFree.before_begin();
  iter = indexFree.begin();
  while (iter != indexFree.end()) {
    if (iter->size == 0) {
      indexFree.erase_after(biter); // don't leave 0 blocks around
      iter = next(biter);
    } else {
      biter++;
      iter++;
    }
  }
#ifdef ADOLC_LOCDEBUG
  std::cerr << "Growing:" << endl;
  iter = indexFree.begin();
  for (; iter != indexFree.end(); iter++)
    std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")"
              << endl;
#endif
}

void StoreManagerLocintBlock::free_loc(size_t loc) {
  assert(loc < maxsize);

  struct FreeBlock &front = indexFree.front();
  if ((loc + 1 == front.next) || (front.next + front.size == loc)) {
    front.size++;
    if (loc + 1 == front.next)
      front.next = loc;
  } else {
    indexFree.emplace_front(
#if defined(_MSC_VER) && _MSC_VER <= 1800
        FreeBlock(
#endif
            loc, 1
#if defined(_MSC_VER) && _MSC_VER <= 1800
            )
#endif
    );
  }

  --currentfill;
#ifdef ADOLC_LOCDEBUG
  std::cerr << "free_loc: " << loc << " fill: " << size()
            << "max: " << maxSize() << endl;
  forward_list<struct FreeBlock>::iterator iter = indexFree.begin();
  for (; iter != indexFree.end(); iter++)
    std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")"
              << endl;
#endif
}

void StoreManagerLocintBlock::consolidateBlocks() {
  indexFree.sort();
  std::forward_list<struct FreeBlock>::iterator iter = indexFree.begin(),
                                                niter = iter++;
  while (iter != indexFree.end()) {
    if (niter->next + niter->size == iter->next) {
      niter->size += iter->size;
      indexFree.erase_after(niter);
      iter = next(niter);
    } else {
      niter++;
      iter++;
    }
  }
#ifdef ADOLC_LOCDEBUG
  std::cerr << "StoreManagerLocintBlock::consolidateBlocks: " << " fill: "
            << size() << "max: " << maxSize() << endl;
  iter = indexFree.begin();
  for (; iter != indexFree.end(); iter++)
    std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")"
              << endl;
#endif
}
