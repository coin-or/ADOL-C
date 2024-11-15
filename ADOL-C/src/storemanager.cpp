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
#include "checkpointing_p.h"
#include "dvlparms.h"
#include "taping_p.h"
#include <adolc/adalloc.h>
#include <adolc/revolve.h>

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
  indexFree = new locint[maxsize];
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
  indexFree = new locint[maxsize];
  for (size_t i = 0; i < maxsize; i++)
    indexFree[i] = stm->indexFree[i];
}

locint StoreManagerLocint::next_loc() {
  if (head == 0) {
    grow();
  }
  assert(head);
  locint const result = head;
  head = indexFree[head];
  ++currentfill;
#ifdef ADOLC_DEBUG
  std::cerr << "next_loc: " << result << " fill: " << size()
            << "max: " << maxSize() << endl;
#endif
  return result;
}

void StoreManagerLocint::free_loc(locint loc) {
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
  fprintf(
      DIAG_OUT,
      "ADOL-C error: Location block required from singleton location store");
  adolc_exit(-4, "ADOL-C error: Location blocks not allowed", __func__,
             __FILE__, __LINE__);
}

void StoreManagerLocint::grow(size_t mingrow) {
  if (maxsize == 0)
    maxsize += initialSize;
  size_t const oldMaxsize = maxsize;
  maxsize *= 2;
  if (maxsize < mingrow)
    maxsize = mingrow;

  if (maxsize > std::numeric_limits<locint>::max()) {
    // encapsulate this error message
    fprintf(DIAG_OUT, "\nADOL-C error:\n");
    fprintf(DIAG_OUT,
            "maximal number (%d) of live active variables exceeded\n\n",
            std::numeric_limits<locint>::max());
    adolc_exit(-3, "", __func__, __FILE__, __LINE__);
  }

#ifdef ADOLC_DEBUG
  std::cerr << "StoreManagerLocint::grow(): increase size from " << oldMaxsize
            << " to " << maxsize << " entries (currently " << size()
            << " entries used)\n";
  assert(oldMaxsize == initialSize or size() == oldMaxsize);
#endif

  double *const oldStore = storePtr;
  locint *const oldIndex = indexFree;
#if defined(ADOLC_TRACK_ACTIVITY)
  char *oldactStore;
  if (activityTracking) {
    oldactStore = actStorePtr;
  }
#endif

#if defined(ADOLC_DEBUG)
  std::cerr << "StoreManagerLocint::grow(): allocate "
            << maxsize * sizeof(double) << " B doubles "
            << "and " << maxsize * sizeof(locint) << " B locints\n";
#endif
  storePtr = new double[maxsize];
  indexFree = new locint[maxsize];
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
              << oldMaxsize * sizeof(locint) << " B\n";
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

/****************************************************************************/
/* Returns the next free location in "adouble" memory.                      */
/****************************************************************************/
locint next_loc() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  return ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->next_loc();
}

/****************************************************************************/
/* frees the specified location in "adouble" memory                         */
/****************************************************************************/
void free_loc(locint loc) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->free_loc(loc);
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

locint StoreManagerLocintBlock::next_loc() {
  if (indexFree.empty())
    grow();

  struct FreeBlock &front = indexFree.front();
  locint const result = front.next;
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

  if (maxsize > std::numeric_limits<locint>::max()) {
    // encapsulate this error message
    fprintf(DIAG_OUT, "\nADOL-C error:\n");
    fprintf(DIAG_OUT,
            "maximal number (%u) of live active variables exceeded\n\n",
            std::numeric_limits<locint>::max());
    adolc_exit(-3, "", __func__, __FILE__, __LINE__);
  }

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

void StoreManagerLocintBlock::free_loc(locint loc) {
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

void ensureContiguousLocations(size_t n) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->ensure_block(n);
}

void setStoreManagerControl(double gcTriggerRatio, size_t gcTriggerMaxSize) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->setStoreManagerControl(
      gcTriggerRatio, gcTriggerMaxSize);
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

void setStoreManagerType(unsigned char type) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->storeType() != type) {
    if (ADOLC_GLOBAL_TAPE_VARS.numLives == 0) {
      ADOLC_GLOBAL_TAPE_VARS.reallocStore(type);
    } else {
      fprintf(
          DIAG_OUT,
          "ADOL-C-warning: called %s after allocating %ld active variables\n"
          "***  WILL NOT CHANGE ***\nto change type deallocate all active "
          "variables\n"
          "continuing ...\n",
          __func__, ADOLC_GLOBAL_TAPE_VARS.numLives);
    }
  } else {
    fprintf(DIAG_OUT,
            "ADOL-C-warning: called %s with same type as before\n"
            "***  NO CHANGE ***\ncontinuing ...\n",
            __func__);
  }
}
