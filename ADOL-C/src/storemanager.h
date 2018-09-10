// -*- c++ -*- hello emacs...
/*---------------------------------------------------------------------------- 
 ADOL-C--  Automatic Differentiation by Overloading in C++ - simplified
 File:     storemanager.h
 Revision: $Id$
 Contents: storemanager.h contains definitions of abstract interface 
           class StoreManager and some derived classes implementing the
           desired functionality.

 Copyright (c) 2006 Johannes Willkomm <johannes.willkomm@rwth-aachen.de>
               2011-2013 Kshitij Kulshreshtha <kshitij@math.upb.de>
               2012 Benjamin Letschert <letschi@mail.upb.de>
               2013 Jean Utke <utke@mcs.anl.gov>

 This file is part of ADOL-C.

 The classes StoreManagerXYZ basically takes the global double *store pointer 
 into their obhut and implement next_loc and free_loc.

 They basic idea is taken from "The C++ Programming Language" by Bjarne
 Stroustrup, from the chapter 19 on iterators and allocators.

 To understand how and why they work do the following:
 1) Have a look at StoreManagerInSitu and convince yourself that it is
    exactly the same as the solution presented in the Stroustrup book,
    except that we always have just one big array instead of a linked
    list of chunks.

    This means in particular that we have to copy the values from the
    old array into the lower half of the new one (we always double the size).

 2) Have a look at StoreManagerLocint and convince yourself that these do
    the same as StoreManagerInSitu except that the linked list of free 
    slots is maintained in a completely different portion of memory. This
    means the values in freed slots remain untouched until they are
    allocated again.


 3) Have a look a class StoreManagerLocintBlock. This class uses a list of
    of free blocks of different sizes instead of free locations.

 class StoreManagerInSitu
 An unsafe implementation is provided as well, but commented out.
 It does not use the indexFeld array which saves between 
 25% and 50% memory relative to the above safe implementation.

 It is most closely modelled after the example found in the 
 Stroustrup book.

 It appears that it works very well, if one does not use the
 trace_on(tag, 1); ... trace_off(); reverse(); way of using ADOL-C.
 If the first sweep is forward it works fine.
 Therefore I left it in here as a comment so an interested user
 with acute main memory scarcity may give it a try.
	   

 History:
          20120427 bl:     add blocking store management
          20110208 kk:     incorporated in ADOL-C; moved some code arround
          20060507 jw:     begin

----------------------------------------------------------------------------*/

#ifndef ADOL_C__STOREMANAGER_H
#define ADOL_C__STOREMANAGER_H

#if defined(ADOLC_INTERNAL)
#    if HAVE_CONFIG_H
#        include "config.h"
#    endif
#endif
#include <adolc/internal/adolc_settings.h>
#include <forward_list>

#if USE_BOOST_POOL 
#include <boost/pool/pool_alloc.hpp>
#endif

#include <adolc/internal/common.h>
#include <adolc/taping.h>

class GlobalTapeVarsCL;
extern "C" void checkInitialStoreSize(GlobalTapeVarsCL* gtv);

class StoreManager {
  friend void checkInitialStoreSize(GlobalTapeVarsCL* gtv);
protected:
  static size_t const initialSize = 4;
  double myGcTriggerRatio;
  size_t myGcTriggerMaxSize;
  virtual void grow(size_t mingrow = 0) = 0;
public:
  StoreManager() : myGcTriggerRatio(1.5), myGcTriggerMaxSize(initialSize) {}
  virtual ~StoreManager() {}
  virtual locint next_loc() = 0;
  virtual void free_loc(locint) = 0;
  virtual void ensure_block(size_t n) = 0;
  void setStoreManagerControl(double gcTriggerRatio, size_t gcTriggerMaxSize) { myGcTriggerRatio=gcTriggerRatio; myGcTriggerMaxSize=gcTriggerMaxSize;}
  double gcTriggerRatio() const {return myGcTriggerRatio;}
  size_t gcTriggerMaxSize() const {return myGcTriggerMaxSize;}
//   // effectively the current size of the store array
  virtual size_t maxSize() const = 0;

//   // the number of slots currently in use
  virtual size_t size() const = 0;
  virtual unsigned char storeType() const = 0;
};

class StoreManagerLocint : public StoreManager {
protected:
  double * &storePtr;
#if defined(ADOLC_TRACK_ACTIVITY)
  char activityTracking;
  static char const* const nowhere;
  char * &actStorePtr;
#endif
  locint * indexFree;
  locint head;
  size_t &maxsize;
  size_t &currentfill;
  virtual void grow(size_t mingrow = 0);
public:

#if defined(ADOLC_TRACK_ACTIVITY)
  StoreManagerLocint(double * &storePtr, char* &actStorePtr, size_t &size, size_t &numlives);
  StoreManagerLocint(const StoreManagerLocint *const stm, double * &storePtr, char* &actStorePtr, size_t &size, size_t &numLives);
#endif
  StoreManagerLocint(double * &storePtr, size_t &size, size_t &numlives);
  StoreManagerLocint(const StoreManagerLocint *const stm, double * &storePtr, size_t &size, size_t &numLives);

  virtual ~StoreManagerLocint();
  virtual inline size_t size() const { return currentfill; }

  virtual inline size_t maxSize() const { return maxsize; }
  virtual inline unsigned char storeType() const { return ADOLC_LOCATION_SINGLETONS; }

  virtual inline bool realloc_on_next_loc() const { 
      return (head == 0);
  }

  virtual locint next_loc();
  virtual void free_loc(locint loc); 
  virtual void ensure_block(size_t n);
};

class StoreManagerLocintBlock : public StoreManager {
protected:
    double * &storePtr;
#if defined(ADOLC_TRACK_ACTIVITY)
    char activityTracking;
    static char const* const nowhere;
    char * &actStorePtr;
#endif
    struct FreeBlock {
	locint next; // next location
	size_t size; // number of following free locations
	FreeBlock(): next(0), size(0) {}
	FreeBlock(const struct FreeBlock &block) :
	    next(block.next),size(block.size) {}
        FreeBlock(const locint& n, const size_t& s) :
            next(n), size(s) {}
	bool operator<(const struct FreeBlock& b) const {
	    return (next < b.next);
	}
    };

    std::forward_list<struct FreeBlock
#if USE_BOOST_POOL
                      , boost::fast_pool_allocator<struct FreeBlock> 
#endif 
                      >  indexFree;
    size_t &maxsize;
    size_t &currentfill;

    void consolidateBlocks();
#ifdef ADOLC_LOCDEBUG
    unsigned int ensure_blockCallsSinceLastConsolidateBlocks;
#endif
    /**
     * when minGrow is specified we asssume that we have already
     * search the blocks and found no block with minGrow locations in it
     */
    virtual void grow(size_t minGrow=0 );
public:
#if defined(ADOLC_TRACK_ACTIVITY)
    StoreManagerLocintBlock(double * &storePtr, char* &actStorePtr, size_t &size, size_t &numlives);
    StoreManagerLocintBlock(const StoreManagerLocintBlock *const stm, double * &storePtr, char* &actStorePtr, size_t &size, size_t &numLives);
#endif
    StoreManagerLocintBlock(double * &storePtr, size_t &size, size_t &numlives);
    StoreManagerLocintBlock(const StoreManagerLocintBlock *const stm, double * &storePtr, size_t &size, size_t &numLives);

    virtual ~StoreManagerLocintBlock();
    virtual inline size_t size() const { return currentfill; }

    virtual inline size_t maxSize() const { return maxsize; }
    virtual inline unsigned char storeType() const { return ADOLC_LOCATION_BLOCKS; }

    virtual locint next_loc();
    virtual void free_loc(locint loc);
    virtual void ensure_block(size_t n);
};

#if 0
/* This implementation is unsafe in that using tace_on with keep=1 and 
   reverse mode directly afterwards will yield incorrect results.
   For all other purposes it seem to work just fine, so it's left here
   for reference as a comment.
*/

/* unsafe - use with care */

class StoreManagerInSitu : public StoreManager {
  //  static size_t const initialeGroesse = 512;
protected:
  double * &storePtr;
  struct Link {
    struct Link *next;
  };
  Link *head;
  size_t groesse;
  size_t anzahl;
public:
  size_t maxIndexUsed;

  StoreManager(double * &storePtr) :
    storePtr(storePtr),
    head(0),
    groesse(initialeGroesse),
    anzahl(0),
    maxIndexUsed(0)
  {
    // while a place in store is unused we want to place
    // a Link stucture (i.e. a pointer) there
    assert(sizeof(double) >= sizeof(void*));
    assert(sizeof(double) >= sizeof(Link));
    std::cerr << "StoreManager::StoreManager()\n";
  }

  virtual ~StoreManager() {
    if (storePtr) {
      delete [] storePtr;
      storePtr = 0;
    }
    std::cerr << "StoreManager::~StoreManager()\n";
  }

  virtual inline size_t size() const { return anzahl; }

  virtual inline size_t maxSize() const { return groesse; }

  virtual locint next_loc(size_t n = 1) {
    assert(n == 1);
    if (head == 0) {
      grow();
    }
    assert(head);
    double * const dPtr = reinterpret_cast<double*>(head);
    head = head->next;
    ++anzahl;
    locint const result = dPtr - storePtr;
    maxIndexUsed = std::max((locint)maxIndexUsed, result);
    return result;
  }

  virtual void free_loc(locint loc) {
    assert(loc < groesse);
    Link *returned = reinterpret_cast<Link*>(storePtr + loc);
    returned->next = head;
    head = returned;
    --anzahl;
  }
private:
  void grow() {
    size_t const alteGroesse = groesse;
    groesse *= 2;
    assert(alteGroesse == initialeGroesse or size() == alteGroesse);
    std::cerr << "StoreManager::grow(): increase size to " << groesse << "\n";
    double *const oldStore = storePtr;
    std::cerr << "StoreManager::grow(): allocate " << groesse * sizeof(double) << " B\n";
    storePtr = new double[groesse];
    size_t i = 0;
    if (alteGroesse != initialeGroesse) { // nicht beim ersten Mal
      std::cerr << "StoreManager::grow(): copy values\n";
      for ( ; i < alteGroesse; ++i) {
        storePtr[i] = oldStore[i];
      }
      std::cerr << "StoreManager::grow(): free " << alteGroesse * sizeof(double) << " B\n";
      delete [] oldStore;
    }
    head = reinterpret_cast<Link*>(storePtr + i);
    for ( ; i < groesse-1; ++i) {
      reinterpret_cast<Link*>(storePtr + i)->next
        = reinterpret_cast<Link*>(storePtr + i + 1);
    }
    reinterpret_cast<Link*>(storePtr + i)->next = 0;
  }

};
#endif /* 0 */

#endif /* ADOL_C__STOREMANAGER_H */

