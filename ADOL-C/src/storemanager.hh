// -*- c++ -*- hello emacs...
/*---------------------------------------------------------------------------- 
 ADOL-C--  Automatic Differentiation by Overloading in C++ - simplified
 File:     storemanager.h
 Revision: $Id$
 Contents: storemanager.h contains definitions of abstract interface 
           class StoreManager and some derived classes implementing the
           desired functionality.

 Copyright (c) 2006 Johannes Willkomm <johannes.willkomm@rwth-aachen.de>
 Written by Johannes Willkomm <johannes.willkomm@rwth-aachen.de>
  
 This file is part of ADOL-C--, an unofficial modification of ADOL-C 1.10.1 
 done by Johannes Willkomm. 
 
 As the original ADOLC-C, this software is provided under the terms of
 the Common Public License. Any use, reproduction, or distribution of the
 software constitutes recipient's acceptance of the terms of this license.
 See the accompanying copy of the Common Public License for more details.


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

 2) Have a look at StoreManagerInteger and StoreManagerLink and convince
    yourself that these do the same as StoreManagerInSitu except that
    the linked list of free slots is maintained in a completely different
    portion of memory. This means the values in freed slots remain untouched
    until they are allocated again.



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
          20060507 jw:     begin

----------------------------------------------------------------------------*/

#ifndef ADOL_C__STOREMANAGER_H
#define ADOL_C__STOREMANAGER_H

#include <limits>
#include <iostream>
#include <assert.h>

#include "common.h"


// #ifdef JW_USE_STOREMANAGER

//#ifdef ADOLC_DEBUG
#define JW_DEBUG_STOREMANAGER
//#endif


struct StoreManager {
  static size_t const initialeGroesse = 4;

//   virtual ~StoreManager() {}

//   virtual locint next_loc() = 0;
//   virtual void free_loc(locint) = 0;

//   // effectively the current size of the store array
//   virtual size_t maxSize() const = 0;

// #ifdef JW_DEBUG_STOREMANAGER
//   // the number of slots currently in use
//   virtual size_t size() const = 0;
// #endif

};



class StoreManagerInteger : public StoreManager {
  double * &storePtr;
  locint * indexFeld;
  locint head;
  size_t &groesse;
#ifdef JW_DEBUG_STOREMANAGER
  size_t anzahl;
#endif
public:

  StoreManagerInteger(double * &storePtr, size_t &size) : 
    storePtr(storePtr),
    indexFeld(0),
    head(0),
    groesse(size)
#ifdef JW_DEBUG_STOREMANAGER
    , anzahl(0)
#endif
  {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
    groesse = initialeGroesse;
  }

  ~StoreManagerInteger() {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerInteger::~StoreManagerInteger()\n";
#endif
    if (storePtr) {
      delete[] storePtr;
      storePtr = 0;
    }
    if (indexFeld) {
      delete[] indexFeld;
      indexFeld = 0;
    }
    groesse = 0;
#ifdef JW_DEBUG_STOREMANAGER
    anzahl = 0;
#endif
    head = 0;
  }

#ifdef JW_DEBUG_STOREMANAGER
  size_t size() const { return anzahl; }
#endif

  size_t maxSize() const { return groesse; }

  bool realloc_on_next_loc() const { 
    return head == 0;
  }

  locint next_loc() {
    if (head == 0) {
      grow();
    }
#ifdef JW_DEBUG_STOREMANAGER
    assert(head);
#endif
    locint const result = head;
    head = indexFeld[head];
#ifdef JW_DEBUG_STOREMANAGER
    ++anzahl;
#ifdef ADOLC_HARDDEBUG
    std::cerr << "next_loc: " << result << " fill: " << size() << endl;
#endif
#endif
    return result;
  }

  void free_loc(locint loc) {
#ifdef JW_DEBUG_STOREMANAGER
    assert(loc < groesse);
#endif
    indexFeld[loc] = head;
    head = loc;
#ifdef JW_DEBUG_STOREMANAGER
    --anzahl;
#ifdef ADOLC_HARDDEBUG
    std::cerr << "free_loc: " << loc << " fill: " << size() << endl;
#endif
#endif
  }

private:
  void grow() {
    size_t const alteGroesse = groesse;
    groesse *= 2;

    if (groesse > std::numeric_limits<locint>::max()) {
      // encapsulate this error message
      fprintf(DIAG_OUT,"\nADOL-C error:\n");
      fprintf(DIAG_OUT,"maximal number (%d) of live active variables exceeded\n\n", 
	      std::numeric_limits<locint>::max());
      exit(-3);
    }

#ifdef JW_DEBUG_STOREMANAGER
    // index 0 is not used, means one slot less
    std::cerr << "StoreManagerInteger::grow(): increase size from " << alteGroesse 
	 << " to " << groesse << " entries (currently " << size() << " entries used)\n";
    assert(alteGroesse == initialeGroesse or size() == (alteGroesse-1));
#endif

    double *const oldStore = storePtr;
    locint *const oldIndex = indexFeld;

#ifdef JW_DEBUG_STOREMANAGER
#if defined(ADOLC_HARDDEBUG)
    std::cerr << "StoreManagerInteger::grow(): allocate " << groesse * sizeof(double) << " B doubles " 
	 << "and " << groesse * sizeof(locint) << " B locints\n";
#endif
#endif
    storePtr = new double[groesse];
    indexFeld = new locint[groesse];

    // we use index 0 as end-of-list marker
    size_t i = 1;
    //     storePtr[0] = nan(""); not available on solaris
    storePtr[0] = (non_num/non_den);

    if (alteGroesse != initialeGroesse) { // not the first time
#ifdef JW_DEBUG_STOREMANAGER
#if defined(ADOLC_HARDDEBUG)
      std::cerr << "StoreManagerInteger::grow(): copy values\n";
#endif
#endif
      for (size_t j = i; j < alteGroesse; ++j) {
	indexFeld[j] = oldIndex[j];
      }
      for (size_t j = i; j < alteGroesse; ++j) {
	storePtr[j] = oldStore[j];
      }

      // reset i to start of new slots (upper half)
      i = alteGroesse;

#ifdef JW_DEBUG_STOREMANAGER
#if defined(ADOLC_HARDDEBUG)
      std::cerr << "StoreManagerInteger::grow(): free " << alteGroesse * sizeof(double)
		<< " + " << alteGroesse * sizeof(locint) << " B\n";
#endif
#endif
      delete [] oldStore;
      delete [] oldIndex;
    }

    head = i;
    // create initial linked list for new slots
    for ( ; i < groesse-1; ++i) {
      indexFeld[i] = i + 1;
    }
    indexFeld[i] = 0; // end marker

#ifdef JW_DEBUG_STOREMANAGER
    assert(i == groesse-1);
#endif

  }

};



class StoreManagerLink : public StoreManager {
public:
  struct Link {
    Link *next;
  };

private:
  double * &storePtr;
  Link * linkFeld;
  Link * head;
  size_t groesse;
#ifdef JW_DEBUG_STOREMANAGER
  size_t anzahl;
#endif

public:

  StoreManagerLink(double * &storePtr) : 
    storePtr(storePtr),
    linkFeld(0),
    head(0),
    groesse(initialeGroesse)
#ifdef JW_DEBUG_STOREMANAGER
    , anzahl(0)
#endif
  {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerLink::StoreManagerLink()\n";
#endif
    if (sizeof(Link) > sizeof(Link*)) {
      std::cerr << "ADOLC: Warning: StoreManagerLink::StoreManagerLink(): die Lösung ist nicht optimal, "
		<< "da sizeof(Link) > sizeof(Link*)\n";
      std::cerr << "ADOLC: This will use sizeof(Link)/sizeof(Link*) times more memory.\n";
      std::cerr << "ADOLC: Depending of what sizeof(locint) is (" << sizeof(locint) << " right now),"
	" consider switching to StoreManagerInteger by hand in file adouble.cpp\n";
    }
  }

  ~StoreManagerLink() {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerLink::~StoreManagerLink()\n";
#endif
    if (storePtr) {
      delete[] storePtr;
      storePtr = 0;
    }
    if (linkFeld) {
      delete[] linkFeld;
      linkFeld = 0;
    }
    groesse = 0;
#ifdef JW_DEBUG_STOREMANAGER
    anzahl = 0;
#endif
    head = 0;
  }

  bool realloc_on_next_loc() const { 
    return head == 0;
  }

#ifdef JW_DEBUG_STOREMANAGER
  size_t size() const { return anzahl; }
#endif

  size_t maxSize() const { return groesse; }

  locint next_loc() {
    if (head == 0) {
      grow();
    }
#ifdef JW_DEBUG_STOREMANAGER
    assert(head);
#endif
    Link *link = head;
    head = head->next;
    locint const result = link - linkFeld;
#ifdef JW_DEBUG_STOREMANAGER
    ++anzahl;
#ifdef ADOLC_HARDDEBUG
    std::cerr << "next_loc: " << result << " fill: " << size() << endl;
#endif
#endif
    return result;
  }

  void free_loc(locint loc) {
#ifdef JW_DEBUG_STOREMANAGER
    assert(loc < groesse);
#endif
    Link *returned = linkFeld + loc;
    returned->next = head;
    head = returned;
#ifdef JW_DEBUG_STOREMANAGER
    --anzahl;
#ifdef ADOLC_HARDDEBUG
    std::cerr << "free_loc: " << loc << " fill: " << size() << endl;
#endif
#endif
  }

private:
  void grow() {
    size_t const alteGroesse = groesse;
    groesse *= 2;

    if (groesse > std::numeric_limits<locint>::max()) {
      // encapsulate this error message
      fprintf(DIAG_OUT,"\nADOL-C error:\n");
      fprintf(DIAG_OUT,"maximal number (%d) of live active variables exceeded\n\n", 
	      std::numeric_limits<locint>::max());
      exit(-3);
    }

#ifdef JW_DEBUG_STOREMANAGER
    std::cerr << "StoreManagerLink::grow(): increase size from " << alteGroesse 
	 << " to " << groesse << " entries (currently " << size() << " entries used)\n";
    assert(alteGroesse == initialeGroesse or size() == alteGroesse);
#endif

    double *const oldStore = storePtr;
    Link *const oldIndex = linkFeld;

#ifdef JW_DEBUG_STOREMANAGER
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerLink::grow(): allocate " << groesse * sizeof(double) << " B doubles " 
	 << "and " << groesse * sizeof(Link) << " B Links\n";
#endif
#endif
    storePtr = new double[groesse];
    linkFeld = new Link[groesse];

    size_t i = 0;

    if (alteGroesse != initialeGroesse) { // not the first time
#ifdef JW_DEBUG_STOREMANAGER
#ifdef ADOLC_HARDDEBUG
      std::cerr << "StoreManagerLink::grow(): copy values\n";
#endif
#endif
      for (size_t j = i; j < alteGroesse; ++j) {
	linkFeld[j] = oldIndex[j];
      }
      for (size_t j = i; j < alteGroesse; ++j) {
	storePtr[j] = oldStore[j];
      }

      // reset i to start of new slots (upper half)
      i = alteGroesse;

#ifdef JW_DEBUG_STOREMANAGER
#ifdef ADOLC_HARDDEBUG
      std::cerr << "StoreManagerLink::grow(): free " << alteGroesse * sizeof(double) 
		<< " + " << alteGroesse * sizeof(locint) << " B\n";
#endif
#endif
      delete [] oldStore;
      delete [] oldIndex;
    }

    head = linkFeld + i;
    // create initial linked list for new slots
    for ( ; i < groesse-1; ++i) {
      linkFeld[i].next = linkFeld + i + 1;
    }
    linkFeld[i].next = 0; // mark end

#ifdef JW_DEBUG_STOREMANAGER
    assert(i == groesse-1);
#endif

  }

};

template <bool SizeofLocintLTSizeofPointer> class StoreManagerExSituImpl;

// specialize for SizeofLocintLTSizeofPointer == true
// using StoreManagerInteger will save sizeof(locint)/sizeof(void*) of the memory for 
// the linked list array
template <> struct StoreManagerExSituImpl<true> : 
public StoreManagerInteger
{
  StoreManagerExSituImpl(double * &storePtr, size_t &sz) : StoreManagerInteger(storePtr, sz) {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerExSitu::StoreManagerExSitu(): using linked list based on locint indices\n";
#endif
  }
};

// specialize for SizeofLocintLTSizeofPointer == false
template <> struct StoreManagerExSituImpl<false> : 
public StoreManagerLink
{
  StoreManagerExSituImpl(double * &storePtr) : StoreManagerLink(storePtr) {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerExSitu::StoreManagerExSitu(): using native pointer linked list\n";
#endif
  }
};

template <class LocintType, class PointerType> struct StoreManagerExSitu : 
  public StoreManagerExSituImpl<sizeof(LocintType) < sizeof(PointerType)>
{
  typedef StoreManagerExSituImpl<sizeof(LocintType) < sizeof(PointerType)> Base;
  StoreManagerExSitu(double * &storePtr) : Base(storePtr) {
#ifdef ADOLC_HARDDEBUG
    std::cerr << "StoreManagerExSitu::StoreManagerExSitu(): select implementation: "
	      << "sizeof(locint) = " << sizeof(LocintType) << ", sizeof(Link) = " << sizeof(PointerType)
	      << " (and sizeof(void*) = " << sizeof(void*) << ")\n";
#endif
  }
};


/* This implementation is unsafe in that using tace_on with keep=1 and 
   reverse mode directly afterwards will yield incorrect results.
   For all other purposes it seem to work just fine, so it's left here
   for reference as a comment.
*/

/* unsafe - use with care 

class StoreManagerInSitu : public StoreManager {
  //  static size_t const initialeGroesse = 512;
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

  ~StoreManager() {
    if (storePtr) {
      delete [] storePtr;
      storePtr = 0;
    }
    std::cerr << "StoreManager::~StoreManager()\n";
  }

  size_t size() const { return anzahl; }

  size_t maxSize() const { return groesse; }

  locint next_loc(size_t n = 1) {
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

  void free_loc(locint loc) {
    assert(loc < groesse);
    Link *returned = reinterpret_cast<Link*>(storePtr + loc);
    returned->next = head;
    head = returned;
    --anzahl;
  }

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

*/

// #endif

#endif

