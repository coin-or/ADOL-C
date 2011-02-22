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
          20110208 kk:     incorporated in ADOL-C; moved some code arround
          20060507 jw:     begin

----------------------------------------------------------------------------*/

#ifndef ADOL_C__STOREMANAGER_H
#define ADOL_C__STOREMANAGER_H

#include <adolc/common.h>

class StoreManager {
protected:
  static size_t const initialeGroesse = 4;
public:
  virtual ~StoreManager() {}

  virtual locint next_loc() = 0;
  virtual void free_loc(locint) = 0;

//   // effectively the current size of the store array
  virtual size_t maxSize() const = 0;

//   // the number of slots currently in use
  virtual size_t size() const = 0;
};



class StoreManagerLocint : public StoreManager {
protected:
  double * &storePtr;
  locint * indexFeld;
  locint head;
  size_t &groesse;
  size_t &anzahl;
private:
  void grow();
public:

  StoreManagerLocint(double * &storePtr, size_t &size, size_t &numlives);

  virtual ~StoreManagerLocint();
  virtual inline size_t size() const { return anzahl; }

  virtual inline size_t maxSize() const { return groesse; }

  virtual inline bool realloc_on_next_loc() const { 
      return (head == 0);
  }

  virtual locint next_loc();
  virtual void free_loc(locint loc); 
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

