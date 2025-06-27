/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adouble_tl.cpp contains that definitions of procedures used to
           define various tapeless adouble operations.
           These operations actually have two purposes.
           The first purpose is to actual compute the function, just as
           the same code written for double precision (single precision -
           complex - interval) arithmetic would.  The second purpose is
           to compute directional derivatives in forward mode of
           automatic differentiation.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adolcerror.h>
#include <adolc/adtl.h>
#include <adolc/dvlparms.h>
#include <cmath>
#include <iostream>
#include <limits>

using std::cout;

namespace adtl {

#ifdef USE_ADTL_REFCOUNTING
size_t refcounter::refcnt = 0;
#endif

#if USE_BOOST_POOL
boost::pool<boost::default_user_allocator_new_delete> *adouble::advalpool =
    new boost::pool<boost::default_user_allocator_new_delete>(
        (adouble::numDir + 1) * sizeof(double), 32, 10000);
#endif

/*******************  i/o operations  ***************************************/
ostream &operator<<(ostream &out, const adouble &a) {
  out << "Value: " << a.PRIMAL_VALUE;
  out << " ADValues (" << adouble::numDir << "): ";
  FOR_I_EQ_1_LTEQ_NUMDIR
  out << a.ADVAL_I << " ";
  out << "(a)";
  return out;
}

istream &operator>>(istream &in, adouble &a) {
  char c;
  do
    in >> c;
  while (c != ':' && !in.eof());
  in >> a.PRIMAL_VALUE;
  unsigned int num;
  do
    in >> c;
  while (c != '(' && !in.eof());
  in >> num;
  if (num > adouble::numDir)
    ADOLCError::fail(ADOLCError::ErrorType::TO_MANY_DIRECTIONS,
                     CURRENT_LOCATION);

  do
    in >> c;
  while (c != ':' && !in.eof());
  FOR_I_EQ_1_LTEQ_NUMDIR
  in >> a.ADVAL_I;
  do
    in >> c;
  while (c != ')' && !in.eof());
  return in;
}

} // namespace adtl
