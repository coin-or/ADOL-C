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

#include <adolc/adtl.h>
#include <cmath>
#include <iostream>
#include <limits>

using std::cout;

namespace adtl {


size_t adouble::numDir = 1;
enum Mode adouble::forward_mode = ADTL_FOV;

size_t refcounter::refcnt = 0;


/*******************  i/o operations  ***************************************/
ostream& operator << ( ostream& out, const adouble& a) {
    if (likely(adouble::_do_val() && adouble::_do_adval())) {
	out << "Value: " << a.val;
	out << " ADValues (" << adouble::numDir << "): ";
	FOR_I_EQ_0_LT_NUMDIR
	    out << a.ADVAL_I << " ";
	out << "(a)";
    }
    return out;
}

istream& operator >> ( istream& in, adouble& a) {
    if(likely(adouble::_do_val() && adouble::_do_adval())) {
	char c;
	do in >> c;
	while (c!=':' && !in.eof());
	in >> a.val;
	unsigned int num;
	do in >> c;
	while (c!='(' && !in.eof());
	in >> num;
	if (num>adouble::numDir)
	{
	    cout << "ADOL-C error: to many directions in input\n";
	    exit(-1);
	}
	do in >> c;
	while (c!=':' && !in.eof());
	FOR_I_EQ_0_LT_NUMDIR
	    in >> a.ADVAL_I;
	do in >> c;
	while (c!=')' && !in.eof());
	return in;
    }
}

/**************** ADOLC_TRACELESS_SPARSE_PATTERN ****************************/
int ADOLC_Init_sparse_pattern(adouble *a, int n, unsigned int start_cnt) {
    for(unsigned int i=0; i < n; i++) {
	a[i].delete_pattern();
	a[i].pattern.push_back( i+start_cnt );
    }
    return 3;
}

int ADOLC_get_sparse_pattern(const adouble *const b, int m, unsigned int **&pat) {
    pat = (unsigned int**) malloc(m*sizeof(unsigned int*));
    for( int i=0; i < m ; i++){
	//const_cast<adouble&>(b[i]).pattern.sort();
	//const_cast<adouble&>(b[i]).pattern.unique();
      if ( b[i].get_pattern_size() > 0 ) {
         pat[i] = (unsigned int*) malloc(sizeof(unsigned int) * (b[i].get_pattern_size() +1) );
         pat[i][0] = b[i].get_pattern_size();
         const list<unsigned int>& tmp_set = b[i].get_pattern();
         list<unsigned int>::const_iterator it;
         unsigned int l=1;
         for(it = tmp_set.begin() ; it != tmp_set.end() ; it++,l++)
             pat[i][l] = *it;
       } else {
          pat[i] = (unsigned int*) malloc(sizeof(unsigned int));
          pat[i][0] =0;
       }
    }
    return 3;
}

}
