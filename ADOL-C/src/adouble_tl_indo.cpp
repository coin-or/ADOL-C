/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble_tl_indo.cpp
 Revision: $Id$
 Contents: adouble_tl.cpp contains that definitions of procedures used for sparse patterns.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adtl_indo.h>
#include <cmath>
#include <iostream>
#include <limits>
#include "dvlparms.h"

using std::cout;

extern "C" void adolc_exit(int errorcode, const char *what, const char* function, const char *file, int line);

namespace adtl_indo {

/*******************  i/o operations  ***************************************/
ostream& operator << ( ostream& out, const adouble& a) {
	out << a.val;
    return out;
}

istream& operator >> ( istream& in, adouble& a) {
	in >> a.val;
	return in;
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
