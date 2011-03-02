/* ---------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++

 Revision: $Id$
 Contents: advector.cpp contains a vector<adouble> implementation
           that is able to trace subscripting operations.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#include <limits>
#include <cmath>

#include <adolc/advector.h>
#include "oplate.h"
#include "taping_p.h"

using std::vector;

bool advector::nondecreasing() const {
    bool ret = true;
    double last = - ADOLC_MATH_NSP::numeric_limits<double>::infinity();
    vector<adouble>::const_iterator iter = data.begin();
    for ( ; iter != data.end() && ret ; iter++) {
	ret = ret && ( iter->value() >= last );
	last = iter->value();
    }
    return ret;
}

adub advector::operator[](const badouble& index) const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    size_t idx = (size_t) trunc(fabs(ADOLC_GLOBAL_TAPE_VARS.store[index.location]));
    locint locat = next_loc();
    size_t n = data.size();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(subscript);
	ADOLC_PUT_LOCINT(index.location);
	ADOLC_PUT_LOCINT(locat);
	ADOLC_PUT_VAL(n);
	for (int i = 0; i < n; i++) 
	    ADOLC_PUT_LOCINT(data[i].location);

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) 
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = ADOLC_GLOBAL_TAPE_VARS.store[data[idx].location];
    return locat;
}

adouble advector::lookupindex(const badouble& x, const badouble& y) const {
    if (!nondecreasing()) {
	fprintf(DIAG_OUT, "ADOL-C error: can only call lookup index if advector ist nondecreasing\n");
	exit(-2);
    }
    if (y.value() < 0) {
	fprintf(DIAG_OUT, "ADOL-C error: index lookup needs a nonnegative denominator\n");
	exit(-2);
    }
    adouble r = 0;
    size_t n = data.size();
    for (int i = 0; i < n; i++) 
	condassign(r, x - data[i]*y, (adouble) (i+1));
    return r;
}
