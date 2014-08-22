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

#include <adolc/adouble.h>
#include "oplate.h"
#include "taping_p.h"
#include "dvlparms.h"

using std::vector;

adubref::adubref( locint lo, locint ref ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    location = lo;
    refloc = (size_t)trunc(fabs(ADOLC_GLOBAL_TAPE_VARS.store[location]));
    if (ref != refloc) {
	fprintf(DIAG_OUT,"ADOL-C error: strange construction of an active"
		" vector subscript reference\n(passed ref = %d, stored refloc = %d)\n",ref,refloc);
	adolc_exit(-2,"",__func__,__FILE__,__LINE__);
    }
}

adubref::operator adub() const {
    locint locat = next_loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(locat,location);
        put_op(ref_copyout);
        ADOLC_PUT_LOCINT(location); // = arg
        ADOLC_PUT_LOCINT(locat);    // = res
        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat]=ADOLC_GLOBAL_TAPE_VARS.store[refloc];
    return locat;
}

adub adubref::operator++( int ) {
    locint locat = next_loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(locat,location);
        put_op(ref_copyout);
        ADOLC_PUT_LOCINT(location); // = arg
        ADOLC_PUT_LOCINT(locat);    // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat]=ADOLC_GLOBAL_TAPE_VARS.store[refloc];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(incr_a,location);
        put_op(ref_incr_a);
        ADOLC_PUT_LOCINT(location); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc]++;
    return locat;
}

adub adubref::operator--( int ) {
    locint locat = next_loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(locat,location);
        put_op(ref_copyout);
        ADOLC_PUT_LOCINT(location); // = arg
        ADOLC_PUT_LOCINT(locat);    // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat]=ADOLC_GLOBAL_TAPE_VARS.store[refloc];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(incr_a,location);
        put_op(ref_decr_a);
        ADOLC_PUT_LOCINT(location); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc]--;
    return locat;
}

adubref& adubref::operator++() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(incr_a,location);
        put_op(ref_incr_a);
        ADOLC_PUT_LOCINT(location); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc]++;
    return *this;
}

adubref& adubref::operator--() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(incr_a,location);
        put_op(ref_decr_a);
        ADOLC_PUT_LOCINT(location); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc]--;
    return *this;
}

adubref& adubref::operator = ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        if (coval == 0) {
            put_op(ref_assign_d_zero);
            ADOLC_PUT_LOCINT(location);   // = res
        } else
            if (coval == 1.0) {
                put_op(ref_assign_d_one);
                ADOLC_PUT_LOCINT(location); // = res
            } else {
                put_op(ref_assign_d);
                ADOLC_PUT_LOCINT(location); // = res
                ADOLC_PUT_VAL(coval);       // = coval
            }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] = coval;
    return *this;
}

adubref& adubref::operator = ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint x_loc = x.loc();
    if (location!=x_loc)
        /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
    { if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old:  write_assign_a(location,x.location);
            put_op(ref_assign_a);
            ADOLC_PUT_LOCINT(x_loc);    // = arg
            ADOLC_PUT_LOCINT(location);   // = res

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
        }

        ADOLC_GLOBAL_TAPE_VARS.store[refloc]=ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
    }
    return *this;
}

adubref& adubref::operator = ( const adubref& x ) {
    *this = adub(x);
    return *this;
}

adubref& adubref::operator <<= ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { 
        ADOLC_CURRENT_TAPE_INFOS.numInds++;

        put_op(ref_assign_ind);
        ADOLC_PUT_LOCINT(location); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] = coval;
    return *this;
}

void adubref::declareIndependent() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        ADOLC_CURRENT_TAPE_INFOS.numInds++;

        put_op(ref_assign_ind);
        ADOLC_PUT_LOCINT(location); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
    }
}

adubref& adubref::operator >>= (double& coval) {
    adub(*this) >>= coval;
    return *this;
}

void adubref::declareDependent() {
    adub(*this).declareDependent();
}

adubref& adubref::operator += ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_d_same_arg(eq_plus_d,location,coval);
        put_op(ref_eq_plus_d);
        ADOLC_PUT_LOCINT(location); // = res
        ADOLC_PUT_VAL(coval);       // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] += coval;
    return *this;
}

adubref& adubref::operator += ( const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint y_loc = y.loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_a_same_arg(eq_plus_a,location,y.location);
        put_op(ref_eq_plus_a);
        ADOLC_PUT_LOCINT(y_loc); // = arg
        ADOLC_PUT_LOCINT(location);   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] += ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
    return *this;
}

adubref& adubref::operator -= ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_d_same_arg(eq_min_d,location,coval);
        put_op(ref_eq_min_d);
        ADOLC_PUT_LOCINT(location); // = res
        ADOLC_PUT_VAL(coval);       // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] -= coval;
    return *this;
}

adubref& adubref::operator -= ( const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint y_loc = y.loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_a_same_arg(eq_min_a,location,y.location);
        put_op(ref_eq_min_a);
        ADOLC_PUT_LOCINT(y_loc); // = arg
        ADOLC_PUT_LOCINT(location);   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] -= ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
    return *this;
}

adubref& adubref::operator *= ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_d_same_arg(eq_mult_d,location,coval);
        put_op(ref_eq_mult_d);
        ADOLC_PUT_LOCINT(location); // = res
        ADOLC_PUT_VAL(coval);       // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] *= coval;
    return *this;
}

adubref& adubref::operator *= ( const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint y_loc = y.loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_a_same_arg(eq_mult_a,location,y.location);
        put_op(ref_eq_mult_a);
        ADOLC_PUT_LOCINT(y_loc); // = arg
        ADOLC_PUT_LOCINT(location);   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[refloc]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[refloc] *= ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
    return *this;
}

void condassign( adubref res,         const badouble &cond,
                 const badouble &arg1, const badouble &arg2 ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_condassign(res.location,cond.location,arg1.location,
        //		     arg2.location);
        put_op(ref_cond_assign);
        ADOLC_PUT_LOCINT(cond.loc()); // = arg
        ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()]);
        ADOLC_PUT_LOCINT(arg1.loc()); // = arg1
        ADOLC_PUT_LOCINT(arg2.loc()); // = arg2
        ADOLC_PUT_LOCINT(res.location);  // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.refloc]);
    }

    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
        ADOLC_GLOBAL_TAPE_VARS.store[res.refloc] = ADOLC_GLOBAL_TAPE_VARS.store[arg1.loc()];
    else
        ADOLC_GLOBAL_TAPE_VARS.store[res.refloc] = ADOLC_GLOBAL_TAPE_VARS.store[arg2.loc()];
}

void condassign( adubref res, const badouble &cond, const badouble &arg ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_condassign2(res.location,cond.location,arg.location);
        put_op(ref_cond_assign_s);
        ADOLC_PUT_LOCINT(cond.loc()); // = arg
        ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()]);
        ADOLC_PUT_LOCINT(arg.loc());  // = arg1
        ADOLC_PUT_LOCINT(res.location);  // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.refloc]);
    }

    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
        ADOLC_GLOBAL_TAPE_VARS.store[res.refloc] = ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()];
}

advector::blocker::blocker(size_t n) {
    dflt = new adouble;
    ensureContiguousLocations(n);
}

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
    size_t idx = (size_t)trunc(fabs(ADOLC_GLOBAL_TAPE_VARS.store[index.loc()]));
    locint locat = next_loc();
    size_t n = data.size();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(subscript);
	ADOLC_PUT_LOCINT(index.loc());
	ADOLC_PUT_LOCINT(locat);
	ADOLC_PUT_VAL(n);
	ADOLC_PUT_LOCINT(data[0].loc());

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) 
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }

    if (idx >= n)
	fprintf(DIAG_OUT, "ADOL-C warning: index out of bounds while subscripting n=%zu, idx=%zu\n", n, idx);

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = ADOLC_GLOBAL_TAPE_VARS.store[data[idx].loc()];
    return locat;
}

adubref advector::operator[](const badouble& index) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    size_t idx = (size_t) trunc(fabs(ADOLC_GLOBAL_TAPE_VARS.store[index.loc()]));
    locint locat = next_loc();
    size_t n = data.size();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(subscript_ref);
	ADOLC_PUT_LOCINT(index.loc());
	ADOLC_PUT_LOCINT(locat);
	ADOLC_PUT_VAL(n);
	ADOLC_PUT_LOCINT(data[0].loc());

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) 
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }

    if (idx >= n)
	fprintf(DIAG_OUT, "ADOL-C warning: index out of bounds while subscripting (ref) n=%zu, idx=%zu\n", n, idx);

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = data[idx].loc();
    return adubref(locat,data[idx].loc());
}

adouble advector::lookupindex(const badouble& x, const badouble& y) const {
    if (!nondecreasing()) {
	fprintf(DIAG_OUT, "ADOL-C error: can only call lookup index if advector ist nondecreasing\n");
	adolc_exit(-2,"",__func__,__FILE__,__LINE__);
    }
    if (y.value() < 0) {
	fprintf(DIAG_OUT, "ADOL-C error: index lookup needs a nonnegative denominator\n");
	adolc_exit(-2,"",__func__,__FILE__,__LINE__);
    }
    adouble r = 0;
    size_t n = data.size();
    for (size_t i = 0; i < n; i++) 
	condassign(r, x - data[i]*y, (adouble) (i+1));
    return r;
}
