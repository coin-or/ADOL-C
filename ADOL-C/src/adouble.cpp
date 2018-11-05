/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: adouble.C contains that definitions of procedures used to 
           define various badouble, adub, and adouble operations. 
           These operations actually have two purposes.
           The first purpose is to actual compute the function, just as 
           the same code written for double precision (single precision -
           complex - interval) arithmetic would.  The second purpose is 
           to write a transcript of the computation for the reverse pass 
           of automatic differentiation.
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Kshitij Kulshreshtha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
   
----------------------------------------------------------------------------*/

#include "taping_p.h"
#include <adolc/adouble.h>
#include "oplate.h"
#include "dvlparms.h"

using namespace std;

/****************************************************************************/
/*                                                        HELPFUL FUNCTIONS */

/*--------------------------------------------------------------------------*/
void condassign( double &res, const double &cond,
                 const double &arg1, const double &arg2 ) {
    res = cond > 0 ? arg1 : arg2;
}

/*--------------------------------------------------------------------------*/
void condassign( double &res, const double &cond,
                 const double &arg) {
    res = cond > 0 ? arg : res;
}

/*--------------------------------------------------------------------------*/
void condeqassign( double &res, const double &cond,
                   const double &arg1, const double &arg2 ) {
    res = cond >= 0 ? arg1 : arg2;
}

/*--------------------------------------------------------------------------*/
void condeqassign( double &res, const double &cond,
                   const double &arg) {
    res = cond >= 0 ? arg : res;
}
/*--------------------------------------------------------------------------*/
/* The remaining routines define the badouble, adub and adouble routines.   */
/*--------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                             CONSTRUCTORS */

/*--------------------------------------------------------------------------*/
#if defined(ADOLC_ADOUBLE_LATEINIT)
void adouble::initInternal(void) {
  if (isInit)
    return;
  location = next_loc();
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

#if defined(ADOLC_ADOUBLE_STDCZERO)
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
#endif
	put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(location);   // = res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[location] = 0.;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[location] = false;
#endif
#endif
  isInit = true;
}
#else
void adouble::initInternal(void) {}
#endif

/*--------------------------------------------------------------------------*/
adouble::adouble() {
    location = next_loc();
    isInit = true;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

#if defined(ADOLC_ADOUBLE_STDCZERO)
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
#endif
	put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(location);   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    
    ADOLC_GLOBAL_TAPE_VARS.store[location] = 0.;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[location] = false;
#endif
#endif
}

/*--------------------------------------------------------------------------*/
adouble::adouble( double coval ) {
    location = next_loc();
    isInit = true;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { 
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
#endif
        if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(location);   // = res
        } else
            if (coval == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(location); // = res
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(location); // = res
                ADOLC_PUT_VAL(coval);       // = coval
            }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[location] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[location] = false;
#endif
}

/*--------------------------------------------------------------------------*/
adouble::adouble( const adouble& a ) {
    location = next_loc();
    isInit = true;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;


    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { 
#if defined(ADOLC_TRACK_ACTIVITY)
     if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.location]) {
#endif
        put_op(assign_a);
        ADOLC_PUT_LOCINT(a.location);   // = arg
        ADOLC_PUT_LOCINT(location);     // = res

       ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
       if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	   ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);

#if defined(ADOLC_TRACK_ACTIVITY)
     } else {
	 if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
	     double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.location];
	     if (coval == 0) {
		 put_op(assign_d_zero);
		 ADOLC_PUT_LOCINT(location);   // = res
	     } else if (coval == 1.0) {
		 put_op(assign_d_one);
		 ADOLC_PUT_LOCINT(location); // = res
	     } else {
		 put_op(assign_d);
		 ADOLC_PUT_LOCINT(location); // = res
		 ADOLC_PUT_VAL(coval);       // = coval
	     }

	     ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	     if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		 ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[location]);

	 }
     }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[location] = ADOLC_GLOBAL_TAPE_VARS.store[a.location];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[location] = ADOLC_GLOBAL_TAPE_VARS.actStore[a.location];
#endif
}

/*--------------------------------------------------------------------------*/
adouble::adouble( const adub& a ) {
    location = next_loc();
    isInit = true;
    locint a_loc = a.loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    int upd = 0;
    /* 981020 olvo  skip upd_resloc(..) if no tracing performed */
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
        upd = upd_resloc_check(a_loc,loc());
    if (upd) { /* olvo 980708 new n2l & 980921 changed interface */
        free_loc(location);
        location = a_loc;
        const_cast<adub&>(a).isInit = false;
    } else {
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(loc(),a_loc);
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc]) {
#endif
            put_op(assign_a);
            ADOLC_PUT_LOCINT(a_loc);    // = arg
            ADOLC_PUT_LOCINT(loc()); // = res

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
          } else {
	    if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
	      if (coval == 0) {
		 put_op(assign_d_zero);
		 ADOLC_PUT_LOCINT(location);   // = res
	      } else if (coval == 1.0) {
		 put_op(assign_d_one);
		 ADOLC_PUT_LOCINT(location); // = res
	      } else {
		 put_op(assign_d);
		 ADOLC_PUT_LOCINT(location); // = res
		 ADOLC_PUT_VAL(coval);       // = coval
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	    }
	  }
#endif
        }
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
	ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc];
#endif
    }
}

/****************************************************************************/
/*                                                              DESTRUCTORS */

/*--------------------------------------------------------------------------*/
adouble::~adouble() {
#ifdef adolc_overwrite
    if (isInit) {
	free_loc(location);
    }
#endif
}

/*--------------------------------------------------------------------------*/
adub::~adub() {
#ifdef adolc_overwrite
    if (isInit) {
	free_loc(location);
    }
#endif
}


/****************************************************************************/
/*                                                                  HELPERS */

adub* adubp_from_adub(const adub& a) {
    locint locat = a.loc();
    const_cast<adub&>(a).isInit = false;
    adub *retp = new adub(locat);
    return retp;
}


/****************************************************************************/
/*                                                                   VALUE */

/*--------------------------------------------------------------------------*/
double badouble::getValue() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[loc()];
}

badouble::operator double const&() const {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[loc()];
}

badouble::operator double&&() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return (double&&)ADOLC_GLOBAL_TAPE_VARS.store[loc()];
}

badouble::operator double() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_GLOBAL_TAPE_VARS.store[loc()];
}

void badouble::setValue( const double x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.store[loc()]=x;
}

/****************************************************************************/
/*                                                              ASSIGNMENTS */

/*--------------------------------------------------------------------------*/
/* Assign an adouble variable a constant value. */
badouble& badouble::operator = ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(loc());   // = res
        } else
            if (coval == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(loc()); // = res
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(loc()); // = res
                ADOLC_PUT_VAL(coval);       // = coval
            }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = false;
#endif
   return *this;
}

/*--------------------------------------------------------------------------*/
/* Assign an adouble variable a constant value. */
adouble& adouble::operator = ( double coval ) {
    this->loc();  // call for late init
    (*this).badouble::operator=(coval);
    return (*this);
}

/*--------------------------------------------------------------------------*/
/* Assign an adouble variable to an independent value. */
badouble& badouble::operator <<= ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { 
        ADOLC_CURRENT_TAPE_INFOS.numInds++;

        put_op(assign_ind);
        ADOLC_PUT_LOCINT(loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = true;
#endif
    return *this;
}

void badouble::declareIndependent() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        ADOLC_CURRENT_TAPE_INFOS.numInds++;

        put_op(assign_ind);
        ADOLC_PUT_LOCINT(loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
    }
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = true;
#endif
}

/*--------------------------------------------------------------------------*/
/* Assign a float variable from a dependent adouble value. */
badouble& badouble::operator >>= ( double& coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
#if defined(ADOLC_TRACK_ACTIVITY)
    if ( ! ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) as dependent.\n");
	locint tmploc = loc();
	double temp = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	if (temp == 0.0) {
	    put_op(assign_d_zero);
	    ADOLC_PUT_LOCINT(tmploc);
	} else if (temp == 1.0) {
	    put_op(assign_d_one);
	    ADOLC_PUT_LOCINT(tmploc);
	} else {
	    put_op(assign_d);
	    ADOLC_PUT_LOCINT(tmploc);
	    ADOLC_PUT_VAL(temp);
	}

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { 
        ADOLC_CURRENT_TAPE_INFOS.numDeps++;

        put_op(assign_dep);
        ADOLC_PUT_LOCINT(loc()); // = res
    }

    coval = double (ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
    return *this;
}

void badouble::declareDependent() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
#if defined(ADOLC_TRACK_ACTIVITY)
    if ( ! ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) as dependent.\n");
	locint tmploc = loc();
	double temp = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	if (temp == 0.0) {
	    put_op(assign_d_zero);
	    ADOLC_PUT_LOCINT(tmploc);
	} else if (temp == 1.0) {
	    put_op(assign_d_one);
	    ADOLC_PUT_LOCINT(tmploc);
	} else {
	    put_op(assign_d);
	    ADOLC_PUT_LOCINT(tmploc);
	    ADOLC_PUT_VAL(temp);
	}

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
#endif
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        ADOLC_CURRENT_TAPE_INFOS.numDeps++;

        put_op(assign_dep);
        ADOLC_PUT_LOCINT(loc()); // = res
    }
}

/*--------------------------------------------------------------------------*/
/* Assign an Badouble variable an Badouble value. */
badouble& badouble::operator = ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint x_loc = x.loc();
    if (loc()!=x_loc)
        /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
    { if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old:  write_assign_a(loc(),x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {
#endif
            put_op(assign_a);
            ADOLC_PUT_LOCINT(x_loc);    // = arg
            ADOLC_PUT_LOCINT(loc());   // = res

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
          } else {
	    if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
	      if (coval == 0) {
		 put_op(assign_d_zero);
		 ADOLC_PUT_LOCINT(location);   // = res
	      } else if (coval == 1.0) {
		 put_op(assign_d_one);
		 ADOLC_PUT_LOCINT(location); // = res
	      } else {
		 put_op(assign_d);
		 ADOLC_PUT_LOCINT(location); // = res
		 ADOLC_PUT_VAL(coval);       // = coval
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	    }
	  }
#endif
        }

        ADOLC_GLOBAL_TAPE_VARS.store[loc()]=ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
	ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc];
#endif
    }
    return *this;
}

/*--------------------------------------------------------------------------*/
/* Assign an Badouble variable an Badouble value. */
adouble& adouble::operator = ( const badouble& x ) {
    this->loc();  // call for late init
    (*this).badouble::operator=(x);
    return (*this);
}

/*--------------------------------------------------------------------------*/
/* Assign an adouble variable an adouble value. */
adouble& adouble::operator = ( const adouble& x ) {
    this->loc();  // call for late init
    x.loc(); // cal for late init
    (*this).badouble::operator=(x);
    return (*this);
}
/*--------------------------------------------------------------------------*/
/* Assign an adouble an adub */
/* olvo 980517 new version griewank */
badouble& badouble::operator = ( const adub& a ) {
    locint a_loc = a.loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    int upd = 0;
    /* 981020 olvo  skip upd_resloc(..) if no tracing performed */
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
#if defined(ADOLC_TRACK_ACTIVITY)
	upd = upd_resloc_check(a_loc,loc());
#else
        upd = upd_resloc(a_loc,loc());
#endif
    if (upd) { /* olvo 980708 new n2l & 980921 changed interface */
#if defined(ADOLC_TRACK_ACTIVITY)
	free_loc(location);
	location = a_loc;
	const_cast<adub&>(a).isInit = false;
#else
        revreal tempVal = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_OVERWRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()],&ADOLC_GLOBAL_TAPE_VARS.store[a_loc]);
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] = tempVal;
#endif
    } else {
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(loc(),a_loc);
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc]) {
#endif
            put_op(assign_a);
            ADOLC_PUT_LOCINT(a_loc);    // = arg
            ADOLC_PUT_LOCINT(loc()); // = res

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
          } else {
	    if (ADOLC_GLOBAL_TAPE_VARS.actStore[location]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
	      if (coval == 0) {
		 put_op(assign_d_zero);
		 ADOLC_PUT_LOCINT(location);   // = res
	      } else if (coval == 1.0) {
		 put_op(assign_d_one);
		 ADOLC_PUT_LOCINT(location); // = res
	      } else {
		 put_op(assign_d);
		 ADOLC_PUT_LOCINT(location); // = res
		 ADOLC_PUT_VAL(coval);       // = coval
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	    }
	  }
#endif
        }
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
	ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc];
#endif
    }

    return *this;
}

/*--------------------------------------------------------------------------*/
/* Assign an adouble an adub */
/* olvo 980517 new version griewank */
adouble& adouble::operator = ( const adub& a ) {
    this->loc();  // call for late init
    (*this).badouble::operator=(a);
    return (*this);
}


/****************************************************************************/
/*                                                           INPUT / OUTPUT */

/*--------------------------------------------------------------------------*/
/* Output an adouble value !!! No tracing of this action */
std::ostream& operator << ( std::ostream& out, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return out << ADOLC_GLOBAL_TAPE_VARS.store[y.loc()] << "(a)" ;
}

/*--------------------------------------------------------------------------*/
/* Input adouble value */
std::istream& operator >> ( std::istream& in, const badouble& y ) {
    double coval;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    in >> coval;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_d(y.loc(),coval);
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
        if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(y.loc());   // = res
        } else
            if (coval == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(y.loc()); // = res
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(y.loc());   // = res
                ADOLC_PUT_VAL(coval);         // = coval
            }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[y.loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[y.loc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()] = false;
#endif
    return in;
}

/****************************************************************************/
/*                                                    INCREMENT / DECREMENT */

/*--------------------------------------------------------------------------*/
/* Postfix increment */
adub adouble::operator++( int ) {
    locint locat = next_loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(locat,loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(assign_a);
        ADOLC_PUT_LOCINT(loc()); // = arg
        ADOLC_PUT_LOCINT(locat);    // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
	 if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	     double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	     if (coval == 0) {
		 put_op(assign_d_zero);
		 ADOLC_PUT_LOCINT(locat);   // = res
	     } else if (coval == 1.0) {
		 put_op(assign_d_one);
		 ADOLC_PUT_LOCINT(locat); // = res
	     } else {
		 put_op(assign_d);
		 ADOLC_PUT_LOCINT(locat); // = res
		 ADOLC_PUT_VAL(coval);       // = coval
	     }

	     ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	     if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		 ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
	 }
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat]=ADOLC_GLOBAL_TAPE_VARS.store[loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[loc()];
#endif

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(incr_a,loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(incr_a);
        ADOLC_PUT_LOCINT(loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()]++;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Postfix decrement */
adub adouble::operator--( int ) {
    locint locat = next_loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(locat,loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(assign_a);
        ADOLC_PUT_LOCINT(loc()); // = arg
        ADOLC_PUT_LOCINT(locat);    // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
	 if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	     double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	     if (coval == 0) {
		 put_op(assign_d_zero);
		 ADOLC_PUT_LOCINT(locat);   // = res
	     } else if (coval == 1.0) {
		 put_op(assign_d_one);
		 ADOLC_PUT_LOCINT(locat); // = res
	     } else {
		 put_op(assign_d);
		 ADOLC_PUT_LOCINT(locat); // = res
		 ADOLC_PUT_VAL(coval);       // = coval
	     }

	     ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	     if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		 ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
	 }
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat]=ADOLC_GLOBAL_TAPE_VARS.store[loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[loc()];
#endif
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(decr_a,loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(decr_a);
        ADOLC_PUT_LOCINT(loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()]--;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Prefix increment */
badouble& adouble::operator++() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(incr_a,loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(incr_a);
        ADOLC_PUT_LOCINT(loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()]++;
    return *this;
}

/*--------------------------------------------------------------------------*/
/* Prefix decrement */
badouble& adouble::operator--() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_incr_decr_a(decr_a,loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(decr_a);
        ADOLC_PUT_LOCINT(loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()]--;
    return *this;
}

/****************************************************************************/
/*                                                   OPERATION + ASSIGNMENT */

/*--------------------------------------------------------------------------*/
/* Adding a floating point to an adouble */
badouble& badouble::operator += ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_d_same_arg(eq_plus_d,loc(),coval);
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_plus_d);
        ADOLC_PUT_LOCINT(loc()); // = res
        ADOLC_PUT_VAL(coval);       // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] += coval;
    return *this;
}


/*--------------------------------------------------------------------------*/
/* Subtracting a floating point from an adouble */
badouble& badouble::operator -= ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_d_same_arg(eq_min_d,loc(),coval);
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_min_d);
        ADOLC_PUT_LOCINT(loc()); // = res
        ADOLC_PUT_VAL(coval);       // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] -= coval;
    return *this;
}

/*--------------------------------------------------------------------------*/
/* Add an adouble to another adouble */
badouble& badouble::operator += ( const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint y_loc = y.loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_a_same_arg(eq_plus_a,loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc] && ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_plus_a);
        ADOLC_PUT_LOCINT(y_loc); // = arg
        ADOLC_PUT_LOCINT(loc());   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	  if (coval) {
	      put_op(plus_d_a);
	      ADOLC_PUT_LOCINT(y_loc); // = arg
	      ADOLC_PUT_LOCINT(loc()); // = res
	      ADOLC_PUT_VAL(coval);
	  } else {
	      put_op(assign_a);
	      ADOLC_PUT_LOCINT(y_loc); // = arg
	      ADOLC_PUT_LOCINT(loc());      // = res
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
	  if (coval) {
	      put_op(eq_plus_d);
	      ADOLC_PUT_LOCINT(loc()); // = res
	      ADOLC_PUT_VAL(coval);       // = coval

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	  }
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] += ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc]);
#endif
    return *this;
}

/*--------------------------------------------------------------------------*/
/* olvo 991122 new version for y += x1 * x2; */
badouble& badouble::operator += ( const adub& a ) {
    locint a_loc = a.loc();
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    int upd = 0;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
#if defined(ADOLC_TRACK_ACTIVITY)
     if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()])
#endif
      {
        upd = upd_resloc_inc_prod(a_loc,loc(),eq_plus_prod);
      }
    if (upd) {
	/* kk 20170720 Activity Tracking: This case only happens if both
	   lhs and rhs have true in actStore, since previous operation
	   that created rhs is mult_a_a and we checked lhs above.
	   So no change is needed */
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] += ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_DELETE_SCAYLOR(&ADOLC_GLOBAL_TAPE_VARS.store[a_loc]);
        --ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	++ADOLC_CURRENT_TAPE_INFOS.num_eq_prod; 
    } else {
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(loc(),a_loc);
#if defined(ADOLC_TRACK_ACTIVITY)
          if (ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc] && ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
            put_op(eq_plus_a);
            ADOLC_PUT_LOCINT(a_loc);    // = arg
            ADOLC_PUT_LOCINT(loc()); // = res

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
	  } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	      if (coval) {
		  put_op(plus_d_a);
		  ADOLC_PUT_LOCINT(a_loc); // = arg
		  ADOLC_PUT_LOCINT(loc()); // = res
		  ADOLC_PUT_VAL(coval);
	      } else {
		  put_op(assign_a);
		  ADOLC_PUT_LOCINT(a_loc); // = arg
		  ADOLC_PUT_LOCINT(loc());      // = res
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

	  } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
	      if (coval) {
		  put_op(eq_plus_d);
		  ADOLC_PUT_LOCINT(loc()); // = res
		  ADOLC_PUT_VAL(coval);       // = coval

		  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
		  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	      }
	  }
#endif
        }
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] += ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
	ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc]);
#endif
    }

    return *this;
}

/*--------------------------------------------------------------------------*/
/* Subtract an adouble from another adouble */
badouble& badouble::operator -= ( const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint y_loc = y.loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_a_same_arg(eq_min_a,loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc] && ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_min_a);
        ADOLC_PUT_LOCINT(y_loc); // = arg
        ADOLC_PUT_LOCINT(loc());   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	  if (coval) {
	      put_op(min_d_a);
	      ADOLC_PUT_LOCINT(y_loc); // = arg
	      ADOLC_PUT_LOCINT(loc()); // = res
	      ADOLC_PUT_VAL(coval);
	  } else {
	      put_op(neg_sign_a);
	      ADOLC_PUT_LOCINT(y_loc); // = arg
	      ADOLC_PUT_LOCINT(loc());      // = res
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
	  if (coval) {
	      put_op(eq_min_d);
	      ADOLC_PUT_LOCINT(loc()); // = res
	      ADOLC_PUT_VAL(coval);       // = coval

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	  }
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] -= ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc]);
#endif
    return *this;
}

/*--------------------------------------------------------------------------*/
/* olvo 991122 new version for y -= x1 * x2; */
badouble& badouble::operator -= ( const adub& a ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint a_loc = a.loc();
    int upd = 0;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
#if defined(ADOLC_TRACK_ACTIVITY)
     if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()])
#endif
      {
        upd = upd_resloc_inc_prod(a_loc,loc(),eq_min_prod);
      }
    if (upd) {
	/* kk 20170720 Activity Tracking: This case only happens if both
	   lhs and rhs have true in actStore, since previous operation
	   that created rhs is mult_a_a and we checked lhs above.
	   So no change is needed */
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] -= ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_DELETE_SCAYLOR(&ADOLC_GLOBAL_TAPE_VARS.store[a_loc]);
        --ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        ++ADOLC_CURRENT_TAPE_INFOS.num_eq_prod;
    } else {
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_assign_a(loc(),a_loc);
#if defined(ADOLC_TRACK_ACTIVITY)
          if (ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc] && ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
            put_op(eq_min_a);
            ADOLC_PUT_LOCINT(a_loc);    // = arg
            ADOLC_PUT_LOCINT(loc()); // = res

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
	  } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	      if (coval) {
		  put_op(min_d_a);
		  ADOLC_PUT_LOCINT(a_loc); // = arg
		  ADOLC_PUT_LOCINT(loc()); // = res
		  ADOLC_PUT_VAL(coval);
	      } else {
		  put_op(neg_sign_a);
		  ADOLC_PUT_LOCINT(a_loc); // = arg
		  ADOLC_PUT_LOCINT(loc());      // = res
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

	  } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	      double coval = ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
	      if (coval) {
		  put_op(eq_min_d);
		  ADOLC_PUT_LOCINT(loc()); // = res
		  ADOLC_PUT_VAL(coval);       // = coval

		  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
		  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
	      }
	  }
#endif
        }
        ADOLC_GLOBAL_TAPE_VARS.store[loc()] -= ADOLC_GLOBAL_TAPE_VARS.store[a_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
	ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[a_loc]);
#endif
    }

    return *this;
}

/*--------------------------------------------------------------------------*/
/* Multiply an adouble by a floating point */
badouble& badouble::operator *= ( double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_d_same_arg(eq_mult_d,loc(),coval);
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_mult_d);
        ADOLC_PUT_LOCINT(loc()); // = res
        ADOLC_PUT_VAL(coval);       // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] *= coval;
    return *this;
}

/*--------------------------------------------------------------------------*/
/* Multiply one adouble by another adouble*/
badouble& badouble::operator *= ( const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint y_loc = y.loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_a_same_arg(eq_mult_a,loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc] && ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_mult_a);
        ADOLC_PUT_LOCINT(y_loc); // = arg
        ADOLC_PUT_LOCINT(loc());   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
	  if (coval == -1.0) {
	      put_op(neg_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(loc()); // = res
	  } else if (coval == 1.0) {
	      put_op(pos_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(loc()); // = res
	  } else {
              put_op(mult_d_a);
              ADOLC_PUT_LOCINT(y_loc); // = arg
              ADOLC_PUT_LOCINT(loc()); // = res
              ADOLC_PUT_VAL(coval);
          }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
	  put_op(eq_mult_d);
	  ADOLC_PUT_LOCINT(loc()); // = res
	  ADOLC_PUT_VAL(coval);       // = coval

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[loc()] *= ADOLC_GLOBAL_TAPE_VARS.store[y_loc];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] = (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y_loc]);
#endif
    return *this;
}

/*--------------------------------------------------------------------------*/
badouble& badouble::operator /= (double y) {
    *this = *this/y;
    return *this;
}

/*--------------------------------------------------------------------------*/
badouble& badouble::operator /= (const badouble& y) {
    *this = *this * (1.0/y);
    return *this;
}

/****************************************************************************/
/*                                                               COMPARISON */
/*--------------------------------------------------------------------------*/
/*   The Not Equal Operator (!=) */
int operator != ( const badouble& v, double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (coval)
        return (-coval+v != 0);
    else {
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[v.loc()]) {
#endif
            put_op(ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] ? neq_zero : eq_zero);
            ADOLC_PUT_LOCINT(v.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  }
#endif
        }
        return (ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] != 0);
    }
}

/*--------------------------------------------------------------------------*/
/*   The Equal Operator (==) */
int operator == ( const badouble& v, double coval) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (coval)
        return (-coval+v == 0);
    else {
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[v.loc()]) {
#endif
            put_op(ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] ? neq_zero : eq_zero);
            ADOLC_PUT_LOCINT(v.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  }
#endif
        }
        return (ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] == 0);
    }
}

/*--------------------------------------------------------------------------*/
/*   The Less than or Equal Operator (<=)      */
int operator <= ( const badouble& v, double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (coval)
        return (-coval+v <= 0);
    else {
        int b = (ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] <= 0);
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[v.loc()]) {
#endif
            put_op(b ? le_zero : gt_zero);
            ADOLC_PUT_LOCINT(v.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  }
#endif
        }
        return b;
    }
}

/*--------------------------------------------------------------------------*/
/*   The Greater than or Equal Operator (>=)      */
int operator >= ( const badouble& v, double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (coval)
        return (-coval+v >= 0);
    else {
        int b = (ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] >= 0);
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[v.loc()]) {
#endif
            put_op(b ? ge_zero : lt_zero);
            ADOLC_PUT_LOCINT(v.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  }
#endif
        }
        return b;
    }
}

/*--------------------------------------------------------------------------*/
/*   The Greater than Operator (>)      */
int operator > ( const badouble& v, double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (coval)
        return (-coval+v > 0);
    else {
        int b = (ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] > 0);
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[v.loc()]) {
#endif
            put_op(b ? gt_zero : le_zero);
            ADOLC_PUT_LOCINT(v.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  }
#endif
        }
        return b;
    }
}

/*--------------------------------------------------------------------------*/
/*   The Less than Operator (<)      */
int operator < ( const badouble& v, double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (coval)
        return (-coval+v < 0);
    else {
        int b = (ADOLC_GLOBAL_TAPE_VARS.store[v.loc()] < 0);
        if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[v.loc()]) {
#endif
            put_op(b ? lt_zero : ge_zero);
            ADOLC_PUT_LOCINT(v.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
	  }
#endif
        }
        return b;
    }
}


/****************************************************************************/
/*                                                          SIGN  OPERATORS */

/*--------------------------------------------------------------------------*/
/* olvo 980709 modified positive sign operator
   ??? possibly there is a better way */
adub operator + ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_pos_sign_a(locat,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/* olvo 980709 modified negative sign operator */
adub operator - ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_neg_sign_a(locat,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (- coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (- coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(- coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = - coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    return locat;
}


/****************************************************************************/
/*                                                         BINARY OPERATORS */

/* NOTE: each operator calculates address of temporary  and returns
         an adub */

/*--------------------------------------------------------------------------*/
/* Adding two adoubles */
adub operator + ( const badouble& x, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()] + ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_two_a_rec(plus_a_a,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] && ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
        put_op(plus_a_a);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
	  put_op(plus_d_a);
	  ADOLC_PUT_LOCINT(x.loc()); // = arg
	  ADOLC_PUT_LOCINT(locat); // = res
	  ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[y.loc()]);

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
	  if (coval) {
	      put_op(plus_d_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	      ADOLC_PUT_VAL(coval);
	  } else {
	      put_op(pos_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	  }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]);
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Adding a adouble and a floating point */
adub operator + ( double coval, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = coval + ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_args_d_a(plus_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
	 /* olvo 980708 test coval to be zero */
	 if (coval) {
            put_op(plus_d_a);
            ADOLC_PUT_LOCINT(y.loc()); // = arg
            ADOLC_PUT_LOCINT(locat);      // = res
            ADOLC_PUT_VAL(coval);         // = coval
	 } else {
            put_op(pos_sign_a);
            ADOLC_PUT_LOCINT(y.loc()); // = arg
            ADOLC_PUT_LOCINT(locat);      // = res
	 }

	 ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	 if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	     ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()];
#endif

    return locat;
}

/*--------------------------------------------------------------------------*/
/* Subtraction of two adoubles */
adub operator - ( const badouble& x, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()] - ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_two_a_rec(min_a_a,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] && ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
        put_op(min_a_a);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
	  double coval = - ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
	  put_op(plus_d_a);
	  ADOLC_PUT_LOCINT(x.loc()); // = arg
	  ADOLC_PUT_LOCINT(locat); // = res
	  ADOLC_PUT_VAL(coval);

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
	  if (coval) {
	      put_op(min_d_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat);  // = res
	      ADOLC_PUT_VAL(coval);
	  } else {
	      put_op(neg_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]);
#endif
    return locat;
}


/*--------------------------------------------------------------------------*/
/* Subtract an adouble from a floating point */
adub operator - ( double coval, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = coval - ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_args_d_a(min_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
	 /* olvo 980708 test coval to be zero */
	 if (coval) {
            put_op(min_d_a);
            ADOLC_PUT_LOCINT(y.loc()); // = arg
            ADOLC_PUT_LOCINT(locat);      // = res
            ADOLC_PUT_VAL(coval);         // = coval
	 } else {
            put_op(neg_sign_a);
            ADOLC_PUT_LOCINT(y.loc()); // = arg
            ADOLC_PUT_LOCINT(locat);      // = res
	 }

	 ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	 if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	     ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()];
#endif

    return locat;
}

/*--------------------------------------------------------------------------*/
/* Multiply two adoubles */
adub operator * ( const badouble& x, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()] * ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_two_a_rec(mult_a_a,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] && ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
        put_op(mult_a_a);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
	  put_op(mult_d_a);
	  ADOLC_PUT_LOCINT(x.loc()); // = arg
	  ADOLC_PUT_LOCINT(locat); // = res
	  ADOLC_PUT_VAL(coval);

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
	  if (coval == -1.0) {
	      put_op(neg_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	  } else if (coval == 1.0) {
	      put_op(pos_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	  } else {
	      put_op(mult_d_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	      ADOLC_PUT_VAL(coval);
	  }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]);
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Multiply an adouble by a floating point */
/* olvo 980709 modified */
adub operator * ( double coval, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = coval * ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_pos_sign_a(locat,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
	 if ( coval == 1.0 ) {
            put_op(pos_sign_a);
            ADOLC_PUT_LOCINT(y.loc()); // = arg
            ADOLC_PUT_LOCINT(locat);      // = res
	 } else if ( coval == -1.0 ) {
	     put_op(neg_sign_a);
	     ADOLC_PUT_LOCINT(y.loc()); // = arg
	     ADOLC_PUT_LOCINT(locat);      // = res
	 } else {
	     put_op(mult_d_a);
	     ADOLC_PUT_LOCINT(y.loc()); // = arg
	     ADOLC_PUT_LOCINT(locat);      // = res
	     ADOLC_PUT_VAL(coval);         // = coval
	 }

	 ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	 if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	     ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()];
#endif

    return locat;
}

/*--------------------------------------------------------------------------*/
/* Divide an adouble by another adouble */
adub operator / ( const badouble& x, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()] / ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_two_a_rec(div_a_a,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] && ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
        put_op(div_a_a);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
	  double coval = 1.0/ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
	  if (coval == -1.0) {
	      put_op(neg_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	  } else if (coval == 1.0) {
	      put_op(pos_sign_a);
	      ADOLC_PUT_LOCINT(y.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	  } else {
	      put_op(mult_d_a);
	      ADOLC_PUT_LOCINT(x.loc()); // = arg
	      ADOLC_PUT_LOCINT(locat); // = res
	      ADOLC_PUT_VAL(coval);
	  }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
	  double coval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
	  put_op(div_d_a);
	  ADOLC_PUT_LOCINT(y.loc()); // = arg
	  ADOLC_PUT_LOCINT(locat); // = res
	  ADOLC_PUT_VAL(coval);

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]);
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Division floating point - adouble */
adub operator / ( double coval, const badouble& y ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = coval / ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_args_d_a(div_d_a,locat,coval,y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
#endif
        put_op(div_d_a);
        ADOLC_PUT_LOCINT(y.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res
        ADOLC_PUT_VAL(coval);         // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()];
#endif
    return locat;
}


/****************************************************************************/
/*                                                        SINGLE OPERATIONS */

/*--------------------------------------------------------------------------*/
/* Compute exponential of adouble */
adub exp ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::exp(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_single_op(exp_op,locat,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(exp_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif

    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Compute logarithm of adouble */
adub log ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::log(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_single_op(log_op,locat,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(log_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Compute sqrt of adouble */
adub sqrt ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::sqrt(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_single_op(sqrt_op,locat,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(sqrt_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Compute cbrt of adouble */
adub cbrt ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::cbrt(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(cbrt_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/****************************************************************************/
/*                                                          QUAD OPERATIONS */

/*--------------------------------------------------------------------------*/
/* Compute sin of adouble
   !!! Sin and Cos are always evaluated together
*/
adub sin ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval1 = ADOLC_MATH_NSP::sin(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);
    double coval2 = ADOLC_MATH_NSP::cos(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(sin_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(sin_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) { /* olvo 980921 changed order */
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[y.loc()]);
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
        }
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	      if (coval1 == 0.0) {
		  put_op(assign_d_zero);
		  ADOLC_PUT_LOCINT(locat);
	      } else if (coval1 == 1.0) {
		  put_op(assign_d_one);
		  ADOLC_PUT_LOCINT(locat);
	      } else {
		  put_op(assign_d);
		  ADOLC_PUT_LOCINT(locat);
		  ADOLC_PUT_VAL(coval1);
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
	  }
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
	      if (coval2 == 0.0) {
		  put_op(assign_d_zero);
		  ADOLC_PUT_LOCINT(y.loc());
	      } else if (coval1 == 1.0) {
		  put_op(assign_d_one);
		  ADOLC_PUT_LOCINT(y.loc());
	      } else {
		  put_op(assign_d);
		  ADOLC_PUT_LOCINT(y.loc());
		  ADOLC_PUT_VAL(coval2);
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[y.loc()]);
	  }
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval1;
    ADOLC_GLOBAL_TAPE_VARS.store[y.loc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif

    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Compute cos of adouble */
adub cos ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval1 = ADOLC_MATH_NSP::cos(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);
    double coval2 = ADOLC_MATH_NSP::sin(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(cos_op, locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(cos_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) { /* olvo 980921 changed order */
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[y.loc()]);
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
        }
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	      if (coval1 == 0.0) {
		  put_op(assign_d_zero);
		  ADOLC_PUT_LOCINT(locat);
	      } else if (coval1 == 1.0) {
		  put_op(assign_d_one);
		  ADOLC_PUT_LOCINT(locat);
	      } else {
		  put_op(assign_d);
		  ADOLC_PUT_LOCINT(locat);
		  ADOLC_PUT_VAL(coval1);
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
	  }
	  if (ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]) {
	      if (coval2 == 0.0) {
		  put_op(assign_d_zero);
		  ADOLC_PUT_LOCINT(y.loc());
	      } else if (coval1 == 1.0) {
		  put_op(assign_d_one);
		  ADOLC_PUT_LOCINT(y.loc());
	      } else {
		  put_op(assign_d);
		  ADOLC_PUT_LOCINT(y.loc());
		  ADOLC_PUT_VAL(coval2);
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[y.loc()]);
	  }
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval1;
    ADOLC_GLOBAL_TAPE_VARS.store[y.loc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif

    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Compute tan of adouble */
adub tan ( const badouble& x ) {
    return sin(x) / cos(x);
}

/*--------------------------------------------------------------------------*/
/* Asin value -- really a quadrature */
adub asin ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::asin(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = 1.0 / sqrt(1.0 - x*x);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old:  write_quad(asin_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(asin_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Acos value -- really a quadrature */
adub acos ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::acos(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = -1.0 / sqrt(1.0 - x*x);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(acos_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(acos_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Atan value -- really a quadrature */
adub atan ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP::atan(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = 1.0 / (1.0 + x*x);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(atan_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(atan_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
adouble atan2( const badouble& y, const badouble& x) {
    adouble a1, a2, ret, sy;
    const double pihalf = ADOLC_MATH_NSP::asin(1.0);
    /* y+0.0 is a hack since condassign is currently not defined for
       badoubles */
    condassign( sy,  y,  (adouble)1.0 , (adouble)-1.0 );
    condassign( a1,  x,  atan(y/x),  atan(y/x)+sy*2*pihalf);
    condassign( a2,  fabs(y), sy*pihalf-atan(x/y), (adouble) 0.0 );
    condassign( ret, fabs(x) - fabs(y), a1, a2 );
    return ret;
}

/*--------------------------------------------------------------------------*/
/* power value -- adouble ^ floating point */
adub pow ( const badouble& x, double coval ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval2 = ADOLC_MATH_NSP::pow(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()],coval);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_args_d_a(pow_op,locat,cocval,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(pow_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res
        ADOLC_PUT_VAL(coval);         // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval2 == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval2 == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval2);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* power value --- floating point ^ adouble */
adouble pow ( double coval, const badouble& y ) {
    adouble ret;

    if (coval <= 0) {
        fprintf(DIAG_OUT,"\nADOL-C message:  exponent at zero/negative constant basis deactivated\n");
    }

    condassign (ret, (adouble) coval, exp(y*ADOLC_MATH_NSP::log(coval)),
		(adouble) ADOLC_MATH_NSP::pow(coval,y.getValue()) );

    return ret;
}

/*--------------------------------------------------------------------------*/
/* power value --- adouble ^ adouble */
adouble pow ( const badouble& x, const badouble& y) {
    adouble a1, a2, ret;
    double vx = x.getValue();
    double vy = y.getValue();

    if (!(vx > 0)) { 
        if (vx < 0 || vy >= 0)
            fprintf(DIAG_OUT,"\nADOL-C message: exponent of zero/negative basis deactivated\n");
        else
            fprintf(DIAG_OUT,"\nADOL-C message: negative exponent and zero basis deactivated\n");
    }
    condassign(a1, -y, (adouble) ADOLC_MATH_NSP::pow(vx,vy), pow(x,vy));
    condassign(a2, fabs(x), pow(x, vy), a1);
    condassign(ret, x, exp(y*log(x)),a2);

    return ret;
}

/*--------------------------------------------------------------------------*/
/* log base 10 of an adouble */
adub log10 ( const badouble& x ) {
    return log(x) / ADOLC_MATH_NSP::log(10.0);
}

/*--------------------------------------------------------------------------*/
/* Hyperbolic Sine of an adouble */
/* 981119 olvo changed as J.M. Aparicio suggested */
adub sinh ( const badouble& x ) {
    if (x.getValue() < 0.0) {
        adouble temp = exp(x);
        return  0.5*(temp - 1.0/temp);
    } else {
        adouble temp = exp(-x);
        return 0.5*(1.0/temp - temp);
    }
}

/*--------------------------------------------------------------------------*/
/* Hyperbolic Cosine of an adouble */
/* 981119 olvo changed as J.M. Aparicio suggested */
adub cosh ( const badouble& x ) {
    adouble temp = (x.getValue() < 0.0) ? exp(x) : exp(-x);
    return 0.5*(temp + 1.0/temp);
}

/*--------------------------------------------------------------------------*/
/*
  Hyperbolic Tangent of an adouble value.
*/
/* 981119 olvo changed as J.M. Aparicio suggested */
adub tanh ( const badouble& x ) {
    if (x.getValue() < 0.0) {
        adouble temp = exp(2.0*x);
        return (temp - 1.0)/(temp + 1.0);
    } else {
        adouble temp = exp((-2.0)*x);
        return (1.0 - temp)/(temp + 1.0);
    }
}

/*--------------------------------------------------------------------------*/
/* Ceiling function (NOTE: This function is nondifferentiable) */
adub ceil ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat=next_loc();

    double coval = ADOLC_MATH_NSP::ceil(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_args_d_a(ceil_op,locat,coval,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(ceil_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res
        ADOLC_PUT_VAL(coval);         // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Floor function (NOTE: This function is nondifferentiable) */
adub floor ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat=next_loc();

    double coval =
        ADOLC_MATH_NSP::floor(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_args_d_a(floor_op,locat,coval,x.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(floor_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg
        ADOLC_PUT_LOCINT(locat);      // = res
        ADOLC_PUT_VAL(coval);         // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    return locat;
}

#ifdef ATRIG_ERF
/* NOTE: enable if your compiler knows asinh, acosh, atanh, erf */

/*--------------------------------------------------------------------------*/
/* Asinh value -- really a quadrature */
adub asinh ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP_ERF::asinh(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = 1.0 / sqrt(1.0 + x*x);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(asinh_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(asinh_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Acosh value -- really a quadrature */
adub acosh ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP_ERF::acosh(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = 1.0 / sqrt(x*x-1.0);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(acosh_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(acosh_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/* Atanh value -- really a quadrature */
adub atanh ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP_ERF::atanh(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = 1.0 / (1.0 - x*x);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(atanh_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(atanh_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

/*--------------------------------------------------------------------------*/
/*  The error function erf */
adub erf( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();
    double coval = ADOLC_MATH_NSP_ERF::erf(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);

    adouble y = 2.0 /
        ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0))*exp(-x*x);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_quad(erf_op,locat,x.loc(),y.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) { // y will have same activity as x and can be considered as second input here
#endif
        put_op(erf_op);
        ADOLC_PUT_LOCINT(x.loc()); // = arg1
        ADOLC_PUT_LOCINT(y.loc()); // = arg2
        ADOLC_PUT_LOCINT(locat);      // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (coval == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (coval == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(coval);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
    return locat;
}

#endif

/*--------------------------------------------------------------------------*/
/* Fabs Function (NOTE: This function is also nondifferentiable at x=0) */
adub fabs ( const badouble& x ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint locat = next_loc();

    double coval = 1.0;
    double temp  = ADOLC_MATH_NSP::fabs(ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]);
    if (temp != ADOLC_GLOBAL_TAPE_VARS.store[x.loc()])
        coval = 0.0;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { /*  write_args_d_a(abs_val,locat,coval,x.loc()); */
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()]) {
#endif
        put_op(abs_val);
        ADOLC_PUT_LOCINT(x.loc());   /* arg */
        ADOLC_PUT_LOCINT(locat);        /* res */
        ADOLC_PUT_VAL(coval);           /* coval */

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
	    ++ADOLC_CURRENT_TAPE_INFOS.numSwitches;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	  if (temp == 0.0) {
	      put_op(assign_d_zero);
	      ADOLC_PUT_LOCINT(locat);
	  } else if (temp == 1.0) {
	      put_op(assign_d_one);
	      ADOLC_PUT_LOCINT(locat);
	  } else {
	      put_op(assign_d);
	      ADOLC_PUT_LOCINT(locat);
	      ADOLC_PUT_VAL(temp);
	  }

	  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
      }
#endif
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = temp;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()];
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/* max and min functions  (changed : 11/15/95) */
adub fmin ( const badouble& x, const badouble& y ) { /* olvo 980702 tested: return 0.5*fabs(x+y-fabs(x-y)); */
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
	return ((x + y - fabs(x - y))/2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if ( ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()] && ! ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] ) {
	locint tmploc = x.loc();
	double temp = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
	if (temp == 0.0) {
	    put_op(assign_d_zero);
	    ADOLC_PUT_LOCINT(tmploc);
	} else if (temp == 1.0) {
	    put_op(assign_d_one);
	    ADOLC_PUT_LOCINT(tmploc);
	} else {
	    put_op(assign_d);
	    ADOLC_PUT_LOCINT(tmploc);
	    ADOLC_PUT_VAL(temp);
	}

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
    if ( ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] && ! ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()] ) {
	locint tmploc = y.loc();
	double temp = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
	if (temp == 0.0) {
	    put_op(assign_d_zero);
	    ADOLC_PUT_LOCINT(tmploc);
	} else if (temp == 1.0) {
	    put_op(assign_d_one);
	    ADOLC_PUT_LOCINT(tmploc);
	} else {
	    put_op(assign_d);
	    ADOLC_PUT_LOCINT(tmploc);
	    ADOLC_PUT_VAL(temp);
	}

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
    }
  }
#endif

    locint locat = next_loc();
    double coval, temp;

    if (ADOLC_GLOBAL_TAPE_VARS.store[y.loc()] < ADOLC_GLOBAL_TAPE_VARS.store[x.loc()]) {
	coval = 0.0;
	temp = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    } else {
	coval = 1.0;
	temp = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    }

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_min_op(x.loc(),y.loc(),locat,0.0);
#if defined(ADOLC_TRACK_ACTIVITY)
	if ( ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()] ) {
#endif
            put_op(min_op);
            ADOLC_PUT_LOCINT(x.loc()); // = arg1
            ADOLC_PUT_LOCINT(y.loc()); // = arg2
            ADOLC_PUT_LOCINT(locat);      // = res
            ADOLC_PUT_VAL(coval);           // = coval

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
#if defined(ADOLC_TRACK_ACTIVITY)
	  } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {
	      if (temp == 0.0) {
		  put_op(assign_d_zero);
		  ADOLC_PUT_LOCINT(locat);
	      } else if (temp == 1.0) {
		  put_op(assign_d_one);
		  ADOLC_PUT_LOCINT(locat);
	      } else {
		  put_op(assign_d);
		  ADOLC_PUT_LOCINT(locat);
		  ADOLC_PUT_VAL(temp);
	      }

	      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
		  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
	  }
#endif
    }

    ADOLC_GLOBAL_TAPE_VARS.store[locat]=temp;
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[locat] = (ADOLC_GLOBAL_TAPE_VARS.actStore[x.loc()] || ADOLC_GLOBAL_TAPE_VARS.actStore[y.loc()]);
#endif
    return locat;
}

/*--------------------------------------------------------------------------*/
/*21.8.96*/
adub fmin ( double d, const badouble& y ) {
    adouble x = d;
    return (fmin (x,y));
}

/*--------------------------------------------------------------------------*/
adub fmin ( const badouble& x, double d ) {
    adouble y = d;
    return (fmin (x,y));
}

/*--------------------------------------------------------------------------*/
adub fmax ( const badouble& x, const badouble& y ) {
    return (-fmin(-x,-y));
}

/*--------------------------------------------------------------------------*/
/*21.8.96*/
adub fmax ( double d, const badouble& y ) {
    adouble x = d;
    return (-fmin(-x,-y));
}

/*--------------------------------------------------------------------------*/
adub fmax ( const badouble& x, double d ) {
    adouble y = d;
    return (-fmin(-x,-y));
}

/*--------------------------------------------------------------------------*/
/* Ldexp Function */
adub ldexp ( const badouble& x, int exp ) {
    return x*ldexp(1.0,exp);
}
/*--------------------------------------------------------------------------*/
/* frexp Function */
adub frexp ( const badouble& x, int* n) {
    double v = frexp(x.value(), n);
    adouble r = x - v;
    adouble z = r - double(*n);
    if (z == 0) {
        return (x - double(*n));
    } else {
        fprintf(stderr,"ADOL-C warning: std::frexp() returned inconsistent results\n");
        return (r - double(*n));
    }
}

/*--------------------------------------------------------------------------*/
/* Macro for user defined quadratures, example myquad is below.*/
/* the forward sweep tests if the tape is executed exactly at  */
/* the same argument point otherwise it stops with a returnval */
#define extend_quad(func,integrand)\
adouble func ( const badouble& arg )\
{  adouble temp; \
    adouble val; \
    integrand; \
    ADOLC_OPENMP_THREAD_NUMBER; \
    ADOLC_OPENMP_GET_THREAD_NUMBER; \
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) \
    { put_op(gen_quad); \
      ADOLC_PUT_LOCINT(arg.loc()); \
      ADOLC_PUT_LOCINT(val.loc()); \
      ADOLC_PUT_LOCINT(temp.loc()); \
      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape; \
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) \
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[temp.loc()]); \
    } \
    ADOLC_GLOBAL_TAPE_VARS.store[temp.loc()]=func(ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()]); \
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) \
    { ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()]); \
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[temp.loc()]); \
    } \
    return temp; }

double myquad(double& x) {
    double res;
    res = ADOLC_MATH_NSP::log(x);
    return res;
}

/* This defines the natural logarithm as a quadrature */

extend_quad(myquad,val = 1/arg)


/****************************************************************************/
/*                                                             CONDITIONALS */
/*--------------------------------------------------------------------------*/
#if defined(ADOLC_ADVANCED_BRANCHING)
adub operator != (const badouble& x, const badouble& y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    double xval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    double yval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    double res = (double)(xval != yval);
    locint locat = next_loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(neq_a_a);
	ADOLC_PUT_LOCINT(x.loc()); // arg
	ADOLC_PUT_LOCINT(y.loc()); // arg1
	ADOLC_PUT_VAL(res);           // check for branch switch
	ADOLC_PUT_LOCINT(locat);      // res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = res;
    return locat;
}
/*--------------------------------------------------------------------------*/
adub operator == (const badouble& x, const badouble& y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    double xval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    double yval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    double res = (double)(xval == yval);
    locint locat = next_loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(eq_a_a);
	ADOLC_PUT_LOCINT(x.loc()); // arg
	ADOLC_PUT_LOCINT(y.loc()); // arg1
	ADOLC_PUT_VAL(res);           // check for branch switch
	ADOLC_PUT_LOCINT(locat);      // res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = res;
    return locat;
}
/*--------------------------------------------------------------------------*/
adub operator <= (const badouble& x, const badouble& y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    double xval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    double yval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    double res = (double)(xval <= yval);
    locint locat = next_loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(le_a_a);
	ADOLC_PUT_LOCINT(x.loc()); // arg
	ADOLC_PUT_LOCINT(y.loc()); // arg1
	ADOLC_PUT_VAL(res);           // check for branch switch
	ADOLC_PUT_LOCINT(locat);      // res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = res;
    return locat;
}
/*--------------------------------------------------------------------------*/
adub operator >= (const badouble& x, const badouble& y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    double xval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    double yval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    double res = (double)(xval >= yval);
    locint locat = next_loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(ge_a_a);
	ADOLC_PUT_LOCINT(x.loc()); // arg
	ADOLC_PUT_LOCINT(y.loc()); // arg1
	ADOLC_PUT_VAL(res);           // check for branch switch
	ADOLC_PUT_LOCINT(locat);      // res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = res;
    return locat;
}
/*--------------------------------------------------------------------------*/
adub operator > (const badouble& x, const badouble& y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    double xval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    double yval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    double res = (double)(xval > yval);
    locint locat = next_loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(gt_a_a);
	ADOLC_PUT_LOCINT(x.loc()); // arg
	ADOLC_PUT_LOCINT(y.loc()); // arg1
	ADOLC_PUT_VAL(res);           // check for branch switch
	ADOLC_PUT_LOCINT(locat);      // res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = res;
    return locat;
}
/*--------------------------------------------------------------------------*/
adub operator < (const badouble& x, const badouble& y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    double xval = ADOLC_GLOBAL_TAPE_VARS.store[x.loc()];
    double yval = ADOLC_GLOBAL_TAPE_VARS.store[y.loc()];
    double res = (double)(xval < yval);
    locint locat = next_loc();
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
	put_op(lt_a_a);
	ADOLC_PUT_LOCINT(x.loc()); // arg
	ADOLC_PUT_LOCINT(y.loc()); // arg1
	ADOLC_PUT_VAL(res);           // check for branch switch
	ADOLC_PUT_LOCINT(locat);      // res

	++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
	if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	    ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[locat]);
    }
    ADOLC_GLOBAL_TAPE_VARS.store[locat] = res;
    return locat;
}
#endif
/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
void condassign( adouble &res,         const badouble &cond,
                 const badouble &arg1, const badouble &arg2 ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_condassign(res.loc(),cond.loc(),arg1.loc(),
        //		     arg2.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if ( ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (! ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()] ) {
            locint tmploc = arg1.loc();
            double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg1.loc()];
            if (temp == 0.0) {
                put_op(assign_d_zero);
                ADOLC_PUT_LOCINT(tmploc);
            } else if (temp == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(tmploc);
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(tmploc);
                ADOLC_PUT_VAL(temp);
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
        }
        if (! ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()] ) {
            locint tmploc = arg2.loc();
            double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg2.loc()];
            if (temp == 0.0) {
                put_op(assign_d_zero);
                ADOLC_PUT_LOCINT(tmploc);
            } else if (temp == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(tmploc);
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(tmploc);
                ADOLC_PUT_VAL(temp);
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
        }
#endif
        put_op(cond_assign);
        ADOLC_PUT_LOCINT(cond.loc()); // = arg
        ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()]);
        ADOLC_PUT_LOCINT(arg1.loc()); // = arg1
        ADOLC_PUT_LOCINT(arg2.loc()); // = arg2
        ADOLC_PUT_LOCINT(res.loc());  // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
          locint x_loc;
          if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
              x_loc = arg1.loc();
          else
              x_loc = arg2.loc();

          if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {
              put_op(assign_a);
              ADOLC_PUT_LOCINT(x_loc);    // = arg
              ADOLC_PUT_LOCINT(res.loc());   // = res

              ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
              if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
          } else {
              if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {
                  double coval = ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
                  if (coval == 0) {
                      put_op(assign_d_zero);
                      ADOLC_PUT_LOCINT(res.loc());   // = res
                  } else if (coval == 1.0) {
                      put_op(assign_d_one);
                      ADOLC_PUT_LOCINT(res.loc()); // = res
                  } else {
                      put_op(assign_d);
                      ADOLC_PUT_LOCINT(res.loc()); // = res
                      ADOLC_PUT_VAL(coval);       // = coval
                  }

                  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
                  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
              }
          }
      }
#endif
    }

    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
        ADOLC_GLOBAL_TAPE_VARS.store[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.store[arg1.loc()];
    else
        ADOLC_GLOBAL_TAPE_VARS.store[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.store[arg2.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    if ( ! ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
            ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =  ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()];
        else
            ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =  ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()];
    } else
        ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];
#endif
}

/*--------------------------------------------------------------------------*/
void condassign( adouble &res, const badouble &cond, const badouble &arg ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_condassign2(res.loc(),cond.loc(),arg.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if ( ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (! ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()] ) {
            locint tmploc = arg.loc();
            double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()];
            if (temp == 0.0) {
                put_op(assign_d_zero);
                ADOLC_PUT_LOCINT(tmploc);
            } else if (temp == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(tmploc);
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(tmploc);
                ADOLC_PUT_VAL(temp);
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
        }
#endif
        put_op(cond_assign_s);
        ADOLC_PUT_LOCINT(cond.loc()); // = arg
        ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()]);
        ADOLC_PUT_LOCINT(arg.loc());  // = arg1
        ADOLC_PUT_LOCINT(res.loc());  // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
          locint x_loc;
          if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0) {
              x_loc = arg.loc();
              if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {
                  put_op(assign_a);
                  ADOLC_PUT_LOCINT(x_loc);    // = arg
                  ADOLC_PUT_LOCINT(res.loc());   // = res

                  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
                  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
              } else {
                  if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {
                      double coval = ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
                      if (coval == 0) {
                          put_op(assign_d_zero);
                          ADOLC_PUT_LOCINT(res.loc());   // = res
                      } else if (coval == 1.0) {
                          put_op(assign_d_one);
                          ADOLC_PUT_LOCINT(res.loc()); // = res
                      } else {
                          put_op(assign_d);
                          ADOLC_PUT_LOCINT(res.loc()); // = res
                          ADOLC_PUT_VAL(coval);       // = coval
                      }

                      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
                      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
                  }
              }
          }
      }
#endif
    }

    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
        ADOLC_GLOBAL_TAPE_VARS.store[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    if ( ! ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
            ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =  ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()];
    } else
        ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];
#endif
}
/*--------------------------------------------------------------------------*/
void condeqassign( adouble &res,         const badouble &cond,
                   const badouble &arg1, const badouble &arg2 ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_condassign(res.loc(),cond.loc(),arg1.loc(),
        //		     arg2.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if ( ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (! ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()] ) {
            locint tmploc = arg1.loc();
            double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg1.loc()];
            if (temp == 0.0) {
                put_op(assign_d_zero);
                ADOLC_PUT_LOCINT(tmploc);
            } else if (temp == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(tmploc);
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(tmploc);
                ADOLC_PUT_VAL(temp);
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
        }
        if (! ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()] ) {
            locint tmploc = arg2.loc();
            double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg2.loc()];
            if (temp == 0.0) {
                put_op(assign_d_zero);
                ADOLC_PUT_LOCINT(tmploc);
            } else if (temp == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(tmploc);
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(tmploc);
                ADOLC_PUT_VAL(temp);
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
        }
#endif
        put_op(cond_eq_assign);
        ADOLC_PUT_LOCINT(cond.loc()); // = arg
        ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()]);
        ADOLC_PUT_LOCINT(arg1.loc()); // = arg1
        ADOLC_PUT_LOCINT(arg2.loc()); // = arg2
        ADOLC_PUT_LOCINT(res.loc());  // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
          locint x_loc;
          if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
              x_loc = arg1.loc();
          else
              x_loc = arg2.loc();

          if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {
              put_op(assign_a);
              ADOLC_PUT_LOCINT(x_loc);    // = arg
              ADOLC_PUT_LOCINT(res.loc());   // = res

              ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
              if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                  ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
          } else {
              if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {
                  double coval = ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
                  if (coval == 0) {
                      put_op(assign_d_zero);
                      ADOLC_PUT_LOCINT(res.loc());   // = res
                  } else if (coval == 1.0) {
                      put_op(assign_d_one);
                      ADOLC_PUT_LOCINT(res.loc()); // = res
                  } else {
                      put_op(assign_d);
                      ADOLC_PUT_LOCINT(res.loc()); // = res
                      ADOLC_PUT_VAL(coval);       // = coval
                  }

                  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
                  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
              }
          }
      }
#endif
    }

    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] >= 0)
        ADOLC_GLOBAL_TAPE_VARS.store[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.store[arg1.loc()];
    else
        ADOLC_GLOBAL_TAPE_VARS.store[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.store[arg2.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    if ( ! ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
            ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =  ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()];
        else
            ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =  ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()];
    } else
        ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];
#endif
}

/*--------------------------------------------------------------------------*/
void condeqassign( adouble &res, const badouble &cond, const badouble &arg ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) { // old: write_condassign2(res.loc(),cond.loc(),arg.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      if ( ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (! ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()] ) {
            locint tmploc = arg.loc();
            double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()];
            if (temp == 0.0) {
                put_op(assign_d_zero);
                ADOLC_PUT_LOCINT(tmploc);
            } else if (temp == 1.0) {
                put_op(assign_d_one);
                ADOLC_PUT_LOCINT(tmploc);
            } else {
                put_op(assign_d);
                ADOLC_PUT_LOCINT(tmploc);
                ADOLC_PUT_VAL(temp);
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
        }
#endif
        put_op(cond_eq_assign_s);
        ADOLC_PUT_LOCINT(cond.loc()); // = arg
        ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()]);
        ADOLC_PUT_LOCINT(arg.loc());  // = arg1
        ADOLC_PUT_LOCINT(res.loc());  // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
          locint x_loc;
          if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0) {
              x_loc = arg.loc();
              if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {
                  put_op(assign_a);
                  ADOLC_PUT_LOCINT(x_loc);    // = arg
                  ADOLC_PUT_LOCINT(res.loc());   // = res

                  ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
                  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
              } else {
                  if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {
                      double coval = ADOLC_GLOBAL_TAPE_VARS.store[x_loc];
                      if (coval == 0) {
                          put_op(assign_d_zero);
                          ADOLC_PUT_LOCINT(res.loc());   // = res
                      } else if (coval == 1.0) {
                          put_op(assign_d_one);
                          ADOLC_PUT_LOCINT(res.loc()); // = res
                      } else {
                          put_op(assign_d);
                          ADOLC_PUT_LOCINT(res.loc()); // = res
                          ADOLC_PUT_VAL(coval);       // = coval
                      }

                      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
                      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
                          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.loc()]);
                  }
              }
          }
      }
#endif
    }

    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] >= 0)
        ADOLC_GLOBAL_TAPE_VARS.store[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    if ( ! ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()] ) {
        if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
            ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =  ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()];
    } else
        ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] = ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];
#endif
}

/****************************************************************************/
/*                                                                THAT'S ALL*/
