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

#include "adouble.h"
#include "dvlparms.h"
#include "oplate.h"
#include "taping_p.h"
#include <adolc/adouble.h>

adouble::adouble() {
  tape_loc_{next_loc()};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

#if defined(ADOLC_ADOUBLE_STDCZERO)
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(assign_d_zero);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = 0.;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = false;
#endif
#endif
}

adouble::adouble(double coval) {
  tape_loc_{next_loc()};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      if (coval == 0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);             // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = false;
#endif
}

adouble::adouble(const tape_location &tape_loc) { tape_loc_{tape_loc}; }

adouble::adouble(const adouble &a) {
  tape_loc_{a.tape_loc_};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
      put_op(assign_a);
      ADOLC_PUT_LOCINT(a.getLoc());     // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
        const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
        if (coval == 0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        } else if (coval == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
          ADOLC_PUT_VAL(coval);             // = coval
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
      }
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] =
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
}

// Moving a's tape_location to the tape_location of this and set a's state to
// invalid (valid = 0). This will tell the destruction that "a" does not own its
// location. Thus, the location is not removed from the tape.
adouble::adouble(adouble &&a) noexcept {
  tape_loc_.loc_{a.tape_loc_.loc};
  a.valid = 0;
}

/*
 * The destructor is used to remove unused locations (tape_loc_.loc_) from the
 * tape. A location is only removed (free_loc), if the destructed adouble owns
 * the location. The adouble does not own its location if it is in an invalid
 * state (valid = 0). The state is only invalid, if the adouble was moved to a
 * new adouble. The location is reused for the new adouble in this case and must
 * remain on the tape.
 */
adouble::~adouble() {
#ifdef adolc_overwrite
  if (valid) {
    free_loc(tape_loc_.loc_);
  }
#endif
}

/****************************************************************************/
/*                                                              ASSIGNMENTS */

adouble &adouble::operator=(double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      if (coval == 0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);             // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = false;
#endif
  return *this;
}

adouble &adouble::operator=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
  if (tape_loc_.loc_ != a.getLoc()) {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(assign_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
          double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
          if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
          } else if (coval == 1.0) {
            put_op(assign_d_one);
            ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
          } else {
            put_op(assign_d);
            ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
            ADOLC_PUT_VAL(coval);             // = coval
          }
          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
        }
      }
#endif
    }
    ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] =
        ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  }
  return *this;
}

// moves the tape_location of "a" to "this" and sets "a" to an invalid state.
adouble &adouble::operator=(adouble &&a) noexcept {
  if (this == &a) {
    return *this;
  }
  // remove location of this from tape to ensure it can be reused
  free_loc(tape_loc_.loc_);

  tape_loc_{a.getLoc()};
  a.valid = 0;

  return *this;
}

adouble &adouble::operator=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(assign_p);
    ADOLC_PUT_LOCINT(p.getLoc());
    ADOLC_PUT_LOCINT(tape_loc_.loc_);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = p.getValue();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif
  return *this;
}

/****************************************************************************/
/*            conversions */

adouble::operator double() const {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  return ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
}

adouble::operator const double &() const {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  return ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
}

/****************************************************************************/
/*                       ARITHMETIC ASSIGNMENT                             */

adouble &adouble::operator+=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_plus_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      ADOLC_PUT_VAL(coval);             // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] += coval;
  return *this;
}

adouble &adouble::operator+=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_plus_a);
      ADOLC_PUT_LOCINT(a.getLoc());     // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(assign_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
      if (coval) {
        put_op(eq_plus_d);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);             // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
      }
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] +=
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]);
#endif
  return *this;
}

adouble &adouble::operator+=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_plus_p);
    ADOLC_PUT_LOCINT(p.getLoc());
    ADOLC_PUT_LOCINT(tape_loc_.loc_);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] += p.getValue();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif

  return *this;
}

adouble &adouble::operator-=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_min_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      ADOLC_PUT_VAL(coval);             // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] -= coval;
  return *this;
}

adouble &adouble::operator-=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_min_a);
      ADOLC_PUT_LOCINT(a.getLoc());     // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
      if (coval) {
        put_op(eq_min_d);
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);             // = coval

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
      }
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] -=
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]);
#endif
  return *this;
}

adouble &adouble::operator-=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_min_p);
    ADOLC_PUT_LOCINT(p.getLoc());
    ADOLC_PUT_LOCINT(tape_loc_.loc_);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] -= p.getValue();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif

  return *this;
}

adouble &adouble::operator*=(const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_mult_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      ADOLC_PUT_VAL(coval);             // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] *= coval;
  return *this;
}

adouble &adouble::operator*=(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_mult_a);
      ADOLC_PUT_LOCINT(a.getLoc());     // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());     // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
      put_op(eq_mult_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      ADOLC_PUT_VAL(coval);             // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] *=
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]);
#endif
  return *this;
}

adouble &adouble::operator*=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(eq_mult_p);
    ADOLC_PUT_LOCINT(p.getLoc());     // = coval
    ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] *= p.getValue();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif

  return *this;
}

adouble &adouble::operator/=(const double coval) {
  *this *= 1.0 / coval;
  return *this;
}

adouble &adouble::operator/=(const adouble &a) {
  *this *= 1.0 / a;
  return *this;
}

/****************************************************************************/
/*                       INCREMENT / DECREMENT                              */

adouble adouble::operator++(int) {
  // create adouble to store old state in it.
  adouble ret_adouble(tape_loc{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(assign_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);       // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {
        const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
        if (coval == 0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        } else if (coval == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
          ADOLC_PUT_VAL(coval);                   // = coval
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(
              ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
      }
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_];
#endif

  // change input adouble to new state
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(incr_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_)]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]++;
  return ret_adouble;
}

adouble adouble::operator--(int) {
  // create adouble to store old state in it.
  adouble ret_adouble(tape_loc{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(assign_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);       // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {
        const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
        if (coval == 0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        } else if (coval == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
          ADOLC_PUT_VAL(coval);                   // = coval
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(
              ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
      }
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_];
#endif

  // write new state into input adouble
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(decr_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]--;
  return ret_adouble;
}

adouble &adouble::operator++() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(incr_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]++;
  return *this;
}

adouble &adouble::operator--() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(decr_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]--;
  return *this;
}

/**************************************************************************
 *           MARK INDEPENDENT AND DEPENDENT
 */

// Assign a double value to an adouble and mark the adouble as independent on
// the tape
adouble &adouble::operator<<=(double input) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ++ADOLC_CURRENT_TAPE_INFOS.numInds;

    put_op(assign_ind);
    ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif
  return *this;
}

// Assign the coval of an adouble to a double reference and mark the adouble as
// dependent variable on the tape. At the end of the function, the double
// reference can be seen as output value of the function given by the trace
// of the adouble.
adouble &adouble::operator>>=(double &output) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
    fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) "
                      "as dependent.\n");
    const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
    if (coval == 0.0) {
      put_op(assign_d_zero);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);
    } else if (coval == 1.0) {
      put_op(assign_d_one);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);
    } else {
      put_op(assign_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);
      ADOLC_PUT_VAL(coval);
    }

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }
#endif
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ++ADOLC_CURRENT_TAPE_INFOS.numDeps;

    put_op(assign_dep);
    ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
  }

  coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
  return *this;
}

void adouble::declareIndependent() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ++ADOLC_CURRENT_TAPE_INFOS.numInds;

    put_op(assign_ind);
    ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif
}

void adouble::declareDependent() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
    fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) "
                      "as dependent.\n");
    const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
    if (coval == 0.0) {
      put_op(assign_d_zero);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);
    } else if (coval == 1.0) {
      put_op(assign_d_one);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);
    } else {
      put_op(assign_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);
      ADOLC_PUT_VAL(coval);
    }

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }
#endif
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ++ADOLC_CURRENT_TAPE_INFOS.numDeps;

    put_op(assign_dep);
    ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
  }
}

/****************************************************************************/
/*                                                           INPUT / OUTPUT */

std::ostream &operator<<(std::ostream &out, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  return out << a.getValue() << "(a)";
}

std::istream &operator>>(std::istream &in, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  double coval;
  in >> coval;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
      if (coval == 0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);         // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] = false;
#endif
  return in;
}

/****************************************************************************/
/*                               COMPARISON                                 */

#ifdef ADOLC_ADVANCED_BRANCHING

adouble operator!=(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  adouble ret_adouble(tape_location{next_loc()});
  const double res = static_cast<double>(a_coval != b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(neq_a_a);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(b.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator!=(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval != b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(neq_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator!=(const adouble &a, adouble &&b) { return std::move(b) != a; }

adouble operator==(const adouble &a, const adouble &b) {
  adouble ret_adouble(tape_location{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval == b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(eq_a_a);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(b.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator==(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval == b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(eq_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator==(const adouble &a, adouble &&b) { return std::move(b) == a; }

adouble operator<=(const adouble &a, const adouble &b) {
  adouble ret_adouble(tape_location{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval <= b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(le_a_a);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(b.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator<=(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval <= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(le_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator<=(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval <= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(le_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(b.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = res;
  return b;
}
adouble operator>=(const adouble &a, const adouble &b) {
  adouble ret_adouble(tape_location{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval >= b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ge_a_a);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(b.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator>=(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval >= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ge_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator>=(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval >= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ge_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(b.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = res;
  return b;
}

adouble operator<(const adouble &a, const adouble &b) {
  adouble ret_adouble(tape_location{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval < b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(lt_a_a);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(b.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator<(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval < b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(lt_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator<(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval < b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(lt_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(b.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = res;
  return b;
}

adouble operator>(const adouble &a, const adouble &b) {
  adouble ret_adouble(tape_location{next_loc()});
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval > b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(gt_a_a);
    ADOLC_PUT_LOCINT(a.getLoc());           // arg
    ADOLC_PUT_LOCINT(b.getLoc());           // arg1
    ADOLC_PUT_VAL(res);                     // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = res;
  return ret_adouble;
}

adouble operator>(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval > b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(gt_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(a.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = res;
  return a;
}

adouble operator>(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  const double b_coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  const double res = static_cast<double>(a_coval > b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(gt_a_a);
    ADOLC_PUT_LOCINT(a.getLoc()); // arg
    ADOLC_PUT_LOCINT(b.getLoc()); // arg1
    ADOLC_PUT_VAL(res);           // check for branch switch
    ADOLC_PUT_LOCINT(b.getLoc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = res;
  return b;
}

#endif // ADOLC_ADVANCED_BRANCHING

bool operator!=(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a != 0);
  else {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] ? neq_zero : eq_zero);
        ADOLC_PUT_LOCIN(Ta.getLoc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return (ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] != 0);
  }
}

bool operator!=(const double coval, const adouble &a) {
  if (coval)
    return (-coval + a != 0);
  else
    return (a != 0);
}

bool operator==(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a == 0);
  else {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] ? neq_zero : eq_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return (ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] == 0);
  }
}

inline bool operator==(const double coval, const adouble &a) {
  if (coval)
    return (-coval + a == 0);
  else
    return (a == 0);
}

bool operator<=(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a <= 0);
  else {
    bool b = (ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] <= 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(b ? le_zero : gt_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return b;
  }
}

inline bool operator<=(const double coval, const adouble &a) {
  if (coval)
    return (-coval + a >= 0);
  else
    return (a >= 0);
}

bool operator>=(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a >= 0);
  else {
    bool b = (ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] >= 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(b ? ge_zero : lt_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return b;
  }
}

bool operator>=(const double coval, const adouble &a) {
  if (coval)
    return (-coval + a <= 0);
  else
    return (a <= 0);
}

bool operator<(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a < 0);
  else {
    bool b = (ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] < 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(b ? lt_zero : ge_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return b;
  }
}

bool operator<(const double coval, const adouble &a) {
  if (coval)
    return (-coval + a > 0);
  else
    return (a > 0);
}

bool operator<(const adouble &a, const double coval) {
  if (coval)
    return (a - coval < 0);
  else
    return (a < 0);
}

bool operator>(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a > 0);
  else {
    bool b = (ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] > 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
        put_op(b ? gt_zero : le_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return b;
  }
}

bool operator>(const double coval, const adouble &a) {
  if (coval)
    return (-coval + a < 0);
  else
    return (a < 0);
}

/****************************************************************************/
/*                           SIGN  OPERATORS                                 */

adouble operator+(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  adouble ret_adouble(tape_location{next_loc()});
  const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif
      put_op(pos_sign_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(pos_sign_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }
  return a;
}

adouble operator-(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(neg_sign_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (-coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (-coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(-coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = -coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(neg_sign_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (-coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (-coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(-coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = -coval;
  return a;
}

/****************************************************************************/
/*                            BINARY OPERATORS                              */

adouble operator+(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] +
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(plus_a_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] +
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(plus_a_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return a;
}

adouble operator+(const adouble &a, adouble &&b) { return std::move(b) + a; }

adouble operator+(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = coval + ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);                   // = coval
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return ret_adouble;
}

adouble operator+(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval + ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());      // = arg
        ADOLC_PUT_LOCINT(a.tape_loc_.loc); // = res
        ADOLC_PUT_VAL(coval);              // = coval
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.tape_loc_.loc]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.tape_loc_.loc]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.tape_loc_.loc);
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT([a.getLoc()]);
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;
  return a;
}

adouble operator+(const adouble &a, const double coval) { return coval + a; }
adouble operator+(adouble &&a, const double coval) {
  return coval + std::move(a);
}

adouble operator-(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] -
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

#endif
      put_op(min_a_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = -ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] -
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

#endif
      put_op(min_a_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = -ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return a;
}

adouble operator-(const adouble &a, adouble &&b) { return -(std::move(b)) + a; }

adouble operator-(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = coval - ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);                   // = coval
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return ret_adouble;
}

adouble operator-(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval - ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);         // = coval
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;

  return a;
}

adouble operator-(const adouble &a, const double coval) { return (-coval) + a; }
adouble operator-(adouble &&a, const double coval) {
  return (-coval) + std::move(a);
}

adouble operator*(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] *
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(mult_a_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      put_op(mult_d_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return ret_adouble;
}

adouble operator*(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] *
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(mult_a_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      put_op(mult_d_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return a;
}

adouble operator*(const adouble &a, adouble &&b) { return std::move(b) * a; }

adouble operator*(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = coval * ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      } else if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);                   // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return ret_adouble;
}

adouble operator*(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval * ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);         // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;
  return a;
}

adouble operator*(const adouble &a, const double coval) { return coval * a; }
adouble operator*(adouble &&a, const double coval) {
  return coval * std::move(a);
}

adouble operator/(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  adouble ret_adouble(tape_location{next_loc()});
  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] /
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(div_a_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = 1.0 / ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.getLoc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return ret_adouble;
}

adouble operator/(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] /
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(div_a_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = 1.0 / ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(a.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return a;
}

adouble operator/(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] /
                        ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(div_a_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(b.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double coval = 1.0 / ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(b.getLoc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.getLoc()); // = arg
        ADOLC_PUT_LOCINT(b.getLoc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.getLoc()); // = arg
        ADOLC_PUT_LOCINT(b.getLoc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg
      ADOLC_PUT_LOCINT(b.getLoc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif
  return b;
}

adouble operator/(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});

  cosnt double coval2 = coval / ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);                   // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  return ret_adouble;
}

adouble operator/(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  cosnt double coval2 = coval / ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);         // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  return a;
}

adouble operator/(const adouble &a, const double coval) {
  return a * 1.0 / coval;
}
adouble operator/(adouble &&a, const double coval) { return a * 1.0 / coval; }
/****************************************************************************/
/*                          UNARY OPERATIONS                                */

adouble exp(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::exp(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(exp_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble exp(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::exp(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(exp_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble log(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_locaton{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::log(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(log_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble log(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::log(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(log_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble log10(const adouble &a) { return log(a) / ADOLC_MATH_NSP::log(10.0); }
adouble log10(adouble &&a) {
  return log(std::move(a)) / ADOLC_MATH_NSP::log(10.0);
}

adouble sqrt(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::sqrt(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(sqrt_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble sqrt(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::sqrt(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(sqrt_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble cbrt(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::cbrt(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(cbrt_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble cbrt(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::cbrt(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(cbrt_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble sin(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});

  const double coval1 =
      ADOLC_MATH_NSP::sin(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  const double coval2 =
      ADOLC_MATH_NSP::cos(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(sin_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(
              ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.getLoc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
      }
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval1;
  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble sin(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble a(tape_location{next_loc()});

  const double coval1 =
      ADOLC_MATH_NSP::sin(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  const double coval2 =
      ADOLC_MATH_NSP::cos(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(sin_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(a.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(a.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(a.getLoc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.getLoc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
      }
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval1;
  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble cos(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});

  const double coval1 =
      ADOLC_MATH_NSP::cos(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  const double coval2 =
      ADOLC_MATH_NSP::sin(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(cos_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.getLoc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(
              ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.getLoc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
      }
    }

#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval1;
  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble cos(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval1 =
      ADOLC_MATH_NSP::cos(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  const double coval2 =
      ADOLC_MATH_NSP::sin(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(cos_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(a.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(a.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(a.getLoc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.getLoc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
      }
    }

#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval1;
  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble tan(const adouble &a) { return sin(a) / cos(a); }
adouble tan(adouble &&a) { return sin(std::move(a)) / cos(std::move(a)); }

adouble asin(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::asin(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / sqrt(1.0 - x * x);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(asin_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble asin(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::asin(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / sqrt(1.0 - x * x);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(asin_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble acos(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::acos(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble y = -1.0 / sqrt(1.0 - x * x);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(acos_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble acos(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::acos(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble y = -1.0 / sqrt(1.0 - x * x);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(acos_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble atan(const adouble &a) {

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::atan(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble y = 1.0 / (1.0 + x * x);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(atan_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble atan(adouble &&a) {

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::atan(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  adouble y = 1.0 / (1.0 + x * x);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(atan_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble sinh(const adouble &a) {
  if (x.getValue() < 0.0) {
    adouble temp = exp(a);
    return 0.5 * (temp - 1.0 / temp);
  } else {
    adouble temp = exp(-a);
    return 0.5 * (1.0 / temp - temp);
  }
}

adouble sinh(adouble &&a) {
  if (a.getValue() < 0.0) {
    adouble temp = exp(std::move(a));
    return 0.5 * (temp - 1.0 / temp);
  } else {
    adouble temp = exp(-std::move(a));
    return 0.5 * (1.0 / temp - temp);
  }
}

adouble cosh(const adouble &a) {
  adouble temp = (a.getValue() < 0.0) ? exp(a) : exp(-a);
  return 0.5 * (temp + 1.0 / temp);
}

adouble cosh(adouble &&a) {
  adouble temp = (a.getValue() < 0.0) ? exp(std::move(a)) : exp(-std::move(a));
  return 0.5 * (temp + 1.0 / temp);
}

adouble tanh(const adouble &a) {
  if (a.getValue() < 0.0) {
    adouble temp = exp(2.0 * a);
    return (temp - 1.0) / (temp + 1.0);
  } else {
    adouble temp = exp((-2.0) * a);
    return (1.0 - temp) / (temp + 1.0);
  }
}

adouble tanh(adouble &&a) {
  if (a.getValue() < 0.0) {
    adouble temp = exp(2.0 * std::move(a));
    return (temp - 1.0) / (temp + 1.0);
  } else {
    adouble temp = exp((-2.0) * std::move(a));
    return (1.0 - temp) / (temp + 1.0);
  }
}

adouble asinh(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP_ERF::asinh(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(asinh_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble asinh(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP_ERF::asinh(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(asinh_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble acosh(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP_ERF::acosh(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(acosh_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble acosh(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP_ERF::acosh(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(acosh_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble atanh(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP_ERF::atanh(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / (1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(atanh_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble atanh(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP_ERF::atanh(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b = 1.0 / (1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(atanh_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble erf(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adoubel ret_adouble(tape_ret_adouble.getLoc() ion{next_loc()});
  const double coval =
      ADOLC_MATH_NSP_ERF::erf(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(erf_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble erf(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP_ERF::erf(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(erf_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble erfc(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP_ERF::erfc(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(erfc_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble erfc(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP_ERF::erfc(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(erfc_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble ceil(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::ceil(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(ceil_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);                   // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return ret_adouble;
}

adouble ceil(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::ceil(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(ceil_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);         // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return a;
}

adouble floor(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval =
      ADOLC_MATH_NSP::floor(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(floor_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);                   // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  return ret_adouble;
}

adouble floor(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval =
      ADOLC_MATH_NSP::floor(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(floor_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);         // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif
  return a;
}

adouble fabs(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_loc()});
  const double coval = 1.0;
  const double temp =
      ADOLC_MATH_NSP::fabs(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (temp != ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()])
    coval = 0.0;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(abs_val);
      ADOLC_PUT_LOCINT(a.getLoc());           // arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // res
      ADOLC_PUT_VAL(coval);                   // coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
        ++ADOLC_CURRENT_TAPE_INFOS.numSwitches;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = temp;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return ret_adouble;
}

adouble fabs(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double temp =
      ADOLC_MATH_NSP::fabs(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

  if (temp != ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()])
    const double coval = 0.0;
  else
    const double coval = 1.0;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(abs_val);
      ADOLC_PUT_LOCINT(a.getLoc()); // arg
      ADOLC_PUT_LOCINT(a.getLoc()); // res
      ADOLC_PUT_VAL(coval);         // coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
        ++ADOLC_CURRENT_TAPE_INFOS.numSwitches;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }
  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = temp;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  return a;
}

adouble fmin(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double temp = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
    }
  }
#endif

  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] <
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]) {
    const double coval = 0.0;
    const double tmp = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  } else {
    const double coval = 1.0;
    const double tmp = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  }

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(min_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg1
      ADOLC_PUT_LOCINT(b.getLoc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);                   // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (tmp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (tmp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(tmp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = tmp;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif

  return ret_adouble;
}

adouble fmin(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double temp = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
    }
  }
#endif

  if (ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] <
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]) {
    const double coval = 0.0;
    const double tmp = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  } else {
    const double coval = 1.0;
    const double tmp = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  }

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(min_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);         // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (tmp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (tmp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(tmp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = tmp;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif

  return a;
}

adouble fmin(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      const double temp = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      const double temp = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.getLoc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
    }
  }
#endif

  if (ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] <
      ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]) {
    const double coval = 0.0;
    const double tmp = ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()];
  } else {
    const double coval = 1.0;
    const double tmp = ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()];
  }

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {
#endif

      put_op(min_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(b.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(b.getLoc()); // = res
      ADOLC_PUT_VAL(coval);         // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]) {

      if (tmp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else if (tmp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.getLoc());
        ADOLC_PUT_VAL(tmp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[b.getLoc()] = tmp;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.getLoc()]);
#endif

  return b;
}

adouble fmin(const double coval, const adouble &a) {
  return fmin(adouble(coval), a);
}

adouble fmin(const double coval, adouble &&a) {
  return fmin(adouble(coval), std::move(a));
}

adouble fmin(const adouble &a, const double coval) {
  return (fmin(a, adouble(coval)));
}

adouble fmin(adouble &&a, const double coval) {
  return (fmin(std::move(a), adouble(coval)));
}

adouble fmax(const adouble &a, const adouble &b) { return (-fmin(-a, -b)); }
adouble fmax(adouble &&a, const adouble &b) {
  return (-fmin(-std::move(a), -b));
}
adouble fmax(const adouble &a, adouble &&b) {
  return (-fmin(-a, -std::move(b)));
}

adouble fmax(const double coval, const adouble &a) {
  return (-fmin(-coval, -a));
}

adouble fmax(const double coval, adouble &&a) {
  return (-fmin(-coval, -std::move(a)));
}

adouble fmax(const adouble &a, const double coval) {
  return (-fmin(-a, -coval));
}
adouble fmax(adouble &&a, const double coval) {
  return (-fmin(-std::move(a), -coval));
}

adouble ldexp(const adouble &a, const int exp) { return a * ldexp(1.0, exp); }
adouble ldexp(adouble &&a, const int exp) { return a * ldexp(1.0, exp); }

adouble frexp(const adouble &a, int *exp) {
  double coval = std::frexp(a.getValue(), exp);
  return adouble(coval);
}

adouble frexp(adouble &&a, int *exp) {
  double coval = std::frexp(a.getValue(), exp);
  adouble.setValue(coval);
  return a;
}

adouble atan2(const adouble &a, const adouble &b) {
  adouble a1, a2, ret, sy;
  const double pihalf = ADOLC_MATH_NSP::asin(1.0);
  condassign(sy, a, adouble{1.0}, adouble{-1.0});
  condassign(a1, a, atan(b / a), atan(b / a) + sy * 2 * pihalf);
  condassign(a2, fabs(b), sy * pihalf - atan(a / b), adouble{0.0});
  condassign(ret, fabs(a) - fabs(b), a1, a2);
  return ret;
}

adouble pow(const adouble &a, const adouble &b) {
  assert((a.getValue() < 0) &&
         "\nADOL-C message: negative basis deactivated\n ");
  assert(a.getValue() == 0 && "\nADOL-C message: zero basis deactivated\n ");

  adouble a1, a2, ret;

  condassign(a1, -b, adouble{ADOLC_MATH_NSP::pow(a.getValue(), b.getValue())},
             pow(a, b.getValue()));
  condassign(a2, fabs(a), pow(a, b.getValue()), a1);
  condassign(ret, a, exp(b * log(a)), a2);
  return ret;
}

adouble pow(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble(tape_location{next_next()});
  const double coval2 =
      ADOLC_MATH_NSP::pow(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()], coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(pow_op);
      ADOLC_PUT_LOCINT(a.getLoc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.getLoc()); // = res
      ADOLC_PUT_VAL(coval);                   // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble pow(adouble &&a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 =
      ADOLC_MATH_NSP::pow(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()], coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {
#endif

      put_op(pow_op);
      ADOLC_PUT_LOCINT(a.getLoc()); // = arg
      ADOLC_PUT_LOCINT(a.getLoc()); // = res
      ADOLC_PUT_VAL(coval);         // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.getLoc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.getLoc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] = coval2;

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.getLoc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble pow(const double base, const adouble &a) {
  assert(coval <= 0 && "\nADOL-C message:  exponent at zero/negative constant "
                       "basis deactivated\n");
  adouble ret;

  condassign(ret, adouble{coval}, exp(a * ADOLC_MATH_NSP::log(coval)),
             adouble{ADOLC_MATH_NSP::pow(coval, a.getValue())});
  return ret;
}

/*--------------------------------------------------------------------------*/
/* Macro for user defined quadratures, example myquad is below.*/
/* the forward sweep tests if the tape is executed exactly at  */
/* the same argument point otherwise it stops with a returnval */
#define extend_quad(func, integrand)                                           \
  adouble func(const adouble &arg) {                                           \
    adouble temp;                                                              \
    adouble val;                                                               \
    integrand;                                                                 \
    ADOLC_OPENMP_THREAD_NUMBER;                                                \
    ADOLC_OPENMP_GET_THREAD_NUMBER;                                            \
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {                                  \
      put_op(gen_quad);                                                        \
      ADOLC_PUT_LOCINT(arg.loc());                                             \
      ADOLC_PUT_LOCINT(val.loc());                                             \
      ADOLC_PUT_LOCINT(temp.loc());                                            \
      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;                                 \
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)                                \
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[temp.loc()]);         \
    }                                                                          \
    ADOLC_GLOBAL_TAPE_VARS.store[temp.loc()] =                                 \
        func(ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()]);                         \
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {                                  \
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[arg.loc()]);                  \
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[temp.loc()]);                 \
    }                                                                          \
    return temp;                                                               \
  }

double myquad(double x) {
  double res;
  res = ADOLC_MATH_NSP::log(x);
  return res;
}

/* This defines the natural logarithm as a quadrature */

extend_quad(myquad, val = 1 / arg);

void condassign(double &res, const double &cond, const double &arg1,
                const double &arg2) {
  res = cond > 0 ? arg1 : arg2;
}

void condassign(double &res, const double &cond, const double &arg) {
  res = cond > 0 ? arg : res;
}

void condeqassign(double &res, const double &cond, const double &arg1,
                  const double &arg2) {
  res = cond >= 0 ? arg1 : arg2;
}

void condeqassign(double &res, const double &cond, const double &arg) {
  res = cond >= 0 ? arg : res;
}

void condassign(adouble &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.getLoc()]) {

        const double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg1.getLoc()];
        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg1.getLoc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg1.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg1.getLoc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[arg1.getLoc()]);
      }

      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.getLoc()]) {

        const double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg2.getLoc()];

        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg2.getLoc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg2.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg2.getLoc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[arg2.getLoc()]);
      }
#endif

      put_op(cond_assign);
      ADOLC_PUT_LOCINT(cond.getLoc()); // = arg
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()]);
      ADOLC_PUT_LOCINT(arg1.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(arg2.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(res.getLoc());  // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.store[cond.loc()] > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (ADOLC_GLOBAL_TAPE_VARS.actStore[arg_loc]) {

        put_op(assign_a);
        ADOLC_PUT_LOCINT(arg_loc);      // = arg
        ADOLC_PUT_LOCINT(res.getLoc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

      } else {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()]) {

          const double coval = ADOLC_GLOBAL_TAPE_VARS.store[arg_loc];
          if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(res.getLoc()); // = res
          } else if (coval == 1.0) {
            put_op(assign_d_one);
            ADOLC_PUT_LOCINT(res.getLoc()); // = res
          } else {
            put_op(assign_d);
            ADOLC_PUT_LOCINT(res.getLoc()); // = res
            ADOLC_PUT_VAL(coval);           // = coval
          }

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);
        }
      }
    }
#endif
  }

  if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
    ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.store[arg1.getLoc()];

  else
    ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.store[arg2.getLoc()];

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.getLoc()];

    else
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.getLoc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()];

#endif
}

void condassign(adouble &res, const adouble &cond, const adouble &arg) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg.getLoc()]) {

        const double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()];
        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg.getLoc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg.getLoc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tmploc]);
      }

#endif

      put_op(cond_assign_s);
      ADOLC_PUT_LOCINT(cond.getLoc()); // = arg
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()]);
      ADOLC_PUT_LOCINT(arg.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(res.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0) {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[arg.getLoc()]) {

          put_op(assign_a);
          ADOLC_PUT_LOCINT(arg.getLoc()); // = arg
          ADOLC_PUT_LOCINT(res.getLoc()); // = res

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

        } else {
          if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()]) {
            const double coval = ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()];

            if (coval == 0) {
              put_op(assign_d_zero);
              ADOLC_PUT_LOCINT(res.getLoc()); // = res
            } else if (coval == 1.0) {
              put_op(assign_d_one);
              ADOLC_PUT_LOCINT(res.getLoc()); // = res
            } else {
              put_op(assign_d);
              ADOLC_PUT_LOCINT(res.getLoc()); // = res
              ADOLC_PUT_VAL(coval);           // = coval
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
              ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);
          }
        }
      }
    }
#endif
  }

  if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
    ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()];

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg.getLoc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()];

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.getLoc()]) {

        const double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg1.getLoc()];

        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg1.getLoc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg1.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg1.getLoc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[arg1.getLoc()]);
      }

      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.getLoc()]) {
        const double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg2.getLoc()];

        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg2.getLoc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg2.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg2.getLoc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[arg2.getLoc()]);
      }
#endif

      put_op(cond_eq_assign);
      ADOLC_PUT_LOCINT(cond.getLoc()); // = arg
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()]);
      ADOLC_PUT_LOCINT(arg1.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(arg2.getLoc()); // = arg2
      ADOLC_PUT_LOCINT(res.getLoc());  // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
        size_t arg_loc = arg1.getLoc();
      else
        size_t arg_loc = arg2.getLoc();

      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {

        put_op(assign_a);
        ADOLC_PUT_LOCINT(arg_loc);      // = arg
        ADOLC_PUT_LOCINT(res.getLoc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

      } else {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()]) {

          const double coval = ADOLC_GLOBAL_TAPE_VARS.store[arg_loc];

          if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(res.getLoc()); // = res
          } else if (coval == 1.0) {
            put_op(assign_d_one);
            ADOLC_PUT_LOCINT(res.getLoc()); // = res
          } else {
            put_op(assign_d);
            ADOLC_PUT_LOCINT(res.getLoc()); // = res
            ADOLC_PUT_VAL(coval);           // = coval
          }

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);
        }
      }
    }
#endif
  }

  if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] >= 0)
    ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.store[arg1.getLoc()];

  else
    ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.store[arg2.getLoc()];

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.getLoc()];

    else
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.getLoc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()];

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg.getLoc()]) {

        const double temp = ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()];
        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg.getLoc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg.getLoc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg.getLoc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()]);
      }
#endif

      put_op(cond_eq_assign_s);
      ADOLC_PUT_LOCINT(cond.getLoc()); // = arg
      ADOLC_PUT_VAL(ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()]);
      ADOLC_PUT_LOCINT(arg.getLoc()); // = arg1
      ADOLC_PUT_LOCINT(res.getLoc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0) {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {

          put_op(assign_a);
          ADOLC_PUT_LOCINT(arg.getLoc()); // = arg
          ADOLC_PUT_LOCINT(res.getLoc()); // = res

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);

        } else {
          if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()]) {

            const double coval = ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()];

            if (coval == 0) {
              put_op(assign_d_zero);
              ADOLC_PUT_LOCINT(res.getLoc()); // = res
            } else if (coval == 1.0) {
              put_op(assign_d_one);
              ADOLC_PUT_LOCINT(res.getLoc()); // = res
            } else {
              put_op(assign_d);
              ADOLC_PUT_LOCINT(res.getLoc()); // = res
              ADOLC_PUT_VAL(coval);           // = coval
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
              ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()]);
          }
        }
      }
    }
#endif
  }

  if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] >= 0)
    ADOLC_GLOBAL_TAPE_VARS.store[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.store[arg.getLoc()];

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()]) {
    if (ADOLC_GLOBAL_TAPE_VARS.store[cond.getLoc()] > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg.getLoc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.getLoc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.getLoc()];

#endif
}

void adolc_vec_copy(adouble *const dest, const adouble *const src,
                    size_t size) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (dest[size - 1].getLoc() - dest[0].getLoc() != size - 1 ||
      src[size - 1].getLoc() - src[0].getLoc() != size - 1)
    fail(ADOLC_VEC_LOCATIONGAP);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(vec_copy);
    ADOLC_PUT_LOCINT(src[0].getLoc());
    ADOLC_PUT_LOCINT(size);
    ADOLC_PUT_LOCINT(dest[0].getLoc());

    for (size_t i = 0; i < size; ++i) {
      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[dest[0].getLoc() + i]);
    }
  }
  for (size_t i = 0; i < size; ++i)
    ADOLC_GLOBAL_TAPE_VARS.store[dest[0].getLoc() + i] =
        ADOLC_GLOBAL_TAPE_VARS.store[src[0].getLoc() + i];
}

// requires a and b to be of size "size"
adouble adolc_vec_dot(const adouble *const vec_a, const adouble *const vec_b,
                      size_t size) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (vec_a[size - 1].getLoc() - vec_a[0].getLoc() != size - 1 ||
      vec_b[size - 1].getLoc() - vec_b[0].getLoc() != size - 1)
    fail(ADOLC_VEC_LOCATIONGAP);

  adouble ret_adouble(tape_location{next_loc()});

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(vec_dot);
    ADOLC_PUT_LOCINT(a[0].getLoc());
    ADOLC_PUT_LOCINT(b[0].getLoc());
    ADOLC_PUT_LOCINT(size);
    ADOLC_PUT_LOCINT(ret_adouble.getLoc());

    ADOLC_CURRENT_TAPE_INFOS.num_eq_prod += 2 * size;

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()]);
  }

  ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] = 0;

  for (size_t i = 0; i < size; ++i)
    ADOLC_GLOBAL_TAPE_VARS.store[ret_adouble.getLoc()] +=
        ADOLC_GLOBAL_TAPE_VARS.store[a[0].getLoc() + i] *
        ADOLC_GLOBAL_TAPE_VARS.store[b[0].getLoc() + i];

  return ret_adouble;
}

// requires res, b and c to be of size "size"
void adolc_vec_axpy(adouble *const res, const adouble &a,
                    const adouble *const vec_a, const adouble *const vec_b,
                    size_t size) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (res[size - 1].getLoc() - res[0].getLoc() != size - 1 ||
      vec_a[size - 1].getLoc() - vec_a[0].getLoc() != size - 1 ||
      vec_b[size - 1].getLoc() - vec_b[0].getLoc() != size - 1)
    fail(ADOLC_VEC_LOCATIONGAP);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(vec_axpy);
    ADOLC_PUT_LOCINT(a.getLoc());
    ADOLC_PUT_LOCINT(vec_a[0].getLoc());
    ADOLC_PUT_LOCINT(vec_b[0].getLoc());
    ADOLC_PUT_LOCINT(size);
    ADOLC_PUT_LOCINT(res[0].getLoc());
    ADOLC_CURRENT_TAPE_INFOS.num_eq_prod += 2 * size - 1;

    for (size_t i = 0; i < size; ++i) {
      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res[0].getLoc() + i]);
    }
  }
  for (size_t i = 0; i < size; ++i)
    ADOLC_GLOBAL_TAPE_VARS.store[res[0].getLoc() + i] =
        ADOLC_GLOBAL_TAPE_VARS.store[a.getLoc()] *
            ADOLC_GLOBAL_TAPE_VARS.store[vec_a[0].getLoc() + i] +
        ADOLC_GLOBAL_TAPE_VARS.store[vec_b[0].getLoc() + i];
}
