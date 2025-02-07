/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adouble.cpp
 Revision: $Id$
 Contents: This file specifies the definitions of the member functions of
           adouble and various arithmetic function that work on adoubles.

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adtb_types.h>
#include <adolc/oplate.h>
#include <cassert>

adouble::adouble() {
  tape_loc_ = tape_location{next_loc()};

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
  tape_loc_ = tape_location{next_loc()};
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

adouble::adouble(const adouble &a) {
  tape_loc_ = tape_location{next_loc()};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
      put_op(assign_a);
      ADOLC_PUT_LOCINT(a.loc());        // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
        const double coval = a.value();
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

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = a.value();
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
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
  if (tape_loc_.loc_ != a.loc()) {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(assign_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
          double coval = a.value();
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
    ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = a.value();
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  }
  return *this;
}

adouble &adouble::operator=(const pdouble &p) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(assign_p);
    ADOLC_PUT_LOCINT(p.loc());
    ADOLC_PUT_LOCINT(tape_loc_.loc_);

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
  }
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = p.value();

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] = true;
#endif
  return *this;
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
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_plus_a);
      ADOLC_PUT_LOCINT(a.loc());        // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(assign_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
      const double coval = a.value();
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
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] += a.value();
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]);
#endif
  return *this;
}

adouble &adouble::operator+=(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  int upd = 0;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()])
#endif
    {
      // if the structure is a*=b*c the call optimizes the temp adouble away.
      upd = upd_resloc_inc_prod(a.loc(), loc(), eq_min_prod);
    }
  if (upd) {
    ADOLC_GLOBAL_TAPE_VARS.store[loc()] +=
        ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_DELETE_SCAYLOR(&ADOLC_GLOBAL_TAPE_VARS.store[a.loc()]);
    --ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    ++ADOLC_CURRENT_TAPE_INFOS.num_eq_prod;
  } else {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
          ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_plus_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(loc());   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
        double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
        if (coval) {
          put_op(plus_d_a);
          ADOLC_PUT_LOCINT(a.loc()); // = arg
          ADOLC_PUT_LOCINT(loc());   // = res
          ADOLC_PUT_VAL(coval);
        } else {
          put_op(assign_a);
          ADOLC_PUT_LOCINT(a.loc()); // = arg
          ADOLC_PUT_LOCINT(loc());   // = res
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
        double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
        if (coval) {
          put_op(eq_plus_d);
          ADOLC_PUT_LOCINT(loc()); // = res
          ADOLC_PUT_VAL(coval);    // = coval

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
        }
      }
#endif
    }
    ADOLC_GLOBAL_TAPE_VARS.store[loc()] +=
        ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] =
        (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] ||
         ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]);
#endif
  }
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
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_min_a);
      ADOLC_PUT_LOCINT(a.loc());        // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
      const double coval = a.value();
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
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] -= a.value();
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]);
#endif
  return *this;
}

adouble &adouble::operator-=(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  int upd = 0;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()])
#endif
    {
      upd = upd_resloc_inc_prod(a.loc(), loc(), eq_min_prod);
    }
  if (upd) {
    ADOLC_GLOBAL_TAPE_VARS.store[loc()] -=
        ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_DELETE_SCAYLOR(&ADOLC_GLOBAL_TAPE_VARS.store[a.loc()]);
    --ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    ++ADOLC_CURRENT_TAPE_INFOS.num_eq_prod;
  } else {
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
          ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
#endif
        put_op(eq_min_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(loc());   // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
        double coval = ADOLC_GLOBAL_TAPE_VARS.store[loc()];
        if (coval) {
          put_op(min_d_a);
          ADOLC_PUT_LOCINT(a.loc()); // = arg
          ADOLC_PUT_LOCINT(loc());   // = res
          ADOLC_PUT_VAL(coval);
        } else {
          put_op(neg_sign_a);
          ADOLC_PUT_LOCINT(a.loc()); // = arg
          ADOLC_PUT_LOCINT(loc());   // = res
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);

      } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()]) {
        double coval = ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
        if (coval) {
          put_op(eq_min_d);
          ADOLC_PUT_LOCINT(loc()); // = res
          ADOLC_PUT_VAL(coval);    // = coval

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc()]);
        }
      }
#endif
    }
    ADOLC_GLOBAL_TAPE_VARS.store[loc()] -=
        ADOLC_GLOBAL_TAPE_VARS.store[a.loc()];
#if defined(ADOLC_TRACK_ACTIVITY)
    ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] =
        (ADOLC_GLOBAL_TAPE_VARS.actStore[loc()] ||
         ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]);
#endif
  }

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
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(eq_mult_a);
      ADOLC_PUT_LOCINT(a.loc());        // = arg
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.loc());        // = arg
        ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
      const double coval = a.value();
      put_op(eq_mult_d);
      ADOLC_PUT_LOCINT(tape_loc_.loc_); // = res
      ADOLC_PUT_VAL(coval);             // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] *= a.value();
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]);
#endif
  return *this;
}
/****************************************************************************/
/*                       INCREMENT / DECREMENT                              */

adouble adouble::operator++(int) {
  // create adouble to store old state in it.
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(assign_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);    // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {
        const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
        if (coval == 0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        } else if (coval == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
          ADOLC_PUT_VAL(coval);                // = coval
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ret_adouble.value());
      }
    }
#endif
  }

  ret_adouble.value(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
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
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]++;
  return ret_adouble;
}

adouble adouble::operator--(int) {
  // create adouble to store old state in it.
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[tape_loc_.loc_]) {
#endif
      put_op(assign_a);
      ADOLC_PUT_LOCINT(tape_loc_.loc_);    // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {
        const double coval = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
        if (coval == 0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        } else if (coval == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
          ADOLC_PUT_VAL(coval);                // = coval
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ret_adouble.value());
      }
    }
#endif
  }

  ret_adouble.value(ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_]);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
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
adouble &adouble::operator<<=(const double input) {
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
  ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_] = input;
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

  output = ADOLC_GLOBAL_TAPE_VARS.store[tape_loc_.loc_];
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
  return out << a.value() << "(a)";
}

std::istream &operator>>(std::istream &in, adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  double coval;
  in >> coval;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
      if (coval == 0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);      // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  a.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] = false;
#endif
  return in;
}

/****************************************************************************/
/*                               COMPARISON                                 */

#ifdef ADOLC_ADVANCED_BRANCHING

adouble operator!=(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  adouble ret_adouble{tape_location{next_loc()}};
  const double res = static_cast<double>(a_coval != b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(neq_a_a);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(b.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator!=(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval != b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(neq_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator!=(const adouble &a, adouble &&b) { return std::move(b) != a; }

adouble operator==(const adouble &a, const adouble &b) {
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval == b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(eq_a_a);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(b.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator==(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval == b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(eq_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator==(const adouble &a, adouble &&b) { return std::move(b) == a; }

adouble operator<=(const adouble &a, const adouble &b) {
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(le_a_a);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(b.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<=(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(le_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator<=(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(le_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(b.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(b.value());
  }

  b.value(res);
  return b;
}
adouble operator>=(const adouble &a, const adouble &b) {
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ge_a_a);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(b.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>=(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ge_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>=(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ge_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(b.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(b.value());
  }

  b.value(res);
  return b;
}

adouble operator<(const adouble &a, const adouble &b) {
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(lt_a_a);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(b.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(lt_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator<(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(lt_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(b.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(b.value());
  }

  b.value(res);
  return b;
}

adouble operator>(const adouble &a, const adouble &b) {
  adouble ret_adouble{tape_location{next_loc()}};
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(gt_a_a);
    ADOLC_PUT_LOCINT(a.loc());           // arg
    ADOLC_PUT_LOCINT(b.loc());           // arg1
    ADOLC_PUT_VAL(res);                  // check for branch switch
    ADOLC_PUT_LOCINT(ret_adouble.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(gt_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(a.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(gt_a_a);
    ADOLC_PUT_LOCINT(a.loc()); // arg
    ADOLC_PUT_LOCINT(b.loc()); // arg1
    ADOLC_PUT_VAL(res);        // check for branch switch
    ADOLC_PUT_LOCINT(b.loc()); // res

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(b.value());
  }

  b.value(res);
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
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(a.value() ? neq_zero : eq_zero);
        ADOLC_PUT_LOCINT(a.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return (a.value() != 0);
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
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(a.value() ? neq_zero : eq_zero);
        ADOLC_PUT_LOCINT(a.loc());
#if defined(ADOLC_TRACK_ACTIVITY)
      }
#endif
    }
    return (a.value() == 0);
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
    bool b = (a.value() <= 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(b ? le_zero : gt_zero);
        ADOLC_PUT_LOCINT(a.loc());
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
    bool b = (a.value() >= 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(b ? ge_zero : lt_zero);
        ADOLC_PUT_LOCINT(a.loc());
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
    bool b = (a.value() < 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(b ? lt_zero : ge_zero);
        ADOLC_PUT_LOCINT(a.loc());
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

bool operator>(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (coval)
    return (-coval + a > 0);
  else {
    bool b = (a.value() > 0);
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
        put_op(b ? gt_zero : le_zero);
        ADOLC_PUT_LOCINT(a.loc());
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
  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif
      put_op(pos_sign_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(pos_sign_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }
  return a;
}

adouble operator-(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(neg_sign_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (-coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (-coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(-coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(-coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(neg_sign_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (-coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (-coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(-coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }
  a.value(-coval);
  return a;
}

/****************************************************************************/
/*                            BINARY OPERATORS                              */

adouble operator+(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = a.value() + b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(plus_a_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(b.value());

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = a.value() + b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(plus_a_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(b.value());

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return a;
}

adouble operator+(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = coval + a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);                // = coval
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  return ret_adouble;
}

adouble operator+(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval + a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      if (coval) {
        put_op(plus_d_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);      // = coval
      } else {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT([a.loc()]);
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);
  return a;
}

adouble operator-(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = a.value() - b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

#endif
      put_op(min_a_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = -b.value();
      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();
      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = a.value() - b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

#endif
      put_op(min_a_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = -b.value();
      put_op(plus_d_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();
      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return a;
}

adouble operator-(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = coval - a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);                // = coval
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  return ret_adouble;
}

adouble operator-(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval - a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      if (coval) {
        put_op(min_d_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);      // = coval
      } else {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);

  return a;
}

adouble operator*(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = a.value() * b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(mult_a_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = b.value();
      put_op(mult_d_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
      const double coval = a.value();

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return ret_adouble;
}

adouble operator*(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = a.value() * b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(mult_a_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = b.value();
      put_op(mult_d_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
      const double coval = a.value();

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return a;
}

adouble operator*(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = coval * a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      } else if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);                // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  return ret_adouble;
}

adouble operator*(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval * a.value();
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);      // = coval
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[locat]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }
  a.value(coval2);
  return a;
}

adouble operator/(const adouble &a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = a.value() / b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(div_a_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.loc());           // = arg
        ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(b.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return ret_adouble;
}

adouble operator/(adouble &&a, const adouble &b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = a.value() / b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(div_a_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(a.loc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(b.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return a;
}

adouble operator/(const adouble &a, adouble &&b) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = a.value() / b.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(div_a_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(b.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        put_op(neg_sign_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(b.loc()); // = res
      } else if (coval == 1.0) {
        put_op(pos_sign_a);
        ADOLC_PUT_LOCINT(b.loc()); // = arg
        ADOLC_PUT_LOCINT(b.loc()); // = res
      } else {
        put_op(mult_d_a);
        ADOLC_PUT_LOCINT(a.loc()); // = arg
        ADOLC_PUT_LOCINT(b.loc()); // = res
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double coval = a.value();

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(b.loc()); // = arg
      ADOLC_PUT_LOCINT(b.loc()); // = res
      ADOLC_PUT_VAL(coval);

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());
    }
#endif
  }

  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif
  return b;
}

adouble operator/(const double coval, const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  const double coval2 = coval / a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);                // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  return ret_adouble;
}

adouble operator/(const double coval, adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = coval / a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(div_d_a);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);      // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  return a;
}

/****************************************************************************/
/*                          UNARY OPERATIONS                                */

adouble exp(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::exp(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(exp_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble exp(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::exp(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(exp_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble log(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::log(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(log_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble log(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::log(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(log_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble sqrt(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::sqrt(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(sqrt_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble sqrt(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::sqrt(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(sqrt_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble cbrt(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::cbrt(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(cbrt_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble cbrt(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::cbrt(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(cbrt_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble sin(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  const double coval1 = ADOLC_MATH_NSP::sin(a.value());
  const double coval2 = ADOLC_MATH_NSP::cos(a.value());

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(sin_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(b.value());
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.loc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ret_adouble.value());
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.loc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(b.value());
      }
    }
#endif
  }

  ret_adouble.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble sin(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval1 = ADOLC_MATH_NSP::sin(a.value());
  const double coval2 = ADOLC_MATH_NSP::cos(a.value());

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(sin_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(b.value());
        ADOLC_WRITE_SCAYLOR(a.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(a.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(a.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(a.loc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(a.value());
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.loc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(b.value());
      }
    }
#endif
  }

  a.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble cos(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};

  const double coval1 = ADOLC_MATH_NSP::cos(a.value());
  const double coval2 = ADOLC_MATH_NSP::sin(a.value());

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(cos_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(b.value());
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(ret_adouble.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(ret_adouble.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(ret_adouble.loc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(ret_adouble.value());
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.loc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(b.value());
      }
    }

#endif
  }

  ret_adouble.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble cos(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval1 = ADOLC_MATH_NSP::cos(a.value());
  const double coval2 = ADOLC_MATH_NSP::sin(a.value());

  adouble b;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(cos_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += 2;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        ADOLC_WRITE_SCAYLOR(b.value());
        ADOLC_WRITE_SCAYLOR(a.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

        if (coval1 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(a.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(a.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(a.loc());
          ADOLC_PUT_VAL(coval1);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(a.value());
      }
      if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

        if (coval2 == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(b.loc());
        } else if (coval1 == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(b.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(b.loc());
          ADOLC_PUT_VAL(coval2);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(b.value());
      }
    }

#endif
  }

  a.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble asin(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::asin(a.value());
  adouble b = 1.0 / sqrt(1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(asin_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble asin(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::asin(a.value());
  adouble b = 1.0 / sqrt(1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(asin_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble acos(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::acos(a.value());

  adouble b = -1.0 / sqrt(1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(acos_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble acos(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::acos(a.value());

  adouble b = -1.0 / sqrt(1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(acos_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble atan(const adouble &a) {

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::atan(a.value());

  adouble b = 1.0 / (1.0 + a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(atan_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble atan(adouble &&a) {

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::atan(a.value());

  adouble b = 1.0 / (1.0 + a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(atan_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble asinh(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP_ERF::asinh(a.value());
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(asinh_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble asinh(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP_ERF::asinh(a.value());
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(asinh_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble acosh(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP_ERF::acosh(a.value());
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(acosh_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble acosh(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP_ERF::acosh(a.value());
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(acosh_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble atanh(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP_ERF::atanh(a.value());
  adouble b = 1.0 / (1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(atanh_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble atanh(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP_ERF::atanh(a.value());
  adouble b = 1.0 / (1.0 - a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(atanh_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble erf(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP_ERF::erf(a.value());
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(erf_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble erf(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP_ERF::erf(a.value());
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(erf_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble erfc(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP_ERF::erfc(a.value());
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(erfc_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble erfc(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP_ERF::erfc(a.value());
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input here
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(erfc_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
}

adouble ceil(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::ceil(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(ceil_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);                // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  return ret_adouble;
}

adouble ceil(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::ceil(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(ceil_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);      // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  return a;
}

adouble floor(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval = ADOLC_MATH_NSP::floor(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(floor_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);                // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  return ret_adouble;
}

adouble floor(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval = ADOLC_MATH_NSP::floor(a.value());

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(floor_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);      // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif
  return a;
}

adouble fabs(const adouble &a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double temp = ADOLC_MATH_NSP::fabs(a.value());
  const double coval = temp != a.value() ? 0.0 : 1.0;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(abs_val);
      ADOLC_PUT_LOCINT(a.loc());           // arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // res
      ADOLC_PUT_VAL(coval);                // coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
        ++ADOLC_CURRENT_TAPE_INFOS.numSwitches;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }
  ret_adouble.value(temp);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  return ret_adouble;
}

adouble fabs(adouble &&a) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double temp = ADOLC_MATH_NSP::fabs(a.value());
  const double coval = temp != a.value() ? 0.0 : 1.0;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(abs_val);
      ADOLC_PUT_LOCINT(a.loc()); // arg
      ADOLC_PUT_LOCINT(a.loc()); // res
      ADOLC_PUT_VAL(coval);      // coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
        ++ADOLC_CURRENT_TAPE_INFOS.numSwitches;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }
  a.value(temp);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
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
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double temp = a.value();

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double temp = b.value();
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());
    }
  }
#endif

  adouble ret_adouble{tape_location{next_loc()}};

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.loc();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(min_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg1
      ADOLC_PUT_LOCINT(b.loc());           // = arg2
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);                // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (tmp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (tmp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(tmp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
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
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double temp = a.value();

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double temp = b.value();
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());
    }
  }
#endif

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(min_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);      // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (tmp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (tmp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(tmp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
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
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      const double temp = a.value();

      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }

    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] &&
        !ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      const double temp = b.value();
      if (temp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.loc());
      } else if (temp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.loc());
        ADOLC_PUT_VAL(temp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());
    }
  }
#endif

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.value();

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
        ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {
#endif

      put_op(min_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg1
      ADOLC_PUT_LOCINT(b.loc()); // = arg2
      ADOLC_PUT_LOCINT(b.loc()); // = res
      ADOLC_PUT_VAL(coval);      // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]) {

      if (tmp == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(b.loc());
      } else if (tmp == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(b.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(b.loc());
        ADOLC_PUT_VAL(tmp);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(b.value());
    }
#endif
  }

  b.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()] =
      (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] ||
       ADOLC_GLOBAL_TAPE_VARS.actStore[b.loc()]);
#endif

  return b;
}

adouble pow(const adouble &a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  adouble ret_adouble{tape_location{next_loc()}};
  const double coval2 = ADOLC_MATH_NSP::pow(a.value(), coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(pow_op);
      ADOLC_PUT_LOCINT(a.loc());           // = arg
      ADOLC_PUT_LOCINT(ret_adouble.loc()); // = res
      ADOLC_PUT_VAL(coval);                // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(ret_adouble.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[ret_adouble.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return ret_adouble;
}

adouble pow(adouble &&a, const double coval) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  const double coval2 = ADOLC_MATH_NSP::pow(a.value(), coval);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {
#endif

      put_op(pow_op);
      ADOLC_PUT_LOCINT(a.loc()); // = arg
      ADOLC_PUT_LOCINT(a.loc()); // = res
      ADOLC_PUT_VAL(coval);      // = coval

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()]) {

      if (coval2 == 0.0) {
        put_op(assign_d_zero);
        ADOLC_PUT_LOCINT(a.loc());
      } else if (coval2 == 1.0) {
        put_op(assign_d_one);
        ADOLC_PUT_LOCINT(a.loc());
      } else {
        put_op(assign_d);
        ADOLC_PUT_LOCINT(a.loc());
        ADOLC_PUT_VAL(coval2);
      }

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(a.value());
    }
#endif
  }

  a.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()] =
      ADOLC_GLOBAL_TAPE_VARS.actStore[a.loc()];
#endif

  ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
  return a;
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
        ADOLC_WRITE_SCAYLOR(temp.value());                                     \
    }                                                                          \
    temp.value(func(arg.value()));                                             \
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {                                  \
      ADOLC_PUT_VAL(arg.value());                                              \
      ADOLC_PUT_VAL(temp.value());                                             \
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

void condassign(adouble &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()]) {

        const double temp = arg1.value();
        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg1.loc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg1.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg1.loc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(arg1.value());
      }

      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()]) {

        const double temp = arg2.value();

        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg2.loc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg2.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg2.loc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(arg2.value());
      }
#endif

      put_op(cond_assign);
      ADOLC_PUT_LOCINT(cond.loc()); // = arg
      ADOLC_PUT_VAL(cond.value());
      ADOLC_PUT_LOCINT(arg1.loc()); // = arg1
      ADOLC_PUT_LOCINT(arg2.loc()); // = arg2
      ADOLC_PUT_LOCINT(res.loc());  // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (cond.value() > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (ADOLC_GLOBAL_TAPE_VARS.actStore[arg_loc]) {

        put_op(assign_a);
        ADOLC_PUT_LOCINT(arg_loc);   // = arg
        ADOLC_PUT_LOCINT(res.loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(res.value());

      } else {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {

          const double coval = ADOLC_GLOBAL_TAPE_VARS.store[arg_loc];
          if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(res.loc()); // = res
          } else if (coval == 1.0) {
            put_op(assign_d_one);
            ADOLC_PUT_LOCINT(res.loc()); // = res
          } else {
            put_op(assign_d);
            ADOLC_PUT_LOCINT(res.loc()); // = res
            ADOLC_PUT_VAL(coval);        // = coval
          }

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(res.value());
        }
      }
    }
#endif
  }

  if (cond.value() > 0)
    res.value(arg1.value());

  else
    res.value(arg2.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
    if (cond.value() > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()];

    else
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];

#endif
}

void condassign(adouble &res, const adouble &cond, const adouble &arg) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()]) {

        const double temp = arg.value();
        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg.loc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg.loc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(arg.value());
      }

#endif

      put_op(cond_assign_s);
      ADOLC_PUT_LOCINT(cond.loc()); // = arg
      ADOLC_PUT_VAL(cond.value());
      ADOLC_PUT_LOCINT(arg.loc()); // = arg1
      ADOLC_PUT_LOCINT(res.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0) {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()]) {

          put_op(assign_a);
          ADOLC_PUT_LOCINT(arg.loc()); // = arg
          ADOLC_PUT_LOCINT(res.loc()); // = res

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(res.value());

        } else {
          if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {
            const double coval = arg.value();

            if (coval == 0) {
              put_op(assign_d_zero);
              ADOLC_PUT_LOCINT(res.loc()); // = res
            } else if (coval == 1.0) {
              put_op(assign_d_one);
              ADOLC_PUT_LOCINT(res.loc()); // = res
            } else {
              put_op(assign_d);
              ADOLC_PUT_LOCINT(res.loc()); // = res
              ADOLC_PUT_VAL(coval);        // = coval
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
              ADOLC_WRITE_SCAYLOR(res.value());
          }
        }
      }
    }
#endif
  }

  if (cond.value() > 0)
    res.value(arg.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
    if (cond.value() > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()]) {

        const double temp = arg1.value();

        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg1.loc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg1.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg1.loc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(arg1.value());
      }

      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()]) {
        const double temp = arg2.value();

        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg2.loc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg2.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg2.loc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(arg2.value());
      }
#endif

      put_op(cond_eq_assign);
      ADOLC_PUT_LOCINT(cond.loc()); // = arg
      ADOLC_PUT_VAL(cond.value());
      ADOLC_PUT_LOCINT(arg1.loc()); // = arg1
      ADOLC_PUT_LOCINT(arg2.loc()); // = arg2
      ADOLC_PUT_LOCINT(res.loc());  // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {

        put_op(assign_a);
        ADOLC_PUT_LOCINT(arg_loc);   // = arg
        ADOLC_PUT_LOCINT(res.loc()); // = res

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(res.value());

      } else {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {

          const double coval = ADOLC_GLOBAL_TAPE_VARS.store[arg_loc];

          if (coval == 0) {
            put_op(assign_d_zero);
            ADOLC_PUT_LOCINT(res.loc()); // = res
          } else if (coval == 1.0) {
            put_op(assign_d_one);
            ADOLC_PUT_LOCINT(res.loc()); // = res
          } else {
            put_op(assign_d);
            ADOLC_PUT_LOCINT(res.loc()); // = res
            ADOLC_PUT_VAL(coval);        // = coval
          }

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(res.value());
        }
      }
    }
#endif
  }

  if (cond.value() >= 0)
    res.value(arg1.value());

  else
    res.value(arg2.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
    if (cond.value() > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg1.loc()];

    else
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg2.loc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
      if (!ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()]) {

        const double temp = arg.value();
        if (temp == 0.0) {
          put_op(assign_d_zero);
          ADOLC_PUT_LOCINT(arg.loc());
        } else if (temp == 1.0) {
          put_op(assign_d_one);
          ADOLC_PUT_LOCINT(arg.loc());
        } else {
          put_op(assign_d);
          ADOLC_PUT_LOCINT(arg.loc());
          ADOLC_PUT_VAL(temp);
        }

        ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
          ADOLC_WRITE_SCAYLOR(arg.value());
      }
#endif

      put_op(cond_eq_assign_s);
      ADOLC_PUT_LOCINT(cond.loc()); // = arg
      ADOLC_PUT_VAL(cond.value());
      ADOLC_PUT_LOCINT(arg.loc()); // = arg1
      ADOLC_PUT_LOCINT(res.loc()); // = res

      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0) {
        if (ADOLC_GLOBAL_TAPE_VARS.actStore[x_loc]) {

          put_op(assign_a);
          ADOLC_PUT_LOCINT(arg.loc()); // = arg
          ADOLC_PUT_LOCINT(res.loc()); // = res

          ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

          if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            ADOLC_WRITE_SCAYLOR(res.value());

        } else {
          if (ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()]) {

            const double coval = arg.value();

            if (coval == 0) {
              put_op(assign_d_zero);
              ADOLC_PUT_LOCINT(res.loc()); // = res
            } else if (coval == 1.0) {
              put_op(assign_d_one);
              ADOLC_PUT_LOCINT(res.loc()); // = res
            } else {
              put_op(assign_d);
              ADOLC_PUT_LOCINT(res.loc()); // = res
              ADOLC_PUT_VAL(coval);        // = coval
            }

            ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

            if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
              ADOLC_WRITE_SCAYLOR(res.value());
          }
        }
      }
    }
#endif
  }

  if (cond.value() >= 0)
    res.value(arg.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()]) {
    if (cond.value() > 0)
      ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
          ADOLC_GLOBAL_TAPE_VARS.actStore[arg.loc()];

  } else
    ADOLC_GLOBAL_TAPE_VARS.actStore[res.loc()] =
        ADOLC_GLOBAL_TAPE_VARS.actStore[cond.loc()];

#endif
}

void adolc_vec_copy(adouble *const dest, const adouble *const src,
                    size_t size) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (dest[size - 1].loc() - dest[0].loc() != size - 1 ||
      src[size - 1].loc() - src[0].loc() != size - 1)
    fail(ADOLC_VEC_LOCATIONGAP);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(vec_copy);
    ADOLC_PUT_LOCINT(src[0].loc());
    ADOLC_PUT_LOCINT(size);
    ADOLC_PUT_LOCINT(dest[0].loc());

    for (size_t i = 0; i < size; ++i) {
      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[dest[0].loc() + i]);
    }
  }
  for (size_t i = 0; i < size; ++i)
    ADOLC_GLOBAL_TAPE_VARS.store[dest[0].loc() + i] =
        ADOLC_GLOBAL_TAPE_VARS.store[src[0].loc() + i];
}

// requires a and b to be of size "size"
adouble adolc_vec_dot(const adouble *const vec_a, const adouble *const vec_b,
                      size_t size) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (vec_a[size - 1].loc() - vec_a[0].loc() != size - 1 ||
      vec_b[size - 1].loc() - vec_b[0].loc() != size - 1)
    fail(ADOLC_VEC_LOCATIONGAP);

  adouble ret_adouble{tape_location{next_loc()}};

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(vec_dot);
    ADOLC_PUT_LOCINT(vec_a[0].loc());
    ADOLC_PUT_LOCINT(vec_b[0].loc());
    ADOLC_PUT_LOCINT(size);
    ADOLC_PUT_LOCINT(ret_adouble.loc());

    ADOLC_CURRENT_TAPE_INFOS.num_eq_prod += 2 * size;

    ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
      ADOLC_WRITE_SCAYLOR(ret_adouble.value());
  }

  ret_adouble.value(0);

  for (size_t i = 0; i < size; ++i)
    ret_adouble.value(ret_adouble.value() +
                      ADOLC_GLOBAL_TAPE_VARS.store[vec_a[0].loc() + i] *
                          ADOLC_GLOBAL_TAPE_VARS.store[vec_b[0].loc() + i]);

  return ret_adouble;
}

// requires res, b and c to be of size "size"
void adolc_vec_axpy(adouble *const res, const adouble &a,
                    const adouble *const vec_a, const adouble *const vec_b,
                    size_t size) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (res[size - 1].loc() - res[0].loc() != size - 1 ||
      vec_a[size - 1].loc() - vec_a[0].loc() != size - 1 ||
      vec_b[size - 1].loc() - vec_b[0].loc() != size - 1)
    fail(ADOLC_VEC_LOCATIONGAP);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    put_op(vec_axpy);
    ADOLC_PUT_LOCINT(a.loc());
    ADOLC_PUT_LOCINT(vec_a[0].loc());
    ADOLC_PUT_LOCINT(vec_b[0].loc());
    ADOLC_PUT_LOCINT(size);
    ADOLC_PUT_LOCINT(res[0].loc());
    ADOLC_CURRENT_TAPE_INFOS.num_eq_prod += 2 * size - 1;

    for (size_t i = 0; i < size; ++i) {
      ++ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[res[0].loc() + i]);
    }
  }
  for (size_t i = 0; i < size; ++i)
    ADOLC_GLOBAL_TAPE_VARS.store[res[0].loc() + i] =
        a.value() * ADOLC_GLOBAL_TAPE_VARS.store[vec_a[0].loc() + i] +
        ADOLC_GLOBAL_TAPE_VARS.store[vec_b[0].loc() + i];
}
