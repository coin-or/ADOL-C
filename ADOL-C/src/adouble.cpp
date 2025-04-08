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
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>

adouble::adouble() : tape_loc_{getDefaultTape()} {
  if (!tape())
    throw ADOLCError("Default tape is nullptr!");

#if defined(ADOLC_ADOUBLE_STDCZERO)
  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(tape_loc_.loc_)) {
#endif
      tape()->put_op(assign_d_zero);
      tape()->put_loc(tape_loc_.loc_); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  tape()->set_ad_value(tape_loc_.loc_, 0.0);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(tape_loc_.loc_, false);
#endif
#endif
}

adouble::adouble(const std::shared_ptr<ValueTape> &in_tape)
    : tape_loc_{in_tape} {

#if defined(ADOLC_ADOUBLE_STDCZERO)
  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(tape_loc_.loc_)) {
#endif
      tape()->put_op(assign_d_zero);
      tape()->put_loc(tape_loc_.loc_); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  tape()->set_ad_value(tape_loc_.loc_, 0.0);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(tape_loc_.loc_, false);
#endif
#endif
}

adouble::adouble(double coval) : tape_loc_{getDefaultTape()} {
  if (!tape())
    throw ADOLCError("Default tape is nullptr!");

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->actStore[loc()]) {
#endif
      if (coval == 0) {
        tape()->put_op(assign_d_zero);
        tape()->put_loc(loc()); // = res
      } else if (coval == 1.0) {
        tape()->put_op(assign_d_one);
        tape()->put_loc(loc()); // = res
      } else {
        tape()->put_op(assign_d);
        tape()->put_loc(loc()); // = res
        tape()->put_val(coval); // = coval
      }

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), false);
#endif
}

adouble::adouble(double coval, const std::shared_ptr<ValueTape> &in_tape)
    : tape_loc_{in_tape} {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->actStore[loc()]) {
#endif
      if (coval == 0) {
        tape()->put_op(assign_d_zero);
        tape()->put_loc(loc()); // = res
      } else if (coval == 1.0) {
        tape()->put_op(assign_d_one);
        tape()->put_loc(loc()); // = res
      } else {
        tape()->put_op(assign_d);
        tape()->put_loc(loc()); // = res
        tape()->put_val(coval); // = coval
      }

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), false);
#endif
}
adouble::adouble(const adouble &a) : tape_loc_{a.tape()} {
  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(a.loc())) {
#endif
      tape()->put_op(assign_a);
      tape()->put_loc(a.loc()); // = arg
      tape()->put_loc(loc());   // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape()->get_active_value(loc())) {
        const double coval = a.value();
        if (coval == 0) {
          tape()->put_op(assign_d_zero);
          tape()->put_loc(loc()); // = res
        } else if (coval == 1.0) {
          tape()->put_op(assign_d_one);
          tape()->put_loc(loc()); // = res
        } else {
          tape()->put_op(assign_d);
          tape()->put_loc(loc()); // = res
          tape()->put_val(coval); // = coval
        }

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());
      }
    }
#endif
  }

  value(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), a.tape()->get_active_value(a.loc()));
#endif
}

/****************************************************************************/
/*                                                              ASSIGNMENTS */

adouble &adouble::operator=(double coval) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      if (coval == 0) {
        tape()->put_op(assign_d_zero);
        tape()->put_loc(loc()); // = res
      } else if (coval == 1.0) {
        tape()->put_op(assign_d_one);
        tape()->put_loc(loc()); // = res
      } else {
        tape()->put_op(assign_d);
        tape()->put_loc(loc()); // = res
        tape()->put_val(coval); // = coval
      }

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), false);
#endif
  return *this;
}

adouble &adouble::operator=(const adouble &a) {

  /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
  if (loc() != a.loc()) {
    if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        tape()->put_op(assign_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
        if (tape()->get_active_value(loc())) {
          double coval = a.value();
          if (coval == 0) {
            tape()->put_op(assign_d_zero);
            tape()->put_loc(loc()); // = res
          } else if (coval == 1.0) {
            tape()->put_op(assign_d_one);
            tape()->put_loc(loc()); // = res
          } else {
            tape()->put_op(assign_d);
            tape()->put_loc(loc()); // = res
            tape()->put_val(coval); // = coval
          }
          tape()->increment_numTays_Tape();
          if (tape()->keepTaylors())
            tape()->write_scaylor(value());
        }
      }
#endif
    }
    value(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    tape()->set_active_value(loc(), tape()->get_active_value(a.loc()));
#endif
  }
  return *this;
}

adouble &adouble::operator=(const pdouble &p) {

  if (tape()->traceFlag()) {

    tape()->put_op(assign_p);
    tape()->put_loc(p.loc());
    tape()->put_loc(loc());

    tape()->increment_numTays_Tape();

    if (tape()->keepTaylors())
      tape()->write_scaylor(value());
  }
  value(p.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), true);
#endif
  return *this;
}

/****************************************************************************/
/*                       ARITHMETIC ASSIGNMENT                             */

adouble &adouble::operator+=(const double coval) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(eq_plus_d);
      tape()->put_loc(loc()); // = res
      tape()->put_val(coval); // = coval

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  value(value() + coval);
  return *this;
}

adouble &adouble::operator+=(const adouble &a) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) &&
        tape()->get_active_value(loc())) {
#endif
      tape()->put_op(eq_plus_a);
      tape()->put_loc(a.loc()); // = arg
      tape()->put_loc(loc());   // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {
      const double coval = value();
      if (coval) {
        tape()->put_op(plus_d_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
        tape()->put_val(coval);
      } else {
        tape()->put_op(assign_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
      }

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());

    } else if (tape()->get_active_value(loc())) {
      const double coval = a.value();
      if (coval) {
        tape()->put_op(eq_plus_d);
        tape()->put_loc(loc()); // = res
        tape()->put_val(coval); // = coval

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());
      }
    }
#endif
  }
  value(value() + a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(
      loc(),
      tape()->get_active_value(loc() || a.tape()->get_active_value(a.loc())));
#endif
  return *this;
}

adouble &adouble::operator+=(adouble &&a) {

  int upd = 0;
  if (tape()->traceFlag())
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc()))
#endif
    {
      // if the structure is a*=b*c the call optimizes the temp adouble away.
      upd = tape()->upd_resloc_inc_prod(a.loc(), loc(), eq_min_prod);
    }
  if (upd) {
    value(value() + a.value());
    if (tape()->keepTaylors())
      tape()->delete_scaylor(a.loc());
    tape()->decrement_numTays_Tape();
    tape()->increment_num_eq_prod();
  } else {
    if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape()->get_active_value(a.loc()) &&
          tape()->get_active_value(loc())) {
#endif
        tape()->put_op(eq_plus_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());

#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (tape()->get_active_value(a.loc())) {
        double coval = value();
        if (coval) {
          tape()->put_op(plus_d_a);
          tape()->put_loc(a.loc()); // = arg
          tape()->put_loc(loc());   // = res
          tape()->put_val(coval);
        } else {
          tape()->put_op(assign_a);
          tape()->put_loc(a.loc()); // = arg
          tape()->put_loc(loc());   // = res
        }

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());

      } else if (tape()->get_active_value(loc())) {
        double coval = a.value();
        if (coval) {
          tape()->put_op(eq_plus_d);
          tape()->put_loc(loc()); // = res
          tape()->put_val(coval); // = coval

          tape()->increment_numTays_Tape();
          if (tape()->keepTaylors())
            tape()->write_scaylor(value());
        }
      }
#endif
    }
    value(value() + a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    tape()->set_active_value(loc(), tape()->get_active_value(loc()) ||
                                        tape()->get_active_value(a.loc()));
#endif
  }
  return *this;
}

adouble &adouble::operator-=(const double coval) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(eq_min_d);
      tape()->put_loc(loc()); // = res
      tape()->put_val(coval); // = coval

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  value(value() - coval);
  return *this;
}

adouble &adouble::operator-=(const adouble &a) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(a.loc()) && tape()->get_active_value(loc())) {
#endif
      tape()->put_op(eq_min_a);
      tape()->put_loc(a.loc()); // = arg
      tape()->put_loc(loc());   // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape()->get_active_value(a.loc())) {
      const double coval = value();
      if (coval) {
        tape()->put_op(min_d_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
        tape()->put_val(coval);
      } else {
        tape()->put_op(neg_sign_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
      }

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());

    } else if (tape()->get_active_value(loc())) {
      const double coval = a.value();
      if (coval) {
        tape()->put_op(eq_min_d);
        tape()->put_loc(loc()); // = res
        tape()->put_val(coval); // = coval

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());
      }
    }
#endif
  }
  value(value() - a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), (tape()->get_active_value(loc()) ||
                                   tape()->get_active_value(a.loc())));
#endif
  return *this;
}

adouble &adouble::operator-=(adouble &&a) {

  int upd = 0;
  if (tape()->traceFlag())
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc()))
#endif
    {
      upd = tape()->upd_resloc_inc_prod(a.loc(), loc(), eq_min_prod);
    }
  if (upd) {
    value(value() - a.value());
    if (tape()->keepTaylors())
      tape()->delete_scaylor(a.loc());
    tape()->decrement_numTays_Tape();
    tape()->increment_num_eq_prod();
  } else {
    if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape()->get_active_value(a.loc()) &&
          tape()->get_active_value(loc())) {
#endif
        tape()->put_op(eq_min_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (tape()->get_active_value(a.loc())) {
        double coval = value();
        if (coval) {
          tape()->put_op(min_d_a);
          tape()->put_loc(a.loc()); // = arg
          tape()->put_loc(loc());   // = res
          tape()->put_val(coval);
        } else {
          tape()->put_op(neg_sign_a);
          tape()->put_loc(a.loc()); // = arg
          tape()->put_loc(loc());   // = res
        }

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(value());

      } else if (tape()->get_active_value(loc())) {
        double coval = a.value();
        if (coval) {
          tape()->put_op(eq_min_d);
          tape()->put_loc(loc()); // = res
          tape()->put_val(coval); // = coval

          tape()->increment_numTays_Tape();
          if (tape()->keepTaylors())
            tape()->write_scaylor(value());
        }
      }
#endif
    }
    value(value() - a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    tape()->set_active_value(loc(), tape()->get_active_value(loc()) ||
                                        tape()->get_active_value(a.loc()));
#endif
  }

  return *this;
}

adouble &adouble::operator*=(const double coval) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(eq_mult_d);
      tape()->put_loc(loc()); // = res
      tape()->put_val(coval); // = coval

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(value() * coval);
  return *this;
}

adouble &adouble::operator*=(const adouble &a) {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(a.loc()) && tape()->get_active_value(loc())) {
#endif
      tape()->put_op(eq_mult_a);
      tape()->put_loc(a.loc()); // = arg
      tape()->put_loc(loc());   // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape()->get_active_value(a.loc())) {
      const double coval = value();
      if (coval == -1.0) {
        tape()->put_op(neg_sign_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
      } else if (coval == 1.0) {
        tape()->put_op(pos_sign_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
      } else {
        tape()->put_op(mult_d_a);
        tape()->put_loc(a.loc()); // = arg
        tape()->put_loc(loc());   // = res
        tape()->put_val(coval);
      }

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());

    } else if (tape()->get_active_value(loc())) {
      const double coval = a.value();
      tape()->put_op(eq_mult_d);
      tape()->put_loc(loc()); // = res
      tape()->put_val(coval); // = coval

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
    }
#endif
  }

  value(value() * a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), (tape()->get_active_value(loc()) ||
                                   tape()->get_active_value(a.loc())));
#endif
  return *this;
}
/****************************************************************************/
/*                       INCREMENT / DECREMENT                              */

adouble adouble::operator++(int) {
  // create adouble to store old state in it.
  adouble ret_adouble{tape()};

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(assign_a);
      tape()->put_loc(loc());             // = arg
      tape()->put_loc(ret_adouble.loc()); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape()->get_active_value(ret_adouble.loc())) {
        const double coval = value();
        if (coval == 0) {
          tape()->put_op(assign_d_zero);
          tape()->put_loc(ret_adouble.loc()); // = res
        } else if (coval == 1.0) {
          tape()->put_op(assign_d_one);
          tape()->put_loc(ret_adouble.loc()); // = res
        } else {
          tape()->put_op(assign_d);
          tape()->put_loc(ret_adouble.loc()); // = res
          tape()->put_val(coval);             // = coval
        }

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(ret_adouble.value());
      }
    }
#endif
  }

  ret_adouble.value(value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), tape()->get_active_value(loc()));
#endif

  // change input adouble to new state
  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(incr_a);
      tape()->put_loc(loc()); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  const double val = value();
  value(val + 1);
  return ret_adouble;
}

adouble adouble::operator--(int) {
  // create adouble to store old state in it.
  adouble ret_adouble{tape()};

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(assign_a);
      tape()->put_loc(loc());             // = arg
      tape()->put_loc(ret_adouble.loc()); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape()->get_active_value(ret_adouble.loc())) {
        const double coval = value();
        if (coval == 0) {
          tape()->put_op(assign_d_zero);
          tape()->put_loc(ret_adouble.loc()); // = res
        } else if (coval == 1.0) {
          tape()->put_op(assign_d_one);
          tape()->put_loc(ret_adouble.loc()); // = res
        } else {
          tape()->put_op(assign_d);
          tape()->put_loc(ret_adouble.loc()); // = res
          tape()->put_val(coval);             // = coval
        }

        tape()->increment_numTays_Tape();
        if (tape()->keepTaylors())
          tape()->write_scaylor(ret_adouble.value());
      }
    }
#endif
  }

  ret_adouble.value(value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(ret_adouble.loc(), tape()->get_active_value(loc()));
#endif

  // write new state into input adouble
  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(decr_a);
      tape()->put_loc(loc()); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(value() - 1);
  return ret_adouble;
}

adouble &adouble::operator++() {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(incr_a);
      tape()->put_loc(loc()); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  value(value() + 1);
  return *this;
}

adouble &adouble::operator--() {

  if (tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape()->get_active_value(loc())) {
#endif
      tape()->put_op(decr_a);
      tape()->put_loc(loc()); // = res

      tape()->increment_numTays_Tape();
      if (tape()->keepTaylors())
        tape()->write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(value() - 1);
  return *this;
}

/**************************************************************************
 *           MARK INDEPENDENT AND DEPENDENT
 */

// Assign a double value to an adouble and mark the adouble as independent on
// the tape
adouble &adouble::operator<<=(const double input) {

  if (tape()->traceFlag()) {
    tape()->increment_numInds();
    tape()->put_op(assign_ind);

    tape()->put_loc(loc()); // = res
    tape()->increment_numTays_Tape();
    if (tape()->keepTaylors())
      tape()->write_scaylor(value());
  }
  value(input);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), true);
#endif
  return *this;
}

// Assign the coval of an adouble to a double reference and mark the adouble
// as dependent variable on the tape. At the end of the function, the double
// reference can be seen as output value of the function given by the trace
// of the adouble.
adouble &adouble::operator>>=(double &output) {

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!tape()->get_active_value(loc())) {
    fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) "
                      "as dependent.\n");
    const double coval = value();
    if (coval == 0.0) {
      tape()->put_op(assign_d_zero);
      tape()->put_loc(loc());
    } else if (coval == 1.0) {
      tape()->put_op(assign_d_one);
      tape()->put_loc(loc());
    } else {
      tape()->put_op(assign_d);
      tape()->put_loc(loc());
      tape()->put_val(coval);
    }

    tape()->increment_numTays_Tape();
    if (tape()->keepTaylors())
      tape()->write_scaylor(value());
  }
#endif
  tape()->traceFlag();

  if (tape()->traceFlag()) {
    tape()->increment_numDeps();

    tape()->put_op(assign_dep);
    tape()->put_loc(loc()); // = res
  }
  output = value();
  return *this;
}

void adouble::declareIndependent() {

  if (tape()->traceFlag()) {
    tape()->increment_numInds();

    tape()->put_op(assign_ind);
    tape()->put_loc(loc()); // = res

    tape()->increment_numTays_Tape();
    if (tape()->keepTaylors())
      tape()->write_scaylor(value());
  }
#if defined(ADOLC_TRACK_ACTIVITY)
  tape()->set_active_value(loc(), true);
#endif
}

void adouble::declareDependent() {

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!tape()->get_active_value(loc())) {
    fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) "
                      "as dependent.\n");
    const double coval = value();
    if (coval == 0.0) {
      tape()->put_op(assign_d_zero);
      tape()->put_loc(loc());
    } else if (coval == 1.0) {
      tape()->put_op(assign_d_one);
      tape()->put_loc(loc());
    } else {
      tape()->put_op(assign_d);
      tape()->put_loc(loc());
      tape()->put_val(coval);
    }

    tape()->increment_numTays_Tape();
    if (tape()->keepTaylors())
      tape()->write_scaylor(value());
  }
#endif
  if (tape()->traceFlag()) {
    tape()->increment_numDeps();

    tape()->put_op(assign_dep);
    tape()->put_loc(loc()); // = res
  }
}

/****************************************************************************/
/*                                                           INPUT / OUTPUT */

std::ostream &operator<<(std::ostream &out, const adouble &a) {

  return out << a.value() << "(a)";
}

std::istream &operator>>(std::istream &in, adouble &a) {

  double coval;
  in >> coval;
  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif
      if (coval == 0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc()); // = res
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc()); // = res
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);   // = coval
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  a.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_variable(a.loc(), false);
#endif
  return in;
}

/****************************************************************************/
/*                               COMPARISON                                 */

#ifdef ADOLC_ADVANCED_BRANCHING

adouble operator!=(const adouble &a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  adouble ret_adouble{a.tape()};
  const double res = static_cast<double>(a_coval != b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(neq_a_a);
    a.tape()->put_loc(a.loc());           // arg
    a.tape()->put_loc(b.loc());           // arg1
    a.tape()->put_val(res);               // check for branch switch
    a.tape()->put_loc(ret_adouble.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator!=(adouble &&a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval != b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(neq_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(a.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator!=(const adouble &a, adouble &&b) { return std::move(b) != a; }

adouble operator==(const adouble &a, const adouble &b) {
  adouble ret_adouble{a.tape()};

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval == b_coval);
  if (a.tape()->traceFlag()) {
    a.tape()->put_op(eq_a_a);
    a.tape()->put_loc(a.loc());           // arg
    a.tape()->put_loc(b.loc());           // arg1
    a.tape()->put_val(res);               // check for branch switch
    a.tape()->put_loc(ret_adouble.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator==(adouble &&a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval == b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(eq_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(a.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator==(const adouble &a, adouble &&b) { return std::move(b) == a; }

adouble operator<=(const adouble &a, const adouble &b) {
  adouble ret_adouble{a.tape()};

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);
  if (a.tape()->traceFlag()) {
    a.tape()->put_op(le_a_a);
    a.tape()->put_loc(a.loc());           // arg
    a.tape()->put_loc(b.loc());           // arg1
    a.tape()->put_val(res);               // check for branch switch
    a.tape()->put_loc(ret_adouble.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<=(adouble &&a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(le_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(a.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator<=(const adouble &a, adouble &&b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(le_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(b.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(b.value());
  }

  b.value(res);
  return b;
}
adouble operator>=(const adouble &a, const adouble &b) {
  adouble ret_adouble{a.tape()};

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);
  if (a.tape()->traceFlag()) {
    a.tape()->put_op(ge_a_a);
    a.tape()->put_loc(a.loc());           // arg
    a.tape()->put_loc(b.loc());           // arg1
    a.tape()->put_val(res);               // check for branch switch
    a.tape()->put_loc(ret_adouble.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>=(adouble &&a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(ge_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(a.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>=(const adouble &a, adouble &&b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(ge_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(b.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(b.value());
  }

  b.value(res);
  return b;
}

adouble operator<(const adouble &a, const adouble &b) {
  adouble ret_adouble{a.tape()};

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);
  if (a.tape()->traceFlag()) {
    a.tape()->put_op(lt_a_a);
    a.tape()->put_loc(a.loc());           // arg
    a.tape()->put_loc(b.loc());           // arg1
    a.tape()->put_val(res);               // check for branch switch
    a.tape()->put_loc(ret_adouble.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<(adouble &&a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(lt_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(a.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator<(const adouble &a, adouble &&b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(lt_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(b.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(b.value());
  }

  b.value(res);
  return b;
}

adouble operator>(const adouble &a, const adouble &b) {
  adouble ret_adouble{a.tape()};

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);
  if (a.tape()->traceFlag()) {
    a.tape()->put_op(gt_a_a);
    a.tape()->put_loc(a.loc());           // arg
    a.tape()->put_loc(b.loc());           // arg1
    a.tape()->put_val(res);               // check for branch switch
    a.tape()->put_loc(ret_adouble.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>(adouble &&a, const adouble &b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(gt_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(a.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>(const adouble &a, adouble &&b) {

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);

  if (a.tape()->traceFlag()) {
    a.tape()->put_op(gt_a_a);
    a.tape()->put_loc(a.loc()); // arg
    a.tape()->put_loc(b.loc()); // arg1
    a.tape()->put_val(res);     // check for branch switch
    a.tape()->put_loc(b.loc()); // res

    a.tape()->increment_numTays_Tape();
    if (a.tape()->keepTaylors())
      a.tape()->write_scaylor(b.value());
  }

  b.value(res);
  return b;
}

#endif // ADOLC_ADVANCED_BRANCHING

bool operator!=(const adouble &a, const double coval) {

  if (coval)
    return (-coval + a != 0);
  else {
    if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        a.tape()->put_op(a.value() ? neq_zero : eq_zero);
        a.tape()->put_loc(a.loc());
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

  if (coval)
    return (-coval + a == 0);
  else {
    if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        a.tape()->put_op(a.value() ? neq_zero : eq_zero);
        a.tape()->put_loc(a.loc());
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

  if (coval)
    return (-coval + a <= 0);
  else {
    bool b = (a.value() <= 0);
    if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        a.tape()->put_op(b ? le_zero : gt_zero);
        a.tape()->put_loc(a.loc());
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

  if (coval)
    return (-coval + a >= 0);
  else {
    bool b = (a.value() >= 0);
    if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        a.tape()->put_op(b ? ge_zero : lt_zero);
        a.tape()->put_loc(a.loc());
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

  if (coval)
    return (-coval + a < 0);
  else {
    bool b = (a.value() < 0);
    if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        a.tape()->put_op(b ? lt_zero : ge_zero);
        a.tape()->put_loc(a.loc());
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

  if (coval)
    return (-coval + a > 0);
  else {
    bool b = (a.value() > 0);
    if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (a.tape()->get_active_value(a.loc())) {
#endif
        a.tape()->put_op(b ? gt_zero : le_zero);
        a.tape()->put_loc(a.loc());
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
/*                           SIGN  OPERATORS */

adouble operator+(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = a.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif
      a.tape()->put_op(pos_sign_a);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {
      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ret_adouble.tape()->set_active_value(ret_adouble.loc(),
                                       a.tape()->get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a) {

  const double coval = a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(pos_sign_a);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (a.tape()->get_active_value(a.loc())) {
      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }
  return a;
}

adouble operator-(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(neg_sign_a);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (-coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (-coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(-coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(-coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a) {

  const double coval = a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(neg_sign_a);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (-coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (-coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(-coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }
  a.value(-coval);
  return a;
}

/****************************************************************************/
/*                            BINARY OPERATORS                              */

adouble operator+(const adouble &a, const adouble &b) {

  adouble ret_adouble{a.tape()};
  const double coval2 = a.value() + b.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(plus_a_a);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      a.tape()->put_op(plus_d_a);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(b.value());

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();

      if (coval) {
        a.tape()->put_op(plus_d_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);
      } else {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ret_adouble.tape()->set_active_variable(
      ret_adouble.loc(), (a.tape()->get_active_value(a.loc()) ||
                          b.tape()->get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a, const adouble &b) {

  const double coval2 = a.value() + b.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(plus_a_a);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      a.tape()->put_op(plus_d_a);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(b.value());

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();

      if (coval) {
        a.tape()->put_op(plus_d_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);
      } else {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), (a.tape()->get_active_value(a.loc()) ||
                                       b.tape()->get_active_value(b.loc())));
#endif
  return a;
}

adouble operator+(const double coval, const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval2 = coval + a.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      if (coval) {
        a.tape()->put_op(plus_d_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);             // = coval
      } else {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble operator+(const double coval, adouble &&a) {

  const double coval2 = coval + a.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      if (coval) {
        a.tape()->put_op(plus_d_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);   // = coval
      } else {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc([a.loc()]);
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
  return a;
}

adouble operator-(const adouble &a, const adouble &b) {

  adouble ret_adouble{a.tape()};
  const double coval2 = a.value() - b.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {

#endif
      a.tape()->put_op(min_a_a);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = -b.value();
      a.tape()->put_op(plus_d_a);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();
      if (coval) {
        a.tape()->put_op(min_d_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);
      } else {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             (a.tape()->get_active_value(a.loc()) ||
                              b.tape()->get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a, const adouble &b) {

  const double coval2 = a.value() - b.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {

#endif
      a.tape()->put_op(min_a_a);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = -b.value();
      a.tape()->put_op(plus_d_a);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();
      if (coval) {
        a.tape()->put_op(min_d_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);
      } else {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), a.tape()->get_active_value(a.loc()) ||
                                          b.tape()->get_active_value(b.loc()));
#endif
  return a;
}

adouble operator-(const double coval, const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval2 = coval - a.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      if (coval) {
        a.tape()->put_op(min_d_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);             // = coval
      } else {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble operator-(const double coval, adouble &&a) {

  const double coval2 = coval - a.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      if (coval) {
        a.tape()->put_op(min_d_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);   // = coval
      } else {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);

  return a;
}

adouble operator*(const adouble &a, const adouble &b) {

  adouble ret_adouble{a.tape()};
  const double coval2 = a.value() * b.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(mult_a_a);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = b.value();
      a.tape()->put_op(mult_d_a);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (b.tape()->get_active_value(b.loc())) {
      const double coval = a.value();

      if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      } else if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             (a.tape()->get_active_value(a.loc()) ||
                              b.tape()->get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator*(adouble &&a, const adouble &b) {

  const double coval2 = a.value() * b.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(mult_a_a);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = b.value();
      a.tape()->put_op(mult_d_a);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (b.tape()->get_active_value(b.loc())) {
      const double coval = a.value();

      if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      } else if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), (a.tape()->get_active_value(a.loc()) ||
                                       b.tape()->get_active_value(b.loc())));
#endif
  return a;
}

adouble operator*(const double coval, const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval2 = coval * a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      } else if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);             // = coval
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[locat]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble operator*(const double coval, adouble &&a) {

  const double coval2 = coval * a.value();
  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      } else if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);   // = coval
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[locat]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }
  a.value(coval2);
  return a;
}

adouble operator/(const adouble &a, const adouble &b) {

  adouble ret_adouble{a.tape()};
  const double coval2 = a.value() / b.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(div_a_a);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      } else if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(b.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(a.loc());           // = arg
        a.tape()->put_loc(ret_adouble.loc()); // = res
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();

      a.tape()->put_op(div_d_a);
      a.tape()->put_loc(b.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             (a.tape()->get_active_value(a.loc()) ||
                              b.tape()->get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator/(adouble &&a, const adouble &b) {

  const double coval2 = a.value() / b.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(div_a_a);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      } else if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(a.loc()); // = res
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();

      a.tape()->put_op(div_d_a);
      a.tape()->put_loc(b.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), (a.tape()->get_active_value(a.loc()) ||
                                       b.tape()->get_active_value(b.loc())));
#endif
  return a;
}

adouble operator/(const adouble &a, adouble &&b) {

  const double coval2 = a.value() / b.value();

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (a.tape()->get_active_value(a.loc()) &&
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(div_a_a);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(b.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        a.tape()->put_op(neg_sign_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(b.loc()); // = res
      } else if (coval == 1.0) {
        a.tape()->put_op(pos_sign_a);
        a.tape()->put_loc(b.loc()); // = arg
        a.tape()->put_loc(b.loc()); // = res
      } else {
        a.tape()->put_op(mult_d_a);
        a.tape()->put_loc(a.loc()); // = arg
        a.tape()->put_loc(b.loc()); // = res
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());

    } else if (b.tape()->get_active_value(b.loc())) {

      const double coval = a.value();

      a.tape()->put_op(div_d_a);
      a.tape()->put_loc(b.loc()); // = arg
      a.tape()->put_loc(b.loc()); // = res
      a.tape()->put_val(coval);

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());
    }
#endif
  }

  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  b.tape()->set_active_value(b.loc(), a.tape()->get_active_value(a.loc()) ||
                                          b.tape()->get_active_value(b.loc()));
#endif
  return b;
}

adouble operator/(const double coval, const adouble &a) {

  adouble ret_adouble{a.tape()};

  const double coval2 = coval / a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(div_d_a);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);             // = coval

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble operator/(const double coval, adouble &&a) {

  const double coval2 = coval / a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(div_d_a);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);   // = coval

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), a.tape()->get_active_value(a.loc()));
#endif
  return a;
}

/****************************************************************************/
/*                          UNARY OPERATIONS                                */

adouble exp(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::exp(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(exp_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }
  ret_adouble.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble exp(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::exp(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(exp_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), a.tape()->get_active_value(a.loc()));
#endif

  return a;
}

adouble log(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::log(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(log_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble log(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::log(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(log_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), a.tape()->get_active_value(a.loc()));
#endif

  return a;
}

adouble sqrt(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::sqrt(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(sqrt_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble sqrt(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::sqrt(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(sqrt_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), a.tape()->get_active_value(a.loc()));
#endif

  return a;
}

adouble cbrt(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::cbrt(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(cbrt_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble cbrt(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::cbrt(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(cbrt_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();
      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble sin(const adouble &a) {

  adouble ret_adouble{a.tape()};

  const double coval1 = ADOLC_MATH_NSP::sin(a.value());
  const double coval2 = ADOLC_MATH_NSP::cos(a.value());

  adouble b(a.tape());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(sin_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->add_numTays_Tape(2);

      if (a.tape()->keepTaylors()) {
        a.tape()->write_scaylor(b.value());
        a.tape()->write_scaylor(ret_adouble.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

        if (coval1 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(ret_adouble.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(ret_adouble.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(ret_adouble.loc());
          a.tape()->put_val(coval1);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(ret_adouble.value());
      }
      if (b.tape()->get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(b.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(b.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(b.loc());
          a.tape()->put_val(coval2);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(b.value());
      }
    }
#endif
  }
  ret_adouble.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             b.tape()->get_active_value(b.loc()) =
                                 a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble sin(adouble &&a) {

  const double coval1 = ADOLC_MATH_NSP::sin(a.value());
  const double coval2 = ADOLC_MATH_NSP::cos(a.value());

  adouble b(a.tape());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(sin_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->add_numTays_Tape(2);

      if (a.tape()->keepTaylors()) {
        a.tape()->write_scaylor(b.value());
        a.tape()->write_scaylor(a.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (a.tape()->get_active_value(a.loc())) {

        if (coval1 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(a.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(a.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(a.loc());
          a.tape()->put_val(coval1);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(a.value());
      }
      if (b.tape()->get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(b.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(b.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(b.loc());
          a.tape()->put_val(coval2);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(b.value());
      }
    }
#endif
  }

  a.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), b.tape()->get_active_value(b.loc()) =
                                          a.tape()->get_active_value(a.loc()));
#endif

  return a;
}

adouble cos(const adouble &a) {

  adouble ret_adouble{a.tape()};

  const double coval1 = ADOLC_MATH_NSP::cos(a.value());
  const double coval2 = ADOLC_MATH_NSP::sin(a.value());

  adouble b(a.tape());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(cos_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->add_numTays_Tape(2);

      if (a.tape()->keepTaylors()) {
        a.tape()->write_scaylor(b.value());
        a.tape()->write_scaylor(ret_adouble.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

        if (coval1 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(ret_adouble.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(ret_adouble.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(ret_adouble.loc());
          a.tape()->put_val(coval1);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(ret_adouble.value());
      }
      if (b.tape()->get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(b.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(b.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(b.loc());
          a.tape()->put_val(coval2);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(b.value());
      }
    }

#endif
  }

  ret_adouble.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             b.tape()->get_active_value(b.loc()) =
                                 a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble cos(adouble &&a) {

  const double coval1 = ADOLC_MATH_NSP::cos(a.value());
  const double coval2 = ADOLC_MATH_NSP::sin(a.value());

  adouble b(a.tape());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(cos_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->add_numTays_Tape(2);

      if (a.tape()->keepTaylors()) {
        a.tape()->write_scaylor(b.value());
        a.tape()->write_scaylor(a.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (a.tape()->get_active_value(a.loc())) {

        if (coval1 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(a.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(a.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(a.loc());
          a.tape()->put_val(coval1);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(a.value());
      }
      if (b.tape()->get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          a.tape()->put_op(assign_d_zero);
          a.tape()->put_loc(b.loc());
        } else if (coval1 == 1.0) {
          a.tape()->put_op(assign_d_one);
          a.tape()->put_loc(b.loc());
        } else {
          a.tape()->put_op(assign_d);
          a.tape()->put_loc(b.loc());
          a.tape()->put_val(coval2);
        }

        a.tape()->increment_numTays_Tape();

        if (a.tape()->keepTaylors())
          a.tape()->write_scaylor(b.value());
      }
    }

#endif
  }

  a.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), b.tape()->get_active_value(b.loc()) =
                                          a.tape()->get_active_value(a.loc()));
#endif

  return a;
}

adouble asin(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::asin(a.value());
  adouble b = 1.0 / sqrt(1.0 - a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(asin_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble asin(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::asin(a.value());
  adouble b = 1.0 / sqrt(1.0 - a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(asin_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), a.tape()->get_active_value(a.loc()));
#endif

  return a;
}

adouble acos(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::acos(a.value());

  adouble b = -1.0 / sqrt(1.0 - a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(acos_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble acos(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::acos(a.value());

  adouble b = -1.0 / sqrt(1.0 - a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(acos_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble atan(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::atan(a.value());

  adouble b = 1.0 / (1.0 + a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(atan_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble atan(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::atan(a.value());

  adouble b = 1.0 / (1.0 + a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(atan_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble asinh(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP_ERF::asinh(a.value());
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(asinh_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble asinh(adouble &&a) {

  const double coval = ADOLC_MATH_NSP_ERF::asinh(a.value());
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(asinh_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble acosh(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP_ERF::acosh(a.value());
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(acosh_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {
      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble acosh(adouble &&a) {

  const double coval = ADOLC_MATH_NSP_ERF::acosh(a.value());
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(acosh_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (a.tape()->get_active_value(a.loc())) {
      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble atanh(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP_ERF::atanh(a.value());
  adouble b = 1.0 / (1.0 - a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(atanh_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble atanh(adouble &&a) {

  const double coval = ADOLC_MATH_NSP_ERF::atanh(a.value());
  adouble b = 1.0 / (1.0 - a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(atanh_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble erf(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP_ERF::erf(a.value());
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(erf_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble erf(adouble &&a) {

  const double coval = ADOLC_MATH_NSP_ERF::erf(a.value());
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(erf_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble erfc(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP_ERF::erfc(a.value());
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(erfc_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble erfc(adouble &&a) {

  const double coval = ADOLC_MATH_NSP_ERF::erfc(a.value());
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(erfc_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble ceil(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::ceil(a.value());

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(ceil_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);             // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble ceil(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::ceil(a.value());

  if (a.tape()->traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(ceil_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);   // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble floor(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double coval = ADOLC_MATH_NSP::floor(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(floor_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);             // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble floor(adouble &&a) {

  const double coval = ADOLC_MATH_NSP::floor(a.value());

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(floor_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);   // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble fabs(const adouble &a) {

  adouble ret_adouble{a.tape()};
  const double temp = ADOLC_MATH_NSP::fabs(a.value());
  const double coval = temp != a.value() ? 0.0 : 1.0;

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(abs_val);
      a.tape()->put_loc(a.loc());           // arg
      a.tape()->put_loc(ret_adouble.loc()); // res
      a.tape()->put_val(coval);             // coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->no_min_max())
        a.tape()->increment_numSwitches();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }
  ret_adouble.value(temp);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble fabs(adouble &&a) {

  const double temp = ADOLC_MATH_NSP::fabs(a.value());
  const double coval = temp != a.value() ? 0.0 : 1.0;

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(abs_val);
      a.tape()->put_loc(a.loc()); // arg
      a.tape()->put_loc(a.loc()); // res
      a.tape()->put_val(coval);   // coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->no_min_max())
        a.tape()->increment_numSwitches();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }
  a.value(temp);

  return a;
}

adouble fmin(const adouble &a, const adouble &b) {

  if (a.tape()->no_min_max())
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (a.tape()->traceFlag()) {
    if (b.tape()->get_active_value(b.loc()) &&
        !a.tape()->get_active_value(a.loc())) {

      const double temp = a.value();

      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }

    if (a.tape()->get_active_value(a.loc()) &&
        !b.tape()->get_active_value(b.loc())) {

      const double temp = b.value();
      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(b.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(b.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(b.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());
    }
  }
#endif

  adouble ret_adouble{a.tape()};

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.loc();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) ||
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(min_op);
      a.tape()->put_loc(a.loc());           // = arg1
      a.tape()->put_loc(b.loc());           // = arg2
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);             // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (tmp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (tmp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(tmp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             (a.tape()->get_active_value(a.loc()) ||
                              b.tape()->get_active_value(b.loc())));
#endif

  return ret_adouble;
}

adouble fmin(adouble &&a, const adouble &b) {

  if (a.tape()->no_min_max())
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (a.tape()->traceFlag()) {
    if (b.tape()->get_active_value(b.loc()) &&
        !a.tape()->get_active_value(a.loc())) {

      const double temp = a.value();

      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }

    if (a.tape()->get_active_value(a.loc()) &&
        !b.tape()->get_active_value(b.loc())) {

      const double temp = b.value();
      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(b.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(b.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(b.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());
    }
  }
#endif

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) ||
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(min_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);   // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (tmp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (tmp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(tmp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(a.loc(), (a.tape()->get_active_value(a.loc()) ||
                                       b.tape()->get_active_value(b.loc())));
#endif

  return a;
}

adouble fmin(const adouble &a, adouble &&b) {

  if (a.tape()->no_min_max())
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (a.tape()->traceFlag()) {
    if (b.tape()->get_active_value(b.loc()) &&
        !a.tape()->get_active_value(a.loc())) {

      const double temp = a.value();

      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }

    if (a.tape()->get_active_value(a.loc()) &&
        !b.tape()->get_active_value(b.loc())) {

      const double temp = b.value();
      if (temp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(b.loc());
      } else if (temp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(b.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(b.loc());
        a.tape()->put_val(temp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());
    }
  }
#endif

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.value();

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc()) ||
        b.tape()->get_active_value(b.loc())) {
#endif

      a.tape()->put_op(min_op);
      a.tape()->put_loc(a.loc()); // = arg1
      a.tape()->put_loc(b.loc()); // = arg2
      a.tape()->put_loc(b.loc()); // = res
      a.tape()->put_val(coval);   // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (b.tape()->get_active_value(b.loc())) {

      if (tmp == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(b.loc());
      } else if (tmp == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(b.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(b.loc());
        a.tape()->put_val(tmp);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(b.value());
    }
#endif
  }

  b.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  b.tape()->set_active_value(b.loc(), (a.tape()->get_active_value(a.loc()) ||
                                       b.tape()->get_active_value(b.loc())));
#endif

  return b;
}

adouble pow(const adouble &a, const double coval) {

  adouble ret_adouble{a.tape()};
  const double coval2 = ADOLC_MATH_NSP::pow(a.value(), coval);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(pow_op);
      a.tape()->put_loc(a.loc());           // = arg
      a.tape()->put_loc(ret_adouble.loc()); // = res
      a.tape()->put_val(coval);             // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(ret_adouble.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(ret_adouble.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  a.tape()->set_active_value(ret_adouble.loc(),
                             a.tape()->get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble pow(adouble &&a, const double coval) {

  const double coval2 = ADOLC_MATH_NSP::pow(a.value(), coval);

  if (a.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (a.tape()->get_active_value(a.loc())) {
#endif

      a.tape()->put_op(pow_op);
      a.tape()->put_loc(a.loc()); // = arg
      a.tape()->put_loc(a.loc()); // = res
      a.tape()->put_val(coval);   // = coval

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (a.tape()->get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        a.tape()->put_op(assign_d_zero);
        a.tape()->put_loc(a.loc());
      } else if (coval2 == 1.0) {
        a.tape()->put_op(assign_d_one);
        a.tape()->put_loc(a.loc());
      } else {
        a.tape()->put_op(assign_d);
        a.tape()->put_loc(a.loc());
        a.tape()->put_val(coval2);
      }

      a.tape()->increment_numTays_Tape();

      if (a.tape()->keepTaylors())
        a.tape()->write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);

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
                                                                               \
    if (a.tape()->traceFlag()) {                                               \
      a.tape()->put_op(gen_quad);                                              \
      a.tape()->put_loc(arg.loc());                                            \
      a.tape()->put_loc(val.loc());                                            \
      a.tape()->put_loc(temp.loc());                                           \
      a.tape()->increment_numTays_Tape();                                      \
      if (a.tape()->keepTaylors())                                             \
        a.tape()->write_scaylor(temp.value());                                 \
    }                                                                          \
    temp.value(func(arg.value()));                                             \
    if (a.tape()->traceFlag()) {                                               \
      a.tape()->put_val(arg.value());                                          \
      a.tape()->put_val(temp.value());                                         \
    }                                                                          \
    return temp;                                                               \
  }

double myquad(double x) {
  double res;
  res = ADOLC_MATH_NSP::log(x);
  return res;
}

/* This defines the natural logarithm as a quadrature */

// extend_quad(myquad, val = 1 / arg);

void condassign(adouble &res, const adouble &cond, const adouble &arg1,
                const adouble &arg2) {

  if (cond.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (cond.tape()->get_active_value(cond.loc())) {
      if (!cond.tape()->globalTapeVars_.actStore[arg1.loc()]) {

        const double temp = arg1.value();
        if (temp == 0.0) {
          cond.tape()->put_op(assign_d_zero);
          cond.tape()->put_loc(arg1.loc());
        } else if (temp == 1.0) {
          cond.tape()->put_op(assign_d_one);
          cond.tape()->put_loc(arg1.loc());
        } else {
          cond.tape()->put_op(assign_d);
          cond.tape()->put_loc(arg1.loc());
          cond.tape()->put_val(temp);
        }

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(arg1.value());
      }

      if (!cond.tape()->globalTapeVars_.actStore[arg2.loc()]) {

        const double temp = arg2.value();

        if (temp == 0.0) {
          cond.tape()->put_op(assign_d_zero);
          cond.tape()->put_loc(arg2.loc());
        } else if (temp == 1.0) {
          cond.tape()->put_op(assign_d_one);
          cond.tape()->put_loc(arg2.loc());
        } else {
          cond.tape()->put_op(assign_d);
          cond.tape()->put_loc(arg2.loc());
          cond.tape()->put_val(temp);
        }

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(arg2.value());
      }
#endif

      cond.tape()->put_op(cond_assign);
      cond.tape()->put_loc(cond.loc()); // = arg
      cond.tape()->put_val(cond.value());
      cond.tape()->put_loc(arg1.loc()); // = arg1
      cond.tape()->put_loc(arg2.loc()); // = arg2
      cond.tape()->put_loc(res.loc());  // = res

      cond.tape()->increment_numTays_Tape();

      if (cond.tape()->keepTaylors())
        cond.tape()->write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (cond.value() > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (cond.tape()->globalTapeVars_.actStore[arg_loc]) {

        cond.tape()->put_op(assign_a);
        cond.tape()->put_loc(arg_loc);   // = arg
        cond.tape()->put_loc(res.loc()); // = res

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(res.value());

      } else {
        if (cond.tape()->get_active_value(res.loc())) {

          const double coval = cond.tape()->globalTapeVars_.store[arg_loc];
          if (coval == 0) {
            cond.tape()->put_op(assign_d_zero);
            cond.tape()->put_loc(res.loc()); // = res
          } else if (coval == 1.0) {
            cond.tape()->put_op(assign_d_one);
            cond.tape()->put_loc(res.loc()); // = res
          } else {
            cond.tape()->put_op(assign_d);
            cond.tape()->put_loc(res.loc()); // = res
            cond.tape()->put_val(coval);     // = coval
          }

          cond.tape()->increment_numTays_Tape();

          if (cond.tape()->keepTaylors)
            cond.tape()->write_scaylor(res.value());
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
  if (!cond.tape()->get_active_value(cond.loc())) {
    if (cond.value() > 0)
      cond.tape()->set_active_value(
          res.loc(), cond.tape()->globalTapeVars_.actStore[arg1.loc()]);

    else
      cond.tape()->set_active_value(
          res.loc(), cond.tape()->globalTapeVars_.actStore[arg2.loc()]);

  } else
    cond.tape()->set_active_value(res.loc(),
                                  cond.tape()->get_active_value(cond.loc()));

#endif
}

void condassign(adouble &res, const adouble &cond, const adouble &arg) {

  if (cond.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (cond.tape()->get_active_value(cond.loc())) {
      if (!cond.tape()->get_active_value(arg.loc())) {

        const double temp = arg.value();
        if (temp == 0.0) {
          cond.tape()->put_op(assign_d_zero);
          cond.tape()->put_loc(arg.loc());
        } else if (temp == 1.0) {
          cond.tape()->put_op(assign_d_one);
          cond.tape()->put_loc(arg.loc());
        } else {
          cond.tape()->put_op(assign_d);
          cond.tape()->put_loc(arg.loc());
          cond.tape()->put_val(temp);
        }

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(arg.value());
      }

#endif

      cond.tape()->put_op(cond_assign_s);
      cond.tape()->put_loc(cond.loc()); // = arg
      cond.tape()->put_val(cond.value());
      cond.tape()->put_loc(arg.loc()); // = arg1
      cond.tape()->put_loc(res.loc()); // = res

      cond.tape()->increment_numTays_Tape();

      if (cond.tape()->keepTaylors())
        cond.tape()->write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0) {
        if (cond.tape()->get_active_value(arg.loc())) {

          cond.tape()->put_op(assign_a);
          cond.tape()->put_loc(arg.loc()); // = arg
          cond.tape()->put_loc(res.loc()); // = res

          cond.tape()->increment_numTays_Tape();

          if (cond.tape()->keepTaylors)
            cond.tape()->write_scaylor(res.value());

        } else {
          if (cond.tape()->get_active_value(res.loc())) {
            const double coval = arg.value();

            if (coval == 0) {
              cond.tape()->put_op(assign_d_zero);
              cond.tape()->put_loc(res.loc()); // = res
            } else if (coval == 1.0) {
              cond.tape()->put_op(assign_d_one);
              cond.tape()->put_loc(res.loc()); // = res
            } else {
              cond.tape()->put_op(assign_d);
              cond.tape()->put_loc(res.loc()); // = res
              cond.tape()->put_val(coval);     // = coval
            }

            cond.tape()->increment_numTays_Tape();

            if (cond.tape()->keepTaylors)
              cond.tape()->write_scaylor(res.value());
          }
        }
      }
    }
#endif
  }

  if (cond.value() > 0)
    res.value(arg.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!cond.tape()->get_active_value(cond.loc())) {
    if (cond.value() > 0)
      cond.tape()->set_active_value(res.loc(),
                                    cond.tape()->get_active_value(arg.loc()));

  } else
    cond.tape()->set_active_value(res.loc(),
                                  cond.tape()->get_active_value(cond.loc()));

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2) {

  if (cond.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (cond.tape()->get_active_value(cond.loc())) {
      if (!cond.tape()->globalTapeVars_.actStore[arg1.loc()]) {

        const double temp = arg1.value();

        if (temp == 0.0) {
          cond.tape()->put_op(assign_d_zero);
          cond.tape()->put_loc(arg1.loc());
        } else if (temp == 1.0) {
          cond.tape()->put_op(assign_d_one);
          cond.tape()->put_loc(arg1.loc());
        } else {
          cond.tape()->put_op(assign_d);
          cond.tape()->put_loc(arg1.loc());
          cond.tape()->put_val(temp);
        }

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(arg1.value());
      }

      if (!cond.tape()->globalTapeVars_.actStore[arg2.loc()]) {
        const double temp = arg2.value();

        if (temp == 0.0) {
          cond.tape()->put_op(assign_d_zero);
          cond.tape()->put_loc(arg2.loc());
        } else if (temp == 1.0) {
          cond.tape()->put_op(assign_d_one);
          cond.tape()->put_loc(arg2.loc());
        } else {
          cond.tape()->put_op(assign_d);
          cond.tape()->put_loc(arg2.loc());
          cond.tape()->put_val(temp);
        }

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(arg2.value());
      }
#endif

      cond.tape()->put_op(cond_eq_assign);
      cond.tape()->put_loc(cond.loc()); // = arg
      cond.tape()->put_val(cond.value());
      cond.tape()->put_loc(arg1.loc()); // = arg1
      cond.tape()->put_loc(arg2.loc()); // = arg2
      cond.tape()->put_loc(res.loc());  // = res

      cond.tape()->increment_numTays_Tape();

      if (cond.tape()->keepTaylors())
        cond.tape()->write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (cond.tape()->get_active_value(arg.loc())) {

        cond.tape()->put_op(assign_a);
        cond.tape()->put_loc(arg_loc);   // = arg
        cond.tape()->put_loc(res.loc()); // = res

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(res.value());

      } else {
        if (cond.tape()->get_active_value(res.loc())) {

          const double coval = cond.tape()->globalTapeVars_.store[arg_loc];

          if (coval == 0) {
            cond.tape()->put_op(assign_d_zero);
            cond.tape()->put_loc(res.loc()); // = res
          } else if (coval == 1.0) {
            cond.tape()->put_op(assign_d_one);
            cond.tape()->put_loc(res.loc()); // = res
          } else {
            cond.tape()->put_op(assign_d);
            cond.tape()->put_loc(res.loc()); // = res
            cond.tape()->put_val(coval);     // = coval
          }

          cond.tape()->increment_numTays_Tape();

          if (cond.tape()->keepTaylors)
            cond.tape()->write_scaylor(res.value());
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
  if (!cond.tape()->get_active_value(cond.loc())) {
    if (cond.value() > 0)
      cond.tape()->set_active_value(
          res.loc(), cond.tape()->globalTapeVars_.actStore[arg1.loc()]);

    else
      cond.tape()->set_active_value(
          res.loc(), cond.tape()->globalTapeVars_.actStore[arg2.loc()]);

  } else
    cond.tape()->set_active_value(res.loc(),
                                  cond.tape()->get_active_value(cond.loc()));

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg) {

  if (cond.tape()->traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (cond.tape()->get_active_value(cond.loc())) {
      if (!cond.tape()->get_active_value(arg.loc())) {

        const double temp = arg.value();
        if (temp == 0.0) {
          cond.tape()->put_op(assign_d_zero);
          cond.tape()->put_loc(arg.loc());
        } else if (temp == 1.0) {
          cond.tape()->put_op(assign_d_one);
          cond.tape()->put_loc(arg.loc());
        } else {
          cond.tape()->put_op(assign_d);
          cond.tape()->put_loc(arg.loc());
          cond.tape()->put_val(temp);
        }

        cond.tape()->increment_numTays_Tape();

        if (cond.tape()->keepTaylors)
          cond.tape()->write_scaylor(arg.value());
      }
#endif

      cond.tape()->put_op(cond_eq_assign_s);
      cond.tape()->put_loc(cond.loc()); // = arg
      cond.tape()->put_val(cond.value());
      cond.tape()->put_loc(arg.loc()); // = arg1
      cond.tape()->put_loc(res.loc()); // = res

      cond.tape()->increment_numTays_Tape();

      if (cond.tape()->keepTaylors())
        cond.tape()->write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0) {
        if (cond.tape()->get_active_value(arg.loc())) {

          cond.tape()->put_op(assign_a);
          cond.tape()->put_loc(arg.loc()); // = arg
          cond.tape()->put_loc(res.loc()); // = res

          cond.tape()->increment_numTays_Tape();

          if (cond.tape()->keepTaylors)
            cond.tape()->write_scaylor(res.value());

        } else {
          if (cond.tape()->get_active_value(res.loc())) {

            const double coval = arg.value();

            if (coval == 0) {
              cond.tape()->put_op(assign_d_zero);
              cond.tape()->put_loc(res.loc()); // = res
            } else if (coval == 1.0) {
              cond.tape()->put_op(assign_d_one);
              cond.tape()->put_loc(res.loc()); // = res
            } else {
              cond.tape()->put_op(assign_d);
              cond.tape()->put_loc(res.loc()); // = res
              cond.tape()->put_val(coval);     // = coval
            }

            cond.tape()->increment_numTays_Tape();

            if (cond.tape()->keepTaylors)
              cond.tape()->write_scaylor(res.value());
          }
        }
      }
    }
#endif
  }

  if (cond.value() >= 0)
    res.value(arg.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!cond.tape()->get_active_value(cond.loc())) {
    if (cond.value() > 0)
      cond.tape()->set_active_value(res.loc(),
                                    cond.tape()->get_active_value(arg.loc()));

  } else
    cond.tape()->set_active_value(res.loc(),
                                  cond.tape()->get_active_value(cond.loc()));

#endif
}

void adolc_vec_copy(adouble *const dest, const adouble *const src,
                    size_t size) {

  if (dest[size - 1].loc() - dest[0].loc() != size - 1 ||
      src[size - 1].loc() - src[0].loc() != size - 1)
    fail(ADOLC_ERRORS::ADOLC_VEC_LOCATIONGAP, std::source_location::current());
  if (dest[0].tape()->traceFlag()) {
    dest[0].tape()->put_op(vec_copy);
    dest[0].tape()->put_loc(src[0].loc());
    dest[0].tape()->put_loc(size);
    dest[0].tape()->put_loc(dest[0].loc());

    for (size_t i = 0; i < size; ++i) {
      dest[0].tape()->increment_numTays_Tape();

      if (dest[0].tape()->keepTaylors())
        dest[0].tape()->write_scaylor(dest[0].value());
    }
  }
  for (size_t i = 0; i < size; ++i)
    dest[0].tape()->set_ad_value(
        dest[0].loc() + i, dest[0].tape()->get_ad_value(src[0].loc() + i));
}

// requires a and b to be of size "size"
adouble adolc_vec_dot(const adouble *const vec_a, const adouble *const vec_b,
                      size_t size) {

  if (vec_a[size - 1].loc() - vec_a[0].loc() != size - 1 ||
      vec_b[size - 1].loc() - vec_b[0].loc() != size - 1)
    fail(ADOLC_ERRORS::ADOLC_VEC_LOCATIONGAP, std::source_location::current());

  adouble ret_adouble{vec_a[0].tape()};

  if (vec_a[0].tape()->traceFlag()) {

    vec_a[0].tape()->put_op(vec_dot);
    vec_a[0].tape()->put_loc(vec_a[0].loc());
    vec_a[0].tape()->put_loc(vec_b[0].loc());
    vec_a[0].tape()->put_loc(size);
    vec_a[0].tape()->put_loc(ret_adouble.loc());

    vec_a[0].tape()->add_num_eq_prod(2 * size);
    vec_a[0].tape()->increment_numTays_Tape();

    if (vec_a[0].tape()->keepTaylors())
      vec_a[0].tape()->write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(0);

  for (size_t i = 0; i < size; ++i)
    ret_adouble.value(ret_adouble.value() +
                      vec_a[0].tape()->get_ad_value(vec_a[0].loc() + i) *
                          vec_b[0].tape()->get_ad_value(vec_b[0].loc() + i));

  return ret_adouble;
}

// requires res, b and c to be of size "size"
void adolc_vec_axpy(adouble *const res, const adouble &a,
                    const adouble *const vec_a, const adouble *const vec_b,
                    size_t size) {

  if (res[size - 1].loc() - res[0].loc() != size - 1 ||
      vec_a[size - 1].loc() - vec_a[0].loc() != size - 1 ||
      vec_b[size - 1].loc() - vec_b[0].loc() != size - 1)
    fail(ADOLC_ERRORS::ADOLC_VEC_LOCATIONGAP, std::source_location::current());
  if (vec_a[0].tape()->traceFlag()) {

    vec_a[0].tape()->put_op(vec_axpy);
    vec_a[0].tape()->put_loc(a.loc());
    vec_a[0].tape()->put_loc(vec_a[0].loc());
    vec_a[0].tape()->put_loc(vec_b[0].loc());
    vec_a[0].tape()->put_loc(size);
    vec_a[0].tape()->put_loc(res[0].loc());
    vec_a[0].tape()->add_num_eq_prod(2 * size - 1);

    for (size_t i = 0; i < size; ++i) {
      vec_a[0].tape()->increment_numTays_Tape();

      if (vec_a[0].tape()->keepTaylors())
        vec_a[0].tape()->write_scaylor(
            vec_a[0].tape()->get_ad_value(vec_a[0].loc() + i));
    }
  }
  for (size_t i = 0; i < size; ++i)
    res[0].tape()->set_ad_value(
        res[0].loc() + i,
        a.value() * vec_a[0].tape()->get_ad_value(vec_a[0].loc() + i) +
            vec_b[0].tape()->get_ad_value(vec_b[0].loc() + i));
}
