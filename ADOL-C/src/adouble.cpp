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

adouble::adouble() {
  ValueTape &tape = currentTape();

#if defined(ADOLC_ADOUBLE_STDCZERO)
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(assign_d_zero);
      tape.put_loc(loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(0.0);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), false);
#endif
#endif
}

adouble::adouble(double coval) {
  ValueTape &tape = currentTape();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.actStore[loc()]) {
#endif
      if (coval == 0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(loc()); // = res
      } else {
        tape.put_op(assign_d);
        tape.put_loc(loc()); // = res
        tape.put_val(coval); // = coval
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), false);
#endif
}

adouble::adouble(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif
      tape.put_op(assign_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(loc());   // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape.get_active_value(loc())) {
        const double coval = a.value();
        if (coval == 0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(loc()); // = res
        } else if (coval == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(loc()); // = res
        } else {
          tape.put_op(assign_d);
          tape.put_loc(loc()); // = res
          tape.put_val(coval); // = coval
        }

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());
      }
    }
#endif
  }

  value(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), tape.get_active_value(a.loc()));
#endif
}

/****************************************************************************/
/*                                                              ASSIGNMENTS */

adouble &adouble::operator=(double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      if (coval == 0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(loc()); // = res
      } else {
        tape.put_op(assign_d);
        tape.put_loc(loc()); // = res
        tape.put_val(coval); // = coval
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), false);
#endif
  return *this;
}

adouble &adouble::operator=(const adouble &a) {
  ValueTape &tape = currentTape();
  /* test this to avoid for x=x statements adjoint(x)=0 in reverse mode */
  if (loc() != a.loc()) {
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(assign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
      } else {
        if (tape.get_active_value(loc())) {
          double coval = a.value();
          if (coval == 0) {
            tape.put_op(assign_d_zero);
            tape.put_loc(loc()); // = res
          } else if (coval == 1.0) {
            tape.put_op(assign_d_one);
            tape.put_loc(loc()); // = res
          } else {
            tape.put_op(assign_d);
            tape.put_loc(loc()); // = res
            tape.put_val(coval); // = coval
          }
          tape.increment_numTays_Tape();
          if (tape.keepTaylors())
            tape.write_scaylor(value());
        }
      }
#endif
    }
    value(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    tape.set_active_value(loc(), tape.get_active_value(a.loc()));
#endif
  }
  return *this;
}

adouble &adouble::operator=(const pdouble &p) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

    tape.put_op(assign_p);
    tape.put_loc(p.loc());
    tape.put_loc(loc());

    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(value());
  }
  value(p.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), true);
#endif
  return *this;
}

/****************************************************************************/
/*                       ARITHMETIC ASSIGNMENT                             */

adouble &adouble::operator+=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(eq_plus_d);
      tape.put_loc(loc()); // = res
      tape.put_val(coval); // = coval

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  value(value() + coval);
  return *this;
}

adouble &adouble::operator+=(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(loc())) {
#endif
      tape.put_op(eq_plus_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(loc());   // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {
      const double coval = value();
      if (coval) {
        tape.put_op(plus_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
        tape.put_val(coval);
      } else {
        tape.put_op(assign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());

    } else if (tape.get_active_value(loc())) {
      const double coval = a.value();
      if (coval) {
        tape.put_op(eq_plus_d);
        tape.put_loc(loc()); // = res
        tape.put_val(coval); // = coval

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());
      }
    }
#endif
  }
  value(value() + a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(
      loc(), tape.get_active_value(loc() || tape.get_active_value(a.loc())));
#endif
  return *this;
}

adouble &adouble::operator+=(adouble &&a) {
  ValueTape &tape = currentTape();
  int upd = 0;
  if (tape.traceFlag())
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc()))
#endif
    {
      // if the structure is a*=b*c the call optimizes the temp adouble away.
      upd = tape.upd_resloc_inc_prod(a.loc(), loc(), eq_plus_prod);
    }
  if (upd) {
    value(value() + a.value());
    if (tape.keepTaylors())
      tape.delete_scaylor(a.loc());
    tape.decrement_numTays_Tape();
    tape.increment_num_eq_prod();
  } else {
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc()) && tape.get_active_value(loc())) {
#endif
        tape.put_op(eq_plus_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());

#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (tape.get_active_value(a.loc())) {
        double coval = value();
        if (coval) {
          tape.put_op(plus_d_a);
          tape.put_loc(a.loc()); // = arg
          tape.put_loc(loc());   // = res
          tape.put_val(coval);
        } else {
          tape.put_op(assign_a);
          tape.put_loc(a.loc()); // = arg
          tape.put_loc(loc());   // = res
        }

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());

      } else if (tape.get_active_value(loc())) {
        double coval = a.value();
        if (coval) {
          tape.put_op(eq_plus_d);
          tape.put_loc(loc()); // = res
          tape.put_val(coval); // = coval

          tape.increment_numTays_Tape();
          if (tape.keepTaylors())
            tape.write_scaylor(value());
        }
      }
#endif
    }
    value(value() + a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    tape.set_active_value(loc(), tape.get_active_value(loc()) ||
                                     tape.get_active_value(a.loc()));
#endif
  }
  return *this;
}

adouble &adouble::operator-=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(eq_min_d);
      tape.put_loc(loc()); // = res
      tape.put_val(coval); // = coval

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  value(value() - coval);
  return *this;
}

adouble &adouble::operator-=(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(loc())) {
#endif
      tape.put_op(eq_min_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(loc());   // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {
      const double coval = value();
      if (coval) {
        tape.put_op(min_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
        tape.put_val(coval);
      } else {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());

    } else if (tape.get_active_value(loc())) {
      const double coval = a.value();
      if (coval) {
        tape.put_op(eq_min_d);
        tape.put_loc(loc()); // = res
        tape.put_val(coval); // = coval

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());
      }
    }
#endif
  }
  value(value() - a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(
      loc(), (tape.get_active_value(loc()) || tape.get_active_value(a.loc())));
#endif
  return *this;
}

adouble &adouble::operator-=(adouble &&a) {
  ValueTape &tape = currentTape();
  int upd = 0;
  if (tape.traceFlag())
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc()))
#endif
    {
      upd = tape.upd_resloc_inc_prod(a.loc(), loc(), eq_min_prod);
    }
  if (upd) {
    value(value() - a.value());
    if (tape.keepTaylors())
      tape.delete_scaylor(a.loc());
    tape.decrement_numTays_Tape();
    tape.increment_num_eq_prod();
  } else {
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc()) && tape.get_active_value(loc())) {
#endif
        tape.put_op(eq_min_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
      } else if (tape.get_active_value(a.loc())) {
        double coval = value();
        if (coval) {
          tape.put_op(min_d_a);
          tape.put_loc(a.loc()); // = arg
          tape.put_loc(loc());   // = res
          tape.put_val(coval);
        } else {
          tape.put_op(neg_sign_a);
          tape.put_loc(a.loc()); // = arg
          tape.put_loc(loc());   // = res
        }

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(value());

      } else if (tape.get_active_value(loc())) {
        double coval = a.value();
        if (coval) {
          tape.put_op(eq_min_d);
          tape.put_loc(loc()); // = res
          tape.put_val(coval); // = coval

          tape.increment_numTays_Tape();
          if (tape.keepTaylors())
            tape.write_scaylor(value());
        }
      }
#endif
    }
    value(value() - a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    tape.set_active_value(loc(), tape.get_active_value(loc()) ||
                                     tape.get_active_value(a.loc()));
#endif
  }

  return *this;
}

adouble &adouble::operator*=(const double coval) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(eq_mult_d);
      tape.put_loc(loc()); // = res
      tape.put_val(coval); // = coval

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(value() * coval);
  return *this;
}

adouble &adouble::operator*=(const adouble &a) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(loc())) {
#endif
      tape.put_op(eq_mult_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(loc());   // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {
      const double coval = value();
      if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
      } else if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(loc());   // = res
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());

    } else if (tape.get_active_value(loc())) {
      const double coval = a.value();
      tape.put_op(eq_mult_d);
      tape.put_loc(loc()); // = res
      tape.put_val(coval); // = coval

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
    }
#endif
  }

  value(value() * a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(
      loc(), (tape.get_active_value(loc()) || tape.get_active_value(a.loc())));
#endif
  return *this;
}
/****************************************************************************/
/*                       INCREMENT / DECREMENT                              */

adouble adouble::operator++(int) {
  ValueTape &tape = currentTape();
  // create adouble to store old state in it.
  adouble ret_adouble;

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(assign_a);
      tape.put_loc(loc());             // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape.get_active_value(ret_adouble.loc())) {
        const double coval = value();
        if (coval == 0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(ret_adouble.loc()); // = res
        } else if (coval == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(ret_adouble.loc()); // = res
        } else {
          tape.put_op(assign_d);
          tape.put_loc(ret_adouble.loc()); // = res
          tape.put_val(coval);             // = coval
        }

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(ret_adouble.value());
      }
    }
#endif
  }

  ret_adouble.value(value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), tape.get_active_value(loc()));
#endif

  // change input adouble to new state
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(incr_a);
      tape.put_loc(loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  const double val = value();
  value(val + 1);
  return ret_adouble;
}

adouble adouble::operator--(int) {
  ValueTape &tape = currentTape();
  // create adouble to store old state in it.
  adouble ret_adouble;

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(assign_a);
      tape.put_loc(loc());             // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape.get_active_value(ret_adouble.loc())) {
        const double coval = value();
        if (coval == 0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(ret_adouble.loc()); // = res
        } else if (coval == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(ret_adouble.loc()); // = res
        } else {
          tape.put_op(assign_d);
          tape.put_loc(ret_adouble.loc()); // = res
          tape.put_val(coval);             // = coval
        }

        tape.increment_numTays_Tape();
        if (tape.keepTaylors())
          tape.write_scaylor(ret_adouble.value());
      }
    }
#endif
  }

  ret_adouble.value(value());
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(loc()));
#endif

  // write new state into input adouble
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(decr_a);
      tape.put_loc(loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  value(value() - 1);
  return ret_adouble;
}

adouble &adouble::operator++() {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(incr_a);
      tape.put_loc(loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }
  value(value() + 1);
  return *this;
}

adouble &adouble::operator--() {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(loc())) {
#endif
      tape.put_op(decr_a);
      tape.put_loc(loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(value());
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
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    tape.increment_numInds();
    tape.put_op(assign_ind);

    tape.put_loc(loc()); // = res
    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(value());
  }
  value(input);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), true);
#endif
  return *this;
}

// Assign the coval of an adouble to a double reference and mark the adouble
// as dependent variable on the tape. At the end of the function, the double
// reference can be seen as output value of the function given by the trace
// of the adouble.
adouble &adouble::operator>>=(double &output) {
  ValueTape &tape = currentTape();
#if defined(ADOLC_TRACK_ACTIVITY)
  if (!tape.get_active_value(loc())) {
    fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) "
                      "as dependent.\n");
    const double coval = value();
    if (coval == 0.0) {
      tape.put_op(assign_d_zero);
      tape.put_loc(loc());
    } else if (coval == 1.0) {
      tape.put_op(assign_d_one);
      tape.put_loc(loc());
    } else {
      tape.put_op(assign_d);
      tape.put_loc(loc());
      tape.put_val(coval);
    }

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(value());
  }
#endif
  tape.traceFlag();

  if (tape.traceFlag()) {
    tape.increment_numDeps();

    tape.put_op(assign_dep);
    tape.put_loc(loc()); // = res
  }

  output = value();
  return *this;
}

void adouble::declareIndependent() {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {
    tape.increment_numInds();

    tape.put_op(assign_ind);
    tape.put_loc(loc()); // = res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(value());
  }
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(loc(), true);
#endif
}

void adouble::declareDependent() {
  ValueTape &tape = currentTape();
#if defined(ADOLC_TRACK_ACTIVITY)
  if (!tape.get_active_value(loc())) {
    fprintf(DIAG_OUT, "ADOL-C warning: marking an inactive variable (constant) "
                      "as dependent.\n");
    const double coval = value();
    if (coval == 0.0) {
      tape.put_op(assign_d_zero);
      tape.put_loc(loc());
    } else if (coval == 1.0) {
      tape.put_op(assign_d_one);
      tape.put_loc(loc());
    } else {
      tape.put_op(assign_d);
      tape.put_loc(loc());
      tape.put_val(coval);
    }

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(value());
  }
#endif
  if (tape.traceFlag()) {
    tape.increment_numDeps();

    tape.put_op(assign_dep);
    tape.put_loc(loc()); // = res
  }
}

/****************************************************************************/
/*                                                           INPUT / OUTPUT */

std::ostream &operator<<(std::ostream &out, const adouble &a) {

  return out << a.value() << "(a)";
}

std::istream &operator>>(std::istream &in, adouble &a) {
  ValueTape &tape = currentTape();
  double coval;
  in >> coval;
  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif
      if (coval == 0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc()); // = res
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);   // = coval
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    }
#endif
  }

  a.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_variable(a.loc(), false);
#endif
  return in;
}

/****************************************************************************/
/*                               COMPARISON                                 */

#ifdef ADOLC_ADVANCED_BRANCHING

adouble operator!=(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  adouble ret_adouble;
  const double res = static_cast<double>(a_coval != b_coval);

  if (tape.traceFlag()) {
    tape.put_op(neq_a_a);
    tape.put_loc(a.loc());           // arg
    tape.put_loc(b.loc());           // arg1
    tape.put_val(res);               // check for branch switch
    tape.put_loc(ret_adouble.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator!=(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval != b_coval);

  if (tape.traceFlag()) {
    tape.put_op(neq_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(a.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator!=(const adouble &a, adouble &&b) { return std::move(b) != a; }

adouble operator==(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval == b_coval);
  if (tape.traceFlag()) {
    tape.put_op(eq_a_a);
    tape.put_loc(a.loc());           // arg
    tape.put_loc(b.loc());           // arg1
    tape.put_val(res);               // check for branch switch
    tape.put_loc(ret_adouble.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator==(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval == b_coval);

  if (tape.traceFlag()) {
    tape.put_op(eq_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(a.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator==(const adouble &a, adouble &&b) { return std::move(b) == a; }

adouble operator<=(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);
  if (tape.traceFlag()) {
    tape.put_op(le_a_a);
    tape.put_loc(a.loc());           // arg
    tape.put_loc(b.loc());           // arg1
    tape.put_val(res);               // check for branch switch
    tape.put_loc(ret_adouble.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<=(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);

  if (tape.traceFlag()) {
    tape.put_op(le_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(a.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator<=(const adouble &a, adouble &&b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval <= b_coval);

  if (tape.traceFlag()) {
    tape.put_op(le_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(b.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(b.value());
  }

  b.value(res);
  return b;
}
adouble operator>=(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);
  if (tape.traceFlag()) {
    tape.put_op(ge_a_a);
    tape.put_loc(a.loc());           // arg
    tape.put_loc(b.loc());           // arg1
    tape.put_val(res);               // check for branch switch
    tape.put_loc(ret_adouble.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>=(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);

  if (tape.traceFlag()) {
    tape.put_op(ge_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(a.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>=(const adouble &a, adouble &&b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval >= b_coval);

  if (tape.traceFlag()) {
    tape.put_op(ge_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(b.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(b.value());
  }

  b.value(res);
  return b;
}

adouble operator<(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);
  if (tape.traceFlag()) {
    tape.put_op(lt_a_a);
    tape.put_loc(a.loc());           // arg
    tape.put_loc(b.loc());           // arg1
    tape.put_val(res);               // check for branch switch
    tape.put_loc(ret_adouble.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator<(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);

  if (tape.traceFlag()) {
    tape.put_op(lt_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(a.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator<(const adouble &a, adouble &&b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval < b_coval);

  if (tape.traceFlag()) {
    tape.put_op(lt_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(b.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(b.value());
  }

  b.value(res);
  return b;
}

adouble operator>(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);
  if (tape.traceFlag()) {
    tape.put_op(gt_a_a);
    tape.put_loc(a.loc());           // arg
    tape.put_loc(b.loc());           // arg1
    tape.put_val(res);               // check for branch switch
    tape.put_loc(ret_adouble.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }
  ret_adouble.value(res);
  return ret_adouble;
}

adouble operator>(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);

  if (tape.traceFlag()) {
    tape.put_op(gt_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(a.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(a.value());
  }

  a.value(res);
  return a;
}

adouble operator>(const adouble &a, adouble &&b) {
  ValueTape &tape = currentTape();
  const double a_coval = a.value();
  const double b_coval = b.value();
  const double res = static_cast<double>(a_coval > b_coval);

  if (tape.traceFlag()) {
    tape.put_op(gt_a_a);
    tape.put_loc(a.loc()); // arg
    tape.put_loc(b.loc()); // arg1
    tape.put_val(res);     // check for branch switch
    tape.put_loc(b.loc()); // res

    tape.increment_numTays_Tape();
    if (tape.keepTaylors())
      tape.write_scaylor(b.value());
  }

  b.value(res);
  return b;
}

#endif // ADOLC_ADVANCED_BRANCHING

bool operator!=(const adouble &a, const double coval) {
  ValueTape &tape = currentTape();
  if (coval)
    return (-coval + a != 0);
  else {
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(a.value() ? neq_zero : eq_zero);
        tape.put_loc(a.loc());
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
  ValueTape &tape = currentTape();
  if (coval)
    return (-coval + a == 0);
  else {
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(a.value() ? neq_zero : eq_zero);
        tape.put_loc(a.loc());
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
  ValueTape &tape = currentTape();
  if (coval)
    return (-coval + a <= 0);
  else {
    bool b = (a.value() <= 0);
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(b ? le_zero : gt_zero);
        tape.put_loc(a.loc());
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
  ValueTape &tape = currentTape();
  if (coval)
    return (-coval + a >= 0);
  else {
    bool b = (a.value() >= 0);
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(b ? ge_zero : lt_zero);
        tape.put_loc(a.loc());
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
  ValueTape &tape = currentTape();
  if (coval)
    return (-coval + a < 0);
  else {
    bool b = (a.value() < 0);
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(b ? lt_zero : ge_zero);
        tape.put_loc(a.loc());
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
  ValueTape &tape = currentTape();
  if (coval)
    return (-coval + a > 0);
  else {
    bool b = (a.value() > 0);
    if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
      if (tape.get_active_value(a.loc())) {
#endif
        tape.put_op(b ? gt_zero : le_zero);
        tape.put_loc(a.loc());
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
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = a.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif
      tape.put_op(pos_sign_a);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {
      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  ret_adouble.tape.set_active_value(ret_adouble.loc(),
                                    tape.get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a) {
  ValueTape &tape = currentTape();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(pos_sign_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (tape.get_active_value(a.loc())) {
      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }
  return a;
}

adouble operator-(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(neg_sign_a);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (-coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (-coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(-coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(-coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(neg_sign_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (-coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (-coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(-coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }
  a.value(-coval);
  return a;
}

/****************************************************************************/
/*                            BINARY OPERATORS                              */

adouble operator+(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = a.value() + b.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(plus_a_a);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      tape.put_op(plus_d_a);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(b.value());

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();

      if (coval) {
        tape.put_op(plus_d_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);
      } else {
        tape.put_op(pos_sign_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  ret_adouble.tape.set_active_variable(
      ret_adouble.loc(),
      (tape.get_active_value(a.loc()) || tape.get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator+(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double coval2 = a.value() + b.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(plus_a_a);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      tape.put_op(plus_d_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(b.value());

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();

      if (coval) {
        tape.put_op(plus_d_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);
      } else {
        tape.put_op(pos_sign_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), (tape.get_active_value(a.loc()) ||
                                  tape.get_active_value(b.loc())));
#endif
  return a;
}

adouble operator+(const double coval, const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = coval + a.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      if (coval) {
        tape.put_op(plus_d_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);             // = coval
      } else {
        tape.put_op(pos_sign_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble operator+(const double coval, adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval2 = coval + a.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      if (coval) {
        tape.put_op(plus_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);   // = coval
      } else {
        tape.put_op(pos_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc([a.loc()]);
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
  return a;
}

adouble operator-(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = a.value() - b.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {

#endif
      tape.put_op(min_a_a);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = -b.value();
      tape.put_op(plus_d_a);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();
      if (coval) {
        tape.put_op(min_d_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);
      } else {
        tape.put_op(neg_sign_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), (tape.get_active_value(a.loc()) ||
                                            tape.get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator-(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double coval2 = a.value() - b.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {

#endif
      tape.put_op(min_a_a);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = -b.value();
      tape.put_op(plus_d_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();
      if (coval) {
        tape.put_op(min_d_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);
      } else {
        tape.put_op(neg_sign_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(a.loc()) ||
                                     tape.get_active_value(b.loc()));
#endif
  return a;
}

adouble operator-(const double coval, const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = coval - a.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      if (coval) {
        tape.put_op(min_d_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);             // = coval
      } else {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble operator-(const double coval, adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval2 = coval - a.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      if (coval) {
        tape.put_op(min_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);   // = coval
      } else {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);

  return a;
}

adouble operator*(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = a.value() * b.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(mult_a_a);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = b.value();
      tape.put_op(mult_d_a);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.get_active_value(b.loc())) {
      const double coval = a.value();

      if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), (tape.get_active_value(a.loc()) ||
                                            tape.get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator*(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double coval2 = a.value() * b.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(mult_a_a);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = b.value();
      tape.put_op(mult_d_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(b.loc())) {
      const double coval = a.value();

      if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), (tape.get_active_value(a.loc()) ||
                                  tape.get_active_value(b.loc())));
#endif
  return a;
}

adouble operator*(const double coval, const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = coval * a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      } else if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);             // = coval
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[locat]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble operator*(const double coval, adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval2 = coval * a.value();
  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      } else if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);   // = coval
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[locat]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }
  a.value(coval2);
  return a;
}

adouble operator/(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = a.value() / b.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(div_a_a);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(b.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(a.loc());           // = arg
        tape.put_loc(ret_adouble.loc()); // = res
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();

      tape.put_op(div_d_a);
      tape.put_loc(b.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), (tape.get_active_value(a.loc()) ||
                                            tape.get_active_value(b.loc())));
#endif
  return ret_adouble;
}

adouble operator/(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  const double coval2 = a.value() / b.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(div_a_a);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(a.loc()); // = res
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();

      tape.put_op(div_d_a);
      tape.put_loc(b.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), (tape.get_active_value(a.loc()) ||
                                  tape.get_active_value(b.loc())));
#endif
  return a;
}

adouble operator/(const adouble &a, adouble &&b) {
  ValueTape &tape = currentTape();
  const double coval2 = a.value() / b.value();

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)

    if (tape.get_active_value(a.loc()) && tape.get_active_value(b.loc())) {
#endif

      tape.put_op(div_a_a);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(b.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(b.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      const double coval = 1.0 / b.value();

      if (coval == -1.0) {
        tape.put_op(neg_sign_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(b.loc()); // = res
      } else if (coval == 1.0) {
        tape.put_op(pos_sign_a);
        tape.put_loc(b.loc()); // = arg
        tape.put_loc(b.loc()); // = res
      } else {
        tape.put_op(mult_d_a);
        tape.put_loc(a.loc()); // = arg
        tape.put_loc(b.loc()); // = res
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(b.value());

    } else if (tape.get_active_value(b.loc())) {

      const double coval = a.value();

      tape.put_op(div_d_a);
      tape.put_loc(b.loc()); // = arg
      tape.put_loc(b.loc()); // = res
      tape.put_val(coval);

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(b.value());
    }
#endif
  }

  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(b.loc(), tape.get_active_value(a.loc()) ||
                                     tape.get_active_value(b.loc()));
#endif
  return b;
}

adouble operator/(const double coval, const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double coval2 = coval / a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(div_d_a);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);             // = coval

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble operator/(const double coval, adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval2 = coval / a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(div_d_a);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);   // = coval

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval2);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(a.loc()));
#endif
  return a;
}

/****************************************************************************/
/*                          UNARY OPERATIONS                                */

adouble exp(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::exp(a.value());
  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(exp_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }
  ret_adouble.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble exp(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::exp(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(exp_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);
#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(a.loc()));
#endif

  return a;
}

adouble log(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::log(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(log_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble log(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::log(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(log_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(a.loc()));
#endif

  return a;
}

adouble sqrt(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::sqrt(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(sqrt_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble sqrt(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::sqrt(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(sqrt_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(a.loc()));
#endif

  return a;
}

adouble cbrt(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::cbrt(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(cbrt_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble cbrt(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::cbrt(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(cbrt_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();
      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble sin(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double coval1 = ADOLC_MATH_NSP::sin(a.value());
  const double coval2 = ADOLC_MATH_NSP::cos(a.value());

  adouble b;

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(sin_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.add_numTays_Tape(2);

      if (tape.keepTaylors()) {
        tape.write_scaylor(b.value());
        tape.write_scaylor(ret_adouble.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

        if (coval1 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(ret_adouble.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(ret_adouble.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(ret_adouble.loc());
          tape.put_val(coval1);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(ret_adouble.value());
      }
      if (tape.get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(b.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(b.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(b.loc());
          tape.put_val(coval2);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(b.value());
      }
    }
#endif
  }
  ret_adouble.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(b.loc()) =
                                               tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble sin(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval1 = ADOLC_MATH_NSP::sin(a.value());
  const double coval2 = ADOLC_MATH_NSP::cos(a.value());

  adouble b;

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(sin_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.add_numTays_Tape(2);

      if (tape.keepTaylors()) {
        tape.write_scaylor(b.value());
        tape.write_scaylor(a.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (tape.get_active_value(a.loc())) {

        if (coval1 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(a.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(a.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(a.loc());
          tape.put_val(coval1);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(a.value());
      }
      if (tape.get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(b.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(b.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(b.loc());
          tape.put_val(coval2);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(b.value());
      }
    }
#endif
  }

  a.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(b.loc()) =
                                     tape.get_active_value(a.loc()));
#endif

  return a;
}

adouble cos(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;

  const double coval1 = ADOLC_MATH_NSP::cos(a.value());
  const double coval2 = ADOLC_MATH_NSP::sin(a.value());

  adouble b;

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(cos_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.add_numTays_Tape(2);

      if (tape.keepTaylors()) {
        tape.write_scaylor(b.value());
        tape.write_scaylor(ret_adouble.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

        if (coval1 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(ret_adouble.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(ret_adouble.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(ret_adouble.loc());
          tape.put_val(coval1);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(ret_adouble.value());
      }
      if (tape.get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(b.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(b.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(b.loc());
          tape.put_val(coval2);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(b.value());
      }
    }

#endif
  }

  ret_adouble.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(b.loc()) =
                                               tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble cos(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval1 = ADOLC_MATH_NSP::cos(a.value());
  const double coval2 = ADOLC_MATH_NSP::sin(a.value());

  adouble b;

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(cos_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.add_numTays_Tape(2);

      if (tape.keepTaylors()) {
        tape.write_scaylor(b.value());
        tape.write_scaylor(a.value());
      }

#if defined(ADOLC_TRACK_ACTIVITY)

    } else {
      if (tape.get_active_value(a.loc())) {

        if (coval1 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(a.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(a.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(a.loc());
          tape.put_val(coval1);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(a.value());
      }
      if (tape.get_active_value(b.loc())) {

        if (coval2 == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(b.loc());
        } else if (coval1 == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(b.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(b.loc());
          tape.put_val(coval2);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors())
          tape.write_scaylor(b.value());
      }
    }

#endif
  }

  a.value(coval1);
  b.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(b.loc()) =
                                     tape.get_active_value(a.loc()));
#endif

  return a;
}

adouble asin(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::asin(a.value());
  adouble b = 1.0 / sqrt(1.0 - a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(asin_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble asin(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::asin(a.value());
  adouble b = 1.0 / sqrt(1.0 - a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(asin_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), tape.get_active_value(a.loc()));
#endif

  return a;
}

adouble acos(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::acos(a.value());

  adouble b = -1.0 / sqrt(1.0 - a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(acos_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble acos(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::acos(a.value());

  adouble b = -1.0 / sqrt(1.0 - a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and an be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(acos_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble atan(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::atan(a.value());

  adouble b = 1.0 / (1.0 + a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(atan_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble atan(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::atan(a.value());

  adouble b = 1.0 / (1.0 + a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(atan_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble asinh(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP_ERF::asinh(a.value());
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(asinh_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble asinh(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP_ERF::asinh(a.value());
  adouble b = 1.0 / sqrt(1.0 + a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(asinh_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble acosh(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP_ERF::acosh(a.value());
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(acosh_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {
      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble acosh(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP_ERF::acosh(a.value());
  adouble b = 1.0 / sqrt(a * a - 1.0);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(acosh_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)

    } else if (tape.get_active_value(a.loc())) {
      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble atanh(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP_ERF::atanh(a.value());
  adouble b = 1.0 / (1.0 - a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(atanh_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble atanh(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP_ERF::atanh(a.value());
  adouble b = 1.0 / (1.0 - a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(atanh_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble erf(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP_ERF::erf(a.value());
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(erf_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble erf(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP_ERF::erf(a.value());
  adouble b =
      2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(erf_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble erfc(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP_ERF::erfc(a.value());
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(erfc_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble erfc(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP_ERF::erfc(a.value());
  adouble b =
      -2.0 / ADOLC_MATH_NSP_ERF::sqrt(ADOLC_MATH_NSP::acos(-1.0)) * exp(-a * a);

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    // a will have same activity as x and can be considered as second input
    // here
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(erfc_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble ceil(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::ceil(a.value());

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(ceil_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);             // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble ceil(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::ceil(a.value());

  if (tape.traceFlag()) {
#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(ceil_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);   // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble floor(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval = ADOLC_MATH_NSP::floor(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(floor_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);             // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif
  return ret_adouble;
}

adouble floor(adouble &&a) {
  ValueTape &tape = currentTape();
  const double coval = ADOLC_MATH_NSP::floor(a.value());

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(floor_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);   // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(coval);

  return a;
}

adouble fabs(const adouble &a) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double temp = ADOLC_MATH_NSP::fabs(a.value());
  const double coval = temp != a.value() ? 0.0 : 1.0;

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(abs_val);
      tape.put_loc(a.loc());           // arg
      tape.put_loc(ret_adouble.loc()); // res
      tape.put_val(coval);             // coval

      tape.increment_numTays_Tape();

      if (tape.no_min_max())
        tape.increment_numSwitches();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }
  ret_adouble.value(temp);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble fabs(adouble &&a) {
  ValueTape &tape = currentTape();
  const double temp = ADOLC_MATH_NSP::fabs(a.value());
  const double coval = temp != a.value() ? 0.0 : 1.0;

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(abs_val);
      tape.put_loc(a.loc()); // arg
      tape.put_loc(a.loc()); // res
      tape.put_val(coval);   // coval

      tape.increment_numTays_Tape();

      if (tape.no_min_max())
        tape.increment_numSwitches();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }
  a.value(temp);

  return a;
}

adouble fmin(const adouble &a, const adouble &b) {
  ValueTape &tape = currentTape();
  if (tape.no_min_max())
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (tape.traceFlag()) {
    if (tape.get_active_value(b.loc()) && !tape.get_active_value(a.loc())) {

      const double temp = a.value();

      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }

    if (tape.get_active_value(a.loc()) && !tape.get_active_value(b.loc())) {

      const double temp = b.value();
      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(b.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(b.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(b.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(b.value());
    }
  }
#endif

  adouble ret_adouble;

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.loc();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) || tape.get_active_value(b.loc())) {
#endif

      tape.put_op(min_op);
      tape.put_loc(a.loc());           // = arg1
      tape.put_loc(b.loc());           // = arg2
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);             // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (tmp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (tmp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(tmp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), (tape.get_active_value(a.loc()) ||
                                            tape.get_active_value(b.loc())));
#endif

  return ret_adouble;
}

adouble fmin(adouble &&a, const adouble &b) {
  ValueTape &tape = currentTape();
  if (tape.no_min_max())
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (tape.traceFlag()) {
    if (tape.get_active_value(b.loc()) && !tape.get_active_value(a.loc())) {

      const double temp = a.value();

      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }

    if (tape.get_active_value(a.loc()) && !tape.get_active_value(b.loc())) {

      const double temp = b.value();
      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(b.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(b.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(b.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(b.value());
    }
  }
#endif

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) || tape.get_active_value(b.loc())) {
#endif

      tape.put_op(min_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);   // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (tmp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (tmp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(tmp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }
#endif
  }

  a.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(a.loc(), (tape.get_active_value(a.loc()) ||
                                  tape.get_active_value(b.loc())));
#endif

  return a;
}

adouble fmin(const adouble &a, adouble &&b) {
  ValueTape &tape = currentTape();
  if (tape.no_min_max())
    return ((a + b - fabs(a - b)) / 2.0);

#if defined(ADOLC_TRACK_ACTIVITY)
  if (tape.traceFlag()) {
    if (tape.get_active_value(b.loc()) && !tape.get_active_value(a.loc())) {

      const double temp = a.value();

      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
    }

    if (tape.get_active_value(a.loc()) && !tape.get_active_value(b.loc())) {

      const double temp = b.value();
      if (temp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(b.loc());
      } else if (temp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(b.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(b.loc());
        tape.put_val(temp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(b.value());
    }
  }
#endif

  const double coval = b.value() < a.value() ? 0.0 : 1.0;
  const double tmp = b.value() < a.value() ? b.value() : a.value();

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc()) || tape.get_active_value(b.loc())) {
#endif

      tape.put_op(min_op);
      tape.put_loc(a.loc()); // = arg1
      tape.put_loc(b.loc()); // = arg2
      tape.put_loc(b.loc()); // = res
      tape.put_val(coval);   // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(b.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(b.loc())) {

      if (tmp == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(b.loc());
      } else if (tmp == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(b.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(b.loc());
        tape.put_val(tmp);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(b.value());
    }
#endif
  }

  b.value(tmp);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(b.loc(), (tape.get_active_value(a.loc()) ||
                                  tape.get_active_value(b.loc())));
#endif

  return b;
}

adouble pow(const adouble &a, const double coval) {
  ValueTape &tape = currentTape();
  adouble ret_adouble;
  const double coval2 = ADOLC_MATH_NSP::pow(a.value(), coval);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(pow_op);
      tape.put_loc(a.loc());           // = arg
      tape.put_loc(ret_adouble.loc()); // = res
      tape.put_val(coval);             // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.globalTapeVars_.actStore[ret_adouble.loc()]) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(ret_adouble.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(ret_adouble.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(ret_adouble.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(ret_adouble.value());
    }
#endif
  }

  ret_adouble.value(coval2);

#if defined(ADOLC_TRACK_ACTIVITY)
  tape.set_active_value(ret_adouble.loc(), tape.get_active_value(a.loc()));
#endif

  return ret_adouble;
}

adouble pow(adouble &&a, const double coval) {
  ValueTape &tape = currentTape();
  const double coval2 = ADOLC_MATH_NSP::pow(a.value(), coval);

  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(a.loc())) {
#endif

      tape.put_op(pow_op);
      tape.put_loc(a.loc()); // = arg
      tape.put_loc(a.loc()); // = res
      tape.put_val(coval);   // = coval

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else if (tape.get_active_value(a.loc())) {

      if (coval2 == 0.0) {
        tape.put_op(assign_d_zero);
        tape.put_loc(a.loc());
      } else if (coval2 == 1.0) {
        tape.put_op(assign_d_one);
        tape.put_loc(a.loc());
      } else {
        tape.put_op(assign_d);
        tape.put_loc(a.loc());
        tape.put_val(coval2);
      }

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(a.value());
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
    ValueTape &tape = currentTape();                                           \
    adouble temp;                                                              \
    adouble val;                                                               \
    integrand;                                                                 \
                                                                               \
    if (tape.traceFlag()) {                                                    \
      tape.put_op(gen_quad);                                                   \
      tape.put_loc(arg.loc());                                                 \
      tape.put_loc(val.loc());                                                 \
      tape.put_loc(temp.loc());                                                \
      tape.increment_numTays_Tape();                                           \
      if (tape.keepTaylors())                                                  \
        tape.write_scaylor(temp.value());                                      \
    }                                                                          \
    temp.value(func(arg.value()));                                             \
    if (tape.traceFlag()) {                                                    \
      tape.put_val(arg.value());                                               \
      tape.put_val(temp.value());                                              \
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
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(cond.loc())) {
      if (!tape.globalTapeVars_.actStore[arg1.loc()]) {

        const double temp = arg1.value();
        if (temp == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(arg1.loc());
        } else if (temp == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(arg1.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(arg1.loc());
          tape.put_val(temp);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(arg1.value());
      }

      if (!tape.globalTapeVars_.actStore[arg2.loc()]) {

        const double temp = arg2.value();

        if (temp == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(arg2.loc());
        } else if (temp == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(arg2.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(arg2.loc());
          tape.put_val(temp);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(arg2.value());
      }
#endif

      tape.put_op(cond_assign);
      tape.put_loc(cond.loc()); // = arg
      tape.put_val(cond.value());
      tape.put_loc(arg1.loc()); // = arg1
      tape.put_loc(arg2.loc()); // = arg2
      tape.put_loc(res.loc());  // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {
      if (cond.value() > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (tape.globalTapeVars_.actStore[arg_loc]) {

        tape.put_op(assign_a);
        tape.put_loc(arg_loc);   // = arg
        tape.put_loc(res.loc()); // = res

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(res.value());

      } else {
        if (tape.get_active_value(res.loc())) {

          const double coval = tape.globalTapeVars_.store[arg_loc];
          if (coval == 0) {
            tape.put_op(assign_d_zero);
            tape.put_loc(res.loc()); // = res
          } else if (coval == 1.0) {
            tape.put_op(assign_d_one);
            tape.put_loc(res.loc()); // = res
          } else {
            tape.put_op(assign_d);
            tape.put_loc(res.loc()); // = res
            tape.put_val(coval);     // = coval
          }

          tape.increment_numTays_Tape();

          if (tape.keepTaylors)
            tape.write_scaylor(res.value());
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
  if (!tape.get_active_value(cond.loc())) {
    if (cond.value() > 0)
      tape.set_active_value(res.loc(),
                            tape.globalTapeVars_.actStore[arg1.loc()]);

    else
      tape.set_active_value(res.loc(),
                            tape.globalTapeVars_.actStore[arg2.loc()]);

  } else
    tape.set_active_value(res.loc(), tape.get_active_value(cond.loc()));

#endif
}

void condassign(adouble &res, const adouble &cond, const adouble &arg) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(cond.loc())) {
      if (!tape.get_active_value(arg.loc())) {

        const double temp = arg.value();
        if (temp == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(arg.loc());
        } else if (temp == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(arg.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(arg.loc());
          tape.put_val(temp);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(arg.value());
      }

#endif

      tape.put_op(cond_assign_s);
      tape.put_loc(cond.loc()); // = arg
      tape.put_val(cond.value());
      tape.put_loc(arg.loc()); // = arg1
      tape.put_loc(res.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0) {
        if (tape.get_active_value(arg.loc())) {

          tape.put_op(assign_a);
          tape.put_loc(arg.loc()); // = arg
          tape.put_loc(res.loc()); // = res

          tape.increment_numTays_Tape();

          if (tape.keepTaylors)
            tape.write_scaylor(res.value());

        } else {
          if (tape.get_active_value(res.loc())) {
            const double coval = arg.value();

            if (coval == 0) {
              tape.put_op(assign_d_zero);
              tape.put_loc(res.loc()); // = res
            } else if (coval == 1.0) {
              tape.put_op(assign_d_one);
              tape.put_loc(res.loc()); // = res
            } else {
              tape.put_op(assign_d);
              tape.put_loc(res.loc()); // = res
              tape.put_val(coval);     // = coval
            }

            tape.increment_numTays_Tape();

            if (tape.keepTaylors)
              tape.write_scaylor(res.value());
          }
        }
      }
    }
#endif
  }

  if (cond.value() > 0)
    res.value(arg.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!tape.get_active_value(cond.loc())) {
    if (cond.value() > 0)
      tape.set_active_value(res.loc(), tape.get_active_value(arg.loc()));

  } else
    tape.set_active_value(res.loc(), tape.get_active_value(cond.loc()));

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg1,
                  const adouble &arg2) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(cond.loc())) {
      if (!tape.globalTapeVars_.actStore[arg1.loc()]) {

        const double temp = arg1.value();

        if (temp == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(arg1.loc());
        } else if (temp == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(arg1.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(arg1.loc());
          tape.put_val(temp);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(arg1.value());
      }

      if (!tape.globalTapeVars_.actStore[arg2.loc()]) {
        const double temp = arg2.value();

        if (temp == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(arg2.loc());
        } else if (temp == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(arg2.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(arg2.loc());
          tape.put_val(temp);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(arg2.value());
      }
#endif

      tape.put_op(cond_eq_assign);
      tape.put_loc(cond.loc()); // = arg
      tape.put_val(cond.value());
      tape.put_loc(arg1.loc()); // = arg1
      tape.put_loc(arg2.loc()); // = arg2
      tape.put_loc(res.loc());  // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0)
        size_t arg_loc = arg1.loc();
      else
        size_t arg_loc = arg2.loc();

      if (tape.get_active_value(arg.loc())) {

        tape.put_op(assign_a);
        tape.put_loc(arg_loc);   // = arg
        tape.put_loc(res.loc()); // = res

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(res.value());

      } else {
        if (tape.get_active_value(res.loc())) {

          const double coval = tape.globalTapeVars_.store[arg_loc];

          if (coval == 0) {
            tape.put_op(assign_d_zero);
            tape.put_loc(res.loc()); // = res
          } else if (coval == 1.0) {
            tape.put_op(assign_d_one);
            tape.put_loc(res.loc()); // = res
          } else {
            tape.put_op(assign_d);
            tape.put_loc(res.loc()); // = res
            tape.put_val(coval);     // = coval
          }

          tape.increment_numTays_Tape();

          if (tape.keepTaylors)
            tape.write_scaylor(res.value());
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
  if (!tape.get_active_value(cond.loc())) {
    if (cond.value() > 0)
      tape.set_active_value(res.loc(),
                            tape.globalTapeVars_.actStore[arg1.loc()]);

    else
      tape.set_active_value(res.loc(),
                            tape.globalTapeVars_.actStore[arg2.loc()]);

  } else
    tape.set_active_value(res.loc(), tape.get_active_value(cond.loc()));

#endif
}

void condeqassign(adouble &res, const adouble &cond, const adouble &arg) {
  ValueTape &tape = currentTape();
  if (tape.traceFlag()) {

#if defined(ADOLC_TRACK_ACTIVITY)
    if (tape.get_active_value(cond.loc())) {
      if (!tape.get_active_value(arg.loc())) {

        const double temp = arg.value();
        if (temp == 0.0) {
          tape.put_op(assign_d_zero);
          tape.put_loc(arg.loc());
        } else if (temp == 1.0) {
          tape.put_op(assign_d_one);
          tape.put_loc(arg.loc());
        } else {
          tape.put_op(assign_d);
          tape.put_loc(arg.loc());
          tape.put_val(temp);
        }

        tape.increment_numTays_Tape();

        if (tape.keepTaylors)
          tape.write_scaylor(arg.value());
      }
#endif

      tape.put_op(cond_eq_assign_s);
      tape.put_loc(cond.loc()); // = arg
      tape.put_val(cond.value());
      tape.put_loc(arg.loc()); // = arg1
      tape.put_loc(res.loc()); // = res

      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(res.value());

#if defined(ADOLC_TRACK_ACTIVITY)
    } else {

      if (cond.value() > 0) {
        if (tape.get_active_value(arg.loc())) {

          tape.put_op(assign_a);
          tape.put_loc(arg.loc()); // = arg
          tape.put_loc(res.loc()); // = res

          tape.increment_numTays_Tape();

          if (tape.keepTaylors)
            tape.write_scaylor(res.value());

        } else {
          if (tape.get_active_value(res.loc())) {

            const double coval = arg.value();

            if (coval == 0) {
              tape.put_op(assign_d_zero);
              tape.put_loc(res.loc()); // = res
            } else if (coval == 1.0) {
              tape.put_op(assign_d_one);
              tape.put_loc(res.loc()); // = res
            } else {
              tape.put_op(assign_d);
              tape.put_loc(res.loc()); // = res
              tape.put_val(coval);     // = coval
            }

            tape.increment_numTays_Tape();

            if (tape.keepTaylors)
              tape.write_scaylor(res.value());
          }
        }
      }
    }
#endif
  }

  if (cond.value() >= 0)
    res.value(arg.value());

#if defined(ADOLC_TRACK_ACTIVITY)
  if (!tape.get_active_value(cond.loc())) {
    if (cond.value() > 0)
      tape.set_active_value(res.loc(), tape.get_active_value(arg.loc()));

  } else
    tape.set_active_value(res.loc(), tape.get_active_value(cond.loc()));

#endif
}

void adolc_vec_copy(adouble *const dest, const adouble *const src,
                    size_t size) {
  ValueTape &tape = currentTape();
  if (dest[size - 1].loc() - dest[0].loc() != size - 1 ||
      src[size - 1].loc() - src[0].loc() != size - 1)
    ADOLCError::fail(ADOLCError::ErrorType::VEC_LOCATIONGAP, CURRENT_LOCATION);
  if (tape.traceFlag()) {
    tape.put_op(vec_copy);
    tape.put_loc(src[0].loc());
    tape.put_loc(size);
    tape.put_loc(dest[0].loc());

    for (size_t i = 0; i < size; ++i) {
      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(dest[0].value());
    }
  }
  for (size_t i = 0; i < size; ++i)
    tape.set_ad_value(dest[0].loc() + i, tape.get_ad_value(src[0].loc() + i));
}

// requires a and b to be of size "size"
adouble adolc_vec_dot(const adouble *const vec_a, const adouble *const vec_b,
                      size_t size) {
  ValueTape &tape = currentTape();
  if (vec_a[size - 1].loc() - vec_a[0].loc() != size - 1 ||
      vec_b[size - 1].loc() - vec_b[0].loc() != size - 1)
    ADOLCError::fail(ADOLCError::ErrorType::VEC_LOCATIONGAP, CURRENT_LOCATION);

  adouble ret_adouble;

  if (tape.traceFlag()) {

    tape.put_op(vec_dot);
    tape.put_loc(vec_a[0].loc());
    tape.put_loc(vec_b[0].loc());
    tape.put_loc(size);
    tape.put_loc(ret_adouble.loc());

    tape.add_num_eq_prod(2 * size);
    tape.increment_numTays_Tape();

    if (tape.keepTaylors())
      tape.write_scaylor(ret_adouble.value());
  }

  ret_adouble.value(0);

  for (size_t i = 0; i < size; ++i)
    ret_adouble.value(ret_adouble.value() +
                      tape.get_ad_value(vec_a[0].loc() + i) *
                          tape.get_ad_value(vec_b[0].loc() + i));

  return ret_adouble;
}

// requires res, b and c to be of size "size"
void adolc_vec_axpy(adouble *const res, const adouble &a,
                    const adouble *const vec_a, const adouble *const vec_b,
                    size_t size) {
  ValueTape &tape = currentTape();
  if (res[size - 1].loc() - res[0].loc() != size - 1 ||
      vec_a[size - 1].loc() - vec_a[0].loc() != size - 1 ||
      vec_b[size - 1].loc() - vec_b[0].loc() != size - 1)
    ADOLCError::fail(ADOLCError::ErrorType::VEC_LOCATIONGAP, CURRENT_LOCATION);
  if (tape.traceFlag()) {

    tape.put_op(vec_axpy);
    tape.put_loc(a.loc());
    tape.put_loc(vec_a[0].loc());
    tape.put_loc(vec_b[0].loc());
    tape.put_loc(size);
    tape.put_loc(res[0].loc());
    tape.add_num_eq_prod(2 * size - 1);

    for (size_t i = 0; i < size; ++i) {
      tape.increment_numTays_Tape();

      if (tape.keepTaylors())
        tape.write_scaylor(tape.get_ad_value(vec_a[0].loc() + i));
    }
  }
  for (size_t i = 0; i < size; ++i)
    tape.set_ad_value(res[0].loc() + i,
                      a.value() * tape.get_ad_value(vec_a[0].loc() + i) +
                          tape.get_ad_value(vec_b[0].loc() + i));
}
