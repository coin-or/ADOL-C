/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     revolve.c
 Revision: $Id$
 Contents: optimal binomial checkpointing adapted for ADOL-C

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/* -----
*   The function REVOLVE coded below is meant to be used as a        * 
*   "controller" for running a time-dependent applications program   *
*   in the reverse mode with checkpointing described in the paper    *
*   "Achieving logarithmic Growth in temporal and spatial complexity *
*   in reverse automatic differentiation", Optimization Methods and  *
*   Software,  Vol.1 pp. 35-54.                                      *
*   A postscript source of that paper can be found in the ftp sites  *
*        info.mcs.anl.gov and nbtf02.math.tu-dresden.de.             *
*   Apart from REVOLVE this file contains five auxiliary routines    * 
*   NUMFORW, EXPENSE, MAXRANGE, and ADJUST.                          *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   To utilize REVOLVE the user must have procedures for             *
*     - Advancing the state of the modeled system to a certain time. *
*     - Saving the current state onto a stack of snapshots.          *
*     - Restoring the the most recently saved snapshot and           *
*       restarting the forward simulation from there.                *
*     - Initializing the adjoints at the end of forward sweep.       *
*     - Performing one combined forward and adjoint step.            * 
*   Through an encoding of its return value REVOLVE asks the         *
*   calling program to perform one of these 'actions', which we will *
*   refer to as                                                      *
*                                                                    *
*       'advance', 'takeshot', 'restore', 'firsturn' and 'youturn'  .*
*   There are two other return values, namely                        *
*       'terminate'   and     'error'                                *
*   which indicate a regular or faulty termination of the calls      *
*   to REVOLVE.                                                      *
*                                                                    *
*   The action 'firsturn' includes a 'youturn', in that it requires  *
*     -advancing through the last time-step with recording           *
*      of intermediates                                              *
*     -initializing the adjoint values (possibly after               *
*      performing some IO)                                           *
*     -reversing the last time step using the record just written    *
*   The action 'firsturn' is obtained when the difference FINE-CAPO  *
*   has been reduced to 1 for the first time.                        *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   The calling sequence is                                          *
*                                                                    *
*               REVOLVE(CHECK,CAPO,FINE,SNAPS,INFO)                  *
*                                                                    *
*   with the return value being one of the actions to be taken. The  *
*   calling parameters are all integers with the following meaning   *
*                                                                    *
*         CHECK     number of checkpoint being written or retrieved  *
*         CAPO      beginning of subrange currently being processed  *
*         FINE      end of subrange currently being processed        *
*         SNAPS     upper bound on number of checkpoints taken       *
*         INFO      determines how much information will be printed  *
*                   and contains information about an error occurred *
*                                                                    *
*   Since REVOLVE involves only a few integer operations its         *
*   run-time is truly negligible within any nontrivial application.  *
*                                                                    *
*   The parameter SNAPS is selected by the user (possibly with the   *
*   help of the routines EXPENSE and ADJUST described below ) and    *
*   remains unchanged throughout.                                    *
*                                                                    *
*   The pair (CAPO,FINE) always represents the initial and final     *
*   state of the subsequence of time steps currently being traversed *
*   backwards.                                                       *
*                                                                    *
*   The conditions                                                   *
*                    CHECK >= -1      and     CAPO <= FINE           *
*   are necessary and sufficient for a regular response of REVOLVE.  *
*   If either condition is violated the value 'error' is returned.   *
*                                                                    *
*   The first call to REVOLVE must be with CHECK=-1 so that          * 
*   appropriate initializations can be performed internally.         *
*                                                                    *
*   When CHECK =-1 and CAPO = FINE  then 'terminate' is returned as  *
*   action value. This combination necessarily arises after a        *
*   sufficiently large number of calls to REVOLVE, which depends     * 
*   only on the initial difference FINE-CAPO.                        *
*                                                                    *
*   The last parameter INFO determines how much information about    *
*   the actions performed will be printed. When INFO =0 no           *
*   information is sent to standard output. When INFO > 0 REVOLVE    *
*   produces an output that contains a prediction of the number of   *    
*   forward steps and of the factor by which the execution will slow *    
*   down. When an error occurs, the return value of INFO contains    *
*   information about the reason:                                    *
*                                                                    *
*     INFO = 10: number of checkpoints stored exceeds CHECKUP,       *
*                increase constant CHECKUP and recompile             *
*     INFO = 11: number of checkpoints stored exceeds SNAPS, ensure  * 
*                SNAPS greater than 0 and increase initial FINE      *
*     INFO = 12: error occurs in NUMFORW                             *
*     INFO = 13: enhancement of FINE, SNAPS checkpoints stored,      *
*                SNAPS must be increased                             *
*     INFO = 14: number of SNAPS exceeds CHECKUP, increase constant  *
*                CHECKUP and recompile                               *
*     INFO = 15: number of REPS exceeds REPSUP, increase constant    *
*                REPSUP and recompile                                *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   Some further explanations and motivations:                       *
*                                                                    *
*   There is an implicit bound on CHECK through the dimensioning of  *
*   the integer array CH[CHEKUP] with CHECKUP = 64 being the default.*
*   If anybody wants to have that even larger he must change the     *
*   source. Also for the variable REPS an upper bound REPSUP is      *
*   defined. The default value equals 64. If during a call to        *
*   TREEVERSE a (CHECKUP+1)-st checkpoint would normally be called   * 
*   for then control is returned after an appropriate error message. * 
*   When the calculated REPS exceeds REPSUP also an error message    *
*   occurs.                                                          *
*   During the forward sweep the user is free to change the last     *
*   three parameters from call to call, except that FINE may never   *
*   be less than the current value of CAPO. This may be useful when  *
*   the total number of time STEPS to be taken is not a priori       *
*   known. The choice FINE=CAPO+1 initiates the reverse sweep, which * 
*   happens automatically if is left constant as CAPO is eventually  * 
*   moved up to it. Once the first reverse or restore action has     *
*   been taken only the last two parameters should be changed.       *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   The necessary number of forward steps without recording is       *
*   calculated by the function                                       *
*                                                                    *
*                      NUMFORW(STEPS,SNAPS)                          *
*                                                                    *
*   STEPS denotes the total number of time steps, i.e. FINE-CAPO     *
*   during the first call of REVOLVE. When SNAPS is less than 1 an   * 
*   error message will be given and -1 is returned as value.         *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   To choose an appropriated value of SNAPS the function            *
*                                                                    *
*                      EXPENSE(STEPS,SNAPS)                          *
*                                                                    *
*   estimates the run-time factor incurred by REVOLVE for a          *
*   particular value of SNAPS. The ratio NUMFORW(STEPS,SNAPS)/STEPS  *
*   is returned. This ratio corresponds to the run-time factor of    *
*   the execution relative to the run-time of one forward time step. *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   The auxiliary function                                           *
*                                                                    *
*                      MAXRANGE(SNAPS,REPS)                          *
*                                                                    *
*   returns the integer (SNAPS+REPS)!/(SNAPS!REPS!) provided         *
*   SNAPS >=0, REPS >= 0. Otherwise there will be appropriate error  *
*   messages and the value -1 will be returned. If the binomial      *
*   expression is not representable as a  signed 4 byte integer,     *
*   greater than 2^31-1, this maximal value is returned and a        *
*   warning message printed.                                         *
*                                                                    *
*--------------------------------------------------------------------*
*                                                                    *
*   Furthermore, the function                                        *
*                                                                    *
*                      ADJUST(STEPS)                                 *
*                                                                    *
*   is provided. It can be used to determine a value of SNAPS so     *
*   that the increase in spatial complexity equals approximately the *
*   increase in temporal complexity. For that ADJUST computes a      *
*   return value satisfying SNAPS ~= log_4 (STEPS) because of the    *
*   theory developed in the paper mentioned above.                   *
*                                                                    *
*--------------------------------------------------------------------*/

#include <adolc/revolve.h>
#include "taping_p.h"

#define MAXINT 2147483647

#ifndef _OPENMP
revolve_nums revolve_numbers;
#else
revolve_nums *revolve_numbers = NULL;
#endif

/* ************************************************************************* */

int numforw(int steps, int snaps) {
    int reps, range, num;

    if (snaps < 1) {
        printf(" error occurs in numforw: snaps < 1\n");
        return -1;
    }
    if (snaps > ADOLC_CHECKUP) {
        printf(" number of snaps=%d exceeds ADOLC_CHECKUP \n",snaps);
        printf(" redefine 'ADOLC_CHECKUP' \n");
        return -1;
    }
    reps = 0;
    range = 1;
    while(range < steps) {
        reps += 1;
        range = range*(reps + snaps)/reps;
    }
    printf("range =  %d \n",range);
    if (reps > ADOLC_REPSUP) {
        printf(" number of reps=%d exceeds ADOLC_REPSUP \n",reps);
        printf(" redefine 'ADOLC_REPSUP' \n");
        return -1;
    }
    num = reps * steps - range*reps/(snaps+1);
    return num;
}

/* ************************************************************************* */

double expense(int steps, int snaps) {
    double ratio;

    if (snaps < 1) {
        printf(" error occurs in expense: snaps < 0\n");
        return -1;
    }
    if (steps < 1) {
        printf(" error occurs in expense: steps < 0\n");
        return -1;
    }
    ratio = ((double) numforw(steps,snaps));
    if (ratio == -1)
        return -1;
    ratio = ratio/steps;
    return ratio;
}

/* ************************************************************************* */

int maxrange(int ss, int tt) {
    int i, ires;
    double res = 1.0;

    if((tt<0) || (ss<0)) {
        printf("error in MAXRANGE: negative parameter");
        return -1;
    }
    for(i=1; i<= tt; i++) {
        res *= (ss + i);
        res /= i;
        if (res > MAXINT) {
            ires=MAXINT;
            printf("warning from MAXRANGE: returned maximal integer %d\n",
                    ires);
            return ires;
        }
    }
    ires = res;
    return ires;
}

/* ************************************************************************* */

int adjust(int steps) {
    int snaps, s, reps;

    snaps = 1;
    reps = 1;
    s = 0;
    while( maxrange(snaps+s, reps+s) > steps )
        s--;
    while( maxrange(snaps+s, reps+s) < steps )
        s++;
    snaps += s;
    reps += s ;
    s = -1;
    while( maxrange(snaps,reps) >= steps ) {
        if (snaps > reps) {
            snaps -= 1;
            s = 0;
        } else {
            reps -= 1;
            s = 1;
        }
    }
    if ( s == 0 )
        snaps += 1 ;
    if ( s == 1 )
        reps += 1;
    return snaps;
}

/* ************************************************************************* */

enum revolve_action revolve
(int* check,int* capo,int* fine,int snaps,int* info) {
    int ds, oldcapo, num, bino1, bino2, bino3, bino4, bino5, bino6;
    /* (*capo,*fine) is the time range currently under consideration */
    /* ch[j] is the number of the state that is stored in checkpoint j */
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    REVOLVE_NUMBERS.commands += 1;
    if ((*check < -1) || (*capo > *fine)) {
        *info = 9;
        return revolve_error;
    }
    if ((*check == -1) && (*capo < *fine)) {
        if (*check == -1)
            REVOLVE_NUMBERS.turn = 0;   /* initialization of turn counter */
        *REVOLVE_NUMBERS.ch = *capo-1;
    }
    switch(*fine-*capo) {
        case 0:   /* reduce capo to previous checkpoint, unless done  */
            if(*check == -1 || *capo==*REVOLVE_NUMBERS.ch ) {
                *check -= 1;
                if (*info > 0) {
                    printf(" \n advances: %5d",REVOLVE_NUMBERS.advances);
                    printf(" \n takeshots: %4d",REVOLVE_NUMBERS.takeshots);
                    printf(" \n commands: %5d \n",REVOLVE_NUMBERS.commands);
                }
                return revolve_terminate;
            } else {
                *capo = REVOLVE_NUMBERS.ch[*check];
                REVOLVE_NUMBERS.oldfine = *fine;
                return revolve_restore;
            }
        case 1:  /* (possibly first) combined forward/reverse step */
            *fine -= 1;
            if(*check >= 0 && REVOLVE_NUMBERS.ch[*check] == *capo)
                *check -= 1;
            if(REVOLVE_NUMBERS.turn == 0) {
                REVOLVE_NUMBERS.turn = 1;
                REVOLVE_NUMBERS.oldfine = *fine;
                return revolve_firsturn;
            } else {
                REVOLVE_NUMBERS.oldfine = *fine;
                return revolve_youturn;
            }
        default:
            if(*check == -1 || REVOLVE_NUMBERS.ch[*check] != *capo) {
                *check += 1 ;
                if(*check >= ADOLC_CHECKUP) {
                    *info = 10;
                    return revolve_error;
                }
                if(*check+1 > snaps) {
                    *info = 11;
                    return revolve_error;
                }
                REVOLVE_NUMBERS.ch[*check] = *capo;
                if (*check == 0) {
                    REVOLVE_NUMBERS.advances = 0;
                    REVOLVE_NUMBERS.takeshots = 0;
                    REVOLVE_NUMBERS.commands = 1;
                    REVOLVE_NUMBERS.oldsnaps = snaps;
                    if (snaps > ADOLC_CHECKUP) {
                        *info = 14;
                        return revolve_error;
                    }
                    if (*info > 0) {
                        num = numforw(*fine-*capo,snaps);
                        if (num == -1) {
                            *info = 12;
                            return revolve_error;
                        }
                        printf(" prediction of needed forward steps: %8d => "
                                "\n",num);
                        printf(" slowdown factor: %8.4f \n\n",
                                ((double) num)/(*fine-*capo));
                    }
                }
                REVOLVE_NUMBERS.takeshots += 1;
                REVOLVE_NUMBERS.oldfine = *fine;
                return revolve_takeshot;
            } else {
                if ((REVOLVE_NUMBERS.oldfine < *fine) &&
                        (snaps == *check+1))
                {
                    *info = 13;
                    return revolve_error;
                }
                oldcapo = *capo;
                ds = snaps - *check;
                if (ds < 1) {
                    *info = 11;
                    return revolve_error;
                }
                REVOLVE_NUMBERS.reps = 0;
                REVOLVE_NUMBERS.range = 1;
                while(REVOLVE_NUMBERS.range < *fine - *capo) {
                    REVOLVE_NUMBERS.reps += 1;
                    REVOLVE_NUMBERS.range = REVOLVE_NUMBERS.range *
                        (REVOLVE_NUMBERS.reps + ds) / REVOLVE_NUMBERS.reps;
                }
                if (REVOLVE_NUMBERS.reps > ADOLC_REPSUP) {
                    *info = 15;
                    return revolve_error;
                }
                if (snaps != REVOLVE_NUMBERS.oldsnaps) {
                    if (snaps > ADOLC_CHECKUP) {
                        *info = 14;
                        return revolve_error;
                    }
                }

                bino1 = REVOLVE_NUMBERS.range * REVOLVE_NUMBERS.reps /
                    (ds+REVOLVE_NUMBERS.reps);
                bino2 = (ds > 1) ? bino1*ds/(ds+REVOLVE_NUMBERS.reps-1) : 1;
                if (ds == 1)
                    bino3 = 0;
                else
                    bino3 = (ds > 2) ? bino2 * (ds - 1) /
                        (ds + REVOLVE_NUMBERS.reps - 2) : 1;
                bino4 = bino2*(REVOLVE_NUMBERS.reps-1)/ds;
                if (ds < 3)
                    bino5 = 0;
                else
                    bino5 = (ds > 3) ? bino3*(ds-2)/REVOLVE_NUMBERS.reps : 1;

                bino6 = 0;

                /* range = beta(c,r) >= l (r -> min)
                 * bino1 = beta(c,r-1)
                 * bino2 = beta(c-1,r-1)
                 * bino3 = beta(c-2,r-1)
                 * bino4 = beta(c,r-2)
                 * bino5 = beta(c-3,r) */

                /* new version by A. Kowarz
                 * l^ as large as possible
                 *         bino6 = beta(c-1,r-2) 
                 
                        if (ds < 1)
                           bino6 = 0;
                        else
                           bino6 = (ds > 1) ? bino2*(reps-1)/(ds+reps-2) : 1;
                 
                        if (*fine-*capo>=range-bino5)
                           *capo += bino1;
                        else
                           if (*fine-*capo>bino1+bino2)
                              *capo = *fine-bino2-bino3;
                           else
                              if (*fine-*capo>=bino1+bino6)
                                 *capo += bino1-bino3;
                              else
                                 *capo = *fine-bino1+bino4; */

                /* new version by A. Kowarz
                 * l^ as small as possible 
                 *         bino6 = beta(c-1,r) */

                bino6 = bino1*ds/REVOLVE_NUMBERS.reps;

                if (*fine-*capo<=bino1+bino3)
                    *capo += bino4;
                else
                    if (*fine-*capo<bino1+bino2)
                        *capo = *fine-bino2-bino3;
                    else
                        if (*fine-*capo<=bino1+bino2+bino5)
                            *capo += bino1-bino3;
                        else
                            *capo = *fine-bino6;

                /* original by A. Walther
                 
                        if (*fine-*capo <= bino1 + bino3)
                          *capo = *capo+bino4;
                        else 
                         {
                          if (*fine-*capo >= range - bino5) 
                            *capo = *capo + bino1; 
                          else 
                             *capo = *fine-bino2-bino3;
                         } */

                if (*capo == oldcapo)
                    *capo = oldcapo+1;
                REVOLVE_NUMBERS.advances = REVOLVE_NUMBERS.advances +
                    *capo - oldcapo;
                REVOLVE_NUMBERS.oldfine = *fine;
                return revolve_advance;
            }
    }
}

