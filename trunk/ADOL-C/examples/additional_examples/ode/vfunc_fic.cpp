/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vfunc_fic.cpp
 Revision: $Id$
 Contents: example for function module containing the flow in a channel
 
   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) int depDim; 
     (4) void initProblemParameters( void )
     (5) void initIndependents( double* indEPS_ )
     (6) void originalVectorFunction( double* indEPS_, double* dEPS_ )
     (7) void tapingVectorFunction( int tag, double* indEPS_, double* dEPS_ )   

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _VFUNC_FIC_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <math.h>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "ficexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;
int depDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
int nIntern;
double r__;


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"FICEXAM Flow in a channel (ADOL-C Example)\n\n");

    /* number of indeps & deps */
    if (indepDim > 0)
        nIntern = indepDim;
    else {
        fprintf(stdout,"    number of independents/8 = ? ");
        fscanf(stdin,"%d",&nIntern);
        fprintf(stdout,"\n");
    }
    indepDim = nIntern*8;
    depDim = indepDim;
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    int i, j, var;
    double xt, h;
    h = 1.0/nIntern;
    xt = 0.0;
    for (i=1; i<=nIntern; i++) {
        var = 8*(i-1);
        indeps[var] = xt*xt*(3.0-2.0*xt);
        indeps[var+1] = 6.0 * xt * (1.0 - xt);
        indeps[var+2] = 6.0 * (1.0 -2.0*xt);
        indeps[var+3] = -12.0;
        for (j=1; j<=4; j++)
            indeps[var+3+j] = 0.0;
        xt = xt + h;
    }
    r__ = 0;
}


/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                         The fic function */
int pdficfj ( int n, double* x, double* fvec, double r__, int nIntern ) {   /* Initialized data */
    static double rho[4] = { .0694318413734436035, .330009490251541138,
                             .66999053955078125,   .930568158626556396 };

    /* System generated locals */
    int i__1;

    /* Local variables */
    static double h__;
    static int i__, j, k, m;
    static double w[5], nf, hm, dw[40]  /* was [5][8] */, rhnfhk[1280]
    /* was [4][8][8][5] */, rhoijh;
    static int eqn, var;
    --fvec;
    --x;

    /* Function Body */
    /*     Initialization. */
    h__ = 1. / (double) (nIntern);
    hm = 1.;
    for (m = 0; m <= 4; ++m) {
        for (i__ = 1; i__ <= 4; ++i__) {
            rhoijh = hm;
            for (j = 0; j <= 7; ++j) {
                nf = 1.;
                for (k = 0; k <= 7; ++k) {
                    rhnfhk[i__ + (j + (k + (m << 3) << 3) << 2) - 1] = rhoijh
                            / nf;
                    nf *= (double) (k + 1);
                }
                rhoijh *= rho[i__ - 1];
            }
        }
        hm *= h__;
    }
    /*     Evaluate the function */
    /*     Initialize arrays. */
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
        fvec[j] = 0.;
    }
    for (k = 1; k <= 8; ++k) {
        for (j = 1; j <= 5; ++j) {
            dw[j + k * 5 - 6] = 0.;
        }
    }
    /*     Set up the boundary equations at t = 0.  u(0) = 0, u'(0) = 0. */
    fvec[1] = x[1];
    fvec[2] = x[2];
    i__1 = nIntern;
    for (i__ = 1; i__ <= i__1; ++i__) {
        var = i__ - 1 << 3;
        /*        Set up the collocation equations. */
        eqn = var + 2;
        for (k = 1; k <= 4; ++k) {
            for (m = 1; m <= 5; ++m) {
                w[m - 1] = 0.;
                for (j = m; j <= 4; ++j) {
                    w[m - 1] += rhnfhk[k + (j - m + (j - m + (j - m << 3) <<

                                                     3) << 2) - 1] * x[var + j];
                    dw[m + j * 5 - 6] = rhnfhk[k + (j - m + (j - m + (j - m
                                                    <<
                                                    3) << 3) << 2) - 1];
                }
                for (j = 1; j <= 4; ++j) {
                    w[m - 1] += rhnfhk[k + (j + 4 - m + (j + 4 - m + (4 - m
                                                         +
                                                         1 << 3) << 3) << 2) - 1] * x[var + 4 + j];
                    dw[m + (j + 4) * 5 - 6] = rhnfhk[k + (j + 4 - m + (j + 4

                                                          - m + (4 - m + 1 << 3) << 3) << 2) - 1];
                }
            }
            fvec[eqn + k] = w[4] - r__ * (w[1] * w[2] - w[0] * w[3]);
        }
        /*        Set up the continuity equations. */
        eqn = var + 6;
        for (m = 1; m <= 4; ++m) {
            w[m - 1] = 0.;
            for (j = m; j <= 4; ++j) {
                w[m - 1] += rhnfhk[(j - m + (j - m << 3)) * 32] * x[var + j]
                            ;

                dw[m + j * 5 - 6] = rhnfhk[(j - m + (j - m << 3)) * 32];
            }
            for (j = 1; j <= 4; ++j) {
                w[m - 1] += rhnfhk[(j + 4 - m + (4 - m + 1 << 3)) * 32] * x[
                                var + 4 + j];
                dw[m + (j + 4) * 5 - 6] = rhnfhk[(j + 4 - m + (4 - m + 1 <<
                                                  3)
                                                 ) * 32];
            }
        }
        if (i__ == nIntern) {
            goto L230;
        }
        for (m = 1; m <= 4; ++m) {
            fvec[eqn + m] = x[var + 8 + m] - w[m - 1];
        }
    }
    /*     Set up the boundary equations at t = 1.  u(1) = 1, u'(1) = 0. */
L230:
    fvec[n - 1] = w[0] - 1.;
    fvec[n] = w[1];
    return 0;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void originalVectorFunction( double* indeps, double* deps ) {
    pdficfj(indepDim,indeps,deps,r__,nIntern);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                  The active fic function */
int dficfj( int n, adouble* x, adouble* fvec, double r__, int nIntern ) {
    /* Initialized data */
    static adouble rho[4] = { .0694318413734436035, .330009490251541138,
                              .66999053955078125,   .930568158626556396 };

    /* System generated locals */
    int i__1;

    /* Local variables */
    static adouble h__;
    static int i__, j, k, m;
    static adouble nf, hm, dw[40]  /* was [5][8] */, rhnfhk[1280]
    /* was [4][8][8][5] */, rhoijh;
    static int eqn, var;
    adouble w[5];
    --fvec;
    --x;

    /* Function Body */
    /*     Initialization. */
    h__ = 1. / (double) (nIntern);
    hm = 1.;
    for (m = 0; m <= 4; ++m) {
        for (i__ = 1; i__ <= 4; ++i__) {
            rhoijh = hm;
            for (j = 0; j <= 7; ++j) {
                nf = 1.;
                for (k = 0; k <= 7; ++k) {
                    rhnfhk[i__ + (j + (k + (m << 3) << 3) << 2) - 1] = rhoijh
                            / nf;
                    nf *= (double) (k + 1);
                }
                rhoijh *= rho[i__ - 1];
            }
        }
        hm *= h__;
    }
    /*     Evaluate the function */
    /*     Initialize arrays. */
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
        fvec[j] = 0.;
    }
    for (k = 1; k <= 8; ++k) {
        for (j = 1; j <= 5; ++j) {
            dw[j + k * 5 - 6] = 0.;
        }
    }
    /*     Set up the boundary equations at t = 0.  u(0) = 0, u'(0) = 0. */
    fvec[1] = x[1];
    fvec[2] = x[2];
    i__1 = nIntern;
    for (i__ = 1; i__ <= i__1; ++i__) {
        var = i__ - 1 << 3;
        /*        Set up the collocation equations. */
        eqn = var + 2;
        for (k = 1; k <= 4; ++k) {
            for (m = 1; m <= 5; ++m) {
                w[m - 1] = 0.;
                for (j = m; j <= 4; ++j) {
                    w[m - 1] += rhnfhk[k + (j - m + (j - m + (j - m << 3) <<

                                                     3) << 2) - 1] * x[var + j];
                    dw[m + j * 5 - 6] = rhnfhk[k + (j - m + (j - m + (j - m
                                                    <<
                                                    3) << 3) << 2) - 1];
                }
                for (j = 1; j <= 4; ++j) {
                    w[m - 1] += rhnfhk[k + (j + 4 - m + (j + 4 - m + (4 - m
                                                         +
                                                         1 << 3) << 3) << 2) - 1] * x[var + 4 + j];
                    dw[m + (j + 4) * 5 - 6] = rhnfhk[k + (j + 4 - m + (j + 4

                                                          - m + (4 - m + 1 << 3) << 3) << 2) - 1];
                }
            }
            fvec[eqn + k] = w[4] - r__ * (w[1] * w[2] - w[0] * w[3]);
        }
        /*        Set up the continuity equations. */
        eqn = var + 6;
        for (m = 1; m <= 4; ++m) {
            w[m - 1] = 0.;
            for (j = m; j <= 4; ++j) {
                w[m - 1] += rhnfhk[(j - m + (j - m << 3)) * 32] * x[var + j]
                            ;
                dw[m + j * 5 - 6] = rhnfhk[(j - m + (j - m << 3)) * 32];
            }
            for (j = 1; j <= 4; ++j) {
                w[m - 1] += rhnfhk[(j + 4 - m + (4 - m + 1 << 3)) * 32] * x[
                                var + 4 + j];
                dw[m + (j + 4) * 5 - 6] = rhnfhk[(j + 4 - m + (4 - m + 1 <<
                                                  3)
                                                 ) * 32];
            }
        }
        if (i__ != nIntern) {
            for (m = 1; m <= 4; ++m) {
                fvec[eqn + m] = x[var + 8 + m] - w[m - 1];
            }
        }
    }
    /*     Set up the boundary equations at t = 1.  u(1) = 1, u'(1) = 0. */
    fvec[n - 1] = w[0] - 1.;
    fvec[n] = w[1];
    return 0;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void tapingVectorFunction( int tag, double* indeps, double* deps ) {
    int i;
    trace_on(tag);
    adouble* activeIndeps = new adouble[indepDim];
    adouble* activeDeps   = new adouble[depDim];
    adouble* aIP = activeIndeps;
    double*  iP  = indeps;
    for (i=0; i<indepDim; i++)
        *aIP++ <<= *iP++;
    dficfj(indepDim,activeIndeps,activeDeps,r__,nIntern);
    aIP = activeDeps;
    iP  = deps;
    for (i=0; i<depDim; i++)
        *aIP++ >>= *iP++;
    trace_off();
}

#undef _VFUNC_FIC_C_





