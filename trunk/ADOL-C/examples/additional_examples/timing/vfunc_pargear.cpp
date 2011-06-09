/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vfunc_pargear.cpp
 Revision: $Id$
 Contents: Example of function module containing the machine tool example
           of gearing (parametrized version)
 
   Each << function module >> contains:
          
     (1) const char* const controlFileName 
     (2) int indepDim; 
     (3) int depDim; 
     (4) void initProblemParameters( void )
     (5) void initIndependents( double* indeps )
     (6) void originalVectorFunction( double* indeps, double* deps )
     (7) void tapingVectorFunction( int tag, double* indeps, double* deps )   
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _VFUNC_PARGEAR_C_


/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include "rotations.h"

#include <cmath>
#include <time.h>
#include <cstdlib>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */
#define Pi 3.141592654

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "pargearexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;
int depDim;
int radMotDegree;
int verMotDegree;
int horMotDegree;
int helMotDegree;
int angMotDegree;
int modRolDegree;


/*--------------------------------------------------------------------------*/
/*                                        important machine tool parameters */
// example bet06-23 convex pinion flank

int konvex=     1;          // konvexe oder konkave Flanke
int    zz=      6;          // Zaehnezahl

double xmk=    -17.50195;   // Messerversatz
double ymk=     80.0;       // MK-Versatz
double kopspw=   0.0;       // Kopfspanwinkel
double flaspw=   0.0;       // Flankenspanwinkel
double meschw=   0.0;       // Messerschwenkwinkel
double flkrrd= 101.44158;   // Spitzenradius, Flugkreisr.
double e=        0.0;       // MK-Exzentrizitaet
double exzenw=   0.0;       // Exzentrizitaetswinkel
double thetas=   0.0;       // Messerkopfschwenkung
double thetan=   0.0;       // Messerkopfneigung
double xmw=     24.63350;   // MK-x
double ymw=    -73.69500;   // MK-y
double zmw=     96.15919;   // MK-z
double thetaw=-314.52420;   // Wiegenwinkel=Waelztrommelwinkel
double m=       0.0;        // Erzeugungs-Achsversatz
double zwr=     0.0;        // Verschiebung Werkradachse
double delta=   14.62090;   // Kegelwinkel
double omega=   0.0;
double c=       0.0;
double r=        2.1;       // Kopfradius
double rs=    2594.425;     // Sphaerikradius
double ys=     876.147;     // Sphaerik-Mitte-Y
double zs=   -2442.015;     // Sphaerik-Mitte-Z

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
//static unsigned short int dx[3]; /* variable needed by erand48(.) */


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"PARGEAREXAM (ADOL-C Example)\n\n");

    /* number of indeps & deps */
    depDim   = 3;
    indepDim = 3;

    fprintf(stdout,"   Radial motion degree = ? [-1=no polynomial,0,...,6]");
    fscanf(stdin,"%d",&radMotDegree);
    fprintf(stdout,"\n");
    if (radMotDegree>=0)
        indepDim += radMotDegree + 1;

    fprintf(stdout,"   Vertical motion degree = ? ");
    fscanf(stdin,"%d",&verMotDegree);
    fprintf(stdout,"\n");
    if (verMotDegree>=0)
        indepDim += verMotDegree + 1;

    fprintf(stdout,"   Horizontal motion degree = ? ");
    fscanf(stdin,"%d",&horMotDegree);
    fprintf(stdout,"\n");
    if (horMotDegree>=0)
        indepDim += horMotDegree + 1;

    fprintf(stdout,"   Helical motion degree = ? ");
    fscanf(stdin,"%d",&helMotDegree);
    fprintf(stdout,"\n");
    if (helMotDegree>=0)
        indepDim += helMotDegree + 1;

    fprintf(stdout,"   Angular motion degree = ? ");
    fscanf(stdin,"%d",&angMotDegree);
    fprintf(stdout,"\n");
    if (angMotDegree>=0)
        indepDim += angMotDegree + 1;

    fprintf(stdout,"   Modified roll degree = ? ");
    fscanf(stdin,"%d",&modRolDegree);
    fprintf(stdout,"\n");
    if (modRolDegree>=0)
        indepDim += modRolDegree + 1;

    /* Init erand48(); */
    struct tm s;
    time_t t;
    time(&t);
    s=*localtime(&t);
    srand(s.tm_sec*s.tm_min);
    /*  dx[0]=rand();
      dx[1]=rand();
      dx[2]=rand();*/
}


/****************************************************************************/
/*                                                        INITIALIZE INDEPs */
void initIndependents( double* indeps ) {
    for (int i=0; i<indepDim; i++)
        indeps[i] = (double)rand();
}



/****************************************************************************/
/*                                                 ORIGINAL SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                       The model function */

/*--------------------------------------------------------------------------*/
// parametrized cutting edge
void def_messer(
    double * z,
    double * messer,
    // jetzt kommen die Parameter
    double hgR,   // Kopfradius
    double hgRs,  // Sphaerikradius
    double hgYs,  // Sphaerik-Mitte-Y
    double hgZs   // Sphaerik-Mitte-Z
) {
    double u0, uOri, phi0, h;

    phi0= asin((hgR+hgYs)/(hgR+hgRs));
    if (konvex==1) {
        u0=hgRs*phi0;
        uOri=1.0;
    } else {
        u0=hgRs*(phi0-Pi);
        uOri=-1.0;
    };

    h= (z[0]-u0)/(uOri*hgRs);
    messer[0]=hgZs+hgRs*cos(h);
    messer[1]=0.0;
    messer[2]=-hgYs-hgRs*sin(h);
}

/*--------------------------------------------------------------------------*/
// the main function
void gearFunction(
    double* z,       // (u,v,t) Parametrisierung der Bewegung der Messerschneide
    double* f,       // (x,y,z) Bewegte Messerschneide
    // jetzt kommen die ganzen Parameter
    double hgXmk,    // Messerversatz
    double hgYmk,    // MK-Versatz
    double hgKopSpW, // Kopfspanwinkel
    double hgFlaSpW, // Flankenspanwinkel
    double hgMeSchW, // Messerschwenkwinkel
    double hgFlKrRd, // Flugkreisradius
    double hgE,      // Exzentrizitaet
    double hgExzenW, // Exzentrizitaetswinkel
    double hgThetaS, // Messerkopfschwenkung
    double hgThetaN, // Messerkopfneigung
    double hgXmw,    // MK-x
    double hgYmw,    // MK-y
    double hgZmw,    // MK-z
    double hgThetaW, // Wiegenwinkel
    double hgM,      // Achsversatz
    double hgZwr,    // Verschiebung Werkradachse
    double hgDelta,  // Teilkegeloeffnungswinkel
    double hgOmega,  //
    double hgC,
    double hgR,      // Kopfradius
    double hgRs,     // Sphaerikradius
    double hgYs,     // Sphaerik-Mitte-Y
    double hgZs,     // Sphaerik-Mitte-Z
    // jetzt die Zusatzbewegungen
    int     radialMotionDegree,
    double* radialMotionCoeff,
    int     verticalMotionDegree,
    double* verticalMotionCoeff,
    int     horizontalMotionDegree,
    double* horizontalMotionCoeff,
    int     helicalMotionDegree,
    double* helicalMotionCoeff,
    int     angularMotionDegree,
    double* angularMotionCoeff,
    int     modifiedRollDegree,
    double* modifiedRollCoeff
) {
    int i;
    double ah;            // Hilfswert

    // Definition der Schneide
    def_messer(z,f,hgR,hgRs,hgYs,hgZs);

    // Position der Schneide am Messerkopf
    // (jetzt die Ber"ucksichtigung von hgKopSpW, hgFlaSpW, hgMeSchW)
    D2T(f,hgMeSchW);     // Messerschwenkwinkel Theta_M
    D3T(f,hgFlaSpW);     // Flankenspanwinkel Theta_F
    D1(f,hgKopSpW);      // Kopfspanwinkel Theta_K
    // Position der Schneide am Messerkopf
    f[0] += hgFlKrRd;    // Flugkreisradius
    f[1] -= hgXmk;       // Messerversatz

    // Messerkopfrotation mit Parameter v
    D3(f,z[1]);

    // Lage des Messerkopfs auf der Wiege
    f[2] -= hgYmk;

    // Beruecksichtigung der Messerkopf-Exzentrizitaet
    f[0] += hgE * cos(hgExzenW);
    f[1] -= hgE * sin(hgExzenW);

    // Eindrehen in Orientierung der Wiege
    ah = f[0];
    f[0] = f[1];
    f[1] = ah;
    f[2] = -f[2];

    // Beruecksichtigung von Messerkopf-Schwenkwinkel hgThetaS
    // und der Messerkopfneigung hgThetaN
    D3T(f,hgThetaS);     // Einschwenken in die Neigungsachse
    D1T(f,hgThetaN);     // Neigung um x-Achse
    D3(f,hgThetaS);      // Rueckschwenken aus der Neigungsachse

    // Verschiebung
    f[0] -= hgXmw;       // FLB1-x-Achse zeigt nach oben     -> (-xNeu)
    f[1] += hgZmw;       // FLB1-z-Achse zeigt nach rechts   ->  (yNeu)
    f[2] += hgYmw;       // FLB1-y-Achse zeigt aus der Wiege ->  (zNeu)

    // Wiegenwinkel thetaW, entspricht dem wert t=0
    D3(f,hgThetaW);

    // ZUSATZBEWEGUNG Radial motion
    if (radialMotionDegree >= 0) {
        ah = 0.0;
        for (i=radialMotionDegree; i>0; i--) {
            ah += radialMotionCoeff[i];
            ah *= z[2];
        }
        ah += radialMotionCoeff[0];
        f[1] += ah;        // radiale Verschiebung des Messerkopfes
    }

    // Wiegenbewegung mit Parameter t
    D3(f,z[2]);

    // ZUSATZBEWEGUNG Vertical motion
    if (verticalMotionDegree >= 0) {
        ah = 0.0;
        for (i=verticalMotionDegree; i>0; i--) {
            ah += verticalMotionCoeff[i];
            ah *= z[2];
        }
        ah += verticalMotionCoeff[0];
        f[0] += ah;        // Achsversatz in positive x-Richtung
    }

    // originaler Achsversatz
    f[0] += hgM;

    // ZUSATZBEWEGUNG Horizontal motion
    if (horizontalMotionDegree >= 0) {
        ah = 0.0;
        for (i=horizontalMotionDegree; i>0; i--) {
            ah += horizontalMotionCoeff[i];
            ah *= z[2];
        }
        ah += horizontalMotionCoeff[0];
        f[1] += ah;        // Achsversatz in positive y-Richtung
    }

    // ZUSATZBEWEGUNG Helical motion
    if (helicalMotionDegree >= 0) {
        ah = 0.0;
        for (i=helicalMotionDegree; i>0; i--) {
            ah += helicalMotionCoeff[i];
            ah *= z[2];
        }
        ah += helicalMotionCoeff[0];
        f[2] -= ah;        // Tiefenposition in negative z-Richtung
    }

    // Eindrehen in Orientierung des Werkrades
    f[0] = -f[0];
    ah = f[1];
    f[1] = -f[2];
    f[2] = -ah;

    // ZUSATZBEWEGUNG Angular motion
    if (angularMotionDegree >= 0) {
        ah = 0.0;
        for (i=angularMotionDegree; i>0; i--) {
            ah += angularMotionCoeff[i];
            ah *= z[2];
        }
        ah += angularMotionCoeff[0];
        D1(f,ah);        // umgekehrte Drehung um die x-Achse
    }

    // Teilkegeloeffnungswinkel delta - y-Achsen entgegengesetzt
    D1(f,hgDelta);

    // neue Verschiebung der Werkradachse
    f[2] += hgZwr; // z-Achse zeigt zu Spitze

    // ZUSATZBEWEGUNG Modified roll
    if (modifiedRollDegree >= 0) {
        ah = 0.0;
        for (i=modifiedRollDegree; i>1; i--) {
            ah += modifiedRollCoeff[i];
            ah *= z[2];
        }
        if (modifiedRollDegree > 0)
            ah += modifiedRollCoeff[1];
        ah += hgOmega;
        ah *= z[2];
        ah += modifiedRollCoeff[0];
    } else {
        ah = hgOmega;
        ah *= z[2];
    }
    ah += hgC*z[1];   // c*v + omega * t
    // gekoppelte Werkraddrehung in Abhaengigkeit von t und v
    D3(f,ah);
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void originalVectorFunction( double* indeps, double* deps ) {
    int  i = 3;
    double * radMotCoeff = indeps+i;
    i += radMotDegree;
    double * verMotCoeff = indeps+i;
    i += verMotDegree;
    double * horMotCoeff = indeps+i;
    i += horMotDegree;
    double * helMotCoeff = indeps+i;
    i += helMotDegree;
    double * angMotCoeff = indeps+i;
    i += angMotDegree;
    double * modRolCoeff = indeps+i;

    gearFunction(
        indeps,
        deps,
        // jetzt kommen die ganzen Parameter
        xmk,    // Messerversatz
        ymk,    // MK-Versatz
        kopspw, // Kopfspanwinkel
        flaspw, // Flankenspanwinkel
        meschw, // Messerschwenkwinkel
        flkrrd, // Flugkreisradius
        e,      // Exzentrizitaet
        exzenw, // Exzentrizitaetswinkel
        thetas, // Messerkopfschwenkung
        thetan, // Messerkopfneigung
        xmw,    // MK-x
        ymw,    // MK-y
        zmw,    // MK-z
        thetaw, // Wiegenwinkel
        m,      // Achsversatz
        zwr,    // Verschiebung Werkradachse
        delta,  // Teilkegeloeffnungswinkel
        omega,  //
        c,
        r,      // Kopfradius
        rs,     // Sphaerikradius
        ys,     // Sphaerik-Mitte-Y
        zs,     // Sphaerik-Mitte-Z
        // jetzt die Zusatzbewegungen
        radMotDegree,
        radMotCoeff,
        verMotDegree,
        verMotCoeff,
        horMotDegree,
        horMotCoeff,
        helMotDegree,
        helMotCoeff,
        angMotDegree,
        angMotCoeff,
        modRolDegree,
        modRolCoeff
    );
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */


/*--------------------------------------------------------------------------*/
/*                                                       The model function */

/*--------------------------------------------------------------------------*/
// parametrized cutting edge
void def_messer(
    adouble * z,
    adouble * messer,
    // jetzt kommen die Parameter
    double hgR,   // Kopfradius
    double hgRs,  // Sphaerikradius
    double hgYs,  // Sphaerik-Mitte-Y
    double hgZs   // Sphaerik-Mitte-Z
) {
    double u0, uOri, phi0;
    adouble h;

    phi0= asin((hgR+hgYs)/(hgR+hgRs));
    if (konvex==1) {
        u0=hgRs*phi0;
        uOri=1.0;
    } else {
        u0=hgRs*(phi0-Pi);
        uOri=-1.0;
    };

    h= (z[0]-u0)/(uOri*hgRs);
    messer[0]=hgZs+hgRs*cos(h);
    messer[1]=0.0;
    messer[2]=-hgYs-hgRs*sin(h);
}

/*--------------------------------------------------------------------------*/
// the main function
void activeGearFunction(
    adouble* z,      // (u,v,t) Parametrisierung der Bewegung der Messerschneide
    adouble* f,      // (x,y,z) Bewegte Messerschneide
    // jetzt kommen die ganzen Parameter
    double hgXmk,    // Messerversatz
    double hgYmk,    // MK-Versatz
    double hgKopSpW, // Kopfspanwinkel
    double hgFlaSpW, // Flankenspanwinkel
    double hgMeSchW, // Messerschwenkwinkel
    double hgFlKrRd, // Flugkreisradius
    double hgE,      // Exzentrizitaet
    double hgExzenW, // Exzentrizitaetswinkel
    double hgThetaS, // Messerkopfschwenkung
    double hgThetaN, // Messerkopfneigung
    double hgXmw,    // MK-x
    double hgYmw,    // MK-y
    double hgZmw,    // MK-z
    double hgThetaW, // Wiegenwinkel
    double hgM,      // Achsversatz
    double hgZwr,    // Verschiebung Werkradachse
    double hgDelta,  // Teilkegeloeffnungswinkel
    double hgOmega,  //
    double hgC,
    double hgR,      // Kopfradius
    double hgRs,     // Sphaerikradius
    double hgYs,     // Sphaerik-Mitte-Y
    double hgZs,     // Sphaerik-Mitte-Z
    // jetzt die Zusatzbewegungen
    int     radialMotionDegree,
    adouble* radialMotionCoeff,
    int     verticalMotionDegree,
    adouble* verticalMotionCoeff,
    int     horizontalMotionDegree,
    adouble* horizontalMotionCoeff,
    int     helicalMotionDegree,
    adouble* helicalMotionCoeff,
    int     angularMotionDegree,
    adouble* angularMotionCoeff,
    int     modifiedRollDegree,
    adouble* modifiedRollCoeff
) {
    int i;
    adouble ah;

    // Definition der Schneide
    def_messer(z,f,hgR,hgRs,hgYs,hgZs);

    // Position der Schneide am Messerkopf
    // (jetzt die Ber"ucksichtigung von hgKopSpW, hgFlaSpW, hgMeSchW)
    D2T(f,hgMeSchW);     // Messerschwenkwinkel Theta_M
    D3T(f,hgFlaSpW);     // Flankenspanwinkel Theta_F
    D1(f,hgKopSpW);      // Kopfspanwinkel Theta_K
    // Position der Schneide am Messerkopf
    f[0] += hgFlKrRd;    // Flugkreisradius
    f[1] -= hgXmk;       // Messerversatz

    // Messerkopfrotation mit Parameter v
    D3(f,z[1]);

    // Lage des Messerkopfs auf der Wiege
    f[2] -= hgYmk;

    // Beruecksichtigung der Messerkopf-Exzentrizitaet
    f[0] += hgE * cos(hgExzenW);
    f[1] -= hgE * sin(hgExzenW);

    // Eindrehen in Orientierung der Wiege
    ah = f[0];
    f[0] = f[1];
    f[1] = ah;
    f[2] = -f[2];

    // Beruecksichtigung von Messerkopf-Schwenkwinkel hgThetaS
    // und der Messerkopfneigung hgThetaN
    D3T(f,hgThetaS);     // Einschwenken in die Neigungsachse
    D1T(f,hgThetaN);     // Neigung um x-Achse
    D3(f,hgThetaS);      // Rueckschwenken aus der Neigungsachse

    // Verschiebung
    f[0] -= hgXmw;       // FLB1-x-Achse zeigt nach oben     -> (-xNeu)
    f[1] += hgZmw;       // FLB1-z-Achse zeigt nach rechts   ->  (yNeu)
    f[2] += hgYmw;       // FLB1-y-Achse zeigt aus der Wiege ->  (zNeu)

    // Wiegenwinkel thetaW, entspricht dem wert t=0
    D3(f,hgThetaW);

    // ZUSATZBEWEGUNG Radial motion
    if (radialMotionDegree >= 0) {
        ah = 0.0;
        for (i=radialMotionDegree; i>0; i--) {
            ah += radialMotionCoeff[i];
            ah *= z[2];
        }
        ah += radialMotionCoeff[0];
        f[1] += ah;        // radiale Verschiebung des Messerkopfes
    }

    // Wiegenbewegung mit Parameter t
    D3(f,z[2]);

    // ZUSATZBEWEGUNG Vertical motion
    if (verticalMotionDegree >= 0) {
        ah = 0.0;
        for (i=verticalMotionDegree; i>0; i--) {
            ah += verticalMotionCoeff[i];
            ah *= z[2];
        }
        ah += verticalMotionCoeff[0];
        f[0] += ah;        // Achsversatz in positive x-Richtung
    }

    // originaler Achsversatz
    f[0] += hgM;

    // ZUSATZBEWEGUNG Horizontal motion
    if (horizontalMotionDegree >= 0) {
        ah = 0.0;
        for (i=horizontalMotionDegree; i>0; i--) {
            ah += horizontalMotionCoeff[i];
            ah *= z[2];
        }
        ah += horizontalMotionCoeff[0];
        f[1] += ah;        // Achsversatz in positive y-Richtung
    }

    // ZUSATZBEWEGUNG Helical motion
    if (helicalMotionDegree >= 0) {
        ah = 0.0;
        for (i=helicalMotionDegree; i>0; i--) {
            ah += helicalMotionCoeff[i];
            ah *= z[2];
        }
        ah += helicalMotionCoeff[0];
        f[2] -= ah;        // Tiefenposition in negative z-Richtung
    }

    // Eindrehen in Orientierung des Werkrades
    f[0] = -f[0];
    ah = f[1];
    f[1] = -f[2];
    f[2] = -ah;

    // ZUSATZBEWEGUNG Angular motion
    if (angularMotionDegree >= 0) {
        ah = 0.0;
        for (i=angularMotionDegree; i>0; i--) {
            ah += angularMotionCoeff[i];
            ah *= z[2];
        }
        ah += angularMotionCoeff[0];
        D1(f,ah);        // umgekehrte Drehung um die x-Achse
    }

    // Teilkegeloeffnungswinkel delta - y-Achsen entgegengesetzt
    D1(f,hgDelta);

    // neue Verschiebung der Werkradachse
    f[2] += hgZwr; // z-Achse zeigt zu Spitze

    // ZUSATZBEWEGUNG Modified roll
    if (modifiedRollDegree >= 0) {
        ah = 0.0;
        for (i=modifiedRollDegree; i>1; i--) {
            ah += modifiedRollCoeff[i];
            ah *= z[2];
        }
        if (modifiedRollDegree > 0)
            ah += modifiedRollCoeff[1];
        ah += hgOmega;
        ah *= z[2];
        ah += modifiedRollCoeff[0];
    } else {
        ah = hgOmega;
        ah *= z[2];
    }
    ah += hgC*z[1];   // c*v + omega * t
    // gekoppelte Werkraddrehung in Abhaengigkeit von t und v
    D3(f,ah);
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

    i = 3;
    adouble * activeRadMotCoeff = activeIndeps+i;
    i += radMotDegree;
    adouble * activeVerMotCoeff = activeIndeps+i;
    i += verMotDegree;
    adouble * activeHorMotCoeff = activeIndeps+i;
    i += horMotDegree;
    adouble * activeHelMotCoeff = activeIndeps+i;
    i += helMotDegree;
    adouble * activeAngMotCoeff = activeIndeps+i;
    i += angMotDegree;
    adouble * activeModRolCoeff = activeIndeps+i;

    activeGearFunction(
        activeIndeps,
        activeDeps,
        // jetzt kommen die ganzen Parameter
        xmk,    // Messerversatz
        ymk,    // MK-Versatz
        kopspw, // Kopfspanwinkel
        flaspw, // Flankenspanwinkel
        meschw, // Messerschwenkwinkel
        flkrrd, // Flugkreisradius
        e,      // Exzentrizitaet
        exzenw, // Exzentrizitaetswinkel
        thetas, // Messerkopfschwenkung
        thetan, // Messerkopfneigung
        xmw,    // MK-x
        ymw,    // MK-y
        zmw,    // MK-z
        thetaw, // Wiegenwinkel
        m,      // Achsversatz
        zwr,    // Verschiebung Werkradachse
        delta,  // Teilkegeloeffnungswinkel
        omega,  //
        c,
        r,      // Kopfradius
        rs,     // Sphaerikradius
        ys,     // Sphaerik-Mitte-Y
        zs,     // Sphaerik-Mitte-Z
        // jetzt die Zusatzbewegungen
        radMotDegree,
        activeRadMotCoeff,
        verMotDegree,
        activeVerMotCoeff,
        horMotDegree,
        activeHorMotCoeff,
        helMotDegree,
        activeHelMotCoeff,
        angMotDegree,
        activeAngMotCoeff,
        modRolDegree,
        activeModRolCoeff
    );

    aIP = activeDeps;
    iP  = deps;
    for (i=0; i<depDim; i++)
        *aIP++ >>= *iP++;
    trace_off();

    delete [] activeDeps;
    delete [] activeIndeps;
}

#undef _VFUNC_GEAR_C_





