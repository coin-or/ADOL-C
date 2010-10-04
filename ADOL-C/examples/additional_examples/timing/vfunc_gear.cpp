/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     vfunc_gear.cpp
 Revision: $Id$
 Contents: Example of function module containing the machine tool example
            of gearing
 
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
#define _VFUNC_GEAR_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>

#include <cmath>
#include <time.h>
#include <cstdlib>


/****************************************************************************/
/*                                                         GLOBAL VARIABLES */

/*--------------------------------------------------------------------------*/
/*                                                        Control file name */
const char* controlFileName = "gearexam.ctrl";

/*--------------------------------------------------------------------------*/
/*                                                               Dimensions */
int indepDim;
int depDim;

/*--------------------------------------------------------------------------*/
/*                                       Other problem dependent parameters */
//static unsigned short int dx[3]; /* variable needed by erand48(.) */


/****************************************************************************/
/*                                                  INIT PROBLEM PARAMETERS */
void initProblemParameters( void ) {
    fprintf(stdout,"GEAREXAM (ADOL-C Example)\n\n");

    /* number of indeps & deps */
    indepDim = 3;
    depDim   = 3;

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
#define Pi 3.141592654

/*--------------------------------------------------------------------------*/
// important machine tool parameters

int konvex=     1;          // konvexe oder konkave Flanke

int zz=         13;         // Zaehnezahl
double delta=   16.46;      // Kegelwinkel
double st=      0.00001;    // Verschiebung Teilkegelspitze

double xmw=     79.33279;   // MK-x
double ymw=     -74.05;     // MK-y
double zmw=     159.49741;  // MK-z
double m=       0.0;        // Erzeugungs-Achsversatz
double zwr=     0.0;        // Reitstockeinstellung
double theta_w= -58.2253;   // Waelztrommelwinkel

double xmk=     17.49874;   // Messerversatz
int zm=         5;          // MK-Gangzahl
double ymk=     80.0;       // MK-Versatz

double rho=     101.64155;  // Spitzenradius, Flugkreisr.
double r=       2.1;        // Kopfradius
double rs=      2594.425;   // Sphaerikradius
double ys=      876.147;    // Sphaerik-Mitte-Y
double zs=      -2442.015;  // Sphaerik-Mitte-Z


/*--------------------------------------------------------------------------*/
// elementary rotations

void D1 ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec2=locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2 ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec2=-locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D3 ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec1=locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] - locSin*vec[1];
    vec[1]=tmpVec1;
}

void D1 ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec2=locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2 ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec2=-locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2 ( adouble *  depVec, adouble * indepVec,  adouble & alpha ) {
    if ( indepVec == depVec ) {
        D2(depVec,alpha);
        return;
    }
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] + locSin*indepVec[2];
    depVec[1]=indepVec[1];
    depVec[2]=-locSin*indepVec[0] + locCos*indepVec[2];
}

void D3 ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec1=locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] - locSin*vec[1];
    vec[1]=tmpVec1;
}

void D1 ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec2=locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2 ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec2=-locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2 ( double * depVec, double * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D2(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] + locSin*indepVec[2];
    depVec[1]=indepVec[1];
    depVec[2]=-locSin*indepVec[0] + locCos*indepVec[2];
}

void D3 ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec1=locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] - locSin*vec[1];
    vec[1]=tmpVec1;
}


/*--------------------------------------------------------------------------*/
// parametrized cutting edge

void def_messer(adouble *z, adouble *messer) {

    double u0, uOri, phi0;
    adouble h;

    phi0= asin((r+ys)/(r+rs));
    if (konvex==1) {
        u0=rs*phi0;
        uOri=1.0;
    } else {
        u0=rs*(phi0-Pi);
        uOri=-1.0;
    };

    h= (z[0]-u0)/(uOri*rs);
    messer[0]=zs+rs*cos(h);
    messer[1]=0.0;
    messer[2]=-ys-rs*sin(h);
}

void def_messer(double *z, double *messer) {

    double u0, uOri, phi0;
    double h;

    phi0= asin((r+ys)/(r+rs));
    if (konvex==1) {
        u0=rs*phi0;
        uOri=1.0;
    } else {
        u0=rs*(phi0-Pi);
        uOri=-1.0;
    };

    h= (z[0]-u0)/(uOri*rs);
    messer[0]=zs+rs*cos(h);
    messer[1]=0.0;
    messer[2]=-ys-rs*sin(h);
}

/*--------------------------------------------------------------------------*/
// the main function

void gearFunction(double* pz, double* pf) {
    double ah;
    double* messer;
    messer= new double[3];

    def_messer(pz, messer);

    // Position der Schneide am Messerkopf
    messer[0]+=rho;
    messer[1]-=xmk;

    // Messerkopfrotation mit Parameter v
    D3(messer,pz[1]);

    // Lage des Messerkopfs auf der Wiege
    messer[2]-=ymk;

    // Eindrehen in Orientierung der Wiege
    ah=messer[0];
    messer[0]=messer[1];
    messer[1]=ah;
    messer[2]=-messer[2];

    // Verschiebung
    messer[0]-=xmw;
    messer[1]+=zmw;
    messer[2]+=ymw;

    // Wiegenwinkel thetaW, entspricht dem wert t=0
    // + Wiegenbewegung mit Parameter t
    ah = theta_w+pz[2];
    D3(messer,ah);

    // Achsversatz
    messer[0]+=m;

    // Eindrehen in Orientierung des Werkrades
    messer[0]=-messer[0];
    ah=messer[1];
    messer[1]=-messer[2];
    messer[2]=-ah;

    // Teilkegeloeffnungswinkel delta, y-Achsen entgegengesetzt
    D1(messer,delta);

    // neue Verschiebung der Werkradachse
    messer[2]+=zwr+st;

    // gekoppelte Werkraddrehung in Abhaengigkeit von t und v
    ah = (1/sin(delta))*pz[2] + (double(zm)/zz)*pz[1];
    D2(pf,messer,ah);

    delete[] messer;
}

/*--------------------------------------------------------------------------*/
/*                                                   The interface function */
void originalVectorFunction( double* indeps, double* deps ) {
    gearFunction(indeps,deps);
}


/****************************************************************************/
/*                                                   TAPING SCALAR FUNCTION */

/*--------------------------------------------------------------------------*/
/*                                                       The model function */
void activeGearFunction(adouble* z, adouble* f) {
    adouble ah;

    adouble* messer = new adouble[3];
    def_messer(z, messer);

    // Position der Schneide am Messerkopf
    messer[0]+=rho;
    messer[1]-=xmk;

    // Messerkopfrotation mit Parameter v
    D3(messer,z[1]);

    // Lage des Messerkopfs auf der Wiege
    messer[2]-=ymk;

    // Eindrehen in Orientierung der Wiege
    ah=messer[0];
    messer[0]=messer[1];
    messer[1]=ah;
    messer[2]=-messer[2];

    // Verschiebung
    messer[0]-=xmw;
    messer[1]+=zmw;
    messer[2]+=ymw;

    // Wiegenwinkel thetaW, entspricht dem wert t=0
    // + Wiegenbewegung mit Parameter t
    ah = theta_w+z[2];
    D3(messer,ah);

    // Achsversatz
    messer[0]+=m;

    // Eindrehen in Orientierung des Werkrades
    messer[0]=-messer[0];
    ah=messer[1];
    messer[1]=-messer[2];
    messer[2]=-ah;

    // Teilkegeloeffnungswinkel delta, y-Achsen entgegengesetzt
    D1(messer,delta);

    // neue Verschiebung der Werkradachse
    messer[2]+=zwr+st;

    // gekoppelte Werkraddrehung in Abhaengigkeit von t und v
    ah = (1/sin(delta))*z[2] + (double(zm)/zz)*z[1];
    D2(f,messer,ah);

    delete[] messer;
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
    activeGearFunction(activeIndeps,activeDeps);
    aIP = activeDeps;
    iP  = deps;
    for (i=0; i<depDim; i++)
        *aIP++ >>= *iP++;
    trace_off();
}

#undef _VFUNC_GEAR_C_





