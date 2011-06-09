/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     rotations.cpp
 Revision: $Id$
 Contents: 

   ... contains elementary rotations used by the machine tool 
       example of gearing (vfunc_pargear.C)
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#define _ROTATIONS_C_

/****************************************************************************/
/*                                                                 INCLUDES */
#include <adolc/adolc.h>
#include "rotations.h"

#include <cmath>

/****************************************************************************/
/*                                                     ELEMENTARY ROTATIONS */

/*--------------------------------------------------------------------------*/
void D1 ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec2=locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D1 ( double * depVec, double * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D1(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=indepVec[0];
    depVec[1]=locCos*indepVec[1] - locSin*indepVec[2];
    depVec[2]=locSin*indepVec[1] + locCos*indepVec[2];
}

void D1T ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec2=-locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D1T ( double * depVec, double * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D1T(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=indepVec[0];
    depVec[1]=locCos*indepVec[1] + locSin*indepVec[2];
    depVec[2]=-locSin*indepVec[1] + locCos*indepVec[2];
}

/*--------------------------------------------------------------------------*/
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

void D2T ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec2=locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2T ( double * depVec, double * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D2T(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] - locSin*indepVec[2];
    depVec[1]=indepVec[1];
    depVec[2]=locSin*indepVec[0] + locCos*indepVec[2];
}

/*--------------------------------------------------------------------------*/
void D3 ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec1=locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] - locSin*vec[1];
    vec[1]=tmpVec1;
}

void D3 ( double * depVec, double * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D3(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] - locSin*indepVec[1];
    depVec[1]=locSin*indepVec[0] + locCos*indepVec[1];
    depVec[2]=indepVec[2];
}

void D3T ( double * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    double tmpVec1=-locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] + locSin*vec[1];
    vec[1]=tmpVec1;
}

void D3T ( double * depVec, double * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D3T(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] + locSin*indepVec[1];
    depVec[1]=-locSin*indepVec[0] + locCos*indepVec[1];
    depVec[2]=indepVec[2];
}


/****************************************************************************/
/*                                           ACTIVATED ELEMENTARY ROTATIONS */

/*--------------------------------------------------------------------------*/
void D1 ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec2=locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D1 ( adouble * depVec, adouble * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D1(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=indepVec[0];
    depVec[1]=locCos*indepVec[1] - locSin*indepVec[2];
    depVec[2]=locSin*indepVec[1] + locCos*indepVec[2];
}

void D1 ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec2=locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D1 ( adouble * depVec, adouble * indepVec,  adouble & alpha ) {
    if ( indepVec == depVec ) {
        D1(depVec,alpha);
        return;
    }
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    depVec[0]=indepVec[0];
    depVec[1]=locCos*indepVec[1] - locSin*indepVec[2];
    depVec[2]=locSin*indepVec[1] + locCos*indepVec[2];
}

void D1T ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec2=-locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D1T ( adouble * depVec, adouble * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D1T(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=indepVec[0];
    depVec[1]=locCos*indepVec[1] + locSin*indepVec[2];
    depVec[2]=-locSin*indepVec[1] + locCos*indepVec[2];
}

void D1T ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec2=-locSin*vec[1] + locCos*vec[2];
    vec[1]=locCos*vec[1] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D1T ( adouble * depVec, adouble * indepVec,  adouble & alpha ) {
    if ( indepVec == depVec ) {
        D1T(depVec,alpha);
        return;
    }
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    depVec[0]=indepVec[0];
    depVec[1]=locCos*indepVec[1] + locSin*indepVec[2];
    depVec[2]=-locSin*indepVec[1] + locCos*indepVec[2];
}

/*--------------------------------------------------------------------------*/
void D2 ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec2=-locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] + locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2 ( adouble * depVec, adouble * indepVec,  double & alpha ) {
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

void D2T ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec2=locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2T ( adouble * depVec, adouble * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D2T(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] - locSin*indepVec[2];
    depVec[1]=indepVec[1];
    depVec[2]=locSin*indepVec[0] + locCos*indepVec[2];
}

void D2T ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec2=locSin*vec[0] + locCos*vec[2];
    vec[0]=locCos*vec[0] - locSin*vec[2];
    vec[2]=tmpVec2;
}

void D2T ( adouble * depVec, adouble * indepVec,  adouble & alpha ) {
    if ( indepVec == depVec ) {
        D2T(depVec,alpha);
        return;
    }
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] - locSin*indepVec[2];
    depVec[1]=indepVec[1];
    depVec[2]=locSin*indepVec[0] + locCos*indepVec[2];
}

/*--------------------------------------------------------------------------*/
void D3 ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec1=locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] - locSin*vec[1];
    vec[1]=tmpVec1;
}

void D3 ( adouble * depVec, adouble * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D3(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] - locSin*indepVec[1];
    depVec[1]=locSin*indepVec[0] + locCos*indepVec[1];
    depVec[2]=indepVec[2];
}

void D3 ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec1=locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] - locSin*vec[1];
    vec[1]=tmpVec1;
}

void D3 ( adouble * depVec, adouble * indepVec,  adouble & alpha ) {
    if ( indepVec == depVec ) {
        D3(depVec,alpha);
        return;
    }
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] - locSin*indepVec[1];
    depVec[1]=locSin*indepVec[0] + locCos*indepVec[1];
    depVec[2]=indepVec[2];
}

void D3T ( adouble * vec, double & alpha ) {
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    adouble tmpVec1=-locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] + locSin*vec[1];
    vec[1]=tmpVec1;
}

void D3T ( adouble * depVec, adouble * indepVec,  double & alpha ) {
    if ( indepVec == depVec ) {
        D3T(depVec,alpha);
        return;
    }
    double locCos=cos(alpha);
    double locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] + locSin*indepVec[1];
    depVec[1]=-locSin*indepVec[0] + locCos*indepVec[1];
    depVec[2]=indepVec[2];
}

void D3T ( adouble * vec, adouble & alpha ) {
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    adouble tmpVec1=-locSin*vec[0] + locCos*vec[1];
    vec[0]=locCos*vec[0] + locSin*vec[1];
    vec[1]=tmpVec1;
}

void D3T ( adouble * depVec, adouble * indepVec,  adouble & alpha ) {
    if ( indepVec == depVec ) {
        D3T(depVec,alpha);
        return;
    }
    adouble locCos=cos(alpha);
    adouble locSin=sin(alpha);
    depVec[0]=locCos*indepVec[0] + locSin*indepVec[1];
    depVec[1]=-locSin*indepVec[0] + locCos*indepVec[1];
    depVec[2]=indepVec[2];
}


#undef _ROTATIONS_C_
