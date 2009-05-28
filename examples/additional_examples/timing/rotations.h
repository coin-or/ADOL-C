/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     rotations.h
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
#ifndef _ROTATIONS_H_
#define _ROTATIONS_H_

class adouble;

/****************************************************************************/
/*                                                     ELEMENTARY ROTATIONS */

/*--------------------------------------------------------------------------*/
void D1  ( double * vec, double & alpha );
void D1  ( double * depVec, double * indepVec, double & alpha );
void D1T ( double * vec, double & alpha );
void D1T ( double * depVec, double * indepVec, double & alpha );

/*--------------------------------------------------------------------------*/
void D2  ( double * vec, double & alpha );
void D2  ( double * depVec, double * indepVec, double & alpha );
void D2T ( double * vec, double & alpha );
void D2T ( double * depVec, double * indepVec, double & alpha );

/*--------------------------------------------------------------------------*/
void D3  ( double * vec, double & alpha );
void D3  ( double * depVec, double * indepVec, double & alpha );
void D3T ( double * vec, double & alpha );
void D3T ( double * depVec, double * indepVec, double & alpha );


/****************************************************************************/
/*                                           ACTIVATED ELEMENTARY ROTATIONS */

/*--------------------------------------------------------------------------*/
void D1  ( adouble * vec, double & alpha );
void D1  ( adouble * depVec, adouble * indepVec, double & alpha );
void D1T ( adouble * vec, double & alpha );
void D1T ( adouble * depVec, adouble * indepVec, double & alpha );
void D1  ( adouble * vec, adouble & alpha );
void D1  ( adouble * depVec, adouble * indepVec, adouble & alpha );
void D1T ( adouble * vec, adouble & alpha );
void D1T ( adouble * depVec, adouble * indepVec, adouble & alpha );

/*--------------------------------------------------------------------------*/
void D2  ( adouble * vec, double & alpha );
void D2  ( adouble * depVec, adouble * indepVec, double & alpha );
void D2T ( adouble * vec, double & alpha );
void D2T ( adouble * depVec, adouble * indepVec, double & alpha );
void D2  ( adouble * vec, adouble & alpha );
void D2  ( adouble * depVec, adouble * indepVec, adouble & alpha );
void D2T ( adouble * vec, adouble & alpha );
void D2T ( adouble * depVec, adouble * indepVec, adouble & alpha );

/*--------------------------------------------------------------------------*/
void D3  ( adouble * vec, double & alpha );
void D3  ( adouble * depVec, adouble * indepVec, double & alpha );
void D3T ( adouble * vec, double & alpha );
void D3T ( adouble * depVec, adouble * indepVec, double & alpha );
void D3  ( adouble * vec, adouble & alpha );
void D3  ( adouble * depVec, adouble * indepVec, adouble & alpha );
void D3T ( adouble * vec, adouble & alpha );
void D3T ( adouble * depVec, adouble * indepVec, adouble & alpha );


#endif

