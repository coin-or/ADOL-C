/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     dvlparms.h
 Revision: $Id$
 Contents: Developer parameters:
           These parameters are intended for use by the developers and 
           maintainers of ADOL-C to specify library wide definitions.
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#if !defined(ADOLC_DVLPARMS_H)
#define ADOLC_DVLPARMS_H 1

/*--------------------------------------------------------------------------*/
/* Directory where tapes are created */
#define TAPE_DIR              "."
/* File names for the tapes */
#define ADOLC_TAYLORS_NAME    "ADOLC-Taylors_"
#define ADOLC_VALUES_NAME     "ADOLC-Values_"
#define ADOLC_LOCATIONS_NAME  "ADOLC-Locations_"
#define ADOLC_OPERATIONS_NAME "ADOLC-Operations_"

#ifdef _WINDOWS
#define PATHSEPARATOR         "\\"
#define __func__   __FUNCTION__
#define snprintf _snprintf
#else
#define PATHSEPARATOR         "/"
#endif

/*--------------------------------------------------------------------------*/
/* TAPE IDENTIFICATION (ADOLC & version check) */
#define statSpace   42
/* NOTE: ADOLC_ID and stats must fit in statSpace locints required!         */

/*--------------------------------------------------------------------------*/
/* ADOL-C configuration (never change this) */
#define adolc_overwrite 1
#define adolc_compsize >

/*--------------------------------------------------------------------------*/
#endif
