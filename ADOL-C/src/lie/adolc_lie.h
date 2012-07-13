/**************************************************************************
 * File: adolc_lie.h
 *
 * AdolCLie - a library for computation of several kinds of Lie derivatives
 *
 * If you do compile this using C include adolc_lie_c.c in your project
 * or Makefile.
 *
 * If you do compile this using C++ include adolc_lie_c.c as well as
 * adolc_lie.cpp in your project or Makefile.
 *
 * You have to provide the path to adolc.h and adouble.h of the software
 * package ADOL-C in order to compile the files properly.
 *
 * You have to link this library against adolc.dll (under Windows) or
 * libadolc.so / libadolc.a (under Linux), respectively in order to
 * build a library or an excecutable.
 * 
 * Use these conditionals (on Windows only):
 * ADOLC_LIE_BUILD: Set this preprocessor definition if you build a 
 *                  library from the sources. Do not set this macro if you
 *                  include this header in sources that import functions from
 *                  the library.
 *
 * Authors: Siquian Wang, Klaus Röbenack, Jan Winkler
 ***************************************************************************/
#if !defined(ADOLC_LIE_TOOL_H)
#define ADOLC_LIE_TOOL_H

#include<adolc/common.h>


// C++ declarations available only when compiling with C++
#if defined(__cplusplus)
ADOLC_DLL_EXPORT int lie_scalar(short, short, short, double*, short, double*);
ADOLC_DLL_EXPORT int lie_scalar(short, short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradient(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradient(short, short, short, short, double*, short, double***);
ADOLC_DLL_EXPORT int lie_covector(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_bracket(short, short, short, double*, short, double**);
#endif



// C-declarations
#if defined (__cplusplus)
extern "C" {
#endif

ADOLC_DLL_EXPORT void accodeout(int, int, int, double***, double***, double***);
ADOLC_DLL_EXPORT void acccov(int, int, double***, double**, double**);          
ADOLC_DLL_EXPORT void accadj(int, int, double***, double***);                   
ADOLC_DLL_EXPORT void accbrac(int, int, double***, double**, double**);         
 
ADOLC_DLL_EXPORT int lie_scalarc(short, short, short, double*, short, double*);
ADOLC_DLL_EXPORT int lie_scalarcv(short, short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradientc(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_gradientcv(short, short, short, short, double*, short, double***);
ADOLC_DLL_EXPORT int lie_covectorv(short, short, short, double*, short, double**);
ADOLC_DLL_EXPORT int lie_bracketv(short, short, short, double*, short, double**);

#if defined (__cplusplus)
}
#endif

#endif

