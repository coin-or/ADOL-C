/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparse_fo_rev.cpp
 Revision: $Id$
 Contents: All "Easy To Use" C++ interfaces of SPARSE package
 
 Copyright (c) Andrea Walther, Christo Mitev
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#include <adolc/sparse/sparse_fo_rev.h>
#include <adolc/interfaces.h>
#include "dvlparms.h"

#include <math.h>

#if defined(__cplusplus)

extern "C" void adolc_exit(int errorcode, const char *what, const char* function, const char *file, int line);

/****************************************************************************/
/*                                    Bit pattern propagation; general call */
/*                                                                          */
int forward( short              tag,
             int                m,
             int                n,
             int                p,
             double             *x,
             unsigned long int  **X,
             double             *y,
             unsigned long int  **Y,
             char               mode)
/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p], mode)                */
{
    int rc = -1;
    if (mode == 1) // tight version
        if (x != NULL)
            rc = int_forward_tight(tag,m,n,p,x,X,y,Y);
        else {
            fprintf(DIAG_OUT,"ADOL-C error:  no basepoint for bit"
                    " pattern forward tight.\n");
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
    else
        if (mode == 0) // safe version
            rc = int_forward_safe(tag,m,n,p,X,Y);
        else {
            fprintf(DIAG_OUT,"ADOL-C error:  bad mode parameter to bit"
                    " pattern forward.\n");
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
    return (rc);
}


/****************************************************************************/
/*                                    Bit pattern propagation; no basepoint */
/*                                                                          */
int forward( short              tag,
             int                m,
             int                n,
             int                p,
             unsigned long int  **X,
             unsigned long int  **Y,
             char               mode)
/* forward(tag, m, n, p, X[n][p], Y[m][p], mode)                            */
{
    if (mode != 0) // not safe
    { fprintf(DIAG_OUT,"ADOL-C error:  bad mode parameter to bit"
                  " pattern forward.\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    return int_forward_safe(tag,m,n,p,X,Y);
}



/****************************************************************************/
/*                                                                          */
/*                                    Bit pattern propagation, general call */
/*                                                                          */
int reverse( short             tag,
             int               m,
             int               n,
             int               q,
             unsigned long int **U,
             unsigned long int **Z,
             char              mode)
/* reverse(tag, m, n, q, U[q][m], Z[q][n]) */
{ int rc=-1;

    /* ! use better the tight version, the safe version supports no subscripts*/

    if (mode == 0) // safe version
        rc = int_reverse_safe(tag,m,n,q,U,Z);
    else
        if (mode == 1)
            rc = int_reverse_tight(tag,m,n,q,U,Z);
        else {
            fprintf(DIAG_OUT,"ADOL-C error:  bad mode parameter"
                    " to bit pattern reverse.\n");
            adolc_exit(-1,"",__func__,__FILE__,__LINE__);
        }
    return rc;
}


/****************************************************************************/

#endif
