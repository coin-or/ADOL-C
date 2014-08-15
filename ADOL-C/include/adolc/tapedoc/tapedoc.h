/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapedoc/tapedoc.h
 Revision: $Id$
 Contents: Contains declaration of tapedoc driver.
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel 
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#if !defined(ADOLC_TAPEDOC_TAPEDOC_H)
#define ADOLC_TAPEDOC_TAPEDOC_H 1

#include <adolc/internal/common.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                                 tape_doc */
/* tape_doc(tag, m, n, x[n], y[m])                                          */

ADOLC_DLL_EXPORT void tape_doc(short, int, int, double*, double*);


/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

#endif
