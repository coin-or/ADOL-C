/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts_p.h
 Revision: $Id: externfcts_p.h 295 2009-02-25 13:32:25Z awalther $
 Contents: private functions and data types for extern (differentiated)
           functions.
 
 Copyright (c) Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
          
----------------------------------------------------------------------------*/

#if !defined(ADOLC_EXTERNFCTS_P_H)
#define ADOLC_EXTERNFCTS_P_H 1

#include <adolc/common.h>
#include <adolc/externfcts.h>

BEGIN_C_DECLS
/****************************************************************************/
/*                                                         Now the C THINGS */

#define EDFCTS_BLOCK_SIZE 10

ext_diff_fct *get_ext_diff_fct(int index);

void init_ext_diff_fct(ext_diff_fct *edfct);

END_C_DECLS

/****************************************************************************/

#endif /* ADOLC_EXTERNFCTS_P_H */

