/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts_p.h
 Revision: $Id$
 Contents: private functions and data types for extern (differentiated)
           functions.
 
 Copyright (c) Andreas Kowarz, Jean Utke
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
          
----------------------------------------------------------------------------*/

#if !defined(ADOLC_EXTERNFCTS_P_H)
#define ADOLC_EXTERNFCTS_P_H 1

#include <adolc/internal/common.h>
#include <adolc/externfcts.h>
#include <adolc/externfcts2.h>

BEGIN_C_DECLS
/****************************************************************************/
/*                                                         Now the C THINGS */

#define EDFCTS_BLOCK_SIZE 10

ext_diff_fct *get_ext_diff_fct(int index);
ext_diff_fct_v2 *get_ext_diff_fct_v2(int index);

END_C_DECLS

/****************************************************************************/

#endif /* ADOLC_EXTERNFCTS_P_H */

