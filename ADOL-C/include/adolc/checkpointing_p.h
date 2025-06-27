/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     checkpointing_p.h
 Revision: $Id$
 Contents: private functions for the checkpointing functions.

 Copyright (c) Andreas Kowarz, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef ADOLC_CHECKPOINTING_P_H
#define ADOLC_CHECKPOINTING_P_H

#include <adolc/adolcexport.h>
#include <adolc/buffer_temp.h>
#include <adolc/checkpointing.h>

#define CP_BLOCK_SIZE 10

void ADOLC_API init_CpInfos(CpInfos *cpInfos);

#endif // ADOLC_CHECKPOINTING_P_H
