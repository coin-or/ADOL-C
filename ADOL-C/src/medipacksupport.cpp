/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     medipacksupport.cpp
 Revision: $Id$

 Copyright (c) Max Sagebaum

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include "taping_p.h"
#include "oplate.h"
#include "adolc/adouble.h"

#ifdef ADOLC_MEDIPACK_SUPPORT

#include <vector>
#include "medipacksupport_p.h"

struct AdolcMediStatic {
    typedef std::vector<medi::HandleBase*> HandleVector;
    std::vector<HandleVector*> tapeHandles;

    ~AdolcMediStatic() {
      for(size_t i = 0; i < tapeHandles.size(); ++i) {
        if(nullptr != tapeHandles[i]) {
          clearHandles(*tapeHandles[i]);

          delete tapeHandles[i];
          tapeHandles[i] = nullptr;
        }
      }
    }

    HandleVector& getTapeVector(short tapeId) {
      return *tapeHandles[tapeId];
    }

    void callHandle(short tapeId, locint index) {
      HandleVector& handleVec = getTapeVector(tapeId);

      medi::HandleBase* handle = handleVec[index];

      handle->func(handle);
    }

    void initTape(short tapeId) {
      if((size_t)tapeId >= tapeHandles.size()) {
        tapeHandles.resize(tapeId + 1, nullptr);
      }

      if(nullptr == tapeHandles[tapeId]) {
        tapeHandles[tapeId] = new HandleVector();
      } else {
        clearHandles(*tapeHandles[tapeId]);
      }
    }

    void freeTape(short tapeId) {
      if((size_t)tapeId < tapeHandles.size() && nullptr != tapeHandles[tapeId]) {
        clearHandles(*tapeHandles[tapeId]);
      }
    }

    void clearHandles(HandleVector& handles) {
      for(size_t i = 0; i < handles.size(); ++i) {
        medi::HandleBase* h = handles[i];

        delete h;
      }

      handles.resize(0);
    }

    locint addHandle(short tapeId, medi::HandleBase* handle) {
      HandleVector& vector = getTapeVector(tapeId);

      locint index = (locint)vector.size();
      vector.push_back(handle);
      return index;
    }
};

AdolcMediStatic* adolcMediStatic;

void mediAddHandle(medi::HandleBase* h) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  // do not need to check trace flag, this is included in the handle check
  put_op(medi_call);
  locint index = adolcMediStatic->addHandle(ADOLC_CURRENT_TAPE_INFOS.tapeID, h);

  ADOLC_PUT_LOCINT(index);
}

void mediCallHandle(short tapeId, locint index, double* primalVec, double* adjointVec) {
  AdolcTool::primalBase = primalVec;
  AdolcTool::adjointBase = adjointVec;

  adolcMediStatic->callHandle(tapeId, index);
}

void mediInitTape(short tapeId) {
  adolcMediStatic->initTape(tapeId);
}

void mediInitStatic() {
  adolcMediStatic = new AdolcMediStatic();
}

void mediFinalizeStatic() {
  delete adolcMediStatic;
}

MPI_Datatype AdolcTool::MpiType;
MPI_Datatype AdolcTool::ModifiedMpiType;
MPI_Datatype AdolcTool::AdjointMpiType;
AdolcTool::MediType* AdolcTool::MPI_TYPE;
medi::AMPI_Datatype AdolcTool::MPI_INT_TYPE;

medi::OperatorHelper<medi::FunctionHelper<adouble, double, double, int, double, AdolcTool>> AdolcTool::operatorHelper;

double* AdolcTool::adjointBase;
double* AdolcTool::primalBase;

#endif
