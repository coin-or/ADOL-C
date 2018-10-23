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

struct AdolcMeDiAdjointInterface : public medi::AdjointInterface {

    double** adjointBase;
    double* primalBase;
    int vecSize;

    AdolcMeDiAdjointInterface(double** adjointBase, double* primalBase, int vecSize) :
      adjointBase(adjointBase),
      primalBase(primalBase),
      vecSize(vecSize) {}

    int computeElements(int elements) const {
      return elements * vecSize;
    }

    int getVectorSize() const {
      return vecSize;
    }

    inline void createAdjointTypeBuffer(void* &buf, size_t size) const {
      buf = (void*)new double[size*vecSize];
    }

    inline void deleteAdjointTypeBuffer(void* &b) const {
      if(NULL != b) {
        double* buf = (double*)b;
        delete [] buf;
        b = NULL;
      }
    }

    inline void createPrimalTypeBuffer(void* &buf, size_t size) const {
      buf = (void*)new double[size];
    }

    inline void deletePrimalTypeBuffer(void* &b) const {
      if(NULL != b) {
        double* buf = (double*)b;
        delete [] buf;
        b = NULL;
      }
    }

    inline void getAdjoints(const void* i, void* a, int elements) const {
      double* adjoints = (double*)a;
      int* indices = (int*)i;

      for(int pos = 0; pos < elements; ++pos) {
        for(int dim = 0; dim < vecSize; ++dim) {
          adjoints[calcIndex(pos, dim)] = adjointBase[dim][indices[pos]];
          adjointBase[dim][indices[pos]] = 0.0;
        }
      }
    }

    inline void updateAdjoints(const void* i, const void* a, int elements) const {
      double* adjoints = (double*)a;
      int* indices = (int*)i;

      for(int pos = 0; pos < elements; ++pos) {
        for(int dim = 0; dim < vecSize; ++dim) {
          adjointBase[dim][indices[pos]] += adjoints[calcIndex(pos, dim)];
        }
      }
    }

    inline void setPrimals(const void* i, const void* p, int elements) const {
      double* primals = (double*)p;
      int* indices = (int*)i;

      for(int pos = 0; pos < elements; ++pos) {
        primalBase[indices[pos]] = primals[pos];
      }
    }

    inline void getPrimals(const void* i, const void* p, int elements) const {
      double* primals = (double*)p;
      int* indices = (int*)i;

      for(int pos = 0; pos < elements; ++pos) {
        primals[pos] = primalBase[indices[pos]];
      }
    }

    inline void combineAdjoints(void* b, const int elements, const int ranks) const {
      double* buf = (double*)b;
      for(int curRank = 1; curRank < ranks; ++curRank) {
        for(int curPos = 0; curPos < elements; ++curPos) {
          for(int dim = 0; dim < vecSize; ++dim) {
            buf[calcIndex(curPos, dim)] += buf[calcIndex(elements * curRank + curPos, dim)];
          }
        }
      }
    }

  private:

    inline int calcIndex(int pos, int dim) const {
      return pos * vecSize + dim;
    }
};

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

    void callHandleReverse(short tapeId, locint index, AdolcMeDiAdjointInterface& interface) {
      HandleVector& handleVec = getTapeVector(tapeId);

      medi::HandleBase* handle = handleVec[index];

      handle->funcReverse(handle, &interface);
    }

    void callHandleForward(short tapeId, locint index, AdolcMeDiAdjointInterface& interface) {
      HandleVector& handleVec = getTapeVector(tapeId);

      medi::HandleBase* handle = handleVec[index];

      handle->funcForward(handle, &interface);
    }

    void callHandlePrimal(short tapeId, locint index, AdolcMeDiAdjointInterface& interface) {
      HandleVector& handleVec = getTapeVector(tapeId);

      medi::HandleBase* handle = handleVec[index];

      handle->funcPrimal(handle, &interface);
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

void mediCallHandleReverse(short tapeId, locint index, double* primalVec, double** adjointVec, int vecSize) {
  AdolcMeDiAdjointInterface interface(adjointVec, primalVec, vecSize);

  adolcMediStatic->callHandleReverse(tapeId, index, interface);
}

void mediCallHandleForward(short tapeId, locint index, double* primalVec, double** adjointVec, int vecSize) {
  AdolcMeDiAdjointInterface interface(adjointVec, primalVec, vecSize);

  adolcMediStatic->callHandleForward(tapeId, index, interface);
}

void mediCallHandlePrimal(short tapeId, locint index, double* primalVec) {
  AdolcMeDiAdjointInterface interface(nullptr, primalVec, 1);

  adolcMediStatic->callHandlePrimal(tapeId, index, interface);
}

void mediInitTape(short tapeId) {
  if(NULL == adolcMediStatic) {
    mediInitStatic();
  }
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
MPI_Datatype AdolcTool::PrimalMpiType;
MPI_Datatype AdolcTool::AdjointMpiType;
AdolcTool::MediType* AdolcTool::MPI_TYPE;
medi::AMPI_Datatype AdolcTool::MPI_INT_TYPE;

medi::OperatorHelper<medi::FunctionHelper<adouble, double, double, int, double, AdolcTool>> AdolcTool::operatorHelper;

#endif
