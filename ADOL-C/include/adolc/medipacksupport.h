/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ampisupportAdolc.cpp
 Revision: $Id$

 Copyright (c) Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_MEDISUPPORTADOLC_H)
#define ADOLC_MEDISUPPORTADOLC_H 1

#include "adouble.h"

#include <medi/medi.hpp>
#include <medi/adjointInterface.hpp>
#include <medi/adToolImplCommon.hpp>
#include <medi/ampi/types/indexTypeHelper.hpp>

void mediAddHandle(medi::HandleBase* h);
void mediInitStatic();
void mediFinalizeStatic();

struct AdolcTool final : public medi::ADToolImplCommon<AdolcTool, true, true, double, double, double, int> {
  typedef adouble Type;
  typedef void AdjointType;
  typedef double ModifiedType;
  typedef double PrimalType;
  typedef int IndexType;

  static MPI_Datatype MpiType;
  static MPI_Datatype ModifiedMpiType;
  static MPI_Datatype PrimalMpiType;
  static MPI_Datatype AdjointMpiType;

  typedef medi::MpiTypeDefault<AdolcTool> MediType;
  static MediType* MPI_TYPE;
  static medi::AMPI_Datatype MPI_INT_TYPE;

  static medi::OperatorHelper<medi::FunctionHelper<adouble, double, double, int, double, AdolcTool>> operatorHelper;

  static void initTypes() {
    // create the mpi type for ADOL-c
    MPI_Type_contiguous(sizeof(adouble), MPI_BYTE, &MpiType);
    MPI_Type_commit(&MpiType);

    ModifiedMpiType = MPI_DOUBLE;
    PrimalMpiType = MPI_DOUBLE;
    AdjointMpiType = MPI_DOUBLE;
  }

  static void init() {
    initTypes();

    MPI_TYPE = new MediType();

    operatorHelper.init(MPI_TYPE);
    MPI_INT_TYPE = operatorHelper.MPI_INT_TYPE;


    mediInitStatic();
  }

  static void finalizeTypes() {
  }

  static void finalize() {

    mediFinalizeStatic();

    operatorHelper.finalize();

    if(nullptr != MPI_TYPE) {
      delete MPI_TYPE;
      MPI_TYPE = nullptr;
    }

    finalizeTypes();
  }

  AdolcTool(MPI_Datatype primalMpiType, MPI_Datatype adjointMpiType) :
    medi::ADToolImplCommon<AdolcTool, true, true, double, double, double, int>(primalMpiType, adjointMpiType) {}


  inline bool isActiveType() const {
    return true;
  }

  inline  bool isHandleRequired() const {
    return isTaping();
  }

  inline bool isOldPrimalsRequired() const {
    return true;
  }

  inline void startAssembly(medi::HandleBase* h) const {
    MEDI_UNUSED(h);

  }

  inline void addToolAction(medi::HandleBase* h) const {
    if(NULL != h) {
      mediAddHandle(h);
    }
  }

  inline void stopAssembly(medi::HandleBase* h) const {
    MEDI_UNUSED(h);
  }

  medi::AMPI_Op convertOperator(medi::AMPI_Op op) const {
    return operatorHelper.convertOperator(op);
  }


  inline void createPrimalTypeBuffer(PrimalType* &buf, size_t size) const {
    buf = new PrimalType[size];
  }

  inline void createIndexTypeBuffer(IndexType* &buf, size_t size) const {
    buf = new IndexType[size];
  }

  inline void deletePrimalTypeBuffer(PrimalType* &buf) const {
    if(NULL != buf) {
      delete [] buf;
      buf = NULL;
    }
  }

  inline void deleteIndexTypeBuffer(IndexType* &buf) const {
    if(NULL != buf) {
      delete [] buf;
      buf = NULL;
    }
  }

  static inline int getIndex(const Type& value) {
    return value.loc();
  }

  static inline void clearIndex(Type& value) {
    // do nothing
  }

  static inline PrimalType getValue(const Type& value) {
    return value.value();
  }

  static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) {
    modValue = value.value();
  }

  static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) {
    value.setValue(modValue);
  }

  static void createIndex(Type& value, IndexType& index) {
    MEDI_UNUSED(value);
    MEDI_UNUSED(index);

    // do nothing indices are created in registerValue
  }

  static inline void registerValue(Type& value, PrimalType& oldPrimal, IndexType& index) {
    MEDI_UNUSED(oldPrimal);
    // do nothing value should have an index

    index = value.loc();
  }

  static PrimalType getPrimalFromMod(const ModifiedType& modValue) {
    return modValue;
  }

  static void setPrimalToMod(ModifiedType& modValue, const PrimalType& value) {
    modValue = value;
  }

  static void modifyDependency(ModifiedType& inval, ModifiedType& inoutval) {
    MEDI_UNUSED(inval);
    MEDI_UNUSED(inoutval);

    // no dependency tracking possible
  }
};

#endif
