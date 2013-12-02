# ----------------------------------------------------------------------------
# ADOL-C -- Automatic Differentiation by Overloading in C++
# File:     Makefile
# Revision: $$
# Contents: Makefile for class myADOLC_NPL_sparse to interface ADOL-C with Ipopt
# 
# Copyright (c) Andrea Walther
#   
# This file is part of ADOL-C. This software is provided as open source.
# Any use, reproduction, or distribution of the software constitutes 
# recipient's acceptance of the terms of the accompanying license file.
# 
# This code is based on the corresponding makefile contained in the
# Ipopt package with the author Andreas Waechter   
#---------------------------------------------------------------------------

##########################################################################
#    You can modify this example makefile to fit for your own program.   #
#    Usually, you only need to change the five CHANGE entries below.     #
##########################################################################

# CHANGEME: This should be the name of your executable
EXE = cpp_example

# CHANGEME: Here is the name of all object files corresponding to the source
#           code that you wrote in order to define the problem statement
OBJS =  cpp_example.o \
	ADOL-C_sparseNLP.o

# Definitions for ADOL-C
# CHANGE THIS TO YOUR ADOL-C BASE DIRECTORY
ADPATH = $(HOME)/AD/ADOL-C/adolc_base_coin/include/adolc
ADLIBDIR = $(HOME)/AD/ADOL-C/adolc_base_coin/lib

# CHANGEME: Additional libraries
ADDLIBS = -Wl,--rpath -Wl,$(ADLIBDIR) -L$(ADLIBDIR) -l${adolclib} $(HOME)/AD/ADOL-C/adolc_base_coin/lib/libColPack.a

# CHANGEME: Additional flags for compilation (e.g., include flags)
ADDINCFLAGS = -I$(ADPATH)  

##########################################################################
#  Usually, you don't have to change anything below.  Note that if you   #
#  change certain compiler options, you might have to recompile Ipopt.   #
##########################################################################


HERE YOU SHOULD USE THE SETTING OF YOUR IPOPT INSTALLATION 

THIS SHOULD WORK WITH COPY AND PASTE
