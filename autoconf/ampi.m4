AC_DEFUN([AMPI_CONF],
[
AC_PREREQ(2.59)

AC_ARG_WITH(ampi,
[AS_HELP_STRING([--with-ampi=AMPI_DIR],
		[full path to the installation of adjoinable MPI (AMPI)])])
	
if test x"$with_ampi" != "x"; 
then 
  if test x"$adolc_ampi_support" = "xno";
  then 
    AC_MSG_ERROR([if --with-ampi is set one  must also --enable_ampi])
  fi
  AMPICPPFLAGS="$CPPFLAGS -I$with_ampi/include"
  AMPILDFLAGS="$LDFLAGS -L$with_ampi/${_lib}"
fi

if test x"$adolc_ampi_support" = "xyes"; 
then 
  AMPILIBS="-lampiCommon -lampiBookkeeping -lampiTape"

  keepCPPFLAGS="$CPPFLAGS"
  keepLIBS="$LIBS"
  keepLDFLAGS="$LDFLAGS"
  keepCC="$CC"
  keepCPP="$CPP"
  keepCXX="$CXX"
  keepLD="$LD"

  CC=$MPICC
  CXX=$MPICXX
  CPP=$MPICXX
  LD=$MPICXX

  CPPFLAGS="$AMPICPPFLAGS $CPPFLAGS"
  LDFLAGS="$AMPILDFLAGS $LDFLAGS"
  LIBS="$AMPILIBS -lampiADtoolStubsOO $LIBS" 

  AC_MSG_CHECKING([libampiCommon (provided by AMPI)])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([#include "ampi/ampi.h"],
                              [AMPI_Finalize_NT() ])],
               [AC_MSG_RESULT([ok])],
               [AC_MSG_RESULT([no])
               AC_MSG_FAILURE([libampiCommon is required by $PACKAGE])])

  LIBS="$keepLIBS"
  LDFLAGS="$keepLDFLAGS"
  CPPFLAGS="$keepCPPFLAGS"

  CC="$keepCC"
  CPP="$keepCPP"
  CXX="$keepCXX"
  LD="$keepLD"
fi
AC_SUBST(AMPICPPFLAGS)
AC_SUBST(AMPILDFLAGS)
AC_SUBST(AMPILIBS)
])
