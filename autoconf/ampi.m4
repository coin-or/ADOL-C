AC_DEFUN([AMPI_CONF],
[
AC_PREREQ(2.59)

AC_ARG_WITH(ampi,
[AS_HELP_STRING([--with-ampi=AMPI_DIR],
		[full path to the installation of adjoinable MPI (AMPI)])])
	
if test x"$with_ampi" != "x"; 
then 
  CPPFLAGS="$CPPFLAGS -I$with_ampi/include"
  LDFLAGS="$LDFLAGS -L$with_ampi/lib"
fi

LIBS="-lampiCommon $LIBS"

keepLIBS="$LIBS"

LIBS="$LIBS -lampiADtoolStubsOO" 

AC_MSG_CHECKING([libampiCommon (provided by AMPI)])
AC_LINK_IFELSE([AC_LANG_PROGRAM([#include "ampi/ampi.h"],
                              [AMPI_Finalize_NT() ])],
               [AC_MSG_RESULT([ok])],
               [AC_MSG_RESULT([no])
               AC_MSG_FAILURE([libampiCommon is required by $PACKAGE])])

LIBS="$keepLIBS"

])
