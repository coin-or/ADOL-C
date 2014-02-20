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
    AC_MSG_ERROR([if --with-ampi is set one  must also --enable-ampi])
  fi
  CPPFLAGS="$CPPFLAGS -I$with_ampi/include"
  LDFLAGS="$LDFLAGS -L$with_ampi/lib -Wl,-rpath,$with_ampi/lib"
  if test x"$_lib" != "xlib" ;
  then
     LDFLAGS="$LDFLAGS -L$with_ampi/${_lib} -Wl,-rpath,$with_ampi/${_lib}"
  fi
fi

if test x"$adolc_ampi_support" = "xyes"; 
then 
  LIBS="-lampiCommon -lampiBookkeeping -lampiTape $LIBS"

  keepLIBS="$LIBS"
  LIBS="$LIBS -lampiADtoolStubsOO" 

  AC_MSG_CHECKING([libampiCommon (provided by AMPI)])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([#include "ampi/ampi.h"],
                              [AMPI_Finalize_NT() ])],
               [AC_MSG_RESULT([ok])],
               [AC_MSG_RESULT([no])
               AC_MSG_FAILURE([libampiCommon is required by $PACKAGE])])

  LIBS="$keepLIBS"
  adolclib=adolc_ampi
fi
])
