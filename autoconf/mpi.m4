AC_DEFUN([MPI_CONF_PRE],
[
AC_PREREQ(2.59)

# MPI root directory
AC_ARG_WITH(mpi_root,
[AC_HELP_STRING([--with-mpi-root=MPIROOT],
		[absolute path to the MPI root directory])])

if test x"$with_mpi_root" != "x"; 
then 
  if test x"$adolc_ampi_support" = "xno";
  then 
    AC_MSG_ERROR([if --with-mpi-root is set one  must also --enable_ampi])
  fi
  MPIROOT="$with_mpi_root"
fi

AC_ARG_WITH(mpicc,
[AC_HELP_STRING([--with-mpicc=MPICC],
		[name of the MPI C++ compiler to use (default mpicc)])])

if test x"$with_mpicc" != "x"; 
then 
  if test x"$adolc_ampi_support" = "xno" && test x"$adolc_medipack_support" = "xno";
  then 
    AC_MSG_ERROR([if --with-mpicc is set one  must also --enable-ampi])
  fi
  MPICC="$with_mpicc"
else 
  MPICC="mpicc"
fi

if test x"$with_mpi_root" != "x"; 
then 
  MPICC="$with_mpi_root/bin/$MPICC"
fi


AC_ARG_WITH(mpicxx,
[AC_HELP_STRING([--with-mpicxx=MPICXX],
		[name of the MPI C++ compiler to use (default mpicxx)])])

if test x"$with_mpicxx" != "x"; 
then 
  if test x"$adolc_ampi_support" = "xno" && test x"$adolc_medipack_support" = "xno";
  then 
    AC_MSG_ERROR([if --with-mpicxx is set one  must also --enable-ampi])
  fi
  MPICXX="$with_mpicxx"
else 
  MPICXX="mpicxx"
fi

if test x"$with_mpi_root" != "x"; 
then 
  MPICXX="$with_mpi_root/bin/$MPICXX"
fi

saveCC="$CC"
saveCXX="$CXX"
AC_SUBST(saveCC)
AC_SUBST(saveCXX)

if test x"$adolc_ampi_support" = "xyes" || test x"$adolc_medipack_support" = "xyes";
then
  CC="$MPICC"
  CXX="$MPICXX"
fi

])

AC_DEFUN([MPI_CONF_POST],
[
AC_PREREQ(2.59)

if test x"$adolc_ampi_support" = "xyes" || test x"$adolc_medipack_support" = "xyes";
then

  AC_MSG_CHECKING([Linking of MPI C programs])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <mpi.h>],
               [MPI_Init(0,0)])],
               [AC_MSG_RESULT([ok])],
               [AC_MSG_RESULT([no])
               AC_MSG_FAILURE([MPI C compiler is required by $PACKAGE])])

fi

if test x"$adolc_ampi_support" = "xyes" || test x"$adolc_medipack_support" = "xyes";
then 

  AC_LANG_PUSH([C++])
  AC_MSG_CHECKING([Linking of MPI C++ programs])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <mpi.h>],
               [MPI_Init(0,0)])],
               [AC_MSG_RESULT([ok])],
               [AC_MSG_RESULT([no])
               AC_MSG_FAILURE([MPI C++ compiler is required by $PACKAGE])])
  AC_LANG_POP([C++])
fi

])
