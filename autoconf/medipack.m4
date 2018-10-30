AC_DEFUN([MEDIPACK_CONF],
[
AC_PREREQ(2.59)

AC_ARG_WITH(medipack,
[AS_HELP_STRING([--with-medipack=MEDIPACK_DIR],
		[full path to the installation of MeDiPack])])

if test x"$with_medipack" != "x";
then
  if test x"$adolc_medipack_support" = "xno";
  then
    AC_MSG_ERROR([if --with-medipack is set one  must also --enable-medipack])
  fi
  CPPFLAGS="$CPPFLAGS -I$with_medipack/include"
fi
])
