dnl check for linking with ColPack
AC_DEFUN([COLPACK_CONF],
[
AC_PREREQ(2.59)

AC_ARG_WITH(colpack,
	[
AS_HELP_STRING([--with-colpack=DIR],[path to the colpack library and headers [default=system libraries]])],
[
colpack=$withval
COLPACK_CFLAGS="-I$colpack/include"
if test x${_lib} != xlib ; then
   D[[0]]="$colpack/${_lib}"
   D[[1]]="$colpack/lib"
else
   D[[0]]="$colpack/${_lib}"
fi
],
[
COLPACK_CFLAGS=""
COLPACK_LIBS="-lColPack"
D[[0]]=""
])

AC_LANG_PUSH([C++])
save_CXXFLAGS="$CXXFLAGS"
save_CPPFLAGS="$CPPFLAGS"
save_LIBS="$LIBS"
CPPFLAGS="$CPPFLAGS $COLPACK_CFLAGS"
CXXFLAGS="$CXXFLAGS $ac_adolc_openmpflag"
AC_CHECK_HEADER([ColPack/ColPackHeaders.h],[have_colpackheaders=yes],[
have_colpackheaders=no
CPPFLAGS="$save_CPPFLAGS"
CXXFLAGS="$save_CXXFLAGS"
])
for ((i=0; i < ${#D[@]} ; i++)); do
COLPACK_LIBDIR="${D[[$i]]}"
if test -n "$COLPACK_LIBDIR" ; then
COLPACK_LIBS="-L$COLPACK_LIBDIR -lColPack -Wl,-rpath,$COLPACK_LIBDIR"
fi
if test x$have_colpackheaders = xyes ; then
   LIBS="$COLPACK_LIBS $LIBS"
   AC_MSG_CHECKING(for libColPack in $COLPACK_LIBDIR)
   AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <ColPack/ColPackHeaders.h>],
			[ColPack::GraphColoring g])],
                       [have_colpack=yes],
                       [have_colpack=no])
   AC_MSG_RESULT($have_colpack)
   if test x$have_colpack = xyes ; then
      break
   else
      LIBS="$save_LIBS"
   fi
else
    have_colpack=no
fi
done
AC_LANG_POP([C++])

AM_CONDITIONAL(HAVE_LIBCOLPACK,[test x$have_colpack = xyes])
if test x$have_colpack = xyes; then 
   sparse=yes
   echo "will build sparse drivers as linking with -lColPack succeeded"
fi

if test x$sparse = xyes && test x$have_colpack = xyes; then
	AC_DEFINE(HAVE_LIBCOLPACK,[1],[Define 1 if ColPack is available])
fi

])
