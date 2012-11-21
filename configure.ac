##############################################################################
# configure.ac -- Process this file with autoconf to produce configure
# Revision: $Id$
#
# Copyright (C) Andrea Walther, Andreas Kowarz
#               
# contains patches from the COIN OR libtool
#
##############################################################################
define([ADOLC_VER], [2])
define([ADOLC_SUB], [3])
define([ADOLC_LVL], [0])

AC_PREREQ(2.67)
AC_INIT(adolc,
        ADOLC_VER.ADOLC_SUB.ADOLC_LVL-trunk,
        [adol-c@list.coin-or.org])
config_flags="$*"
AC_REVISION($Revision$)
AC_CONFIG_SRCDIR([ADOL-C/src/adouble.cpp])
AC_CONFIG_AUX_DIR(autoconf)
AC_CONFIG_HEADERS([ADOL-C/src/config.h])
AC_PREFIX_DEFAULT(${HOME}/adolc_base)
AM_INIT_AUTOMAKE([1.11 dist-bzip2 foreign])
AM_SILENT_RULES([yes])

if test "x$CFLAGS" == x ; then
    ac_shell_cflags="not_set"
else
    ac_shell_cflags="$CFLAGS"
fi
if test "x$CXXFLAGS" == x ; then
    ac_shell_cxxflags="not_set"
else
    ac_shell_cxxflags="$CXXFLAGS"
fi
libdir_set_on_cmd="no"
for i in $config_flags ; do
    case $i in
    -libdir | --libdir | --libdi | --libd | -libdir=* | --libdir=* | --libdi=* | --libd=*)
            libdir_set_on_cmd="yes"
            break
            ;;
    esac
done

# checks for programs
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL

AC_CHECK_SIZEOF([void *])
if test "x$ac_cv_sizeof_void_p" = "x8" ; then
   _lib=lib64
else
   _lib=lib
fi
if test "x$libdir_set_on_cmd" = "xno" ; then
   libdir='${exec_prefix}/${_lib}'
   AC_SUBST(_lib)
   AC_SUBST(libdir)
fi

# checks for header files
AC_HEADER_STDC
AC_HEADER_TIME
AC_HEADER_STDBOOL
AC_CHECK_HEADERS([stddef.h stdlib.h stdio.h string.h unistd.h sys/timeb.h])

# checks for types
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_STRUCT_TM

# Checks for library functions
AC_FUNC_MALLOC
AC_FUNC_REALLOC

# Checks for libraries and fuctions
AC_SEARCH_LIBS([pow], [m])
AC_CHECK_FUNCS([floor fmax fmin ftime pow sqrt strchr strtol trunc])

# substitutions
AC_SUBST(ac_aux_dir)

# ADOL-C configuration
AC_DEFINE(ADOLC_VERSION,ADOLC_VER,[ADOL-C Version])
AC_DEFINE(ADOLC_SUBVERSION,ADOLC_SUB,[ADOL-C Subversion])
AC_DEFINE(ADOLC_PATCHLEVEL,ADOLC_LVL,[ADOL-C Patchlevel])
ADOLC_VERSION=ADOLC_VER
ADOLC_SUBVERSION=ADOLC_SUB
ADOLC_PATCHLEVEL=ADOLC_LVL
AC_SUBST(ADOLC_VERSION)
AC_SUBST(ADOLC_SUBVERSION)
AC_SUBST(ADOLC_PATCHLEVEL)

AC_MSG_CHECKING(whether to use ADOL-C debug mode)
AC_ARG_ENABLE(debug,
              [
AS_HELP_STRING([--enable-debug],[enable ADOL-C debug mode [default=no]])],
  [
  adolc_debug=$enableval
   AC_MSG_RESULT($adolc_debug)
   ],
  [
   adolc_debug=no
   AC_MSG_RESULT(no)
   ])

AC_MSG_CHECKING(whether to use ADOL-C hard debug mode)
AC_ARG_ENABLE(harddebug,
              [
AS_HELP_STRING([--enable-harddebug],[enable ADOL-C hard debug mode
 [default=no]])],
  [
   adolc_harddebug=$enableval
   AC_MSG_RESULT($adolc_harddebug)
   ],
  [
   adolc_harddebug=no
   AC_MSG_RESULT(no)
   ])

if test x$adolc_harddebug == xyes ; then
adolc_debug=yes
AC_DEFINE(ADOLC_HARDDEBUG,1,[ADOL-C hard debug mode])
fi

if test x$adolc_debug == xyes ; then
AC_DEFINE(ADOLC_DEBUG,1,[ADOL-C debug mode])
fi

AC_MSG_CHECKING(whether standard adouble constructor shall zero value)
AC_ARG_ENABLE(stdczero,
              [
AS_HELP_STRING([--enable-stdczero],[adouble standard constructor zeros value 
(at the cost of performance) [default=no]])],
  [
  adolc_stdczero=$enableval
  AC_MSG_RESULT($adolc_stdczero)
  ],
  [
  adolc_stdczero=no
  AC_MSG_RESULT(no)
  ])

if test x$adolc_stdczero == xyes ; then
AC_DEFINE(ADOLC_ADOUBLE_STDCZERO,1,[ADOL-C adouble zeroing mode])
fi

AC_MSG_CHECKING(whether errno is thread save)
AC_ARG_ENABLE(tserrno,
              [
AS_HELP_STRING([--enable-tserrno],[use errno as thread number cache [default=no]])],
  [
  adolc_tserrno=$enableval
  AC_MSG_RESULT($adolc_tserrno)
   ],
  [
  adolc_tserrno=no
  AC_MSG_RESULT(no)
  ])

if test x$adolc_tserrno == xyes ; then
AC_DEFINE(ADOLC_THREADSAVE_ERRNO,1,[ADOL-C thread save errno mode])
fi

AC_MSG_CHECKING(which flag to use to enable OpenMP)
AC_ARG_WITH(openmp-flag,
            [AS_HELP_STRING([--with-openmp-flag=FLAG],
                            [use FLAG to enable OpenMP at compile time [default=none]])],
            [ac_adolc_openmpflag=$withval
             AC_MSG_RESULT($ac_adolc_openmpflag)],
            [ac_adolc_openmpflag=""
             AC_MSG_RESULT(none)])


# ADOL-C Sparse facility
AC_MSG_CHECKING(whether to build sparse drivers)
AC_ARG_ENABLE(sparse,
            [
AS_HELP_STRING([--enable-sparse],[build sparse drivers [default=disabled]])],
            [sparse=$enableval
             AC_MSG_RESULT($sparse)],
            [sparse="no"
             AC_MSG_RESULT(no)])

AC_ARG_WITH(colpack,
	[
AS_HELP_STRING([--with-colpack=DIR],[path to the colpack library and headers [default=system libraries]])],
[
colpack=$withval
COLPACK_CFLAGS="$ac_adolc_openmpflag -I$colpack/include"
COLPACK_LIBDIR="$colpack/${_lib}"
COLPACK_LIBS="$ac_adolc_openmpflag -L$COLPACK_LIBDIR -lColPack"
],
[
COLPACK_CFLAGS="$ac_adolc_openmpflag"
COLPACK_LIBDIR=""
COLPACK_LIBS="-lColPack"
])

AC_LANG_PUSH([C++])
save_CPPFLAGS="$CPPFLAGS"
save_LIBS="$LIBS"
CPPFLAGS="$CPPFLAGS $COLPACK_CFLAGS"
LIBS="$COLPACK_LIBS $LIBS"
AC_MSG_CHECKING(for ColPack in $COLPACK_LIBDIR)
AC_CHECK_HEADER([ColPack/ColPackHeaders.h],[have_colpackheaders=yes],[have_colpackheaders=no])
if test x$have_colpackheaders = xyes ; then
   AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <ColPack/ColPackHeaders.h>],
			[ColPack::GraphColoring g])],
                       [have_colpack=yes
                        AC_MSG_RESULT(linking with -lColPack succeeded)],
                       [have_colpack=no,
                        AC_MSG_RESULT(linking with -lColPack failed)])
else
    have_colpack=no
fi
CPPFLAGS="$save_CPPFLAGS"
LIBS="$save_LIBS"
AC_LANG_POP([C++])

AM_CONDITIONAL(HAVE_LIBCOLPACK,[test x$have_colpack = xyes])
if test x$have_colpack = xyes; then 
   sparse=yes
   echo "will build sparse drivers as linking with -lColPack succeeded"
fi
AM_CONDITIONAL(SPARSE,[test x${sparse} = xyes])

if test x$sparse = xyes && test x$have_colpack = xyes; then
	AC_DEFINE(HAVE_LIBCOLPACK,[1],[Define 1 if ColPack is available])
else
	COLPACK_CFLAGS=""
	COLPACK_LIBS=""
fi
AC_SUBST(COLPACK_CFLAGS)
AC_SUBST(COLPACK_LIBS)

# ADOL-C Examples
AC_MSG_CHECKING(whether to build documented examples)
AC_ARG_ENABLE(docexa,
            [
AS_HELP_STRING([--enable-docexa],[build documented examples [default=disabled]])],
            [docexa=$enableval
             AC_MSG_RESULT(yes)],
            [docexa="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(DOCEXA,[test x${docexa} = xyes])

AC_MSG_CHECKING(whether to build additional examples)
AC_ARG_ENABLE(addexa,
            [AS_HELP_STRING([--enable-addexa],
                            [build additional examples [default=disabled]])],
            [addexa=$enableval
             AC_MSG_RESULT(yes)],
            [addexa="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(ADDEXA,[test x${addexa} = xyes])

AC_MSG_CHECKING(whether to build parallel example)
AC_ARG_ENABLE(parexa,
            [AS_HELP_STRING([--enable-parexa],
                            [build parallel example [default=disabled], if enabled -with-openmp-flag=FLAG required])],
            [parexa=$enableval
             AC_MSG_RESULT(yes)],
            [parexa="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(PAREXA,[test x${parexa} = xyes])

# ADOL-C Compilation flags
AC_MSG_CHECKING(which CFLAGS to use)
AC_ARG_WITH(cflags,
            [AS_HELP_STRING([--with-cflags=FLAGS],
                            [use CFLAGS=FLAGS (default: -O2)])],
            [ac_adolc_cflags="$withval"
             if test x${adolc_debug} == xyes ||
                test x${adolc_harddebug} == xyes ; then
                ac_adolc_cflags="$ac_adolc_cflags -g -O0"
             fi
             ac_adolc_cflags="$ac_adolc_cflags $ac_adolc_openmpflag"
             AC_SUBST(ac_adolc_cflags)
             AC_MSG_RESULT($ac_adolc_cflags)],
            [if test "$ac_shell_cflags" = "not_set"; then
                ac_adolc_cflags="-O2"
                if test x${adolc_debug} == xyes ||
                   test x${adolc_harddebug} == xyes ; then
                   ac_adolc_cflags="-g -O3 -Wall -ansi"
                fi
             else
                ac_adolc_cflags="$CFLAGS"
             fi
             ac_adolc_cflags="$ac_adolc_cflags $ac_adolc_openmpflag"

             AC_SUBST(ac_adolc_cflags)
             AC_MSG_RESULT($ac_adolc_cflags)])

AC_MSG_CHECKING(which CXXFLAGS to use)
AC_ARG_WITH(cxxflags,
            [AS_HELP_STRING([--with-cxxflags=FLAGS],
                            [use CXXFLAGS=FLAGS (default: -O2)])],
            [ac_adolc_cxxflags="$withval"
             if test x${adolc_debug} == xyes ||
                test x${adolc_harddebug} == xyes ; then
                ac_adolc_cxxflags="$ac_adolc_cxxflags -g -O0"
             fi
             ac_adolc_cxxflags="$ac_adolc_cxxflags $ac_adolc_openmpflag"
             AC_SUBST(ac_adolc_cxxflags)
             AC_MSG_RESULT($ac_adolc_cxxflags)],
            [if test "$ac_shell_cxxflags" = "not_set"; then
                ac_adolc_cxxflags="-O2"
                if test x${adolc_debug} == xyes ||
                   test x${adolc_harddebug} == xyes ; then
                   ac_adolc_cxxflags="-g -O0 -Wall"
                fi
             else
                ac_adolc_cxxflags="$CXXFLAGS"
             fi
             ac_adolc_cxxflags="$ac_adolc_cxxflags $ac_adolc_openmpflag"
            AC_SUBST(ac_adolc_cxxflags)
             AC_MSG_RESULT($ac_adolc_cxxflags)])

###########################################################################
#                      COIN_PATCH_LIBTOOL_CYGWIN                          #
###########################################################################

# Patches to libtool for cygwin. Lots for cl, a few for GCC.
# For cl:
# - cygpath is not correctly quoted in fix_srcfile_path
# - paths generated for .lib files is not run through cygpath -w

#
# REQUIRED FOR COMPILATION WITH CYGWIN !!!
#
 

AC_DEFUN([AC_COIN_PATCH_LIBTOOL_CYGWIN],
[ case "$CXX" in
    cl* | */cl* | CL* | */CL* | icl* | */icl* | ICL* | */ICL*) 
      AC_MSG_NOTICE(Applying patches to libtool for cl compiler)
      sed -e 's|fix_srcfile_path=\"`cygpath -w \"\$srcfile\"`\"|fix_srcfile_path=\"\\\`'"$CYGPATH_W"' \\\"\\$srcfile\\\"\\\`\"|' \
          -e 's|fix_srcfile_path=\"\"|fix_srcfile_path=\"\\\`'"$CYGPATH_W"' \\\"\\$srcfile\\\"\\\`\"|' \
          -e 's%compile_deplibs=\"\$dir/\$old_library \$compile_deplibs\"%compile_deplibs="'\`"$CYGPATH_W"' \$dir/\$old_library | sed -e '"'"'sY\\\\\\\\Y/Yg'"'"\`' \$compile_deplibs\"'% \
          -e 's%compile_deplibs=\"\$dir/\$linklib \$compile_deplibs\"%compile_deplibs="'\`"$CYGPATH_W"' \$dir/\$linklib | sed -e '"'"'sY\\\\\\\\Y/Yg'"'"\`' \$compile_deplibs\"'% \
          -e 's%lib /OUT:%lib -OUT:%' \
          -e "s%cygpath -w%$CYGPATH_W%" \
          -e 's%$AR x \\$f_ex_an_ar_oldlib%bla=\\`lib -nologo -list \\$f_ex_an_ar_oldlib | xargs echo '"$mydos2unix"'\\`; echo \\$bla; for i in \\$bla; do lib -nologo -extract:\\$i \\$f_ex_an_ar_oldlib; done%' \
          -e 's/$AR t/lib -nologo -list/' \
          -e 's%f_ex_an_ar_oldlib="\($?*1*\)"%f_ex_an_ar_oldlib='\`"$CYGPATH_W"' \1`%' \ 
          -e 's%^archive_cmds=.*%archive_cmds="\\$CC -o \\$lib \\$libobjs \\$compiler_flags \\\\\\`echo \\\\\\"\\$deplibs\\\\\\" | \\$SED -e '"\'"'s/ -lc\\$//'"\'"'\\\\\\` -link -dll~linknames="%' \
          -e 's%old_archive_cmds="lib -OUT:\\$oldlib\\$oldobjs\\$old_deplibs"%old_archive_cmds="if test -r \\$oldlib; then bla=\\"\\$oldlib\\"; else bla=; fi; lib -OUT:\\$oldlib \\\\\\$bla\\$oldobjs\\$old_deplibs"%' \
      libtool > conftest.bla

      mv conftest.bla libtool
      chmod 755 libtool
      ;;
    *)
      AC_MSG_NOTICE(Applying patches to libtool for GNU compiler)
      sed -e 's|fix_srcfile_path=\"`cygpath -w \"\$srcfile\"`\"|fix_srcfile_path=\"\\\`'"$CYGPATH_W"' \\\"\\$srcfile\\\"\\\`\"|' \
          -e 's|"lib /OUT:\\$oldlib\\$oldobjs\\$old_deplibs"|"\\$AR \\$AR_FLAGS \\$oldlib\\$oldobjs\\$old_deplibs~\\$RANLIB \\$oldlib"|' \
          -e 's|libext="lib"|libext="a"|' \
      libtool > conftest.bla

      mv conftest.bla libtool
      chmod 755 libtool
      ;;
  esac ]) # COIN_PATCH_LIBTOOL_CYGWIN
 

# output
AC_CONFIG_FILES(Makefile
                ADOL-C/Makefile
                ADOL-C/include/Makefile
                ADOL-C/include/adolc/Makefile
                ADOL-C/include/adolc/drivers/Makefile 
                ADOL-C/include/adolc/sparse/Makefile
                ADOL-C/include/adolc/tapedoc/Makefile
                ADOL-C/src/Makefile
                ADOL-C/src/drivers/Makefile 
                ADOL-C/src/sparse/Makefile
                ADOL-C/src/tapedoc/Makefile
                ADOL-C/doc/version.tex
                ADOL-C/examples/Makefile
                ADOL-C/examples/additional_examples/Makefile
                ADOL-C/examples/additional_examples/clock/Makefile
                ADOL-C/examples/additional_examples/checkpointing/Makefile
                ADOL-C/examples/additional_examples/ext_diff_func/Makefile
                ADOL-C/examples/additional_examples/fixpoint_exam/Makefile
                ADOL-C/examples/additional_examples/hessmat/Makefile
                ADOL-C/examples/additional_examples/lufact/Makefile
                ADOL-C/examples/additional_examples/openmp_exam/Makefile
                ADOL-C/examples/additional_examples/scal/Makefile
                ADOL-C/examples/additional_examples/speelpenning/Makefile
                ADOL-C/examples/additional_examples/taylor/Makefile
                ADOL-C/examples/additional_examples/detexam/Makefile
                ADOL-C/examples/additional_examples/helm/Makefile
                ADOL-C/examples/additional_examples/lighthouse/Makefile
                ADOL-C/examples/additional_examples/ode/Makefile
                ADOL-C/examples/additional_examples/sparse/Makefile
                ADOL-C/examples/additional_examples/tapesave/Makefile
                ADOL-C/examples/additional_examples/pow/Makefile
                ADOL-C/examples/additional_examples/timing/Makefile
                ADOL-C/test/Makefile
                MSVisualStudio/v10/sparse/config.h
                MSVisualStudio/v10/nosparse/config.h
                MSVisualStudio/v10/x64/sparse/config.h
                MSVisualStudio/v10/x64/nosparse/config.h
		adolc.spec
)

AC_OUTPUT

#call libtool for cygwin 
AC_COIN_PATCH_LIBTOOL_CYGWIN 
warn=false
# echo configuration
echo \
"
-----------------------------------------------------------------------------
Configuration:

  C compiler:                       ${CC}
  C++ compiler:                     ${CXX}
  Linker:                           ${LD}
  Source code location:             `pwd`
  Install path:                     ${prefix}
  
  CFLAGS:                           ${ac_adolc_cflags}
  CXXFLAGS:                         ${ac_adolc_cxxflags}

  Use ADOL-C debug mode:            ${adolc_debug}
  Use ADOL-C hard debug mode:       ${adolc_harddebug}
  Use ADOL-C adouble zeroing mode:  ${adolc_stdczero}"

if [[ "x${ac_adolc_openmpflag}" != "x" ]] ; then
    echo \
"  Access thread number via errno:   ${adolc_tserrno}"
fi

echo \
"  Use ADOL-C safe extern mode:      ${adolc_safe_extern}

  Build sparse drivers:             ${sparse}
  Build with ColPack:		    ${have_colpack}"
if test x$sparse = xyes && test x$have_colpack = xno; then
    echo -e \
"  \033@<:@1;31mCompressed sparse structures will not be available
  Only sparsity patterns can be computed\033@<:@0m"
    warn=true
fi
echo

echo \
"  Build documented examples:        ${docexa}
  Build additional examples:        ${addexa}

  See ADOL-C/src/config.h for further configuration information.
-----------------------------------------------------------------------------"

# check for make install problems resulting from missing write permission
pdir=${prefix}
writable=true
while test "x${pdir}" != "x/" ; do
    if test -e ${pdir} && test ! -w ${pdir}; then
        writable=false
        break
    elif test ! -e ${pdir} ; then
        pdir=`dirname ${pdir}`
    else
        break
    fi
done
if test "x${pdir}" == "x/" ; then
    writable=false
fi
if ! ${writable} ; then
    echo -e \
"  \033@<:@1;31mPrefix ${prefix} not writable
  please run ./configure --prefix=DIR
  with a writable directory path otherwise administrator privilages
  will be required to run \`make install'\033@<:@0m"
    warn=true
fi

case ${prefix} in 
   ${HOME}|${HOME}/*)
if $warn ; then
    for (( i=5; i>0; i-- )); do
        echo -ne "\rPlease read the above messages carefully, continuing in ${i} seconds"
        sleep 1
    done
    echo -e "\rPlease read the above messages carefully                                 "
fi
echo \
"
*****************************************************************************

  To successfully compile and run programs using the ADOL-C shared library do
  the following things:
     compiling:
        * add \"-I${prefix}/include\" to your compiler call
     linking:
        * add \"-L${prefix}/${_lib} -ladolc\" to your linker call
        * extend your linker call by \"-Wl,--rpath -Wl,${prefix}/${_lib}\"
          (if you wish to skip the point \"executing\")
     executing (do one of the following things):
        * add ${prefix}/${_lib} to your LD_LIBRARY_PATH variable
        * ask your system administrator for adding ${prefix}/${_lib} to
          the global file containing library search paths (/etc/ld.so.conf)

     (or use the static library by replacing
        \"-L${prefix}/${_lib} -ladolc\" with
        \"${prefix}/${_lib}/libadolc.a\" when linking)

  See README for instructions on how to change to a nonlocal installation!

*****************************************************************************
"
;;
esac
