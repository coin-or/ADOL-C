##############################################################################
# configure.ac -- Process this file with autoconf to produce configure
# Revision: $Id: configure.ac 289 2009-02-24 09:44:15Z awalther $
#
# Copyright (C) Andrea Walther, Andreas Kowarz
#
##############################################################################
define([ADOLC_VER], [2])
define([ADOLC_SUB], [0])
define([ADOLC_LVL], [0])

AC_PREREQ(2.57)
AC_INIT(adolc,
        ADOLC_VER.ADOLC_SUB.ADOLC_LVL-trunk,
        [adol-c@tu-dresden.de])
AC_REVISION($Revision: 289 $)
AC_CONFIG_SRCDIR([adolc/adolc.h])
AC_CONFIG_AUX_DIR(autoconf)
AC_CONFIG_HEADERS([adolc/config.h])
AC_PREFIX_DEFAULT(${HOME}/adolc_base)
AM_INIT_AUTOMAKE

# checks for programs
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL

# Checks for libraries
AC_CHECK_LIB([m], [pow])

# checks for header files
AC_HEADER_STDC
AC_HEADER_TIME
AC_HEADER_STDBOOL
AC_CHECK_HEADERS([stddef.h stdlib.h stdio.h string.h sys/timeb.h])

# checks for types
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_STRUCT_TM

# Checks for library functions
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([floor ftime pow sqrt strchr strtol])

# substitutions
AC_SUBST(ac_aux_dir)

# ADOL-C configuration
AC_DEFINE(ADOLC_VERSION,ADOLC_VER,[ADOL-C Version])
AC_DEFINE(ADOLC_SUBVERSION,ADOLC_SUB,[ADOL-C Subversion])
AC_DEFINE(ADOLC_PATCHLEVEL,ADOLC_LVL,[ADOL-C Patchlevel])

AC_MSG_CHECKING(whether to use ADOL-C debug mode)
AC_ARG_ENABLE(debug,
              [
AS_HELP_STRING([--enable-debug],[enable ADOL-C debug mode [default=no]])],
  [AC_DEFINE(ADOLC_DEBUG,1,[ADOL-C debug mode])
   AC_MSG_RESULT(yes)
   adolc_debug="yes"],
  [AC_MSG_RESULT(no)
   adolc_debug="no"])

AC_MSG_CHECKING(whether to use ADOL-C hard debug mode)
AC_ARG_ENABLE(harddebug,
              [
AS_HELP_STRING([--enable-harddebug],[enable ADOL-C hard debug mode
 [default=no]])],
  [AC_DEFINE(ADOLC_HARDDEBUG,1,[ADOL-C hard debug mode])
   AC_DEFINE(ADOLC_DEBUG,1,[ADOL-C debug mode])
   AC_MSG_RESULT(yes)
   adolc_harddebug="yes"],
  [AC_MSG_RESULT(no)
   adolc_harddebug="no"])

AC_MSG_CHECKING(whether standard adouble constructor shall zero value)
AC_ARG_ENABLE(stdczero,
              [
AS_HELP_STRING([--enable-stdczero],[adouble standard constructor zeros value 
(at the cost of performance) [default=no]])],
  [AC_DEFINE(ADOLC_ADOUBLE_STDCZERO,1,[ADOL-C adouble zeroing mode])
   AC_MSG_RESULT(yes)
   adolc_stdczero="yes"],
  [AC_MSG_RESULT(no)
   adolc_stdczero="no"])

AC_MSG_CHECKING(whether errno is thread save)
AC_ARG_ENABLE(tserrno,
              [
AS_HELP_STRING([--enable-tserrno],[use errno as thread number cache [default=no]])],
  [AC_DEFINE(ADOLC_THREADSAVE_ERRNO,1,[ADOL-C thread save errno mode])
   AC_MSG_RESULT(yes)
   adolc_tserrno="yes"],
  [AC_MSG_RESULT(no)
   adolc_tserrno="no"])

AC_MSG_CHECKING(whether external function call forces backup of all adoubles)
AC_ARG_ENABLE(safe_extern,
              [
AS_HELP_STRING([--enable-safe-extern],[backup all adoubles before calling an external function [default=no]])],
  [AC_DEFINE(ADOLC_SAFE_EXTERN,1,[ADOL-C safe extern mode])
   AC_MSG_RESULT(yes)
   adolc_safe_extern="yes"],
  [AC_MSG_RESULT(no)
   adolc_safe_extern="no"])

# ADOL-C Sparse facility
AC_MSG_CHECKING(whether to build sparse drivers)
AC_ARG_WITH(sparse,
            [
AS_HELP_STRING([--with-sparse],[build sparse drivers [default=no]])],
            [sparse="yes"
             AC_MSG_RESULT(yes)],
            [sparse="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(SPARSE,[test x${sparse} = xyes])

# ADOL-C Examples
AC_MSG_CHECKING(whether to build documented examples)
AC_ARG_WITH(docexa,
            [
AS_HELP_STRING([--with-docexa],[build documented examples [default=no]])],
            [docexa="yes"
             AC_MSG_RESULT(yes)],
            [docexa="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(DOCEXA,[test x${docexa} = xyes])

AC_MSG_CHECKING(whether to build additional examples)
AC_ARG_WITH(addexa,
            [AS_HELP_STRING([--with-addexa],
                            [build additional examples [default=no]])],
            [addexa="yes"
             AC_MSG_RESULT(yes)],
            [addexa="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(ADDEXA,[test x${addexa} = xyes])

AC_MSG_CHECKING(whether to build parallel example)
AC_ARG_WITH(parexa,
            [AS_HELP_STRING([--with-parexa],
                            [build parallel example [default=no], if yes -with-openmp-flag=FLAG required])],
            [parexa="yes"
             AC_MSG_RESULT(yes)],
            [parexa="no"
             AC_MSG_RESULT(no)])
AM_CONDITIONAL(PAREXA,[test x${parexa} = xyes])

# ADOL-C Compilation flags
AC_MSG_CHECKING(which flag to use to enable OpenMP)
AC_ARG_WITH(openmp-flag,
            [AS_HELP_STRING([--with-openmp-flag=FLAG],
                            [use FLAG to enable OpenMP at compile time [default=none]])],
            [ac_adolc_openmpflag=$withval
             msg_result=$withval
             AC_MSG_RESULT($ac_adolc_openmpflag)],
            [ac_adolc_openmpflag=""
             msg_result="none"
             AC_MSG_RESULT($ac_adolc_openmpflag)])

AC_MSG_CHECKING(which CFLAGS to use)
AC_ARG_WITH(cflags,
            [AS_HELP_STRING([--with-cflags=FLAGS],
                            [use CFLAGS=FLAGS (default: -O3 -Wall -ansi)])],
            [ac_adolc_cflags="$withval"
             if test x${adolc_debug} == xyes ||
                test x${adolc_harddebug} == xyes ; then
                ac_adolc_cflags="$ac_adolc_cflags -g -O0"
             fi
             ac_adolc_cflags="$ac_adolc_cflags $ac_adolc_openmpflag"
             AC_SUBST(ac_adolc_cflags)
             AC_MSG_RESULT($ac_adolc_cflags)],
            [if test x"$CFLAGS" = x; then
                ac_adolc_cflags="-O3 -Wall -ansi"
                if test x${adolc_debug} == xyes ||
                   test x${adolc_harddebug} == xyes ; then
                   ac_adolc_cflags="-g -O0 -Wall -ansi"
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
                            [use CXXFLAGS=FLAGS (default: -O3 -Wall)])],
            [ac_adolc_cxxflags="$withval"
             if test x${adolc_debug} == xyes ||
                test x${adolc_harddebug} == xyes ; then
                ac_adolc_cxxflags="$ac_adolc_cxxflags -g -O0"
             fi
             ac_adolc_cxxflags="$ac_adolc_cxxflags $ac_adolc_openmpflag"
             AC_SUBST(ac_adolc_cxxflags)
             AC_MSG_RESULT($ac_adolc_cxxflags)],
            [if test x"$CXXFLAGS" = x; then
                ac_adolc_cxxflags="-O3 -Wall"
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

# output
AC_CONFIG_FILES(Makefile
                adolc/Makefile
                adolc/drivers/Makefile 
                adolc/sparse/Makefile
                adolc/tapedoc/Makefile
                doc/Makefile
                examples/Makefile
                examples/additional_examples/Makefile
                examples/additional_examples/clock/Makefile
                examples/additional_examples/checkpointing/Makefile
                examples/additional_examples/ext_diff_func/Makefile
                examples/additional_examples/fixpoint_exam/Makefile
                examples/additional_examples/hessmat/Makefile
                examples/additional_examples/lufact/Makefile
                examples/additional_examples/openmp_exam/Makefile
                examples/additional_examples/scal/Makefile
                examples/additional_examples/speelpenning/Makefile
                examples/additional_examples/taylor/Makefile
                examples/additional_examples/detexam/Makefile
                examples/additional_examples/helm/Makefile
                examples/additional_examples/lighthouse/Makefile
                examples/additional_examples/ode/Makefile
                examples/additional_examples/sparse/Makefile
                examples/additional_examples/tapesave/Makefile
                examples/additional_examples/pow/Makefile
                examples/additional_examples/timing/Makefile)

AC_OUTPUT

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

  Build documented examples:        ${docexa}
  Build additional examples:        ${addexa}

  See adolc/config.h for further configuration information.
-----------------------------------------------------------------------------"

# check for make install problems resulting from missing write permission
if test x${prefix} == x${HOME}/adolc_base ; then
echo \
"
*****************************************************************************

  To successfully compile and run programs using the ADOL-C shared library do
  the following things:
     compiling:
        * add \"-I\${HOME}/adolc_base/include\" to your compiler call
     linking:
        * add \"-L\${HOME}/adolc_base/lib -ladolc\" to your linker call
        * extend your linker call by \"-Wl,--rpath -Wl,\${HOME}/adolc_base/lib\"
          (if you wish to skip the point \"executing\")
     executing (do one of the following things):
        * add \${HOME}/adolc_base/lib to your LD_LIBRARY_PATH variable
        * ask your system administrator for adding \${HOME}/adolc_base/lib to
          the global file containing library search paths (/etc/ld.so.conf)

     (or use the static library by replacing
        \"-L\${HOME}/adolc_base/lib -ladolc\" with
        \"\${HOME}/adolc_base/lib/libadolc.a\" when linking)

  See README for instructions on how to change to a nonlocal installation!

*****************************************************************************
"
fi
