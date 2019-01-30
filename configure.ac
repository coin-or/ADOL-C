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
define([ADOLC_SUB], [7])
define([ADOLC_LVL], [0])

AC_PREREQ(2.67)
AC_INIT(adolc,
        ADOLC_VER.ADOLC_SUB.ADOLC_LVL-trunk,
        [adol-c@list.coin-or.org])
config_flags="$*"
AC_REVISION($Revision$)
AC_CONFIG_SRCDIR([ADOL-C/src/adouble.cpp])
AC_CONFIG_AUX_DIR([autoconf])
AC_CONFIG_MACRO_DIR([autoconf])
AC_CONFIG_HEADERS([ADOL-C/src/config.h])
AC_PREFIX_DEFAULT(${HOME}/adolc_base)
AM_INIT_AUTOMAKE([1.11 dist-bzip2 foreign subdir-objects])
AM_SILENT_RULES([yes])


adolclib=adolc

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

AM_CONDITIONAL(DARWIN, [test "x${host_os#darwin}" != "x${host_os}"])

AC_MSG_CHECKING(whether to build ADOL-C with adjoinable MPI (AMPI) support)
AC_ARG_ENABLE(ampi,
              [AS_HELP_STRING([--enable-ampi],[build ADOL-C with adjoinable MPI (AMPI) support [default=disabled]])],
	      [adolc_ampi_support=$enableval],[adolc_ampi_support=no])
AC_MSG_RESULT($adolc_ampi_support)
AM_CONDITIONAL(BUILD_ADOLC_AMPI_SUPPORT,[test x${adolc_ampi_support} = xyes])
if test x"${adolc_ampi_support}" = xyes ; then
AC_DEFINE(ADOLC_AMPI_SUPPORT,1,[defined if adjoinable MPI support is to be compiled in])
fi

AC_MSG_CHECKING(whether to build ADOL-C with MeDiPack (MPI) support)
AC_ARG_ENABLE(medipack,
              [AS_HELP_STRING([--enable-medipack],[build ADOL-C with MeDiPack (MPI) support [default=disabled]])],
	      [adolc_medipack_support=$enableval],[adolc_medipack_support=no])
AC_MSG_RESULT($adolc_medipack_support)
AM_CONDITIONAL(BUILD_ADOLC_MEDIPACK_SUPPORT,[test x${adolc_medipack_support} = xyes])
if test x"${adolc_medipack_support}" = xyes ; then
AC_DEFINE(ADOLC_MEDIPACK_SUPPORT,1,[defined if MeDiPack support is to be compiled in])
fi

LT_PREREQ([2.2.6])
# checks for programs
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11(noext,mandatory)
AC_PROG_CC
AC_PROG_CC_C_O
AC_PROG_CC_C99
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
LT_INIT([disable-static win32-dll dlopen])

if test "x$enable_static" != "xno" ; then
   AC_MSG_WARN([Linking a user program with static ADOL-C library will always result in a segmentation fault])
fi

MPI_CONF_PRE

AC_CHECK_SIZEOF([void *])
if test "x$ac_cv_sizeof_void_p" = "x8" ; then
   _lib=lib64
else
   _lib=lib
fi
if test "x$libdir_set_on_cmd" = "xno" ; then
   libdir='${exec_prefix}'/"${_lib}"
   AC_SUBST(_lib)
   AC_SUBST(libdir)
fi

# check for builtin
ACX_C_BUILTIN_EXPECT

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


# Checks for libraries and fuctions
AC_SEARCH_LIBS([pow], [m])
AC_CHECK_FUNCS([floor fmax fmin ftime pow sqrt cbrt strchr strtol trunc])

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

# Checks for library functions
AC_FUNC_MALLOC
AC_FUNC_REALLOC
if test x$ac_cv_func_malloc_0_nonnull != xyes ; then
AC_DEFINE_UNQUOTED([calloc],[rpl_calloc],[Define to rpl_calloc if the replacement function should be used.])
fi

AC_MSG_CHECKING(whether to use calloc or malloc for memory allocation)
AC_ARG_ENABLE(use-calloc,[AS_HELP_STRING([--disable-use-calloc],
  [disable use of calloc and use malloc instead for memory allocation [default=calloc is used]])],
  [use_calloc=$enableval],[use_calloc=yes])

if test x$use_calloc = xyes ; then
AC_MSG_RESULT(calloc)
AC_DEFINE(ADOLC_USE_CALLOC,1,[Use calloc instead of malloc for memory allocation])
else
AC_MSG_RESULT(malloc)
fi

AC_MSG_CHECKING(whether to enable atrig/erf functions)
AC_ARG_ENABLE(atrig-erf,[AS_HELP_STRING([--enable-atrig-erf],
  [enable atrig and erf functions in ADOL-C [default=no]])],
  [use_atrig_erf=$enableval],[use_atrig_erf=no])
AC_MSG_RESULT($use_atrig_erf)

if test x$use_atrig_erf = xyes ; then
  ATRIG_ERF="#define ATRIG_ERF 1"
else
  ATRIG_ERF="#undef ATRIG_ERF"
fi
AC_SUBST(ATRIG_ERF)

AC_MSG_CHECKING(whether to use 32-bit or 64-bit locations)
AC_ARG_ENABLE(ulong,[AS_HELP_STRING([--enable-ulong],
  [enable 64-bit locations (only available on 64-bit systems) [default=32-bit]])],
  [use_ulong=$enableval],[use_ulong=no])

if test x$use_ulong = xyes -a 0$ac_cv_sizeof_void_p -ge 8 ; then
  UINT_TYPE=uint64_t
  AC_MSG_RESULT(64-bit)
else
  UINT_TYPE=uint32_t
  AC_MSG_RESULT(32-bit)
fi
AC_SUBST(UINT_TYPE)

AC_MSG_CHECKING(whether to use single or double precision)
AC_ARG_ENABLE(double,[AS_HELP_STRING([--disable-double],
  [disable double precision arithmetic [untested, default=double is enabled]])],
  [use_double=$enableval],[use_double=yes])

if test x$use_double = xno ; then
  REAL_TYPE=float
  echo
  echo -n "using ${REAL_TYPE} is not well tested, please report bugs if you find any..."
else
  REAL_TYPE=double
fi
AC_MSG_RESULT($REAL_TYPE)
AC_SUBST(REAL_TYPE)

AC_MSG_CHECKING(whether to enable advanced branching operations)
AC_ARG_ENABLE(advanced-branching, [AS_HELP_STRING([--enable-advanced-branching],
  [enable advanced branching operations to reduce retaping [default=no].
The boolean valued comparison operators with two adouble arguments will 
not return boolean results but the active results may be used to automatically 
switch branches in conjunction with condassign or advector (see manual).])],
  [use_advbranch=$enableval],[use_advbranch=no])
AC_MSG_RESULT($use_advbranch)

if test x$use_advbranch = xyes ; then
  ADVBRANCH="#define ADOLC_ADVANCED_BRANCHING 1"
else
  ADVBRANCH="#undef ADOLC_ADVANCED_BRANCHING"
fi
AC_SUBST(ADVBRANCH)

AC_MSG_CHECKING(whether to enable reference counting for tapeless numbers)
AC_ARG_ENABLE(traceless-refcounting, [AS_HELP_STRING([--enable-traceless-refcounting],
  [enable reference counting for tapeless numbers [default=no].
With this enabled some additional checks will be conducted when setting the
number of directional derivatives for tapeless numbers using the SetNumDir()
function. ])],
  [use_adtl_refcnt=$enableval],[use_adtl_refcnt=no])
AC_MSG_RESULT($use_adtl_refcnt)

if test x$use_adtl_refcnt = xyes ; then
  ADTL_REFCNT="#define USE_ADTL_REFCOUNTING 1"
else
  ADTL_REFCNT="#undef USE_ADTL_REFCOUNTING"
fi
AC_SUBST(ADTL_REFCNT)

AC_MSG_CHECKING(whether to enable activity tracking)
AC_ARG_ENABLE(activity-tracking, [AS_HELP_STRING([--enable-activity-tracking],
  [enable activity tracking to reduce trace size but increased tracing time [default=no].
Only the operations involving actual dependency relationships from the
independent variables will be recorded on the trace, this however
requires more checks to be performed during the tracing and increases
tracing time. Useful only if memory is a constraint and tracing is
done fewer times than derivative computations.])],
  [use_activities=$enableval],[use_activities=no])
AC_MSG_RESULT($use_activities)

if test x$use_activities = xyes ; then
AC_DEFINE(ADOLC_TRACK_ACTIVITY,1,[ADOL-C activity tracking in live variables])
fi

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

AC_MSG_CHECKING(whether the default adouble constructor should initialize the value to 0)
AC_ARG_ENABLE(stdczero,
              [
AS_HELP_STRING([--disable-stdczero],[adouble default constructor does not initialize the value to zero
(improves performance but yields incorrect results for implicit array initializations, see manual) [default=enabled]])],
  [
  adolc_stdczero=no
  AC_MSG_RESULT(no)
  ],
  [
  adolc_stdczero=yes
  AC_MSG_RESULT(yes)
  ])

if test x$adolc_stdczero == xyes; then
AC_DEFINE(ADOLC_ADOUBLE_STDCZERO,1,[ADOL-C adouble zeroing mode])
fi

AC_MSG_CHECKING(whether the adouble variables have a late initialize option for functions like malloc/realloc)
AC_ARG_ENABLE(lateinit,
              [
AS_HELP_STRING([--enable-lateinit],[adouble constructors need to be called. With malloc or realloc that is not the case. With this option the adouble can do a late initialization.) [default=disabled]])],
  [
  adolc_late_init=yes
  AC_MSG_RESULT(yes)
  ],
  [
  adolc_late_init=no
  AC_MSG_RESULT(no)
  ])

if test x$adolc_late_init == xyes; then
AC_DEFINE(ADOLC_ADOUBLE_LATEINIT,1,[ADOL-C adouble late initialization mode])
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

AX_BOOST_BASE([1.54],[boost_new=yes],[boost_new=no])
if test x$boost_new = xyes ; then
CPPFLAGS="$CPPFLAGS $BOOST_CPPFLAGS"
AC_LANG_PUSH([C++])
AC_CHECK_HEADERS([boost/pool/pool_alloc.hpp])
AC_LANG_POP([C++])
AX_BOOST_SYSTEM
LDFLAGS="$LDFLAGS $BOOST_LDFLAGS $BOOST_SYSTEM_LIB"
fi

if test "x$want_boost" = xyes -a "x$boost_new" = xyes -a "x$ac_cv_header_boost_pool_pool_alloc_hpp" = xyes -a "x$link_system" = xyes ; then
    using_boost_pool=yes
    USE_BOOST_POOL="#define USE_BOOST_POOL 1"
else
    using_boost_pool=no
    USE_BOOST_POOL="#define USE_BOOST_POOL 0"
fi
AC_SUBST(USE_BOOST_POOL)

if test "x$using_boost_pool" = xyes -a "x$ac_adolc_openmpflag" = x ; then
   AC_DEFINE(BOOST_POOL_NO_MT,1,[Boost pool should not assume multithreading])
fi

# ADOL-C Sparse facility
AC_MSG_CHECKING(whether to build sparse drivers)
AC_ARG_ENABLE(sparse,
            [
AS_HELP_STRING([--enable-sparse],[build sparse drivers [default=disabled]])],
            [sparse=$enableval
             AC_MSG_RESULT($sparse)],
            [sparse="no"
             AC_MSG_RESULT(no)])

COLPACK_CONF

AM_CONDITIONAL(SPARSE,[test x${sparse} = xyes])

SPARSE_DRIVERS="#undef SPARSE_DRIVERS"

if test x$sparse = xyes ; then
   AC_DEFINE(SPARSE,[1],[Define 1 if sparse derivative propagation is to be enabled])
   SPARSE_DRIVERS="#define SPARSE_DRIVERS 1"
fi

AC_SUBST(SPARSE_DRIVERS)

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

MPI_CONF_POST
AMPI_CONF
MEDIPACK_CONF

AC_ARG_WITH(soname,
            [AS_HELP_STRING([--with-soname=NAME],
              [user can choose what to call his library here [default: if ampi enabled then adolc_ampi otherwise adolc]])],
      [adolclib=$withval])

AC_SUBST(adolclib)

AC_MSG_CHECKING(whether tape_doc should compute values as it prints the tape contents)
AC_ARG_ENABLE(tapedoc_values,
              [
		AS_HELP_STRING([--disable-tapedoc-values],
			       [should the tape_doc routine compute the values as it interprets and prints the tape contents [default=enabled]])],
  [
  tapedoc_values=no
  ],
  [
  tapedoc_values=yes
  ])
AC_MSG_RESULT($tapedoc_values)

if test x$tapedoc_values == xyes; then
AC_DEFINE(ADOLC_TAPE_DOC_VALUES,1,[ADOL-C tape_doc routine computes values])
fi

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
    clang* | */clang*) 
	# do nothing but don't want to accidentally match the 'cl' patterns below
	;;
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
		ADOL-C/include/adolc/internal/adolc_settings.h
                ADOL-C/include/adolc/drivers/Makefile 
                ADOL-C/include/adolc/internal/Makefile
                ADOL-C/include/adolc/lie/Makefile
                ADOL-C/include/adolc/sparse/Makefile
                ADOL-C/include/adolc/tapedoc/Makefile
                ADOL-C/src/Makefile
                ADOL-C/src/drivers/Makefile 
                ADOL-C/src/lie/Makefile
                ADOL-C/src/sparse/Makefile
                ADOL-C/src/tapedoc/Makefile
                ADOL-C/doc/version.tex
                ADOL-C/examples/Makefile
                ADOL-C/examples/additional_examples/Makefile
                ADOL-C/examples/additional_examples/cuda/Makefile
                ADOL-C/examples/additional_examples/clock/Makefile
                ADOL-C/examples/additional_examples/checkpointing/Makefile
                ADOL-C/examples/additional_examples/ext_diff_func/Makefile
                ADOL-C/examples/additional_examples/fixpoint_exam/Makefile
                ADOL-C/examples/additional_examples/hessmat/Makefile
                ADOL-C/examples/additional_examples/lie/Makefile
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
                ADOL-C/examples/additional_examples/param/Makefile
                ADOL-C/test/Makefile
                ADOL-C/swig/Makefile
                MSVisualStudio/v14/sparse/config.h
                MSVisualStudio/v14/nosparse/config.h
                MSVisualStudio/v14/x64/sparse/config.h
                MSVisualStudio/v14/x64/nosparse/config.h
		adolc.spec
                adolclib.pc
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

  C compiler:                         ${CC}
  C++ compiler:                       ${CXX}
  Linker:                             ${LD}
  Source code location:               `pwd`
  Install path:                       ${prefix}
  
  CFLAGS:                             ${ac_adolc_cflags}
  CXXFLAGS:                           ${ac_adolc_cxxflags}

  Use Boost pool allocator:           ${using_boost_pool}
  Use ADOL-C debug mode:              ${adolc_debug}
  Use ADOL-C hard debug mode:         ${adolc_harddebug}
  Zero value in adouble default ctor: ${adolc_stdczero}"

if [[ "x${ac_adolc_openmpflag}" != "x" ]] ; then
    echo \
"  Access thread number via errno:     ${adolc_tserrno}"
fi

echo \
"  Use ADOL-C activity tracking:       ${use_activities}
  Use ADOL-C late init mode:          ${adolc_late_init}

  Build sparse drivers:               ${sparse}
  Build with adjoinable MPI support:  ${adolc_ampi_support}
  Build with MediPack support:        ${adolc_medipack_support}
  Build with ColPack:		      ${have_colpack}"
if test x$sparse = xyes && test x$have_colpack = xno; then
    echo -e \
"  \033@<:@1;31mCompressed sparse structures will not be available
  Only sparsity patterns can be computed\033@<:@0m"
    warn=true
fi
echo

echo \
"  Build documented examples:          ${docexa}
  Build additional examples:          ${addexa}
  tape_doc routine computes values:   ${tapedoc_values}

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
  with a writable directory path otherwise administrator privileges
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
        * add \"-L${libdir} -l$adolclib\" to your linker call
        * extend your linker call by \"-Wl,--rpath -Wl,${libdir}\"
          (if you wish to skip the point \"executing\")
     executing (do one of the following things):
        * add ${libdir} to your LD_LIBRARY_PATH variable
        * ask your system administrator for adding ${libdir} to
          the global file containing library search paths (/etc/ld.so.conf)

     (or use the static library by replacing
        \"-L${libdir} -l$adolclib\" with
        \"${libdir}/lib$adolclib.a\" when linking)

  See README for instructions on how to change to a nonlocal installation!

*****************************************************************************
"
;;
esac
if test "x$enable_static" != "xno" ; then
echo \
" Linking a user program with a static ADOL-C library will always result
  in a segmentation fault. Please always use the shared library.
*****************************************************************************
"
fi
