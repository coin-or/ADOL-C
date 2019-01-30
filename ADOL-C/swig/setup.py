##############################################################################
## setup.py -- Script to generate swig based interface for python
## Revision: $Id$
##
## Copyright (C) Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

from __future__ import print_function
from swigprocess import prepare_flat_header
from numpy.distutils import misc_util as np_dist
from distutils.core import setup, Extension
from distutils.cmd import Command
from distutils.command.build_ext import build_ext
from distutils.command.build import build
from distutils.command.install import install
import os
import subprocess

def compile_dynlib(prefix,colpackdir,boostdir):
    import subprocess
    from multiprocessing import cpu_count
    nproc = cpu_count()
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    librootdir = os.path.abspath(uppath(os.getcwd(),2))
    try:
        out = subprocess.check_output(['autoreconf','-fi'],cwd=librootdir,universal_newlines=True)
        print(out)
        out = subprocess.check_output(['./configure',
                                       '--prefix=%s' % prefix,
                                       '--with-colpack=%s' % colpackdir,
                                       '--with-boost=%s' % boostdir ],
                                      cwd=librootdir,
                                      universal_newlines=True)
        print(out)
        out = subprocess.check_output(['make', '-j%d' % nproc, 'install'],cwd=librootdir,
                                      universal_newlines=True)
        print(out)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print('cmd = ', e.cmd)
        raise SystemExit()

class build_adolc(Command):
    user_options = [
        ('lib-prefix=', None, 'prefix to install adolc library'),
        ('colpack-dir=', None, 'directory in which colpack is installed'),
        ('boost-dir=', None, 'directory in which boost is installed') ]
    command_name = 'build_lib'

    def initialize_options(self):
        self.lib_prefix = None
        self.colpack_dir = None
        self.boost_dir = None

    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('lib_prefix','lib_prefix'),
                                   ('colpack_dir', 'colpack_dir'),
                                  ('boost_dir', 'boost_dir') )

    def run(self):
        compile_dynlib(self.lib_prefix,self.colpack_dir,self.boost_dir)


class build_swigadolc(build_ext,object):
    command_name = 'build_ext'
    user_options = build_ext.user_options + [
        ('lib-prefix=', None, 'prefix of installed adolc library') ]
    def initialize_options(self):
        super(build_swigadolc,self).initialize_options()
        self.lib_prefix = None

    def finalize_options(self):
        super(build_swigadolc,self).finalize_options()
        self.set_undefined_options('build',
                                   ('lib_prefix','lib_prefix'))
        prefix = self.lib_prefix
        print('prefix = ', prefix)
        self.include_dirs.append(os.path.join(prefix,'include'))
        self.library_dirs.append(os.path.join(prefix,'lib64'))
        self.rpath.append(os.path.join(prefix,'lib64'))
        prepare_flat_header()


class buildthis(build,object):
    command_name = 'build'
    user_options = build.user_options + [
        ('lib-prefix=', None, 'prefix to install adolc library'),
        ('colpack-dir=', None, 'directory in which colpack is installed'),
        ('boost-dir=', None, 'directory in which boost is installed') ]

    def lib_doesnot_exist(self):
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler
        compiler = new_compiler()
        customize_compiler(compiler)
        lib_file = compiler.find_library_file([os.path.join(self.lib_prefix,'lib64')],'adolc')
        return lib_file is None

    def initialize_options(self):
        super(buildthis,self).initialize_options()
        self.lib_prefix = None
        self.colpack_dir = None
        self.boost_dir = None

    #sub_commands = [ ('build_lib', lib_doesnot_exist),
    #                 ('build_ext', None) ]
    sub_commands = [ ('build_lib', None),
                     ('build_ext', None) ]

class installthis(install,object):
    command_name = 'install'
    user_options = install.user_options + [
        ('lib-prefix=', None, 'prefix to install adolc library'),
        ('colpack-dir=', None, 'directory in which colpack is installed'),
        ('boost-dir=', None, 'directory in which boost is installed') ]

    def initialize_options(self):
        self.lib_prefix = None
        self.colpack_dir = None
        self.boost_dir = None
        super(installthis,self).initialize_options()

    def finalize_options(self):
        super(installthis,self).finalize_options()
        if self.lib_prefix is None:
            self.lib_prefix = os.path.join(os.environ['HOME'],'adolc_base')
        if self.colpack_dir is None:
            self.colpack_dir = os.path.join(os.environ['HOME'],'adolc_base')
        if self.boost_dir is None:
            self.boost_dir = '/usr'
        self.finalized = 1
        buildobj = self.distribution.get_command_obj('build')
        buildobj.set_undefined_options('install',
                                   ('lib_prefix','lib_prefix'),
                                   ('colpack_dir', 'colpack_dir'),
                                   ('boost_dir', 'boost_dir') )
        
incdirs = np_dist.get_numpy_include_dirs()
python_ldflags = subprocess.check_output(['python-config','--ldflags'],universal_newlines=True)

adolc_mod = Extension('_adolc',
                      sources=['adolc-python.i'],
                      depends=['adolc_all.hpp'],
                      swig_opts=['-c++', '-dirvtable'],
                      libraries=['adolc'],
                      include_dirs=incdirs,
                      extra_compile_args=['-std=c++11', '-fPIC', '-w'],
                      extra_link_args=['-Wl,-no-undefined ' + python_ldflags.rstrip()])

setup(name='adolc',
      ext_modules=[adolc_mod],
      py_modules=['adolc'],
      version='2.7-trunk',
      cmdclass = { 'build_lib': build_adolc,
                   'build_ext': build_swigadolc,
                   'build': buildthis,
                   'install': installthis
               }
)
