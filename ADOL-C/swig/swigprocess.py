##############################################################################
## swigprocess.py -- Script to generate swig based interface for python and R
## Revision: $Id$
##
## Copyright (C) Kshitij Kulshreshtha
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

import re
import os.path
import sys
import argparse
import glob
import shlex
import shutil
import subprocess
import atexit

noerrors = True

@atexit.register
def call_make_clean():
    if noerrors:
        pass
    elif len(glob.glob('[Mm]akefile')) > 0:
        subprocess.call(['make','clean'])

def readFile(name):
    alllines = ''
    inputfile= open(name, 'r')
    for line in inputfile:
        alllines+=line
    inputfile.close()
    return alllines

def writeOutput(outstr, outfile):
    outputfile= open(outfile, 'w')
    outputfile.write(outstr)
    outputfile.close()


def appendFile(outfile, outstr):
    outputfile = open(outfile,'a')
    outputfile.write(outstr)
    outputfile.write('\n')
    outputfile.close()

def comment_all_includes(lines):
    s = r'(#\s*include\s*<\S*>)'
    p = re.compile(s,re.M|re.S)
    newlines = p.sub(r'//\1',lines)
    return newlines

def uncomment_local_includes(lines):
    s = r'//(#\s*include\s*<adolc/\S*>)'
    p = re.compile(s,re.M|re.S)
    newlines = p.sub(r'\1',lines)
    return newlines

def invoke_cpp(infile,outfile):
    s = os.environ['CXX'] + ' -std=c++11 -E -C -P -o ' + outfile + ' -Iinclude -nostdinc -DSWIGPRE ' + infile
    print('invoking:', s)
    cmd = shlex.split(s)
    try:
        warn = subprocess.check_output(cmd,universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print("error in cmd = ", e.cmd)
        exit()
    if len(warn) > 0:
        print(warn)

def reinstate_nonlocal_include(lines):
    s = r'//(#\s*include\s*<\S*>)'
    p = re.compile(s,re.M|re.S)
    newlines = p.sub(r'\1',lines)
    return newlines    

def cleanup(intfile):
    shutil.rmtree('include')
    os.remove(intfile)


def invoke_swig_compile(lang,infile,outfile,modname):
    try:
        os.mkdir(lang)
    except:
        shutil.rmtree(lang)
        os.mkdir(lang)

    if lang == 'R':
        s = 'swig -r -c++ -o ' + outfile + ' ' + infile
        print('invoking:', s)
        cmd = shlex.split(s)
        warn = ''
        try:
            warn += subprocess.check_output(cmd,stderr=subprocess.STDOUT,universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            print("error in cmd = ", e.cmd)
            exit()
        s = 'R CMD SHLIB -o ' + modname + '.so ' + outfile + ' -L../.libs -ladolc'
        if sys.platform.startswith('linux'):
            s += ' -Wl,-no-undefined'
        evars = os.environ 
        evars['PKG_CPPFLAGS'] = "-I../include -std=c++11" 
        print('invoking:', s)
        cmd = shlex.split(s)
        try:
            warn += subprocess.check_output(cmd,env=evars,stderr=subprocess.STDOUT,universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            print("error in cmd = ", e.cmd)
            exit()            
        shutil.move(modname + '.so', lang)
        shutil.move(modname + '.R', lang)
        shutil.move(outfile,lang)
        writeOutput(warn,'warnings-r.txt')
    elif lang == 'octave':
        s = 'swig -octave -c++ -o ' + outfile + ' ' + infile
        print('invoking:', s)
        cmd = shlex.split(s)
        warn = ''
        try:
            warn += subprocess.check_output(cmd,stderr=subprocess.STDOUT,universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            print("error in cmd = ", e.cmd)
            exit()
        s = 'mkoctfile -o ' + modname + '.oct ' + ' -I../include -std=c++11'  + ' -L../.libs -ladolc ' + outfile
        if sys.platform.startswith('linux'):
            s += ' -Wl,-no-undefined '
        print('invoking:', s)
        cmd = shlex.split(s)
        try:
            warn += subprocess.check_output(cmd,stderr=subprocess.STDOUT,universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            print("error in cmd = ", e.cmd)
            exit()            
        shutil.move(modname + '.oct', lang)
        shutil.move(outfile,lang)
        writeOutput(warn,'warnings-octave.txt')
    elif lang == 'python':
        python_cflags = subprocess.check_output(['python-config','--cflags'],universal_newlines=True)
        python_ldflags = subprocess.check_output(['python-config','--ldflags'],universal_newlines=True)
        from numpy.distutils import misc_util as npy_dist
        incp = npy_dist.get_numpy_include_dirs()
        npy_cflags = ''
        for p in incp:
            npy_cflags += ' -I' + p 

        s = 'swig -python -c++ -dirvtable -o ' + outfile + ' ' + infile
        print('invoking:', s)
        cmd = shlex.split(s)
        warn = ''
        try:
            warn += subprocess.check_output(cmd,stderr=subprocess.STDOUT,universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            print("error in cmd = ", e.cmd)
        p = re.compile(r'(.*)\.cxx',re.M|re.S)
        outhead = p.sub(r'\1.h',outfile)
        s = os.environ['CXX'] + ' -I../include -std=c++11 -fPIC -Wall -shared -o _' + modname + '.so ' + python_cflags.rstrip() + npy_cflags + ' ' + outfile + ' -L../.libs -ladolc ' + python_ldflags.rstrip() 
        if sys.platform.startswith('linux'):
            s += ' -Wl,-no-undefined'
        print('invoking:', s)
        cmd = shlex.split(s)
        try:
            warn += subprocess.check_output(cmd,stderr=subprocess.STDOUT,universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            print("error in cmd = ", e.cmd)
            exit()            
        shutil.move('_' + modname + '.so', lang)
        shutil.move(modname + '.py', lang)
        shutil.move(outfile,lang)
        shutil.move(outhead,lang)
        writeOutput(warn,'warnings-python.txt')

def finalClean(headfile,outfiles):
    if os.path.isfile(headfile):
        os.remove(headfile)
    for f in outfiles:
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob('*.o'):
        os.remove(f)

def prepare_flat_header():
    sys.path = [ os.getcwd() ] + sys.path
    p = os.getcwd() + '/../include/adolc'
    for (dp, dn, fn) in os.walk(p):
        ndp = re.sub(r'\.\./',r'',dp)
        for f in iter(fn):
            lines = readFile(dp + "/" + f)
            lines = comment_all_includes(lines)
            lines = uncomment_local_includes(lines)
            try:
                os.makedirs(ndp)
            except:
                pass
            writeOutput(lines, ndp + "/" + f)
    
    invoke_cpp('adolc_all_in.hpp', 'adolc_all_pre.hpp')
    lines = readFile('adolc_all_pre.hpp')
    lines = reinstate_nonlocal_include(lines)
    writeOutput(lines,'adolc_all.hpp')
    cleanup('adolc_all_pre.hpp')


def main(args):
    global noerrors
    noerrors = False
    prepare_flat_header()
    if args.py or args.all:
        invoke_swig_compile('python','adolc-python.i','adolc_python_wrap.cxx','adolc')
    if args.r or args.all:
        invoke_swig_compile('R','adolc-r.i','adolc_r_wrap.cpp','adolc')
    if args.oc or args.all:
        invoke_swig_compile('octave','adolc-octave.i','adolc_octave_wrap.cpp','adolc')
    finalClean('adolc_all.hpp',['adolc_python_wrap.cxx','adolc_python_wrap.h','adolc_r_wrap.cpp'])
    noerrors = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile Swig Interfaces')
    parser.add_argument('--py', '--python', action='store_true', 
                        help='compile python interface')
    parser.add_argument('--r', action='store_true',
                        help='compile R interface')
    parser.add_argument('--oc', '--octave', action='store_true',
                        help='compile Octave interface')
    parser.add_argument('--all', action='store_true', default=True,
                        help='compile all interfaces (python, R, octave) [default]')
    args = parser.parse_args()
    if args.py or args.r or args.oc:
        args.all = False
    try:
        cxx = os.environ['CXX']
    except KeyError:
        os.environ['CXX'] = 'c++'
    main(args)
