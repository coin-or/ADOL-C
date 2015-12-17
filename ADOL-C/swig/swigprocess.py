import re
import os.path
import sys
import argparse
import glob
import shlex
import shutil
import subprocess

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
    s = 'c++ -std=c++11 -E -C -P -o ' + outfile + ' -Iinclude -nostdinc -DSWIGPRE ' + infile
    cmd = shlex.split(s)
    subprocess.call(cmd)

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
        cmd = shlex.split(s)
        warn = subprocess.check_output(cmd,stderr=subprocess.STDOUT)
        s = 'R CMD SHLIB -o ' + modname + '.so ' + outfile + ' -L../.libs -ladolc' + ' -Wl,-no-undefined'
        evars = os.environ 
        evars['PKG_CPPFLAGS'] = "-I../include -std=c++11" 
        cmd = shlex.split(s)
        warn += subprocess.check_output(cmd,env=evars,stderr=subprocess.STDOUT)
        shutil.move(modname + '.so', lang)
        shutil.move(modname + '.R', lang)
        writeOutput(warn,'warnings-r.txt')
    elif lang == 'python':
        python_cflags = subprocess.check_output(['python-config','--cflags'])
        python_ldflags = subprocess.check_output(['python-config','--ldflags'])
        s = 'swig -python -c++ -o ' + outfile + ' ' + infile
        cmd = shlex.split(s)
        warn = subprocess.check_output(cmd,stderr=subprocess.STDOUT)
        s = 'c++ -I../include -std=c++11 -fPIC -Wall -shared -o _' + modname + '.so ' + python_cflags.rstrip() + ' ' + outfile + ' -L../.libs -ladolc ' + python_ldflags.rstrip() + ' -Wl,-no-undefined'
        cmd = shlex.split(s)
        warn += subprocess.check_output(cmd,stderr=subprocess.STDOUT)
        shutil.move('_' + modname + '.so', lang)
        shutil.move(modname + '.py', lang)
        writeOutput(warn,'warnings-python.txt')

    os.remove(outfile)
    for f in glob.glob('*.o'):
        os.remove(f)

def main():
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
    
    invoke_cpp('adolc_all_in.h', 'adolc_all_pre.h')
    lines = readFile('adolc_all_pre.h')
    lines = reinstate_nonlocal_include(lines)
    writeOutput(lines,'adolc_all.h')
    cleanup('adolc_all_pre.h')
    invoke_swig_compile('python','adolc-python.i','adolc_python_wrap.cxx','adolc')
    invoke_swig_compile('R','adolc-r.i','adolc_r_wrap.cpp','adolc')

if __name__ == '__main__':
    main()
