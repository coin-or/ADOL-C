#!/bin/bash -x
INCFILES=(adolc.h adalloc.h adouble.h adutils.h adutilsc.h \
         common.h convolut.h dvlparms.h fortutils.h \
         interfaces.h taping.h usrparms.h \
         externfcts.h checkpointing.h fixpoint.h \
         adolc_sparse.h adolc_openmp.h \
         revolve.h)
INCFILES_SPARSE=(sparsedrivers.h sparse_fo_rev.h)

INCFILES_DRIVERS=(drivers.h odedrivers.h taylor.h)

INCFILES_TAPEDOC=(tapedoc.h)

if [[ $# < 1 ]]; then
	echo "usage: installer.sh config [arch]"
	echo "where config is sparse|nosparse"
	echo "and arch is optionally Win32 or x64"
	echo "if omitted arch defaults to Win32"
	exit 0
fi

if [[ $# > 1 ]]; then 
	case $2 in 
		win32|Win32) conf=$1
		       arch=.
		       cf=$1_win32
		       suffix=x86
		       ;;
		x64) conf=x64/$1
		     arch=x64
		     cf=$1_x64
		     suffix=x64
		     ;;
	 esac
else
	 conf=$1
	 arch=.
	 cf=$1_win32
	 suffix=x86
fi

mkdir -p tmp
mkdir -p tmp/include/adolc
mkdir -p tmp/lib
mkdir -p tmp/bin
mkdir -p tmp/doc

mkdir -p tmp/include/adolc/sparse
mkdir -p tmp/include/adolc/drivers
mkdir -p tmp/include/adolc/tapedoc

for i in ${INCFILES[*]} ; do
	cp ../../ADOL-C/src/$i tmp/include/adolc
done

for i in ${INCFILES_SPARSE[*]} ; do
	cp ../../ADOL-C/src/sparse/$i tmp/include/adolc/sparse
done

for i in ${INCFILES_DRIVERS[*]} ; do
	cp ../../ADOL-C/src/drivers/$i tmp/include/adolc/drivers
done

for i in ${INCFILES_TAPEDOC[*]} ; do
	cp ../../ADOL-C/src/tapedoc/$i tmp/include/adolc/tapedoc
done

cp $conf/adolc.dll tmp/bin
cp $conf/adolc.lib tmp/lib
cp ../../ADOL-C/doc/* tmp/doc
cp $arch/vcredist_$suffix.exe tmp/
echo "@echo off" > tmp/setup.bat
echo "vcredist_${suffix}.exe" >> tmp/setup.bat
cd tmp
zip -r ../adolc_$cf.zip ./
cd ..
rm -rf tmp
