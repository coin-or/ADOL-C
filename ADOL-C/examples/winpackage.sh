NAMES=(detexam odexam powexam speelpenning tapeless_scalar tapeless_vector)

if [[ $# < 1 ]]; then
	echo "usage: $0 config [arch]"
	echo "where config is sparse|nosparse"
	echo "and arch is optionally Win32 or x64"
	echo "if ommitted arch defaults to Win32"
	exit 0
fi

if [[ $# > 1 ]]; then
	case $2 in
		win32|Win32) conf=$1
			arch=.
			cf=$1_win32
			;;
		x64)    conf=x64/$1
			arch=x64
			cf=$1_x64
			;;
	esac
else
	conf=$1
	arch=.
	cf=$1_win32
fi

mkdir -p tmp
mkdir -p tmp/bin

for i in ${NAMES[*]} ; do
	cp $conf/$i.exe tmp/bin
done

cd tmp
zip -r ../adolc_examples_$cf.zip ./
cd ..
rm -rf tmp
