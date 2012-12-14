#!/bin/bash

autoreconf -fi

./configure

pushd ADOL-C/doc

pdflatex adolc-manual.tex
pdflatex adolc-manual.tex
pdflatex adolc-manual.tex

latex adolc-manual.tex
latex adolc-manual.tex
latex adolc-manual.tex
dvips adolc-manual.dvi

popd
