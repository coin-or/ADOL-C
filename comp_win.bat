@if "%1"=="--help" goto help

@if "%1"=="--with-exa" goto exa

:normal     rem build library only
@set e=0
@set c=%1
@goto make

:exa
@set e=1
@set c=%2
@goto make

:make
@make -f Makefile.win EXA=%e% %c%
@goto end

:help
@echo **********************************************************
@echo * Usage: "comp_win [--with-exa] | [--help] | [<target>]" *
@echo *                                                        *
@echo *      --with-exa   build including examples             *
@echo *      --help       print this help screen               *
@echo *                                                        *
@echo *      target:      library - as the name says           *
@echo *                   clean   - clean up package           *
@echo **********************************************************
@goto end

:end
