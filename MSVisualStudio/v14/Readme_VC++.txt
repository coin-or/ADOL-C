The solution and project files were generated on Visual Studio 2010 Ultimate.
Compatibility with older versions is not guarenteed.

Before adolc may be built using Visual studio the following step must
be taken:
- Extract the boost library headers into the subdirectory
MSVisualStudio/v14/boost
This directory will be added to the build as additional include directory
Compile boost using instructions from the boost website and choose
the stage directory for 32 bit static build as static32 and for 64 bit static
build as static64
- Extract and place the ColPack sources in the subdirectory
MSVisualStudio/v14/ColPack
This directory will be used to build ColPack too. Copy ColPack.vcxproj into 
it. ColPack will be built by adolc.sln

The solution file adolc.sln can build both the sparse as well as
the nosparse versions of adolc.dll. In visual studio open this solution file
and select the solution 'adolc' in the Solution Explorer, from the toolbar
select the build configuration sparse or nosparse, then from the Build
menu select Build Solution (shortcut key F7).

