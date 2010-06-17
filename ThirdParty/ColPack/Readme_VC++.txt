In order to compile the ColPack sources from the tarball with VC++ using
the project file ColPack.vcxproj please apply the following patch:

colpack_vcxcompile.patch

The complete steps are as follows

1. Go to directory ThirdParty/

tar -xzf ColPack.tar.gz
cd ColPack
patch -p0 <colpack_vcxcompile.patch

2. In Visual Studio open the solution file

windows/adolc.sln

select the solution 'adolc' in the Solution Explorer, from the toolbar select
the build configuration sparse and from Build menu select Build Solution 
(shortcut key F7).

This will build ColPack and link it with the sparse version of adolc.dll

