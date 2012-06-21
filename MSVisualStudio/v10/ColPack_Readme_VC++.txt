The complete steps are as follows

1. Unpack the ColPack sources

tar -xzf ColPack-$VERSION.tar.gz

or use your favourite extraction utility under windows. Rename the
subdirectory to be simply ColPack. Then move the file ColPack.vcxproj into 
this subdirectory

2. In Visual Studio open the solution file

adolc.sln

select the solution 'adolc' in the Solution Explorer, rightclick and select
Add -> Existing project and then choose the ColPack.vcxproj project from the
ColPack subdirectory. Right click on adolc and select Project Dependencies 
and tick the check mark on ColPack. Then select from the toolbar select 
the build onfiguration sparse and from Build menu select Build Solution 
(shortcut F7).

This will build ColPack and link it with the sparse version of adolc.dll
