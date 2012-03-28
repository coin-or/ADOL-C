The solution and project files were generated on Visual Studio 2010 Ultimate.
Compatibility with older versions is not guarenteed.

Before adolc may be built using Visual studio the following step must
be taken:
- Create a symbolic link ADOL-C\adolc pointing to ADOL-C\src
   Under Windows Vista/Windows 7/Windows 2008 Server
   use the mklink command
     cd ADOL-C
     mklink /j src adolc
   Under Windows XP/Windows 2003 Server
   use the junction command from Windows sysinternals suite:
   http://technet.microsoft.com/en-us/sysinternals/bb842062.aspx
     cd ADOL-C
     junction src adolc

The solution file adolc.sln can build both the sparse as well as
the nosparse versions of adolc.dll. In visual studio open this solution file
and select the solution 'adolc' in the Solution Explorer, from the toolbar
select the build configuration sparse or nosparse, then from the Build
menu select Build Solution (shortcut key F7).

In order to build the nosparse version nothing further is needed. For the 
sparse version the source code of ColPack must be extracted into 
this directory. Read the ColPack Readme_VC++.txt file for further
instructions.
