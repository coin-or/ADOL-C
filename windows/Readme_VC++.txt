The solution file windows/adolc.sln can build both the sparse as well as
the nosparse versions of adolc.dll. In visual studio open this solution file
and select the solution 'adolc' in the Solution Explorer, then from the Build
menu select Build Solution (shortcut key F7).

In order to build the nosparse version nothing further is needed. For the 
sparse version the source code of ColPack must be extracted into the
ThirdParty/ directory. Read the Readme_VC++.txt file in ThirdParty/ColPack 
for further instructions.
