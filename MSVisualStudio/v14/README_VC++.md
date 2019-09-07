# Build ADOL-C with Visual Studio

The solution and project files were generated on Visual Studio 2010 Ultimate.
Compatibility with older versions is not guaranteed.

## Dependencies
Before ADOL-C may be built using Visual Studio the following step must
be taken:
- Extract the **boost library** headers into the subdirectory MSVisualStudio/v14/boost.
  This directory will be added to the build as additional include directory.
  Compile boost using instructions from the boost website and choose
  the stage directory for 32 bit static build as static32 and for 64 bit static
  build as static64

  Extract all boost files and folders and build boost:
  ```cmd
  > cd boost\
  > b2.exe address-model=64 threading=multi runtime-link=static link=static --stagedir=64
  ```
  If you want to build for 32 bit change 64 accordingly.


- Extract and place the **ColPack** sources in the subdirectory
  `MSVisualStudio/v14/ColPack`
  This directory will be used to build ColPack too. Copy ColPack.vcxproj into
  it. ColPack will be built by adolc.sln

The structure should now look like
```
| - MSVisualStudio/
  | - v14/
    | - boost/
    | - static64/           // Install boost lib files here
    | - boost/
      | - boost sub folders and headers go here
      | - ...
    | - ColPac
```

## Build with MS Visual Studio solution
The solution file `adolc.sln` can build both the sparse as well as
the nosparse versions of adolc.dll. In Visual Studio open this solution file
and select the solution 'adolc' in the Solution Explorer, from the toolbar
select the build configuration sparse or nosparse, then from the build
menu select Build Solution (shortcut key F7).

## Test the ADOL-C build
To test your freshly build ADOL-C run the examples MSVC solution `ADOL-C/examples/adolc_examples.sln`.
Make sure to test the right solution platform (x64 or Win32). There are tests for sparse and non-sparse examples.
