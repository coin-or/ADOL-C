# spec file for package adolc (Version 2.5.0)

# norootforbuild

%define packver 2.5.0-stable

Name:           adolc
Version:        2.5.0
Release:        0.1
License:        GPLv2 or CPL
Summary:        Algorithmic Differentiation Library for C/C++
Url:            http://projects.coin-or.org/ADOL-C
Group:          Development/Languages/C and C++
Source:         %{name}-%{packver}.tar.bz2
Source1:        ColPack.tar.gz
BuildRequires:  gcc-c++ libstdc++-devel
BuildRoot:      %{_tmppath}/%{name}-%{version}-build
AutoReqProv:    on

%description  
The package ADOL-C (Automatic Differentiation by OverLoading in C++)
facilitates the evaluation of first and higher derivatives of vector
functions that are defined by computer programs written in C or
C++. The resulting derivative evaluation routines may be called from
C/C++, Fortran, or any other language that can be linked with C.

The numerical values of derivative vectors are obtained free of
truncation errors at a small multiple of the run time and randomly
accessed memory of the given function evaluation program.

%package -n libadolc1
Summary:        Algorithmic Differentiation Library for C/C++
Group:          Development/Languages/C and C++

%description -n libadolc1
The package ADOL-C (Automatic Differentiation by OverLoading in C++)
facilitates the evaluation of first and higher derivatives of vector
functions that are defined by computer programs written in C or
C++. The resulting derivative evaluation routines may be called from
C/C++, Fortran, or any other language that can be linked with C.

The numerical values of derivative vectors are obtained free of
truncation errors at a small multiple of the run time and randomly
accessed memory of the given function evaluation program.

%package devel
Summary:        Algorithmic Differentiation Library for C/C++ -- development files
Group:          Development/Languages/C and C++
Requires:       libadolc1 = %{version}

%description devel
The package ADOL-C (Automatic Differentiation by OverLoading in C++)
facilitates the evaluation of first and higher derivatives of vector
functions that are defined by computer programs written in C or
C++. The resulting derivative evaluation routines may be called from
C/C++, Fortran, or any other language that can be linked with C.

The numerical values of derivative vectors are obtained free of
truncation errors at a small multiple of the run time and randomly
accessed memory of the given function evaluation program.

This package provides the development environment for adolc

%package doc
Summary:        Algorithmic Differentiation Library for C/C++ -- documentation
Group:          Development/Languages/C and C++
BuildArch:      noarch

%description doc
The package ADOL-C (Automatic Differentiation by OverLoading in C++)
facilitates the evaluation of first and higher derivatives of vector
functions that are defined by computer programs written in C or
C++. The resulting derivative evaluation routines may be called from
C/C++, Fortran, or any other language that can be linked with C.

The numerical values of derivative vectors are obtained free of
truncation errors at a small multiple of the run time and randomly
accessed memory of the given function evaluation program.

This package provides the user´s manual for adolc

%prep
%setup -q -n %{name}-%{packver} -b 1
pushd ThirdParty
mv %{_builddir}/ColPack/* ColPack/
rm -rf %{_builddir}/ColPack
popd

%build
pushd ThirdParty/ColPack
make %{_smp_mflags}
popd
autoreconf -v --install --force
%configure --prefix=/usr
make %{_smp_mflags}

%install
%makeinstall
install -d %{buildroot}%{_datadir}/doc/packages/%{name}
install -m 644 README AUTHORS BUGS LICENSE INSTALL TODO %{buildroot}%{_datadir}/doc/packages/%{name}
install -m 644 ADOL-C/doc/adolc-manual.pdf %{buildroot}%{_datadir}/doc/packages/%{name}
install -m 644 ADOL-C/doc/short_ref.pdf %{buildroot}%{_datadir}/doc/packages/%{name}
find %{buildroot} -type f -name '*.la' -delete -print

%clean
rm -rf %{buildroot}
rm -rf %{_builddir}/%{name}-%{packver}

%post -n libadolc1 -p /sbin/ldconfig
%postun -n libadolc1 -p /sbin/ldconfig

%files -n libadolc1
%defattr(-,root,root)
%{_libdir}/libadolc.so.*

%files devel
%defattr(-,root,root)
%dir %{_includedir}/adolc
%dir %{_includedir}/adolc/drivers
%dir %{_includedir}/adolc/sparse
%dir %{_includedir}/adolc/tapedoc
%{_includedir}/adolc/*.h
%{_includedir}/adolc/drivers/*.h
%{_includedir}/adolc/sparse/*.h
%{_includedir}/adolc/tapedoc/*.h
%{_libdir}/libadolc.so
%{_libdir}/libadolc.a

%files doc
%defattr(-,root,root)
%dir %{_datadir}/doc/packages/%{name}
%{_datadir}/doc/packages/%{name}/*

%changelog
