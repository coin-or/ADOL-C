{
  description = "A Package for Automatic Differentiation of Algorithms Written in C/C++ ";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { flake-utils, nixpkgs, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages = rec {
          adolc-cmake = pkgs.stdenv.mkDerivation {
            name = "adolc";
            src = ./.;
            nativeBuildInputs = [ pkgs.cmake ];
          };

          adolc-autotools = adolc-cmake.overrideAttrs (
            _: _: {
              nativeBuildInputs = [ pkgs.autoreconfHook ];
            }
          );

          default = adolc-cmake;
        };
      }
    );
}
