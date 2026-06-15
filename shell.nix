{ pkgs ? import <nixpkgs> {} }:
let
  pythonEnv = pkgs.python313.withPackages (ps: with ps; [
    matplotlib
    pyqt6
    numpy
    jax
  ]);
in
pkgs.mkShell {
  buildInputs = [
    pkgs.python313
    pkgs.stdenv.cc.cc.lib
    pkgs.libz
    pkgs.xorg.libX11
  ];
  packages = [
    pythonEnv
  ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libz}/lib";
}
