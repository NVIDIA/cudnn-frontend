{
  pkgs,
  cmake,
  pkg-config,
  catch2_3,
  fetchFromGitHub,
  cuda-pkgs ? pkgs.cudaPackages_12,
  cudnn-frontend-version ? import ./version.nix,
  doCheck ? false,
}:
let
  src = pkgs.stdenv.mkDerivation{
    name = "cudnn-frontend-src";
    phases = ["installPhase"];
    installPhase = ''
      mkdir -p $out
      cp ${../CMakeLists.txt} $out/CMakeLists.txt
      cp -r ${../cmake} $out/cmake
      cp -r ${../include} $out/include
      '' + (if doCheck then "cp -r ${../test} $out/test;" else "") + ''
    '';
  };
in
cuda-pkgs.backendStdenv.mkDerivation{
  pname = "cudnn-frontend";
  version = cudnn-frontend-version;
  inherit src;
  nativeBuildInputs = [
    cmake
    pkg-config
  ];
  propagatedBuildInputs = [
    cuda-pkgs.cudnn
    pkgs.autoAddDriverRunpath
    cuda-pkgs.cudatoolkit
  ];

  # add testing, if enabled
  inherit doCheck;
  cmakeFlags = [
    "-DCUDNN_FRONTEND_BUILD_SAMPLES=OFF"
    "-DCUDNN_FRONTEND_BUILD_UNIT_TESTS=${if doCheck then "ON" else "OFF"}"
  ];
  checkInputs = [ catch2_3 ];
}
