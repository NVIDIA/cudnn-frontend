{
  pkgs,
  cmake,
  pkg-config,
  fetchFromGitHub,
  patchelf,
  cuda-pkgs ? pkgs.cudaPackages_12,
  cudnn-frontend ? pkgs.callPackage ./cudnn-frontend.nix {},
  dlpack ? pkgs.callPackage ./dlpack.nix {},
  cudnn-frontend-version ? import ./version.nix,
  doCheck ? false,
}:
let
  compiled-extension-src = pkgs.stdenv.mkDerivation{
    name = "cudnn-frontend-python-extension-src";
    phases = ["installPhase"];
    installPhase = ''
      cp -r ${../python} $out
    '';
  };
  compiled-extension = cuda-pkgs.backendStdenv.mkDerivation {
    name = "cudnn-frontend-python-compiled-extension";
    src = compiled-extension-src;
    propagatedBuildInputs = with pkgs.python3Packages; [
      cudnn-frontend
    ];
    nativeBuildInputs = with pkgs.python3Packages; [
      cmake
      pybind11
      dlpack
    ];
    postInstall = ''
      patchelf \
        --add-rpath ${cuda-pkgs.cudnn}/lib \
        $out/python/cudnn/*.so
    '';
  };

  python-src = pkgs.stdenv.mkDerivation{
    name = "cudnn-frontend-python-src";
    phases = ["installPhase"];
    installPhase = ''
      mkdir -p $out
      cp ${../setup.py} $out/setup.py

      # in a nix build, the compiled extension is already built, so we don't
      # need pybind11, cmake etc to be available at this point
      cat ${../requirements.txt} | grep -v pybind | grep -v clang-format > $out/requirements.txt
      echo '[build-system]' > $out/pyproject.toml
      echo 'requires = ["setuptools>=64"]' >> $out/pyproject.toml
      tail -n+3 ${../pyproject.toml} >> $out/pyproject.toml

      mkdir -p $out/python/cudnn
      # force ctypes.CDLL to load the cudnn library from the correct path
      cat ${../python/cudnn/__init__.py} \
          | sed 's~ctypes\.CDLL("libcudnn.so")~ctypes.CDLL("${cuda-pkgs.cudnn}/lib/libcudnn.so")~' \
          >> $out/python/cudnn/__init__.py
      cp -r ${../python/cudnn/datatypes.py} $out/python/cudnn/datatypes.py
    '' + (if doCheck then ''
      mkdir -p $out/test
      cp -r ${../test/python_fe} $out/test/python_fe
    '' else "");
  };
in
pkgs.python3Packages.buildPythonPackage{
  pname = "cudnn-frontend-python";
  version = cudnn-frontend-version;
  format = "pyproject";
  src = python-src;
  nativeBuildInputs = with pkgs.python3Packages; [];
  propagatedBuildInputs = with pkgs.python3Packages; [
    compiled-extension
    setuptools
    cuda-pkgs.cudnn
    pkgs.autoAddDriverRunpath
  ];
  # add testing, if enabled
  inherit doCheck;
  checkInputs = with pkgs.python3Packages; [
    pytest
    pytorch
    looseversion
  ];
  checkPhase = ''
    if [ -d test ]; then
      cd test/python_fe
      python -m pytest
    fi
  '';
}
