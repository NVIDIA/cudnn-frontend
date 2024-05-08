{
  stdenv,
  fetchFromGitHub,
  cmake
}:
stdenv.mkDerivation rec{
  name = "dlpack";
  version = "0.8";
  src = fetchFromGitHub {
    owner = "dmlc";
    repo = "${name}";
    rev = "v${version}";
    sha256 = sha256:IcfCoz3PfDdRetikc2MZM1sJFOyRgKonWMk21HPbrso=;
  };
  nativeBuildInputs = [
    cmake
  ];
  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DBUILD_MOCK=OFF"
  ];
}
