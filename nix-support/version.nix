let
  cmakelists = builtins.readFile ../CMakeLists.txt;
  versions =
    builtins.match
    ''.*project\(cudnn_frontend VERSION ([^)]+)\).*''
    cmakelists;
  version = if builtins.length versions > 0 then
    builtins.elemAt versions 0
  else
    builtins.trace "Warning: Unable to find version. Defaulting to 0.0.0" "0.0.0";
in
  version
