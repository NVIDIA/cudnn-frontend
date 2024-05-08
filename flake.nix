{
  description = "A C++ header-only library that wraps the cuDNN C backend API.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  nixConfig = {
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
    extra-trusted-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      cudnn-frontend = pkgs.callPackage ./nix-support/cudnn-frontend.nix {};
      cudnn-frontend-python = pkgs.callPackage ./nix-support/cudnn-frontend-python.nix { inherit cudnn-frontend; };
    in {
      packages = {
        cudnn-frontend = cudnn-frontend;
        cudnn-frontend-python = cudnn-frontend-python;
        default = cudnn-frontend;
      };
      devShell = pkgs.mkShell {
        buildInputs = [ cudnn-frontend cudnn-frontend-python ];
      };
    });
}
