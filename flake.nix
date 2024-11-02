{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {self, nixpkgs, ... }@inp:
    let
      nixpkgs_configs = {
        default={allowUnfree= true;};
        with_cuda={
          cudaCapabilities = ["8.6"];
          cudaSupport = true;
          allowUnfree = true;
        };
      };
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        nixpkgs.lib.attrsets.mapAttrs (name: config:
          let pkgs = import nixpkgs { 
                inherit system config;
              };
              python = pkgs.python311.override {};
          in pkgs.mkShell {
              buildInputs = [
                  pkgs.cudaPackages.cudatoolkit
                  (python.withPackages (p: with p; [
                    numpy
                    matplotlib
                    pip
                  ]))
                  pkgs.vulkan-loader
                  pkgs.vulkan-headers
              ];
              shellHook = ''
                export PYTHONPATH=$PYTHONPATH:$(pwd)
                pip install wgpu
              '';
            }
        ) nixpkgs_configs
      );
    };
}
