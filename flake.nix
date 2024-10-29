{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nixgl.url = "github:kenranunderscore/nixGL";
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
      # Define supported systems
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      # Helper function to generate attributes for all supported systems
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      # enter this python environment by executing `nix shell .#with_cuda`
      devShells = forAllSystems (system:
        nixpkgs.lib.attrsets.mapAttrs (name: config:
          let pkgs = import nixpkgs { 
                # overlays=[nixgl.overlay]; 
                inherit system config;
              };
              python = pkgs.python311.override {
                # packageOverrides = import ./nix/python-overrides.nix;
              };
          in pkgs.mkShell {
              buildInputs = [
                  # pkgs.nixgl.nixGLIntel
                  pkgs.cudaPackages.cudatoolkit
                  (python.withPackages (p: with p; [
                    jax
                    jaxlib
                    matplotlib
                    pygame
                  ]))
              ];
              shellHook = ''
                export PYTHONPATH=$PYTHONPATH:$(pwd) # to allow importing local packages as editable
                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib
                # export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}
              '';
            }
        ) nixpkgs_configs
      );
    };
}
