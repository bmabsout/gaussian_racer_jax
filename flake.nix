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
              python-with-packages = python.withPackages (p: with p; [
                numpy
                matplotlib
                moderngl
                glfw
                pip
              ]);
              
          in pkgs.mkShell {
                buildInputs = [
                    # pkgs.cudaPackages.cudatoolkit
                    python-with-packages
                    pkgs.vulkan-loader
                    pkgs.vulkan-headers
                ];
                shellHook = ''
                  python -m venv .venv --system-site-packages
                  source .venv/bin/activate
                  ln -s ${python-with-packages}/${python-with-packages.sitePackages}/* .venv/${python-with-packages.sitePackages}/
                  export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:${pkgs.glfw}/lib:$LD_LIBRARY_PATH
                  export PYTHONPATH=$(pwd):$PYTHONPATH
                  pip install wgpu
              '';
            }
        ) nixpkgs_configs
      );
    };
}
