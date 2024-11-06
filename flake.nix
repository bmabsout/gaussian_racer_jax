{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python311.override {
          packageOverrides = self: super: {
            wgpu = self.buildPythonPackage rec {
              pname = "wgpu";
              version = "0.19.0";
              format = "wheel";

              src = self.fetchPypi {
                inherit pname version;
                format = "wheel";
                dist = "py3";
                python = "py3";
                abi = "none";
                platform = {
                  x86_64-linux = "manylinux_2_28_x86_64";
                  aarch64-linux = "manylinux_2_28_aarch64";
                  x86_64-darwin = "macosx_10_9_x86_64";
                  aarch64-darwin = "macosx_11_0_arm64";
                }.${system};
                hash = {
                  x86_64-linux = "sha256-TtVeNyQq5JghPdoUrDgdM7gnAnFDRuIKLDI8xCe7h64=";
                  aarch64-linux = "sha256-syxfUZbZqloO7nn9etaCNyXpQv5vTiRhmhlGh7WVkok=";
                  x86_64-darwin = "sha256-0Xq1xjUJc4Zh86LRQOLGS7n61BYpi/0ovIUXzSKzwpc=";
                  aarch64-darwin = "sha256-CNXF+deGjPAWteG9FTFtk+HzBkM7X15pqeYqa4YrJEQ=";
                }.${system};
              };

              propagatedBuildInputs = with self; [
                cffi
                numpy
                typing-extensions
                glfw
              ] ++ [(if pkgs.stdenv.isDarwin then rubicon-objc else null)];

              doCheck = false;
            };
          };
        };

        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          matplotlib
          glfw
          pip
          wgpu
        ]);

      in {
        apps.default = {
          type = "app";
          program = toString (pkgs.writeShellScript "gaussian-racer" ''
            export PYTHONPATH=${./.}:$PYTHONPATH
            ${if !pkgs.stdenv.isDarwin then "export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH" else ""}
            ${pythonEnv}/bin/python ${./src/gaussian_game.py}
          '');
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
          shellHook = ''
            export PYTHONPATH=$(pwd):$PYTHONPATH
            ${if !pkgs.stdenv.isDarwin then "export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH" else ""}
          '';
        };
      }
    );
}
