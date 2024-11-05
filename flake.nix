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
          let 
            pkgs = import nixpkgs { 
              inherit system config;
            };
            python = pkgs.python311.override {
              packageOverrides = self: super: with self; {
                wgpu = buildPythonPackage rec {
                  pname = "wgpu";
                  version = "0.19.0";
                  format = "wheel";

                  src = fetchPypi {
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

                  propagatedBuildInputs = with pkgs; [
                    cffi
                    numpy
                    typing-extensions
                    glfw
                    vulkan-loader
                  ];

                  buildInputs = with pkgs; [
                    vulkan-headers
                    freetype
                    fontconfig
                    glfw
                  ] ++ (lib.optionals stdenv.isDarwin (with darwin.apple_sdk.frameworks; [
                    CoreServices
                    QuartzCore
                    AppKit
                    Metal
                    MetalKit
                  ]));

                  doCheck = false;
                };
              };
            };
            
          in pkgs.mkShell {
              buildInputs = [
                (python.withPackages (p: with p; [
                  numpy
                  matplotlib
                  glfw
                  pip
                  wgpu
                ]))
                pkgs.vulkan-loader
                pkgs.vulkan-headers
                pkgs.freetype
                pkgs.fontconfig
              ] ++ (pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs.darwin.apple_sdk.frameworks; [
                Cocoa
                Metal
                MetalKit
                QuartzCore
                CoreServices
                AppKit
              ]));
              shellHook = ''
                export PYTHONPATH=$(pwd):$PYTHONPATH
              '';
            }
        ) nixpkgs_configs
      );
    };
}
