{
  description = "Categorize requirements";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    # Define 'forAllSystems' for properties that shall be build for x86_64 *and* aarch64
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
    pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
  in {
    packages = forAllSystems (system: {
      setfit = pkgs.${system}.python3Packages.buildPythonPackage rec {
        pname = "setfit";
        version = "1.1.3";
        pyproject = true;

        src = pkgs.${system}.fetchPypi {
          inherit pname;
          inherit version;
          hash = "sha256-21SntuhhGDQydcxww5dJ9x4QqiJv71EFQ7zXyLALnxk==";
        };

        dependencies = with pkgs.${system}.python313Packages; [
          setuptools
          datasets
          sentence-transformers
          evaluate
          huggingface-hub
          scikit-learn
        ];
      };
    });

    devShells = forAllSystems (system: {
      default = pkgs.${system}.mkShell {
        buildInputs = with pkgs.${system}; [
          # basics
          gitMinimal

          #python
          python313
          ruff
          dataset
          self.packages.${system}.setfit

          # nix
          nil
          alejandra
        ];
      };
    });
  };
}