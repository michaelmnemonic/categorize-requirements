{
  description = "Categorize requirements";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    setfit = {
      url = "github:michaelmnemonic/setfit-flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    setfit,
  }: let
    # Define 'forAllSystems' for properties that shall be build for x86_64 *and* aarch64
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
    pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
  in {
    devShells = forAllSystems (system: {
      default = pkgs.${system}.mkShell {
        buildInputs = with pkgs.${system}; [
          # basics
          gitMinimal

          #python
          python313
          ruff
          python313Packages.datasets
          python313Packages.accelerate
          python313Packages.pytest
          setfit.packages.${system}.setfit

          # nix
          nil
          alejandra
        ];
      };
    });
  };
}
