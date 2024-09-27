{
	inputs.nixpkgs.url = "github:NixOS/nixpkgs";
	inputs.poetry2nix = {
		url = "github:nix-community/poetry2nix";
		inputs.nixpkgs.follows = "nixpkgs";
	};
	inputs.flake-utils.url = "github:numtide/flake-utils";

	outputs = { self, nixpkgs, poetry2nix, flake-utils }: 
	flake-utils.lib.eachDefaultSystem (system:
		let
			pkgs = nixpkgs.legacyPackages.${system};
			inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv;
			pyEnv = mkPoetryEnv {
				projectDir = ./.;
				preferWheels = true;
			};
		in {
			devShells.default = pkgs.mkShell {
				nativeBuildInputs = [ pyEnv pkgs.poetry ];
			};
		}
	);
}
