"""
MYGO.jl - Molecular descriptors and ADMET prediction module in Julia

This module provides molecular descriptor calculation, fingerprint generation,
and ADMET property prediction for integration with the MYGO molecular generation project.

# Features
- Molecular descriptor calculation (2D, topological, constitutional)
- Fingerprint generation (Morgan, RDKit, MACCS)
- ADMET property prediction (BBB, metabolism, toxicity)
- Python integration via PyCall

# Example
```julia
using MYGO

# Parse SMILES
mol = parse_smiles("CCO")

# Calculate descriptors
descriptors = calculate_descriptors(mol)

# Calculate fingerprints
fp = calculate_fingerprint(mol, "morgan")

# Predict BBB permeability
result = predict_bbb(mol)
```
"""
module MYGO

# Import required Julia packages
using Statistics
using LinearAlgebra
using Random
using SparseArrays
using Printf
using Distributions

# Export public functions
export parse_smiles
export calculate_descriptors
export calculate_fingerprint
export predict_bbb
export predict_metabolism
export predict_toxicity
export get_molecular_weight
export get_logp
export get_tpsa
export get_num_hbd
export get_num_hba
export get_num_rotatable_bonds
export get_num_rings

# Include submodules
include("molecule.jl")
include("descriptors.jl")
include("fingerprints.jl")
include("admet_predictor.jl")

# Initialize module
function __init__()
    @info "MYGO.jl loaded - Molecular descriptors and ADMET prediction"
end

end # module
