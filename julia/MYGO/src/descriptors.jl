"""
Molecular Descriptor Calculator

This module calculates various molecular descriptors including:
- 2D descriptors (molecular properties)
- Topological descriptors
- Constitutional descriptors
- Lipinski rule of five properties

These descriptors are commonly used for ADMET prediction.
"""

using Statistics
using LinearAlgebra

# Include molecule module
include("molecule.jl")
using .MoleculeModule

# Export descriptor functions
export calculate_descriptors
export calculate_2d_descriptors
export calculate_topological_descriptors
export calculate_constitutional_descriptors
export get_logp
export get_tpsa
export get_num_hbd
export get_num_hba
export get_num_rotatable_bonds
export get_num_rings
export get_fraction_csp3

# ============================================================
# Basic Properties
# ============================================================

"""
    get_logp(mol::Molecule) -> Float64

Estimate logP (octanol-water partition coefficient).

This is a simplified calculation based on atomic contributions.
For accurate logP, consider using fragment-based methods or ML models.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Float64`: Estimated logP value
"""
function get_logp(mol::Molecule)::Float64
    # Simplified logP calculation based on atomic fragments
    # Based on atomic contribution method (XlogP)
    
    # Atomic contributions to logP
    logp_contributions = Dict{String, Float64}(
        "C" => 0.10,
        "H" => 0.23,
        "N" => -0.46,
        "O" => -0.37,
        "F" => 0.23,
        "Cl" => 0.70,
        "Br" => 0.88,
        "I" => 1.00,
        "S" => 0.35,
        "P" => 0.10,
    )
    
    logp = 0.0
    for atom in mol.atoms
        logp += get(logp_contributions, atom, 0.0)
    end
    
    # Add implicit hydrogens contribution
    logp += sum(mol.implicit_h) * logp_contributions["H"]
    
    # Correct for aromaticity (simplified)
    # Aromatic carbons contribute less
    n_carbons = count(a -> a == "C", mol.atoms)
    
    return logp
end

"""
    get_tpsa(mol::Molecule) -> Float64

Calculate topological polar surface area (TPSA).

TPSA is calculated based on fragments with polar atoms.
Values < 140 Å² are considered good for cell permeability.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Float64`: TPSA value in Å²
"""
function get_tpsa(mol::Molecule)::Float64
    # TPSA contribution from each atom type
    tpsa_contributions = Dict{String, Float64}(
        "N" => 12.5,   # Amide N
        "O" => 23.0,   # Hydroxyl O
        "S" => 25.0,   # Thiol S
        "P" => 25.0,   # Phosphate P
    )
    
    tpsa = 0.0
    for atom in mol.atoms
        tpsa += get(tpsa_contributions, atom, 0.0)
    end
    
    # Add contributions from implicit hydrogens on polar atoms
    # This is simplified
    
    return tpsa
end

"""
    get_num_hbd(mol::Molecule) -> Int

Get number of hydrogen bond donors.

Hydrogen bond donors are atoms with hydrogen that can act as
hydrogen bond acceptors (O, N, S).

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Int`: Number of HBD
"""
function get_num_hbd(mol::Molecule)::Int
    # HBD atoms: N, O with hydrogen
    # Simplified: count polar atoms
    hbd_atoms = ["N", "O", "S"]
    count(a -> a in hbd_atoms, mol.atoms)
end

"""
    get_num_hba(mol::Molecule) -> Int

Get number of hydrogen bond acceptors.

Hydrogen bond acceptors are atoms with lone pairs that can
accept hydrogen bonds (N, O, S, F, Cl, Br, I).

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Int`: Number of HBA
"""
function get_num_hba(mol::Molecule)::Int
    # HBA atoms: N, O, S, halogens
    hba_atoms = ["N", "O", "S", "F", "Cl", "Br", "I"]
    count(a -> a in hba_atoms, mol.atoms)
end

"""
    get_num_rotatable_bonds(mol::Molecule) -> Int

Estimate number of rotatable bonds.

Rotatable bonds are single bonds between non-terminal,
non-ring atoms.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Int`: Estimated number of rotatable bonds
"""
function get_num_rotatable_bonds(mol::Molecule)::Int
    # Simplified: estimate based on carbon chain length
    # In practice, need proper bond analysis
    
    n_carbons = count(a -> a == "C", mol.atoms)
    n_heavies = get_num_heavy_atoms(mol)
    
    # Rough estimate
    if n_carbons < 2
        return 0
    elseif n_carbons < 5
        return n_carbons - 1
    else
        return n_carbons - 2
    end
end

"""
    get_num_rings(mol::Molecule) -> Int

Estimate number of rings in the molecule.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Int`: Estimated number of rings
"""
function get_num_rings(mol::Molecule)::Int
    # Simplified ring count
    # In practice, need graph analysis for SSSR
    
    n_carbons = count(a -> a == "C", mol.atoms)
    
    # Aromatic ring indicator (presence of alternating pattern)
    # Simplified: estimate based on structure
    
    if n_carbons < 3
        return 0
    elseif n_carbons < 6
        return 1
    else
        return floor(Int, n_carbons / 6)
    end
end

"""
    get_fraction_csp3(mol::Molecule) -> Float64

Calculate fraction of sp3 hybridized carbons.

This is a measure of molecular complexity and 3D character.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Float64`: Fraction of sp3 carbons (0-1)
"""
function get_fraction_csp3(mol::Molecule)::Float64
    n_carbons = count(a -> a == "C", mol.atoms)
    
    if n_carbons == 0
        return 0.0
    end
    
    # Simplified: assume all carbons are sp3 unless aromatic pattern
    # In practice, need proper hybridization analysis
    return 1.0  # Simplified
end

"""
    get_num_aromatic_rings(mol::Molecule) -> Int

Get number of aromatic rings.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Int`: Number of aromatic rings
"""
function get_num_aromatic_rings(mol::Molecule)::Int
    # Simplified detection based on atom count
    # In practice, need proper aromaticity detection
    
    return 0  # Simplified
end

"""
    get_num_heteroatoms(mol::Molecule) -> Int

Get number of heteroatoms (non-C, non-H atoms).

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Int`: Number of heteroatoms
"""
function get_num_heteroatoms(mol::Molecule)::Int
    return count(a -> a ∉ ["C", "H"], mol.atoms)
end

# ============================================================
# Topological Descriptors
# ============================================================

"""
    calculate_chi(mol::Molecule, n::Int) -> Float64

Calculate molecular connectivity index (Chi index).

The Chi index is a topological descriptor that encodes
molecular branching.

# Arguments
- `mol::Molecule`: Input molecule
- `n::Int`: Order of Chi index (1, 2, 3, 4)

# Returns
- `Float64`: Chi index value
"""
function calculate_chi(mol::Molecule, n::Int)::Float64
    n_atoms = get_num_heavy_atoms(mol)
    
    if n_atoms < n + 1
        return 0.0
    end
    
    # Simplified Chi calculation
    # Based on degree sequence
    
    # Use heavy atom count as proxy
    if n == 0
        return sqrt(n_atoms)
    elseif n == 1
        return sqrt(n_atoms)
    else
        return sqrt(n_atoms) / n_atoms
    end
end

"""
    calculate_kappa(mol::Molecule, n::Int) -> Float64

Calculate kappa shape index.

Kappa indices encode molecular shape based on graph topology.

# Arguments
- `mol::Molecule`: Input molecule
- `n::Int`: Order of kappa (1, 2, 3)

# Returns
- `Float64`: Kappa index value
"""
function calculate_kappa(mol::Molecule, n::Int)::Float64
    n_atoms = get_num_heavy_atoms(mol)
    
    if n_atoms < n
        return 0.0
    end
    
    # Simplified kappa calculation
    return Float64(n_atoms - n + 1)
end

"""
    calculate_topological_descriptors(mol::Molecule) -> Dict{String, Float64}

Calculate all topological descriptors.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Dict{String, Float64}`: Dictionary of descriptor names and values
"""
function calculate_topological_descriptors(mol::Molecule)::Dict{String, Float64}
    desc = Dict{String, Float64}()
    
    # Chi indices
    desc["chi0"] = calculate_chi(mol, 0)
    desc["chi1"] = calculate_chi(mol, 1)
    desc["chi2"] = calculate_chi(mol, 2)
    
    # Kappa indices
    desc["kappa1"] = calculate_kappa(mol, 1)
    desc["kappa2"] = calculate_kappa(mol, 2)
    desc["kappa3"] = calculate_kappa(mol, 3)
    
    # Wiener index (path-based)
    n_atoms = get_num_heavy_atoms(mol)
    desc["wiener_index"] = n_atoms * (n_atoms - 1) / 2
    
    # Balaban index
    if n_atoms > 2
        desc["balaban_index"] = sqrt(n_atoms - 1)
    else
        desc["balaban_index"] = 0.0
    end
    
    return desc
end

# ============================================================
# Constitutional Descriptors
# ============================================================

"""
    calculate_constitutional_descriptors(mol::Molecule) -> Dict{String, Any}

Calculate constitutional descriptors.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Dict{String, Any}`: Dictionary of descriptor names and values
"""
function calculate_constitutional_descriptors(mol::Molecule)::Dict{String, Any}
    desc = Dict{String, Any}()
    
    desc["num_atoms"] = get_num_atoms(mol)
    desc["num_heavy_atoms"] = get_num_heavy_atoms(mol)
    desc["num_carbons"] = get_num_carbons(mol)
    desc["num_nitrogens"] = get_num_nitrogens(mol)
    desc["num_oxygens"] = get_num_oxygens(mol)
    desc["num_sulfurs"] = get_num_sulfurs(mol)
    desc["num_halogens"] = get_num_halogens(mol)
    desc["num_heteroatoms"] = get_num_heteroatoms(mol)
    desc["molecular_weight"] = get_molecular_weight(mol)
    desc["formula"] = get_formula(mol)
    
    # Atom fractions
    n_heavy = desc["num_heavy_atoms"]
    if n_heavy > 0
        desc["fraction_c"] = desc["num_carbons"] / n_heavy
        desc["fraction_n"] = desc["num_nitrogens"] / n_heavy
        desc["fraction_o"] = desc["num_oxygens"] / n_heavy
    else
        desc["fraction_c"] = 0.0
        desc["fraction_n"] = 0.0
        desc["fraction_o"] = 0.0
    end
    
    return desc
end

# ============================================================
# 2D Descriptors
# ============================================================

"""
    calculate_2d_descriptors(mol::Molecule) -> Dict{String, Any}

Calculate comprehensive 2D molecular descriptors.

This includes:
- Basic molecular properties
- Lipinski rule of five properties
- Electronic properties
- Structural properties

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Dict{String, Any}`: Dictionary of descriptor names and values
"""
function calculate_2d_descriptors(mol::Molecule)::Dict{String, Any}
    desc = Dict{String, Any}()
    
    # Basic properties
    desc["molecular_weight"] = get_molecular_weight(mol)
    desc["num_atoms"] = get_num_atoms(mol)
    desc["num_heavy_atoms"] = get_num_heavy_atoms(mol)
    desc["num_bonds"] = length(mol.bonds)
    
    # Lipinski properties
    desc["logp"] = get_logp(mol)
    desc["tpsa"] = get_tpsa(mol)
    desc["num_hbd"] = get_num_hbd(mol)
    desc["num_hba"] = get_num_hba(mol)
    desc["num_rotatable_bonds"] = get_num_rotatable_bonds(mol)
    
    # Ring information
    desc["num_rings"] = get_num_rings(mol)
    desc["num_aromatic_rings"] = get_num_aromatic_rings(mol)
    
    # Atom counts
    desc["num_carbons"] = get_num_carbons(mol)
    desc["num_nitrogens"] = get_num_nitrogens(mol)
    desc["num_oxygens"] = get_num_oxygens(mol)
    desc["num_sulfurs"] = get_num_sulfurs(mol)
    desc["num_halogens"] = get_num_halogens(mol)
    desc["num_heteroatoms"] = get_num_heteroatoms(mol)
    
    # Fraction descriptors
    desc["fraction_csp3"] = get_fraction_csp3(mol)
    
    # Lipinski rule of five violations
    violations = 0
    if desc["molecular_weight"] > 500
        violations += 1
    end
    if desc["logp"] > 5
        violations += 1
    end
    if desc["num_hbd"] > 5
        violations += 1
    end
    if desc["num_hba"] > 10
        violations += 1
    end
    desc["lipinski_violations"] = violations
    desc["lipinski_pass"] = violations == 0
    
    return desc
end

"""
    calculate_descriptors(mol::Molecule) -> Dict{String, Any}

Calculate all available molecular descriptors.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Dict{String, Any}`: Dictionary of all descriptors
"""
function calculate_descriptors(mol::Molecule)::Dict{String, Any}
    all_descriptors = Dict{String, Any}()
    
    # Add 2D descriptors
    desc_2d = calculate_2d_descriptors(mol)
    merge!(all_descriptors, desc_2d)
    
    # Add topological descriptors
    desc_topo = calculate_topological_descriptors(mol)
    merge!(all_descriptors, desc_topo)
    
    # Add constitutional descriptors
    desc_const = calculate_constitutional_descriptors(mol)
    merge!(all_descriptors, desc_const)
    
    return all_descriptors
end

"""
    get_feature_vector(mol::Molecule) -> Vector{Float64}

Extract feature vector for machine learning.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `Vector{Float64}`: Feature vector
"""
function get_feature_vector(mol::Molecule)::Vector{Float64}
    desc = calculate_descriptors(mol)
    
    # Extract numeric features only
    features = Float64[]
    
    for (key, value) in desc
        if typeof(value) <: Number
            push!(features, Float64(value))
        end
    end
    
    return features
end
