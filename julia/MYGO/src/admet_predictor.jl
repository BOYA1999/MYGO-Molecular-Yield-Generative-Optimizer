"""
ADMET Property Predictor

This module provides ADMET (Absorption, Distribution, Metabolism,
Excretion, Toxicity) property prediction using rule-based and
machine learning approaches.

Supported predictions:
- Blood-Brain Barrier (BBB) permeability
- Metabolism prediction
- Toxicity prediction
"""

using Statistics
using LinearAlgebra
using Random
using Distributions

# Include required modules
include("molecule.jl")
include("descriptors.jl")
using .MoleculeModule
using .DescriptorModule

# Export prediction functions
export predict_bbb
export predict_metabolism
export predict_toxicity
export BBBResult
export MetabolismResult
export ToxicityResult

# ============================================================
# Result Structures
# ============================================================

"""
    BBBResult

Blood-Brain Barrier permeability prediction result.
"""
struct BBBResult
    penetration::String           # "Yes", "No", "Moderate"
    score::Float64               # Permeability score (0-1)
    brain_plasma_ratio::Float64 # Estimated brain-to-plasma ratio
    confidence::Float64          # Confidence of prediction
    factors::Vector{String}      # Contributing factors
    properties::Dict{String, Any} # Molecular properties used
end

"""
    MetabolismResult

Metabolism prediction result.
"""
struct MetabolismResult
    metabolic_stability::Float64  # 0-1 score
    predicted_pathways::Vector{String}
    cytochrome_interaction::String
    half_life_estimate::Float64   # in hours
    confidence::Float64
end

"""
    ToxicityResult

Toxicity prediction result.
"""
struct ToxicityResult
    toxicity_risk::String         # "Low", "Medium", "High"
    hERG_toxicity::String        # Cardiotoxicity
    hepatotoxicity::String       # Liver toxicity
    mutagenicity::String         # Genetic toxicity
    carcinogenicity::String      # Cancer risk
    confidence::Float64
end

# ============================================================
# BBB Permeability Prediction
# ============================================================

"""
    predict_bbb(mol::Molecule) -> BBBResult

Predict blood-brain barrier permeability.

This function uses a rule-based approach based on Lipinski's CNS
drug criteria and other known BBB penetration rules.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `BBBResult`: BBB prediction result

# Example
```julia
mol = parse_smiles("CCO")
result = predict_bbb(mol)
println(result.penetration)
```
"""
function predict_bbb(mol::Molecule)::BBBResult
    # Calculate molecular properties
    mw = get_molecular_weight(mol)
    logp = get_logp(mol)
    tpsa = get_tpsa(mol)
    hbd = get_num_hbd(mol)
    hba = get_num_hba(mol)
    rot_bonds = get_num_rotatable_bonds(mol)
    num_rings = get_num_rings(mol)
    
    # Initialize BBB score
    bbb_score = 0.0
    factors = String[]
    
    # Molecular weight factor (lower is better for BBB)
    # BBB+ compounds typically have MW < 450
    if mw < 400
        bbb_score += 0.25
        push!(factors, "Optimal MW for BBB penetration (<400)")
    elseif mw < 500
        bbb_score += 0.15
        push!(factors, "Moderate MW (400-500)")
    else
        bbb_score -= 0.25
        push!(factors, "High MW reduces BBB penetration (>$500)")
    end
    
    # LogP factor (optimal range 1-4 for CNS drugs)
    if 1.0 <= logp <= 3.5
        bbb_score += 0.25
        push!(factors, "Optimal LogP for BBB (1-3.5)")
    elseif 0.5 <= logp <= 5.0
        bbb_score += 0.15
        push!(factors, "Moderate LogP (0.5-5)")
    elseif logp < 0
        bbb_score -= 0.2
        push!(factors, "Very low LogP (too polar)")
    else
        bbb_score -= 0.15
        push!(factors, "High LogP may reduce BBB")
    end
    
    # TPSA factor (lower is better)
    # TPSA < 90 Å² is favorable for BBB
    if tpsa < 70
        bbb_score += 0.2
        push!(factors, "Low TPSA favors BBB (<70)")
    elseif tpsa < 90
        bbb_score += 0.1
        push!(factors, "Moderate TPSA (70-90)")
    else
        bbb_score -= 0.2
        push!(factors, "High TPSA reduces BBB (>$90)")
    end
    
    # HBD factor
    if hbd <= 2
        bbb_score += 0.15
        push!(factors, "Low HBD count (<=2)")
    elseif hbd <= 4
        bbb_score += 0.05
        push!(factors, "Moderate HBD (2-4)")
    else
        bbb_score -= 0.15
        push!(factors, "High HBD count reduces BBB")
    end
    
    # HBA factor
    if hba <= 5
        bbb_score += 0.1
        push!(factors, "Low HBA count (<=5)")
    elseif hba <= 8
        bbb_score += 0.05
        push!(factors, "Moderate HBA (5-8)")
    else
        push!(factors, "High HBA may reduce BBB")
    end
    
    # Rotatable bonds factor
    if rot_bonds <= 8
        bbb_score += 0.1
        push!(factors, "Low flexibility (<=8 rotatable bonds)")
    elseif rot_bonds > 15
        bbb_score -= 0.1
        push!(factors, "High flexibility may reduce BBB")
    end
    
    # Ring count factor
    if num_rings >= 2
        bbb_score += 0.1
        push!(factors, "Rigid structure (>=2 rings)")
    end
    
    # Normalize score to 0-1 range
    bbb_score = max(0.0, min(1.0, bbb_score))
    
    # Determine penetration category
    if bbb_score >= 0.7
        penetration = "Yes (High likelihood)"
        brain_plasma_ratio = 0.8 + (bbb_score - 0.7) * 2.0
    elseif bbb_score >= 0.4
        penetration = "Moderate"
        brain_plasma_ratio = 0.3 + (bbb_score - 0.4) * 1.5
    else
        penetration = "No (Low likelihood)"
        brain_plasma_ratio = bbb_score * 0.75
    end
    
    # Limit brain/plasma ratio
    brain_plasma_ratio = max(0.01, min(3.0, brain_plasma_ratio))
    
    # Calculate confidence based on how well molecule fits rules
    confidence = 0.5 + bbb_score * 0.3
    
    properties = Dict{String, Any}(
        "molecular_weight" => mw,
        "logp" => logp,
        "tpsa" => tpsa,
        "num_hbd" => hbd,
        "num_hba" => hba,
        "num_rotatable_bonds" => rot_bonds,
        "num_rings" => num_rings
    )
    
    return BBBResult(
        penetration,
        bbb_score,
        brain_plasma_ratio,
        confidence,
        factors,
        properties
    )
end

# ============================================================
# Metabolism Prediction
# ============================================================

"""
    predict_metabolism(mol::Molecule) -> MetabolismResult

Predict metabolic stability and metabolic pathways.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `MetabolismResult`: Metabolism prediction result
"""
function predict_metabolism(mol::Molecule)::MetabolismResult
    # Calculate properties
    mw = get_molecular_weight(mol)
    logp = get_logp(mol)
    tpsa = get_tpsa(mol)
    num_carbons = get_num_carbons(mol)
    num_rings = get_num_rings(mol)
    num_heteroatoms = get_num_heteroatoms(mol)
    
    # Metabolic stability score (0-1)
    stability = 0.5
    
    # Factors that increase metabolic stability
    if num_rings >= 2
        stability += 0.15
    end
    
    if num_carbons > 5
        stability += 0.1
    end
    
    if logp > 2
        stability += 0.1
    end
    
    # Factors that decrease metabolic stability
    if num_heteroatoms > 5
        stability -= 0.15
    end
    
    if mw < 300
        stability -= 0.15
    end
    
    # Normalize
    stability = max(0.0, min(1.0, stability))
    
    # Predict metabolic pathways
    pathways = String[]
    
    # Cytochrome P450 interaction
    cychrome = "Unknown"
    
    # Check for metabolically unstable groups
    has_phenol = count(a -> a == "O", mol.atoms) > 1
    has_amine = count(a -> a == "N", mol.atoms) > 0
    
    if has_phenol
        push!(pathways, "Phase I: Oxidation (CYP450)")
        cychrome = "High"
        stability -= 0.2
    end
    
    if has_amine
        push!(pathways, "Phase I: N-dealkylation")
        cychrome = "Moderate"
        stability -= 0.1
    end
    
    if isempty(pathways)
        push!(pathways, "Phase I: Minimal metabolism predicted")
        cychrome = "Low"
        stability += 0.1
    end
    
    # Phase II pathways
    if has_phenol || has_amine
        push!(pathways, "Phase II: Glucuronidation possible")
        push!(pathways, "Phase II: Sulfation possible")
    end
    
    # Estimate half-life (hours)
    half_life = 1.0 / stability * 2.0  # Simplified model
    half_life = max(0.1, min(24.0, half_life))
    
    confidence = 0.6 + stability * 0.2
    
    return MetabolismResult(
        stability,
        pathways,
        cychrome,
        half_life,
        confidence
    )
end

# ============================================================
# Toxicity Prediction
# ============================================================

"""
    predict_toxicity(mol::Molecule) -> ToxicityResult

Predict toxicity properties.

# Arguments
- `mol::Molecule`: Input molecule

# Returns
- `ToxicityResult`: Toxicity prediction result
"""
function predict_toxicity(mol::Molecule)::ToxicityResult
    # Calculate properties
    mw = get_molecular_weight(mol)
    logp = get_logp(mol)
    num_heavies = get_num_heavy_atoms(mol)
    num_heteroatoms = get_num_heteroatoms(mol)
    num_carbons = get_num_carbons(mol)
    num_nitrogens = get_num_nitrogens(mol)
    num_oxygens = get_num_oxygens(mol)
    num_halogens = get_num_halogens(mol)
    
    # hERG toxicity (cardiotoxicity)
    # Associated with certain structural features
    hERG_risk = "Low"
    
    # High logP and nitrogen-containing compounds may have hERG risk
    if logp > 4 && num_nitrogens > 1
        hERG_risk = "Medium"
    end
    
    if logp > 5 && num_nitrogens > 2
        hERG_risk = "High"
    end
    
    # Halogenated compounds may have cardiac effects
    if num_halogens >= 2
        if hERG_risk == "Low"
            hERG_risk = "Medium"
        end
    end
    
    # Hepatotoxicity (liver toxicity)
    hepatotox = "Low"
    
    if num_heavies > 30
        hepatotox = "Medium"
    end
    
    if num_heteroatoms > 8
        hepatotox = "Medium"
    end
    
    # Known hepatotoxic elements
    if num_nitrogens > 5
        hepatotox = "Medium"
    end
    
    # Mutagenicity
    mutagen = "Low"
    
    # Some fragments are associated with mutagenicity
    # Simplified rules
    if num_halogens > 2
        mutagen = "Medium"
    end
    
    if num_heavies > 40
        mutagen = "Medium"
    end
    
    # Carcinogenicity
    carcin = "Low"
    
    # Large PAHs may be carcinogenic
    if num_carbons > 15 && num_rings(mol) >= 3
        carcin = "Medium"
    end
    
    # Overall toxicity risk
    risk_score = 0.0
    
    if hERG_risk == "High"
        risk_score += 0.3
    elseif hERG_risk == "Medium"
        risk_score += 0.15
    end
    
    if hepatotox == "High"
        risk_score += 0.3
    elseif hepatotox == "Medium"
        risk_score += 0.15
    end
    
    if mutagen == "High"
        risk_score += 0.3
    elseif mutagen == "Medium"
        risk_score += 0.15
    end
    
    if carcin == "High"
        risk_score += 0.3
    elseif carcin == "Medium"
        risk_score += 0.15
    end
    
    if risk_score >= 0.6
        overall_risk = "High"
    elseif risk_score >= 0.3
        overall_risk = "Medium"
    else
        overall_risk = "Low"
    end
    
    confidence = 0.55 + (1.0 - risk_score) * 0.25
    
    return ToxicityResult(
        overall_risk,
        hERG_risk,
        hepatotox,
        mutagen,
        carcin,
        confidence
    )
end

# Helper function for toxicity prediction
function num_rings(mol::Molecule)::Int
    return get_num_rings(mol)
end
