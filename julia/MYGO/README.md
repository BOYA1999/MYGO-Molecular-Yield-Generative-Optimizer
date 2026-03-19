# MYGO.jl

Molecular descriptors and ADMET prediction module in Julia for integration with the MYGO project.

## Installation

```julia
using Pkg
Pkg.add("MYGO")
```

Or from the package directory:

```julia
using Pkg
Pkg.add(".")
```

## Dependencies

- Julia 1.8+
- Grapholecules - Molecular graph operations
- MolecularDescriptor - Molecular descriptors
- RandomForest - ML models

## Usage

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

## Python Integration

This package can be used with Python via PyCall.jl:

```python
import julia
from julia import MYGO
```

Or use the provided Python wrapper:

```python
from mygo_julia import MolecularDescriptorJulia

descriptor = MolecularDescriptorJulia()
features = descriptor.extract("CCO")
```
