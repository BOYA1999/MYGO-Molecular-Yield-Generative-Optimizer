"""
Molecular Descriptor Extractor

Extracts comprehensive molecular descriptors for ADMET prediction.
"""

import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdMD


class DescriptorExtractor:
    """
    Extracts molecular descriptors for machine learning models.
    
    Supports multiple types of descriptors:
    - 2D descriptors (molecular properties)
    - Fingerprints (Morgan, RDKit, MACCS)
    - Topological descriptors
    - Constitutional descriptors
    """
    
    def __init__(self, include_fingerprints: bool = True, fingerprint_radius: int = 2, n_bits: int = 2048):
        """
        Initialize descriptor extractor.
        
        Args:
            include_fingerprints: Whether to include fingerprint descriptors
            fingerprint_radius: Radius for Morgan fingerprint
            n_bits: Number of bits for fingerprints
        """
        self.include_fingerprints = include_fingerprints
        self.fingerprint_radius = fingerprint_radius
        self.n_bits = n_bits
    
    def extract_all(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """
        Extract all available descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with descriptor names as keys and arrays as values
        """
        descriptors = {}
        
        # Basic 2D descriptors
        descriptors.update(self.extract_2d_descriptors(mol))
        
        # Fingerprints
        if self.include_fingerprints:
            descriptors.update(self.extract_fingerprints(mol))
        
        # Topological descriptors
        descriptors.update(self.extract_topological_descriptors(mol))
        
        return descriptors
    
    def extract_2d_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract 2D molecular descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with descriptor names and values
        """
        descriptors = {}
        
        # Basic properties
        descriptors['molecular_weight'] = Descriptors.MolWt(mol)
        descriptors['logp'] = Crippen.MolLogP(mol)
        descriptors['tpsa'] = Descriptors.TPSA(mol)
        descriptors['num_atoms'] = mol.GetNumAtoms()
        descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        descriptors['num_bonds'] = mol.GetNumBonds()
        
        # Lipinski properties
        descriptors['num_hbd'] = Lipinski.NumHDonors(mol)
        descriptors['num_hba'] = Lipinski.NumHAcceptors(mol)
        descriptors['num_rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
        
        # Ring information
        descriptors['num_rings'] = rdMD.CalcNumRings(mol)
        descriptors['num_aromatic_rings'] = rdMD.CalcNumAromaticRings(mol)
        descriptors['num_saturated_rings'] = rdMD.CalcNumSaturatedRings(mol)
        descriptors['num_aliphatic_rings'] = rdMD.CalcNumAliphaticRings(mol)
        
        # Atom counts
        descriptors['num_carbons'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        descriptors['num_nitrogens'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
        descriptors['num_oxygens'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        descriptors['num_sulfurs'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'S')
        descriptors['num_halogens'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br', 'I'])
        
        # Additional descriptors
        descriptors['num_heteroatoms'] = rdMD.CalcNumHeteroatoms(mol)
        descriptors['num_radicals'] = Descriptors.NumRadicalElectrons(mol)
        descriptors['num_valence_electrons'] = Descriptors.NumValenceElectrons(mol)
        descriptors['fraction_csp3'] = rdMD.CalcFractionCsp3(mol)
        descriptors['num_amides'] = rdMD.CalcNumAmideBonds(mol)
        descriptors['num_aromatic_heterocycles'] = rdMD.CalcNumAromaticHeterocycles(mol)
        descriptors['num_saturated_heterocycles'] = rdMD.CalcNumSaturatedHeterocycles(mol)
        
        # Complexity
        descriptors['balaban_j'] = Descriptors.BalabanJ(mol)
        descriptors['bertz_ct'] = Descriptors.BertzCT(mol)
        
        return descriptors
    
    def extract_fingerprints(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """
        Extract molecular fingerprints.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with fingerprint names and bit vectors
        """
        fingerprints = {}
        
        # Morgan fingerprint (circular fingerprint)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.fingerprint_radius, nBits=self.n_bits
        )
        fingerprints['morgan'] = np.array(morgan_fp)
        
        # RDKit fingerprint
        rdkit_fp = Chem.RDKFingerprint(mol)
        # Convert to fixed-length vector
        rdkit_array = np.zeros(self.n_bits, dtype=np.int32)
        for bit in rdkit_fp.GetOnBits():
            if bit < self.n_bits:
                rdkit_array[bit] = 1
        fingerprints['rdkit'] = rdkit_array
        
        # MACCS keys (166 bits)
        try:
            maccs_fp = rdMD.GetMACCSKeysFingerprint(mol)
            fingerprints['maccs'] = np.array(maccs_fp)
        except:
            fingerprints['maccs'] = np.zeros(167, dtype=np.int32)  # 167 bits (0-166)
        
        return fingerprints
    
    def extract_topological_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract topological descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with topological descriptor names and values
        """
        descriptors = {}
        
        # Connectivity indices
        descriptors['chi0'] = Descriptors.Chi0(mol)
        descriptors['chi1'] = Descriptors.Chi1(mol)
        descriptors['chi0n'] = Descriptors.Chi0n(mol)
        descriptors['chi1n'] = Descriptors.Chi1n(mol)
        descriptors['chi2n'] = Descriptors.Chi2n(mol)
        descriptors['chi3n'] = Descriptors.Chi3n(mol)
        descriptors['chi4n'] = Descriptors.Chi4n(mol)
        
        # Kappa indices
        descriptors['kappa1'] = Descriptors.Kappa1(mol)
        descriptors['kappa2'] = Descriptors.Kappa2(mol)
        descriptors['kappa3'] = Descriptors.Kappa3(mol)
        
        # HallKierAlpha
        descriptors['hallkier_alpha'] = Descriptors.HallKierAlpha(mol)
        
        return descriptors
    
    def extract_feature_vector(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract a single feature vector combining all descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Concatenated feature vector as numpy array
        """
        all_descriptors = self.extract_all(mol)
        
        # Combine all descriptors into a single vector
        feature_list = []
        
        # Add 2D descriptors
        feature_list.extend(list(all_descriptors['molecular_weight'].values()) if isinstance(all_descriptors.get('molecular_weight'), dict) else [all_descriptors.get('molecular_weight', 0)])
        
        # Flatten all descriptors
        for key, value in all_descriptors.items():
            if isinstance(value, np.ndarray):
                feature_list.extend(value.tolist())
            elif isinstance(value, (int, float)):
                feature_list.append(value)
            elif isinstance(value, dict):
                feature_list.extend(value.values())
        
        return np.array(feature_list, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        # Create a dummy molecule to get feature names
        dummy_mol = Chem.MolFromSmiles('C')
        all_descriptors = self.extract_all(dummy_mol)
        
        feature_names = []
        for key, value in all_descriptors.items():
            if isinstance(value, np.ndarray):
                feature_names.extend([f"{key}_{i}" for i in range(len(value))])
            elif isinstance(value, (int, float)):
                feature_names.append(key)
            elif isinstance(value, dict):
                feature_names.extend([f"{key}_{k}" for k in value.keys()])
        
        return feature_names
