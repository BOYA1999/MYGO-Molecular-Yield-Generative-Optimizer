"""
Physical Interaction Encoder for Molecular Generation

This module encodes explicit physical interactions (electrostatic, van der Waals, 
hydrogen bonding) to enhance the model's ability to capture physical forces,
especially important for non-standard pockets and induced fit scenarios.
"""

import torch
import torch.nn as nn
from models.common import MLP


class PhysicalInteractionEncoder(nn.Module):
    """
    Encodes explicit physical interactions between atoms:
    - Electrostatic interactions (Coulomb potential)
    - Van der Waals interactions (Lennard-Jones potential)
    - Hydrogen bonding (geometric and chemical constraints)
    
    This addresses the limitation of relying solely on geometric distances
    by incorporating physics-based interaction terms.
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=64, use_learnable_params=True):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for interaction networks
            use_learnable_params: If True, use learnable atomic properties; 
                                 if False, use fixed physical constants
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.use_learnable_params = use_learnable_params
        
        # Atomic property embeddings
        # These can be learned or initialized with physical constants
        max_atom_types = 100  # Support up to 100 atom types
        if use_learnable_params:
            # Learnable embeddings for atomic properties
            self.electronegativity_emb = nn.Embedding(max_atom_types, 1)
            self.vdw_radius_emb = nn.Embedding(max_atom_types, 1)
            self.partial_charge_emb = nn.Embedding(max_atom_types, 1)
            # Initialize with reasonable values
            self._init_physical_properties()
        else:
            # Fixed physical constants (can be loaded from database)
            self.register_buffer('electronegativity', self._get_default_electronegativity())
            self.register_buffer('vdw_radius', self._get_default_vdw_radius())
            self.register_buffer('partial_charge', torch.zeros(max_atom_types, 1))
        
        # Interaction computation networks
        # Electrostatic: E_elec ∝ q_i * q_j / r
        self.electrostatic_net = MLP(edge_dim + 3, hidden_dim, hidden_dim, num_layer=2)
        
        # Van der Waals: Simplified LJ potential
        # E_vdw ∝ (r_min/r)^12 - (r_min/r)^6
        self.vdw_net = MLP(edge_dim + 3, hidden_dim, hidden_dim, num_layer=2)
        
        # Hydrogen bonding: geometric and chemical constraints
        self.hbond_net = MLP(edge_dim + 5, hidden_dim, hidden_dim, num_layer=2)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def _init_physical_properties(self):
        """Initialize embeddings with physical constants"""
        # Electronegativity (Pauling scale, approximate)
        elec_values = torch.tensor([
            0.0, 2.20, 0.0, 0.0, 0.0, 0.0, 2.55,  # H, He, Li, Be, B, C, N
            3.44, 3.98, 0.0, 0.0, 0.0, 0.0, 0.0,  # O, F, Ne, Na, Mg, Al, Si
            2.19, 2.58, 0.0, 0.0, 0.0, 0.0, 0.0,  # P, S, Cl, Ar, K, Ca, Sc
        ] + [0.0] * 73)  # Pad to 100
        self.electronegativity_emb.weight.data[:, 0] = elec_values
        
        # Van der Waals radius (Angstrom, approximate)
        vdw_values = torch.tensor([
            0.0, 1.40, 0.0, 0.0, 0.0, 0.0, 1.70,  # H, He, Li, Be, B, C, N
            1.52, 1.47, 0.0, 0.0, 0.0, 0.0, 0.0,  # O, F, Ne, Na, Mg, Al, Si
            1.80, 1.80, 1.75, 0.0, 0.0, 0.0, 0.0,  # P, S, Cl, Ar, K, Ca, Sc
        ] + [1.80] * 73)  # Default to 1.8 for others
        self.vdw_radius_emb.weight.data[:, 0] = vdw_values
        
    def _get_default_electronegativity(self):
        """Get default electronegativity values"""
        return torch.tensor([
            0.0, 2.20, 0.0, 0.0, 0.0, 0.0, 2.55,
            3.44, 3.98, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.19, 2.58, 0.0, 0.0, 0.0, 0.0, 0.0,
        ] + [0.0] * 79).unsqueeze(-1)
    
    def _get_default_vdw_radius(self):
        """Get default van der Waals radius values"""
        return torch.tensor([
            0.0, 1.40, 0.0, 0.0, 0.0, 0.0, 1.70,
            1.52, 1.47, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.80, 1.80, 1.75, 0.0, 0.0, 0.0, 0.0,
        ] + [1.80] * 79).unsqueeze(-1)
    
    def forward(self, h_node, edge_index, distance, atom_types, h_edge=None):
        """
        Compute physical interaction features for edges.
        
        Args:
            h_node: (N, node_dim) Node features
            edge_index: (2, E) Edge indices [source, target]
            distance: (E,) Interatomic distances
            atom_types: (N,) Atom type indices
            h_edge: (E, edge_dim) Existing edge features (optional)
            
        Returns:
            physical_feat: (E, hidden_dim) Physical interaction features
        """
        row, col = edge_index  # source and target nodes
        E = edge_index.size(1)
        device = distance.device
        
        # Get atomic properties
        if self.use_learnable_params:
            elec_i = self.electronegativity_emb(atom_types[row])  # (E, 1)
            elec_j = self.electronegativity_emb(atom_types[col])
            vdw_i = self.vdw_radius_emb(atom_types[row])
            vdw_j = self.vdw_radius_emb(atom_types[col])
            charge_i = self.partial_charge_emb(atom_types[row])
            charge_j = self.partial_charge_emb(atom_types[col])
        else:
            elec_i = self.electronegativity[atom_types[row]]
            elec_j = self.electronegativity[atom_types[col]]
            vdw_i = self.vdw_radius[atom_types[row]]
            vdw_j = self.vdw_radius[atom_types[col]]
            charge_i = self.partial_charge[atom_types[row]]
            charge_j = self.partial_charge[atom_types[col]]
        
        # 1. Electrostatic interaction: E_elec ∝ q_i * q_j / r
        # Simplified: use electronegativity difference as proxy for charge
        elec_diff = torch.abs(elec_i - elec_j)
        elec_product = elec_i * elec_j
        # Avoid division by zero
        elec_interaction = elec_product / (distance.unsqueeze(-1) + 1e-6)
        
        # Prepare electrostatic input
        if h_edge is not None:
            elec_input = torch.cat([
                h_edge,
                elec_interaction,
                elec_diff,
                distance.unsqueeze(-1)
            ], dim=-1)
        else:
            elec_input = torch.cat([
                elec_interaction,
                elec_diff,
                distance.unsqueeze(-1)
            ], dim=-1)
        
        elec_feat = self.electrostatic_net(elec_input)
        
        # 2. Van der Waals interaction: Simplified Lennard-Jones potential
        # E_vdw ∝ (r_min/r)^12 - (r_min/r)^6
        vdw_sum = vdw_i + vdw_j  # Sum of vdW radii
        vdw_ratio = vdw_sum / (distance.unsqueeze(-1) + 1e-6)
        # Simplified LJ: use power terms
        vdw_attractive = vdw_ratio ** 3  # (r_min/r)^6 approximation
        vdw_repulsive = vdw_ratio ** 6   # (r_min/r)^12 approximation
        vdw_interaction = vdw_repulsive - vdw_attractive
        
        # Prepare vdW input
        if h_edge is not None:
            vdw_input = torch.cat([
                h_edge,
                vdw_interaction,
                vdw_sum,
                distance.unsqueeze(-1)
            ], dim=-1)
        else:
            vdw_input = torch.cat([
                vdw_interaction,
                vdw_sum,
                distance.unsqueeze(-1)
            ], dim=-1)
        
        vdw_feat = self.vdw_net(vdw_input)
        
        # 3. Hydrogen bonding: detect potential H-bonds
        # Criteria: (1) Donor: N, O, F with H; (2) Acceptor: N, O, F; (3) Distance < 3.5 Å
        # Atom types: 1=H, 7=N, 8=O, 9=F
        is_donor = (atom_types[row] == 7) | (atom_types[row] == 8) | (atom_types[row] == 9)
        is_acceptor = (atom_types[col] == 7) | (atom_types[col] == 8) | (atom_types[col] == 9)
        hbond_candidate = is_donor & is_acceptor
        
        # H-bond strength depends on distance and angle (simplified)
        hbond_distance_factor = 1.0 / (distance.unsqueeze(-1) + 1e-6)
        hbond_strength = hbond_candidate.float().unsqueeze(-1) * hbond_distance_factor
        
        # Additional H-bond features
        hbond_elec_sum = elec_i + elec_j  # Higher electronegativity = stronger H-bond
        
        # Prepare H-bond input
        if h_edge is not None:
            hbond_input = torch.cat([
                h_edge,
                hbond_strength,
                hbond_elec_sum,
                elec_diff,
                distance.unsqueeze(-1),
                hbond_candidate.float().unsqueeze(-1)
            ], dim=-1)
        else:
            hbond_input = torch.cat([
                hbond_strength,
                hbond_elec_sum,
                elec_diff,
                distance.unsqueeze(-1),
                hbond_candidate.float().unsqueeze(-1)
            ], dim=-1)
        
        hbond_feat = self.hbond_net(hbond_input)
        
        # Combine all interaction features
        combined_feat = torch.cat([elec_feat, vdw_feat, hbond_feat], dim=-1)
        physical_feat = self.output_proj(combined_feat)
        
        return physical_feat


def get_atom_type_from_node_feature(node_feature, node_type_embedder):
    """
    Extract atom type from node features.
    This is a helper function to get atom types when they're not directly available.
    
    Args:
        node_feature: (N, node_dim) Node features
        node_type_embedder: Embedding layer for node types
        
    Returns:
        atom_types: (N,) Atom type indices (approximate)
    """
    # This is a simplified approach - in practice, atom types should be passed directly
    # For now, we use a heuristic based on feature similarity
    with torch.no_grad():
        # Get all possible atom type embeddings
        num_types = node_type_embedder.num_embeddings
        all_embeddings = node_type_embedder.weight  # (num_types, emb_dim)
        
        # Find closest match (this is approximate)
        # Extract the embedding part from node features
        emb_dim = all_embeddings.size(1)
        node_emb = node_feature[:, :emb_dim]
        
        # Compute similarity
        similarity = torch.matmul(node_emb, all_embeddings.t())  # (N, num_types)
        atom_types = similarity.argmax(dim=-1)
    
    return atom_types

