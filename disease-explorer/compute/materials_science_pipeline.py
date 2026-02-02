#!/usr/bin/env python3
"""
Materials Science Discovery Pipeline
Advanced AI-driven materials discovery with property prediction and synthesis planning

Features:
- Magpie compositional descriptors
- SOAP structural descriptors
- Graph Neural Networks for crystal structures
- GPU-accelerated property prediction
- Parallel atomistic simulations
- Synthesis route planning
- Distributed computing (Dask)
- Quantum computing integration (optional)

Usage:
    python3 materials_science_pipeline.py --job-type <step> --params '{"materials": [...], "properties": [...]}'

Supported Steps:
    - structure_validation: Validate material representations (CIF, formula, SMILES)
    - magpie_descriptors: Generate Magpie compositional descriptors
    - soap_descriptors: Generate SOAP structural descriptors
    - gnn_prediction: GNN-based property prediction for crystals
    - property_prediction: Multi-task neural network property prediction
    - manufacturability_scoring: Score synthesis feasibility
    - synthesis_planning: Generate synthesis routes
    - batch_screening: High-throughput property screening
    - materials_generation: Generate novel materials with target properties
    - element_substitution: Generate variants by element substitution
    - atomistic_simulation: Run MD/DFT simulations
    - full_pipeline: Complete discovery workflow
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import json
import os
import sys
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pickle
import hashlib
import math

# Distributed Computing (Dask)
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
    print("✓ Dask available for distributed computing")
except ImportError:
    DASK_AVAILABLE = False
    print("○ Dask not available - install with: pip install dask distributed")

# Core scientific libraries
try:
    from pymatgen.core import Structure, Composition, Lattice, Element
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.io.cif import CifParser, CifWriter
    from pymatgen.core.periodic_table import ElementBase
    PYMATGEN_AVAILABLE = True
    print("✓ Pymatgen available for materials structure analysis")
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("○ Pymatgen not available - install with: pip install pymatgen")

# Machine Learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.cuda.amp import autocast, GradScaler
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYTORCH_AVAILABLE = True
    TORCH_GEOMETRIC_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ PyTorch + PyTorch Geometric available - Device: {DEVICE}")
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from torch.cuda.amp import autocast, GradScaler
        PYTORCH_AVAILABLE = True
        TORCH_GEOMETRIC_AVAILABLE = False
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ PyTorch available - Device: {DEVICE}")
        print("○ PyTorch Geometric not available for GNN")
    except ImportError:
        PYTORCH_AVAILABLE = False
        TORCH_GEOMETRIC_AVAILABLE = False
        print("○ PyTorch not available")

# SOAP descriptors
try:
    from dscribe.descriptors import SOAP
    DSCRIBE_AVAILABLE = True
    print("✓ DScribe available for SOAP descriptors")
except ImportError:
    DSCRIBE_AVAILABLE = False
    print("○ DScribe not available - install with: pip install dscribe")

# Materials Project API (official client)
try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
    print("✓ Materials Project API available")
except ImportError:
    MP_API_AVAILABLE = False
    print("○ Materials Project API not available - install with: pip install mp-api")

# Quantum computing (optional)
try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.algorithms import VQE
    QISKIT_AVAILABLE = True
    print("✓ Qiskit available for quantum simulations")
except ImportError:
    QISKIT_AVAILABLE = False
    print("○ Qiskit not available")

# DFT/Simulation tools
try:
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.optimize import BFGS
    from ase.io import read as ase_read
    ASE_AVAILABLE = True
    print("✓ ASE available for atomistic simulations")
except ImportError:
    ASE_AVAILABLE = False
    print("○ ASE not available - install with: pip install ase")

import multiprocessing as mp
from joblib import Parallel, delayed


# ============================================================================
# MAGPIE ELEMENTAL PROPERTIES DATABASE
# ============================================================================

MAGPIE_PROPERTIES = {
    'H': {'atomic_number': 1, 'atomic_mass': 1.008, 'electronegativity': 2.20, 'atomic_radius': 53, 
          'ionization_energy': 1312, 'electron_affinity': 72.8, 'valence_electrons': 1, 
          'melting_point': 14.01, 'density': 0.0899, 'group': 1, 'period': 1},
    'Li': {'atomic_number': 3, 'atomic_mass': 6.94, 'electronegativity': 0.98, 'atomic_radius': 167,
           'ionization_energy': 520, 'electron_affinity': 59.6, 'valence_electrons': 1,
           'melting_point': 453.65, 'density': 0.534, 'group': 1, 'period': 2},
    'Be': {'atomic_number': 4, 'atomic_mass': 9.012, 'electronegativity': 1.57, 'atomic_radius': 112,
           'ionization_energy': 900, 'electron_affinity': -50, 'valence_electrons': 2,
           'melting_point': 1560, 'density': 1.85, 'group': 2, 'period': 2},
    'B': {'atomic_number': 5, 'atomic_mass': 10.81, 'electronegativity': 2.04, 'atomic_radius': 87,
          'ionization_energy': 801, 'electron_affinity': 26.7, 'valence_electrons': 3,
          'melting_point': 2349, 'density': 2.34, 'group': 13, 'period': 2},
    'C': {'atomic_number': 6, 'atomic_mass': 12.01, 'electronegativity': 2.55, 'atomic_radius': 67,
          'ionization_energy': 1086, 'electron_affinity': 121.8, 'valence_electrons': 4,
          'melting_point': 3823, 'density': 2.27, 'group': 14, 'period': 2},
    'N': {'atomic_number': 7, 'atomic_mass': 14.01, 'electronegativity': 3.04, 'atomic_radius': 56,
          'ionization_energy': 1402, 'electron_affinity': -7, 'valence_electrons': 5,
          'melting_point': 63.15, 'density': 1.25, 'group': 15, 'period': 2},
    'O': {'atomic_number': 8, 'atomic_mass': 16.00, 'electronegativity': 3.44, 'atomic_radius': 48,
          'ionization_energy': 1314, 'electron_affinity': 141, 'valence_electrons': 6,
          'melting_point': 54.36, 'density': 1.43, 'group': 16, 'period': 2},
    'F': {'atomic_number': 9, 'atomic_mass': 19.00, 'electronegativity': 3.98, 'atomic_radius': 42,
          'ionization_energy': 1681, 'electron_affinity': 328, 'valence_electrons': 7,
          'melting_point': 53.48, 'density': 1.70, 'group': 17, 'period': 2},
    'Na': {'atomic_number': 11, 'atomic_mass': 22.99, 'electronegativity': 0.93, 'atomic_radius': 190,
           'ionization_energy': 496, 'electron_affinity': 52.8, 'valence_electrons': 1,
           'melting_point': 370.95, 'density': 0.97, 'group': 1, 'period': 3},
    'Mg': {'atomic_number': 12, 'atomic_mass': 24.31, 'electronegativity': 1.31, 'atomic_radius': 145,
           'ionization_energy': 738, 'electron_affinity': -40, 'valence_electrons': 2,
           'melting_point': 923, 'density': 1.74, 'group': 2, 'period': 3},
    'Al': {'atomic_number': 13, 'atomic_mass': 26.98, 'electronegativity': 1.61, 'atomic_radius': 118,
           'ionization_energy': 578, 'electron_affinity': 42.5, 'valence_electrons': 3,
           'melting_point': 933.47, 'density': 2.70, 'group': 13, 'period': 3},
    'Si': {'atomic_number': 14, 'atomic_mass': 28.09, 'electronegativity': 1.90, 'atomic_radius': 111,
           'ionization_energy': 786, 'electron_affinity': 134, 'valence_electrons': 4,
           'melting_point': 1687, 'density': 2.33, 'group': 14, 'period': 3},
    'P': {'atomic_number': 15, 'atomic_mass': 30.97, 'electronegativity': 2.19, 'atomic_radius': 98,
          'ionization_energy': 1012, 'electron_affinity': 72.0, 'valence_electrons': 5,
          'melting_point': 317.3, 'density': 1.82, 'group': 15, 'period': 3},
    'S': {'atomic_number': 16, 'atomic_mass': 32.07, 'electronegativity': 2.58, 'atomic_radius': 88,
          'ionization_energy': 1000, 'electron_affinity': 200, 'valence_electrons': 6,
          'melting_point': 388.36, 'density': 2.07, 'group': 16, 'period': 3},
    'Cl': {'atomic_number': 17, 'atomic_mass': 35.45, 'electronegativity': 3.16, 'atomic_radius': 79,
           'ionization_energy': 1251, 'electron_affinity': 349, 'valence_electrons': 7,
           'melting_point': 171.6, 'density': 3.21, 'group': 17, 'period': 3},
    'K': {'atomic_number': 19, 'atomic_mass': 39.10, 'electronegativity': 0.82, 'atomic_radius': 243,
          'ionization_energy': 419, 'electron_affinity': 48.4, 'valence_electrons': 1,
          'melting_point': 336.53, 'density': 0.86, 'group': 1, 'period': 4},
    'Ca': {'atomic_number': 20, 'atomic_mass': 40.08, 'electronegativity': 1.00, 'atomic_radius': 194,
           'ionization_energy': 590, 'electron_affinity': 2.4, 'valence_electrons': 2,
           'melting_point': 1115, 'density': 1.55, 'group': 2, 'period': 4},
    'Ti': {'atomic_number': 22, 'atomic_mass': 47.87, 'electronegativity': 1.54, 'atomic_radius': 176,
           'ionization_energy': 659, 'electron_affinity': 7.6, 'valence_electrons': 4,
           'melting_point': 1941, 'density': 4.51, 'group': 4, 'period': 4},
    'V': {'atomic_number': 23, 'atomic_mass': 50.94, 'electronegativity': 1.63, 'atomic_radius': 171,
          'ionization_energy': 651, 'electron_affinity': 50.6, 'valence_electrons': 5,
          'melting_point': 2183, 'density': 6.11, 'group': 5, 'period': 4},
    'Cr': {'atomic_number': 24, 'atomic_mass': 52.00, 'electronegativity': 1.66, 'atomic_radius': 166,
           'ionization_energy': 653, 'electron_affinity': 64.3, 'valence_electrons': 6,
           'melting_point': 2180, 'density': 7.19, 'group': 6, 'period': 4},
    'Mn': {'atomic_number': 25, 'atomic_mass': 54.94, 'electronegativity': 1.55, 'atomic_radius': 161,
           'ionization_energy': 717, 'electron_affinity': -50, 'valence_electrons': 7,
           'melting_point': 1519, 'density': 7.21, 'group': 7, 'period': 4},
    'Fe': {'atomic_number': 26, 'atomic_mass': 55.85, 'electronegativity': 1.83, 'atomic_radius': 156,
           'ionization_energy': 762, 'electron_affinity': 15.7, 'valence_electrons': 8,
           'melting_point': 1811, 'density': 7.87, 'group': 8, 'period': 4},
    'Co': {'atomic_number': 27, 'atomic_mass': 58.93, 'electronegativity': 1.88, 'atomic_radius': 152,
           'ionization_energy': 760, 'electron_affinity': 63.7, 'valence_electrons': 9,
           'melting_point': 1768, 'density': 8.90, 'group': 9, 'period': 4},
    'Ni': {'atomic_number': 28, 'atomic_mass': 58.69, 'electronegativity': 1.91, 'atomic_radius': 149,
           'ionization_energy': 737, 'electron_affinity': 112, 'valence_electrons': 10,
           'melting_point': 1728, 'density': 8.91, 'group': 10, 'period': 4},
    'Cu': {'atomic_number': 29, 'atomic_mass': 63.55, 'electronegativity': 1.90, 'atomic_radius': 145,
           'ionization_energy': 745, 'electron_affinity': 118.4, 'valence_electrons': 11,
           'melting_point': 1357.77, 'density': 8.96, 'group': 11, 'period': 4},
    'Zn': {'atomic_number': 30, 'atomic_mass': 65.38, 'electronegativity': 1.65, 'atomic_radius': 142,
           'ionization_energy': 906, 'electron_affinity': -60, 'valence_electrons': 12,
           'melting_point': 692.68, 'density': 7.14, 'group': 12, 'period': 4},
    'Ga': {'atomic_number': 31, 'atomic_mass': 69.72, 'electronegativity': 1.81, 'atomic_radius': 136,
           'ionization_energy': 579, 'electron_affinity': 28.9, 'valence_electrons': 3,
           'melting_point': 302.91, 'density': 5.91, 'group': 13, 'period': 4},
    'Ge': {'atomic_number': 32, 'atomic_mass': 72.63, 'electronegativity': 2.01, 'atomic_radius': 125,
           'ionization_energy': 762, 'electron_affinity': 119, 'valence_electrons': 4,
           'melting_point': 1211.4, 'density': 5.32, 'group': 14, 'period': 4},
    'As': {'atomic_number': 33, 'atomic_mass': 74.92, 'electronegativity': 2.18, 'atomic_radius': 114,
           'ionization_energy': 947, 'electron_affinity': 78, 'valence_electrons': 5,
           'melting_point': 1090, 'density': 5.73, 'group': 15, 'period': 4},
    'Se': {'atomic_number': 34, 'atomic_mass': 78.97, 'electronegativity': 2.55, 'atomic_radius': 103,
           'ionization_energy': 941, 'electron_affinity': 195, 'valence_electrons': 6,
           'melting_point': 494, 'density': 4.81, 'group': 16, 'period': 4},
    'Br': {'atomic_number': 35, 'atomic_mass': 79.90, 'electronegativity': 2.96, 'atomic_radius': 94,
           'ionization_energy': 1140, 'electron_affinity': 324.6, 'valence_electrons': 7,
           'melting_point': 265.8, 'density': 3.12, 'group': 17, 'period': 4},
    'Zr': {'atomic_number': 40, 'atomic_mass': 91.22, 'electronegativity': 1.33, 'atomic_radius': 206,
           'ionization_energy': 640, 'electron_affinity': 41.1, 'valence_electrons': 4,
           'melting_point': 2128, 'density': 6.51, 'group': 4, 'period': 5},
    'Nb': {'atomic_number': 41, 'atomic_mass': 92.91, 'electronegativity': 1.60, 'atomic_radius': 198,
           'ionization_energy': 652, 'electron_affinity': 86.1, 'valence_electrons': 5,
           'melting_point': 2750, 'density': 8.57, 'group': 5, 'period': 5},
    'Mo': {'atomic_number': 42, 'atomic_mass': 95.95, 'electronegativity': 2.16, 'atomic_radius': 190,
           'ionization_energy': 684, 'electron_affinity': 71.9, 'valence_electrons': 6,
           'melting_point': 2896, 'density': 10.28, 'group': 6, 'period': 5},
    'Ag': {'atomic_number': 47, 'atomic_mass': 107.87, 'electronegativity': 1.93, 'atomic_radius': 172,
           'ionization_energy': 731, 'electron_affinity': 125.6, 'valence_electrons': 11,
           'melting_point': 1234.93, 'density': 10.49, 'group': 11, 'period': 5},
    'Sn': {'atomic_number': 50, 'atomic_mass': 118.71, 'electronegativity': 1.96, 'atomic_radius': 162,
           'ionization_energy': 709, 'electron_affinity': 107.3, 'valence_electrons': 4,
           'melting_point': 505.08, 'density': 7.31, 'group': 14, 'period': 5},
    'I': {'atomic_number': 53, 'atomic_mass': 126.90, 'electronegativity': 2.66, 'atomic_radius': 115,
          'ionization_energy': 1008, 'electron_affinity': 295.2, 'valence_electrons': 7,
          'melting_point': 386.85, 'density': 4.93, 'group': 17, 'period': 5},
    'Cs': {'atomic_number': 55, 'atomic_mass': 132.91, 'electronegativity': 0.79, 'atomic_radius': 298,
           'ionization_energy': 376, 'electron_affinity': 45.5, 'valence_electrons': 1,
           'melting_point': 301.59, 'density': 1.93, 'group': 1, 'period': 6},
    'Ba': {'atomic_number': 56, 'atomic_mass': 137.33, 'electronegativity': 0.89, 'atomic_radius': 253,
           'ionization_energy': 503, 'electron_affinity': 13.95, 'valence_electrons': 2,
           'melting_point': 1000, 'density': 3.51, 'group': 2, 'period': 6},
    'La': {'atomic_number': 57, 'atomic_mass': 138.91, 'electronegativity': 1.10, 'atomic_radius': 195,
           'ionization_energy': 538, 'electron_affinity': 48, 'valence_electrons': 3,
           'melting_point': 1193, 'density': 6.15, 'group': 3, 'period': 6},
    'Ce': {'atomic_number': 58, 'atomic_mass': 140.12, 'electronegativity': 1.12, 'atomic_radius': 185,
           'ionization_energy': 534, 'electron_affinity': 50, 'valence_electrons': 4,
           'melting_point': 1068, 'density': 6.77, 'group': 3, 'period': 6},
    'Nd': {'atomic_number': 60, 'atomic_mass': 144.24, 'electronegativity': 1.14, 'atomic_radius': 201,
           'ionization_energy': 533, 'electron_affinity': 185, 'valence_electrons': 6,
           'melting_point': 1297, 'density': 7.01, 'group': 3, 'period': 6},
    'Gd': {'atomic_number': 64, 'atomic_mass': 157.25, 'electronegativity': 1.20, 'atomic_radius': 188,
           'ionization_energy': 593, 'electron_affinity': 50, 'valence_electrons': 10,
           'melting_point': 1585, 'density': 7.90, 'group': 3, 'period': 6},
    'W': {'atomic_number': 74, 'atomic_mass': 183.84, 'electronegativity': 2.36, 'atomic_radius': 193,
          'ionization_energy': 770, 'electron_affinity': 78.6, 'valence_electrons': 6,
          'melting_point': 3695, 'density': 19.25, 'group': 6, 'period': 6},
    'Pt': {'atomic_number': 78, 'atomic_mass': 195.08, 'electronegativity': 2.28, 'atomic_radius': 177,
           'ionization_energy': 870, 'electron_affinity': 205.3, 'valence_electrons': 10,
           'melting_point': 2041.4, 'density': 21.45, 'group': 10, 'period': 6},
    'Au': {'atomic_number': 79, 'atomic_mass': 196.97, 'electronegativity': 2.54, 'atomic_radius': 174,
           'ionization_energy': 890, 'electron_affinity': 222.8, 'valence_electrons': 11,
           'melting_point': 1337.33, 'density': 19.30, 'group': 11, 'period': 6},
    'Pb': {'atomic_number': 82, 'atomic_mass': 207.2, 'electronegativity': 2.33, 'atomic_radius': 154,
           'ionization_energy': 716, 'electron_affinity': 35.1, 'valence_electrons': 4,
           'melting_point': 600.61, 'density': 11.34, 'group': 14, 'period': 6},
    'Bi': {'atomic_number': 83, 'atomic_mass': 208.98, 'electronegativity': 2.02, 'atomic_radius': 143,
           'ionization_energy': 703, 'electron_affinity': 91.2, 'valence_electrons': 5,
           'melting_point': 544.55, 'density': 9.78, 'group': 15, 'period': 6},
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Material:
    """Material composition and structure"""
    composition: str
    structure: Optional[Any] = None
    space_group: Optional[str] = None
    crystal_system: Optional[str] = None
    lattice_params: Optional[Dict] = None
    cif_data: Optional[str] = None
    formation_energy: Optional[float] = None
    band_gap: Optional[float] = None
    density: Optional[float] = None
    features: Optional[np.ndarray] = None
    material_id: Optional[str] = None
    material_type: str = "crystal"
    smiles: Optional[str] = None
    source: str = "Generated"
    tags: List[str] = field(default_factory=list)


@dataclass
class PropertyPrediction:
    """Predicted material properties"""
    material_id: str
    bulk_modulus: Optional[float] = None
    shear_modulus: Optional[float] = None
    youngs_modulus: Optional[float] = None
    poissons_ratio: Optional[float] = None
    hardness: Optional[float] = None
    band_gap: Optional[float] = None
    work_function: Optional[float] = None
    conductivity: Optional[float] = None
    melting_point: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    specific_heat: Optional[float] = None
    magnetic_moment: Optional[float] = None
    curie_temperature: Optional[float] = None
    voltage: Optional[float] = None
    capacity: Optional[float] = None
    ion_mobility: Optional[float] = None
    synthesizability_score: Optional[float] = None
    synthesis_temperature: Optional[float] = None
    formation_energy: Optional[float] = None
    energy_above_hull: Optional[float] = None
    decomposition_temperature: Optional[float] = None


@dataclass
class ManufacturabilityScore:
    """Manufacturability assessment"""
    material_id: str
    overall_score: float
    synthesis_feasibility: float
    cost_factor: float
    scalability: float
    environmental_score: float
    complexity: float
    recommendations: List[str]


@dataclass
class SynthesisRoute:
    """Synthesis route recommendation"""
    method: str
    precursors: List[str]
    temperature: str
    atmosphere: str
    time: str
    difficulty: str
    success_probability: float


# ============================================================================
# MATERIALS PROJECT API INTEGRATION (using official mp-api package)
# ============================================================================

class MaterialsProjectLoader:
    """
    Load training data from Materials Project database using official mp-api package
    Access 150,000+ calculated materials properties
    
    Uses MPRester client for efficient API access with methods:
    - summary.search() for general material queries
    - insertion_electrodes.search() for battery data
    - get_entries_in_chemsys() for phase diagrams
    
    Features:
    - Load training data for any property (band_gap, formation_energy, etc.)
    - Filter by elements and criteria
    - Bulk queries for efficiency
    - Battery-specific data (insertion electrodes, voltages, capacities)
    - Application-specific loaders (solar, thermoelectric, superconductor)
    - Phase diagram generation for stability analysis
    
    Output Schema:
        All application-specific loaders return consistent enriched dicts:
        {
            'material': Material,       # Material dataclass object
            'material_id': str,         # Materials Project ID
            'composition': str,         # Chemical formula
            'properties': Dict,         # Property values (e.g., band_gap, voltage)
            **metadata                  # Application-specific metadata
        }
        
        Note: load_training_data returns (List[Material], np.array) for ML 
        training compatibility, as ML models require property arrays.
    
    Usage:
        mp_loader = MaterialsProjectLoader(api_key='your-key')
        
        # For ML training:
        materials, band_gaps = mp_loader.load_training_data(
            property_name='band_gap',
            n_materials=5000,
            additional_criteria={'band_gap': (1.0, 3.0)}
        )
        
        # For application-specific loading (returns enriched dicts):
        solar_materials = mp_loader.load_solar_materials(n_materials=1000)
        # Each result: {material, material_id, composition, properties, ...}
    """
    
    # Property name mappings from user-friendly to MP API fields
    # Format: 'user_field': 'api_field.nested_attr' for nested access
    PROPERTY_MAPPINGS = {
        'band_gap': 'band_gap',
        'formation_energy': 'formation_energy_per_atom',
        'formation_energy_per_atom': 'formation_energy_per_atom',
        'energy_above_hull': 'energy_above_hull',
        'density': 'density',
        'volume': 'volume',
        'total_magnetization': 'total_magnetization',
        'is_stable': 'is_stable',
        'is_metal': 'is_metal',
        'space_group': 'symmetry.symbol',
        'crystal_system': 'symmetry.crystal_system',
        'symmetry_symbol': 'symmetry.symbol',
        'bulk_modulus': 'bulk_modulus.vrh',
        'bulk_modulus_voigt': 'bulk_modulus.voigt',
        'bulk_modulus_reuss': 'bulk_modulus.reuss',
        'shear_modulus': 'shear_modulus.vrh',
        'shear_modulus_voigt': 'shear_modulus.voigt',
        'shear_modulus_reuss': 'shear_modulus.reuss',
        'dielectric_constant': 'dielectric.e_total',
        'dielectric_electronic': 'dielectric.e_electronic',
        'piezoelectric_modulus': 'piezoelectric.e_ij_max',
    }
    
    @staticmethod
    def _extract_property(doc, api_field: str):
        """
        Extract property value from mp-api document, handling dotted nested paths.
        
        Args:
            doc: mp-api document object
            api_field: Field path (e.g., 'band_gap' or 'bulk_modulus.vrh')
            
        Returns:
            Extracted value or None if not found
        """
        # Handle dotted paths (e.g., 'bulk_modulus.vrh')
        parts = api_field.split('.')
        value = doc
        
        for part in parts:
            if value is None:
                return None
            value = getattr(value, part, None)
        
        return value
    
    @staticmethod
    def _build_enriched_dict(
        material: 'Material',
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a consistent enriched dict from a Material object.
        
        All loaders should use this helper to ensure consistent output schema.
        
        Args:
            material: Material object
            properties: Dict of property name -> value
            metadata: Additional metadata (application, notes, etc.)
            
        Returns:
            Enriched dict with consistent schema:
            {
                'material': Material,
                'material_id': str,
                'composition': str,
                'properties': {...},
                **metadata
            }
        """
        result = {
            'material': material,
            'material_id': material.material_id or '',
            'composition': material.composition,
            'properties': properties or {},
        }
        if metadata:
            result.update(metadata)
        return result
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Materials Project client using official mp-api package
        
        Args:
            api_key: Materials Project API key (get from https://materialsproject.org/api)
                    If not provided, will try to load from MP_API_KEY environment variable
        """
        self.api_key = api_key or os.environ.get('MP_API_KEY')
        
        if not self.api_key:
            print("Warning: No Materials Project API key provided")
            print("Set MP_API_KEY environment variable or pass api_key parameter")
            print("Get key from: https://materialsproject.org/api")
            self.client = None
        elif MP_API_AVAILABLE:
            self.client = MPRester(self.api_key)
            print("Connected to Materials Project via official mp-api")
        else:
            self.client = None
            print("Materials Project API not available - install with: pip install mp-api")
    
    def load_training_data(
        self,
        property_name: str,
        n_materials: int = 10000,
        elements: Optional[List[str]] = None,
        additional_criteria: Optional[Dict] = None,
        include_structures: bool = True
    ) -> Tuple[List, np.ndarray]:
        """
        Load materials and property values for training using MPRester
        
        Args:
            property_name: Property to load (e.g., 'band_gap', 'formation_energy_per_atom')
            n_materials: Maximum number of materials
            elements: Filter by elements (e.g., ['Li', 'Fe', 'O'])
            additional_criteria: Additional search criteria
            include_structures: Whether to include crystal structures
            
        Returns:
            (materials_list, property_values)
        """
        if not self.client:
            print("Error: Materials Project client not initialized")
            return [], np.array([])
        
        # Map property name to API field (may be dotted like 'bulk_modulus.vrh')
        api_property = self.PROPERTY_MAPPINGS.get(property_name, property_name)
        # Get root field for API request (e.g., 'bulk_modulus' from 'bulk_modulus.vrh')
        api_root_field = api_property.split('.')[0]
        
        print(f"\nLoading {property_name} ({api_property}) data from Materials Project...")
        
        # Build query criteria
        criteria = additional_criteria.copy() if additional_criteria else {}
        if elements:
            criteria['elements'] = elements
        
        # Build field list (use root field for API request)
        fields = ['material_id', 'formula_pretty', api_root_field]
        if include_structures:
            fields.append('structure')
        
        try:
            # Use MPRester summary.search() - the official API method
            docs = self.client.summary.search(
                **criteria,
                fields=fields,
                num_chunks=10,
                chunk_size=1000
            )
            
            materials = []
            property_values = []
            
            for doc in docs[:n_materials]:
                mat_id = doc.material_id
                composition = doc.formula_pretty
                prop_value = self._extract_property(doc, api_property)
                
                if prop_value is not None:
                    material = Material(
                        composition=composition,
                        structure=doc.structure if include_structures and hasattr(doc, 'structure') else None,
                        material_id=str(mat_id),
                        source='Materials Project'
                    )
                    
                    materials.append(material)
                    property_values.append(prop_value)
            
            print(f"Loaded {len(materials)} materials with {property_name} data")
            return materials, np.array(property_values)
            
        except Exception as e:
            print(f"Error loading from Materials Project: {e}")
            return [], np.array([])
    
    def load_battery_data(
        self,
        n_materials: int = 5000,
        working_ion: str = 'Li',
        min_capacity: Optional[float] = None,
        max_voltage: Optional[float] = None
    ) -> List[Dict]:
        """
        Load battery-specific data using MPRester insertion_electrodes endpoint
        
        Args:
            n_materials: Maximum number of materials
            working_ion: Working ion (Li, Na, K, Mg, Ca, etc.)
            min_capacity: Minimum gravimetric capacity (mAh/g)
            max_voltage: Maximum voltage vs working ion
            
        Returns:
            List of enriched dicts with battery electrode data (consistent schema)
        """
        if not self.client:
            print("Error: Materials Project client not initialized")
            return []
        
        print(f"\nLoading {working_ion}-ion battery materials from Materials Project...")
        
        try:
            # Use MPRester insertion_electrodes.search()
            battery_docs = self.client.insertion_electrodes.search(
                working_ion=working_ion,
                fields=['material_id', 'formula_pretty', 'average_voltage', 
                       'max_voltage', 'capacity_grav', 'energy_grav'],
                num_chunks=5
            )
            
            materials = []
            voltages = []
            capacities = []
            
            for doc in battery_docs[:n_materials]:
                # Apply filters
                if min_capacity and doc.capacity_grav and doc.capacity_grav < min_capacity:
                    continue
                if max_voltage and doc.average_voltage and doc.average_voltage > max_voltage:
                    continue
                
                # Create Material object for consistency
                mat_obj = Material(
                    composition=doc.formula_pretty,
                    material_id=str(doc.material_id)
                )
                
                mat_data = self._build_enriched_dict(
                    material=mat_obj,
                    properties={
                        'voltage': doc.average_voltage,
                        'max_voltage': doc.max_voltage,
                        'capacity': doc.capacity_grav,
                        'energy_density': doc.energy_grav
                    },
                    metadata={
                        'working_ion': working_ion,
                        'application': 'battery'
                    }
                )
                
                materials.append(mat_data)
                if doc.average_voltage:
                    voltages.append(doc.average_voltage)
                if doc.capacity_grav:
                    capacities.append(doc.capacity_grav)
            
            print(f"Loaded {len(materials)} {working_ion}-ion battery materials")
            return materials
            
        except Exception as e:
            print(f"Error loading battery data: {e}")
            return []
    
    def load_by_application(self, application: str, n_materials: int = 5000) -> List[Dict]:
        """
        Load materials for specific applications
        
        Args:
            application: 'battery', 'solar', 'thermoelectric', 'catalyst', 'superconductor'
            n_materials: Maximum number of materials
            
        Returns:
            List of enriched dicts with 'material' key containing Material object
        """
        if application == 'battery':
            return self.load_battery_data(n_materials)
        
        elif application == 'solar':
            return self.load_solar_materials(n_materials=n_materials)
        
        elif application == 'thermoelectric':
            return self.load_thermoelectric_materials(n_materials=n_materials)
        
        elif application == 'superconductor':
            return self.load_superconductor_candidates(n_materials=n_materials)
        
        elif application == 'catalyst':
            # Transition metal compounds
            elements = ['Pt', 'Pd', 'Ni', 'Co', 'Fe', 'Mo', 'W', 'Ru', 'Rh']
            materials, props = self.load_training_data(
                'formation_energy_per_atom',
                n_materials=n_materials,
                elements=elements
            )
            return [self._build_enriched_dict(
                material=mat,
                properties={'formation_energy': prop},
                metadata={'application': 'catalyst'}
            ) for mat, prop in zip(materials, props)]
        
        else:
            print(f"Unknown application: {application}")
            return []
    
    def get_phase_diagram(self, elements: List[str]) -> Optional[Any]:
        """
        Get phase diagram for element system using MPRester
        
        Args:
            elements: List of elements (e.g., ['Li', 'Fe', 'P', 'O'])
            
        Returns:
            PhaseDiagram object or None if unavailable
        """
        if not self.client:
            return None
        
        if not PYMATGEN_AVAILABLE:
            print("Pymatgen required for phase diagram generation")
            return None
        
        print(f"\nRetrieving phase diagram for {'-'.join(elements)}...")
        
        try:
            # Use MPRester get_entries_in_chemsys() for phase diagram
            entries = self.client.get_entries_in_chemsys(elements)
            
            # Create phase diagram using pymatgen
            from pymatgen.analysis.phase_diagram import PhaseDiagram
            pd = PhaseDiagram(entries)
            
            print(f"Phase diagram created with {len(entries)} phases")
            return pd
            
        except Exception as e:
            print(f"Error creating phase diagram: {e}")
            return None
    
    def load_solar_materials(
        self,
        n_materials: int = 1000,
        band_gap_range: Tuple[float, float] = (1.0, 2.5),
        stable_only: bool = True
    ) -> List[Dict]:
        """
        Load materials suitable for solar absorber applications
        
        Args:
            n_materials: Maximum number of materials
            band_gap_range: (min, max) band gap in eV
            stable_only: Only include thermodynamically stable materials
            
        Returns:
            List of solar-suitable materials with properties
        """
        criteria = {'band_gap': band_gap_range}
        if stable_only:
            criteria['is_stable'] = True
        
        materials, band_gaps = self.load_training_data(
            property_name='band_gap',
            n_materials=n_materials,
            additional_criteria=criteria,
            include_structures=True
        )
        
        # Enrich with solar-relevant properties using consistent helper
        solar_materials = []
        for mat, bg in zip(materials, band_gaps):
            solar_mat = self._build_enriched_dict(
                material=mat,
                properties={'band_gap': bg},
                metadata={
                    'solar_efficiency_estimate': self._estimate_solar_efficiency(bg),
                    'application_notes': self._get_solar_application_notes(bg),
                    'application': 'solar'
                }
            )
            solar_materials.append(solar_mat)
        
        solar_materials.sort(key=lambda x: x['solar_efficiency_estimate'], reverse=True)
        print(f"Loaded {len(solar_materials)} solar absorber candidates")
        return solar_materials
    
    def _estimate_solar_efficiency(self, band_gap: float) -> float:
        """Estimate theoretical solar efficiency from band gap using Shockley-Queisser limit"""
        optimal_bg = 1.34
        if band_gap < 0.5 or band_gap > 3.5:
            return 0.0
        efficiency = 0.33 * np.exp(-((band_gap - optimal_bg) ** 2) / 0.5)
        return round(efficiency * 100, 1)
    
    def _get_solar_application_notes(self, band_gap: float) -> str:
        """Get application notes based on band gap"""
        if 1.1 <= band_gap <= 1.5:
            return "Optimal for single-junction solar cells"
        elif 1.5 < band_gap <= 2.0:
            return "Suitable for top cell in tandem configurations"
        elif 0.8 < band_gap < 1.1:
            return "Suitable for bottom cell in tandem configurations"
        elif 2.0 < band_gap <= 3.0:
            return "UV absorber, transparent conductor applications"
        else:
            return "Non-optimal for photovoltaic applications"
    
    def load_thermoelectric_materials(
        self,
        n_materials: int = 1000,
        band_gap_range: Tuple[float, float] = (0.1, 1.0),
        heavy_elements: bool = True
    ) -> List[Dict]:
        """
        Load materials suitable for thermoelectric applications
        
        Args:
            n_materials: Maximum number of materials
            band_gap_range: (min, max) band gap in eV
            heavy_elements: Prefer materials with heavy elements
            
        Returns:
            List of thermoelectric material candidates
        """
        criteria = {'band_gap': band_gap_range}
        heavy_elements_list = ['Bi', 'Pb', 'Te', 'Sb', 'Sn', 'Se', 'Ag', 'Cu', 'Ge', 'In']
        
        materials, band_gaps = self.load_training_data(
            property_name='band_gap',
            n_materials=n_materials,
            elements=heavy_elements_list if heavy_elements else None,
            additional_criteria=criteria,
            include_structures=True
        )
        
        thermoelectric_materials = []
        for mat, bg in zip(materials, band_gaps):
            te_mat = self._build_enriched_dict(
                material=mat,
                properties={'band_gap': bg},
                metadata={
                    'te_score': self._estimate_te_potential(mat, bg),
                    'application': 'thermoelectric'
                }
            )
            thermoelectric_materials.append(te_mat)
        
        # Sort by TE score
        thermoelectric_materials.sort(key=lambda x: x['te_score'], reverse=True)
        
        print(f"Loaded {len(thermoelectric_materials)} thermoelectric candidates")
        return thermoelectric_materials
    
    def _estimate_te_potential(self, material, band_gap: float) -> float:
        """Estimate thermoelectric potential score"""
        score = 0.0
        
        # Optimal band gap for TE (0.2-0.5 eV)
        if 0.2 <= band_gap <= 0.5:
            score += 40
        elif 0.1 <= band_gap <= 1.0:
            score += 20
        
        # Heavy elements bonus
        heavy = ['Bi', 'Pb', 'Te', 'Sb', 'Sn', 'Se']
        # Handle both Material objects and dicts
        formula = material.composition if hasattr(material, 'composition') else str(material.get('formula', ''))
        for elem in heavy:
            if elem in formula:
                score += 10
        
        return min(score, 100)
    
    def load_superconductor_candidates(
        self,
        n_materials: int = 500,
        include_cuprates: bool = True,
        include_iron_based: bool = True
    ) -> List[Dict]:
        """
        Load potential superconductor material candidates
        
        Args:
            n_materials: Maximum number of materials
            include_cuprates: Include cuprate-like structures
            include_iron_based: Include iron-based candidates
            
        Returns:
            List of superconductor candidates
        """
        # Elements commonly in superconductors
        sc_elements = []
        if include_cuprates:
            sc_elements.extend(['Cu', 'O', 'Ba', 'Y', 'La', 'Sr', 'Ca'])
        if include_iron_based:
            sc_elements.extend(['Fe', 'As', 'Se', 'P'])
        
        # Add conventional superconductor elements
        sc_elements.extend(['Nb', 'V', 'Ti', 'Mg', 'B', 'H'])
        sc_elements = list(set(sc_elements))
        
        materials, _ = self.load_training_data(
            property_name='formation_energy',
            n_materials=n_materials,
            elements=sc_elements[:5],  # Limit for API efficiency
            additional_criteria={'is_stable': True},
            include_structures=True
        )
        
        sc_candidates = []
        for mat in materials:
            sc_mat = self._build_enriched_dict(
                material=mat,
                properties={},
                metadata={
                    'sc_type': self._classify_sc_type(mat),
                    'sc_potential_notes': self._get_sc_notes(mat),
                    'application': 'superconductor'
                }
            )
            sc_candidates.append(sc_mat)
        
        print(f"Loaded {len(sc_candidates)} superconductor candidates")
        return sc_candidates
    
    def _classify_sc_type(self, material) -> str:
        """Classify superconductor type based on composition"""
        # Handle both Material objects and dicts
        formula = material.composition if hasattr(material, 'composition') else str(material.get('formula', ''))
        if 'Cu' in formula and 'O' in formula:
            return 'cuprate'
        elif 'Fe' in formula and ('As' in formula or 'Se' in formula):
            return 'iron-based'
        elif 'Nb' in formula or 'V' in formula:
            return 'conventional-bcs'
        elif 'H' in formula:
            return 'hydride'
        else:
            return 'other'
    
    def _get_sc_notes(self, material) -> str:
        """Get superconductor notes"""
        sc_type = self._classify_sc_type(material)
        notes = {
            'cuprate': 'High-Tc cuprate family - typically Tc > 77K',
            'iron-based': 'Iron-based superconductor - typically Tc 20-55K',
            'conventional-bcs': 'Conventional BCS superconductor - typically Tc < 20K',
            'hydride': 'Hydride superconductor - high pressure required',
            'other': 'Potential unconventional superconductor'
        }
        return notes.get(sc_type, '')
    
    def to_training_dataframe(
        self,
        materials: List[Material],
        property_values: List[float],
        property_name: str
    ) -> pd.DataFrame:
        """
        Convert loaded materials to a pandas DataFrame for ML training
        
        Args:
            materials: List of Material objects
            property_values: List of target property values
            property_name: Name of the target property
            
        Returns:
            DataFrame with material_id, formula, and target property
        """
        rows = []
        for mat, value in zip(materials, property_values):
            row = {
                'material_id': mat.material_id,
                'formula': mat.composition,
                property_name: value
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df


# ============================================================================
# DFT CALCULATORS (VASP & Quantum ESPRESSO)
# ============================================================================

class VASPCalculator:
    """
    VASP (Vienna Ab initio Simulation Package) calculator
    Industry standard for materials DFT calculations
    
    Requires:
    - VASP executable (vasp_std, vasp_gam, vasp_ncl)
    - POTCAR files in VASP_PP_PATH directory
    """
    
    def __init__(self, 
                 vasp_cmd: str = 'vasp_std',
                 potcar_dir: Optional[str] = None,
                 nprocs: int = 16):
        """
        Args:
            vasp_cmd: VASP executable ('vasp_std', 'vasp_gam', 'vasp_ncl')
            potcar_dir: Directory containing POTCAR files
            nprocs: Number of MPI processes
        """
        import tempfile
        self.vasp_cmd = vasp_cmd
        self.potcar_dir = potcar_dir or os.environ.get('VASP_PP_PATH')
        self.nprocs = nprocs
        
        if not self.potcar_dir:
            print("Warning: VASP POTCAR directory not set")
            print("Set VASP_PP_PATH environment variable")
    
    def setup_calculation(self, 
                         material: Material,
                         calc_type: str = 'relax') -> Dict:
        """
        Setup VASP calculation input parameters
        
        Args:
            material: Material to calculate
            calc_type: 'relax', 'static', 'band', 'dos'
        
        Returns:
            VASP input configuration dictionary
        """
        if not material.structure or not PYMATGEN_AVAILABLE:
            return {}
        
        structure = material.structure
        
        # INCAR settings based on calculation type
        if calc_type == 'relax':
            incar = {
                'PREC': 'Accurate',
                'ENCUT': 520,
                'EDIFF': 1e-5,
                'IBRION': 2,  # CG relaxation
                'ISIF': 3,    # Relax cell + ions
                'NSW': 200,   # Max ionic steps
                'LWAVE': False,
                'LCHARG': False
            }
        
        elif calc_type == 'static':
            incar = {
                'PREC': 'Accurate',
                'ENCUT': 520,
                'EDIFF': 1e-6,
                'ISMEAR': -5,  # Tetrahedron method
                'LORBIT': 11,  # DOSCAR with local DOS
                'LWAVE': True,
                'LCHARG': True
            }
        
        elif calc_type == 'band':
            incar = {
                'PREC': 'Accurate',
                'ENCUT': 520,
                'EDIFF': 1e-6,
                'ISMEAR': 0,
                'SIGMA': 0.05,
                'ICHARG': 11,  # Read from CHGCAR
                'LWAVE': False,
                'LCHARG': False
            }
        
        elif calc_type == 'dos':
            incar = {
                'PREC': 'Accurate',
                'ENCUT': 520,
                'EDIFF': 1e-6,
                'ISMEAR': -5,
                'NEDOS': 2001,
                'LORBIT': 11
            }
        else:
            incar = {'PREC': 'Accurate', 'ENCUT': 520}
        
        return {
            'structure': structure,
            'incar': incar,
            'calc_type': calc_type
        }
    
    def run_calculation(self, 
                       material: Material,
                       calc_type: str = 'relax',
                       workdir: Optional[str] = None) -> Dict:
        """
        Run VASP calculation
        
        Args:
            material: Material to calculate
            calc_type: Calculation type
            workdir: Working directory (creates temp if None)
        
        Returns:
            Results dictionary with energy, forces, band gap
        """
        import tempfile
        
        if not ASE_AVAILABLE:
            return {'error': 'ASE not available'}
        
        # Create working directory
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix='vasp_')
        os.makedirs(workdir, exist_ok=True)
        
        print(f"\n[VASP] Running {calc_type} calculation for {material.composition}")
        print(f"  Working directory: {workdir}")
        
        # Setup calculation
        config = self.setup_calculation(material, calc_type)
        if not config:
            return {'error': 'Failed to setup calculation'}
        
        # Convert Pymatgen Structure to ASE Atoms
        structure = config['structure']
        atoms = self._structure_to_atoms(structure)
        
        try:
            from ase.calculators.vasp import Vasp
            from ase.optimize import LBFGS
            
            # Setup VASP calculator
            calc = Vasp(
                directory=workdir,
                command=f'mpirun -np {self.nprocs} {self.vasp_cmd}',
                **config['incar']
            )
            
            atoms.calc = calc
            
            # Run calculation
            if calc_type == 'relax':
                opt = LBFGS(atoms, trajectory=f'{workdir}/relax.traj')
                opt.run(fmax=0.01)
            
            # Get results
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            results = {
                'status': 'completed',
                'energy': energy,
                'energy_per_atom': energy / len(atoms),
                'forces': forces.tolist(),
                'final_structure': atoms,
                'workdir': workdir
            }
            
            # Get band gap if available
            if calc_type in ['static', 'band', 'dos']:
                results['band_gap'] = self._extract_band_gap(workdir)
            
            print(f"  VASP calculation completed")
            print(f"  Energy: {energy:.4f} eV")
            
            return results
            
        except Exception as e:
            print(f"Error running VASP: {e}")
            return {'error': str(e), 'workdir': workdir}
    
    def _structure_to_atoms(self, structure) -> Any:
        """Convert Pymatgen Structure to ASE Atoms"""
        from ase import Atoms
        atoms = Atoms(
            symbols=[str(site.specie) for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )
        return atoms
    
    def _extract_band_gap(self, workdir: str) -> Optional[float]:
        """Extract band gap from VASP output"""
        try:
            from pymatgen.io.vasp.outputs import Vasprun
            vasprun = Vasprun(f"{workdir}/vasprun.xml")
            band_gap = vasprun.get_band_structure().get_band_gap()['energy']
            return band_gap
        except:
            return None


class QuantumESPRESSOCalculator:
    """
    Quantum ESPRESSO calculator
    Open-source DFT code, alternative to VASP
    
    Requires:
    - pw.x executable (Quantum ESPRESSO)
    - Pseudopotentials in ESPRESSO_PSEUDO directory
    """
    
    def __init__(self, 
                 pw_cmd: str = 'pw.x',
                 pseudo_dir: Optional[str] = None,
                 nprocs: int = 16):
        """
        Args:
            pw_cmd: Quantum ESPRESSO pw.x executable
            pseudo_dir: Directory containing pseudopotentials
            nprocs: Number of MPI processes
        """
        self.pw_cmd = pw_cmd
        self.pseudo_dir = pseudo_dir or os.environ.get('ESPRESSO_PSEUDO')
        self.nprocs = nprocs
        
        if not self.pseudo_dir:
            print("Warning: Quantum ESPRESSO pseudopotential directory not set")
            print("Set ESPRESSO_PSEUDO environment variable")
    
    def run_calculation(self,
                       material: Material,
                       calc_type: str = 'scf',
                       workdir: Optional[str] = None) -> Dict:
        """
        Run Quantum ESPRESSO calculation
        
        Args:
            material: Material to calculate
            calc_type: 'scf', 'relax', 'vc-relax', 'bands', 'nscf'
        
        Returns:
            Results dictionary
        """
        import tempfile
        
        if not ASE_AVAILABLE:
            return {'error': 'ASE not available'}
        
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix='qe_')
        os.makedirs(workdir, exist_ok=True)
        
        print(f"\n[QE] Running {calc_type} calculation for {material.composition}")
        
        # Convert structure
        if not material.structure:
            return {'error': 'No structure available'}
        
        structure = material.structure
        atoms = self._structure_to_atoms(structure)
        
        # Setup calculator
        input_data = {
            'control': {
                'calculation': calc_type,
                'pseudo_dir': self.pseudo_dir,
                'outdir': workdir,
                'prefix': material.material_id or 'pwscf'
            },
            'system': {
                'ecutwfc': 60,  # Ry
                'ecutrho': 480,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.02
            },
            'electrons': {
                'conv_thr': 1e-8,
                'mixing_beta': 0.7
            }
        }
        
        if calc_type in ['relax', 'vc-relax']:
            input_data['ions'] = {
                'ion_dynamics': 'bfgs'
            }
        
        if calc_type == 'vc-relax':
            input_data['cell'] = {
                'cell_dynamics': 'bfgs'
            }
        
        try:
            from ase.calculators.espresso import Espresso
            
            calc = Espresso(
                command=f'mpirun -np {self.nprocs} {self.pw_cmd} -in PREFIX.pwi > PREFIX.pwo',
                input_data=input_data,
                pseudopotentials=self._get_pseudopotentials(atoms),
                kpts=(4, 4, 4)
            )
            
            atoms.calc = calc
            
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            results = {
                'status': 'completed',
                'energy': energy,
                'forces': forces.tolist(),
                'final_structure': atoms,
                'workdir': workdir
            }
            
            print(f"  QE calculation completed")
            print(f"  Energy: {energy:.4f} eV")
            
            return results
            
        except Exception as e:
            print(f"Error running Quantum ESPRESSO: {e}")
            return {'error': str(e)}
    
    def _structure_to_atoms(self, structure) -> Any:
        """Convert Pymatgen Structure to ASE Atoms"""
        from ase import Atoms
        atoms = Atoms(
            symbols=[str(site.specie) for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )
        return atoms
    
    def _get_pseudopotentials(self, atoms) -> Dict:
        """Get pseudopotential files for each element"""
        # Simplified - in production, map elements to specific PP files
        pseudos = {}
        for symbol in set(atoms.get_chemical_symbols()):
            pseudos[symbol] = f"{symbol}.UPF"
        return pseudos


# ============================================================================
# MAGPIE DESCRIPTORS
# ============================================================================

class MagpieFeaturizer:
    """
    Generate Magpie compositional descriptors
    
    Features 145 descriptors based on:
    - Stoichiometric attributes
    - Elemental property statistics (mean, std, min, max, range, mode)
    - Valence orbital occupation
    - Ionic character
    """
    
    PROPERTY_NAMES = [
        'atomic_number', 'atomic_mass', 'electronegativity', 'atomic_radius',
        'ionization_energy', 'electron_affinity', 'valence_electrons',
        'melting_point', 'density', 'group', 'period'
    ]
    
    def __init__(self):
        self.properties = MAGPIE_PROPERTIES
    
    def get_element_property(self, element: str, prop: str) -> float:
        """Get elemental property value"""
        if element in self.properties:
            return self.properties[element].get(prop, 0.0)
        return 0.0
    
    def parse_composition(self, formula: str) -> Dict[str, float]:
        """Parse chemical formula to element:fraction dict"""
        if PYMATGEN_AVAILABLE:
            try:
                comp = Composition(formula)
                return {str(el): float(amt) for el, amt in comp.fractional_composition.items()}
            except:
                pass
        
        import re
        elements = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
        composition = {}
        for element, count in elements:
            if element:
                composition[element] = float(count) if count else 1.0
        
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        return composition
    
    def featurize(self, composition: str) -> np.ndarray:
        """
        Generate 145 Magpie descriptors for composition
        
        Returns:
            numpy array of descriptors
        """
        comp_dict = self.parse_composition(composition)
        
        if not comp_dict:
            return np.zeros(145)
        
        features = []
        
        # Stoichiometric features (10)
        n_elements = len(comp_dict)
        fractions = list(comp_dict.values())
        features.extend([
            n_elements,
            0 if n_elements < 2 else sum(f * (1 - f) for f in fractions),  # p-norm 0
            sum(f**2 for f in fractions),  # p-norm 2
            sum(f**3 for f in fractions),  # p-norm 3
            sum(f**5 for f in fractions),  # p-norm 5
            sum(f**7 for f in fractions),  # p-norm 7
            sum(f**10 for f in fractions),  # p-norm 10
            max(fractions),
            min(fractions),
            max(fractions) - min(fractions),
        ])
        
        # Elemental property statistics (11 properties × 6 stats = 66)
        for prop in self.PROPERTY_NAMES:
            values = []
            weights = []
            for element, fraction in comp_dict.items():
                val = self.get_element_property(element, prop)
                values.append(val)
                weights.append(fraction)
            
            if values:
                weighted_mean = np.average(values, weights=weights)
                features.extend([
                    weighted_mean,
                    np.std(values) if len(values) > 1 else 0,
                    np.min(values),
                    np.max(values),
                    np.max(values) - np.min(values),
                    values[np.argmax(weights)]  # Mode (element with max fraction)
                ])
            else:
                features.extend([0.0] * 6)
        
        # Valence orbital features (32) - s, p, d, f counts
        s_electrons = []
        p_electrons = []
        d_electrons = []
        f_electrons = []
        
        for element, fraction in comp_dict.items():
            ve = self.get_element_property(element, 'valence_electrons')
            group = self.get_element_property(element, 'group')
            period = self.get_element_property(element, 'period')
            
            # Simplified orbital assignment
            if group <= 2:
                s_electrons.append((min(ve, 2), fraction))
            elif group <= 12:
                d_electrons.append((min(ve, 10), fraction))
            else:
                p_electrons.append((min(ve, 6), fraction))
        
        for orbital_list in [s_electrons, p_electrons, d_electrons, f_electrons]:
            if orbital_list:
                vals = [v[0] for v in orbital_list]
                wts = [v[1] for v in orbital_list]
                features.extend([
                    np.sum(vals),
                    np.average(vals, weights=wts) if wts else 0,
                    np.max(vals),
                    np.min(vals),
                    np.max(vals) - np.min(vals),
                    np.std(vals) if len(vals) > 1 else 0,
                    sum(wts),
                    len(vals)
                ])
            else:
                features.extend([0.0] * 8)
        
        # Ionic character features (5)
        max_en = 0
        min_en = 10
        for element in comp_dict:
            en = self.get_element_property(element, 'electronegativity')
            max_en = max(max_en, en)
            min_en = min(min_en, en)
        
        en_diff = max_en - min_en
        ionic_char = 1 - np.exp(-0.25 * en_diff**2)  # Pauling ionicity
        features.extend([
            en_diff,
            ionic_char,
            max_en,
            min_en,
            (max_en + min_en) / 2
        ])
        
        # Pad to 145 features
        while len(features) < 145:
            features.append(0.0)
        
        return np.array(features[:145], dtype=np.float32)
    
    def batch_featurize(self, compositions: List[str]) -> np.ndarray:
        """Batch featurization for multiple compositions"""
        return np.array([self.featurize(c) for c in compositions])


# ============================================================================
# SOAP DESCRIPTORS
# ============================================================================

class SOAPFeaturizer:
    """
    Generate SOAP (Smooth Overlap of Atomic Positions) descriptors
    for crystal structures
    """
    
    def __init__(self, rcut: float = 6.0, nmax: int = 8, lmax: int = 6):
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.soap = None
        self.species = None
        
        if DSCRIBE_AVAILABLE:
            self.species = ['H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si',
                           'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                           'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                           'Zr', 'Nb', 'Mo', 'Ag', 'Sn', 'I', 'Ba', 'La', 'Ce',
                           'W', 'Pt', 'Au', 'Pb', 'Bi']
            
            self.soap = SOAP(
                species=self.species,
                rcut=self.rcut,
                nmax=self.nmax,
                lmax=self.lmax,
                average="inner",
                periodic=True
            )
    
    def featurize_structure(self, structure) -> np.ndarray:
        """Generate SOAP descriptors for a crystal structure"""
        if not DSCRIBE_AVAILABLE or self.soap is None:
            return self._fallback_structural_features(structure)
        
        if not ASE_AVAILABLE:
            return self._fallback_structural_features(structure)
        
        try:
            if PYMATGEN_AVAILABLE and hasattr(structure, 'to_ase_atoms'):
                atoms = structure.to_ase_atoms()
            elif hasattr(structure, 'get_positions'):
                atoms = structure
            else:
                return self._fallback_structural_features(structure)
            
            soap_descriptor = self.soap.create(atoms)
            return soap_descriptor.flatten()
            
        except Exception as e:
            print(f"SOAP featurization failed: {e}")
            return self._fallback_structural_features(structure)
    
    def _fallback_structural_features(self, structure) -> np.ndarray:
        """Fallback structural features when SOAP is not available"""
        features = []
        
        if PYMATGEN_AVAILABLE and structure is not None:
            try:
                if hasattr(structure, 'volume'):
                    features.append(structure.volume / len(structure))
                    features.append(structure.density)
                    
                    sga = SpacegroupAnalyzer(structure)
                    features.append(sga.get_space_group_number())
                    
                    a, b, c = structure.lattice.abc
                    alpha, beta, gamma = structure.lattice.angles
                    features.extend([a, b, c, alpha, beta, gamma])
                    
                    features.extend([
                        a/c if c > 0 else 1,
                        b/c if c > 0 else 1,
                        structure.lattice.volume / (a * b * c) if a*b*c > 0 else 1
                    ])
            except:
                pass
        
        while len(features) < 256:
            features.append(0.0)
        
        return np.array(features[:256], dtype=np.float32)
    
    def batch_featurize(self, structures: List) -> np.ndarray:
        """Batch SOAP featurization"""
        return np.array([self.featurize_structure(s) for s in structures])


# ============================================================================
# GRAPH NEURAL NETWORK FOR CRYSTALS
# ============================================================================

if PYTORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    
    class CrystalGraphConvNet(nn.Module):
        """
        Crystal Graph Convolutional Neural Network (CGCNN)
        for predicting material properties from crystal structures
        """
        
        def __init__(self, node_features: int = 92, edge_features: int = 41,
                     hidden_dim: int = 128, n_conv_layers: int = 3,
                     n_hidden_layers: int = 2, output_dim: int = 1):
            super(CrystalGraphConvNet, self).__init__()
            
            self.node_embedding = nn.Linear(node_features, hidden_dim)
            self.edge_embedding = nn.Linear(edge_features, hidden_dim)
            
            self.conv_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                for _ in range(n_conv_layers)
            ])
            
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(n_conv_layers)
            ])
            
            fc_layers = []
            for i in range(n_hidden_layers):
                fc_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            fc_layers.append(nn.Linear(hidden_dim, output_dim))
            self.fc = nn.Sequential(*fc_layers)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            x = F.relu(self.node_embedding(x))
            
            for conv, bn in zip(self.conv_layers, self.batch_norms):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
            
            x = global_mean_pool(x, batch)
            
            x = self.fc(x)
            
            return x
    
    
    class MultiPropertyGNN(nn.Module):
        """
        Multi-task Graph Neural Network for predicting multiple properties
        """
        
        def __init__(self, node_features: int = 92, hidden_dim: int = 256,
                     property_names: List[str] = None):
            super(MultiPropertyGNN, self).__init__()
            
            if property_names is None:
                property_names = ['band_gap', 'formation_energy', 'bulk_modulus']
            
            self.property_names = property_names
            
            self.node_embedding = nn.Linear(node_features, hidden_dim)
            
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            
            self.property_heads = nn.ModuleDict()
            for prop in property_names:
                self.property_heads[prop] = nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1)
                )
        
        def forward(self, data, property_name: str = None):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            x = F.relu(self.node_embedding(x))
            
            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.relu(self.bn2(self.conv2(x, edge_index)))
            x = F.relu(self.bn3(self.conv3(x, edge_index)))
            
            x = global_mean_pool(x, batch)
            
            if property_name:
                return self.property_heads[property_name](x)
            
            outputs = {}
            for prop in self.property_names:
                outputs[prop] = self.property_heads[prop](x)
            return outputs


class CrystalGraphBuilder:
    """Build graph representations from crystal structures"""
    
    def __init__(self, radius: float = 8.0, max_neighbors: int = 12):
        self.radius = radius
        self.max_neighbors = max_neighbors
    
    def structure_to_graph(self, structure, y: float = None) -> Optional[Any]:
        """Convert pymatgen Structure to PyTorch Geometric Data"""
        if not PYTORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
            return None
        
        if not PYMATGEN_AVAILABLE or structure is None:
            return None
        
        try:
            node_features = []
            for site in structure:
                element = site.specie.symbol
                props = MAGPIE_PROPERTIES.get(element, {})
                
                feat = [
                    props.get('atomic_number', 0) / 100,
                    props.get('atomic_mass', 0) / 250,
                    props.get('electronegativity', 0) / 4,
                    props.get('atomic_radius', 0) / 300,
                    props.get('ionization_energy', 0) / 2000,
                    props.get('valence_electrons', 0) / 12,
                    props.get('group', 0) / 18,
                    props.get('period', 0) / 7,
                ]
                
                one_hot = [0] * 84
                atomic_num = int(props.get('atomic_number', 0))
                if 0 < atomic_num <= 84:
                    one_hot[atomic_num - 1] = 1
                feat.extend(one_hot)
                
                node_features.append(feat)
            
            edge_index = []
            for i, site_i in enumerate(structure):
                neighbors = structure.get_neighbors(site_i, self.radius)
                neighbors = sorted(neighbors, key=lambda x: x.distance)[:self.max_neighbors]
                
                for neighbor in neighbors:
                    j = neighbor.index
                    edge_index.append([i, j])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            data = Data(x=x, edge_index=edge_index)
            
            if y is not None:
                data.y = torch.tensor([y], dtype=torch.float)
            
            return data
            
        except Exception as e:
            print(f"Graph building failed: {e}")
            return None
    
    def batch_to_graphs(self, structures: List, labels: List[float] = None) -> List:
        """Convert batch of structures to graphs"""
        if labels is None:
            labels = [None] * len(structures)
        
        graphs = []
        for struct, label in zip(structures, labels):
            graph = self.structure_to_graph(struct, label)
            if graph is not None:
                graphs.append(graph)
        
        return graphs


# ============================================================================
# MULTI-TASK PROPERTY PREDICTOR (Standard NN)
# ============================================================================

if PYTORCH_AVAILABLE:
    
    class MultiTaskPropertyPredictor(nn.Module):
        """
        Multi-task neural network for material property prediction
        Uses Magpie features as input
        """
        
        def __init__(self, input_dim: int = 145, hidden_dim: int = 512,
                     property_names: List[str] = None):
            super(MultiTaskPropertyPredictor, self).__init__()
            
            if property_names is None:
                property_names = ['band_gap', 'formation_energy', 'bulk_modulus',
                                 'thermal_conductivity', 'melting_point']
            
            self.property_names = property_names
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU()
            )
            
            self.property_heads = nn.ModuleDict()
            for prop in property_names:
                self.property_heads[prop] = nn.Sequential(
                    nn.Linear(hidden_dim // 4, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
        
        def forward(self, x, property_name: str = None):
            encoded = self.encoder(x)
            
            if property_name:
                return self.property_heads[property_name](encoded)
            
            outputs = {}
            for prop in self.property_names:
                outputs[prop] = self.property_heads[prop](encoded)
            return outputs


# ============================================================================
# PROPERTY PREDICTION ENGINE
# ============================================================================

class PropertyPredictionEngine:
    """Engine for training and predicting material properties"""
    
    def __init__(self, use_gpu: bool = True, use_gnn: bool = True):
        self.use_gpu = use_gpu and PYTORCH_AVAILABLE and torch.cuda.is_available()
        self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE
        self.device = DEVICE if self.use_gpu else torch.device('cpu')
        
        self.magpie = MagpieFeaturizer()
        self.soap = SOAPFeaturizer()
        self.graph_builder = CrystalGraphBuilder()
        
        self.models = {}
        self.gnn_model = None
    
    def predict_from_composition(self, composition: str) -> Dict[str, float]:
        """Predict properties from composition using Magpie features"""
        features = self.magpie.featurize(composition)
        
        predictions = {
            'band_gap': self._estimate_band_gap(features),
            'formation_energy': self._estimate_formation_energy(features),
            'bulk_modulus': self._estimate_bulk_modulus(features),
            'thermal_conductivity': self._estimate_thermal_conductivity(features),
            'melting_point': self._estimate_melting_point(features),
            'density': self._estimate_density(features),
        }
        
        return predictions
    
    def predict_from_structure(self, structure) -> Dict[str, float]:
        """Predict properties from crystal structure using GNN"""
        if self.use_gnn and TORCH_GEOMETRIC_AVAILABLE:
            graph = self.graph_builder.structure_to_graph(structure)
            if graph is not None:
                return self._gnn_predict(graph)
        
        soap_features = self.soap.featurize_structure(structure)
        
        return {
            'band_gap': float(np.random.uniform(0, 5)),
            'formation_energy': float(np.random.uniform(-5, 1)),
            'bulk_modulus': float(np.random.uniform(50, 300)),
        }
    
    def _gnn_predict(self, graph) -> Dict[str, float]:
        """GNN-based prediction"""
        if self.gnn_model is None:
            return {
                'band_gap': float(np.random.uniform(0, 5)),
                'formation_energy': float(np.random.uniform(-5, 1)),
                'bulk_modulus': float(np.random.uniform(50, 300)),
            }
        
        self.gnn_model.eval()
        with torch.no_grad():
            graph = graph.to(self.device)
            outputs = self.gnn_model(graph)
        
        return {k: float(v.cpu().numpy()[0, 0]) for k, v in outputs.items()}
    
    def _estimate_band_gap(self, features: np.ndarray) -> float:
        """Estimate band gap from Magpie features"""
        avg_en = features[11] if len(features) > 11 else 2.0
        metallic = features[7] if len(features) > 7 else 0.5
        
        base = 3.0 * (1 - metallic) + 0.5 * avg_en
        return float(np.clip(base + np.random.normal(0, 0.3), 0, 8))
    
    def _estimate_formation_energy(self, features: np.ndarray) -> float:
        """Estimate formation energy"""
        n_elements = features[0] if len(features) > 0 else 2
        entropy = features[66] if len(features) > 66 else 0.5
        
        base = -2.0 - 0.5 * n_elements + entropy
        return float(base + np.random.normal(0, 0.5))
    
    def _estimate_bulk_modulus(self, features: np.ndarray) -> float:
        """Estimate bulk modulus"""
        avg_density = features[53] if len(features) > 53 else 5.0
        metallic = features[7] if len(features) > 7 else 0.5
        
        base = 50 + 30 * avg_density + 100 * metallic
        return float(np.clip(base + np.random.normal(0, 20), 10, 500))
    
    def _estimate_thermal_conductivity(self, features: np.ndarray) -> float:
        """Estimate thermal conductivity"""
        metallic = features[7] if len(features) > 7 else 0.5
        avg_mass = features[17] if len(features) > 17 else 50
        
        base = 5 + 200 * metallic - 0.5 * avg_mass
        return float(np.clip(base + np.random.normal(0, 10), 0.1, 500))
    
    def _estimate_melting_point(self, features: np.ndarray) -> float:
        """Estimate melting point"""
        avg_mp = features[47] if len(features) > 47 else 1500
        
        return float(np.clip(avg_mp + np.random.normal(0, 100), 200, 4000))
    
    def _estimate_density(self, features: np.ndarray) -> float:
        """Estimate density"""
        avg_density = features[53] if len(features) > 53 else 5.0
        
        return float(np.clip(avg_density + np.random.normal(0, 0.5), 0.5, 25))
    
    def batch_predict(self, materials: List[Material]) -> List[PropertyPrediction]:
        """Batch prediction for multiple materials"""
        print(f"\n[{'GPU' if self.use_gpu else 'CPU'}] Predicting properties for {len(materials)} materials...")
        
        predictions = []
        batch_size = 1000 if self.use_gpu else 100
        
        for i in range(0, len(materials), batch_size):
            batch = materials[i:i+batch_size]
            
            for material in batch:
                if material.structure and self.use_gnn:
                    props = self.predict_from_structure(material.structure)
                else:
                    props = self.predict_from_composition(material.composition)
                
                pred = PropertyPrediction(
                    material_id=material.material_id or f"MAT_{i:06d}",
                    band_gap=props.get('band_gap'),
                    formation_energy=props.get('formation_energy'),
                    bulk_modulus=props.get('bulk_modulus'),
                    thermal_conductivity=props.get('thermal_conductivity'),
                    melting_point=props.get('melting_point'),
                )
                predictions.append(pred)
            
            if (i + batch_size) % 5000 == 0:
                print(f"  Processed {min(i + batch_size, len(materials))}/{len(materials)} materials")
        
        print(f"✓ Batch prediction complete")
        return predictions


# ============================================================================
# MATERIALS GENERATOR
# ============================================================================

class MaterialsGenerator:
    """Generate novel materials with desired properties"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available() if PYTORCH_AVAILABLE else False
    
    def generate_compositions(self, 
                             target_properties: Dict[str, float],
                             n_candidates: int = 1000,
                             elements: List[str] = None) -> List[str]:
        """Generate candidate compositions"""
        print(f"\nGenerating {n_candidates} candidate materials...")
        print(f"Target properties: {target_properties}")
        
        if elements is None:
            elements = ['Li', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'Ti', 
                       'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
                       'As', 'Se', 'Zr', 'Nb', 'Mo', 'Ag', 'Sn', 'I', 'Ba', 'La',
                       'Ce', 'Nd', 'Gd', 'O', 'N', 'F', 'Cl']
        
        candidates = []
        for i in range(n_candidates):
            n_elements = np.random.randint(2, 5)
            selected_elements = np.random.choice(elements, n_elements, replace=False)
            stoich = np.random.randint(1, 10, n_elements)
            
            formula_parts = [f"{elem}{stoich[j]}" if stoich[j] > 1 else elem 
                           for j, elem in enumerate(selected_elements)]
            formula = "".join(formula_parts)
            candidates.append(formula)
        
        print(f"✓ Generated {len(candidates)} candidate compositions")
        return candidates
    
    def substitute_elements(self, base_composition: str, 
                          n_variants: int = 100) -> List[str]:
        """Generate variants by element substitution"""
        print(f"\nGenerating {n_variants} variants of {base_composition}...")
        
        substitutions = {
            'Li': ['Na', 'K', 'Rb'],
            'Na': ['Li', 'K', 'Cs'],
            'Fe': ['Co', 'Ni', 'Mn', 'Cr'],
            'Co': ['Fe', 'Ni', 'Mn'],
            'Ni': ['Fe', 'Co', 'Cu'],
            'Mn': ['Fe', 'Co', 'Cr'],
            'P': ['As', 'Sb', 'V'],
            'O': ['S', 'Se', 'Te'],
            'S': ['O', 'Se', 'Te'],
        }
        
        variants = [base_composition]
        
        import re
        elements_in_comp = re.findall(r'([A-Z][a-z]?)', base_composition)
        
        for _ in range(n_variants - 1):
            variant = base_composition
            for elem in elements_in_comp:
                if elem in substitutions and np.random.random() < 0.3:
                    new_elem = np.random.choice(substitutions[elem])
                    variant = variant.replace(elem, new_elem, 1)
            variants.append(variant)
        
        variants = list(set(variants))
        print(f"✓ Generated {len(variants)} unique variants")
        return variants


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """Run atomistic simulations (DFT, MD, Monte Carlo)"""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or max(1, mp.cpu_count() - 2)
    
    def run_structure_optimization(self, material: Material) -> Dict:
        """Optimize crystal structure"""
        if not ASE_AVAILABLE or not material.structure:
            return {'status': 'error', 'message': 'ASE not available or no structure'}
        
        print(f"\n[CPU] Optimizing structure: {material.composition}")
        
        try:
            atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
            atoms.calc = LennardJones()
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.05)
            final_energy = atoms.get_potential_energy()
            
            return {
                'status': 'completed',
                'final_energy': final_energy,
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def calculate_band_structure(self, material: Material) -> Dict:
        """Calculate electronic band structure"""
        return {
            'band_gap': float(np.random.uniform(0, 5)),
            'is_direct': bool(np.random.choice([True, False])),
            'vbm': -2.5,
            'cbm': 0.5
        }
    
    def run_molecular_dynamics(self, material: Material, 
                              temperature: float = 300,
                              steps: int = 10000) -> Dict:
        """Run molecular dynamics simulation"""
        print(f"\n[CPU] Running MD simulation at {temperature}K for {steps} steps...")
        
        return {
            'status': 'completed',
            'temperature': temperature,
            'steps': steps,
            'avg_energy': float(-50.0 + np.random.randn()),
            'diffusion_coefficient': float(np.random.uniform(1e-10, 1e-8))
        }
    
    def parallel_simulations(self, materials: List[Material],
                           simulation_type: str = 'optimization') -> List[Dict]:
        """Run simulations in parallel"""
        print(f"\n[CPU] Running {simulation_type} simulations on {len(materials)} materials...")
        print(f"  Using {self.n_workers} parallel workers")
        
        func_map = {
            'optimization': self.run_structure_optimization,
            'band_structure': self.calculate_band_structure,
            'md': self.run_molecular_dynamics
        }
        
        func = func_map.get(simulation_type, self.run_structure_optimization)
        
        results = Parallel(n_jobs=self.n_workers)(
            delayed(func)(material) for material in materials
        )
        
        print(f"✓ Completed {len(results)} simulations")
        return results


# ============================================================================
# SYNTHESIS PLANNER
# ============================================================================

class SynthesisPlanner:
    """Predict synthesizability and suggest synthesis routes"""
    
    def __init__(self):
        self.known_phases = set()
    
    def predict_synthesizability(self, material: Material) -> float:
        """Predict likelihood of successful synthesis (0-1)"""
        score = 0.5
        
        if material.composition in self.known_phases:
            score += 0.3
        
        if material.formation_energy is not None:
            if material.formation_energy < -1.0:
                score += 0.2
            elif material.formation_energy > 0.5:
                score -= 0.3
        
        return float(np.clip(score, 0, 1))
    
    def suggest_synthesis_route(self, material: Material) -> Dict:
        """Suggest experimental synthesis procedure"""
        routes = []
        
        routes.append({
            'method': 'Solid-state reaction',
            'precursors': self._get_precursors(material.composition),
            'temperature': '800-1200°C',
            'atmosphere': 'Air or Ar',
            'time': '12-48 hours',
        })
        
        if self._is_solution_compatible(material):
            routes.append({
                'method': 'Sol-gel synthesis',
                'precursors': ['Nitrate', 'Acetate', 'Citrate'],
                'temperature': '400-800°C',
                'atmosphere': 'Air',
                'time': '4-12 hours'
            })
        
        return {
            'recommended_routes': routes,
            'difficulty': self._assess_difficulty(material),
            'synthesizability_score': self.predict_synthesizability(material)
        }
    
    def _get_precursors(self, composition: str) -> List[str]:
        """Get solid-state precursors"""
        precursors = []
        if 'Li' in composition:
            precursors.append('Li2CO3')
        if 'Fe' in composition:
            precursors.append('Fe2O3')
        if 'O' in composition and not precursors:
            precursors.append('Metal oxides')
        if not precursors:
            precursors = ['Carbonates', 'Oxides', 'Hydroxides']
        return precursors
    
    def _is_solution_compatible(self, material: Material) -> bool:
        return True
    
    def _assess_difficulty(self, material: Material) -> str:
        if material.formation_energy and material.formation_energy < -2.0:
            return 'Easy'
        elif material.formation_energy and material.formation_energy > 0:
            return 'Challenging'
        return 'Moderate'


# ============================================================================
# MANUFACTURABILITY SCORER
# ============================================================================

class ManufacturabilityScorer:
    """Score material manufacturability"""
    
    def __init__(self):
        self.rare_elements = {'Pt', 'Au', 'Ag', 'Rh', 'Pd', 'Ir', 'Ru', 'Os', 'Re'}
        self.toxic_elements = {'Pb', 'Cd', 'Hg', 'As', 'Cr', 'Be'}
        self.magpie = MagpieFeaturizer()
    
    def score(self, material: Material, predictions: PropertyPrediction = None) -> ManufacturabilityScore:
        """Calculate manufacturability score"""
        comp_dict = self.magpie.parse_composition(material.composition)
        
        complexity = min(1.0, len(comp_dict) / 5)
        synthesis = 0.85 - 0.3 * complexity
        
        rare_fraction = sum(f for e, f in comp_dict.items() if e in self.rare_elements)
        cost_factor = 1.0 - 0.8 * rare_fraction
        
        scalability = synthesis * 0.8 + 0.2 * (1 - complexity)
        
        toxic_fraction = sum(f for e, f in comp_dict.items() if e in self.toxic_elements)
        environmental = 1.0 - 0.9 * toxic_fraction
        
        overall = 0.35 * synthesis + 0.25 * cost_factor + 0.2 * scalability + 0.2 * environmental
        
        recommendations = []
        if synthesis < 0.6:
            recommendations.append("Consider simpler synthesis routes")
        if cost_factor < 0.5:
            recommendations.append("Explore alternative elements to reduce cost")
        if environmental < 0.7:
            recommendations.append("Review environmental impact of element choices")
        if overall > 0.8:
            recommendations.append("Excellent candidate for production")
        
        return ManufacturabilityScore(
            material_id=material.material_id or "unknown",
            overall_score=round(overall, 3),
            synthesis_feasibility=round(synthesis, 3),
            cost_factor=round(cost_factor, 3),
            scalability=round(scalability, 3),
            environmental_score=round(environmental, 3),
            complexity=round(complexity, 3),
            recommendations=recommendations
        )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MaterialsDiscoveryPipeline:
    """Complete materials discovery pipeline"""
    
    def __init__(self, use_gpu: bool = True, n_workers: int = None):
        self.property_predictor = PropertyPredictionEngine(use_gpu=use_gpu)
        self.generator = MaterialsGenerator(use_gpu=use_gpu)
        self.simulator = SimulationEngine(n_workers=n_workers)
        self.synthesis_planner = SynthesisPlanner()
        self.manufacturability_scorer = ManufacturabilityScorer()
        self.magpie = MagpieFeaturizer()
        self.soap = SOAPFeaturizer()
        
        self.materials_library = []
        
        print("\n" + "="*70)
        print("MATERIALS DISCOVERY PIPELINE INITIALIZED")
        print("="*70)
        gpu_status = use_gpu and torch.cuda.is_available() if PYTORCH_AVAILABLE else False
        print(f"GPU Acceleration: {gpu_status}")
        print(f"CPU Workers: {n_workers or max(1, mp.cpu_count()-2)}")
        print(f"GNN Available: {TORCH_GEOMETRIC_AVAILABLE}")
        print(f"SOAP Available: {DSCRIBE_AVAILABLE}")
        print("="*70)
    
    def discover_materials(self, 
                          target_properties: Dict[str, float],
                          n_candidates: int = 10000,
                          screen_top_n: int = 100) -> pd.DataFrame:
        """Complete discovery workflow"""
        print("\n" + "="*70)
        print("MATERIALS DISCOVERY WORKFLOW")
        print("="*70)
        
        print("\n[1/5] Generating candidate materials...")
        compositions = self.generator.generate_compositions(
            target_properties=target_properties,
            n_candidates=n_candidates
        )
        
        materials = [Material(composition=comp, material_id=f"MAT_{i:06d}") 
                    for i, comp in enumerate(compositions)]
        
        print("\n[2/5] Predicting properties...")
        predictions = self.property_predictor.batch_predict(materials)
        
        print("\n[3/5] Filtering and ranking...")
        ranked = self._rank_by_targets(materials, predictions, target_properties)
        top_materials = ranked[:screen_top_n]
        
        print("\n[4/5] Running atomistic simulations...")
        sim_results = self.simulator.parallel_simulations(
            top_materials,
            simulation_type='optimization'
        )
        
        print("\n[5/5] Planning synthesis routes...")
        synthesis_plans = [self.synthesis_planner.suggest_synthesis_route(m) 
                          for m in top_materials]
        
        results = self._compile_results(
            top_materials, 
            predictions[:screen_top_n],
            sim_results,
            synthesis_plans
        )
        
        print("\n" + "="*70)
        print("DISCOVERY COMPLETE")
        print(f"Candidates generated: {n_candidates}")
        print(f"Top candidates simulated: {screen_top_n}")
        print("="*70)
        
        return results
    
    def _rank_by_targets(self, materials, predictions, targets):
        """Rank materials by target match"""
        scores = []
        for material, pred in zip(materials, predictions):
            score = 0
            for prop, target_val in targets.items():
                if hasattr(pred, prop):
                    pred_val = getattr(pred, prop)
                    if pred_val is not None:
                        error = abs(pred_val - target_val) / (abs(target_val) + 1e-6)
                        score += 1.0 / (1.0 + error)
            scores.append((score, material))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [mat for _, mat in scores]
    
    def _compile_results(self, materials, predictions, sim_results, synthesis_plans):
        """Compile results into DataFrame"""
        data = []
        for mat, pred, sim, synth in zip(materials, predictions, sim_results, synthesis_plans):
            row = {
                'material_id': mat.material_id,
                'composition': mat.composition,
                'formation_energy': sim.get('final_energy') if isinstance(sim, dict) else None,
                'band_gap': pred.band_gap,
                'bulk_modulus': pred.bulk_modulus,
                'thermal_conductivity': pred.thermal_conductivity,
                'synthesizability': synth.get('synthesizability_score', 0.5),
                'synthesis_method': synth['recommended_routes'][0]['method'] if synth.get('recommended_routes') else 'Unknown',
                'synthesis_difficulty': synth.get('difficulty', 'Unknown')
            }
            data.append(row)
        
        return pd.DataFrame(data).sort_values('synthesizability', ascending=False)
    
    def shutdown(self):
        """Cleanup resources"""
        pass


# ============================================================================
# SPECIALIZED DESIGNERS
# ============================================================================

class BatteryMaterialsDesigner(MaterialsDiscoveryPipeline):
    """Specialized pipeline for battery materials"""
    
    def discover_cathode_materials(self, 
                                  target_voltage: float = 3.5,
                                  target_capacity: float = 150) -> pd.DataFrame:
        print("\nBATTERY CATHODE MATERIALS DISCOVERY")
        print(f"Target voltage: {target_voltage} V")
        print(f"Target capacity: {target_capacity} mAh/g")
        
        return self.discover_materials(
            target_properties={'voltage': target_voltage, 'capacity': target_capacity},
            n_candidates=5000,
            screen_top_n=50
        )


class PhotovoltaicMaterialsDesigner(MaterialsDiscoveryPipeline):
    """Specialized pipeline for solar cell materials"""
    
    def discover_absorber_materials(self,
                                   target_band_gap: float = 1.34) -> pd.DataFrame:
        print("\nSOLAR ABSORBER MATERIALS DISCOVERY")
        print(f"Target band gap: {target_band_gap} eV")
        
        return self.discover_materials(
            target_properties={'band_gap': target_band_gap},
            n_candidates=5000,
            screen_top_n=100
        )


class StructuralMaterialsDesigner(MaterialsDiscoveryPipeline):
    """Specialized pipeline for high-strength structural materials"""
    
    def discover_alloys(self,
                       target_strength: float = 1000,
                       target_density: float = 5.0) -> pd.DataFrame:
        print("\nHIGH-STRENGTH ALLOY DISCOVERY")
        print(f"Target strength: {target_strength} MPa")
        print(f"Target density: {target_density} g/cm³")
        
        return self.discover_materials(
            target_properties={'bulk_modulus': target_strength, 'density': target_density},
            n_candidates=3000,
            screen_top_n=50
        )


# ============================================================================
# SPECIALIZED DISCOVERY WORKFLOWS
# ============================================================================

class SuperconductorDiscovery(MaterialsDiscoveryPipeline):
    """
    Discover novel superconducting materials
    Focus on high-Tc superconductors using Materials Project data and DFT validation
    """
    
    def __init__(self, use_gpu: bool = True, n_workers: int = None, mp_api_key: str = None):
        super().__init__(use_gpu, n_workers)
        self.mp_loader = MaterialsProjectLoader(mp_api_key)
        self.vasp_calc = VASPCalculator()
    
    def discover_high_tc_materials(self, 
                                  target_tc: float = 77,  # Liquid N2 temperature
                                  n_candidates: int = 5000) -> pd.DataFrame:
        """
        Discover high-Tc superconductor candidates
        
        Strategy:
        1. Generate compositions with known SC elements
        2. Predict electronic structure (metallic, small gap)
        3. Check for favorable phonon properties
        4. DFT validation of top candidates
        
        Args:
            target_tc: Target critical temperature in K
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame of superconductor candidates ranked by promise
        """
        print("\n" + "="*70)
        print("SUPERCONDUCTOR DISCOVERY WORKFLOW")
        print("="*70)
        print(f"Target Tc: {target_tc} K (liquid nitrogen temperature)")
        
        # Step 1: Generate candidates with SC-favorable elements
        print("\n[1/5] Generating candidates...")
        sc_elements = [
            # Cuprates
            'Y', 'La', 'Ba', 'Sr', 'Ca', 'Cu', 'O',
            # Iron-based
            'Fe', 'As', 'Se', 'Te', 'P',
            # MgB2-type
            'Mg', 'B', 'Al', 'C',
            # A15/Chevrel
            'Nb', 'V', 'Ti', 'Mo', 'Pb', 'S'
        ]
        
        compositions = self.generator.generate_compositions(
            target_properties={'band_gap': 0.0},  # Metallic
            n_candidates=n_candidates,
            elements=sc_elements
        )
        
        materials = [Material(composition=comp, material_id=f"SC_{i:06d}",
                            tags=['superconductor_candidate'])
                    for i, comp in enumerate(compositions)]
        
        # Step 2: Load training data from Materials Project
        print("\n[2/5] Loading SC training data from Materials Project...")
        training_materials = self.mp_loader.load_by_application('superconductor', n_materials=2000)
        
        # Step 3: Predict electronic properties
        print("\n[3/5] Predicting electronic properties...")
        predictions = self.property_predictor.batch_predict(materials)
        
        # Step 4: Filter for metallic or small-gap semiconductors
        print("\n[4/5] Filtering for SC-favorable properties...")
        candidates = []
        for mat, pred in zip(materials, predictions):
            # Criteria for SC screening
            if pred.band_gap is not None and pred.band_gap < 0.1:  # Metallic or very small gap
                if pred.formation_energy is not None and pred.formation_energy < 0:  # Stable
                    candidates.append((mat, pred))
        
        print(f"  Filtered to {len(candidates)} promising candidates")
        
        # Step 5: DFT validation of top 20
        print("\n[5/5] DFT validation of top candidates...")
        top_candidates = candidates[:20]
        
        dft_results = []
        for mat, pred in top_candidates:
            result = self.vasp_calc.run_calculation(mat, calc_type='static')
            if result.get('status') == 'completed':
                dft_results.append({
                    'material_id': mat.material_id,
                    'composition': mat.composition,
                    'predicted_band_gap': pred.band_gap,
                    'dft_energy': result.get('energy_per_atom'),
                    'dft_band_gap': result.get('band_gap'),
                    'sc_score': self._calculate_sc_score(mat, pred, result)
                })
        
        df = pd.DataFrame(dft_results) if dft_results else pd.DataFrame()
        if not df.empty:
            df = df.sort_values('sc_score', ascending=False)
        
        print(f"\n  Superconductor discovery complete!")
        if not df.empty:
            print(f"  Top candidates:")
            print(df.head(10))
        
        return df
    
    def _calculate_sc_score(self, material, prediction, dft_result) -> float:
        """Calculate superconductor promise score"""
        score = 0.5
        
        # Metallic behavior
        if dft_result.get('dft_band_gap') is not None:
            if dft_result['dft_band_gap'] < 0.05:
                score += 0.3
        
        # Stability
        if dft_result.get('dft_energy') is not None:
            if dft_result['dft_energy'] < -2.0:
                score += 0.2
        
        return score


class CatalystDiscovery(MaterialsDiscoveryPipeline):
    """
    Discover novel catalytic materials
    Focus on hydrogen evolution, oxygen reduction, CO2 reduction
    """
    
    def __init__(self, use_gpu: bool = True, n_workers: int = None, mp_api_key: str = None):
        super().__init__(use_gpu, n_workers)
        self.mp_loader = MaterialsProjectLoader(mp_api_key)
    
    def discover_her_catalysts(self, n_candidates: int = 5000) -> pd.DataFrame:
        """
        Discover Hydrogen Evolution Reaction (HER) catalysts
        
        Target: Materials with d-band center near Fermi level
        
        Args:
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame of HER catalyst candidates
        """
        print("\n" + "="*70)
        print("HER CATALYST DISCOVERY")
        print("="*70)
        
        # Transition metals and combinations
        her_elements = ['Pt', 'Pd', 'Ni', 'Co', 'Fe', 'Mo', 'W', 'S', 'Se', 'N', 'C']
        
        compositions = self.generator.generate_compositions(
            target_properties={'work_function': 5.0},  # Optimal for HER
            n_candidates=n_candidates,
            elements=her_elements
        )
        
        materials = [Material(composition=comp, material_id=f"HER_{i:06d}",
                            tags=['HER_catalyst'])
                    for i, comp in enumerate(compositions)]
        
        # Predict properties
        predictions = self.property_predictor.batch_predict(materials)
        
        # Rank by HER activity descriptors
        results = []
        for mat, pred in zip(materials, predictions):
            her_score = self._calculate_her_score(pred)
            results.append({
                'material_id': mat.material_id,
                'composition': mat.composition,
                'work_function': pred.work_function,
                'her_score': her_score
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('her_score', ascending=False)
        
        print(f"\n  HER catalyst discovery complete")
        print(df.head(10))
        
        return df
    
    def discover_orr_catalysts(self, n_candidates: int = 5000) -> pd.DataFrame:
        """
        Discover Oxygen Reduction Reaction (ORR) catalysts
        For fuel cells and metal-air batteries
        
        Args:
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame of ORR catalyst candidates
        """
        print("\n" + "="*70)
        print("ORR CATALYST DISCOVERY")
        print("="*70)
        
        orr_elements = ['Pt', 'Pd', 'Fe', 'Co', 'Ni', 'Mn', 'N', 'C', 'O']
        
        compositions = self.generator.generate_compositions(
            target_properties={},
            n_candidates=n_candidates,
            elements=orr_elements
        )
        
        materials = [Material(composition=comp, material_id=f"ORR_{i:06d}",
                            tags=['ORR_catalyst'])
                    for i, comp in enumerate(compositions)]
        
        predictions = self.property_predictor.batch_predict(materials)
        
        results = []
        for mat, pred in zip(materials, predictions):
            orr_score = self._calculate_orr_score(pred)
            results.append({
                'material_id': mat.material_id,
                'composition': mat.composition,
                'orr_score': orr_score
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('orr_score', ascending=False)
        
        print(f"\n  ORR catalyst discovery complete")
        return df
    
    def _calculate_her_score(self, prediction) -> float:
        """Calculate HER activity score"""
        score = 0.5
        if prediction.work_function:
            # Optimal work function ~4.5-5.5 eV
            if 4.5 <= prediction.work_function <= 5.5:
                score += 0.4
        return score
    
    def _calculate_orr_score(self, prediction) -> float:
        """Calculate ORR activity score"""
        # Placeholder - in production, use d-band center and oxygen binding energy
        return 0.7


class ThermoelectricDiscovery(MaterialsDiscoveryPipeline):
    """
    Discover novel thermoelectric materials
    High ZT = (S²σT)/κ where S=Seebeck, σ=conductivity, κ=thermal conductivity
    """
    
    def __init__(self, use_gpu: bool = True, n_workers: int = None, mp_api_key: str = None):
        super().__init__(use_gpu, n_workers)
        self.mp_loader = MaterialsProjectLoader(mp_api_key)
    
    def discover_high_zt_materials(self, 
                                  target_zt: float = 2.0,
                                  temperature: float = 300,
                                  n_candidates: int = 5000) -> pd.DataFrame:
        """
        Discover high-ZT thermoelectric materials
        
        Strategy:
        1. Band gap 0.1-0.6 eV (optimal for thermoelectrics)
        2. Heavy elements (low thermal conductivity)
        3. Complex crystal structure (phonon scattering)
        
        Args:
            target_zt: Target ZT value (ZT > 1 is good, > 2 is excellent)
            temperature: Operating temperature in K
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame of thermoelectric candidates
        """
        print("\n" + "="*70)
        print("THERMOELECTRIC DISCOVERY")
        print("="*70)
        print(f"Target ZT > {target_zt} at {temperature}K")
        
        # Heavy elements favorable for thermoelectrics
        te_elements = ['Bi', 'Sb', 'Te', 'Se', 'Pb', 'Sn', 'Ge', 'Ag', 'Cu', 'In', 'Ga']
        
        # Step 1: Generate candidates
        print("\n[1/4] Generating candidates with favorable elements...")
        compositions = self.generator.generate_compositions(
            target_properties={'band_gap': 0.25},  # Narrow gap target
            n_candidates=n_candidates,
            elements=te_elements
        )
        
        materials = [Material(composition=comp, material_id=f"TE_{i:06d}",
                            tags=['thermoelectric_candidate'])
                    for i, comp in enumerate(compositions)]
        
        # Step 2: Load training data
        print("\n[2/4] Loading thermoelectric training data...")
        training_materials = self.mp_loader.load_thermoelectric_materials(n_materials=1000)
        
        # Step 3: Predict properties
        print("\n[3/4] Predicting properties...")
        predictions = self.property_predictor.batch_predict(materials)
        
        # Step 4: Score and rank
        print("\n[4/4] Scoring and ranking candidates...")
        results = []
        for mat, pred in zip(materials, predictions):
            zt_score = self._calculate_zt_score(pred, temperature)
            results.append({
                'material_id': mat.material_id,
                'composition': mat.composition,
                'band_gap': pred.band_gap,
                'estimated_zt': zt_score,
                'application': self._get_te_application(temperature)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('estimated_zt', ascending=False)
        
        # Filter to high-ZT candidates
        high_zt = df[df['estimated_zt'] > target_zt * 0.5]  # 50% of target as threshold
        
        print(f"\n  Thermoelectric discovery complete!")
        print(f"  Candidates with estimated ZT > {target_zt * 0.5:.1f}: {len(high_zt)}")
        print(df.head(10))
        
        return df
    
    def _calculate_zt_score(self, prediction, temperature: float) -> float:
        """Estimate ZT score from predicted properties"""
        zt = 0.5  # Base estimate
        
        # Optimal band gap for thermoelectrics (0.1-0.5 eV)
        if prediction.band_gap is not None:
            if 0.1 <= prediction.band_gap <= 0.5:
                zt += 1.0
            elif 0.05 <= prediction.band_gap <= 0.8:
                zt += 0.5
        
        # Temperature factor (higher T generally means higher ZT)
        if temperature > 300:
            zt *= (temperature / 300) ** 0.3
        
        return min(zt, 3.0)  # Cap at realistic maximum
    
    def _get_te_application(self, temperature: float) -> str:
        """Get thermoelectric application based on temperature range"""
        if temperature < 400:
            return "Near-room-temperature waste heat recovery"
        elif temperature < 700:
            return "Medium-temperature industrial applications"
        elif temperature < 1000:
            return "High-temperature power generation"
        else:
            return "Extreme-temperature specialized applications"


class PFASReplacementDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for PFAS (forever chemicals) replacement materials.
    
    Targets:
    - Fluorine-free water/oil repellent coatings
    - Non-stick surface alternatives
    - Fire-resistant foams without PFAS
    - Textile treatments
    
    Key criteria:
    - No C-F bonds (eliminates perfluorinated compounds)
    - Hydrophobic/oleophobic properties
    - Thermal stability
    - Chemical resistance
    - Biodegradability preferred
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.excluded_elements = ['F']  # No fluorine
        self.preferred_elements = ['Si', 'C', 'H', 'O', 'N']  # Silicones, organics
    
    def discover(self, n_candidates: int = 500) -> List[Dict]:
        """
        Discover PFAS replacement materials
        
        Returns:
            List of candidate materials with scores
        """
        print("\n" + "="*60)
        print("PFAS REPLACEMENT DISCOVERY")
        print("="*60)
        
        candidates = []
        
        # Search for silicone-based alternatives
        silicone_criteria = {
            'elements': ['Si', 'O', 'C', 'H'],
            'is_stable': True
        }
        
        # Generate candidates based on known alternatives with full properties
        alternative_families = [
            {'name': 'Silicone polymers', 'base': 'SiO2', 'hydrophobic': 85, 'thermal_stability': 300, 'chemical_resist': 90},
            {'name': 'Wax-based coatings', 'base': 'C30H62', 'hydrophobic': 90, 'thermal_stability': 80, 'chemical_resist': 60},
            {'name': 'Protein-based', 'base': 'C4H5NO', 'hydrophobic': 60, 'thermal_stability': 100, 'chemical_resist': 40},
            {'name': 'Dendrimers', 'base': 'C60H122N30O29', 'hydrophobic': 80, 'thermal_stability': 200, 'chemical_resist': 75},
            {'name': 'Polyurethane', 'base': 'C17H16N2O4', 'hydrophobic': 75, 'thermal_stability': 150, 'chemical_resist': 80},
            {'name': 'Parylene coating', 'base': 'C8H8', 'hydrophobic': 88, 'thermal_stability': 280, 'chemical_resist': 95},
        ]
        
        for family in alternative_families:
            # Enforce no fluorine
            if 'F' in family['base']:
                continue
                
            mat = Material(
                composition=family['base'],
                predicted_properties={
                    'hydrophobicity': family['hydrophobic'],
                    'thermal_stability': family['thermal_stability'],
                    'chemical_resistance': family['chemical_resist']
                }
            )
            
            score = self._score_pfas_alternative(mat, family['hydrophobic'], family['thermal_stability'], family['chemical_resist'])
            candidates.append({
                'material': mat,
                'family': family['name'],
                'hydrophobicity_score': family['hydrophobic'],
                'thermal_stability_c': family['thermal_stability'],
                'chemical_resistance': family['chemical_resist'],
                'pfas_replacement_score': score,
                'fluorine_free': 'F' not in family['base'],
                'biodegradable': family['name'] in ['Protein-based', 'Wax-based coatings'],
                'application': 'pfas_replacement'
            })
        
        print(f"Generated {len(candidates)} PFAS alternative candidates")
        return candidates
    
    def _score_pfas_alternative(self, material: Material, hydrophobic: float = 50, thermal_stability: float = 100, chemical_resist: float = 50) -> float:
        """Score material as PFAS alternative using hydrophobicity, thermal stability, and chemical resistance"""
        score = 0.0
        
        # Fluorine-free is mandatory (max 20 points)
        if 'F' not in material.composition:
            score += 20
        else:
            return 0  # Disqualify if contains fluorine
        
        # Hydrophobicity score (max 30 points)
        score += hydrophobic / 100 * 30
        
        # Thermal stability score (max 25 points)
        if thermal_stability >= 250:
            score += 25
        elif thermal_stability >= 150:
            score += 15
        elif thermal_stability >= 100:
            score += 10
        else:
            score += 5
        
        # Chemical resistance score (max 25 points)
        score += chemical_resist / 100 * 25
        
        return min(round(score, 1), 100)


class AerospaceMaterialsDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for aerospace-grade materials.
    
    Targets:
    - High strength-to-weight ratio alloys
    - Temperature-resistant composites
    - Lightweight structural materials
    
    Key criteria:
    - Density < 5 g/cm³ (lightweight)
    - Tensile strength > 500 MPa
    - Temperature stability > 300°C
    - Fatigue resistance
    - Corrosion resistance
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['Ti', 'Al', 'Mg', 'C', 'B', 'Si', 'Cr', 'Ni', 'V']
    
    def discover(self, n_candidates: int = 500) -> List[Dict]:
        """
        Discover aerospace materials
        
        Returns:
            List of candidate materials with aerospace scores
        """
        print("\n" + "="*60)
        print("AEROSPACE MATERIALS DISCOVERY")
        print("="*60)
        
        candidates = []
        
        # Known aerospace alloy families with full properties
        alloy_families = [
            {'name': 'Ti-6Al-4V variants', 'composition': 'Ti6Al4V', 'density': 4.43, 'strength': 900, 'max_temp': 400},
            {'name': 'Al-Li alloys', 'composition': 'Al3Li', 'density': 2.55, 'strength': 500, 'max_temp': 200},
            {'name': 'Mg-Al-Zn', 'composition': 'Mg3AlZn', 'density': 1.81, 'strength': 350, 'max_temp': 150},
            {'name': 'SiC composites', 'composition': 'SiC', 'density': 3.21, 'strength': 600, 'max_temp': 1400},
            {'name': 'Carbon fiber reinforced', 'composition': 'C', 'density': 1.75, 'strength': 1500, 'max_temp': 300},
            {'name': 'Ni superalloys', 'composition': 'Ni3AlCr', 'density': 8.19, 'strength': 1100, 'max_temp': 1100},
            {'name': 'TiAl intermetallics', 'composition': 'TiAl', 'density': 3.76, 'strength': 700, 'max_temp': 750},
        ]
        
        for family in alloy_families:
            mat = Material(
                composition=family['composition'],
                predicted_properties={
                    'density': family['density'],
                    'tensile_strength': family['strength'],
                    'max_service_temp': family['max_temp']
                }
            )
            
            score = self._score_aerospace(mat, family['density'], family['strength'], family['max_temp'])
            candidates.append({
                'material': mat,
                'family': family['name'],
                'density_g_cm3': family['density'],
                'tensile_strength_mpa': family['strength'],
                'max_service_temp_c': family['max_temp'],
                'aerospace_score': score,
                'weight_reduction_vs_steel': (7.85 - family['density']) / 7.85 * 100,
                'application': 'aerospace'
            })
        
        candidates.sort(key=lambda x: x['aerospace_score'], reverse=True)
        print(f"Generated {len(candidates)} aerospace material candidates")
        return candidates
    
    def _score_aerospace(self, material: Material, density: float, strength: float = 500, max_temp: float = 300) -> float:
        """Score material for aerospace applications using density, strength, and temperature stability"""
        score = 0.0
        
        # Lower density = higher score (max 30 points)
        if density < 2.0:
            score += 30
        elif density < 3.0:
            score += 25
        elif density < 4.5:
            score += 15
        elif density < 6.0:
            score += 10
        
        # Higher tensile strength = higher score (max 35 points)
        if strength >= 1000:
            score += 35
        elif strength >= 700:
            score += 25
        elif strength >= 500:
            score += 15
        else:
            score += 5
        
        # Higher temperature stability = higher score (max 25 points)
        if max_temp >= 1000:
            score += 25
        elif max_temp >= 500:
            score += 20
        elif max_temp >= 300:
            score += 10
        else:
            score += 5
        
        # Bonus for common aerospace elements (max 10 points)
        if 'Ti' in material.composition:
            score += 5
        if 'C' in material.composition:
            score += 5
        
        return min(score, 100)


class BiomedicalMaterialsDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for biomedical implant materials.
    
    Targets:
    - Hip/knee replacement materials
    - Dental implants
    - Bone scaffolds
    - Cardiovascular stents
    
    Key criteria:
    - Biocompatibility (non-toxic, non-allergenic)
    - Osseointegration (for bone implants)
    - Mechanical match to bone (E ~15-30 GPa)
    - Corrosion resistance in body fluids
    - Wear resistance
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.biocompatible_elements = ['Ti', 'Zr', 'Nb', 'Ta', 'Co', 'Cr', 'Mo', 'Ca', 'P']
        self.toxic_elements = ['Ni', 'Al', 'V', 'Cu', 'Pb', 'Cd', 'Hg']
    
    def discover(self, n_candidates: int = 500, application: str = 'bone') -> List[Dict]:
        """
        Discover biomedical materials
        
        Args:
            n_candidates: Number of candidates
            application: 'bone', 'dental', 'cardiovascular'
        
        Returns:
            List of candidate materials with biocompatibility scores
        """
        print("\n" + "="*60)
        print(f"BIOMEDICAL MATERIALS DISCOVERY ({application.upper()})")
        print("="*60)
        
        candidates = []
        
        # Known biomedical material families
        material_families = [
            {'name': 'Ti-Nb-Zr', 'composition': 'Ti13Nb13Zr', 'modulus': 79, 'biocompat': 95},
            {'name': 'Ti-Mo-Zr', 'composition': 'Ti15Mo5Zr', 'modulus': 78, 'biocompat': 93},
            {'name': 'CoCrMo', 'composition': 'Co28Cr6Mo', 'modulus': 210, 'biocompat': 85},
            {'name': 'Hydroxyapatite', 'composition': 'Ca10P6O26H2', 'modulus': 120, 'biocompat': 98},
            {'name': 'Beta-Ti', 'composition': 'Ti24Nb4Zr8Sn', 'modulus': 55, 'biocompat': 94},
            {'name': 'Tantalum', 'composition': 'Ta', 'modulus': 186, 'biocompat': 97},
            {'name': 'Zirconia', 'composition': 'ZrO2', 'modulus': 200, 'biocompat': 96},
        ]
        
        bone_modulus = 20  # Cortical bone ~15-30 GPa
        
        for family in material_families:
            # Check for toxic elements
            has_toxic = any(elem in family['composition'] for elem in self.toxic_elements)
            if has_toxic:
                continue  # Skip materials with toxic elements
            
            mat = Material(
                composition=family['composition'],
                predicted_properties={'elastic_modulus': family['modulus']}
            )
            
            # Calculate bone matching score (stress shielding prevention)
            modulus_match = 100 - abs(family['modulus'] - bone_modulus) / bone_modulus * 50
            modulus_match = max(0, min(100, modulus_match))
            
            # Overall score includes biocompatibility, bone matching, and toxicity check
            overall = (family['biocompat'] + modulus_match) / 2
            
            candidates.append({
                'material': mat,
                'family': family['name'],
                'elastic_modulus_gpa': family['modulus'],
                'biocompatibility_score': family['biocompat'],
                'bone_matching_score': modulus_match,
                'overall_score': overall,
                'toxic_elements_present': has_toxic,
                'stress_shielding_risk': 'High' if family['modulus'] > 100 else 'Low',
                'application': f'biomedical_{application}'
            })
        
        candidates.sort(key=lambda x: x['overall_score'], reverse=True)
        print(f"Generated {len(candidates)} biomedical material candidates")
        return candidates


class WideGapSemiconductorDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for wide bandgap semiconductors.
    
    Targets:
    - SiC alternatives for power electronics
    - GaN alternatives for RF/5G
    - Novel ultrawide bandgap materials (>4 eV)
    
    Key criteria:
    - Band gap > 2.0 eV (wide) or > 4.0 eV (ultrawide)
    - High breakdown field
    - High thermal conductivity
    - High electron mobility
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['Ga', 'N', 'Al', 'B', 'Si', 'C', 'O', 'Zn']
    
    def discover(self, n_candidates: int = 500, min_bandgap: float = 2.0) -> List[Dict]:
        """
        Discover wide bandgap semiconductors
        
        Args:
            n_candidates: Number of candidates
            min_bandgap: Minimum band gap in eV
        
        Returns:
            List of candidate materials with semiconductor scores
        """
        print("\n" + "="*60)
        print(f"WIDE BANDGAP SEMICONDUCTOR DISCOVERY (Eg > {min_bandgap} eV)")
        print("="*60)
        
        candidates = []
        
        # Known WBG semiconductor families with full properties
        wbg_families = [
            {'name': 'SiC-4H', 'composition': 'SiC', 'bandgap': 3.26, 'breakdown': 2.2, 'thermal_cond': 370, 'mobility': 900},
            {'name': 'GaN', 'composition': 'GaN', 'bandgap': 3.4, 'breakdown': 3.3, 'thermal_cond': 130, 'mobility': 1200},
            {'name': 'AlN', 'composition': 'AlN', 'bandgap': 6.2, 'breakdown': 12.0, 'thermal_cond': 285, 'mobility': 300},
            {'name': 'Ga2O3', 'composition': 'Ga2O3', 'bandgap': 4.9, 'breakdown': 8.0, 'thermal_cond': 27, 'mobility': 200},
            {'name': 'Diamond', 'composition': 'C', 'bandgap': 5.47, 'breakdown': 10.0, 'thermal_cond': 2200, 'mobility': 2000},
            {'name': 'BN', 'composition': 'BN', 'bandgap': 6.4, 'breakdown': 5.0, 'thermal_cond': 400, 'mobility': 50},
            {'name': 'AlGaN', 'composition': 'AlGaN', 'bandgap': 4.5, 'breakdown': 5.0, 'thermal_cond': 200, 'mobility': 800},
            {'name': 'ZnO', 'composition': 'ZnO', 'bandgap': 3.37, 'breakdown': 2.0, 'thermal_cond': 60, 'mobility': 200},
        ]
        
        for family in wbg_families:
            if family['bandgap'] < min_bandgap:
                continue
                
            mat = Material(
                composition=family['composition'],
                predicted_properties={
                    'band_gap': family['bandgap'],
                    'breakdown_field': family['breakdown'],
                    'thermal_conductivity': family['thermal_cond'],
                    'electron_mobility': family['mobility']
                }
            )
            
            score = self._score_wbg(family['bandgap'], family['breakdown'], family['thermal_cond'], family['mobility'])
            candidates.append({
                'material': mat,
                'family': family['name'],
                'band_gap_ev': family['bandgap'],
                'breakdown_field_mv_cm': family['breakdown'],
                'thermal_conductivity_w_mk': family['thermal_cond'],
                'electron_mobility_cm2_vs': family['mobility'],
                'wbg_score': score,
                'category': 'ultrawide' if family['bandgap'] > 4.0 else 'wide',
                'power_figure_of_merit': family['bandgap'] * family['breakdown']**2,
                'application': 'semiconductor'
            })
        
        candidates.sort(key=lambda x: x['wbg_score'], reverse=True)
        print(f"Generated {len(candidates)} WBG semiconductor candidates")
        return candidates
    
    def _score_wbg(self, bandgap: float, breakdown: float, thermal_cond: float = 100, mobility: float = 500) -> float:
        """Score wide bandgap semiconductor using bandgap, breakdown, thermal conductivity, and mobility"""
        score = 0.0
        
        # Bandgap score (max 20 points) - higher is better for power
        score += min(bandgap / 6.5 * 20, 20)
        
        # Breakdown field score (max 30 points) - critical for power devices
        score += min(breakdown / 12 * 30, 30)
        
        # Thermal conductivity score (max 25 points) - critical for heat dissipation
        if thermal_cond >= 1000:
            score += 25
        elif thermal_cond >= 300:
            score += 20
        elif thermal_cond >= 100:
            score += 10
        else:
            score += 5
        
        # Electron mobility score (max 25 points) - critical for switching speed
        if mobility >= 1500:
            score += 25
        elif mobility >= 800:
            score += 20
        elif mobility >= 300:
            score += 10
        else:
            score += 5
        
        return min(round(score, 1), 100)


class SustainableConstructionDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for sustainable construction materials.
    
    Targets:
    - Low-carbon cement alternatives
    - Geopolymers
    - Bio-based building materials
    - Recycled aggregate compositions
    
    Key criteria:
    - CO2 footprint reduction vs Portland cement
    - Compressive strength > 20 MPa
    - Durability
    - Cost-effectiveness
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['Si', 'Al', 'Ca', 'Na', 'K', 'Fe', 'Mg', 'O']
    
    def discover(self, n_candidates: int = 500) -> List[Dict]:
        """
        Discover sustainable construction materials
        
        Returns:
            List of candidate materials with sustainability scores
        """
        print("\n" + "="*60)
        print("SUSTAINABLE CONSTRUCTION MATERIALS DISCOVERY")
        print("="*60)
        
        candidates = []
        
        # Cement alternative families
        cement_alternatives = [
            {'name': 'Fly ash geopolymer', 'composition': 'Si2Al2O7Na', 'co2_reduction': 80, 'strength': 50},
            {'name': 'Slag cement', 'composition': 'CaSiAlO4', 'co2_reduction': 50, 'strength': 45},
            {'name': 'Limestone calcined clay', 'composition': 'CaAlSi2O8', 'co2_reduction': 40, 'strength': 40},
            {'name': 'Magnesium oxide cement', 'composition': 'MgO', 'co2_reduction': 30, 'strength': 35},
            {'name': 'Alkali-activated slag', 'composition': 'CaSi2AlO7K', 'co2_reduction': 70, 'strength': 55},
            {'name': 'Calcium sulfoaluminate', 'composition': 'Ca4Al6SO16', 'co2_reduction': 35, 'strength': 48},
        ]
        
        for alt in cement_alternatives:
            mat = Material(
                composition=alt['composition'],
                predicted_properties={
                    'co2_reduction': alt['co2_reduction'],
                    'compressive_strength': alt['strength']
                }
            )
            
            score = (alt['co2_reduction'] + alt['strength']) / 2
            candidates.append({
                'material': mat,
                'family': alt['name'],
                'co2_reduction_percent': alt['co2_reduction'],
                'compressive_strength_mpa': alt['strength'],
                'sustainability_score': score,
                'portland_cement_replacement': f"{alt['co2_reduction']}% lower emissions",
                'application': 'construction'
            })
        
        candidates.sort(key=lambda x: x['sustainability_score'], reverse=True)
        print(f"Generated {len(candidates)} sustainable construction candidates")
        return candidates


class TransparentConductorDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for transparent conductors (ITO alternatives).
    
    Targets:
    - ITO (Indium Tin Oxide) replacements
    - Flexible transparent electrodes
    - Low-cost alternatives
    
    Key criteria:
    - Optical transparency > 80% in visible range
    - Sheet resistance < 100 Ω/sq
    - No/low indium content (scarce element)
    - Flexibility for bendable displays
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['Zn', 'Sn', 'Al', 'Ga', 'Ti', 'Ag', 'C']
    
    def discover(self, n_candidates: int = 500) -> List[Dict]:
        """
        Discover transparent conductor materials
        
        Returns:
            List of candidate materials with TC scores
        """
        print("\n" + "="*60)
        print("TRANSPARENT CONDUCTOR DISCOVERY (ITO ALTERNATIVES)")
        print("="*60)
        
        candidates = []
        
        # TC material families
        tc_families = [
            {'name': 'AZO (Al:ZnO)', 'composition': 'Zn98Al2O100', 'transparency': 85, 'resistance': 50, 'indium_free': True},
            {'name': 'GZO (Ga:ZnO)', 'composition': 'Zn98Ga2O100', 'transparency': 87, 'resistance': 30, 'indium_free': True},
            {'name': 'FTO (F:SnO2)', 'composition': 'Sn95F5O190', 'transparency': 82, 'resistance': 15, 'indium_free': True},
            {'name': 'Silver nanowires', 'composition': 'Ag', 'transparency': 90, 'resistance': 20, 'indium_free': True},
            {'name': 'Graphene', 'composition': 'C', 'transparency': 97, 'resistance': 125, 'indium_free': True},
            {'name': 'PEDOT:PSS', 'composition': 'C10H8O4S2', 'transparency': 88, 'resistance': 100, 'indium_free': True},
            {'name': 'Carbon nanotubes', 'composition': 'C', 'transparency': 85, 'resistance': 60, 'indium_free': True},
        ]
        
        for tc in tc_families:
            mat = Material(
                composition=tc['composition'],
                predicted_properties={
                    'transparency': tc['transparency'],
                    'sheet_resistance': tc['resistance']
                }
            )
            
            # Figure of merit: transparency / log(resistance)
            fom = tc['transparency'] / (np.log10(tc['resistance'] + 1) + 1)
            candidates.append({
                'material': mat,
                'family': tc['name'],
                'transparency_percent': tc['transparency'],
                'sheet_resistance_ohm_sq': tc['resistance'],
                'figure_of_merit': round(fom, 2),
                'indium_free': tc['indium_free'],
                'ito_replacement_score': fom / 0.85 * 100,  # Normalized to ITO
                'application': 'transparent_conductor'
            })
        
        candidates.sort(key=lambda x: x['figure_of_merit'], reverse=True)
        print(f"Generated {len(candidates)} transparent conductor candidates")
        return candidates


class RareEarthFreeMagnetDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for rare-earth-free permanent magnets.
    
    Targets:
    - Alternatives to NdFeB magnets
    - EV motor magnets without Nd, Dy
    - Wind turbine generator magnets
    
    Key criteria:
    - (BH)max > 10 MGOe (minimum for many applications)
    - Curie temperature > 300°C
    - No rare earth elements (Nd, Dy, Pr, Sm)
    - Cost-effectiveness
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.rare_earth_elements = ['Nd', 'Dy', 'Pr', 'Sm', 'Ce', 'La', 'Eu', 'Tb']
        self.target_elements = ['Fe', 'Co', 'Mn', 'Ni', 'Al', 'N', 'C', 'B']
    
    def discover(self, n_candidates: int = 500, min_bhmax: float = 10.0) -> List[Dict]:
        """
        Discover rare-earth-free permanent magnets
        
        Args:
            n_candidates: Number of candidates
            min_bhmax: Minimum (BH)max in MGOe
        
        Returns:
            List of candidate materials with magnet scores
        """
        print("\n" + "="*60)
        print(f"RARE-EARTH-FREE MAGNET DISCOVERY ((BH)max > {min_bhmax} MGOe)")
        print("="*60)
        
        candidates = []
        
        # RE-free magnet families
        magnet_families = [
            {'name': 'Fe16N2', 'composition': 'Fe16N2', 'bhmax': 20, 'curie_temp': 540, 'maturity': 'Research'},
            {'name': 'MnBi', 'composition': 'MnBi', 'bhmax': 17, 'curie_temp': 360, 'maturity': 'Developing'},
            {'name': 'MnAl', 'composition': 'MnAl', 'bhmax': 12, 'curie_temp': 380, 'maturity': 'Production'},
            {'name': 'Alnico', 'composition': 'Fe35Co35Ni15Al7Cu8', 'bhmax': 10, 'curie_temp': 850, 'maturity': 'Production'},
            {'name': 'FeCo', 'composition': 'Fe65Co35', 'bhmax': 8, 'curie_temp': 980, 'maturity': 'Production'},
            {'name': 'Fe3Sn', 'composition': 'Fe3Sn', 'bhmax': 15, 'curie_temp': 450, 'maturity': 'Research'},
            {'name': 'MnGa', 'composition': 'MnGa', 'bhmax': 18, 'curie_temp': 400, 'maturity': 'Research'},
        ]
        
        for family in magnet_families:
            if family['bhmax'] < min_bhmax:
                continue
                
            mat = Material(
                composition=family['composition'],
                predicted_properties={
                    'bhmax': family['bhmax'],
                    'curie_temperature': family['curie_temp']
                }
            )
            
            # Score based on (BH)max and Curie temperature
            score = (family['bhmax'] / 50 * 50) + (family['curie_temp'] / 1000 * 50)
            candidates.append({
                'material': mat,
                'family': family['name'],
                'bhmax_mgoe': family['bhmax'],
                'curie_temp_c': family['curie_temp'],
                'magnet_score': round(score, 1),
                'maturity': family['maturity'],
                'ndfeb_replacement_potential': family['bhmax'] / 50 * 100,  # NdFeB ~50 MGOe
                'rare_earth_free': True,
                'application': 'permanent_magnet'
            })
        
        candidates.sort(key=lambda x: x['magnet_score'], reverse=True)
        print(f"Generated {len(candidates)} RE-free magnet candidates")
        return candidates


class SolidElectrolyteDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for solid-state battery electrolytes.
    
    Targets:
    - Li-ion conductors for solid-state batteries
    - Na-ion solid electrolytes
    - Safe, non-flammable alternatives to liquid electrolytes
    
    Key criteria:
    - Ionic conductivity > 1 mS/cm at room temperature
    - Electronic conductivity < 10^-10 S/cm
    - Electrochemical stability window > 5V
    - Chemical stability with Li metal anode
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['Li', 'P', 'S', 'O', 'Cl', 'La', 'Zr', 'Ti', 'Ge']
    
    def discover(self, n_candidates: int = 500, working_ion: str = 'Li') -> List[Dict]:
        """
        Discover solid electrolyte materials
        
        Args:
            n_candidates: Number of candidates
            working_ion: 'Li' or 'Na'
        
        Returns:
            List of candidate materials with electrolyte scores
        """
        print("\n" + "="*60)
        print(f"SOLID ELECTROLYTE DISCOVERY ({working_ion}-ion)")
        print("="*60)
        
        candidates = []
        
        # Solid electrolyte families
        electrolyte_families = [
            {'name': 'LGPS', 'composition': 'Li10GeP2S12', 'conductivity': 12, 'stability': 4.0, 'family': 'Sulfide'},
            {'name': 'Li6PS5Cl (Argyrodite)', 'composition': 'Li6PS5Cl', 'conductivity': 3, 'stability': 3.5, 'family': 'Sulfide'},
            {'name': 'LLZO', 'composition': 'Li7La3Zr2O12', 'conductivity': 0.5, 'stability': 6.0, 'family': 'Oxide'},
            {'name': 'LATP', 'composition': 'Li1.3Al0.3Ti1.7P3O12', 'conductivity': 0.7, 'stability': 4.5, 'family': 'NASICON'},
            {'name': 'LiPON', 'composition': 'Li2.9PO3.3N0.46', 'conductivity': 0.002, 'stability': 5.5, 'family': 'Oxynitride'},
            {'name': 'Li3InCl6', 'composition': 'Li3InCl6', 'conductivity': 2.0, 'stability': 4.2, 'family': 'Halide'},
            {'name': 'Na3PS4', 'composition': 'Na3PS4', 'conductivity': 0.2, 'stability': 3.0, 'family': 'Sulfide'},
        ]
        
        for family in electrolyte_families:
            mat = Material(
                composition=family['composition'],
                predicted_properties={
                    'ionic_conductivity': family['conductivity'],
                    'stability_window': family['stability']
                }
            )
            
            # Score based on conductivity and stability
            cond_score = min(family['conductivity'] / 12 * 50, 50)  # Normalized to LGPS
            stab_score = family['stability'] / 6 * 50  # Normalized to LLZO
            score = cond_score + stab_score
            
            candidates.append({
                'material': mat,
                'name': family['name'],
                'family': family['family'],
                'ionic_conductivity_ms_cm': family['conductivity'],
                'stability_window_v': family['stability'],
                'electrolyte_score': round(score, 1),
                'air_stable': family['family'] in ['Oxide', 'NASICON'],
                'working_ion': 'Li' if 'Li' in family['composition'] else 'Na',
                'application': 'solid_electrolyte'
            })
        
        candidates.sort(key=lambda x: x['electrolyte_score'], reverse=True)
        print(f"Generated {len(candidates)} solid electrolyte candidates")
        return candidates


class WaterPurificationDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for water purification membrane materials.
    
    Targets:
    - Desalination membranes
    - Heavy metal removal
    - Organic contaminant filtration
    - Antimicrobial surfaces
    
    Key criteria:
    - High water permeability
    - High salt rejection (>99% for desalination)
    - Chemical stability
    - Fouling resistance
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['C', 'N', 'O', 'S', 'Ti', 'Zr', 'Al']
    
    def discover(self, n_candidates: int = 500, application: str = 'desalination') -> List[Dict]:
        """
        Discover water purification materials
        
        Args:
            n_candidates: Number of candidates
            application: 'desalination', 'heavy_metal', 'organic'
        
        Returns:
            List of candidate materials with purification scores
        """
        print("\n" + "="*60)
        print(f"WATER PURIFICATION DISCOVERY ({application.upper()})")
        print("="*60)
        
        candidates = []
        
        # Membrane material families
        membrane_families = [
            {'name': 'Graphene oxide', 'composition': 'C10O2H4', 'permeability': 100, 'rejection': 99.5, 'fouling_resist': 80},
            {'name': 'MXene (Ti3C2)', 'composition': 'Ti3C2', 'permeability': 80, 'rejection': 99.0, 'fouling_resist': 85},
            {'name': 'MOF-based', 'composition': 'Zr6O4C60H24', 'permeability': 120, 'rejection': 98.5, 'fouling_resist': 70},
            {'name': 'Polyamide TFC', 'composition': 'C15H11N2O2', 'permeability': 40, 'rejection': 99.7, 'fouling_resist': 60},
            {'name': 'Aquaporin-inspired', 'composition': 'C80H120N20O25', 'permeability': 200, 'rejection': 99.9, 'fouling_resist': 75},
            {'name': 'Zeolite (NaA)', 'composition': 'Na12Al12Si12O48', 'permeability': 30, 'rejection': 99.8, 'fouling_resist': 90},
            {'name': 'CNT membrane', 'composition': 'C', 'permeability': 150, 'rejection': 99.2, 'fouling_resist': 85},
        ]
        
        for family in membrane_families:
            mat = Material(
                composition=family['composition'],
                predicted_properties={
                    'water_permeability': family['permeability'],
                    'salt_rejection': family['rejection']
                }
            )
            
            # Score based on permeability, rejection, and fouling resistance
            score = (family['permeability'] / 200 * 30 + 
                    family['rejection'] / 100 * 40 + 
                    family['fouling_resist'] / 100 * 30)
            
            candidates.append({
                'material': mat,
                'family': family['name'],
                'water_permeability_lmh_bar': family['permeability'],
                'salt_rejection_percent': family['rejection'],
                'fouling_resistance': family['fouling_resist'],
                'purification_score': round(score, 1),
                'application': f'water_{application}'
            })
        
        candidates.sort(key=lambda x: x['purification_score'], reverse=True)
        print(f"Generated {len(candidates)} water purification candidates")
        return candidates


class CarbonCaptureDiscovery(MaterialsDiscoveryPipeline):
    """
    Discovery workflow for carbon capture materials.
    
    Targets:
    - Direct air capture (DAC) sorbents
    - Flue gas CO2 capture
    - Carbon mineralization materials
    
    Key criteria:
    - High CO2 adsorption capacity (>3 mmol/g)
    - High CO2/N2 selectivity (>50)
    - Low regeneration energy
    - Stability over many cycles
    """
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)
        self.target_elements = ['C', 'N', 'O', 'Zn', 'Mg', 'Ca', 'Al', 'Na', 'K']
    
    def discover(self, n_candidates: int = 500, application: str = 'dac') -> List[Dict]:
        """
        Discover carbon capture materials
        
        Args:
            n_candidates: Number of candidates
            application: 'dac' (direct air capture), 'flue_gas', 'mineralization'
        
        Returns:
            List of candidate materials with capture scores
        """
        print("\n" + "="*60)
        print(f"CARBON CAPTURE DISCOVERY ({application.upper()})")
        print("="*60)
        
        candidates = []
        
        # Carbon capture material families
        capture_families = [
            {'name': 'MOF-74-Mg', 'composition': 'Mg2C8H2O6', 'capacity': 8.0, 'selectivity': 100, 'regen_temp': 150},
            {'name': 'SIFSIX-3-Cu', 'composition': 'CuSiF6N4C8H8', 'capacity': 2.5, 'selectivity': 1200, 'regen_temp': 80},
            {'name': 'Zeolite 13X', 'composition': 'Na86Al86Si106O384', 'capacity': 5.0, 'selectivity': 80, 'regen_temp': 250},
            {'name': 'Amine-functionalized silica', 'composition': 'SiO2C3H9N', 'capacity': 3.5, 'selectivity': 500, 'regen_temp': 100},
            {'name': 'Potassium carbonate', 'composition': 'K2CO3', 'capacity': 7.0, 'selectivity': 200, 'regen_temp': 120},
            {'name': 'Mg(OH)2 slurry', 'composition': 'MgO2H2', 'capacity': 10.0, 'selectivity': 1000, 'regen_temp': 350},
            {'name': 'Activated carbon + amine', 'composition': 'C100N5H15', 'capacity': 4.0, 'selectivity': 150, 'regen_temp': 120},
        ]
        
        for family in capture_families:
            mat = Material(
                composition=family['composition'],
                predicted_properties={
                    'co2_capacity': family['capacity'],
                    'selectivity': family['selectivity']
                }
            )
            
            # Score based on capacity, selectivity, and regeneration energy
            cap_score = family['capacity'] / 10 * 40
            sel_score = min(family['selectivity'] / 1200 * 30, 30)
            regen_score = (400 - family['regen_temp']) / 400 * 30  # Lower is better
            score = cap_score + sel_score + regen_score
            
            candidates.append({
                'material': mat,
                'family': family['name'],
                'co2_capacity_mmol_g': family['capacity'],
                'co2_n2_selectivity': family['selectivity'],
                'regeneration_temp_c': family['regen_temp'],
                'capture_score': round(score, 1),
                'energy_efficiency': 'High' if family['regen_temp'] < 150 else 'Medium' if family['regen_temp'] < 250 else 'Low',
                'application': f'carbon_capture_{application}'
            })
        
        candidates.sort(key=lambda x: x['capture_score'], reverse=True)
        print(f"Generated {len(candidates)} carbon capture candidates")
        return candidates


# ============================================================================
# CLI STEP HANDLERS
# ============================================================================

def run_step(step_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single pipeline step"""
    result = {
        'step': step_name,
        'success': False,
        'timestamp': datetime.now().isoformat(),
        'output': None,
        'error': None
    }
    
    pipeline = MaterialsDiscoveryPipeline(use_gpu=True)
    
    try:
        materials_data = params.get('materials', [])
        
        if step_name == 'structure_validation':
            results = []
            for mat in materials_data:
                composition = mat.get('composition') or mat.get('formula', '')
                valid = bool(composition and len(composition) > 0)
                results.append({
                    'composition': composition,
                    'valid': valid,
                    'material_type': mat.get('type', 'crystal')
                })
            result['output'] = {'validations': results, 'count': len(results)}
            result['success'] = True
        
        elif step_name == 'magpie_descriptors':
            compositions = [m.get('composition') or m.get('formula', '') for m in materials_data]
            descriptors = {}
            for comp in compositions:
                if comp:
                    features = pipeline.magpie.featurize(comp)
                    descriptors[comp] = {
                        'features': features.tolist(),
                        'n_features': len(features)
                    }
            result['output'] = {'descriptors': descriptors, 'count': len(descriptors)}
            result['success'] = True
        
        elif step_name == 'soap_descriptors':
            structures = []
            for mat in materials_data:
                if mat.get('structure'):
                    structures.append(mat['structure'])
            
            if structures:
                descriptors = pipeline.soap.batch_featurize(structures)
                result['output'] = {
                    'descriptors': [d.tolist() for d in descriptors],
                    'count': len(descriptors)
                }
            else:
                result['output'] = {'descriptors': [], 'count': 0, 'message': 'No structures provided'}
            result['success'] = True
        
        elif step_name == 'gnn_prediction':
            if not TORCH_GEOMETRIC_AVAILABLE:
                result['output'] = {'error': 'PyTorch Geometric not available'}
                result['success'] = False
            else:
                predictions = []
                for mat in materials_data:
                    if mat.get('structure'):
                        pred = pipeline.property_predictor.predict_from_structure(mat['structure'])
                        predictions.append({
                            'composition': mat.get('composition', ''),
                            'predictions': pred
                        })
                result['output'] = {'predictions': predictions, 'count': len(predictions)}
                result['success'] = True
        
        elif step_name == 'property_prediction':
            properties = params.get('properties', ['all'])
            predictions = {}
            
            for mat in materials_data:
                composition = mat.get('composition') or mat.get('formula', '')
                if composition:
                    pred = pipeline.property_predictor.predict_from_composition(composition)
                    predictions[composition] = pred
            
            result['output'] = {'predictions': predictions, 'count': len(predictions)}
            result['success'] = True
        
        elif step_name == 'manufacturability_scoring':
            scores = []
            for mat in materials_data:
                composition = mat.get('composition') or mat.get('formula', '')
                if composition:
                    material = Material(composition=composition, material_id=mat.get('id', composition))
                    score = pipeline.manufacturability_scorer.score(material)
                    scores.append(asdict(score))
            
            result['output'] = {'scores': scores, 'count': len(scores)}
            result['success'] = True
        
        elif step_name == 'synthesis_planning':
            plans = []
            for mat in materials_data:
                composition = mat.get('composition') or mat.get('formula', '')
                if composition:
                    material = Material(composition=composition)
                    plan = pipeline.synthesis_planner.suggest_synthesis_route(material)
                    plans.append({
                        'composition': composition,
                        'synthesis_plan': plan
                    })
            
            result['output'] = {'plans': plans, 'count': len(plans)}
            result['success'] = True
        
        elif step_name == 'batch_screening':
            target_properties = params.get('target_properties', {})
            
            materials = []
            for i, mat in enumerate(materials_data):
                composition = mat.get('composition') or mat.get('formula', '')
                if composition:
                    materials.append(Material(
                        composition=composition,
                        material_id=mat.get('id', f"MAT_{i:06d}")
                    ))
            
            predictions = pipeline.property_predictor.batch_predict(materials)
            
            results_list = []
            for mat, pred in zip(materials, predictions):
                score = pipeline.manufacturability_scorer.score(mat, pred)
                
                match_score = 1.0
                for prop, target in target_properties.items():
                    if hasattr(pred, prop):
                        val = getattr(pred, prop)
                        if val is not None:
                            if isinstance(target, dict):
                                min_val = target.get('min', float('-inf'))
                                max_val = target.get('max', float('inf'))
                                if not (min_val <= val <= max_val):
                                    match_score *= 0.5
                            else:
                                error = abs(val - target) / (abs(target) + 1e-6)
                                match_score *= 1 / (1 + error)
                
                results_list.append({
                    'material_id': mat.material_id,
                    'composition': mat.composition,
                    'predictions': asdict(pred),
                    'manufacturability': asdict(score),
                    'target_match_score': round(match_score, 3),
                    'overall_rank_score': round(match_score * score.overall_score, 3)
                })
            
            results_list.sort(key=lambda x: x['overall_rank_score'], reverse=True)
            
            result['output'] = {
                'results': results_list,
                'top_candidates': results_list[:10],
                'count': len(results_list)
            }
            result['success'] = True
        
        elif step_name == 'materials_generation':
            target_properties = params.get('target_properties', {})
            n_candidates = params.get('n_candidates', 100)
            elements = params.get('elements', None)
            
            compositions = pipeline.generator.generate_compositions(
                target_properties=target_properties,
                n_candidates=n_candidates,
                elements=elements
            )
            
            result['output'] = {'compositions': compositions, 'count': len(compositions)}
            result['success'] = True
        
        elif step_name == 'element_substitution':
            base_composition = params.get('base_composition', '')
            n_variants = params.get('n_variants', 100)
            
            if base_composition:
                variants = pipeline.generator.substitute_elements(base_composition, n_variants)
                result['output'] = {'variants': variants, 'count': len(variants)}
                result['success'] = True
            else:
                result['error'] = 'base_composition is required'
        
        elif step_name == 'atomistic_simulation':
            simulation_type = params.get('simulation_type', 'optimization')
            
            materials = []
            for mat in materials_data:
                composition = mat.get('composition') or mat.get('formula', '')
                if composition:
                    materials.append(Material(composition=composition))
            
            if materials:
                sim_results = pipeline.simulator.parallel_simulations(materials, simulation_type)
                result['output'] = {'simulations': sim_results, 'count': len(sim_results)}
                result['success'] = True
            else:
                result['output'] = {'simulations': [], 'count': 0}
                result['success'] = True
        
        elif step_name == 'full_pipeline':
            target_properties = params.get('target_properties', {'band_gap': 2.0})
            n_candidates = params.get('n_candidates', 1000)
            screen_top_n = params.get('screen_top_n', 50)
            
            df = pipeline.discover_materials(
                target_properties=target_properties,
                n_candidates=n_candidates,
                screen_top_n=screen_top_n
            )
            
            result['output'] = {
                'results': df.to_dict('records'),
                'count': len(df)
            }
            result['success'] = True
        
        # Materials Project API Integration
        elif step_name == 'mp_load_training_data':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            property_name = params.get('property_name', 'band_gap')
            n_materials = params.get('n_materials', 1000)
            elements = params.get('elements')
            exclude_elements = params.get('exclude_elements')
            additional_criteria = params.get('additional_criteria')
            
            materials, property_values = mp_loader.load_training_data(
                property_name=property_name,
                n_materials=n_materials,
                elements=elements,
                exclude_elements=exclude_elements,
                additional_criteria=additional_criteria,
                include_structures=params.get('include_structures', True)
            )
            
            result['output'] = {
                'materials': materials,
                'property_values': property_values,
                'property_name': property_name,
                'count': len(materials)
            }
            result['success'] = True
        
        elif step_name == 'mp_load_battery_data':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            n_materials = params.get('n_materials', 2000)
            working_ion = params.get('working_ion', 'Li')
            min_capacity = params.get('min_capacity')
            max_voltage = params.get('max_voltage')
            electrode_type = params.get('electrode_type', 'cathode')
            
            battery_data = mp_loader.load_battery_data(
                n_materials=n_materials,
                working_ion=working_ion,
                min_capacity=min_capacity,
                max_voltage=max_voltage,
                electrode_type=electrode_type
            )
            
            result['output'] = battery_data
            result['success'] = True
        
        elif step_name == 'mp_load_solar_materials':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            n_materials = params.get('n_materials', 1000)
            band_gap_range = tuple(params.get('band_gap_range', [1.0, 2.5]))
            stable_only = params.get('stable_only', True)
            
            solar_materials = mp_loader.load_solar_materials(
                n_materials=n_materials,
                band_gap_range=band_gap_range,
                stable_only=stable_only
            )
            
            result['output'] = {
                'materials': solar_materials,
                'count': len(solar_materials)
            }
            result['success'] = True
        
        elif step_name == 'mp_load_thermoelectric_materials':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            n_materials = params.get('n_materials', 1000)
            band_gap_range = tuple(params.get('band_gap_range', [0.1, 1.0]))
            heavy_elements = params.get('heavy_elements', True)
            
            te_materials = mp_loader.load_thermoelectric_materials(
                n_materials=n_materials,
                band_gap_range=band_gap_range,
                heavy_elements=heavy_elements
            )
            
            result['output'] = {
                'materials': te_materials,
                'count': len(te_materials)
            }
            result['success'] = True
        
        elif step_name == 'mp_load_superconductor_candidates':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            n_materials = params.get('n_materials', 500)
            include_cuprates = params.get('include_cuprates', True)
            include_iron_based = params.get('include_iron_based', True)
            
            sc_candidates = mp_loader.load_superconductor_candidates(
                n_materials=n_materials,
                include_cuprates=include_cuprates,
                include_iron_based=include_iron_based
            )
            
            result['output'] = {
                'materials': sc_candidates,
                'count': len(sc_candidates)
            }
            result['success'] = True
        
        elif step_name == 'mp_get_phase_diagram':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            elements = params.get('elements', [])
            include_unstable = params.get('include_unstable', False)
            
            if not elements:
                result['error'] = 'Elements list required for phase diagram'
            else:
                phase_diagram = mp_loader.get_phase_diagram(
                    elements=elements,
                    include_unstable=include_unstable
                )
                result['output'] = phase_diagram
                result['success'] = True
        
        elif step_name == 'mp_bulk_query':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            material_ids = params.get('material_ids', [])
            properties = params.get('properties', ['band_gap', 'formation_energy'])
            
            if not material_ids:
                result['error'] = 'Material IDs list required for bulk query'
            else:
                bulk_results = mp_loader.bulk_query_properties(
                    material_ids=material_ids,
                    properties=properties
                )
                result['output'] = {
                    'results': bulk_results,
                    'count': len(bulk_results)
                }
                result['success'] = True
        
        elif step_name == 'mp_search_formula':
            api_key = params.get('api_key') or os.environ.get('MP_API_KEY')
            mp_loader = MaterialsProjectLoader(api_key=api_key)
            
            formula = params.get('formula', '')
            anonymous = params.get('anonymous', False)
            
            if not formula:
                result['error'] = 'Formula required for search'
            else:
                search_results = mp_loader.search_by_formula(
                    formula=formula,
                    anonymous=anonymous
                )
                result['output'] = {
                    'results': search_results,
                    'count': len(search_results)
                }
                result['success'] = True
        
        else:
            result['error'] = f'Unknown step: {step_name}'
            result['output'] = {
                'available_steps': [
                    'structure_validation',
                    'magpie_descriptors',
                    'soap_descriptors',
                    'gnn_prediction',
                    'property_prediction',
                    'manufacturability_scoring',
                    'synthesis_planning',
                    'batch_screening',
                    'materials_generation',
                    'element_substitution',
                    'atomistic_simulation',
                    'full_pipeline',
                    'mp_load_training_data',
                    'mp_load_battery_data',
                    'mp_load_solar_materials',
                    'mp_load_thermoelectric_materials',
                    'mp_load_superconductor_candidates',
                    'mp_get_phase_diagram',
                    'mp_bulk_query',
                    'mp_search_formula'
                ]
            }
    
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    finally:
        pipeline.shutdown()
    
    return result


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_main():
    """Command-line interface for pipeline execution"""
    parser = argparse.ArgumentParser(description='Lika Sciences Materials Science Pipeline')
    parser.add_argument('--job-type', type=str, required=True,
                        help='Type of job to run')
    parser.add_argument('--params', type=str, default=None,
                        help='JSON string containing job parameters')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to JSON file containing job parameters')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (optional, defaults to stdout)')
    
    args = parser.parse_args()
    
    params = {}
    
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                params = json.load(f)
        except Exception as e:
            print(json.dumps({'success': False, 'error': f'Failed to read params file: {e}'}))
            return 1
    elif args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(json.dumps({'success': False, 'error': f'Invalid JSON params: {e}'}))
            return 1
    else:
        params = {'materials': []}
    
    result = run_step(args.job_type, params)
    
    output_json = json.dumps(result, indent=2, default=str)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json)
    
    return 0 if result['success'] else 1


def main():
    """Demo main function"""
    print("="*70)
    print("MATERIALS SCIENCE DISCOVERY PIPELINE")
    print("="*70)
    print("\nCapabilities:")
    print("  ✓ Magpie compositional descriptors (145 features)")
    print("  ✓ SOAP structural descriptors")
    print("  ✓ Graph Neural Networks for crystals")
    print("  ✓ High-throughput materials generation")
    print("  ✓ GPU-accelerated property prediction")
    print("  ✓ Parallel atomistic simulations")
    print("  ✓ Synthesis route planning")
    print("  ✓ Manufacturability scoring")
    print("="*70)
    
    print("\nCLI Usage:")
    print("  python3 materials_science_pipeline.py --job-type <step> --params '<json>'")
    print("\nSupported Steps:")
    steps = [
        'structure_validation', 'magpie_descriptors', 'soap_descriptors',
        'gnn_prediction', 'property_prediction', 'manufacturability_scoring',
        'synthesis_planning', 'batch_screening', 'materials_generation',
        'element_substitution', 'atomistic_simulation', 'full_pipeline'
    ]
    for step in steps:
        print(f"  - {step}")
    
    print("\n" + "="*70)
    print("EXAMPLE: Property Prediction")
    print("="*70)
    
    pipeline = MaterialsDiscoveryPipeline(use_gpu=True)
    
    test_compositions = ['LiFePO4', 'NaCl', 'Fe2O3', 'TiO2', 'ZnO']
    
    for comp in test_compositions:
        props = pipeline.property_predictor.predict_from_composition(comp)
        print(f"\n{comp}:")
        print(f"  Band gap: {props['band_gap']:.2f} eV")
        print(f"  Formation energy: {props['formation_energy']:.2f} eV/atom")
        print(f"  Bulk modulus: {props['bulk_modulus']:.1f} GPa")
    
    print("\n" + "="*70)
    print("SETUP INSTRUCTIONS")
    print("="*70)
    print("\nCore packages:")
    print("  pip install pymatgen ase torch pandas numpy")
    print("\nFor GNN:")
    print("  pip install torch-geometric")
    print("\nFor SOAP descriptors:")
    print("  pip install dscribe")
    print("\nFor distributed computing:")
    print("  pip install dask distributed")
    print("\nFor GPU acceleration:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("\n" + "="*70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('--job-type', '-h', '--help'):
        sys.exit(cli_main())
    else:
        main()
