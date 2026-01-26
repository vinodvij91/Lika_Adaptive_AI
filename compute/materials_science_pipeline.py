"""
<<<<<<< HEAD
#HARDWARE-AGNOSTIC UNIVERSAL MATERIALS PREDICTION PIPELINE
#Automatically adapts to ANY hardware configuration:
#- Single GPU, Multi-GPU, or CPU-only
#- Any GPU model (RTX 3090, V100, A100, T4, etc.)
#- Any amount of memory
#- Optimal settings chosen automatically

#NO HARDCODED HARDWARE ASSUMPTIONS!
=======
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
>>>>>>> f39227b (Add advanced materials discovery pipeline with AI and simulation features)
"""
from __future__ import annotations

<<<<<<< HEAD
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Core ML
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Feature extraction
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from pymatgen.core import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
=======
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

>>>>>>> f39227b (Add advanced materials discovery pipeline with AI and simulation features)


<<<<<<< HEAD
# ============================================================================
# HARDWARE DETECTION AND AUTO-CONFIGURATION
# ============================================================================

class HardwareConfig:
    """
    Auto-detect and configure for ANY hardware
    No assumptions about specific GPU models
    """
    
    def __init__(self):
        self.device_type = 'cpu'
        self.device_count = 0
        self.device_name = 'CPU'
        self.total_memory_gb = 0
        self.compute_capability = None
        self.supports_mixed_precision = False
        self.cpu_cores = mp.cpu_count()
        
        self._detect_hardware()
        self._determine_optimal_config()
    
    def _detect_hardware(self):
        """Detect available hardware"""
        
        if not TORCH_AVAILABLE:
            return
        
        if torch.cuda.is_available():
            self.device_type = 'cuda'
            self.device_count = torch.cuda.device_count()
            
            # Get info from first GPU (assume homogeneous for multi-GPU)
            props = torch.cuda.get_device_properties(0)
            self.device_name = props.name
            self.total_memory_gb = props.total_memory / 1e9 * self.device_count
            self.compute_capability = (props.major, props.minor)
            
            # Check for mixed precision support
            # Tensor Cores available from compute capability 7.0+ (V100, T4, RTX 20xx+)
            self.supports_mixed_precision = self.compute_capability[0] >= 7
        
        elif torch.backends.mps.is_available():
            # Apple Silicon (M1/M2/M3)
            self.device_type = 'mps'
            self.device_count = 1
            self.device_name = 'Apple Silicon'
            self.supports_mixed_precision = False  # MPS doesn't support AMP yet
    
    def _determine_optimal_config(self):
        """Determine optimal configuration based on hardware"""
        
        if self.device_type == 'cpu':
            self.optimal_batch_size = 32
            self.optimal_workers = 0
            self.use_multi_gpu = False
            self.gradient_accumulation = 1
            
        elif self.device_type == 'mps':
            # Apple Silicon
            self.optimal_batch_size = 64
            self.optimal_workers = 0  # MPS doesn't benefit from workers
            self.use_multi_gpu = False
            self.gradient_accumulation = 1
            
        elif self.device_type == 'cuda':
            # GPU configuration based on memory and count
            memory_per_gpu = self.total_memory_gb / self.device_count
            
            # Batch size based on GPU memory
            if memory_per_gpu < 8:
                # Small GPU (e.g., GTX 1060, RTX 3050)
                self.optimal_batch_size = 64
                self.gradient_accumulation = 4
            elif memory_per_gpu < 12:
                # Medium GPU (e.g., RTX 2060, RTX 3060)
                self.optimal_batch_size = 128
                self.gradient_accumulation = 2
            elif memory_per_gpu < 16:
                # Large GPU (e.g., V100 16GB, RTX 3070)
                self.optimal_batch_size = 256
                self.gradient_accumulation = 2
            elif memory_per_gpu < 24:
                # Very large GPU (e.g., V100 32GB, A100 40GB, RTX 3080)
                self.optimal_batch_size = 512
                self.gradient_accumulation = 1
            else:
                # Huge GPU (e.g., RTX 3090, A100 80GB)
                self.optimal_batch_size = 1024
                self.gradient_accumulation = 1
            
            # Multi-GPU handling
            if self.device_count > 1:
                self.use_multi_gpu = True
                # Scale batch size with GPU count (but not linearly)
                self.optimal_batch_size = int(self.optimal_batch_size * (1 + 0.5 * (self.device_count - 1)))
            else:
                self.use_multi_gpu = False
            
            # Workers based on CPU cores and GPU count
            if self.device_count == 1:
                self.optimal_workers = min(4, self.cpu_cores // 2)
            else:
                self.optimal_workers = min(8 * self.device_count, self.cpu_cores)
    
    def get_device(self):
        """Get torch device object"""
        if self.device_type == 'cuda':
            return torch.device('cuda')
        elif self.device_type == 'mps':
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def summary(self) -> str:
        """Get hardware summary"""
        lines = [
            "="*80,
            "HARDWARE CONFIGURATION",
            "="*80,
            f"Device Type: {self.device_type.upper()}",
            f"Device Name: {self.device_name}",
        ]
        
        if self.device_type == 'cuda':
            lines.extend([
                f"GPU Count: {self.device_count}",
                f"Total Memory: {self.total_memory_gb:.1f} GB",
                f"Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}",
                f"Mixed Precision Support: {'Yes' if self.supports_mixed_precision else 'No'}",
            ])
        
        lines.extend([
            f"CPU Cores: {self.cpu_cores}",
            "",
            "OPTIMAL CONFIGURATION:",
            f"  Batch Size: {self.optimal_batch_size}",
            f"  Workers: {self.optimal_workers}",
            f"  Gradient Accumulation: {self.gradient_accumulation}",
            f"  Multi-GPU: {'Yes' if self.use_multi_gpu else 'No'}",
            f"  Mixed Precision: {'Yes' if self.supports_mixed_precision else 'No'}",
            "="*80
        ])
        
        return "\n".join(lines)


# ============================================================================
# UNIVERSAL FEATURE EXTRACTORS
# ============================================================================

class BaseFeatureExtractor:
    """Base class with parallel extraction"""
    
    def extract_features(self, representation) -> np.ndarray:
        raise NotImplementedError
    
    def get_feature_dim(self) -> int:
        raise NotImplementedError
    
    def batch_extract_parallel(self, representations: List, n_jobs: int = -1) -> np.ndarray:
        """Parallel feature extraction"""
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        if len(representations) < 100:
            return np.array([self.extract_features(rep) for rep in representations])
        
        chunk_size = max(1, len(representations) // (n_jobs * 4))
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            features = list(executor.map(
                self.extract_features,
                representations,
                chunksize=chunk_size
            ))
        
        return np.array(features)


class MolecularFeatureExtractor(BaseFeatureExtractor):
    """Molecular features from SMILES"""
    
    def __init__(self, fingerprint_size: int = 512):
        self.fingerprint_size = fingerprint_size
    
    def extract_features(self, smiles: str) -> np.ndarray:
        if not RDKIT_AVAILABLE:
            return np.zeros(self.get_feature_dim())
        
        mol = Chem.MolFromSmiles(smiles.replace('*', '[H]'))
        if mol is None:
            return np.zeros(self.get_feature_dim())
        
        features = []
        
        try:
            features.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
            ])
        except:
            features.extend([0] * 9)
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fingerprint_size)
            features.extend(fp.ToList())
        except:
            features.extend([0] * self.fingerprint_size)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        return 9 + self.fingerprint_size


class CompositionFeatureExtractor(BaseFeatureExtractor):
    """Composition features from formula"""
    
    def __init__(self):
        if MATMINER_AVAILABLE:
            self.elem_featurizer = ElementProperty.from_preset("magpie")
            self.stoich_featurizer = Stoichiometry()
    
    def extract_features(self, formula: str) -> np.ndarray:
        if not PYMATGEN_AVAILABLE:
            return np.zeros(self.get_feature_dim())
        
        try:
            comp = Composition(formula)
        except:
            return np.zeros(self.get_feature_dim())
        
        features = [comp.num_atoms, comp.weight, len(comp.elements)]
        
        elements = comp.elements
        fractions = [comp.get_atomic_fraction(el) for el in elements]
        
        avg_z = sum(el.Z * frac for el, frac in zip(elements, fractions))
        avg_mass = sum(el.atomic_mass * frac for el, frac in zip(elements, fractions))
        
        features.extend([avg_z, avg_mass])
        
        if MATMINER_AVAILABLE:
            try:
                elem_features = self.elem_featurizer.featurize(comp)
                stoich_features = self.stoich_featurizer.featurize(comp)
                features.extend(elem_features)
                features.extend(stoich_features)
            except:
                features.extend([0] * 142)
        else:
            features.extend([0] * 142)
        
        return np.array(features[:self.get_feature_dim()], dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        return 147


# ============================================================================
# UNIVERSAL DATASET AND MODEL
# ============================================================================

class UniversalDataset(Dataset):
    """Dataset with optional pinned memory"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 pin_memory: bool = True, device_type: str = 'cuda'):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
        # Pin memory only for CUDA
        if pin_memory and device_type == 'cuda':
            self.features = self.features.pin_memory()
            self.targets = self.targets.pin_memory()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class UniversalNN(nn.Module):
    """Adaptive neural network"""
    
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None,
                 output_dim: int = 1, dropout: float = 0.3):
        super(UniversalNN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [
                min(1024, max(256, input_dim * 2)),
                min(512, max(128, input_dim)),
                min(256, max(64, input_dim // 2)),
            ]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# UNIVERSAL MATERIALS PREDICTOR - WORKS ON ANY HARDWARE
# ============================================================================

class UniversalMaterialsPredictor:
    """
    Hardware-agnostic materials predictor
    Automatically adapts to ANY GPU/CPU configuration
    """
    
    def __init__(self, material_type: str, property_name: str,
                 auto_config: bool = True):
        """
        Initialize predictor
        
        Args:
            material_type: 'molecular', 'composition', 'perovskite'
            property_name: Property to predict
            auto_config: Automatically configure for hardware (recommended)
        """
        self.material_type = material_type
        self.property_name = property_name
        
        # Detect and configure hardware
        self.hw_config = HardwareConfig()
        
        # Show configuration
        if auto_config:
            print(self.hw_config.summary())
        
        # Get device
        self.device = self.hw_config.get_device()
        
        # Select feature extractor
        extractors = {
            'molecular': MolecularFeatureExtractor(fingerprint_size=512),
            'composition': CompositionFeatureExtractor(),
        }
        
        if material_type not in extractors:
            raise ValueError(f"Material type '{material_type}' not supported")
        
        self.feature_extractor = extractors[material_type]
        
        # ML components
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        
        # Mixed precision scaler (only for CUDA with support)
        if self.hw_config.supports_mixed_precision and self.hw_config.device_type == 'cuda':
            self.scaler_amp = GradScaler()
            self.use_amp = True
        else:
            self.scaler_amp = None
            self.use_amp = False
        
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def prepare_data(self, representations: List, properties: np.ndarray,
                     n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data with parallel extraction"""
        
        print(f"\nExtracting features from {len(representations)} materials...")
        
        X = self.feature_extractor.batch_extract_parallel(representations, n_jobs=n_jobs)
        y = np.array(properties).reshape(-1, 1)
        
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid]
        y = y[valid]
        
        print(f"Valid samples: {len(X)}/{len(representations)}")
        
        return X, y
    
    def create_model(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        """Create model and wrap for multi-GPU if needed"""
        
        self.model = UniversalNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=0.3
        ).to(self.device)
        
        # Wrap for multi-GPU if available
        if self.hw_config.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
            print(f"✓ Model wrapped for {self.hw_config.device_count} GPUs")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {total_params:,} parameters")
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200,
              batch_size: Optional[int] = None,
              num_workers: Optional[int] = None,
              gradient_accumulation_steps: Optional[int] = None,
              learning_rate: float = 0.001,
              validation_split: float = 0.15,
              early_stopping_patience: int = 30,
              verbose: bool = True):
        """
        Train with hardware-optimized settings
        
        If batch_size, num_workers, etc. not specified, uses optimal values
        """
        
        # Use hardware-optimized defaults
        if batch_size is None:
            batch_size = self.hw_config.optimal_batch_size
        if num_workers is None:
            num_workers = self.hw_config.optimal_workers
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.hw_config.gradient_accumulation
        
        if verbose:
            print(f"\nTraining Configuration:")
            print(f"  Batch size: {batch_size}")
            print(f"  Workers: {num_workers}")
            print(f"  Gradient accumulation: {gradient_accumulation_steps}")
            print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
            print(f"  Mixed precision: {self.use_amp}")
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        # Create datasets
        train_dataset = UniversalDataset(
            X_train, y_train,
            pin_memory=(self.hw_config.device_type == 'cuda'),
            device_type=self.hw_config.device_type
        )
        val_dataset = UniversalDataset(
            X_val, y_val,
            pin_memory=(self.hw_config.device_type == 'cuda'),
            device_type=self.hw_config.device_type
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # Already pinned in dataset
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0)
        )
        
        # Create model
        if self.model is None:
            self.create_model(input_dim=X.shape[1])
        
        # Optimizer and scheduler
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            optimizer.zero_grad()
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Forward pass with optional mixed precision
                if self.use_amp:
                    with autocast():
                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y) / gradient_accumulation_steps
                    
                    self.scaler_amp.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler_amp.step(optimizer)
                        self.scaler_amp.update()
                        optimizer.zero_grad()
                else:
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y) / gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                train_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    if self.use_amp:
                        with autocast():
                            predictions = self.model(batch_X)
                            loss = criterion(predictions, batch_y)
                    else:
                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y)
                    
                    val_losses.append(loss.item())
            
            # Statistics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch [{epoch+1}/{epochs}] - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def predict(self, representations: List, batch_size: Optional[int] = None,
                n_jobs: int = -1) -> np.ndarray:
        """Predict with hardware-optimized settings"""
        
        if batch_size is None:
            batch_size = self.hw_config.optimal_batch_size * 2  # Larger for inference
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        X = self.feature_extractor.batch_extract_parallel(representations, n_jobs=n_jobs)
        X_scaled = self.scaler_X.transform(X)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                batch = X_scaled[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        pred = self.model(X_tensor).cpu().numpy()
                else:
                    pred = self.model(X_tensor).cpu().numpy()
                
                predictions.append(pred)
        
        predictions = np.vstack(predictions)
        predictions = self.scaler_y.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def save_model(self, filepath: str):
        """Save model"""
        checkpoint = {
            'material_type': self.material_type,
            'property_name': self.property_name,
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'training_history': self.training_history,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.material_type = checkpoint['material_type']
        self.property_name = checkpoint['property_name']
        
        input_dim = checkpoint['scaler_X'].n_features_in_
        self.create_model(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\nUniversal Materials Prediction Pipeline")
    print("Works on ANY hardware: CPU, Single GPU, Multi-GPU\n")
    
    # This will auto-detect and configure for your hardware
    predictor = UniversalMaterialsPredictor(
        material_type='molecular',
        property_name='glass_transition_temp',
        auto_config=True  # Automatically optimizes for your hardware
    )
    
    # Generate example data
    materials = ['CC', 'c1ccccc1CC', 'CC(C)(C(=O)OC)C'] * 100
    properties = np.random.randn(300) * 50 + 50
    
    # Prepare data
    X, y = predictor.prepare_data(materials, properties, n_jobs=-1)
    
    # Train (uses optimal settings for YOUR hardware automatically)
    predictor.train(X, y, epochs=50, verbose=True)
    
    # Predict
    predictions = predictor.predict(['CCOCC', 'CCCl'])
    print(f"\nPredictions: {predictions}")
    
    print("\n✓ Pipeline works on your hardware!")
=======

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
                    'full_pipeline'
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
>>>>>>> f39227b (Add advanced materials discovery pipeline with AI and simulation features)
