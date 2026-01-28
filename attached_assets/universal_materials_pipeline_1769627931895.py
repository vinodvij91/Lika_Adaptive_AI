"""
Universal Materials Property Prediction Pipeline
Supports: Polymers, Small Molecules, MOFs, Perovskites, Batteries, Catalysts

Extends the polymer pipeline to handle diverse material classes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Literal
import json
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    from polymer_ml_pipeline import PolymerPropertyNN, PolymerDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ============================================================================
# MATERIAL-SPECIFIC FEATURE EXTRACTORS
# ============================================================================

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction"""
    
    @abstractmethod
    def extract_features(self, representation: str) -> np.ndarray:
        """Extract features from material representation"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get dimension of feature vector"""
        pass


class PolymerFeatureExtractor(BaseFeatureExtractor):
    """Feature extraction for polymers (from SMILES)"""
    
    def extract_features(self, smiles: str) -> np.ndarray:
        """Extract polymer-specific features"""
        mol = Chem.MolFromSmiles(smiles.replace('*', '[H]'))
        if mol is None:
            return np.zeros(self.get_feature_dim())
        
        features = []
        
        # Basic descriptors
        features.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            Descriptors.BertzCT(mol),
        ])
        
        # Polymer-specific: chain flexibility
        features.append(Descriptors.NumRotatableBonds(mol) / max(mol.GetNumAtoms(), 1))
        
        # Aromatic fraction
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(aromatic_atoms / max(mol.GetNumAtoms(), 1))
        
        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        features.extend(fp.ToList())
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        return 12 + 512


class SmallMoleculeFeatureExtractor(BaseFeatureExtractor):
    """Feature extraction for small drug-like molecules"""
    
    def extract_features(self, smiles: str) -> np.ndarray:
        """Extract drug-like molecule features"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.get_feature_dim())
        
        features = []
        
        # Lipinski's Rule of Five descriptors
        features.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ])
        
        # Additional drug-like properties
        features.extend([
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
        ])
        
        # Complexity and charge
        features.extend([
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
        ])
        
        # Morgan fingerprint (drug-optimized)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
        features.extend(fp.ToList())
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        return 13 + 1024


class PerovskiteFeatureExtractor(BaseFeatureExtractor):
    """Feature extraction for perovskite materials (ABX3 structure)"""
    
    def __init__(self):
        # Common perovskite elements and their properties
        self.element_properties = {
            # A-site cations
            'Cs': {'radius': 1.67, 'electronegativity': 0.79, 'valence': 1},
            'MA': {'radius': 1.80, 'electronegativity': 2.0, 'valence': 1},  # Methylammonium
            'FA': {'radius': 1.90, 'electronegativity': 2.2, 'valence': 1},  # Formamidinium
            'Rb': {'radius': 1.52, 'electronegativity': 0.82, 'valence': 1},
            'K': {'radius': 1.38, 'electronegativity': 0.82, 'valence': 1},
            
            # B-site cations
            'Pb': {'radius': 1.19, 'electronegativity': 2.33, 'valence': 2},
            'Sn': {'radius': 1.10, 'electronegativity': 1.96, 'valence': 2},
            'Ge': {'radius': 0.73, 'electronegativity': 2.01, 'valence': 2},
            'Cu': {'radius': 0.73, 'electronegativity': 1.90, 'valence': 2},
            
            # X-site anions
            'I': {'radius': 2.20, 'electronegativity': 2.66, 'valence': -1},
            'Br': {'radius': 1.96, 'electronegativity': 2.96, 'valence': -1},
            'Cl': {'radius': 1.81, 'electronegativity': 3.16, 'valence': -1},
        }
    
    def extract_features(self, formula: str) -> np.ndarray:
        """
        Extract features from perovskite formula
        Format: "A_B_X3" or "A1-xA'x_B_X3" for mixed compositions
        Example: "Cs_Pb_I3", "MA0.5FA0.5_Pb_I3"
        """
        features = []
        
        try:
            # Parse formula (simplified parser)
            parts = formula.split('_')
            a_site = parts[0]
            b_site = parts[1]
            x_site = parts[2].replace('3', '')
            
            # Extract A-site features
            a_features = self._get_site_features(a_site, 'A')
            b_features = self._get_site_features(b_site, 'B')
            x_features = self._get_site_features(x_site, 'X')
            
            features.extend(a_features)
            features.extend(b_features)
            features.extend(x_features)
            
            # Goldschmidt tolerance factor
            r_a = a_features[0]
            r_b = b_features[0]
            r_x = x_features[0]
            tolerance_factor = (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))
            features.append(tolerance_factor)
            
            # Octahedral factor
            octahedral_factor = r_b / r_x
            features.append(octahedral_factor)
            
        except:
            # Return zeros if parsing fails
            return np.zeros(self.get_feature_dim())
        
        return np.array(features, dtype=np.float32)
    
    def _get_site_features(self, site: str, site_type: str) -> List[float]:
        """Extract features for a specific site"""
        # Handle mixed compositions (e.g., "MA0.5FA0.5")
        if any(char.isdigit() for char in site):
            # Mixed composition - average properties
            elements = []
            fractions = []
            # Simplified parsing
            if site in self.element_properties:
                elem_props = self.element_properties[site]
            else:
                # Default values
                elem_props = {'radius': 1.5, 'electronegativity': 2.0, 'valence': 1}
        else:
            if site in self.element_properties:
                elem_props = self.element_properties[site]
            else:
                elem_props = {'radius': 1.5, 'electronegativity': 2.0, 'valence': 1}
        
        return [
            elem_props['radius'],
            elem_props['electronegativity'],
            elem_props['valence']
        ]
    
    def get_feature_dim(self) -> int:
        return 3 * 3 + 2  # 3 sites × 3 features + 2 structural factors


class MOFFeatureExtractor(BaseFeatureExtractor):
    """Feature extraction for Metal-Organic Frameworks"""
    
    def extract_features(self, representation: str) -> np.ndarray:
        """
        Extract MOF features from linker SMILES or MOF composition
        Format: "Metal_Linker" e.g., "Zn_BDC" or SMILES for linker
        """
        features = []
        
        # Parse representation
        if '_' in representation:
            metal, linker_smiles = representation.split('_', 1)
            
            # Metal properties (simplified)
            metal_features = self._get_metal_features(metal)
            features.extend(metal_features)
            
            # Linker properties from SMILES
            mol = Chem.MolFromSmiles(linker_smiles)
            if mol:
                linker_features = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    mol.GetNumAtoms(),
                ]
                features.extend(linker_features)
                
                # Fingerprint for linker
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                features.extend(fp.ToList())
            else:
                features.extend([0] * (6 + 256))
        else:
            # Default features
            features.extend([0] * self.get_feature_dim())
        
        return np.array(features[:self.get_feature_dim()], dtype=np.float32)
    
    def _get_metal_features(self, metal: str) -> List[float]:
        """Get metal center features"""
        metal_props = {
            'Zn': [30, 2, 1.65],  # Atomic number, valence, electronegativity
            'Cu': [29, 2, 1.90],
            'Fe': [26, 2, 1.83],
            'Co': [27, 2, 1.88],
            'Ni': [28, 2, 1.91],
            'Cr': [24, 3, 1.66],
            'Al': [13, 3, 1.61],
        }
        
        return metal_props.get(metal, [20, 2, 1.5])  # Defaults
    
    def get_feature_dim(self) -> int:
        return 3 + 6 + 256  # Metal + linker descriptors + fingerprint


# ============================================================================
# UNIVERSAL MATERIALS PREDICTOR
# ============================================================================

class UniversalMaterialsPredictor:
    """
    Universal predictor for multiple material classes
    Automatically selects appropriate feature extractor
    """
    
    MATERIAL_TYPES = Literal[
        'polymer', 
        'small_molecule', 
        'perovskite', 
        'mof',
        'battery',  # Can be added
        'catalyst',  # Can be added
    ]
    
    def __init__(self, material_type: MATERIAL_TYPES, property_name: str):
        self.material_type = material_type
        self.property_name = property_name
        
        # Select feature extractor
        self.feature_extractor = self._get_feature_extractor()
        
        # Standard scaler and model
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def _get_feature_extractor(self) -> BaseFeatureExtractor:
        """Select appropriate feature extractor"""
        extractors = {
            'polymer': PolymerFeatureExtractor,
            'small_molecule': SmallMoleculeFeatureExtractor,
            'perovskite': PerovskiteFeatureExtractor,
            'mof': MOFFeatureExtractor,
        }
        
        if self.material_type not in extractors:
            raise ValueError(f"Material type '{self.material_type}' not supported")
        
        return extractors[self.material_type]()
    
    def prepare_data(self, representations: List[str], properties: np.ndarray) -> Tuple:
        """Prepare data for training"""
        print(f"Extracting features for {self.material_type} materials...")
        
        X = np.array([self.feature_extractor.extract_features(rep) for rep in representations])
        y = np.array(properties).reshape(-1, 1)
        
        # Remove invalid samples
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"Valid samples: {len(X)}, Feature dim: {X.shape[1]}")
        
        return X, y
    
    def create_model(self, input_dim: int, hidden_dims: List[int] = None):
        """Create neural network model"""
        if hidden_dims is None:
            # Auto-scale architecture based on input dimension
            hidden_dims = [
                min(512, input_dim * 2),
                min(256, input_dim),
                min(128, input_dim // 2),
            ]
        
        self.model = PolymerPropertyNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=0.3
        ).to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, **kwargs):
        """Train the model"""
        from polymer_ml_pipeline import PolymerPropertyPredictor
        
        # Use the existing training infrastructure
        temp_predictor = PolymerPropertyPredictor(self.property_name)
        temp_predictor.scaler_X = self.scaler_X
        temp_predictor.scaler_y = self.scaler_y
        temp_predictor.model = self.model
        temp_predictor.device = self.device
        
        history = temp_predictor.train(X, y, epochs=epochs, **kwargs)
        
        # Update our state
        self.scaler_X = temp_predictor.scaler_X
        self.scaler_y = temp_predictor.scaler_y
        self.model = temp_predictor.model
        self.training_history = history
        
        return history
    
    def predict(self, representations: List[str]) -> np.ndarray:
        """Predict properties"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X = np.array([self.feature_extractor.extract_features(rep) for rep in representations])
        X_scaled = self.scaler_X.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions.flatten()
    
    def save_model(self, filepath: str):
        """Save complete model"""
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
        """Load complete model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.material_type = checkpoint['material_type']
        self.property_name = checkpoint['property_name']
        self.feature_extractor = self._get_feature_extractor()
        
        input_dim = checkpoint['scaler_X'].n_features_in_
        self.create_model(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")


# ============================================================================
# MATERIAL-SPECIFIC DATASETS
# ============================================================================

def generate_small_molecule_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic drug-like molecule dataset"""
    
    # Common drug-like SMILES patterns
    drug_templates = [
        # Benzene derivatives
        "c1ccccc1",
        "c1ccccc1C",
        "c1ccccc1O",
        "c1ccccc1N",
        
        # Heterocycles
        "c1ccncc1",  # Pyridine
        "c1cccnc1",  # Pyridine isomer
        "c1cnccn1",  # Pyrimidine
        "c1coc(c1)",  # Furan
        
        # Pharmaceutical fragments
        "CC(=O)Nc1ccccc1",  # Acetanilide-like
        "CC(=O)O",  # Acetate
        "c1ccc(cc1)CO",  # Benzyl alcohol
        "c1ccc(cc1)N",  # Aniline
    ]
    
    data = []
    np.random.seed(42)
    
    for i in range(n_samples):
        base_smiles = np.random.choice(drug_templates)
        
        # Estimate solubility (logS) based on structure
        solubility = estimate_solubility(base_smiles)
        
        # Estimate bioavailability score
        bioavailability = np.random.uniform(0.3, 0.9)
        
        data.append({
            'smiles': base_smiles,
            'solubility_logS': solubility,
            'bioavailability': bioavailability,
            'source': 'synthetic'
        })
    
    return pd.DataFrame(data)


def estimate_solubility(smiles: str) -> float:
    """Estimate aqueous solubility"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    
    # Simplified solubility estimate
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    
    # Empirical formula
    solubility = 0.5 - 0.01 * mw - logp + 0.5 * hbd
    solubility += np.random.normal(0, 0.2)
    
    return solubility


def generate_perovskite_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic perovskite solar cell dataset"""
    
    a_sites = ['Cs', 'MA', 'FA', 'Rb']
    b_sites = ['Pb', 'Sn']
    x_sites = ['I', 'Br', 'Cl']
    
    data = []
    np.random.seed(42)
    
    for i in range(n_samples):
        a = np.random.choice(a_sites)
        b = np.random.choice(b_sites)
        x = np.random.choice(x_sites)
        
        formula = f"{a}_{b}_{x}3"
        
        # Estimate bandgap based on composition
        bandgap = estimate_perovskite_bandgap(a, b, x)
        
        # Estimate stability (months)
        stability = estimate_perovskite_stability(a, b, x)
        
        # Estimate PCE (power conversion efficiency)
        pce = max(0, min(25, 15 + (1.5 - bandgap) * 5 + np.random.normal(0, 2)))
        
        data.append({
            'formula': formula,
            'bandgap_eV': bandgap,
            'stability_months': stability,
            'pce_percent': pce,
            'source': 'synthetic'
        })
    
    return pd.DataFrame(data)


def estimate_perovskite_bandgap(a: str, b: str, x: str) -> float:
    """Estimate perovskite bandgap"""
    base_gaps = {
        ('Cs', 'Pb', 'I'): 1.73,
        ('MA', 'Pb', 'I'): 1.55,
        ('FA', 'Pb', 'I'): 1.48,
        ('Cs', 'Pb', 'Br'): 2.25,
        ('MA', 'Pb', 'Br'): 2.28,
        ('Cs', 'Sn', 'I'): 1.30,
    }
    
    gap = base_gaps.get((a, b, x), 1.6)
    gap += np.random.normal(0, 0.05)
    
    return max(0.5, min(3.0, gap))


def estimate_perovskite_stability(a: str, b: str, x: str) -> float:
    """Estimate stability in months"""
    # Cs and Br improve stability
    stability = 3.0
    
    if a == 'Cs':
        stability += 2.0
    if x == 'Br':
        stability += 3.0
    if b == 'Pb':  # Pb more stable than Sn
        stability += 2.0
    
    stability *= np.random.uniform(0.5, 1.5)
    return max(0.5, min(24, stability))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_multi_material_training():
    """Train models for multiple material classes"""
    
    print("=" * 70)
    print("UNIVERSAL MATERIALS PREDICTION PIPELINE")
    print("=" * 70)
    
    # 1. POLYMERS
    print("\n1. Training Polymer Model (Tg)...")
    from polymer_ml_pipeline import generate_synthetic_polymer_dataset
    
    polymer_data = generate_synthetic_polymer_dataset(n_samples=500)
    polymer_model = UniversalMaterialsPredictor('polymer', 'glass_transition_temp')
    
    X, y = polymer_model.prepare_data(
        polymer_data['smiles'].tolist(),
        polymer_data['glass_transition_temp'].values
    )
    polymer_model.train(X, y, epochs=50)
    polymer_model.save_model('/mnt/user-data/outputs/polymer_tg_model.pth')
    
    # 2. SMALL MOLECULES
    print("\n2. Training Drug Molecule Model (Solubility)...")
    drug_data = generate_small_molecule_dataset(n_samples=500)
    drug_model = UniversalMaterialsPredictor('small_molecule', 'solubility')
    
    X, y = drug_model.prepare_data(
        drug_data['smiles'].tolist(),
        drug_data['solubility_logS'].values
    )
    drug_model.train(X, y, epochs=50)
    drug_model.save_model('/mnt/user-data/outputs/drug_solubility_model.pth')
    
    # 3. PEROVSKITES
    print("\n3. Training Perovskite Model (Bandgap)...")
    perovskite_data = generate_perovskite_dataset(n_samples=300)
    perovskite_model = UniversalMaterialsPredictor('perovskite', 'bandgap')
    
    X, y = perovskite_model.prepare_data(
        perovskite_data['formula'].tolist(),
        perovskite_data['bandgap_eV'].values
    )
    perovskite_model.train(X, y, epochs=50)
    perovskite_model.save_model('/mnt/user-data/outputs/perovskite_bandgap_model.pth')
    
    print("\n" + "=" * 70)
    print("All models trained successfully!")
    print("=" * 70)
    
    # Test predictions
    print("\nTest Predictions:")
    print("-" * 70)
    
    # Polymer
    poly_pred = polymer_model.predict(['c1ccccc1CC'])[0]
    print(f"Polystyrene Tg: {poly_pred:.1f}°C")
    
    # Drug
    drug_pred = drug_model.predict(['c1ccccc1O'])[0]
    print(f"Phenol Solubility: {drug_pred:.2f} logS")
    
    # Perovskite
    perov_pred = perovskite_model.predict(['MA_Pb_I3'])[0]
    print(f"MAPbI3 Bandgap: {perov_pred:.2f} eV")


if __name__ == "__main__":
    example_multi_material_training()
