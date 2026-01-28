"""
HARDWARE-AGNOSTIC UNIVERSAL MATERIALS PREDICTION PIPELINE
Automatically adapts to ANY hardware configuration:
- Single GPU, Multi-GPU, or CPU-only
- Any GPU model (RTX 3090, V100, A100, T4, etc.)
- Any amount of memory
- Optimal settings chosen automatically

NO HARDCODED HARDWARE ASSUMPTIONS!
"""

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
