"""
Advanced Polymer ML Pipeline
Supports real datasets (PolyInfo, PI1M) and transfer learning
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
import requests
from pathlib import Path


class PolyInfoDataLoader:
    """Load and process polymer data from various sources"""
    
    def __init__(self, cache_dir: str = "./polymer_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_polyinfo_style_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load PolyInfo-style dataset
        
        Real PolyInfo data format typically includes:
        - Polymer name
        - SMILES representation
        - Glass transition temperature (Tg)
        - Melting temperature (Tm)
        - Other properties
        
        For demonstration, we'll create a realistic synthetic dataset
        """
        
        if filepath and Path(filepath).exists():
            return pd.read_csv(filepath)
        
        # Create realistic polymer dataset based on known polymers
        polymers = []
        
        # Common commercial polymers with realistic properties
        polymer_data = [
            # Polymer name, SMILES (repeating unit), Tg (°C), Tm (°C), Density (g/cm³)
            ("Polyethylene (HDPE)", "CC", -120, 135, 0.95),
            ("Polypropylene (isotactic)", "CC(C)", -20, 165, 0.90),
            ("Polystyrene", "c1ccccc1CC", 100, None, 1.05),
            ("Poly(methyl methacrylate)", "CC(C)(C(=O)OC)C", 105, None, 1.18),
            ("Polyethylene terephthalate", "c1ccc(C(=O)OCC)cc1", 70, 265, 1.38),
            ("Polycarbonate", "c1ccc(OC(=O)Oc2ccc(C(C)(C))cc2)cc1", 150, None, 1.20),
            ("Nylon 6", "CCCCCC(=O)N", 50, 220, 1.14),
            ("Nylon 6,6", "CCCCCCNC(=O)CCCCC(=O)N", 60, 265, 1.14),
            ("Polyvinyl chloride", "CCCl", 80, None, 1.38),
            ("Polytetrafluoroethylene", "C(F)(F)C(F)(F)", 115, 327, 2.20),
            ("Poly(vinyl alcohol)", "CCO", 85, None, 1.19),
            ("Polyethylene oxide", "CCO", -67, 65, 1.13),
            ("Polypropylene oxide", "CC(C)O", -75, None, 1.00),
            ("Polybutadiene", "C=CCC", -100, None, 0.90),
            ("Polyisoprene", "CC(=C)CC", -70, 28, 0.93),
            ("Styrene-butadiene rubber", "c1ccccc1CCC=C", -50, None, 0.94),
            ("Polylactic acid", "CC(O)C(=O)", 60, 175, 1.25),
            ("Polycaprolactone", "CCCCCC(=O)O", -60, 60, 1.15),
            ("Polyurethane (generic)", "CC(=O)NCCOC(=O)NC", -50, None, 1.20),
            ("Epoxy resin", "c1ccc(OCC2CO2)cc1", 150, None, 1.16),
        ]
        
        # Add variations with noise to increase dataset size
        for name, smiles, tg, tm, density in polymer_data:
            # Add base entry
            polymers.append({
                'polymer_name': name,
                'smiles': smiles,
                'glass_transition_temp': tg,
                'melting_temp': tm,
                'density': density,
                'source': 'literature'
            })
            
            # Add variations (simulating different measurement conditions)
            for i in range(5):
                tg_var = tg + np.random.normal(0, 5) if tg is not None else None
                tm_var = tm + np.random.normal(0, 10) if tm is not None else None
                density_var = density + np.random.normal(0, 0.02)
                
                polymers.append({
                    'polymer_name': f"{name} (variant {i+1})",
                    'smiles': smiles,
                    'glass_transition_temp': tg_var,
                    'melting_temp': tm_var,
                    'density': density_var,
                    'source': 'synthetic_variation'
                })
        
        df = pd.DataFrame(polymers)
        return df
    
    def load_pi1m_style_data(self) -> pd.DataFrame:
        """
        Load PI1M (Polymer Informatics 1 Million) style dataset
        
        PI1M is a large-scale polymer property database
        Here we create a representative sample
        """
        
        # Generate diverse polymer structures
        n_samples = 1000
        
        # Building blocks for polymer generation
        backbones = ['CC', 'CCC', 'CCCC', 'C=C', 'c1ccccc1C']
        functionals = ['', 'O', 'Cl', 'F', 'N', 'C(=O)O', 'C(=O)N']
        
        polymers = []
        np.random.seed(42)
        
        for i in range(n_samples):
            backbone = np.random.choice(backbones)
            functional = np.random.choice(functionals)
            
            smiles = backbone + functional
            
            # Estimate properties based on structure
            # This is a simplified model
            tg = self._estimate_tg(smiles)
            density = self._estimate_density(smiles)
            
            polymers.append({
                'polymer_id': f'PI1M_{i:06d}',
                'smiles': smiles,
                'glass_transition_temp': tg,
                'density': density,
                'source': 'pi1m_synthetic'
            })
        
        return pd.DataFrame(polymers)
    
    def _estimate_tg(self, smiles: str) -> float:
        """Estimate Tg based on structural features"""
        base_tg = 0
        
        # Aromatic groups increase Tg
        if 'c1ccccc1' in smiles:
            base_tg += 100
        
        # Halogens increase Tg
        base_tg += smiles.count('Cl') * 30
        base_tg += smiles.count('F') * 20
        
        # Oxygen in backbone (ether) decreases Tg
        if 'O' in smiles and '=' not in smiles:
            base_tg -= 40
        
        # Carbonyl groups increase Tg
        base_tg += smiles.count('C(=O)') * 25
        
        # Nitrogen (H-bonding) increases Tg
        base_tg += smiles.count('N') * 30
        
        # Add noise
        base_tg += np.random.normal(0, 15)
        
        return base_tg
    
    def _estimate_density(self, smiles: str) -> float:
        """Estimate density based on structural features"""
        base_density = 1.0
        
        # Aromatic groups increase density
        if 'c1ccccc1' in smiles:
            base_density += 0.1
        
        # Halogens increase density
        base_density += smiles.count('Cl') * 0.15
        base_density += smiles.count('F') * 0.3
        
        # Add noise
        base_density += np.random.normal(0, 0.05)
        
        return max(0.85, min(2.0, base_density))
    
    def merge_datasets(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """Merge multiple polymer datasets"""
        merged = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates based on SMILES
        merged = merged.drop_duplicates(subset=['smiles'], keep='first')
        
        return merged


class TransferLearningPredictor:
    """
    Transfer learning for polymer property prediction
    
    Allows pre-training on large dataset and fine-tuning on specific properties
    """
    
    def __init__(self, base_model_path: Optional[str] = None):
        from polymer_ml_pipeline import PolymerPropertyPredictor
        
        self.predictor = PolymerPropertyPredictor()
        
        if base_model_path and Path(base_model_path).exists():
            print(f"Loading pre-trained model from {base_model_path}")
            self.predictor.load_model(base_model_path)
            self.is_pretrained = True
        else:
            self.is_pretrained = False
    
    def pretrain(self, smiles_list: List[str], properties: np.ndarray,
                 epochs: int = 100, **kwargs):
        """Pre-train on large dataset"""
        print("Pre-training on large dataset...")
        
        X, y = self.predictor.prepare_data(smiles_list, properties)
        self.predictor.train(X, y, epochs=epochs, **kwargs)
        
        self.is_pretrained = True
        print("Pre-training complete!")
    
    def fine_tune(self, smiles_list: List[str], properties: np.ndarray,
                  epochs: int = 50, learning_rate: float = 0.0001, **kwargs):
        """Fine-tune on specific dataset"""
        
        if not self.is_pretrained:
            print("Warning: No pre-trained model. Using regular training.")
            return self.pretrain(smiles_list, properties, epochs=epochs, **kwargs)
        
        print("Fine-tuning on specific dataset...")
        
        X, y = self.predictor.prepare_data(smiles_list, properties)
        
        # Use lower learning rate for fine-tuning
        self.predictor.train(
            X, y,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs
        )
        
        print("Fine-tuning complete!")
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions"""
        return self.predictor.predict(smiles_list)
    
    def save(self, filepath: str):
        """Save model"""
        self.predictor.save_model(filepath)


def create_training_pipeline():
    """Complete training pipeline with real-world workflow"""
    
    print("=" * 70)
    print("Advanced Polymer ML Training Pipeline")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading polymer datasets...")
    loader = PolyInfoDataLoader()
    
    polyinfo_data = loader.load_polyinfo_style_data()
    print(f"   Loaded {len(polyinfo_data)} samples from PolyInfo-style data")
    
    pi1m_data = loader.load_pi1m_style_data()
    print(f"   Loaded {len(pi1m_data)} samples from PI1M-style data")
    
    # 2. Merge datasets
    print("\n2. Merging datasets...")
    all_data = loader.merge_datasets(polyinfo_data, pi1m_data)
    print(f"   Total samples after merging: {len(all_data)}")
    
    # Remove entries with missing Tg values
    all_data = all_data.dropna(subset=['glass_transition_temp'])
    print(f"   Samples with Tg values: {len(all_data)}")
    
    # 3. Pre-train on large dataset
    print("\n3. Pre-training on combined dataset...")
    transfer_model = TransferLearningPredictor()
    
    transfer_model.pretrain(
        all_data['smiles'].tolist(),
        all_data['glass_transition_temp'].values,
        epochs=80,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Save pre-trained model
    transfer_model.save('/mnt/user-data/outputs/pretrained_polymer_model.pth')
    print("   Pre-trained model saved!")
    
    # 4. Fine-tune on specific polymer class (e.g., only polyesters)
    print("\n4. Fine-tuning on specific polymer class...")
    
    # Filter for polyesters (contain C(=O)O pattern)
    polyester_data = all_data[all_data['smiles'].str.contains('C\(=O\)O', regex=True)]
    print(f"   Found {len(polyester_data)} polyester samples")
    
    if len(polyester_data) > 10:
        transfer_model.fine_tune(
            polyester_data['smiles'].tolist(),
            polyester_data['glass_transition_temp'].values,
            epochs=30,
            learning_rate=0.0001
        )
        
        transfer_model.save('/mnt/user-data/outputs/polyester_finetuned_model.pth')
        print("   Fine-tuned model saved!")
    
    # 5. Evaluate and compare
    print("\n5. Model Evaluation")
    print("-" * 70)
    
    # Test on some examples
    test_polymers = [
        ("Polyethylene terephthalate", "c1ccc(C(=O)OCCOC(=O))cc1"),
        ("Polycarbonate", "c1ccc(OC(=O)Oc2ccc(C(C)(C))cc2)cc1"),
        ("PMMA", "CC(C)(C(=O)OC)C"),
        ("Polystyrene", "c1ccccc1CC"),
    ]
    
    print("\nPredictions on test polymers:")
    for name, smiles in test_polymers:
        pred = transfer_model.predict([smiles])[0]
        print(f"  {name:30s} -> Tg = {pred:6.1f} °C")
    
    # 6. Create summary report
    print("\n6. Creating summary report...")
    
    report = {
        'dataset_summary': {
            'total_samples': len(all_data),
            'polyinfo_samples': len(polyinfo_data),
            'pi1m_samples': len(pi1m_data),
            'unique_polymers': all_data['smiles'].nunique()
        },
        'model_info': {
            'architecture': 'Feed-forward Neural Network',
            'input_features': 'Molecular descriptors + Morgan fingerprints',
            'training_approach': 'Transfer learning with fine-tuning'
        },
        'test_predictions': [
            {'name': name, 'smiles': smiles, 'predicted_tg': float(transfer_model.predict([smiles])[0])}
            for name, smiles in test_polymers
        ]
    }
    
    with open('/mnt/user-data/outputs/pipeline_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save datasets
    all_data.to_csv('/mnt/user-data/outputs/combined_polymer_dataset.csv', index=False)
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - pretrained_polymer_model.pth: Pre-trained model")
    print("  - polyester_finetuned_model.pth: Fine-tuned model")
    print("  - combined_polymer_dataset.csv: Training data")
    print("  - pipeline_report.json: Summary report")


if __name__ == "__main__":
    create_training_pipeline()
