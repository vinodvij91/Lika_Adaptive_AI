"""
Task Classification Matrix for Vaccine Discovery Pipeline
==========================================================

This module provides a comprehensive breakdown of all computational tasks
in the vaccine discovery workflow, classified by their compute requirements
and optimal execution environment.

Hardware Routing Map:
- GPU_INTENSIVE: Deep learning, MD simulations (15-200x speedup)
- GPU_PREFERRED: Benefits from GPU but works well on CPU (2-6x speedup)
- CPU_INTENSIVE: Parallel processing, highly threaded workloads
- CPU_ONLY: Sequential tasks, I/O bound, no parallelization benefit
- HYBRID: Benefits from both CPU and GPU working together
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass


class TaskComputeType(Enum):
    GPU_INTENSIVE = "GPU_INTENSIVE"
    GPU_PREFERRED = "GPU_PREFERRED"
    CPU_INTENSIVE = "CPU_INTENSIVE"
    CPU_ONLY = "CPU_ONLY"
    HYBRID = "HYBRID"


TASK_CLASSIFICATION: Dict[str, Any] = {
    
    "genome_analysis": {
        "stage": 1,
        "stage_name": "Target Identification & Antigen Selection",
        "tasks": {
            "sequence_extraction": {
                "type": "CPU_ONLY",
                "reason": "File I/O and parsing",
                "cpu_cores": 1,
                "memory_gb": 2,
                "estimated_time_minutes": 1
            },
            "gene_annotation": {
                "type": "CPU_INTENSIVE",
                "reason": "BLAST searches, parallel annotation",
                "cpu_cores": 16,
                "memory_gb": 16,
                "estimated_time_minutes": 30,
                "tools": ["BLAST", "Prokka", "GeneMark"]
            },
            "protein_translation": {
                "type": "CPU_ONLY",
                "reason": "Simple translation, no parallelization benefit",
                "cpu_cores": 1,
                "memory_gb": 1,
                "estimated_time_minutes": 1
            }
        }
    },
    
    "protein_function_prediction": {
        "stage": 1,
        "stage_name": "Target Identification & Antigen Selection",
        "tasks": {
            "esm2_embedding": {
                "type": "GPU_INTENSIVE",
                "reason": "Transformer model, massive matrix operations",
                "gpu_memory_gb": 8,
                "cpu_cores": 4,
                "memory_gb": 16,
                "estimated_time_minutes": 5,
                "speedup_gpu_vs_cpu": "50x",
                "batch_size_gpu": 32,
                "batch_size_cpu": 1,
                "tools": ["BioNeMo ESM-2", "ESM-2 650M/3B"]
            },
            "functional_annotation": {
                "type": "CPU_INTENSIVE",
                "reason": "Database lookups, GO term assignment",
                "cpu_cores": 8,
                "memory_gb": 8,
                "estimated_time_minutes": 10
            }
        }
    },
    
    "structure_prediction": {
        "stage": 1,
        "stage_name": "Target Identification & Antigen Selection",
        "tasks": {
            "esmfold_prediction": {
                "type": "GPU_INTENSIVE",
                "reason": "Primary structure prediction - Deep learning inference, attention mechanisms, no MSA required",
                "gpu_memory_gb": 16,
                "cpu_cores": 8,
                "memory_gb": 32,
                "estimated_time_minutes": 15,
                "speedup_gpu_vs_cpu": "100x",
                "structures_per_hour_gpu": 4,
                "structures_per_hour_cpu": 0.04,
                "tools": ["ESMFold", "BioNeMo"],
                "priority": "PRIMARY",
                "supported_pipelines": ["drug_discovery", "vaccine_discovery", "materials_science"],
                "max_sequence_length": 400,
                "api_key_required": False,
                "notes": "Recommended for all structure predictions. Fast, free, and accurate."
            },
            "alphafold2_prediction": {
                "type": "GPU_INTENSIVE",
                "reason": "Alternative for long sequences (>400 residues) - Multi-stage deep learning, MSA generation",
                "gpu_memory_gb": 24,
                "cpu_cores": 16,
                "memory_gb": 64,
                "estimated_time_minutes": 45,
                "speedup_gpu_vs_cpu": "200x",
                "tools": ["AlphaFold2", "OpenFold"],
                "priority": "FALLBACK",
                "notes": "Use only when sequence exceeds ESMFold limit (400 residues) or requires MSA for accuracy"
            },
            "homology_modeling": {
                "type": "CPU_INTENSIVE",
                "reason": "Template search, alignment, energy minimization",
                "cpu_cores": 8,
                "memory_gb": 16,
                "estimated_time_minutes": 120,
                "tools": ["MODELLER", "SWISS-MODEL"]
            },
            "structure_quality_assessment": {
                "type": "CPU_ONLY",
                "reason": "Geometry checks, Ramachandran analysis",
                "cpu_cores": 1,
                "memory_gb": 2,
                "estimated_time_minutes": 2,
                "tools": ["MolProbity", "ProCheck"]
            }
        }
    },
    
    "conservation_analysis": {
        "stage": 1,
        "stage_name": "Target Identification & Antigen Selection",
        "tasks": {
            "multiple_sequence_alignment": {
                "type": "CPU_INTENSIVE",
                "reason": "Dynamic programming, parallelizable across sequences",
                "cpu_cores": 32,
                "memory_gb": 64,
                "estimated_time_minutes": 60,
                "sequences_supported": 10000,
                "tools": ["MAFFT", "Clustal Omega", "MUSCLE"]
            },
            "phylogenetic_analysis": {
                "type": "CPU_INTENSIVE",
                "reason": "Tree building, bootstrap analysis",
                "cpu_cores": 16,
                "memory_gb": 32,
                "estimated_time_minutes": 120,
                "tools": ["RAxML", "IQ-TREE", "FastTree"]
            },
            "conservation_scoring": {
                "type": "CPU_ONLY",
                "reason": "Simple statistical calculations",
                "cpu_cores": 4,
                "memory_gb": 4,
                "estimated_time_minutes": 5
            }
        }
    },
    
    "b_cell_epitope_prediction": {
        "stage": 2,
        "stage_name": "Epitope Prediction & Design",
        "tasks": {
            "linear_epitope_bepipred": {
                "type": "CPU_ONLY",
                "reason": "Rule-based prediction, no heavy computation",
                "cpu_cores": 1,
                "memory_gb": 2,
                "estimated_time_minutes": 1,
                "tools": ["BepiPred 3.0"]
            },
            "conformational_epitope_ellipro": {
                "type": "CPU_INTENSIVE",
                "reason": "Surface accessibility calculations",
                "cpu_cores": 8,
                "memory_gb": 8,
                "estimated_time_minutes": 10,
                "tools": ["ElliPro", "DiscoTope"]
            },
            "dl_based_epitope_prediction": {
                "type": "GPU_PREFERRED",
                "reason": "Neural network inference, benefits from GPU but works on CPU",
                "gpu_memory_gb": 4,
                "cpu_cores": 8,
                "memory_gb": 16,
                "estimated_time_minutes_gpu": 2,
                "estimated_time_minutes_cpu": 10,
                "speedup_gpu_vs_cpu": "5x"
            }
        }
    },
    
    "t_cell_epitope_prediction": {
        "stage": 2,
        "stage_name": "Epitope Prediction & Design",
        "tasks": {
            "netmhcpan_mhc1": {
                "type": "CPU_INTENSIVE",
                "reason": "Neural network per allele, highly parallelizable",
                "cpu_cores": 64,
                "memory_gb": 32,
                "estimated_time_minutes": 30,
                "alleles_per_batch": 100,
                "peptides_per_second_per_core": 1000,
                "tools": ["NetMHCpan 4.1"]
            },
            "netmhcpan_mhc1_gpu": {
                "type": "GPU_PREFERRED",
                "reason": "Batch neural network inference",
                "gpu_memory_gb": 8,
                "cpu_cores": 8,
                "memory_gb": 16,
                "estimated_time_minutes": 5,
                "speedup_gpu_vs_cpu": "6x",
                "batch_size": 10000
            },
            "netmhciipan_mhc2": {
                "type": "CPU_INTENSIVE",
                "reason": "Similar to MHC-I, CPU-based neural networks",
                "cpu_cores": 64,
                "memory_gb": 32,
                "estimated_time_minutes": 45,
                "tools": ["NetMHCIIpan 4.0"]
            },
            "population_coverage": {
                "type": "CPU_ONLY",
                "reason": "Statistical calculations",
                "cpu_cores": 4,
                "memory_gb": 4,
                "estimated_time_minutes": 5,
                "tools": ["IEDB Population Coverage"]
            },
            "tap_transport_prediction": {
                "type": "CPU_ONLY",
                "reason": "Simple scoring function",
                "cpu_cores": 1,
                "memory_gb": 1,
                "estimated_time_minutes": 1,
                "tools": ["NetCTL", "TAPPred"]
            },
            "proteasomal_cleavage": {
                "type": "CPU_ONLY",
                "reason": "Matrix-based scoring",
                "cpu_cores": 1,
                "memory_gb": 1,
                "estimated_time_minutes": 1,
                "tools": ["NetChop", "ProteaSMM"]
            }
        }
    },
    
    "protein_sequence_design": {
        "stage": 3,
        "stage_name": "Antigen Design & Optimization",
        "tasks": {
            "proteinmpnn_design": {
                "type": "GPU_INTENSIVE",
                "reason": "Message passing neural network, GPU-optimized",
                "gpu_memory_gb": 8,
                "cpu_cores": 4,
                "memory_gb": 16,
                "estimated_time_minutes": 10,
                "designs_per_minute_gpu": 100,
                "designs_per_minute_cpu": 2,
                "speedup_gpu_vs_cpu": "50x",
                "tools": ["ProteinMPNN", "BioNeMo"]
            },
            "rosetta_design": {
                "type": "CPU_INTENSIVE",
                "reason": "Monte Carlo sampling, parallelizable",
                "cpu_cores": 32,
                "memory_gb": 64,
                "estimated_time_minutes": 240,
                "tools": ["Rosetta", "RoseTTAFold"]
            },
            "stability_prediction": {
                "type": "GPU_PREFERRED",
                "reason": "Machine learning models",
                "gpu_memory_gb": 4,
                "cpu_cores": 4,
                "memory_gb": 8,
                "estimated_time_minutes_gpu": 2,
                "estimated_time_minutes_cpu": 10
            },
            "aggregation_prediction": {
                "type": "CPU_ONLY",
                "reason": "Sequence-based algorithms",
                "cpu_cores": 1,
                "memory_gb": 2,
                "estimated_time_minutes": 5,
                "tools": ["AGGRESCAN", "Zyggregator"]
            }
        }
    },
    
    "mrna_vaccine_design": {
        "stage": 3,
        "stage_name": "Antigen Design & Optimization",
        "tasks": {
            "codon_optimization": {
                "type": "CPU_ONLY",
                "reason": "Lookup table operations, no parallelization benefit",
                "cpu_cores": 1,
                "memory_gb": 1,
                "estimated_time_minutes": 1,
                "tools": ["Python CodonW", "OPTIMIZER"]
            },
            "rna_secondary_structure": {
                "type": "CPU_INTENSIVE",
                "reason": "Dynamic programming, can parallelize multiple sequences",
                "cpu_cores": 16,
                "memory_gb": 16,
                "estimated_time_minutes": 15,
                "tools": ["ViennaRNA", "RNAfold"]
            },
            "rna_secondary_structure_gpu": {
                "type": "GPU_PREFERRED",
                "reason": "Can be accelerated with GPU for large sequences",
                "gpu_memory_gb": 4,
                "cpu_cores": 4,
                "memory_gb": 8,
                "estimated_time_minutes": 3,
                "speedup_gpu_vs_cpu": "5x"
            },
            "utr_optimization": {
                "type": "CPU_INTENSIVE",
                "reason": "Sequence search and optimization",
                "cpu_cores": 8,
                "memory_gb": 8,
                "estimated_time_minutes": 30
            },
            "gc_content_adjustment": {
                "type": "CPU_ONLY",
                "reason": "Simple sequence manipulation",
                "cpu_cores": 1,
                "memory_gb": 1,
                "estimated_time_minutes": 2
            }
        }
    },
    
    "immune_simulation": {
        "stage": 4,
        "stage_name": "Immunogenicity Prediction & Validation",
        "tasks": {
            "c_immsim_simulation": {
                "type": "CPU_INTENSIVE",
                "reason": "Agent-based modeling, parallelizable parameter sweeps",
                "cpu_cores": 64,
                "memory_gb": 128,
                "estimated_time_minutes": 180,
                "simulations_parallel": 64,
                "tools": ["C-ImmSim"]
            },
            "immunogrid_simulation": {
                "type": "CPU_INTENSIVE",
                "reason": "Large-scale cellular automata",
                "cpu_cores": 32,
                "memory_gb": 64,
                "estimated_time_minutes": 240
            }
        }
    },
    
    "antibody_prediction": {
        "stage": 4,
        "stage_name": "Immunogenicity Prediction & Validation",
        "tasks": {
            "antibody_structure_prediction": {
                "type": "GPU_INTENSIVE",
                "reason": "AlphaFold-based antibody modeling",
                "gpu_memory_gb": 16,
                "cpu_cores": 8,
                "memory_gb": 32,
                "estimated_time_minutes": 30,
                "tools": ["AlphaFold-Multimer", "ABodyBuilder2"]
            },
            "antibody_antigen_docking": {
                "type": "HYBRID",
                "reason": "Benefits from GPU for scoring, CPU for conformational search",
                "gpu_memory_gb": 8,
                "cpu_cores": 32,
                "memory_gb": 64,
                "estimated_time_minutes": 60,
                "tools": ["HADDOCK", "ClusPro"]
            },
            "binding_affinity_prediction": {
                "type": "GPU_PREFERRED",
                "reason": "Machine learning models",
                "gpu_memory_gb": 4,
                "cpu_cores": 4,
                "memory_gb": 8,
                "estimated_time_minutes_gpu": 5,
                "estimated_time_minutes_cpu": 20
            }
        }
    },
    
    "safety_assessment": {
        "stage": 4,
        "stage_name": "Immunogenicity Prediction & Validation",
        "tasks": {
            "allergenicity_prediction": {
                "type": "CPU_ONLY",
                "reason": "Sequence comparison and simple ML",
                "cpu_cores": 4,
                "memory_gb": 4,
                "estimated_time_minutes": 5,
                "tools": ["AllerTop", "AlgPred"]
            },
            "autoimmunity_screen": {
                "type": "CPU_INTENSIVE",
                "reason": "BLAST against human proteome",
                "cpu_cores": 16,
                "memory_gb": 32,
                "estimated_time_minutes": 20,
                "tools": ["BLAST", "PSI-BLAST"]
            },
            "toxicity_prediction": {
                "type": "GPU_PREFERRED",
                "reason": "Deep learning models",
                "gpu_memory_gb": 2,
                "cpu_cores": 2,
                "memory_gb": 4,
                "estimated_time_minutes_gpu": 1,
                "estimated_time_minutes_cpu": 5,
                "tools": ["ToxIBTL", "DeepTox"]
            }
        }
    },
    
    "molecular_dynamics": {
        "stage": 5,
        "stage_name": "Advanced Analysis",
        "tasks": {
            "md_simulation_gpu": {
                "type": "GPU_INTENSIVE",
                "reason": "Force calculations, highly parallel",
                "gpu_memory_gb": 16,
                "cpu_cores": 8,
                "memory_gb": 32,
                "estimated_time_minutes_per_ns": 12,
                "speedup_gpu_vs_cpu": "100x",
                "ns_per_day_gpu": 120,
                "ns_per_day_cpu": 1.2,
                "tools": ["OpenMM", "GROMACS", "AMBER"]
            },
            "md_simulation_cpu": {
                "type": "CPU_INTENSIVE",
                "reason": "CPU-only MD, very slow",
                "cpu_cores": 64,
                "memory_gb": 128,
                "estimated_time_minutes_per_ns": 1200,
                "note": "Not recommended, 100x slower than GPU"
            },
            "trajectory_analysis": {
                "type": "CPU_INTENSIVE",
                "reason": "RMSD, RMSF calculations, parallelizable",
                "cpu_cores": 16,
                "memory_gb": 32,
                "estimated_time_minutes": 30,
                "tools": ["MDAnalysis", "CPPTRAJ"]
            },
            "free_energy_calculations": {
                "type": "HYBRID",
                "reason": "MD simulations (GPU) + statistical analysis (CPU)",
                "gpu_memory_gb": 16,
                "cpu_cores": 32,
                "memory_gb": 64,
                "estimated_time_minutes": 480,
                "tools": ["GROMACS", "FEP+"]
            }
        }
    },
    
    "visualization": {
        "stage": 5,
        "stage_name": "Advanced Analysis",
        "tasks": {
            "structure_rendering": {
                "type": "CPU_ONLY",
                "reason": "PyMOL, ChimeraX rendering",
                "cpu_cores": 4,
                "memory_gb": 8,
                "estimated_time_minutes": 5,
                "tools": ["PyMOL", "ChimeraX", "VMD"]
            },
            "surface_analysis": {
                "type": "CPU_INTENSIVE",
                "reason": "SASA calculations, electrostatics",
                "cpu_cores": 8,
                "memory_gb": 16,
                "estimated_time_minutes": 10,
                "tools": ["APBS", "PDB2PQR"]
            },
            "interactive_visualization": {
                "type": "CPU_ONLY",
                "reason": "Web-based viewers",
                "cpu_cores": 1,
                "memory_gb": 2,
                "estimated_time_minutes": 1,
                "tools": ["Mol*", "NGL Viewer"]
            }
        }
    }
}


HARDWARE_REQUIREMENTS: Dict[str, Any] = {
    
    "minimum_configuration": {
        "description": "Can run pipeline, but slow for GPU-intensive tasks",
        "cpu": {
            "cores": 16,
            "frequency_ghz": 2.5,
            "recommended": "Intel Xeon or AMD EPYC"
        },
        "memory_gb": 64,
        "storage_tb": 2,
        "gpu": "Optional (pipeline works CPU-only)",
        "estimated_pipeline_time_hours": 48
    },
    
    "recommended_configuration": {
        "description": "Balanced performance for most vaccine discovery projects",
        "cpu": {
            "cores": 32,
            "frequency_ghz": 3.0,
            "recommended": "AMD EPYC 7003 or Intel Xeon Scalable"
        },
        "memory_gb": 128,
        "storage_tb": 5,
        "gpu": {
            "model": "NVIDIA RTX 4090 or A4000",
            "vram_gb": 24,
            "count": 1
        },
        "estimated_pipeline_time_hours": 8
    },
    
    "high_performance_configuration": {
        "description": "Fast turnaround for multiple projects",
        "cpu": {
            "cores": 64,
            "frequency_ghz": 3.5,
            "recommended": "AMD EPYC 9004 or Intel Xeon Platinum"
        },
        "memory_gb": 512,
        "storage_tb": 10,
        "gpu": {
            "model": "NVIDIA A100 80GB",
            "vram_gb": 80,
            "count": 4
        },
        "estimated_pipeline_time_hours": 3
    },
    
    "enterprise_configuration": {
        "description": "Maximum throughput, multiple concurrent projects",
        "cpu": {
            "cores": 128,
            "frequency_ghz": 3.0,
            "recommended": "Dual AMD EPYC 9004"
        },
        "memory_gb": 1024,
        "storage_tb": 50,
        "gpu": {
            "model": "NVIDIA H100 80GB",
            "vram_gb": 80,
            "count": 8
        },
        "cloud_burst": {
            "cpu_instances": "100x c6a.16xlarge",
            "gpu_instances": "10x p4d.24xlarge"
        },
        "estimated_pipeline_time_hours": 1.5
    }
}


COST_ANALYSIS: Dict[str, Any] = {
    
    "gpu_vs_cpu_comparison": {
        "structure_prediction_1000_proteins": {
            "gpu_time_hours": 250,
            "cpu_time_hours": 25000,
            "gpu_cost_usd": 1250,
            "cpu_cost_usd": 5000,
            "gpu_savings_pct": 75,
            "note": "100x faster on GPU, 75% cost savings despite higher $/hour"
        },
        "md_simulation_100ns": {
            "gpu_time_hours": 10,
            "cpu_time_hours": 1000,
            "gpu_cost_usd": 50,
            "cpu_cost_usd": 2000,
            "gpu_savings_pct": 97.5,
            "note": "MD is where GPU truly shines - 100x speedup"
        },
        "epitope_prediction_1000_sequences": {
            "gpu_time_hours": 5,
            "cpu_time_hours": 30,
            "gpu_cost_usd": 25,
            "cpu_cost_usd": 60,
            "gpu_savings_pct": 58,
            "note": "Moderate GPU benefit, CPU is still viable"
        }
    },
    
    "annual_cost_estimates": {
        "on_premise_gpu_cluster": {
            "hardware_capex": {
                "4x_a100_80gb": 80000,
                "servers": 20000,
                "networking": 10000,
                "storage": 30000,
                "total": 140000
            },
            "annual_opex": {
                "power_cooling": 15000,
                "maintenance": 10000,
                "personnel": 150000,
                "total": 175000
            },
            "total_year_1": 315000,
            "total_year_2_onwards": 175000
        },
        "cloud_hybrid": {
            "reserved_cpu": 50000,
            "spot_gpu": 30000,
            "storage": 12000,
            "personnel": 100000,
            "total_annual": 192000,
            "note": "More flexible, lower upfront cost"
        }
    }
}


class TaskRouter:
    """
    Routes computational tasks to optimal hardware based on the classification matrix.
    """
    
    def __init__(self, available_gpus: int = 0, available_cpu_cores: int = 8, 
                 memory_gb: float = 16.0, gpu_memory_gb: float = 0.0):
        self.available_gpus = available_gpus
        self.available_cpu_cores = available_cpu_cores
        self.memory_gb = memory_gb
        self.gpu_memory_gb = gpu_memory_gb
    
    def get_task_info(self, category: str, task_name: str) -> Optional[Dict[str, Any]]:
        """Get task information from the classification matrix."""
        if category in TASK_CLASSIFICATION:
            tasks = TASK_CLASSIFICATION[category].get("tasks", {})
            return tasks.get(task_name)
        return None
    
    def get_optimal_target(self, category: str, task_name: str) -> str:
        """
        Determine optimal execution target (GPU or CPU) for a task.
        
        Returns:
            'gpu' if GPU is optimal and available
            'cpu' if CPU is optimal or GPU not available
            'hybrid' for tasks that benefit from both
        """
        task_info = self.get_task_info(category, task_name)
        if not task_info:
            return "cpu"
        
        task_type = task_info.get("type", "CPU_ONLY")
        
        if task_type == "GPU_INTENSIVE":
            if self.available_gpus > 0:
                gpu_mem_required = task_info.get("gpu_memory_gb", 8)
                if self.gpu_memory_gb >= gpu_mem_required:
                    return "gpu"
            return "cpu"
        
        elif task_type == "GPU_PREFERRED":
            if self.available_gpus > 0:
                return "gpu"
            return "cpu"
        
        elif task_type == "HYBRID":
            if self.available_gpus > 0:
                return "hybrid"
            return "cpu"
        
        else:
            return "cpu"
    
    def estimate_time(self, category: str, task_name: str, 
                      target: Optional[str] = None) -> float:
        """
        Estimate execution time in minutes for a task.
        
        Args:
            category: Task category
            task_name: Specific task name
            target: Execution target ('gpu' or 'cpu'), auto-detected if None
            
        Returns:
            Estimated time in minutes
        """
        task_info = self.get_task_info(category, task_name)
        if not task_info:
            return 60.0
        
        if target is None:
            target = self.get_optimal_target(category, task_name)
        
        if target == "gpu":
            return task_info.get("estimated_time_minutes_gpu", 
                               task_info.get("estimated_time_minutes", 60))
        else:
            return task_info.get("estimated_time_minutes_cpu",
                               task_info.get("estimated_time_minutes", 60))
    
    def get_resource_requirements(self, category: str, task_name: str) -> Dict[str, Any]:
        """Get resource requirements for a task."""
        task_info = self.get_task_info(category, task_name)
        if not task_info:
            return {"cpu_cores": 1, "memory_gb": 2}
        
        return {
            "cpu_cores": task_info.get("cpu_cores", 1),
            "memory_gb": task_info.get("memory_gb", 2),
            "gpu_memory_gb": task_info.get("gpu_memory_gb", 0),
            "task_type": task_info.get("type", "CPU_ONLY")
        }
    
    def get_all_tasks_by_type(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all tasks organized by compute type.
        
        Returns:
            Dict with keys: GPU_INTENSIVE, GPU_PREFERRED, CPU_INTENSIVE, CPU_ONLY, HYBRID
        """
        result: Dict[str, List[Dict[str, str]]] = {
            "GPU_INTENSIVE": [],
            "GPU_PREFERRED": [],
            "CPU_INTENSIVE": [],
            "CPU_ONLY": [],
            "HYBRID": []
        }
        
        for category, category_data in TASK_CLASSIFICATION.items():
            tasks = category_data.get("tasks", {})
            stage_name = category_data.get("stage_name", "Unknown")
            
            for task_name, task_info in tasks.items():
                task_type = task_info.get("type", "CPU_ONLY")
                if task_type in result:
                    result[task_type].append({
                        "category": category,
                        "task": task_name,
                        "stage": stage_name,
                        "reason": task_info.get("reason", ""),
                        "speedup": task_info.get("speedup_gpu_vs_cpu", "N/A"),
                        "tools": ", ".join(task_info.get("tools", []))
                    })
        
        return result
    
    def get_stage_summary(self) -> List[Dict[str, Any]]:
        """Get summary of tasks organized by pipeline stage."""
        stages: Dict[int, Dict[str, Any]] = {}
        
        for category, category_data in TASK_CLASSIFICATION.items():
            stage_num = category_data.get("stage", 0)
            stage_name = category_data.get("stage_name", "Unknown")
            
            if stage_num not in stages:
                stages[stage_num] = {
                    "stage": stage_num,
                    "name": stage_name,
                    "categories": [],
                    "gpu_intensive_count": 0,
                    "gpu_preferred_count": 0,
                    "cpu_intensive_count": 0,
                    "cpu_only_count": 0,
                    "hybrid_count": 0,
                    "total_estimated_time_gpu_minutes": 0,
                    "total_estimated_time_cpu_minutes": 0
                }
            
            stages[stage_num]["categories"].append(category)
            
            tasks = category_data.get("tasks", {})
            for task_name, task_info in tasks.items():
                task_type = task_info.get("type", "CPU_ONLY")
                
                if task_type == "GPU_INTENSIVE":
                    stages[stage_num]["gpu_intensive_count"] += 1
                elif task_type == "GPU_PREFERRED":
                    stages[stage_num]["gpu_preferred_count"] += 1
                elif task_type == "CPU_INTENSIVE":
                    stages[stage_num]["cpu_intensive_count"] += 1
                elif task_type == "CPU_ONLY":
                    stages[stage_num]["cpu_only_count"] += 1
                elif task_type == "HYBRID":
                    stages[stage_num]["hybrid_count"] += 1
                
                gpu_time = task_info.get("estimated_time_minutes_gpu",
                                         task_info.get("estimated_time_minutes", 0))
                cpu_time = task_info.get("estimated_time_minutes_cpu",
                                         task_info.get("estimated_time_minutes", 0))
                
                stages[stage_num]["total_estimated_time_gpu_minutes"] += gpu_time
                stages[stage_num]["total_estimated_time_cpu_minutes"] += cpu_time
        
        return sorted(stages.values(), key=lambda x: x["stage"])


def get_task_classification_summary() -> Dict[str, Any]:
    """
    Get a complete summary of the task classification matrix.
    
    Returns:
        Dict containing:
        - tasks_by_type: Tasks organized by compute type
        - stages: Pipeline stages with task counts
        - hardware_requirements: Recommended hardware configurations
        - cost_analysis: Cost comparison data
    """
    router = TaskRouter()
    
    return {
        "tasks_by_type": router.get_all_tasks_by_type(),
        "stages": router.get_stage_summary(),
        "hardware_requirements": HARDWARE_REQUIREMENTS,
        "cost_analysis": COST_ANALYSIS,
        "total_categories": len(TASK_CLASSIFICATION),
        "total_tasks": sum(
            len(cat.get("tasks", {})) 
            for cat in TASK_CLASSIFICATION.values()
        )
    }


if __name__ == "__main__":
    import json
    
    print("=" * 80)
    print("VACCINE DISCOVERY TASK CLASSIFICATION MATRIX")
    print("=" * 80)
    
    summary = get_task_classification_summary()
    
    print(f"\nTotal Categories: {summary['total_categories']}")
    print(f"Total Tasks: {summary['total_tasks']}")
    
    print("\n" + "=" * 80)
    print("TASKS BY COMPUTE TYPE")
    print("=" * 80)
    
    for task_type, tasks in summary["tasks_by_type"].items():
        print(f"\n{task_type} ({len(tasks)} tasks):")
        for task in tasks[:3]:
            print(f"  - {task['category']}/{task['task']}: {task['reason']}")
        if len(tasks) > 3:
            print(f"  ... and {len(tasks) - 3} more")
    
    print("\n" + "=" * 80)
    print("PIPELINE STAGES")
    print("=" * 80)
    
    for stage in summary["stages"]:
        print(f"\nStage {stage['stage']}: {stage['name']}")
        print(f"  GPU-Intensive: {stage['gpu_intensive_count']}, "
              f"GPU-Preferred: {stage['gpu_preferred_count']}, "
              f"CPU-Intensive: {stage['cpu_intensive_count']}, "
              f"CPU-Only: {stage['cpu_only_count']}")
