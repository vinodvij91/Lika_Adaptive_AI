# Lika Sciences Platform

## Overview
Lika Sciences is a scientific SaaS platform designed to accelerate drug discovery and molecular research. It provides tools for managing molecule registries, research campaigns, and SMILES data, aiming to streamline scientific workflows, enable data-driven decisions, and offer robust data visualization and professional scientific presentation. The platform supports both drug discovery and materials science domains, with future extensibility for advanced computational chemistry and quantum computing.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
The platform utilizes a full-stack TypeScript architecture. The frontend is built with React, Vite, shadcn/ui, and Tailwind CSS, using TanStack React Query for state management. The backend is developed with Node.js, Express, TypeScript, and PostgreSQL, employing Drizzle ORM for database interactions. Shared schemas and types are managed in a `shared/` directory.

### Key Domain Features
- **Molecule & Project Management**: Handles chemical compounds (SMILES notation) and organizes them into projects and research campaigns across various modalities.
- **Data Import & Curation**: Supports batch import of SMILES with validation, duplicate detection, and provides curated domain-aware SMILES libraries.
- **Scientific Terminology**: Integrates professional drug discovery and materials science terminology.
- **Compute Nodes**: Manages multi-provider infrastructure (Hetzner, Vast.ai, AWS, Azure, GCP, On-Prem) for ML, docking, quantum, and agent workloads, including SSH key management.
- **Advanced Scientific Modules**: Includes Multi-Target SAR, Translational Medicine, Wet-Lab Integration, and Literature & IP for drug discovery. For materials science, it supports polymers, crystals, composites, catalysts, membranes, and coatings with structure-first and property-first discovery modes.
- **Collaboration & Templates**: Supports multi-organization collaboration with role-based access and provides disease-specific pipeline templates.
- **Enterprise-Scale Processing Infrastructure**: Features a robust processing jobs system for orchestration, status tracking, and fault tolerance, including artifact storage and ingestion. Precomputed aggregations provide dashboard data and per-variant metrics.

### API Design
The platform uses RESTful API endpoints under `/api/`, with specific `/api/agent/` endpoints for AI interaction and service account roles.

### Design System
Adheres to Material Design principles with Carbon Design data patterns, using Inter and JetBrains Mono fonts, and a custom Tailwind theme for scientific visualization.

### Python Compute Pipelines
The platform integrates production-grade Python pipelines for both drug discovery and materials science, designed for executing computational workloads.
- **Drug Discovery Pipeline**: Features Dask for distributed computing, RAPIDS cuML for GPU-accelerated ML, PyTorch mixed precision training, AutoDock Vina integration, and XGBoost with GPU acceleration. Supports steps like SMILES validation, fingerprint generation, property calculation, ML prediction, Vina docking, scoring, and rule filtering.
- **Vaccine Discovery Pipeline** (`compute/vaccine_discovery_pipeline.py`): GPU-agnostic pipeline with automatic hardware detection (NVIDIA CUDA, AMD ROCm, Apple Metal, CPU). Intelligently routes tasks between CPU and GPU based on workload characteristics.
- **Complete Vaccine Pipeline** (`compute/complete_vaccine_pipeline.py`): Production-ready vaccine design with all bioinformatics tool integrations:
  - DSSP surface analysis and secondary structure
  - DiscoTope-3.0 B-cell epitope prediction
  - NetMHCpan-4.1 T-cell epitope prediction (MHC-I/II)
  - MAFFT conservation analysis
  - Linker design for multi-epitope constructs
  - JCat codon optimization
  - ViennaRNA secondary structure prediction
  - End-to-end pipeline for protein subunit, mRNA, and multi-epitope vaccines
  - **Task Classification Matrix** (`compute/task_classification_matrix.py`): Comprehensive hardware routing map for all vaccine discovery tasks organized by pipeline stages:
    - **Stage 1 - Target Identification**: Genome analysis, protein function prediction, structure prediction, conservation analysis
    - **Stage 2 - Epitope Prediction**: B-cell epitopes (linear, conformational), T-cell epitopes (MHC-I/II binding, population coverage)
    - **Stage 3 - Antigen Design**: Protein sequence design (ProteinMPNN, Rosetta), mRNA vaccine design (codon optimization, UTR optimization)
    - **Stage 4 - Immunogenicity**: Immune simulation (C-ImmSim), antibody prediction, safety assessment
    - **Stage 5 - Advanced Analysis**: Molecular dynamics, visualization
  - **Task Compute Types**:
    - **GPU_INTENSIVE** (15-200x speedup): Structure prediction (ESMFold/AlphaFold2), MD simulation, ProteinMPNN design
    - **GPU_PREFERRED** (2-6x speedup): MHC binding prediction, toxicity prediction, RNA secondary structure
    - **CPU_INTENSIVE**: Epitope prediction (NetMHCpan), sequence alignment, Rosetta design, immune simulation
    - **CPU_ONLY**: Codon optimization, file I/O, structure quality assessment
    - **HYBRID**: Antibody-antigen docking, free energy calculations
  - **PDB File Support**: All vaccine pipeline endpoints now support PDB file input. Upload PDB files to extract protein sequences automatically for pipeline processing.
  - **API Endpoints**:
    - `POST /api/compute/vaccine/upload-pdb` - Upload PDB structure file (extracts sequence automatically)
    - `GET /api/compute/vaccine/pdb-uploads` - List uploaded PDB files for vaccine discovery
    - `POST /api/compute/vaccine/pipeline` - Run full vaccine discovery pipeline (accepts `pdbFileId` parameter)
    - `POST /api/compute/vaccine/complete-pipeline` - Run complete pipeline with all bioinformatics tools (DSSP, DiscoTope, NetMHCpan, MAFFT, etc.)
    - `GET /api/compute/vaccine/complete-task-registry` - Get complete pipeline task registry with all tools and status
    - `POST /api/compute/vaccine/structure` - Predict protein structure (GPU-intensive, accepts `pdbFileId` parameter)
    - `POST /api/compute/vaccine/epitopes` - Predict MHC binding/epitopes (CPU-intensive)
    - `POST /api/compute/vaccine/codon-optimize` - Codon optimization (CPU-only)
    - `POST /api/compute/vaccine/mrna-design` - mRNA construct design (CPU-intensive)
    - `POST /api/compute/vaccine/md-simulation` - Molecular dynamics (GPU-intensive, accepts `pdbFileId` parameter)
    - `GET /api/compute/vaccine/hardware` - Get hardware performance report
    - `GET /api/compute/vaccine/task-matrix` - Get full task classification matrix with hardware requirements and cost analysis
- **SandboxAQ AQAffinity Integration** (`compute/aqaffinity_integration.py`): Open-source AI model for fast, structure-free prediction of protein-ligand binding affinities, built on OpenFold3. **Note:** Currently in simulation mode - returns deterministic predictions based on input hashing. Real AQAffinity integration requires installing the package (`pip install git+https://huggingface.co/SandboxAQ/aqaffinity`) and GPU compute resources.
  - **Key Features**:
    - Structure-free prediction (sequence + SMILES only, no protein structure required)
    - Fast "fail fast" drug candidate screening
    - Apache 2.0 licensed (free for academic & commercial use)
    - Trained on GOSTAR assay database
  - **Supported Pipelines**:
    - **Drug Discovery**: Screen drug candidates against therapeutic targets
    - **Vaccine Discovery**: Complementary epitope-MHC binding analysis
    - **Materials Discovery**: Catalyst-substrate and polymer binding prediction
  - **API Endpoints**:
    - `POST /api/compute/aqaffinity/predict` - Single protein-ligand binding affinity prediction
    - `POST /api/compute/aqaffinity/batch-predict` - Batch predictions for multiple ligands against one target
    - `POST /api/compute/aqaffinity/screen-library` - Screen compound library against target with hit classification
    - `GET /api/compute/aqaffinity/info` - Get AQAffinity integration info and capabilities
  - **Input/Output**:
    - Input: `proteinSequence` (amino acid sequence), `ligandSmiles` (SMILES string), `pipeline` (discovery type)
    - Output: `predictedAffinity` (IC50 in nM), `confidenceScore` (0-1), `isStrongBinder` (boolean)
  - **Installation**: `pip install git+https://huggingface.co/SandboxAQ/aqaffinity`
  - **References**: https://huggingface.co/SandboxAQ/AQAffinity, https://www.sandboxaq.com/aqaffinity
- **Materials Science Pipeline**: Includes Magpie and SOAP descriptors, Graph Neural Networks (CGCNN, Multi-Property GNN), Multi-task Neural Networks, GPU acceleration, materials generation, element substitution, synthesis planning, and atomistic simulations (VASP, Quantum ESPRESSO). It also incorporates specialized designers for various material types (e.g., Battery, Photovoltaic, Structural) and discovery workflows (e.g., Superconductor, Catalyst).
- **Universal Hardware-Agnostic Materials Pipeline**: Automatically adapts to any hardware configuration (CPU, GPU, Apple Silicon) for optimal performance, supporting molecular, compositional, and polymer material types. It includes specialized feature extractors and predictors for various polymer properties (e.g., Tg, tensile strength).

### scRNA + PGD Trajectory Analysis Module
Automated biomarker discovery and target identification from public single-cell RNA-seq datasets using Pseudotime Graph Diffusion (PGD).
- **Public Dataset Registry**: 50+ curated scRNA-seq datasets from GEO covering 36 disease areas: Alzheimer's, COVID-19, Cancer, Diabetes, IBD, Parkinson's, ALS, Multiple Sclerosis, Rheumatoid Arthritis, Lupus, Asthma, COPD, Pulmonary Fibrosis, Heart Failure, Kidney Disease, NASH, Psoriasis, Atopic Dermatitis, Glioblastoma, Melanoma, Prostate Cancer, Ovarian Cancer, Pancreatic Cancer, Leukemia, Lymphoma, Osteoarthritis, Osteoporosis, Depression, Schizophrenia, Autism, Huntington's, Epilepsy, Hepatitis, Scleroderma, Sjogren's, Sepsis, HIV, Tuberculosis, Endometriosis, Preeclampsia, Macular Degeneration, and Glaucoma
- **PGD Trajectory Inference**: Simulates UMAP embedding and pseudotime trajectory analysis with configurable smoothing parameters
- **Biomarker Detection**: Identifies genes with significant expression changes at trajectory branch points
- **Druggable Target Extraction**: Automatically identifies targetable genes from biomarker analysis
- **Assay Template Generation**: Auto-generates assay templates from detected targets with binding, functional, and cellular assay suggestions
- **BioNeMo Integration**: Predicts inhibitor binding affinity for identified targets using compound SMILES input
- **API Endpoints**:
  - `GET /api/trajectory/datasets` - List public scRNA-seq datasets (filterable by disease)
  - `GET /api/trajectory/datasets/:id` - Get dataset details
  - `GET /api/trajectory/diseases` - Get available disease areas with dataset counts
  - `POST /api/trajectory/analyze/:datasetId` - Run PGD trajectory analysis on a dataset
  - `POST /api/trajectory/assay-template` - Generate assay template from target gene
  - `POST /api/trajectory/predict-inhibitors` - Predict binding affinity for SMILES against target
- **Frontend**: Accessible at `/trajectory-analysis` with interactive UMAP visualization, biomarker table, and BioNeMo prediction dialog

## External Dependencies
- **PostgreSQL**: Primary relational database.
- **Radix UI primitives**: For building UI components.
- **cmdk**: For command palette functionality.
- **embla-carousel-react**: For carousel components.
- **date-fns**: For date utility functions.
- **DigitalOcean Spaces**: S3-compatible storage for assets.
- **Materials Project API**: (via `mp-api` Python package) For materials data and integrations.