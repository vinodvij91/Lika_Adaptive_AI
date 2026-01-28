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
  - **GPU-Intensive Tasks** (15-100x speedup): Structure prediction (ESMFold/AlphaFold2), MD simulation (OpenMM/GROMACS)
  - **GPU-Preferred Tasks** (2-5x speedup): MHC binding prediction with deep learning models
  - **CPU-Intensive Tasks**: Epitope prediction (NetMHCpan), mRNA secondary structure (ViennaRNA)
  - **CPU-Only Tasks**: Codon optimization, file I/O
  - **API Endpoints**:
    - `POST /api/compute/vaccine/pipeline` - Run full vaccine discovery pipeline
    - `POST /api/compute/vaccine/structure` - Predict protein structure (GPU-intensive)
    - `POST /api/compute/vaccine/epitopes` - Predict MHC binding/epitopes (CPU-intensive)
    - `POST /api/compute/vaccine/codon-optimize` - Codon optimization (CPU-only)
    - `POST /api/compute/vaccine/mrna-design` - mRNA construct design (CPU-intensive)
    - `POST /api/compute/vaccine/md-simulation` - Molecular dynamics (GPU-intensive)
    - `GET /api/compute/vaccine/hardware` - Get hardware performance report
- **Materials Science Pipeline**: Includes Magpie and SOAP descriptors, Graph Neural Networks (CGCNN, Multi-Property GNN), Multi-task Neural Networks, GPU acceleration, materials generation, element substitution, synthesis planning, and atomistic simulations (VASP, Quantum ESPRESSO). It also incorporates specialized designers for various material types (e.g., Battery, Photovoltaic, Structural) and discovery workflows (e.g., Superconductor, Catalyst).
- **Universal Hardware-Agnostic Materials Pipeline**: Automatically adapts to any hardware configuration (CPU, GPU, Apple Silicon) for optimal performance, supporting molecular, compositional, and polymer material types. It includes specialized feature extractors and predictors for various polymer properties (e.g., Tg, tensile strength).

## External Dependencies
- **PostgreSQL**: Primary relational database.
- **Radix UI primitives**: For building UI components.
- **cmdk**: For command palette functionality.
- **embla-carousel-react**: For carousel components.
- **date-fns**: For date utility functions.
- **DigitalOcean Spaces**: S3-compatible storage for assets.
- **Materials Project API**: (via `mp-api` Python package) For materials data and integrations.