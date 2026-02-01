# Lika Sciences Platform

## Overview
Lika Sciences is a scientific SaaS platform accelerating drug discovery and molecular research. It manages molecule registries, research campaigns, and SMILES data to streamline scientific workflows, enable data-driven decisions, and provide robust data visualization and professional scientific presentations. The platform supports drug discovery and materials science, with future extensibility for advanced computational chemistry and quantum computing.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
The platform uses a full-stack TypeScript architecture. The frontend is built with React, Vite, shadcn/ui, and Tailwind CSS, using TanStack React Query. The backend uses Node.js, Express, TypeScript, PostgreSQL, and Drizzle ORM. Shared schemas and types are in a `shared/` directory.

### Key Domain Features
- **Molecule & Project Management**: Manages chemical compounds (SMILES) and organizes them into projects and research campaigns.
- **Data Import & Curation**: Supports batch import of SMILES with validation, duplicate detection, and provides curated domain-aware SMILES libraries.
- **Scientific Terminology**: Integrates professional drug discovery and materials science terminology.
- **Compute Nodes**: Manages multi-provider infrastructure (Hetzner, Vast.ai, AWS, Azure, GCP, On-Prem) for ML, docking, quantum, and agent workloads, including SSH key management.
- **Advanced Scientific Modules**: Includes Multi-Target SAR, Translational Medicine, Wet-Lab Integration, and Literature & IP for drug discovery. For materials science, it supports various material types with structure-first and property-first discovery modes.
- **Collaboration & Templates**: Supports multi-organization collaboration with role-based access and provides disease-specific pipeline templates.
- **Enterprise-Scale Processing Infrastructure**: Features a robust processing jobs system for orchestration, status tracking, and fault tolerance, including artifact storage and ingestion. Precomputed aggregations provide dashboard data and per-variant metrics.

### API Design
The platform uses RESTful API endpoints under `/api/`, with specific `/api/agent/` endpoints for AI interaction.

### Design System
Adheres to Material Design principles with Carbon Design data patterns, using Inter and JetBrains Mono fonts, and a custom Tailwind theme for scientific visualization.

### Python Compute Pipelines
The platform integrates production-grade Python pipelines for drug discovery and materials science, designed for executing computational workloads.
- **Drug Discovery Pipeline**: Features Dask for distributed computing, RAPIDS cuML for GPU-accelerated ML, PyTorch mixed precision training, AutoDock Vina integration, and XGBoost with GPU acceleration.
- **Vaccine Discovery Pipeline**: A GPU-agnostic pipeline with automatic hardware detection and intelligent task routing. It includes a complete vaccine pipeline with bioinformatics tool integrations (DSSP, DiscoTope, NetMHCpan, MAFFT, Linker design, JCat, ViennaRNA) and supports PDB file input. A task classification matrix details hardware routing by pipeline stage.
- **ESMFold Integration**: Primary structure prediction service using Meta's free API for drug discovery, vaccine discovery, and materials science.
- **OpenFold3 NIM Integration**: AlphaFold3-compatible structure prediction for protein-ligand complexes via NVIDIA NIM API. Serves as a fallback for sequences >400 residues or complex multi-chain predictions. Includes result caching.
- **SandboxAQ AQAffinity Integration**: Open-source AI model for fast, structure-free prediction of protein-ligand binding affinities, currently in simulation mode. Supports drug discovery, vaccine discovery, and materials discovery pipelines.
- **Materials Science Pipeline**: Includes Magpie and SOAP descriptors, Graph Neural Networks, Multi-task Neural Networks, GPU acceleration, materials generation, element substitution, synthesis planning, and atomistic simulations (VASP, Quantum ESPRESSO). It also incorporates specialized designers for various material types and discovery workflows. A universal hardware-agnostic materials pipeline adapts to any hardware configuration.

### scRNA + PGD Trajectory Analysis Module
Automated biomarker discovery and target identification from public single-cell RNA-seq datasets using Pseudotime Graph Diffusion (PGD). Includes a public dataset registry, PGD trajectory inference, biomarker detection, druggable target extraction, assay template generation, and BioNeMo integration for inhibitor binding affinity prediction.

## External Dependencies
- **PostgreSQL**: Primary relational database.
- **Radix UI primitives**: For building UI components.
- **cmdk**: For command palette functionality.
- **embla-carousel-react**: For carousel components.
- **date-fns**: For date utility functions.
- **DigitalOcean Spaces**: S3-compatible storage for assets.
- **Materials Project API**: For materials data and integrations.