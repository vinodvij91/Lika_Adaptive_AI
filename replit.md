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
- **Molecule Optimization**: Ported from Python pipeline Stage 6 (MoleculeOptimizer). Available in Campaign SAR tab series popups. Analyzes solubility/exposure, permeability/CNS, safety (cardiotoxicity), metabolic stability, and dose/indication repurposing. Two actions: "Optimize Properties" generates analogs (ester→amide, hydroxyl→methoxy, etc.) with ADMET predictions; "Optimize Dose & Indication" generates dose scenarios and repurposing hints. Disease-context-aware (CNS, oncology, obesity, etc.). Service: `server/services/molecule-optimizer.ts`. Endpoints: `POST /api/campaigns/:id/sar/optimize-properties` and `POST /api/campaigns/:id/sar/optimize-dose`. Campaign-level aggregation via `GET /api/campaigns/:id/sar/optimization-summary` provides original/optimized counts, improvement stats (solubility, toxicity, CNS, metabolic), property deltas, dose scenarios, and per-series optimization maps. SAR overview displays summary panel with badges. SeriesCards show N original / M optimized counts with "Optimized" badge. Multi-Target tab has Profile selector (Original/Optimized/Both) with dual radar traces comparing original vs optimized averages. Matrix view shows Opt/Orig badges per molecule.
- **Collaboration & Templates**: Supports multi-organization collaboration with role-based access and provides disease-specific pipeline templates.
- **Enterprise-Scale Processing Infrastructure**: Features a robust processing jobs system for orchestration, status tracking, and fault tolerance, including artifact storage and ingestion. Precomputed aggregations provide dashboard data and per-variant metrics.

### API Design
The platform uses RESTful API endpoints under `/api/`, with specific `/api/agent/` endpoints for AI interaction.

### Design System
Adheres to Material Design principles with Carbon Design data patterns, using Inter and JetBrains Mono fonts, and a custom Tailwind theme for scientific visualization.

### Python Compute Pipelines
Production pipeline scripts live in `pipelines/` directory (organized by domain), replacing the old `compute/` scripts. All scripts share a standardized CLI contract: `--job-type <type> --params '<json>' [--params-file <path>] [--output <path>]` and return `{step, success, output, error}` JSON envelopes.
- **Drug Discovery Pipeline** (`pipelines/drug_discovery/lika_drug_discovery_pipeline.py`): 8-stage pipeline covering 750+ diseases. Job types: full_pipeline, smiles_validation, property_calculation, fingerprint_generation, ml_prediction, scoring. Config: `disease_discovery_config.yaml`.
- **Vaccine Discovery Pipeline** (`pipelines/vaccine_discovery/complete_vaccine_pipeline_production.py`): Full vaccine discovery with bioinformatics integrations (DSSP, DiscoTope, NetMHCpan, MAFFT, JCat, ViennaRNA). Job types: full_pipeline, conservation_analysis, dssp_analysis, bcell_epitope_prediction, tcell_epitope_prediction, optimize_codons, design_mrna, rna_structure, run_md.
- **Alzheimer's 12-Target Platform** (`pipelines/alzheimers/alzheimers_12target_platform.py`): Multi-target scoring across 12 Alzheimer's protein targets. Job types: full_pipeline, list_targets, execution_plan, phase1_screening, phase2_validation, phase3_optimization, toggle_target.
- **Materials Science Pipeline** (`pipelines/materials_science/universal_hardware_agnostic_pipeline.py`): Hardware-agnostic materials predictor with automatic hardware detection. Job types: hardware_detect, train, predict, prepare_data, batch_screening, full_pipeline.
- **ESMFold Integration**: Primary structure prediction service using Meta's free API for drug discovery, vaccine discovery, and materials science.
- **OpenFold3 NIM Integration**: AlphaFold3-compatible structure prediction for protein-ligand complexes via NVIDIA NIM API. Serves as a fallback for sequences >400 residues or complex multi-chain predictions. Includes result caching.
- **SandboxAQ AQAffinity Integration**: Open-source AI model for fast, structure-free prediction of protein-ligand binding affinities, currently in simulation mode.
- **Fc Effector Pipeline** (`pipelines/fc_effector/fc_effector_pipeline.py`): Disease/vaccine-agnostic Fc effector modeling. CPU layers: FcgammaR/FcRn atlas, ADCC/CDC scoring, species-translation similarity. GPU hook: BioNeMo Fc-receptor affinity prediction. LLM hook: OpenAI structured UI guidance. Job types: build_atlas, build_variants, build_species_similarity, bionemo_fc_affinity, openai_guidance, build_fc_bundle, plot_atlas, plot_effector, plot_species, full_pipeline. Graceful no-op when BioNeMo/OpenAI credentials are absent.
- **Omics Integration Pipeline** (`pipelines/omics_integration/omics_integration_pipeline.py`): Disease/vaccine-agnostic multi-omics integration engine for all ~360 disease and 10 vaccine pipelines. CPU layers: per-target evidence aggregation (genomics, transcriptomics, proteomics, metabolomics) with weighted integrated scoring. GPU hook: BioNeMo sequence property enrichment (stability, disorder, aggregation) with CPU fallback. LLM hook: OpenAI JSON-mode structured UI text (panel_title, panel_subtitle, tooltips, narrative). Job types: build_table, bionemo_enrich, openai_guidance, build_bundle, full_pipeline. Main function `build_omics_bundle()` returns JSON-serializable dict with context, omics_table, sequence_properties, ui_text. Graceful no-op when BioNeMo/OpenAI credentials are absent.
- **Remote Execution**: Scripts are staged to `/opt/lika-compute/` on remote compute nodes via SSH. `compute-executor.ts` handles pipeline + config file staging. `buildPipelineCommand()` in `compute-adapters.ts` generates execution commands. Deploy route uploads all scripts from `pipelines/` to remote nodes.

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