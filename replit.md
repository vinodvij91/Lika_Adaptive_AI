# Lika Sciences Platform

## Overview

Lika Sciences is a scientific SaaS platform for drug discovery and molecular research. It focuses on managing molecule registries, research campaigns, and SMILES data. The platform aims to provide clear information, robust data visualization, and professional scientific presentation, with future extensibility for advanced chemistry computations. The business vision is to streamline drug discovery workflows, accelerate molecular research, and enable data-driven decisions in scientific R&D.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture
The platform utilizes a full-stack TypeScript architecture. The frontend is built with React, Vite, and shadcn/ui, styled with Tailwind CSS, and manages state using TanStack React Query. The backend runs on Node.js with Express, leveraging TypeScript and PostgreSQL for data persistence. Drizzle ORM is used for database interactions, with schema validation via drizzle-zod. Shared schemas and types reside in a `shared/` directory, accessible by both frontend and backend.

### Key Domain Features
- **Molecule Registry**: Manages chemical compounds using SMILES notation.
- **Project & Campaign Management**: Organizes molecules into projects and facilitates research campaign setup with various modalities (small molecule, PROTAC, peptide, fragment).
- **SMILES Import**: Supports batch import with validation and duplicate detection.
- **Curated Libraries**: Provides domain-aware SMILES libraries with cleaning and validation workflows.
- **Scientific Terminology**: Incorporates professional drug discovery terminology for job types, campaign tabs, and process descriptions (e.g., Hit identification, SAR feedback, Hit-to-Lead optimization).
- **Compute Nodes**: Manages multi-provider infrastructure (Hetzner, Vast.ai, AWS, Azure, GCP, On-Prem) for ML, docking, quantum, and agent workloads, supporting various GPU types and connection methods (SSH, Cloud API). Features include SSH key management and usage tracking.
- **Multi-Target SAR (v1)**: Enables multi-target profiling with assay panels, a Mechanism of Action (MoA) knowledge graph, and advanced visualizations (radar charts, trade-off plots, activity heatmaps) for multi-target optimization and safety flagging.
- **Translational Medicine (v1)**: Integrates target variants, disease context signals, and organizes campaigns into higher-level drug discovery programs. Introduces extended molecule scoring (translational, synthesis, variant robustness, uncertainty metrics, IP screening).
- **Wet-Lab Integration (v1)**: Defines wet-lab assays, provides AI-generated experiment recommendations, and incorporates experimental outcomes.
- **Literature & IP (v1)**: Allows for literature annotations and IP risk screening.
- **Multi-Organization Collaboration (v1)**: Supports creating organizations with role-based access for team collaboration and sharing assets.
- **Disease-Specific Pipeline Templates (v1)**: Offers pre-configured pipeline templates for various disease domains (Alzheimer's, Oncology, Neuroinflammation, Metabolic Disease, etc.) with customizable targets, scoring weights, and visualization presets.
- **Materials Science (v1)**: First-class materials science domain supporting polymers, crystals, composites, catalysts, membranes, and coatings with structure-first and property-first discovery modes.

### Materials Science Data Model (v1)

#### Core Tables
- **material_entities**: Core material identity table
  - `id`: UUID primary key
  - `name`: Material name
  - `type`: Structure type (polymer, crystal, composite, surface, membrane, catalyst, coating)
  - `representation`: JSONB for structural data (SMILES, BigSMILES, CIF, graph, repeat unit, lattice)
  - `baseFamily`: Base family (e.g., "polyamide", "perovskite", "carbon composite")
  - `metadata`: JSONB for additional metadata
  - `isCurated`: Boolean curation flag
  - `companyId`: Organization/company reference
  - `createdAt`: Timestamp

- **material_variants**: Structural variants for systematic exploration
  - `id`: UUID primary key
  - `materialId`: Foreign key to material_entities
  - `variantParams`: JSONB for substitutions, chain length, dopants, ratios
  - `generatedBy`: Generation method (human, ml, genetic, quantum)
  - `simulationState`: Current simulation status
  - `manufacturabilityScore`: Manufacturing feasibility score
  - `createdAt`: Timestamp

- **material_properties**: Property predictions and measurements
  - Links to material_entities via materialId
  - Supports ML, simulation, and experimental sources

- **materials_programs**: High-level materials research programs
- **materials_campaigns**: Materials discovery campaigns with pipeline configurations
- **materials_oracle_scores**: Scoring with property breakdown, synthesis feasibility, manufacturing cost

### Enterprise-Scale Processing Infrastructure

#### Scalability Design
The platform is designed for industrial-scale materials discovery:
- Materials libraries: 100K-5M+ polymer, crystal, or composite variants
- Property prediction pipelines: batch sizes of 10K-500K variants
- Multiple concurrent campaigns per organization
- Distributed processing across compute nodes
- Resumable and fault-tolerant execution

#### Processing Jobs System
- **processing_jobs**: Core job orchestration table
  - `id`: UUID primary key
  - `type`: Job type (property_prediction, simulation, variant_generation, optimization, screening, aggregation)
  - `status`: State machine (queued, dispatched, running, succeeded, failed, cancelled, paused)
  - `priority`: Job priority for queue ordering
  - `itemsTotal/itemsCompleted`: Progress tracking for batch operations
  - `progressPercent`: Real-time progress percentage
  - `checkpointData`: JSONB for resumable execution state
  - `inputPayload/outputPayload`: JSONB for job data
  - `computeNodeId`: Assigned compute node
  - `retryCount/maxRetries`: Fault tolerance configuration
  - `heartbeatAt`: Worker liveness tracking

- **processing_job_runs**: Retry/resumption tracking per job
- **processing_job_events**: Streaming partial results and events

#### Precomputed Aggregations
- **materials_campaign_aggregates**: Campaign-level dashboard data
  - Total materials/variants counts
  - Score distributions and histograms
  - Top variant IDs for quick access
  - Property correlations
  - Auto-refresh on job completion

- **material_variant_metrics**: Per-variant computed metrics
  - Property scores breakdown
  - Aggregate scores with ranking
  - Percentile positioning
  - Quality flags

### Processing Job Endpoints
- `GET /api/processing-jobs` - List jobs with filters (status, type, campaignId)
- `GET /api/processing-jobs/:id` - Get job with runs and events
- `POST /api/processing-jobs` - Create new processing job
- `PATCH /api/processing-jobs/:id` - Update job status/data
- `POST /api/processing-jobs/:id/progress` - Update job progress with checkpoints
- `POST /api/processing-jobs/:id/events` - Stream partial results
- `POST /api/materials/:id/variants/batch-submit` - Async batch variant generation
- `GET /api/materials-campaigns/:id/aggregates` - Get precomputed aggregates
- `POST /api/materials-campaigns/:id/aggregates/refresh` - Schedule aggregate refresh
- `GET /api/materials-campaigns/:id/variant-metrics` - Get variant rankings

### Materials Science Endpoints
- `GET /api/materials` - List material entities (optional `?type=` filter)
- `GET /api/materials/:id` - Get material with properties
- `POST /api/materials` - Create material entity
- `PATCH /api/materials/:id` - Update material entity
- `DELETE /api/materials/:id` - Delete material entity
- `POST /api/materials/:id/properties` - Add property to material
- `GET /api/materials/:id/variants` - List variants for a material
- `POST /api/materials/:id/variants` - Create single variant
- `POST /api/materials/:id/variants/batch` - Synchronous batch create variants
- `GET /api/material-variants/:id` - Get variant by ID
- `PATCH /api/material-variants/:id` - Update variant
- `DELETE /api/material-variants/:id` - Delete variant
- `GET /api/materials-programs` - List materials programs
- `POST /api/materials-programs` - Create materials program
- `GET /api/materials-campaigns` - List materials campaigns
- `POST /api/materials-campaigns` - Create materials campaign
- Agent endpoints: `/api/agent/materials`, `/api/agent/materials-programs`, `/api/agent/materials-campaigns`

### API Design
The platform uses RESTful API endpoints under the `/api/` prefix, with specific agent-friendly endpoints designed for AI interaction, including `GET /api/agent/campaigns/pending`, `POST /api/agent/campaigns/:id/start`, and various learning graph, library, and recommendation management endpoints. Service account roles (`agent_pipeline_copilot`, `agent_operator`, `agent_readonly`) define agent permissions.

### Design System
Adheres to Material Design principles with Carbon Design data patterns, using Inter and JetBrains Mono fonts. Emphasizes information hierarchy and data-dense layouts with a custom Tailwind theme for scientific visualization.

### Quantum Compute Integration
The platform is designed to integrate with quantum computation services (e.g., IonQ/IBM Quantum) for optimization and scoring, with placeholder configurations for future use.

## External Dependencies

- **PostgreSQL**: Primary relational database.
- **Radix UI primitives**: For UI components (dialogs, dropdowns, forms, navigation).
- **cmdk**: For command palette functionality.
- **embla-carousel-react**: For carousel components.
- **date-fns**: For date utility functions.
