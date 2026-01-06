# Lika Sciences Platform

## Overview

Lika Sciences is a scientific SaaS platform designed for drug discovery and molecular research workflows. The application provides tools for managing molecule registries, research campaigns, and SMILES (chemical structure notation) data imports. Built with a focus on information clarity, data visualization, and professional scientific presentation.

The platform follows a full-stack TypeScript architecture with React frontend and Express backend, using PostgreSQL for data persistence. It's designed with future extensibility for Python/RDKit microservices for chemistry-specific computations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript, bundled using Vite
- **UI Components**: shadcn/ui component library with Radix UI primitives
- **Styling**: Tailwind CSS with custom design tokens for scientific data presentation
- **State Management**: TanStack React Query for server state
- **Forms**: React Hook Form with Zod validation (via @hookform/resolvers)
- **Path Aliases**: `@/` maps to `client/src/`, `@shared/` maps to `shared/`

### Backend Architecture
- **Runtime**: Node.js with Express
- **Language**: TypeScript (ESM modules)
- **Build Tool**: tsx for development, custom build script for production
- **Session Storage**: PostgreSQL-backed sessions via connect-pg-simple

### Data Layer
- **Database**: PostgreSQL
- **ORM**: Drizzle ORM with drizzle-zod for schema validation
- **Schema Location**: `shared/schema.ts` (shared between frontend and backend)
- **Migrations**: Managed via drizzle-kit, output to `./migrations`

### Project Structure
```
client/          # React frontend application
  src/
    components/  # UI components (shadcn/ui structure)
    lib/         # Utility functions
    hooks/       # Custom React hooks
server/          # Express backend
shared/          # Shared code (schema, types)
migrations/      # Database migrations
attached_assets/ # Specification documents
```

### Design System
- Uses Material Design principles with Carbon Design data patterns
- Typography: Inter (primary), JetBrains Mono (technical/SMILES data)
- Focus on information hierarchy and data-dense layouts
- Custom Tailwind theme with scientific color tokens

### Key Domain Features
- **Molecule Registry**: Storage and management of chemical compounds via SMILES notation
- **Project Organization**: Molecules linked to projects via junction tables
- **Campaign Configurator**: Workflow for research campaign setup with modality selection (small molecule, PROTAC, peptide, fragment)
- **SMILES Import**: Batch import with duplicate detection and validation
- **Curated Libraries**: Domain-aware SMILES libraries with scaffolds, cleaning workflows, and agent validation
- **Compute Nodes**: Infrastructure management for ML, docking, quantum, and agent workloads (Hetzner, Vast.ai)
- **SSH Key Management**: User SSH key registration for compute node access
- **Usage Tracking**: Resource consumption tracking (CPU, GPU, storage) per campaign/project with source attribution
- **Credit Wallet**: Foundation for credits-based billing (wallet balance, transactions) - stub only in v0

### Translational Medicine Features (v1)
- **Target Variants**: Track clinically significant genetic variants for variant-aware scoring
- **Disease Context Signals**: Store disease-specific context signals (GWAS, clinical trial data, biomarker relevance)
- **Programs**: Organize campaigns into higher-level drug discovery programs with disease area tracking
- **Oracle Versions**: Model governance via version tracking for reproducibility (stores component versions as JSONB)
- **Extended Molecule Scoring**: 
  - `translationalScore`: Disease context relevance
  - `synthesisScore`: Synthesis feasibility prediction
  - `variantRobustnessScore`: Cross-variant robustness
  - Uncertainty metrics: `dockingUncertainty`, `admetUncertainty`, `qsarUncertainty`, `translationalUncertainty`
  - IP screening: `ipSimilarity`, `mostSimilarPatentId`
  - `applicabilityDomainFlag`: Confidence in predictions

### Wet-Lab Integration Features (v1)
- **Assays**: Define wet-lab assays with estimated cost and duration (binding, functional, in_vivo, pk, admet)
- **Experiment Recommendations**: AI-generated suggestions for which molecules to test next
- **Assay Results**: Feedback loop for experimental outcomes with outcome labels (active, inactive, toxic, no_effect, inconclusive)

### Literature & IP Features (v1)
- **Literature Annotations**: Store PubMed/literature references with relevance scores for targets and molecules
- **IP Risk Screening**: Similarity scoring against patent databases (stored in molecule_scores)

### Multi-Organization Collaboration (v1)
- **Organizations**: Create and manage organizations for team collaboration
- **Organization Members**: Role-based access (admin, member, viewer)
- **Shared Assets**: Share SMILES libraries, pipeline templates, or programs between organizations with read/fork permissions

### API Design Pattern
RESTful endpoints under `/api/` prefix. Example endpoint structure:
- `POST /api/molecules/import-smiles` - Batch SMILES import with validation

### Agent-Friendly API Endpoints
Designed for AI agents and bots to interact with the platform:
- `GET /api/agent/campaigns/pending` - Returns minimal data for campaigns needing action
- `GET /api/agent/campaigns/:id/analytics` - Returns compact JSON summary of scores and status
- `POST /api/agent/campaigns/:id/start` - Triggers the orchestrator to start a campaign
- `GET /api/agent/learning-graph/unlabeled` - Returns entries needing labeling
- `POST /api/agent/learning-graph/label` - Labels a learning graph entry
- `POST /api/agent/quantum-recommendation` - Recommends whether quantum optimization should be used
- `GET /api/agent/libraries/curated` - Returns curated libraries available for campaigns
- `GET /api/agent/libraries/:id/status` - Returns library validation progress and readiness
- `POST /api/agent/libraries/:id/validate` - Validates or invalidates library molecules
- `POST /api/agent/libraries/:id/classify` - Classifies molecules with domain annotations
- `GET /api/agent/variants/:targetId` - Returns target variants for variant-aware scoring
- `GET /api/agent/programs` - Returns programs with disease area info
- `GET /api/agent/oracle-versions` - Returns oracle versions for model governance
- `GET /api/agent/assays` - Returns available assays with cost estimates
- `GET /api/agent/campaigns/:campaignId/recommendations` - Returns experiment recommendations
- `POST /api/agent/assay-results` - Bulk create assay results from wet-lab feedback
- `GET /api/agent/literature/:targetId` - Returns literature annotations for a target
- `POST /api/agent/literature` - Creates literature annotation
- `POST /api/agent/recommendations/:id/approve` - Approves an experiment recommendation
- `POST /api/agent/recommendations/:id/reject` - Rejects an experiment recommendation

### Compute Infrastructure Endpoints
- `GET /api/compute-nodes` - List all compute nodes
- `GET /api/compute-nodes/:id` - Get node details with SSH key registrations
- `POST /api/compute-nodes` - Register a new compute node
- `POST /api/compute-nodes/:id/register-key` - Request SSH key registration (mock, future automation)
- `GET /api/ssh-keys` - List user's SSH public keys
- `POST /api/ssh-keys` - Upload a new SSH public key

### Usage & Credits Endpoints (v0 Foundation)
- `GET /api/usage` - List usage meters for the current user (filterable by projectId, campaignId)
- `GET /api/usage/summary/:projectId` - Aggregated usage summary by resource type for a project
- `GET /api/credits/wallet` - Get or create user's credit wallet
- `GET /api/credits/transactions` - List credit transactions for user's wallet
- `POST /api/credits/purchase` - Stub endpoint (returns 501, placeholder for billing)
- `POST /api/credits/apply` - Stub endpoint (returns 501, placeholder for billing)

### Service Account Roles
The platform supports service accounts for agents with defined roles:
- `agent_pipeline_copilot` - Can configure and start campaigns
- `agent_operator` - Can start campaigns and label learning graph entries
- `agent_readonly` - Can read data but not modify

### Quantum Compute Integration (IonQ/IBM Quantum Ready)
The platform is prepared for quantum computation integration:
- **QuantumClient**: Stub implementation for quantum optimization jobs
- **Pipeline Steps**: Support for `provider: "quantum"` in pipeline configuration
- **Job Types**: `quantum_optimization` and `quantum_scoring` job types
- **Environment Variables**: `QUANTUM_API_BASE_URL`, `QUANTUM_API_KEY` (placeholders for future use)

## External Dependencies

### Database
- **PostgreSQL**: Primary data store, connection via `DATABASE_URL` environment variable

### Future Integrations (Planned)
- **Python/RDKit Microservice**: Chemistry validation and molecular property calculations (currently stubbed in Node backend)

### UI Component Libraries
- Radix UI primitives (dialogs, dropdowns, forms, navigation)
- cmdk (command palette)
- embla-carousel-react (carousels)
- date-fns (date utilities)

### Development Tools
- Replit-specific plugins for development (cartographer, dev-banner, error overlay)