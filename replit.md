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
- **Campaign Configurator**: Workflow for research campaign setup
- **SMILES Import**: Batch import with duplicate detection and validation

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