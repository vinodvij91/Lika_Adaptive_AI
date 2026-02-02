# Lika Sciences Design Guidelines

## Design Approach

**Selected Approach:** Design System Foundation (Material Design principles + Carbon Design data patterns)

**Rationale:** Scientific SaaS platforms require information clarity, consistent data visualization, and professional credibility over visual novelty. This is a utility-first application where researchers need to process complex molecular data efficiently.

**Core Principles:**
- Information hierarchy over aesthetics
- Efficiency in data-dense workflows
- Scientific credibility through clean, professional presentation
- Progressive disclosure for complex features

---

## Typography

**Font Stack:**
- Primary: Inter (Google Fonts) - excellent for data tables and UI density
- Monospace: JetBrains Mono - for SMILES strings, molecular IDs, technical data

**Hierarchy:**
- Page Titles: text-3xl font-semibold
- Section Headers: text-xl font-semibold
- Data Labels: text-sm font-medium
- Body/Table Content: text-sm
- Technical Data: text-xs font-mono
- Captions/Metadata: text-xs text-gray-600

---

## Layout System

**Spacing Primitives:** Use Tailwind units of 2, 4, 6, and 8 consistently
- Component padding: p-4 or p-6
- Section spacing: gap-6 or gap-8
- Tight data layouts: gap-2 or gap-4
- Page margins: p-6 or p-8

**Grid Patterns:**
- Dashboard cards: grid-cols-1 md:grid-cols-2 lg:grid-cols-3
- Data tables: Full-width with horizontal scroll
- Detail panels: Two-column split (2/3 main content, 1/3 sidebar)
- Pipeline configurator: Vertical stepper layout

**Container Strategy:**
- Max-width: max-w-7xl for most views
- Full-width: Tables and data visualizations
- Constrained: Forms and configurators (max-w-3xl)

---

## Component Library

**Navigation:**
- Fixed left sidebar (w-64) with project/target/molecule sections
- Top bar with user menu, notifications, search
- Breadcrumbs for deep navigation hierarchy

**Data Display:**
- Tables: Striped rows, sticky headers, sortable columns, fixed action columns
- Cards: Subtle shadows (shadow-sm), rounded-lg, hover:shadow-md transitions
- Metrics: Large numbers (text-3xl) with labels below
- Status badges: Rounded pills with semantic colors (pending/running/completed/failed)

**Forms & Inputs:**
- Clean outlined inputs with floating labels
- Multi-step wizards for campaign configuration
- Slider controls for oracle weightings (w_docking, w_admet, w_qsar)
- Dropdown selects for disease areas, pipeline steps

**Data Visualization:**
- Pipeline graphs: Horizontal flow diagram with step nodes
- Score distributions: Histograms and scatter plots
- Learning graph: Network visualization of molecule relationships

**Specialized Components:**
- Molecule viewer: Card with SMILES display + 2D structure preview placeholder
- Campaign monitor: Timeline view of jobs with status indicators
- Collaboration: Comment threads with timestamps and user avatars

---

## Key Page Layouts

**Landing Page:**
- Hero section (h-screen) with gradient background
- Large headline emphasizing "Adaptive AI Drug Discovery"
- Three-column feature showcase
- CTA to platform entry

**Dashboard:**
- 3-column metric cards at top (molecules evaluated, active campaigns, recent hits)
- Two-column split: Recent projects (left) + Campaign status feed (right)
- Domain breakdown chart (CNS/Oncology/Rare distribution)

**Campaign Configurator:**
- Vertical stepper on left (w-1/4)
- Configuration panel on right (w-3/4)
- Bottom action bar with Save/Start buttons
- Visual pipeline preview at top

**Campaign Detail:**
- Header with campaign metadata
- Pipeline visualization as read-only flow
- Tabs: Overview, Jobs, Top Molecules, Activity
- Data table with molecule scores, sortable and filterable

**Reports & Learning Graph:**
- Filter sidebar (w-1/4)
- Main visualization area (w-3/4)
- Export controls in top-right corner

---

## Animations

Use sparingly, only for:
- Loading states (subtle spinners)
- Page transitions (fade-in)
- Status changes (badge color transitions)
- Table row expansion (accordion-style)

**No hover animations** on data elements to maintain scan-ability.

---

## Images

**Hero Image:** Abstract representation of molecular structures or AI neural networks - stylized, professional, conveys scientific innovation without being overly technical. Full-width background with overlay gradient for text legibility.

**Dashboard:** Small icon illustrations for empty states (e.g., "No campaigns yet")

**No product screenshots** - this is the product itself.