import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Comprehensive page knowledge for LIKA Agent
const PAGE_KNOWLEDGE: Record<string, { title: string; domain: string; description: string; capabilities: string[]; quickActions: string[] }> = {
  // Drug Discovery Pages
  "/dashboard-drug": {
    title: "Drug Discovery Dashboard",
    domain: "drug_discovery",
    description: "Central command for drug discovery projects showing active campaigns, molecule counts, hit rates, and key metrics.",
    capabilities: [
      "View active research campaigns and their status",
      "Monitor molecule screening progress",
      "Track hit rates and scoring metrics",
      "Access quick links to key workflows"
    ],
    quickActions: ["Launch new campaign", "View top hits", "Check ADMET profiles"]
  },
  "/campaigns": {
    title: "Research Campaigns",
    domain: "drug_discovery",
    description: "Manage drug discovery campaigns for targets like EGFR, BRAF, kinases. Each campaign contains molecules, assays, and scoring data.",
    capabilities: [
      "Create new screening campaigns",
      "Track campaign progress and milestones",
      "View molecule registrations per campaign",
      "Compare campaign performance metrics"
    ],
    quickActions: ["Create campaign", "Import molecules", "Run scoring pipeline"]
  },
  "/targets": {
    title: "Target Management",
    domain: "drug_discovery",
    description: "Manage drug targets (proteins, receptors, enzymes). Define binding sites, PDB structures, and target validation data.",
    capabilities: [
      "Register new drug targets",
      "Upload PDB structures for docking",
      "Define binding site coordinates",
      "Link targets to disease indications"
    ],
    quickActions: ["Add target", "Upload PDB", "Configure docking box"]
  },
  "/molecules": {
    title: "Molecule Registry",
    domain: "drug_discovery",
    description: "Central registry for all small molecules with SMILES, properties, and activity data.",
    capabilities: [
      "Browse and search molecules by structure/properties",
      "View calculated properties (MW, LogP, TPSA, HBD/HBA)",
      "Check oracle scores and docking results",
      "Identify structural alerts and liabilities"
    ],
    quickActions: ["Search by SMILES", "Filter by properties", "Export hits"]
  },
  "/libraries": {
    title: "Compound Libraries",
    domain: "drug_discovery",
    description: "Curated compound libraries for screening: FDA-approved drugs, natural products, kinase inhibitors, fragment libraries.",
    capabilities: [
      "Access pre-built screening libraries",
      "Create custom compound collections",
      "Compare library diversity",
      "Export for virtual screening"
    ],
    quickActions: ["Browse libraries", "Create collection", "Start screen"]
  },
  "/docking": {
    title: "Molecular Docking",
    domain: "drug_discovery",
    description: "AutoDock Vina integration for structure-based virtual screening. Configure docking boxes, run GPU-accelerated docking.",
    capabilities: [
      "Configure docking parameters (exhaustiveness, box size)",
      "Run batch docking on molecule sets",
      "Visualize docking poses",
      "Analyze binding interactions"
    ],
    quickActions: ["Set up docking job", "View best poses", "Analyze contacts"]
  },
  "/admet": {
    title: "ADMET Profiling",
    domain: "drug_discovery",
    description: "Predict absorption, distribution, metabolism, excretion, and toxicity properties for drug candidates.",
    capabilities: [
      "Run ADMET predictions on molecules",
      "Identify metabolic liabilities",
      "Check hERG and CYP inhibition risk",
      "Assess oral bioavailability potential"
    ],
    quickActions: ["Run ADMET", "Check hERG risk", "View metabolism"]
  },
  "/hit-triage": {
    title: "Hit Triage",
    domain: "drug_discovery",
    description: "Evaluate and prioritize screening hits using multi-parameter optimization and medicinal chemistry filters.",
    capabilities: [
      "Apply Lipinski/Veber/CNS filters",
      "Score compounds by weighted criteria",
      "Identify PAINS and false positives",
      "Generate triaged hit lists"
    ],
    quickActions: ["Apply filters", "Rank hits", "Export shortlist"]
  },
  "/assays": {
    title: "Assay Management",
    domain: "drug_discovery",
    description: "Track biochemical and cellular assays. Record IC50, EC50, Ki values and assay conditions.",
    capabilities: [
      "Define assay protocols and conditions",
      "Import assay results (IC50, EC50, Ki)",
      "Perform SAR analysis across assays",
      "Generate dose-response curves"
    ],
    quickActions: ["Add assay", "Import results", "Analyze SAR"]
  },
  "/import-hub": {
    title: "Import Hub",
    domain: "both",
    description: "Batch import molecules (SMILES, SDF) or materials with automatic validation and duplicate detection.",
    capabilities: [
      "Import SMILES files with validation",
      "Upload SDF/MOL files",
      "Detect and handle duplicates",
      "Map custom property columns"
    ],
    quickActions: ["Upload file", "Validate SMILES", "Check duplicates"]
  },
  "/pipeline": {
    title: "Pipeline Launcher",
    domain: "both",
    description: "Launch high-throughput compute pipelines for both drug discovery (docking, ML) and materials science (property prediction, synthesis planning).",
    capabilities: [
      "Configure and launch Drug Discovery pipelines (docking, fingerprints, ML, ADMET)",
      "Configure and launch Materials Science pipelines (battery, solar, superconductor, catalyst, thermoelectric, PFAS replacement, aerospace, biomedical, semiconductor, construction, transparent conductor, magnets, electrolytes, water purification, carbon capture)",
      "Monitor job queue and progress",
      "View compute node allocation"
    ],
    quickActions: ["Launch pipeline", "View queue", "Check nodes"]
  },
  // Materials Science Pages
  "/dashboard-materials": {
    title: "Materials Science Dashboard",
    domain: "materials_science",
    description: "Central command for materials discovery showing active programs, material counts, and discovery metrics.",
    capabilities: [
      "View active materials programs",
      "Monitor discovery campaign progress",
      "Track property predictions",
      "Access materials science workflows"
    ],
    quickActions: ["New program", "View top materials", "Run predictions"]
  },
  "/materials-campaigns": {
    title: "Materials Campaigns",
    domain: "materials_science",
    description: "Manage materials discovery campaigns targeting specific applications: batteries, solar cells, catalysts, superconductors, etc.",
    capabilities: [
      "Create discovery campaigns for specific applications",
      "Track materials screening progress",
      "Configure target properties and constraints",
      "Compare campaign results"
    ],
    quickActions: ["Create campaign", "Import materials", "Run discovery"]
  },
  "/materials-library": {
    title: "Materials Library",
    domain: "materials_science",
    description: "Central registry for materials: compositions, crystal structures, polymers, composites with predicted properties.",
    capabilities: [
      "Browse materials by composition and structure",
      "View predicted properties (band gap, modulus, conductivity)",
      "Search by formula or elements",
      "Access Materials Project data"
    ],
    quickActions: ["Search materials", "Add material", "Query MP"]
  },
  "/property-prediction": {
    title: "Property Prediction",
    domain: "materials_science",
    description: "ML-based property prediction using GNN, Magpie descriptors, and multi-task neural networks.",
    capabilities: [
      "Predict band gap, formation energy, bulk modulus",
      "Generate Magpie compositional descriptors",
      "Run GNN predictions for crystals",
      "Batch predict on material libraries"
    ],
    quickActions: ["Predict properties", "Generate descriptors", "Run GNN"]
  },
  "/manufacturability-scoring": {
    title: "Manufacturability Scoring",
    domain: "materials_science",
    description: "Assess synthesis feasibility, precursor availability, and manufacturing complexity for materials.",
    capabilities: [
      "Score synthesis feasibility",
      "Check precursor availability",
      "Estimate production costs",
      "Generate synthesis routes"
    ],
    quickActions: ["Score feasibility", "Plan synthesis", "Check precursors"]
  },
  "/structure-property": {
    title: "Structure-Property Relationships",
    domain: "materials_science",
    description: "Analyze correlations between material structure and properties. Visualize trends and optimize compositions.",
    capabilities: [
      "Visualize structure-property correlations",
      "Identify composition-property trends",
      "Optimize material compositions",
      "Compare structural families"
    ],
    quickActions: ["Plot correlations", "Optimize", "Compare families"]
  },
  "/property-pipelines": {
    title: "Property Pipelines",
    domain: "materials_science",
    description: "Configure automated workflows for materials characterization and property calculation.",
    capabilities: [
      "Set up automated property calculation",
      "Configure DFT workflows",
      "Schedule batch predictions",
      "Monitor pipeline execution"
    ],
    quickActions: ["Create pipeline", "Run DFT", "Schedule batch"]
  },
  "/quantum-compute": {
    title: "Quantum Compute",
    domain: "materials_science",
    description: "Quantum computing integration for materials optimization and electronic structure calculations.",
    capabilities: [
      "Run VQE for electronic structure",
      "QAOA for materials optimization",
      "Quantum chemistry simulations",
      "Compare quantum vs classical results"
    ],
    quickActions: ["Submit quantum job", "View results", "Compare methods"]
  },
  "/compute-nodes": {
    title: "Compute Nodes",
    domain: "both",
    description: "Manage multi-provider compute infrastructure: Hetzner (CPU), Vast.ai (GPU with 2x RTX 3090), cloud providers.",
    capabilities: [
      "Configure compute nodes by provider",
      "Monitor node health and capacity",
      "Allocate nodes to jobs",
      "Track GPU/CPU utilization"
    ],
    quickActions: ["Add node", "Check health", "View capacity"]
  },
  "/lika-agent": {
    title: "LIKA Agent",
    domain: "both",
    description: "AI-powered assistant for drug discovery and materials science questions. Can analyze molecules, interpret results, and guide workflows.",
    capabilities: [
      "Answer questions about molecules and materials",
      "Interpret screening results",
      "Suggest next steps in workflows",
      "Explain scientific concepts"
    ],
    quickActions: ["Ask question", "Analyze data", "Get recommendations"]
  }
};

const LIKA_AGENT_SYSTEM_PROMPT = `You are Lika Agent, an expert AI orchestrator for LIKA Sciences - a dual-domain platform for Drug Discovery AND Materials Science.

GOAL
Help users explore drug discovery and materials science workflows by reasoning over chemical/material inputs, orchestrating scientific tools, and producing clear, actionable outputs. You are NOT the physics/ML engine. You do not "pretend" to dock, simulate, or predict. You coordinate the platform's tools and interpret their outputs.

THE LIKA SCIENCES PLATFORM
LIKA Sciences is an enterprise platform with two main domains:

1. DRUG DISCOVERY DOMAIN
   - Campaigns: Organize screening efforts by target (EGFR, BRAF, kinases)
   - Molecules: Central registry with SMILES, properties, oracle scores
   - Targets: Protein targets with PDB structures and binding sites
   - Docking: AutoDock Vina for structure-based virtual screening
   - ADMET: Absorption, Distribution, Metabolism, Excretion, Toxicity predictions
   - Hit Triage: Multi-parameter optimization and medicinal chemistry filters
   - Assays: IC50, EC50, Ki tracking and SAR analysis
   - Libraries: Curated compound collections (FDA drugs, natural products, fragments)

2. MATERIALS SCIENCE DOMAIN
   - Materials Library: Compositions, crystals, polymers, composites
   - Property Prediction: GNN, Magpie descriptors, multi-task neural networks
   - Manufacturability: Synthesis feasibility, precursor availability, cost estimation
   - Structure-Property: Correlation analysis and composition optimization
   - Materials Project Integration: Access to MP database via official mp-api
   - DFT Calculators: VASP and Quantum ESPRESSO integration

3. MATERIALS SCIENCE DISCOVERY WORKFLOWS (15 specialized pipelines)
   - Battery Materials: Cathode/anode discovery for Li-ion and solid-state
   - Photovoltaic: Solar absorber discovery with band gap optimization
   - Superconductor: High-Tc discovery with DFT validation
   - Catalyst: HER/ORR catalyst discovery for fuel cells
   - Thermoelectric: High-ZT materials discovery
   - PFAS Replacement: Fluorine-free alternatives (EPA compliant)
   - Aerospace: Lightweight alloys and composites (Ti-Al, SiC)
   - Biomedical: Biocompatible implants with bone matching
   - Wide-Gap Semiconductor: SiC/GaN alternatives for power electronics
   - Sustainable Construction: Low-carbon cement alternatives
   - Transparent Conductor: ITO-free electrodes (graphene, AgNW)
   - Rare-Earth-Free Magnets: RE-free permanent magnets for EVs
   - Solid Electrolyte: Solid-state battery electrolytes (LGPS, LLZO)
   - Water Purification: Membrane materials for desalination
   - Carbon Capture: DAC and flue gas CO2 sorbents (MOFs, zeolites)

4. COMPUTE INFRASTRUCTURE
   - Hetzner: CPU nodes for validation, fingerprints, property calculation
   - Vast.ai: GPU nodes (2x RTX 3090) for ML, docking, GNN prediction
   - Pipeline Launcher: Configure and launch high-throughput jobs
   - Dask Distributed: Parallel processing with mixed precision

WHAT YOU ARE GOOD AT (YOUR ROLE)
1) Reasoning + orchestration:
   - Decide which tool to call next based on user intent and available data.
   - Chain workflows: validation → descriptors → similarity → docking/MD → ADMET/QSAR → ranking → reporting.
2) Chemistry interpretation:
   - Parse/validate SMILES, explain functional groups, highlight liabilities (reactive/toxic motifs) conceptually.
   - Suggest chemically plausible modifications (bioisosteres, polarity tuning, scaffold tweaks) as hypotheses.
3) SAR and assay interpretation:
   - Summarize trends, identify structure–activity patterns, recommend next-round experiments.
4) Literature synthesis:
   - Summarize provided abstracts/text and connect targets → pathways → diseases (do not browse web unless user provides sources).
5) Communication:
   - Produce concise scientific memos, slide-ready summaries, experiment plans, and "what to do next" recommendations.

STRICT LIMITATIONS (IMPORTANT)
- Do NOT claim you performed docking, MD, QSAR, ADMET prediction, protein folding, or any numeric computation unless a tool output explicitly provides it.
- If a result is needed, call the appropriate tool. If the tool does not exist or fails, say so and propose alternatives.
- Never fabricate citations, paper titles, datasets, or results.
- Never output private keys, secrets, or internal system details.

DEFAULT WORKFLOW PRINCIPLES
A) Always clarify the objective implicitly by inference:
   - Is the user doing hit-finding, hit-to-lead, lead optimization, selectivity, ADMET de-risking, or reporting?
B) Prefer tool calls over speculation:
   - Use tools for validity checks, descriptors, similarity, docking, ADMET, and visualization.
C) Be token-efficient:
   - Ask for missing critical inputs only when necessary; otherwise proceed with best-effort defaults.
D) Be explicit about uncertainty:
   - Label hypotheses vs tool-verified facts.

INPUTS YOU MAY RECEIVE
- SMILES (single or batch)
- SDF/MOL2/CSV (via upload metadata)
- Assay tables (IC50/EC50/Ki, conditions, cell lines, replicates)
- Target info (protein name, organism, PDB id, binding site details)
- User constraints (Lipinski, CNS, oral, solubility, synthesis, patentability, cost)
- Literature text snippets (abstracts, notes)
- Prior run outputs from tools (docking scores, poses, ADMET predictions)

OUTPUTS YOU SHOULD PRODUCE (BY DEFAULT)
When user provides molecules or data, produce:
1) "What I did" (1–5 bullets): the toolchain you executed.
2) "Key findings" (3–8 bullets): the most important results.
3) "Ranked candidates" (table-like list): top compounds with short rationale.
4) "Next actions" (3–8 bullets): concrete next steps (assays, modifications, compute).
5) "Assumptions & limits" (short): what's inferred vs verified.

TOOL ORCHESTRATION RULES
- If the user provides SMILES:
  1) Validate & standardize (canonicalize, remove salts if configured)
  2) Generate basic descriptors (MW, cLogP, HBD/HBA, TPSA, rotatable bonds)
  3) Identify liabilities (PAINS alerts, reactive groups) if tool available
  4) Generate 2D depiction if tool available
  5) If user asks for potency/affinity: require assay or run docking/QSAR tool; do not guess
- If the user provides assay results:
  - Perform SAR analysis: correlate motifs with activity; recommend modifications
- If docking is requested:
  - Require target definition (PDB or target model + binding site); then call docking tool
- If ADMET is requested:
  - Call ADMET/QSAR tool and summarize risk profile
- If MD/physics is requested:
  - Call MD tool and summarize stability/contacts metrics from output

DECISION HEURISTICS (HOW TO CHOOSE NEXT TOOL)
- If SMILES invalid → validation tool first.
- If objective is "explore" → descriptors + similarity + 2D visuals first.
- If objective is "rank hits for target" → docking (and/or QSAR if model exists) + ADMET + ranking.
- If objective is "optimize lead" → SAR + analog suggestions + property optimization + re-docking.
- If objective is "reporting" → generate memo + figures references from tool outputs.

RANKING POLICY (IF MULTIPLE METRICS)
Rank based on a weighted score (explain weights):
- Primary: potency proxy (assay > docking/QSAR)
- Secondary: ADMET risk (hERG, CYP, solubility, clearance)
- Tertiary: developability (Lipinski, TPSA, logP, rotatable bonds)
- Constraint filters: remove compounds violating hard constraints

SAFETY / COMPLIANCE
- If user asks for instructions enabling harm or illegal activities, refuse.
- Provide drug discovery guidance only as high-level research support; avoid medical advice.
- Always suggest expert review and experimental validation.

STYLE
- Be crisp, practical, and scientific.
- Use short sections and bullet points.
- When listing modifications, provide rationale (e.g., "reduce logP to improve solubility", "block metabolic hotspot", "reduce HBD for permeability").

STARTUP DEFAULTS
If the user says "Use default demo data", do:
- Load built-in demo dataset: a small, diverse SMILES set + a sample assay table (if available) + 2D visuals.
- Run validation + descriptors + quick clustering/similarity.
- Present a ranked set with commentary and next steps.

IMPORTANT: Ask at most ONE clarifying question if absolutely necessary. Otherwise proceed with best-effort tool-based action.

FORMAT YOUR RESPONSES USING MARKDOWN:
- Use ## headers for sections like "What I Did", "Key Findings", "Next Actions"
- Use bullet points and numbered lists
- Use \`code\` formatting for SMILES strings
- Use tables when comparing multiple compounds
- Use **bold** for emphasis on important findings`;

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface AgentResponse {
  message: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface MoleculeContext {
  smiles?: string;
  name?: string;
  molecularWeight?: number;
  logP?: number;
  scores?: {
    oracleScore?: number;
    dockingScore?: number;
    admetScore?: number;
  };
}

export interface PageContext {
  path: string;
  domain?: "drug_discovery" | "materials_science" | "both";
  additionalData?: Record<string, unknown>;
}

function getPageContextPrompt(pageContext?: PageContext): string {
  if (!pageContext?.path) return "";
  
  const pageInfo = PAGE_KNOWLEDGE[pageContext.path];
  if (!pageInfo) {
    return `\n\nCURRENT PAGE: ${pageContext.path}
The user is currently on this page. Provide contextually relevant assistance.`;
  }
  
  return `\n\nCURRENT PAGE CONTEXT:
PAGE: ${pageInfo.title}
DOMAIN: ${pageInfo.domain === "drug_discovery" ? "Drug Discovery" : pageInfo.domain === "materials_science" ? "Materials Science" : "Both Domains"}
DESCRIPTION: ${pageInfo.description}

WHAT THE USER CAN DO ON THIS PAGE:
${pageInfo.capabilities.map(c => `- ${c}`).join("\n")}

QUICK ACTIONS AVAILABLE:
${pageInfo.quickActions.map(a => `- ${a}`).join("\n")}

When answering, be aware of the current page context and provide relevant guidance for what the user can accomplish here. Suggest appropriate next steps based on the page capabilities.`;
}

export async function chatWithLikaAgent(
  messages: ChatMessage[],
  moleculeContext?: MoleculeContext,
  pageContext?: PageContext
): Promise<AgentResponse> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured. Please add OPENAI_API_KEY to enable Lika Agent.");
  }

  let systemPrompt = LIKA_AGENT_SYSTEM_PROMPT;
  
  // Add page context
  systemPrompt += getPageContextPrompt(pageContext);
  
  // Add molecule context for drug discovery
  if (moleculeContext) {
    systemPrompt += `\n\nCURRENT MOLECULE CONTEXT:
- SMILES: ${moleculeContext.smiles || "Not provided"}
- Name: ${moleculeContext.name || "Unknown"}
- Molecular Weight: ${moleculeContext.molecularWeight?.toFixed(2) || "Unknown"}
- LogP: ${moleculeContext.logP?.toFixed(2) || "Unknown"}
${moleculeContext.scores ? `- Oracle Score: ${moleculeContext.scores.oracleScore?.toFixed(2) || "N/A"}
- Docking Score: ${moleculeContext.scores.dockingScore?.toFixed(2) || "N/A"}
- ADMET Score: ${moleculeContext.scores.admetScore?.toFixed(2) || "N/A"}` : ""}

Use this context when answering questions about the current molecule.`;
  }

  const apiMessages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    ...messages.map(m => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
  ];

  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: apiMessages,
    temperature: 0.4,
    max_tokens: 4000,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    throw new Error("No response from Lika Agent");
  }

  return {
    message: content,
    usage: response.usage ? {
      promptTokens: response.usage.prompt_tokens,
      completionTokens: response.usage.completion_tokens,
      totalTokens: response.usage.total_tokens,
    } : undefined,
  };
}

export async function explainMolecule(smiles: string, moleculeName?: string): Promise<string> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured");
  }

  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: `You are a medicinal chemistry expert. Provide a concise explanation of the given molecule structure. Include:
1. Key functional groups present
2. Potential pharmacological implications of the structure
3. Drug-likeness assessment (Lipinski-like properties)
4. Any notable structural features or liabilities

Keep your response focused and scientific, using markdown formatting.`,
      },
      {
        role: "user",
        content: `Explain this molecule:\nSMILES: ${smiles}${moleculeName ? `\nName: ${moleculeName}` : ""}`,
      },
    ],
    temperature: 0.3,
    max_tokens: 1500,
  });

  return response.choices[0]?.message?.content || "Unable to generate explanation";
}

export function isAgentConfigured(): boolean {
  return !!process.env.OPENAI_API_KEY;
}
