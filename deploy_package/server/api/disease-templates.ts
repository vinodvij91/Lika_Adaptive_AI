import { searchPubChemAssays, getAssayById, type PubChemAssay } from "./pubchem-harvester";
import { searchChEMBLAssays, type ChEMBLAssay } from "./chembl-harvester";
import { type AssayCategory } from "./assay-classifier";

export interface DiseaseTarget {
  name: string;
  symbol: string;
  uniprotId?: string;
  chemblId?: string;
  role: "primary" | "secondary" | "safety";
}

export interface RecommendedAssay {
  id: string;
  name: string;
  description: string;
  source: "PubChem" | "ChEMBL" | "Manual";
  category: AssayCategory;
  targetName?: string;
  confidence: number;
}

export interface DiseaseTemplate {
  disease: string;
  therapeuticArea: string;
  targets: DiseaseTarget[];
  recommendedAssays: Record<AssayCategory, RecommendedAssay[]>;
  lastUpdated: string;
}

const DISEASE_TARGETS: Record<string, DiseaseTarget[]> = {
  "Alzheimer's Disease": [
    { name: "Amyloid Beta 42", symbol: "APP", uniprotId: "P05067", role: "primary" },
    { name: "Alpha-synuclein", symbol: "SNCA", uniprotId: "P37840", role: "primary" },
    { name: "NLRP3 Inflammasome", symbol: "NLRP3", uniprotId: "Q96P20", role: "primary" },
    { name: "TFEB", symbol: "TFEB", uniprotId: "P19484", role: "secondary" },
    { name: "ULK1", symbol: "ULK1", uniprotId: "O75385", role: "secondary" },
    { name: "Tau Protein", symbol: "MAPT", uniprotId: "P10636", role: "primary" },
    { name: "BACE1", symbol: "BACE1", uniprotId: "P56817", role: "primary" },
    { name: "Acetylcholinesterase", symbol: "ACHE", uniprotId: "P22303", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Oncology": [
    { name: "EGFR", symbol: "EGFR", uniprotId: "P00533", role: "primary" },
    { name: "KRAS", symbol: "KRAS", uniprotId: "P01116", role: "primary" },
    { name: "TP53", symbol: "TP53", uniprotId: "P04637", role: "primary" },
    { name: "BCL2", symbol: "BCL2", uniprotId: "P10415", role: "secondary" },
    { name: "CDK4", symbol: "CDK4", uniprotId: "P11802", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Metabolic": [
    { name: "GLUT4", symbol: "SLC2A4", uniprotId: "P14672", role: "primary" },
    { name: "PPAR-gamma", symbol: "PPARG", uniprotId: "P37231", role: "primary" },
    { name: "AMPK", symbol: "PRKAA1", uniprotId: "Q13131", role: "primary" },
    { name: "GLP-1R", symbol: "GLP1R", uniprotId: "P43220", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Vaccines": [
    { name: "Spike Protein", symbol: "S", role: "primary" },
    { name: "Nucleocapsid", symbol: "N", role: "secondary" },
    { name: "RBD", symbol: "RBD", role: "primary" }
  ],
  "Materials": [
    { name: "Physical Properties", symbol: "PHYS", role: "primary" },
    { name: "Stability", symbol: "STAB", role: "primary" },
    { name: "Formulation", symbol: "FORM", role: "secondary" }
  ]
};

const CURATED_ASSAYS: Record<string, RecommendedAssay[]> = {
  "Alzheimer's Disease": [
    {
      id: "AID 720543",
      name: "Aβ42 aggregation inhibition",
      description: "Primary screen for compounds that inhibit Aβ42 peptide aggregation using ThT fluorescence",
      source: "PubChem",
      category: "binding",
      targetName: "Amyloid Beta 42",
      confidence: 0.95
    },
    {
      id: "AID 1508",
      name: "Beta-secretase (BACE1) inhibition",
      description: "High-throughput screen for BACE1 inhibitors using FRET-based assay",
      source: "PubChem",
      category: "binding",
      targetName: "BACE1",
      confidence: 0.92
    },
    {
      id: "CHEMBL3215112",
      name: "Tau aggregation inhibition",
      description: "Screen for compounds preventing tau protein aggregation",
      source: "ChEMBL",
      category: "functional",
      targetName: "Tau Protein",
      confidence: 0.88
    },
    {
      id: "NLRP3-FUNC-001",
      name: "NLRP3 inflammasome inhibition",
      description: "IL-1β secretion assay in THP-1 cells stimulated with LPS+ATP",
      source: "Manual",
      category: "functional",
      targetName: "NLRP3",
      confidence: 0.90
    },
    {
      id: "BBB-PERM-001",
      name: "hiPSC-BBB permeability",
      description: "Blood-brain barrier permeability using human iPSC-derived brain microvascular endothelial cells",
      source: "Manual",
      category: "adme",
      targetName: "CNS Penetration",
      confidence: 0.85
    },
    {
      id: "PGP-EFFLUX-001",
      name: "P-gp efflux ratio",
      description: "MDCK-MDR1 bidirectional transport assay for P-glycoprotein substrate assessment",
      source: "Manual",
      category: "adme",
      targetName: "CNS Penetration",
      confidence: 0.88
    },
    {
      id: "HERG-001",
      name: "hERG channel inhibition",
      description: "Automated patch clamp assay for cardiac safety",
      source: "Manual",
      category: "safety",
      targetName: "hERG",
      confidence: 0.95
    },
    {
      id: "NEURO-TOX-001",
      name: "Neuronal cytotoxicity",
      description: "Primary cortical neuron viability assay (ATP-based)",
      source: "Manual",
      category: "safety",
      targetName: "Neurons",
      confidence: 0.87
    }
  ]
};

export function getDiseaseTargets(disease: string): DiseaseTarget[] {
  return DISEASE_TARGETS[disease] || [];
}

export function getCuratedAssays(disease: string): RecommendedAssay[] {
  return CURATED_ASSAYS[disease] || [];
}

export async function harvestAssaysForDisease(disease: string): Promise<RecommendedAssay[]> {
  const targets = getDiseaseTargets(disease);
  const curatedAssays = getCuratedAssays(disease);
  const harvestedAssays: RecommendedAssay[] = [...curatedAssays];

  for (const target of targets.slice(0, 3)) {
    try {
      const pubchemAssays = await searchPubChemAssays(target.symbol, 5);
      for (const assay of pubchemAssays) {
        if (!harvestedAssays.some(a => a.id === `AID ${assay.aid}`)) {
          harvestedAssays.push({
            id: `AID ${assay.aid}`,
            name: assay.name,
            description: assay.description,
            source: "PubChem",
            category: assay.category,
            targetName: target.name,
            confidence: assay.confidence
          });
        }
      }

      const chemblAssays = await searchChEMBLAssays(target.symbol, 5);
      for (const assay of chemblAssays) {
        if (!harvestedAssays.some(a => a.id === assay.chemblId)) {
          harvestedAssays.push({
            id: assay.chemblId,
            name: assay.name,
            description: assay.description,
            source: "ChEMBL",
            category: assay.category,
            targetName: target.name,
            confidence: assay.confidence
          });
        }
      }
    } catch (err) {
      console.error(`Error harvesting assays for target ${target.symbol}:`, err);
    }
  }

  return harvestedAssays;
}

export async function buildDiseaseTemplate(disease: string, useCache: boolean = true): Promise<DiseaseTemplate> {
  const targets = getDiseaseTargets(disease);
  const assays = useCache ? getCuratedAssays(disease) : await harvestAssaysForDisease(disease);

  const categorizedAssays: Record<AssayCategory, RecommendedAssay[]> = {
    binding: [],
    functional: [],
    adme: [],
    safety: [],
    physicochemical: []
  };

  for (const assay of assays) {
    categorizedAssays[assay.category].push(assay);
  }

  const therapeuticAreaMap: Record<string, string> = {
    "Alzheimer's Disease": "CNS",
    "Parkinson's Disease": "CNS",
    "Oncology": "Oncology",
    "Metabolic": "Cardiometabolic",
    "Vaccines": "Infectious",
    "Materials": "Other"
  };

  return {
    disease,
    therapeuticArea: therapeuticAreaMap[disease] || "Other",
    targets,
    recommendedAssays: categorizedAssays,
    lastUpdated: new Date().toISOString()
  };
}

export function getAvailableDiseases(): string[] {
  return Object.keys(DISEASE_TARGETS);
}
