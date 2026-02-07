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
  "Huntington's Disease": [
    { name: "Huntingtin", symbol: "HTT", uniprotId: "P42858", role: "primary" },
    { name: "mTOR", symbol: "MTOR", uniprotId: "P42345", role: "primary" },
    { name: "HDAC4", symbol: "HDAC4", uniprotId: "P56524", role: "primary" },
    { name: "PDE10A", symbol: "PDE10A", uniprotId: "Q9Y233", role: "primary" },
    { name: "Caspase-6", symbol: "CASP6", uniprotId: "P55212", role: "secondary" },
    { name: "BDNF", symbol: "BDNF", uniprotId: "P23560", role: "secondary" },
    { name: "SIRT1", symbol: "SIRT1", uniprotId: "Q96EB6", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Parkinson's Disease": [
    { name: "Alpha-synuclein", symbol: "SNCA", uniprotId: "P37840", role: "primary" },
    { name: "LRRK2", symbol: "LRRK2", uniprotId: "Q5S007", role: "primary" },
    { name: "PINK1", symbol: "PINK1", uniprotId: "Q9BXM7", role: "primary" },
    { name: "Parkin", symbol: "PRKN", uniprotId: "O60260", role: "primary" },
    { name: "GBA1", symbol: "GBA", uniprotId: "P04062", role: "secondary" },
    { name: "Dopamine Transporter", symbol: "SLC6A3", uniprotId: "Q01959", role: "secondary" },
    { name: "MAO-B", symbol: "MAOB", uniprotId: "P27338", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Breast Cancer": [
    { name: "HER2", symbol: "ERBB2", uniprotId: "P04626", role: "primary" },
    { name: "Estrogen Receptor Alpha", symbol: "ESR1", uniprotId: "P03372", role: "primary" },
    { name: "CDK4/6", symbol: "CDK4", uniprotId: "P11802", role: "primary" },
    { name: "PI3K alpha", symbol: "PIK3CA", uniprotId: "P42336", role: "primary" },
    { name: "BRCA1", symbol: "BRCA1", uniprotId: "P38398", role: "secondary" },
    { name: "PARP1", symbol: "PARP1", uniprotId: "P09874", role: "secondary" },
    { name: "mTOR", symbol: "MTOR", uniprotId: "P42345", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Type 2 Diabetes": [
    { name: "GLP-1R", symbol: "GLP1R", uniprotId: "P43220", role: "primary" },
    { name: "DPP-4", symbol: "DPP4", uniprotId: "P27487", role: "primary" },
    { name: "SGLT2", symbol: "SLC5A2", uniprotId: "P31639", role: "primary" },
    { name: "PPAR-gamma", symbol: "PPARG", uniprotId: "P37231", role: "primary" },
    { name: "AMPK", symbol: "PRKAA1", uniprotId: "Q13131", role: "secondary" },
    { name: "Glucokinase", symbol: "GCK", uniprotId: "P35557", role: "secondary" },
    { name: "GLUT4", symbol: "SLC2A4", uniprotId: "P14672", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Rheumatoid Arthritis": [
    { name: "TNF-alpha", symbol: "TNF", uniprotId: "P01375", role: "primary" },
    { name: "JAK1", symbol: "JAK1", uniprotId: "P23458", role: "primary" },
    { name: "JAK3", symbol: "JAK3", uniprotId: "P52333", role: "primary" },
    { name: "IL-6", symbol: "IL6", uniprotId: "P05231", role: "primary" },
    { name: "BTK", symbol: "BTK", uniprotId: "Q06187", role: "secondary" },
    { name: "SYK", symbol: "SYK", uniprotId: "P43405", role: "secondary" },
    { name: "p38 MAPK", symbol: "MAPK14", uniprotId: "Q16539", role: "secondary" },
    { name: "hERG", symbol: "KCNH2", role: "safety" }
  ],
  "Oncology": [
    { name: "EGFR", symbol: "EGFR", uniprotId: "P00533", role: "primary" },
    { name: "KRAS", symbol: "KRAS", uniprotId: "P01116", role: "primary" },
    { name: "TP53", symbol: "TP53", uniprotId: "P04637", role: "primary" },
    { name: "BCL2", symbol: "BCL2", uniprotId: "P10415", role: "secondary" },
    { name: "CDK4", symbol: "CDK4", uniprotId: "P11802", role: "secondary" },
    { name: "PD-L1", symbol: "CD274", uniprotId: "Q9NZQ7", role: "secondary" },
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
    { name: "RBD", symbol: "RBD", role: "primary" },
    { name: "Hemagglutinin", symbol: "HA", role: "primary" },
    { name: "Neuraminidase", symbol: "NA", role: "secondary" }
  ],
  "Materials": [
    { name: "Physical Properties", symbol: "PHYS", role: "primary" },
    { name: "Stability", symbol: "STAB", role: "primary" },
    { name: "Formulation", symbol: "FORM", role: "secondary" }
  ]
};

const GENERIC_ADME_ASSAYS: RecommendedAssay[] = [
  {
    id: "ADME-SOLU-001",
    name: "Kinetic solubility (pH 7.4)",
    description: "Nephelometric kinetic solubility assay in phosphate buffer at physiological pH",
    source: "Manual",
    category: "adme",
    targetName: "Solubility",
    confidence: 0.90
  },
  {
    id: "ADME-PERM-001",
    name: "Caco-2 permeability",
    description: "Bidirectional Caco-2 cell monolayer permeability assay for intestinal absorption prediction",
    source: "Manual",
    category: "adme",
    targetName: "Permeability",
    confidence: 0.88
  },
  {
    id: "ADME-MET-001",
    name: "Microsomal stability (human)",
    description: "Human liver microsome metabolic stability assay measuring intrinsic clearance",
    source: "Manual",
    category: "adme",
    targetName: "Metabolism",
    confidence: 0.92
  },
  {
    id: "ADME-PPB-001",
    name: "Plasma protein binding",
    description: "Rapid equilibrium dialysis assay for plasma protein binding determination",
    source: "Manual",
    category: "adme",
    targetName: "Distribution",
    confidence: 0.85
  }
];

const GENERIC_SAFETY_ASSAYS: RecommendedAssay[] = [
  {
    id: "HERG-001",
    name: "hERG channel inhibition",
    description: "Automated patch clamp assay for cardiac safety - hERG potassium channel inhibition measurement",
    source: "Manual",
    category: "safety",
    targetName: "hERG",
    confidence: 0.95
  },
  {
    id: "SAFETY-CYTO-001",
    name: "HepG2 cytotoxicity",
    description: "Hepatocyte cytotoxicity assay using HepG2 cells with ATP-based viability readout",
    source: "Manual",
    category: "safety",
    targetName: "Hepatotoxicity",
    confidence: 0.90
  },
  {
    id: "SAFETY-GENO-001",
    name: "Mini-Ames mutagenicity",
    description: "Abbreviated Ames test for mutagenicity screening using Salmonella strains TA98 and TA100",
    source: "Manual",
    category: "safety",
    targetName: "Genotoxicity",
    confidence: 0.85
  }
];

const GENERIC_PHYSCHEM_ASSAYS: RecommendedAssay[] = [
  {
    id: "PHYSCHEM-LOGD-001",
    name: "LogD 7.4 determination",
    description: "Shake-flask LogD measurement at pH 7.4 for lipophilicity assessment",
    source: "Manual",
    category: "physicochemical",
    targetName: "Lipophilicity",
    confidence: 0.88
  },
  {
    id: "PHYSCHEM-PKA-001",
    name: "pKa determination",
    description: "Potentiometric pKa measurement for ionization constant determination",
    source: "Manual",
    category: "physicochemical",
    targetName: "Ionization",
    confidence: 0.85
  }
];

const CURATED_ASSAYS: Record<string, RecommendedAssay[]> = {
  "Alzheimer's Disease": [
    {
      id: "AID 720543",
      name: "A\u03B242 aggregation inhibition",
      description: "Primary screen for compounds that inhibit A\u03B242 peptide aggregation using ThT fluorescence",
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
      description: "IL-1\u03B2 secretion assay in THP-1 cells stimulated with LPS+ATP",
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
      id: "NEURO-TOX-001",
      name: "Neuronal cytotoxicity",
      description: "Primary cortical neuron viability assay (ATP-based)",
      source: "Manual",
      category: "safety",
      targetName: "Neurons",
      confidence: 0.87
    }
  ],
  "Huntington's Disease": [
    {
      id: "HD-HTT-BIND-001",
      name: "Mutant Huntingtin aggregation inhibition",
      description: "Filter retardation assay measuring inhibition of mHTT exon 1 polyQ aggregation in vitro",
      source: "Manual",
      category: "binding",
      targetName: "Huntingtin",
      confidence: 0.93
    },
    {
      id: "HD-PDE10A-001",
      name: "PDE10A enzymatic inhibition",
      description: "Radiometric phosphodiesterase 10A enzymatic assay measuring cAMP/cGMP hydrolysis inhibition",
      source: "Manual",
      category: "binding",
      targetName: "PDE10A",
      confidence: 0.91
    },
    {
      id: "HD-HDAC4-001",
      name: "HDAC4 selective inhibition",
      description: "Fluorogenic class IIa HDAC4 deacetylase activity assay for selective inhibitor identification",
      source: "Manual",
      category: "binding",
      targetName: "HDAC4",
      confidence: 0.89
    },
    {
      id: "HD-MTOR-FUNC-001",
      name: "mTOR autophagy induction",
      description: "LC3-II/LC3-I ratio measurement in striatal cells to assess mTOR-dependent autophagy induction for mHTT clearance",
      source: "Manual",
      category: "functional",
      targetName: "mTOR",
      confidence: 0.88
    },
    {
      id: "HD-BDNF-FUNC-001",
      name: "BDNF transcription rescue",
      description: "BDNF promoter-driven luciferase reporter assay in mHTT-expressing striatal neurons",
      source: "Manual",
      category: "functional",
      targetName: "BDNF",
      confidence: 0.86
    },
    {
      id: "HD-CASP6-FUNC-001",
      name: "Caspase-6 cleavage inhibition",
      description: "Fluorogenic caspase-6 substrate cleavage assay measuring inhibition of HTT proteolysis at residue 586",
      source: "Manual",
      category: "functional",
      targetName: "Caspase-6",
      confidence: 0.87
    },
    {
      id: "BBB-PERM-002",
      name: "CNS penetration (MDCK-MDR1)",
      description: "MDCK-MDR1 bidirectional transport assay for blood-brain barrier penetration assessment",
      source: "Manual",
      category: "adme",
      targetName: "CNS Penetration",
      confidence: 0.88
    },
    {
      id: "HD-NEURO-TOX-001",
      name: "Striatal neuron viability",
      description: "STHdhQ111 striatal cell viability assay with serum deprivation stress",
      source: "Manual",
      category: "safety",
      targetName: "Striatal Neurons",
      confidence: 0.89
    }
  ],
  "Parkinson's Disease": [
    {
      id: "PD-SNCA-BIND-001",
      name: "Alpha-synuclein aggregation inhibition",
      description: "ThT fluorescence assay measuring inhibition of alpha-synuclein fibril formation",
      source: "Manual",
      category: "binding",
      targetName: "Alpha-synuclein",
      confidence: 0.94
    },
    {
      id: "PD-LRRK2-001",
      name: "LRRK2 kinase inhibition",
      description: "TR-FRET kinase assay for LRRK2 G2019S mutant inhibitor screening",
      source: "Manual",
      category: "binding",
      targetName: "LRRK2",
      confidence: 0.92
    },
    {
      id: "PD-GBA-001",
      name: "GBA1 glucocerebrosidase activity",
      description: "4-MU-Glc fluorogenic assay for GBA1 enzyme activity enhancement (chaperone screening)",
      source: "Manual",
      category: "functional",
      targetName: "GBA1",
      confidence: 0.87
    },
    {
      id: "PD-MITO-FUNC-001",
      name: "PINK1/Parkin mitophagy activation",
      description: "mt-Keima mitophagy reporter assay in PINK1-deficient dopaminergic neurons",
      source: "Manual",
      category: "functional",
      targetName: "PINK1/Parkin",
      confidence: 0.86
    },
    {
      id: "PD-MAOB-001",
      name: "MAO-B inhibition",
      description: "Amplex Red fluorometric assay for monoamine oxidase B selective inhibition",
      source: "Manual",
      category: "binding",
      targetName: "MAO-B",
      confidence: 0.90
    },
    {
      id: "BBB-PERM-003",
      name: "CNS permeability (PAMPA-BBB)",
      description: "PAMPA-BBB parallel artificial membrane permeability assay for CNS drug candidates",
      source: "Manual",
      category: "adme",
      targetName: "CNS Penetration",
      confidence: 0.86
    },
    {
      id: "PD-DA-TOX-001",
      name: "Dopaminergic neuron toxicity",
      description: "iPSC-derived dopaminergic neuron viability assay under oxidative stress conditions",
      source: "Manual",
      category: "safety",
      targetName: "DA Neurons",
      confidence: 0.88
    }
  ],
  "Breast Cancer": [
    {
      id: "BC-HER2-BIND-001",
      name: "HER2 kinase inhibition",
      description: "HTRF kinase assay measuring ErbB2/HER2 tyrosine kinase inhibition with ATP competition",
      source: "Manual",
      category: "binding",
      targetName: "HER2",
      confidence: 0.94
    },
    {
      id: "BC-ESR1-001",
      name: "Estrogen receptor alpha binding",
      description: "Fluorescence polarization competitive binding assay for ER-alpha with estradiol displacement",
      source: "Manual",
      category: "binding",
      targetName: "Estrogen Receptor Alpha",
      confidence: 0.93
    },
    {
      id: "BC-CDK46-001",
      name: "CDK4/6 kinase inhibition",
      description: "ADP-Glo kinase assay for selective CDK4/6 inhibitor identification using Rb substrate",
      source: "Manual",
      category: "binding",
      targetName: "CDK4/6",
      confidence: 0.91
    },
    {
      id: "BC-PI3K-001",
      name: "PI3K alpha selective inhibition",
      description: "ADP-Glo assay for PI3K-alpha (H1047R mutant) kinase activity inhibition screening",
      source: "Manual",
      category: "binding",
      targetName: "PI3K alpha",
      confidence: 0.90
    },
    {
      id: "BC-PROLIF-001",
      name: "MCF-7 proliferation inhibition",
      description: "CellTiter-Glo proliferation assay in ER+ MCF-7 breast cancer cells",
      source: "Manual",
      category: "functional",
      targetName: "ER+ Breast Cancer",
      confidence: 0.89
    },
    {
      id: "BC-PARP-FUNC-001",
      name: "PARP trapping assay",
      description: "PARP-DNA trapping assay in BRCA1-mutant HCC1937 cells measuring chromatin-bound PARP1",
      source: "Manual",
      category: "functional",
      targetName: "PARP1",
      confidence: 0.87
    },
    {
      id: "BC-CARDIO-001",
      name: "Cardiomyocyte contractility",
      description: "hiPSC-cardiomyocyte impedance assay for cardiac safety (HER2 inhibitor liability)",
      source: "Manual",
      category: "safety",
      targetName: "Cardiac Safety",
      confidence: 0.91
    }
  ],
  "Type 2 Diabetes": [
    {
      id: "T2D-DPP4-001",
      name: "DPP-4 enzymatic inhibition",
      description: "Fluorogenic Gly-Pro-AMC substrate assay for DPP-4 inhibitor screening",
      source: "Manual",
      category: "binding",
      targetName: "DPP-4",
      confidence: 0.93
    },
    {
      id: "T2D-GLP1R-001",
      name: "GLP-1R agonist potency",
      description: "cAMP accumulation assay in HEK293-GLP1R cells for incretin mimetic screening",
      source: "Manual",
      category: "binding",
      targetName: "GLP-1R",
      confidence: 0.92
    },
    {
      id: "T2D-SGLT2-001",
      name: "SGLT2 transport inhibition",
      description: "14C-AMG uptake assay in SGLT2-expressing CHO cells for gliflozin-class compound screening",
      source: "Manual",
      category: "binding",
      targetName: "SGLT2",
      confidence: 0.91
    },
    {
      id: "T2D-GLUT-FUNC-001",
      name: "Glucose uptake in adipocytes",
      description: "2-NBDG fluorescent glucose uptake assay in differentiated 3T3-L1 adipocytes with insulin stimulation",
      source: "Manual",
      category: "functional",
      targetName: "GLUT4",
      confidence: 0.88
    },
    {
      id: "T2D-GSIS-001",
      name: "Glucose-stimulated insulin secretion",
      description: "GSIS assay in INS-1E beta cells measuring insulin release in response to glucose challenge",
      source: "Manual",
      category: "functional",
      targetName: "Beta Cell Function",
      confidence: 0.87
    },
    {
      id: "T2D-HYPO-001",
      name: "Hypoglycemia risk assessment",
      description: "In vitro insulin secretion assay at low glucose (2.8 mM) to assess hypoglycemia liability",
      source: "Manual",
      category: "safety",
      targetName: "Hypoglycemia",
      confidence: 0.86
    }
  ],
  "Rheumatoid Arthritis": [
    {
      id: "RA-JAK1-001",
      name: "JAK1 kinase inhibition",
      description: "LanthaScreen TR-FRET kinase assay for JAK1 selective inhibitor screening",
      source: "Manual",
      category: "binding",
      targetName: "JAK1",
      confidence: 0.93
    },
    {
      id: "RA-JAK3-001",
      name: "JAK3 kinase inhibition",
      description: "ADP-Glo kinase assay for JAK3 inhibition selectivity profiling",
      source: "Manual",
      category: "binding",
      targetName: "JAK3",
      confidence: 0.91
    },
    {
      id: "RA-TNF-001",
      name: "TNF-alpha neutralization",
      description: "L929 cell-based TNF-alpha cytotoxicity neutralization assay for anti-TNF compound screening",
      source: "Manual",
      category: "binding",
      targetName: "TNF-alpha",
      confidence: 0.90
    },
    {
      id: "RA-IL6-FUNC-001",
      name: "IL-6/STAT3 signaling inhibition",
      description: "STAT3 phosphorylation ELISA in THP-1 cells stimulated with IL-6 for pathway inhibitor screening",
      source: "Manual",
      category: "functional",
      targetName: "IL-6/STAT3",
      confidence: 0.88
    },
    {
      id: "RA-FLS-FUNC-001",
      name: "FLS proliferation inhibition",
      description: "Fibroblast-like synoviocyte (RA-FLS) proliferation and MMP secretion assay",
      source: "Manual",
      category: "functional",
      targetName: "Synoviocytes",
      confidence: 0.86
    },
    {
      id: "RA-IMMUNO-001",
      name: "Immunosuppression liability",
      description: "T-cell proliferation assay (CFSE dilution) to assess immunosuppression risk at therapeutic concentrations",
      source: "Manual",
      category: "safety",
      targetName: "Immune Function",
      confidence: 0.88
    }
  ],
  "Oncology": [
    {
      id: "ONC-EGFR-001",
      name: "EGFR kinase inhibition",
      description: "HTRF kinase assay measuring EGFR tyrosine kinase inhibition with poly-GT substrate",
      source: "Manual",
      category: "binding",
      targetName: "EGFR",
      confidence: 0.94
    },
    {
      id: "ONC-KRAS-001",
      name: "KRAS G12C covalent binding",
      description: "Mass spectrometry-based assay for KRAS G12C covalent inhibitor engagement",
      source: "Manual",
      category: "binding",
      targetName: "KRAS",
      confidence: 0.91
    },
    {
      id: "ONC-BCL2-001",
      name: "BCL2 BH3 mimetic binding",
      description: "TR-FRET assay measuring displacement of BIM BH3 peptide from BCL-2 protein",
      source: "Manual",
      category: "binding",
      targetName: "BCL2",
      confidence: 0.89
    },
    {
      id: "ONC-PROLIF-001",
      name: "NCI-60 cell line panel",
      description: "Multi-cell-line proliferation screen across NCI-60 tumor panel for broad anti-cancer activity",
      source: "Manual",
      category: "functional",
      targetName: "Tumor Proliferation",
      confidence: 0.90
    },
    {
      id: "ONC-APOP-001",
      name: "Caspase 3/7 apoptosis induction",
      description: "Caspase-Glo 3/7 assay measuring apoptosis induction in A549 and HCT116 cells",
      source: "Manual",
      category: "functional",
      targetName: "Apoptosis",
      confidence: 0.87
    },
    {
      id: "ONC-MYELOSUPP-001",
      name: "Myelosuppression liability",
      description: "Colony-forming unit assay (CFU-GM, BFU-E) in human bone marrow progenitors",
      source: "Manual",
      category: "safety",
      targetName: "Bone Marrow",
      confidence: 0.89
    }
  ],
  "Metabolic": [
    {
      id: "MET-PPARG-001",
      name: "PPAR-gamma transactivation",
      description: "GAL4-PPAR-gamma chimeric receptor transactivation assay for agonist screening",
      source: "Manual",
      category: "binding",
      targetName: "PPAR-gamma",
      confidence: 0.91
    },
    {
      id: "MET-AMPK-001",
      name: "AMPK activation",
      description: "TR-FRET assay measuring AMPK alpha phosphorylation at Thr172 for activator screening",
      source: "Manual",
      category: "binding",
      targetName: "AMPK",
      confidence: 0.89
    },
    {
      id: "MET-GLUT-FUNC-001",
      name: "Glucose uptake enhancement",
      description: "2-NBDG fluorescent glucose analog uptake in L6 myotubes with insulin co-stimulation",
      source: "Manual",
      category: "functional",
      targetName: "GLUT4",
      confidence: 0.88
    },
    {
      id: "MET-ADIPO-001",
      name: "Adipogenesis modulation",
      description: "Oil Red O staining assay in 3T3-L1 pre-adipocytes measuring adipocyte differentiation",
      source: "Manual",
      category: "functional",
      targetName: "Adipogenesis",
      confidence: 0.85
    },
    {
      id: "MET-HYPO-001",
      name: "Hypoglycemia risk assessment",
      description: "Insulin secretion at low glucose concentrations (2.8 mM) in INS-1E cells",
      source: "Manual",
      category: "safety",
      targetName: "Hypoglycemia",
      confidence: 0.86
    }
  ],
  "Vaccines": [
    {
      id: "VAX-RBD-001",
      name: "RBD-ACE2 binding inhibition",
      description: "ELISA-based assay measuring inhibition of spike RBD binding to human ACE2 receptor",
      source: "Manual",
      category: "binding",
      targetName: "RBD",
      confidence: 0.93
    },
    {
      id: "VAX-NEUT-001",
      name: "Pseudovirus neutralization",
      description: "Lentiviral pseudovirus neutralization assay in HEK293-ACE2 cells for antibody titer measurement",
      source: "Manual",
      category: "functional",
      targetName: "Neutralization",
      confidence: 0.92
    },
    {
      id: "VAX-TCELL-001",
      name: "T-cell response (IFN-gamma ELISpot)",
      description: "IFN-gamma ELISpot assay measuring antigen-specific T-cell responses to vaccine candidates",
      source: "Manual",
      category: "functional",
      targetName: "T-cell Immunity",
      confidence: 0.90
    },
    {
      id: "VAX-REACT-001",
      name: "Reactogenicity assessment",
      description: "In vitro cytokine storm panel (IL-1b, IL-6, TNF-a, IFN-g) in human PBMC cultures",
      source: "Manual",
      category: "safety",
      targetName: "Reactogenicity",
      confidence: 0.88
    }
  ],
  "Materials": [
    {
      id: "MAT-STAB-001",
      name: "Thermal stability screening",
      description: "Differential scanning calorimetry (DSC) screening for thermal decomposition temperature",
      source: "Manual",
      category: "physicochemical",
      targetName: "Thermal Stability",
      confidence: 0.90
    },
    {
      id: "MAT-MECH-001",
      name: "Mechanical property assessment",
      description: "Nanoindentation hardness and elastic modulus measurement for materials screening",
      source: "Manual",
      category: "physicochemical",
      targetName: "Mechanical Properties",
      confidence: 0.88
    }
  ]
};

const THERAPEUTIC_AREA_MAP: Record<string, string> = {
  "Alzheimer's Disease": "CNS",
  "Huntington's Disease": "CNS",
  "Parkinson's Disease": "CNS",
  "Breast Cancer": "Oncology",
  "Oncology": "Oncology",
  "Type 2 Diabetes": "Cardiometabolic",
  "Metabolic": "Cardiometabolic",
  "Rheumatoid Arthritis": "Immunology",
  "Vaccines": "Infectious",
  "Materials": "Other"
};

export function getDiseaseTargets(disease: string): DiseaseTarget[] {
  return DISEASE_TARGETS[disease] || [];
}

export function getCuratedAssays(disease: string): RecommendedAssay[] {
  return CURATED_ASSAYS[disease] || [];
}

const GENERIC_BINDING_ASSAYS: RecommendedAssay[] = [
  { id: "GEN-BIND-001", name: "Target binding affinity (SPR)", description: "Surface plasmon resonance measurement of compound-target binding kinetics (ka, kd, KD)", source: "Manual", category: "binding", targetName: "Primary Target", confidence: 0.85 },
  { id: "GEN-BIND-002", name: "Radioligand displacement assay", description: "Competitive radioligand binding assay to determine Ki values against primary target", source: "Manual", category: "binding", targetName: "Primary Target", confidence: 0.83 },
];

const GENERIC_FUNCTIONAL_ASSAYS: RecommendedAssay[] = [
  { id: "GEN-FUNC-001", name: "Cell-based functional assay", description: "Cellular reporter assay measuring compound-induced modulation of target pathway activity", source: "Manual", category: "functional", targetName: "Pathway Activity", confidence: 0.82 },
  { id: "GEN-FUNC-002", name: "Dose-response curve (IC50/EC50)", description: "8-point dose-response determination in target-expressing cell line", source: "Manual", category: "functional", targetName: "Efficacy", confidence: 0.84 },
];

function getGenericFallbacks(disease: string): RecommendedAssay[] {
  const existing = CURATED_ASSAYS[disease] || [];
  const existingIds = new Set(existing.map(a => a.id));
  const fallbacks: RecommendedAssay[] = [];

  const hasBinding = existing.some(a => a.category === "binding");
  const hasFunctional = existing.some(a => a.category === "functional");
  const hasAdme = existing.some(a => a.category === "adme");
  const hasSafety = existing.some(a => a.category === "safety");
  const hasPhyschem = existing.some(a => a.category === "physicochemical");

  if (!hasBinding) {
    for (const a of GENERIC_BINDING_ASSAYS) {
      if (!existingIds.has(a.id)) fallbacks.push(a);
    }
  }
  if (!hasFunctional) {
    for (const a of GENERIC_FUNCTIONAL_ASSAYS) {
      if (!existingIds.has(a.id)) fallbacks.push(a);
    }
  }
  if (!hasAdme) {
    for (const a of GENERIC_ADME_ASSAYS) {
      if (!existingIds.has(a.id)) fallbacks.push(a);
    }
  }
  if (!hasSafety) {
    for (const a of GENERIC_SAFETY_ASSAYS) {
      if (!existingIds.has(a.id)) fallbacks.push(a);
    }
  }
  if (!hasPhyschem && disease !== "Materials") {
    for (const a of GENERIC_PHYSCHEM_ASSAYS) {
      if (!existingIds.has(a.id)) fallbacks.push(a);
    }
  }

  return fallbacks;
}

export async function harvestAssaysForDisease(disease: string): Promise<RecommendedAssay[]> {
  const targets = getDiseaseTargets(disease);
  const curatedAssays = getCuratedAssays(disease);
  const fallbacks = getGenericFallbacks(disease);
  const harvestedAssays: RecommendedAssay[] = [...curatedAssays, ...fallbacks];

  for (const target of targets.filter(t => t.role !== "safety").slice(0, 3)) {
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
  const assays = useCache
    ? [...getCuratedAssays(disease), ...getGenericFallbacks(disease)]
    : await harvestAssaysForDisease(disease);

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

  return {
    disease,
    therapeuticArea: THERAPEUTIC_AREA_MAP[disease] || "Other",
    targets,
    recommendedAssays: categorizedAssays,
    lastUpdated: new Date().toISOString()
  };
}

export function getAvailableDiseases(): string[] {
  return Object.keys(DISEASE_TARGETS);
}
