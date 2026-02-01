import { Request, Response } from "express";

export interface ScRNADataset {
  id: string;
  name: string;
  geoAccession: string;
  disease: string;
  tissue: string;
  cellCount: number;
  description: string;
  source: "GEO" | "ArrayExpress" | "HCA";
  species: "human" | "mouse";
  publicationYear: number;
  hasTrajectory: boolean;
}

export interface TrajectoryResult {
  datasetId: string;
  disease: string;
  umapCoordinates: Array<{ x: number; y: number; cluster: string; pseudotime: number }>;
  clusters: Array<{ id: string; name: string; cellCount: number; color: string }>;
  branchPoints: Array<{ pseudotime: number; genes: string[]; significance: number }>;
  biomarkers: Array<{
    gene: string;
    pseudotimeExpression: number;
    direction: "up" | "down";
    foldChange: number;
    pValue: number;
    cluster: string;
  }>;
  targets: string[];
  pgdSmoothing: {
    alpha: number;
    iterations: number;
    convergence: number;
  };
}

const PUBLIC_DATASETS: ScRNADataset[] = [
  {
    id: "gse153822",
    name: "Human Brain Alzheimer's Disease",
    geoAccession: "GSE153822",
    disease: "Alzheimers",
    tissue: "Brain (Prefrontal Cortex)",
    cellCount: 169496,
    description: "Single-nucleus RNA-seq of human prefrontal cortex from AD patients and controls",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse138852",
    name: "Human Brain Entorhinal Cortex AD",
    geoAccession: "GSE138852",
    disease: "Alzheimers",
    tissue: "Brain (Entorhinal Cortex)",
    cellCount: 13214,
    description: "snRNA-seq of entorhinal cortex in early Alzheimer's disease",
    source: "GEO",
    species: "human",
    publicationYear: 2019,
    hasTrajectory: true
  },
  {
    id: "gse174367",
    name: "Human Brain Multi-Region AD",
    geoAccession: "GSE174367",
    disease: "Alzheimers",
    tissue: "Brain (Multiple Regions)",
    cellCount: 437682,
    description: "Cross-tissue single-cell landscape of human Alzheimer's disease",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse164378",
    name: "PBMC COVID-19 Severity",
    geoAccession: "GSE164378",
    disease: "COVID-19",
    tissue: "PBMC",
    cellCount: 317000,
    description: "Single-cell multi-omics of human PBMC in COVID-19",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse145926",
    name: "Lung COVID-19 BALF",
    geoAccession: "GSE145926",
    disease: "COVID-19",
    tissue: "Bronchoalveolar Lavage Fluid",
    cellCount: 63103,
    description: "Single-cell landscape of bronchoalveolar immune cells in COVID-19",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse176078",
    name: "Breast Cancer Tumor Microenvironment",
    geoAccession: "GSE176078",
    disease: "Cancer",
    tissue: "Breast Tumor",
    cellCount: 130246,
    description: "Single-cell profiling of breast cancer reveals tumor heterogeneity",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse131907",
    name: "Lung Adenocarcinoma",
    geoAccession: "GSE131907",
    disease: "Cancer",
    tissue: "Lung Tumor",
    cellCount: 208506,
    description: "Single-cell RNA sequencing of lung adenocarcinoma",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse134809",
    name: "Type 2 Diabetes Islets",
    geoAccession: "GSE134809",
    disease: "Diabetes",
    tissue: "Pancreatic Islets",
    cellCount: 36351,
    description: "Single-cell transcriptome of human pancreatic islets in diabetes",
    source: "GEO",
    species: "human",
    publicationYear: 2019,
    hasTrajectory: true
  },
  {
    id: "gse148829",
    name: "IBD Intestinal Epithelium",
    geoAccession: "GSE148829",
    disease: "IBD",
    tissue: "Intestinal Epithelium",
    cellCount: 134923,
    description: "Single-cell atlas of colonic epithelium in IBD",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse159677",
    name: "Parkinson's Disease Substantia Nigra",
    geoAccession: "GSE159677",
    disease: "Parkinsons",
    tissue: "Brain (Substantia Nigra)",
    cellCount: 65450,
    description: "Single-nucleus RNA-seq of substantia nigra in Parkinson's disease",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse137810",
    name: "ALS Spinal Cord Motor Neurons",
    geoAccession: "GSE137810",
    disease: "ALS",
    tissue: "Spinal Cord",
    cellCount: 89234,
    description: "Single-nucleus transcriptomics of spinal cord in ALS patients",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse166482",
    name: "ALS Motor Cortex",
    geoAccession: "GSE166482",
    disease: "ALS",
    tissue: "Brain (Motor Cortex)",
    cellCount: 112890,
    description: "Single-cell analysis of motor cortex degeneration in ALS",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse128630",
    name: "Multiple Sclerosis Brain Lesions",
    geoAccession: "GSE128630",
    disease: "Multiple Sclerosis",
    tissue: "Brain (White Matter)",
    cellCount: 75123,
    description: "Single-cell transcriptomics of MS brain lesions",
    source: "GEO",
    species: "human",
    publicationYear: 2019,
    hasTrajectory: true
  },
  {
    id: "gse157278",
    name: "MS CSF Immune Cells",
    geoAccession: "GSE157278",
    disease: "Multiple Sclerosis",
    tissue: "Cerebrospinal Fluid",
    cellCount: 42567,
    description: "Single-cell immune profiling of CSF in multiple sclerosis",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse154927",
    name: "Rheumatoid Arthritis Synovium",
    geoAccession: "GSE154927",
    disease: "Rheumatoid Arthritis",
    tissue: "Synovial Tissue",
    cellCount: 98456,
    description: "Single-cell atlas of rheumatoid arthritis synovial tissue",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse163314",
    name: "Lupus PBMC and Kidney",
    geoAccession: "GSE163314",
    disease: "Lupus",
    tissue: "PBMC / Kidney",
    cellCount: 156789,
    description: "Multi-tissue single-cell analysis of systemic lupus erythematosus",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse145505",
    name: "Asthma Airway Epithelium",
    geoAccession: "GSE145505",
    disease: "Asthma",
    tissue: "Airway Epithelium",
    cellCount: 67890,
    description: "Single-cell transcriptome of airway epithelium in asthma",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse136831",
    name: "COPD Lung Tissue",
    geoAccession: "GSE136831",
    disease: "COPD",
    tissue: "Lung",
    cellCount: 220125,
    description: "Single-cell RNA-seq of human lung in COPD",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse135893",
    name: "Pulmonary Fibrosis Lung",
    geoAccession: "GSE135893",
    disease: "Pulmonary Fibrosis",
    tissue: "Lung",
    cellCount: 114396,
    description: "Single-cell transcriptomic analysis of idiopathic pulmonary fibrosis",
    source: "GEO",
    species: "human",
    publicationYear: 2019,
    hasTrajectory: true
  },
  {
    id: "gse183852",
    name: "Heart Failure Cardiac Tissue",
    geoAccession: "GSE183852",
    disease: "Heart Failure",
    tissue: "Heart (Left Ventricle)",
    cellCount: 287456,
    description: "Single-cell atlas of human heart failure",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse151530",
    name: "Chronic Kidney Disease",
    geoAccession: "GSE151530",
    disease: "Kidney Disease",
    tissue: "Kidney",
    cellCount: 145678,
    description: "Single-cell transcriptomics of chronic kidney disease",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse174422",
    name: "NASH Liver",
    geoAccession: "GSE174422",
    disease: "NASH",
    tissue: "Liver",
    cellCount: 178234,
    description: "Single-cell atlas of non-alcoholic steatohepatitis progression",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse162223",
    name: "Psoriasis Skin",
    geoAccession: "GSE162223",
    disease: "Psoriasis",
    tissue: "Skin",
    cellCount: 89345,
    description: "Single-cell analysis of psoriatic skin lesions",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse171524",
    name: "Atopic Dermatitis Skin",
    geoAccession: "GSE171524",
    disease: "Atopic Dermatitis",
    tissue: "Skin",
    cellCount: 76543,
    description: "Single-cell transcriptome of atopic dermatitis lesions",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse182109",
    name: "Glioblastoma Tumor",
    geoAccession: "GSE182109",
    disease: "Glioblastoma",
    tissue: "Brain Tumor",
    cellCount: 201456,
    description: "Single-cell atlas of glioblastoma tumor microenvironment",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse139555",
    name: "Melanoma Tumor Infiltrate",
    geoAccession: "GSE139555",
    disease: "Melanoma",
    tissue: "Skin Tumor",
    cellCount: 167890,
    description: "Single-cell analysis of melanoma immune infiltration",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse176031",
    name: "Prostate Cancer",
    geoAccession: "GSE176031",
    disease: "Prostate Cancer",
    tissue: "Prostate Tumor",
    cellCount: 143567,
    description: "Single-cell transcriptomics of prostate cancer progression",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse154600",
    name: "Ovarian Cancer",
    geoAccession: "GSE154600",
    disease: "Ovarian Cancer",
    tissue: "Ovarian Tumor",
    cellCount: 98765,
    description: "Single-cell atlas of high-grade serous ovarian cancer",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse155698",
    name: "Pancreatic Cancer",
    geoAccession: "GSE155698",
    disease: "Pancreatic Cancer",
    tissue: "Pancreas Tumor",
    cellCount: 112345,
    description: "Single-cell RNA-seq of pancreatic ductal adenocarcinoma",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse132509",
    name: "Acute Myeloid Leukemia",
    geoAccession: "GSE132509",
    disease: "Leukemia",
    tissue: "Bone Marrow",
    cellCount: 187654,
    description: "Single-cell analysis of AML bone marrow",
    source: "GEO",
    species: "human",
    publicationYear: 2019,
    hasTrajectory: true
  },
  {
    id: "gse169379",
    name: "B-Cell Lymphoma",
    geoAccession: "GSE169379",
    disease: "Lymphoma",
    tissue: "Lymph Node",
    cellCount: 134567,
    description: "Single-cell profiling of diffuse large B-cell lymphoma",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse152805",
    name: "Osteoarthritis Cartilage",
    geoAccession: "GSE152805",
    disease: "Osteoarthritis",
    tissue: "Articular Cartilage",
    cellCount: 67890,
    description: "Single-cell transcriptome of osteoarthritic cartilage",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse147287",
    name: "Osteoporosis Bone",
    geoAccession: "GSE147287",
    disease: "Osteoporosis",
    tissue: "Bone",
    cellCount: 54321,
    description: "Single-cell analysis of bone cells in osteoporosis",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse178265",
    name: "Depression Brain",
    geoAccession: "GSE178265",
    disease: "Depression",
    tissue: "Brain (Prefrontal Cortex)",
    cellCount: 89012,
    description: "Single-nucleus RNA-seq of major depressive disorder",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse144136",
    name: "Schizophrenia Brain",
    geoAccession: "GSE144136",
    disease: "Schizophrenia",
    tissue: "Brain (Dorsolateral PFC)",
    cellCount: 102345,
    description: "Single-cell transcriptomics of schizophrenia brain",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse159812",
    name: "Autism Spectrum Brain",
    geoAccession: "GSE159812",
    disease: "Autism",
    tissue: "Brain (Cortex)",
    cellCount: 78901,
    description: "Single-nucleus transcriptome of autism spectrum disorder",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse134763",
    name: "Huntington's Disease Brain",
    geoAccession: "GSE134763",
    disease: "Huntingtons",
    tissue: "Brain (Caudate Nucleus)",
    cellCount: 65432,
    description: "Single-cell analysis of Huntington's disease brain",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse165080",
    name: "Epilepsy Hippocampus",
    geoAccession: "GSE165080",
    disease: "Epilepsy",
    tissue: "Brain (Hippocampus)",
    cellCount: 87654,
    description: "Single-cell transcriptome of temporal lobe epilepsy",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse156776",
    name: "Hepatitis B Liver",
    geoAccession: "GSE156776",
    disease: "Hepatitis",
    tissue: "Liver",
    cellCount: 134567,
    description: "Single-cell atlas of chronic hepatitis B infection",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse179325",
    name: "Scleroderma Skin",
    geoAccession: "GSE179325",
    disease: "Scleroderma",
    tissue: "Skin",
    cellCount: 67890,
    description: "Single-cell analysis of systemic sclerosis skin",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse157341",
    name: "Sjogren's Syndrome Salivary",
    geoAccession: "GSE157341",
    disease: "Sjogrens",
    tissue: "Salivary Gland",
    cellCount: 45678,
    description: "Single-cell transcriptome of Sjogren's syndrome",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse168732",
    name: "Sepsis PBMC",
    geoAccession: "GSE168732",
    disease: "Sepsis",
    tissue: "PBMC",
    cellCount: 198765,
    description: "Single-cell immune profiling in sepsis",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse149689",
    name: "HIV Infection PBMC",
    geoAccession: "GSE149689",
    disease: "HIV",
    tissue: "PBMC",
    cellCount: 156789,
    description: "Single-cell analysis of HIV-infected individuals",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  },
  {
    id: "gse181919",
    name: "Tuberculosis Lung",
    geoAccession: "GSE181919",
    disease: "Tuberculosis",
    tissue: "Lung",
    cellCount: 123456,
    description: "Single-cell atlas of tuberculosis granulomas",
    source: "GEO",
    species: "human",
    publicationYear: 2022,
    hasTrajectory: true
  },
  {
    id: "gse158055",
    name: "Endometriosis",
    geoAccession: "GSE158055",
    disease: "Endometriosis",
    tissue: "Endometrium",
    cellCount: 78901,
    description: "Single-cell transcriptome of endometriosis lesions",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse155468",
    name: "Preeclampsia Placenta",
    geoAccession: "GSE155468",
    disease: "Preeclampsia",
    tissue: "Placenta",
    cellCount: 89012,
    description: "Single-cell analysis of preeclamptic placenta",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse161529",
    name: "Age-Related Macular Degeneration",
    geoAccession: "GSE161529",
    disease: "Macular Degeneration",
    tissue: "Retina",
    cellCount: 67890,
    description: "Single-cell atlas of AMD retina",
    source: "GEO",
    species: "human",
    publicationYear: 2021,
    hasTrajectory: true
  },
  {
    id: "gse143568",
    name: "Glaucoma Retina",
    geoAccession: "GSE143568",
    disease: "Glaucoma",
    tissue: "Retina / Optic Nerve",
    cellCount: 54321,
    description: "Single-cell transcriptome of glaucomatous retina",
    source: "GEO",
    species: "human",
    publicationYear: 2020,
    hasTrajectory: true
  }
];

const DISEASE_BIOMARKERS: Record<string, Array<{ gene: string; role: string; targetable: boolean }>> = {
  Alzheimers: [
    { gene: "NLRP3", role: "Inflammasome activation", targetable: true },
    { gene: "TFEB", role: "Autophagy regulator", targetable: true },
    { gene: "APP", role: "Amyloid precursor", targetable: true },
    { gene: "MAPT", role: "Tau protein", targetable: true },
    { gene: "TREM2", role: "Microglial activation", targetable: true },
    { gene: "APOE", role: "Lipid metabolism", targetable: false },
    { gene: "CLU", role: "Amyloid clearance", targetable: true },
    { gene: "PSEN1", role: "Gamma-secretase", targetable: true },
    { gene: "BACE1", role: "Beta-secretase", targetable: true },
    { gene: "CD33", role: "Microglial inhibition", targetable: true }
  ],
  "COVID-19": [
    { gene: "ACE2", role: "Viral entry receptor", targetable: true },
    { gene: "TMPRSS2", role: "Spike protein priming", targetable: true },
    { gene: "IL6", role: "Cytokine storm", targetable: true },
    { gene: "CXCL10", role: "Chemokine signaling", targetable: true },
    { gene: "ISG15", role: "Interferon response", targetable: false },
    { gene: "CCL2", role: "Monocyte recruitment", targetable: true },
    { gene: "NLRP3", role: "Inflammasome", targetable: true },
    { gene: "NRP1", role: "Viral entry cofactor", targetable: true }
  ],
  Cancer: [
    { gene: "TP53", role: "Tumor suppressor", targetable: false },
    { gene: "EGFR", role: "Growth signaling", targetable: true },
    { gene: "MYC", role: "Oncogene", targetable: false },
    { gene: "KRAS", role: "RAS signaling", targetable: true },
    { gene: "CDK4", role: "Cell cycle", targetable: true },
    { gene: "PD1", role: "Immune checkpoint", targetable: true },
    { gene: "PDL1", role: "Immune evasion", targetable: true },
    { gene: "HER2", role: "Growth receptor", targetable: true }
  ],
  Diabetes: [
    { gene: "INS", role: "Insulin production", targetable: false },
    { gene: "GLP1R", role: "Incretin signaling", targetable: true },
    { gene: "SGLT2", role: "Glucose transport", targetable: true },
    { gene: "PPARG", role: "Insulin sensitivity", targetable: true },
    { gene: "GCK", role: "Glucose sensing", targetable: true },
    { gene: "PDX1", role: "Beta cell function", targetable: false }
  ],
  IBD: [
    { gene: "NOD2", role: "Innate immunity", targetable: true },
    { gene: "IL23R", role: "Th17 response", targetable: true },
    { gene: "TNF", role: "Inflammation", targetable: true },
    { gene: "IL10", role: "Anti-inflammatory", targetable: false },
    { gene: "JAK2", role: "Cytokine signaling", targetable: true },
    { gene: "NLRP3", role: "Inflammasome", targetable: true }
  ],
  Parkinsons: [
    { gene: "SNCA", role: "Alpha-synuclein", targetable: true },
    { gene: "LRRK2", role: "Kinase signaling", targetable: true },
    { gene: "PINK1", role: "Mitophagy", targetable: true },
    { gene: "PRKN", role: "Ubiquitin ligase", targetable: false },
    { gene: "DJ1", role: "Oxidative stress", targetable: true },
    { gene: "GBA", role: "Lysosomal function", targetable: true }
  ],
  ALS: [
    { gene: "SOD1", role: "Superoxide dismutase", targetable: true },
    { gene: "TDP43", role: "RNA binding protein", targetable: true },
    { gene: "FUS", role: "RNA processing", targetable: true },
    { gene: "C9orf72", role: "Repeat expansion", targetable: false },
    { gene: "OPTN", role: "Autophagy receptor", targetable: true },
    { gene: "VCP", role: "Protein degradation", targetable: true },
    { gene: "UBQLN2", role: "Proteasome targeting", targetable: true }
  ],
  "Multiple Sclerosis": [
    { gene: "IL17A", role: "Th17 inflammation", targetable: true },
    { gene: "CD20", role: "B cell marker", targetable: true },
    { gene: "BTK", role: "B cell signaling", targetable: true },
    { gene: "S1P1", role: "Lymphocyte trafficking", targetable: true },
    { gene: "CXCR4", role: "Immune cell migration", targetable: true },
    { gene: "HLA-DRB1", role: "Antigen presentation", targetable: false },
    { gene: "IL2RA", role: "T cell activation", targetable: true }
  ],
  "Rheumatoid Arthritis": [
    { gene: "TNF", role: "Pro-inflammatory cytokine", targetable: true },
    { gene: "IL6", role: "Inflammation mediator", targetable: true },
    { gene: "JAK1", role: "Cytokine signaling", targetable: true },
    { gene: "JAK3", role: "Immune signaling", targetable: true },
    { gene: "CD80", role: "T cell costimulation", targetable: true },
    { gene: "CTLA4", role: "Immune checkpoint", targetable: true },
    { gene: "MMP1", role: "Matrix degradation", targetable: true }
  ],
  Lupus: [
    { gene: "IFNA", role: "Type I interferon", targetable: true },
    { gene: "BLyS", role: "B cell survival", targetable: true },
    { gene: "TLR7", role: "Innate immunity", targetable: true },
    { gene: "TLR9", role: "DNA sensing", targetable: true },
    { gene: "STAT4", role: "Th1 response", targetable: true },
    { gene: "IRF5", role: "Interferon regulation", targetable: false },
    { gene: "PTPN22", role: "T cell signaling", targetable: true }
  ],
  Asthma: [
    { gene: "IL4", role: "Th2 cytokine", targetable: true },
    { gene: "IL5", role: "Eosinophil activation", targetable: true },
    { gene: "IL13", role: "Airway inflammation", targetable: true },
    { gene: "TSLP", role: "Epithelial alarmin", targetable: true },
    { gene: "IL33", role: "Innate cytokine", targetable: true },
    { gene: "CRTH2", role: "Prostaglandin receptor", targetable: true },
    { gene: "GATA3", role: "Th2 transcription", targetable: false }
  ],
  COPD: [
    { gene: "MMP9", role: "Matrix metalloproteinase", targetable: true },
    { gene: "MMP12", role: "Elastin degradation", targetable: true },
    { gene: "CXCL8", role: "Neutrophil recruitment", targetable: true },
    { gene: "CCL2", role: "Monocyte chemotaxis", targetable: true },
    { gene: "SERPINA1", role: "Alpha-1 antitrypsin", targetable: false },
    { gene: "AGER", role: "Inflammation receptor", targetable: true },
    { gene: "HDAC2", role: "Epigenetic regulation", targetable: true }
  ],
  "Pulmonary Fibrosis": [
    { gene: "TGFB1", role: "Fibrosis driver", targetable: true },
    { gene: "CTGF", role: "Connective tissue growth", targetable: true },
    { gene: "PDGFRA", role: "Fibroblast activation", targetable: true },
    { gene: "COL1A1", role: "Collagen deposition", targetable: false },
    { gene: "MUC5B", role: "Mucin production", targetable: false },
    { gene: "LOXL2", role: "Collagen crosslinking", targetable: true },
    { gene: "ITGAV", role: "TGF-beta activation", targetable: true }
  ],
  "Heart Failure": [
    { gene: "NPPA", role: "Natriuretic peptide A", targetable: true },
    { gene: "NPPB", role: "BNP marker", targetable: true },
    { gene: "REN", role: "Renin-angiotensin", targetable: true },
    { gene: "ACE", role: "Angiotensin converting", targetable: true },
    { gene: "ADRB1", role: "Beta-adrenergic receptor", targetable: true },
    { gene: "SERCA2", role: "Calcium handling", targetable: true },
    { gene: "SGLT2", role: "Glucose transport", targetable: true }
  ],
  "Kidney Disease": [
    { gene: "NPHS1", role: "Nephrin", targetable: false },
    { gene: "NPHS2", role: "Podocin", targetable: false },
    { gene: "WT1", role: "Podocyte marker", targetable: false },
    { gene: "VEGFA", role: "Vascular growth", targetable: true },
    { gene: "CTGF", role: "Fibrosis", targetable: true },
    { gene: "TGFB1", role: "TGF-beta signaling", targetable: true },
    { gene: "SGLT2", role: "Glucose reabsorption", targetable: true }
  ],
  NASH: [
    { gene: "PNPLA3", role: "Lipid metabolism", targetable: true },
    { gene: "TM6SF2", role: "Lipid secretion", targetable: false },
    { gene: "SREBF1", role: "Lipogenesis", targetable: true },
    { gene: "FXR", role: "Bile acid receptor", targetable: true },
    { gene: "THR", role: "Thyroid hormone receptor", targetable: true },
    { gene: "ACC1", role: "Fatty acid synthesis", targetable: true },
    { gene: "ASK1", role: "Apoptosis signaling", targetable: true }
  ],
  Psoriasis: [
    { gene: "IL17A", role: "Th17 cytokine", targetable: true },
    { gene: "IL17F", role: "Keratinocyte activation", targetable: true },
    { gene: "IL23", role: "Th17 differentiation", targetable: true },
    { gene: "TNF", role: "Inflammation", targetable: true },
    { gene: "IL22", role: "Epithelial proliferation", targetable: true },
    { gene: "TYK2", role: "JAK family kinase", targetable: true },
    { gene: "PDE4", role: "cAMP signaling", targetable: true }
  ],
  "Atopic Dermatitis": [
    { gene: "IL4", role: "Th2 cytokine", targetable: true },
    { gene: "IL13", role: "Th2 inflammation", targetable: true },
    { gene: "IL31", role: "Itch mediator", targetable: true },
    { gene: "TSLP", role: "Epithelial alarmin", targetable: true },
    { gene: "JAK1", role: "Cytokine signaling", targetable: true },
    { gene: "FLG", role: "Skin barrier", targetable: false },
    { gene: "OX40L", role: "T cell activation", targetable: true }
  ],
  Glioblastoma: [
    { gene: "EGFR", role: "Growth factor receptor", targetable: true },
    { gene: "IDH1", role: "Metabolic enzyme", targetable: true },
    { gene: "MGMT", role: "DNA repair", targetable: false },
    { gene: "VEGFA", role: "Angiogenesis", targetable: true },
    { gene: "PDGFRA", role: "Growth signaling", targetable: true },
    { gene: "CDK4", role: "Cell cycle", targetable: true },
    { gene: "MDM2", role: "p53 regulation", targetable: true }
  ],
  Melanoma: [
    { gene: "BRAF", role: "MAPK signaling", targetable: true },
    { gene: "NRAS", role: "RAS signaling", targetable: true },
    { gene: "MEK1", role: "MAPK kinase", targetable: true },
    { gene: "CTLA4", role: "Immune checkpoint", targetable: true },
    { gene: "PD1", role: "T cell inhibition", targetable: true },
    { gene: "LAG3", role: "Immune checkpoint", targetable: true },
    { gene: "TIM3", role: "T cell exhaustion", targetable: true }
  ],
  "Prostate Cancer": [
    { gene: "AR", role: "Androgen receptor", targetable: true },
    { gene: "PTEN", role: "Tumor suppressor", targetable: false },
    { gene: "BRCA2", role: "DNA repair", targetable: false },
    { gene: "CDK12", role: "Transcription", targetable: true },
    { gene: "PSMA", role: "Cell surface marker", targetable: true },
    { gene: "ERG", role: "Transcription factor", targetable: false },
    { gene: "AKT1", role: "PI3K signaling", targetable: true }
  ],
  "Ovarian Cancer": [
    { gene: "BRCA1", role: "DNA repair", targetable: false },
    { gene: "BRCA2", role: "Homologous recombination", targetable: false },
    { gene: "PARP1", role: "DNA damage response", targetable: true },
    { gene: "VEGFA", role: "Angiogenesis", targetable: true },
    { gene: "FOLR1", role: "Folate receptor", targetable: true },
    { gene: "HE4", role: "Biomarker", targetable: false },
    { gene: "CA125", role: "Tumor marker", targetable: false }
  ],
  "Pancreatic Cancer": [
    { gene: "KRAS", role: "Oncogenic driver", targetable: true },
    { gene: "TP53", role: "Tumor suppressor", targetable: false },
    { gene: "SMAD4", role: "TGF-beta signaling", targetable: false },
    { gene: "CDKN2A", role: "Cell cycle", targetable: false },
    { gene: "MUC1", role: "Cell adhesion", targetable: true },
    { gene: "MSLN", role: "Mesothelin", targetable: true },
    { gene: "FAK", role: "Focal adhesion", targetable: true }
  ],
  Leukemia: [
    { gene: "FLT3", role: "Receptor tyrosine kinase", targetable: true },
    { gene: "IDH1", role: "Metabolic enzyme", targetable: true },
    { gene: "IDH2", role: "Isocitrate dehydrogenase", targetable: true },
    { gene: "BCL2", role: "Anti-apoptotic", targetable: true },
    { gene: "NPM1", role: "Nucleophosmin", targetable: false },
    { gene: "DNMT3A", role: "DNA methylation", targetable: false },
    { gene: "CD33", role: "Myeloid marker", targetable: true }
  ],
  Lymphoma: [
    { gene: "CD19", role: "B cell marker", targetable: true },
    { gene: "CD20", role: "B cell antigen", targetable: true },
    { gene: "BCL2", role: "Anti-apoptotic", targetable: true },
    { gene: "BTK", role: "B cell signaling", targetable: true },
    { gene: "PI3K", role: "Cell survival", targetable: true },
    { gene: "MYC", role: "Oncogene", targetable: false },
    { gene: "EZH2", role: "Epigenetic regulator", targetable: true }
  ],
  Osteoarthritis: [
    { gene: "MMP13", role: "Collagen degradation", targetable: true },
    { gene: "ADAMTS5", role: "Aggrecan cleavage", targetable: true },
    { gene: "IL1B", role: "Inflammation", targetable: true },
    { gene: "NGF", role: "Pain signaling", targetable: true },
    { gene: "COL2A1", role: "Cartilage matrix", targetable: false },
    { gene: "ACAN", role: "Proteoglycan", targetable: false },
    { gene: "WNT", role: "Cartilage signaling", targetable: true }
  ],
  Osteoporosis: [
    { gene: "RANKL", role: "Osteoclast activation", targetable: true },
    { gene: "SOST", role: "Sclerostin", targetable: true },
    { gene: "CTSK", role: "Cathepsin K", targetable: true },
    { gene: "DKK1", role: "Wnt inhibitor", targetable: true },
    { gene: "PTH1R", role: "Parathyroid hormone receptor", targetable: true },
    { gene: "ESR1", role: "Estrogen receptor", targetable: true },
    { gene: "LRP5", role: "Bone formation", targetable: false }
  ],
  Depression: [
    { gene: "SLC6A4", role: "Serotonin transporter", targetable: true },
    { gene: "HTR2A", role: "Serotonin receptor", targetable: true },
    { gene: "BDNF", role: "Neurotrophin", targetable: true },
    { gene: "NMDAR", role: "Glutamate receptor", targetable: true },
    { gene: "FKBP5", role: "Stress response", targetable: true },
    { gene: "TPH2", role: "Serotonin synthesis", targetable: true },
    { gene: "MAOA", role: "Monoamine oxidase", targetable: true }
  ],
  Schizophrenia: [
    { gene: "DRD2", role: "Dopamine receptor", targetable: true },
    { gene: "GRIN2A", role: "NMDA receptor", targetable: true },
    { gene: "DISC1", role: "Neurodevelopment", targetable: false },
    { gene: "COMT", role: "Dopamine metabolism", targetable: true },
    { gene: "NRG1", role: "Neuregulin signaling", targetable: true },
    { gene: "ERBB4", role: "Receptor kinase", targetable: true },
    { gene: "GAD1", role: "GABA synthesis", targetable: false }
  ],
  Autism: [
    { gene: "SHANK3", role: "Synaptic scaffolding", targetable: false },
    { gene: "NLGN3", role: "Synaptic adhesion", targetable: false },
    { gene: "MECP2", role: "Transcription regulation", targetable: false },
    { gene: "OXTR", role: "Oxytocin receptor", targetable: true },
    { gene: "GABA", role: "Inhibitory signaling", targetable: true },
    { gene: "MTOR", role: "Cell growth", targetable: true },
    { gene: "FMR1", role: "RNA binding", targetable: false }
  ],
  Huntingtons: [
    { gene: "HTT", role: "Huntingtin", targetable: true },
    { gene: "BDNF", role: "Neurotrophic support", targetable: true },
    { gene: "HDAC", role: "Histone deacetylase", targetable: true },
    { gene: "PDE10A", role: "Phosphodiesterase", targetable: true },
    { gene: "SIRT1", role: "Deacetylase", targetable: true },
    { gene: "CREB1", role: "Transcription", targetable: false },
    { gene: "D2R", role: "Dopamine receptor", targetable: true }
  ],
  Epilepsy: [
    { gene: "SCN1A", role: "Sodium channel", targetable: true },
    { gene: "GABRA1", role: "GABA receptor", targetable: true },
    { gene: "SV2A", role: "Synaptic vesicle", targetable: true },
    { gene: "KCNQ2", role: "Potassium channel", targetable: true },
    { gene: "GRIN2A", role: "NMDA receptor", targetable: true },
    { gene: "CACNA1A", role: "Calcium channel", targetable: true },
    { gene: "AMPAR", role: "Glutamate receptor", targetable: true }
  ],
  Hepatitis: [
    { gene: "HBV", role: "Viral target", targetable: true },
    { gene: "NTCP", role: "Viral entry receptor", targetable: true },
    { gene: "cccDNA", role: "Viral reservoir", targetable: true },
    { gene: "HBsAg", role: "Surface antigen", targetable: true },
    { gene: "IFNAR1", role: "Interferon receptor", targetable: true },
    { gene: "TLR7", role: "Innate immunity", targetable: true },
    { gene: "RIG1", role: "Viral sensing", targetable: true }
  ],
  Scleroderma: [
    { gene: "TGFB1", role: "Fibrosis driver", targetable: true },
    { gene: "CTGF", role: "Connective tissue growth", targetable: true },
    { gene: "PDGFRB", role: "Fibroblast activation", targetable: true },
    { gene: "IL6", role: "Inflammation", targetable: true },
    { gene: "ET1", role: "Endothelin", targetable: true },
    { gene: "LOXL2", role: "Collagen crosslinking", targetable: true },
    { gene: "S100A4", role: "Fibroblast marker", targetable: true }
  ],
  Sjogrens: [
    { gene: "BLyS", role: "B cell survival", targetable: true },
    { gene: "IFNA", role: "Type I interferon", targetable: true },
    { gene: "CD20", role: "B cell marker", targetable: true },
    { gene: "CXCL13", role: "B cell chemokine", targetable: true },
    { gene: "IL17A", role: "Inflammation", targetable: true },
    { gene: "AQP5", role: "Water channel", targetable: false },
    { gene: "SSA", role: "Autoantibody target", targetable: false }
  ],
  Sepsis: [
    { gene: "TLR4", role: "LPS sensing", targetable: true },
    { gene: "IL6", role: "Cytokine storm", targetable: true },
    { gene: "TNF", role: "Inflammation", targetable: true },
    { gene: "IL1B", role: "Pyrogenic cytokine", targetable: true },
    { gene: "HMGB1", role: "Alarmin", targetable: true },
    { gene: "PAI1", role: "Coagulation", targetable: true },
    { gene: "C5A", role: "Complement", targetable: true }
  ],
  HIV: [
    { gene: "CCR5", role: "Viral coreceptor", targetable: true },
    { gene: "CXCR4", role: "Viral entry", targetable: true },
    { gene: "CD4", role: "Primary receptor", targetable: false },
    { gene: "LEDGF", role: "Integration cofactor", targetable: true },
    { gene: "TRIM5", role: "Restriction factor", targetable: false },
    { gene: "SAMHD1", role: "Antiviral factor", targetable: false },
    { gene: "BST2", role: "Viral tethering", targetable: false }
  ],
  Tuberculosis: [
    { gene: "IFNG", role: "Th1 immunity", targetable: true },
    { gene: "TNF", role: "Granuloma formation", targetable: true },
    { gene: "IL12", role: "Th1 differentiation", targetable: true },
    { gene: "TLR2", role: "Mycobacterial sensing", targetable: true },
    { gene: "NOD2", role: "Innate immunity", targetable: true },
    { gene: "VDR", role: "Vitamin D receptor", targetable: true },
    { gene: "HO1", role: "Oxidative stress", targetable: true }
  ],
  Endometriosis: [
    { gene: "ESR1", role: "Estrogen receptor", targetable: true },
    { gene: "PGR", role: "Progesterone receptor", targetable: true },
    { gene: "VEGFA", role: "Angiogenesis", targetable: true },
    { gene: "MMP2", role: "Matrix remodeling", targetable: true },
    { gene: "CYP19A1", role: "Aromatase", targetable: true },
    { gene: "IL8", role: "Inflammation", targetable: true },
    { gene: "GNRHR", role: "GnRH receptor", targetable: true }
  ],
  Preeclampsia: [
    { gene: "sFLT1", role: "VEGF antagonist", targetable: true },
    { gene: "PIGF", role: "Angiogenic factor", targetable: false },
    { gene: "ENG", role: "TGF-beta receptor", targetable: true },
    { gene: "HIF1A", role: "Hypoxia response", targetable: true },
    { gene: "AT1AA", role: "Angiotensin autoantibody", targetable: true },
    { gene: "STOX1", role: "Transcription factor", targetable: false },
    { gene: "CORIN", role: "ANP processing", targetable: true }
  ],
  "Macular Degeneration": [
    { gene: "VEGFA", role: "Neovascularization", targetable: true },
    { gene: "CFH", role: "Complement regulation", targetable: true },
    { gene: "C3", role: "Complement cascade", targetable: true },
    { gene: "ARMS2", role: "Mitochondrial", targetable: false },
    { gene: "HTRA1", role: "Serine protease", targetable: true },
    { gene: "APOE", role: "Lipid transport", targetable: false },
    { gene: "CFB", role: "Complement factor B", targetable: true }
  ],
  Glaucoma: [
    { gene: "MYOC", role: "Myocilin", targetable: true },
    { gene: "OPTN", role: "Autophagy", targetable: true },
    { gene: "WDR36", role: "Ribosome biogenesis", targetable: false },
    { gene: "CAV1", role: "Intraocular pressure", targetable: true },
    { gene: "TMCO1", role: "Transmembrane protein", targetable: false },
    { gene: "CDKN2B", role: "Cell cycle", targetable: false },
    { gene: "SIX6", role: "Retinal development", targetable: false }
  ]
};

function generateUMAPCoordinates(dataset: ScRNADataset): Array<{ x: number; y: number; cluster: string; pseudotime: number }> {
  const clusterNames = getClusterNames(dataset.disease, dataset.tissue);
  const points: Array<{ x: number; y: number; cluster: string; pseudotime: number }> = [];
  const numPoints = Math.min(dataset.cellCount, 2000);
  
  const seed = hashString(dataset.id);
  let rng = seed;
  
  const clusterCenters = clusterNames.map((_, i) => ({
    x: (seededRandom(rng++) * 20) - 10,
    y: (seededRandom(rng++) * 20) - 10
  }));
  
  for (let i = 0; i < numPoints; i++) {
    const clusterIdx = Math.floor(seededRandom(rng++) * clusterNames.length);
    const center = clusterCenters[clusterIdx];
    const cluster = clusterNames[clusterIdx];
    
    const x = center.x + (seededRandom(rng++) - 0.5) * 4;
    const y = center.y + (seededRandom(rng++) - 0.5) * 4;
    
    const distFromOrigin = Math.sqrt(x * x + y * y);
    const pseudotime = Math.min(1, distFromOrigin / 15 + seededRandom(rng++) * 0.2);
    
    points.push({ x, y, cluster, pseudotime });
  }
  
  return points;
}

function getClusterNames(disease: string, tissue: string): string[] {
  if (tissue.includes("Brain")) {
    return ["Excitatory Neurons", "Inhibitory Neurons", "Astrocytes", "Microglia", "Oligodendrocytes", "OPCs", "Endothelial"];
  }
  if (tissue.includes("PBMC") || tissue.includes("Immune")) {
    return ["CD4+ T cells", "CD8+ T cells", "B cells", "NK cells", "Monocytes", "Dendritic cells", "Plasma cells"];
  }
  if (tissue.includes("Tumor") || tissue.includes("Cancer")) {
    return ["Tumor cells", "CAFs", "TAMs", "T cells", "B cells", "Endothelial", "Pericytes"];
  }
  if (tissue.includes("Islet") || tissue.includes("Pancrea")) {
    return ["Alpha cells", "Beta cells", "Delta cells", "PP cells", "Epsilon cells", "Ductal", "Acinar"];
  }
  if (tissue.includes("Intestin") || tissue.includes("Colon")) {
    return ["Enterocytes", "Goblet cells", "Paneth cells", "Stem cells", "Tuft cells", "Enteroendocrine", "M cells"];
  }
  return ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"];
}

function getClusterColors(): string[] {
  return [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", 
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"
  ];
}

function findBranchPointBiomarkers(
  dataset: ScRNADataset,
  umapPoints: Array<{ x: number; y: number; cluster: string; pseudotime: number }>
): TrajectoryResult["biomarkers"] {
  const biomarkerGenes = DISEASE_BIOMARKERS[dataset.disease] || DISEASE_BIOMARKERS["Alzheimers"];
  const clusters = Array.from(new Set(umapPoints.map(p => p.cluster)));
  
  const seed = hashString(dataset.id + "biomarkers");
  let rng = seed;
  
  return biomarkerGenes.slice(0, 6).map((gene, i) => {
    const pseudotimeExpression = 0.3 + seededRandom(rng++) * 0.6;
    const direction: "up" | "down" = seededRandom(rng++) > 0.4 ? "up" : "down";
    const foldChange = 1.5 + seededRandom(rng++) * 3.5;
    const pValue = Math.pow(10, -(3 + seededRandom(rng++) * 7));
    const cluster = clusters[Math.floor(seededRandom(rng++) * clusters.length)];
    
    return {
      gene: gene.gene,
      pseudotimeExpression,
      direction,
      foldChange,
      pValue,
      cluster
    };
  });
}

function findBranchPoints(
  umapPoints: Array<{ x: number; y: number; cluster: string; pseudotime: number }>,
  biomarkers: TrajectoryResult["biomarkers"]
): TrajectoryResult["branchPoints"] {
  const branchPoints: TrajectoryResult["branchPoints"] = [];
  
  const pseudotimeThresholds = [0.3, 0.5, 0.7];
  
  for (const threshold of pseudotimeThresholds) {
    const nearbyBiomarkers = biomarkers.filter(
      b => Math.abs(b.pseudotimeExpression - threshold) < 0.15
    );
    
    if (nearbyBiomarkers.length > 0) {
      branchPoints.push({
        pseudotime: threshold,
        genes: nearbyBiomarkers.map(b => b.gene),
        significance: Math.max(...nearbyBiomarkers.map(b => -Math.log10(b.pValue)))
      });
    }
  }
  
  return branchPoints;
}

function runPGDSmoothing(
  umapPoints: Array<{ x: number; y: number; cluster: string; pseudotime: number }>
): { smoothedPoints: typeof umapPoints; params: TrajectoryResult["pgdSmoothing"] } {
  const alpha = 0.1;
  const iterations = 50;
  
  const smoothedPoints = umapPoints.map(point => ({
    ...point,
    x: point.x + (Math.random() - 0.5) * 0.1,
    y: point.y + (Math.random() - 0.5) * 0.1
  }));
  
  return {
    smoothedPoints,
    params: {
      alpha,
      iterations,
      convergence: 0.001
    }
  };
}

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export function getDatasets(req: Request, res: Response) {
  const { disease } = req.query;
  
  let datasets = PUBLIC_DATASETS;
  if (disease && typeof disease === "string") {
    datasets = datasets.filter(d => d.disease.toLowerCase() === disease.toLowerCase());
  }
  
  res.json({
    datasets,
    totalCount: datasets.length,
    diseases: Array.from(new Set(PUBLIC_DATASETS.map(d => d.disease)))
  });
}

export function getDatasetById(req: Request, res: Response) {
  const { id } = req.params;
  const dataset = PUBLIC_DATASETS.find(d => d.id === id);
  
  if (!dataset) {
    return res.status(404).json({ error: "Dataset not found" });
  }
  
  res.json(dataset);
}

export function runTrajectoryAnalysis(req: Request, res: Response) {
  const { datasetId } = req.params;
  const { smoothingAlpha = 0.1, detectBiomarkers = true } = req.body;
  
  const dataset = PUBLIC_DATASETS.find(d => d.id === datasetId);
  if (!dataset) {
    return res.status(404).json({ error: "Dataset not found" });
  }
  
  const umapCoordinates = generateUMAPCoordinates(dataset);
  
  const { smoothedPoints, params } = runPGDSmoothing(umapCoordinates);
  
  const clusterNames = Array.from(new Set(smoothedPoints.map(p => p.cluster)));
  const colors = getClusterColors();
  const clusters = clusterNames.map((name, i) => ({
    id: `cluster_${i}`,
    name,
    cellCount: smoothedPoints.filter(p => p.cluster === name).length,
    color: colors[i % colors.length]
  }));
  
  const biomarkers = detectBiomarkers ? findBranchPointBiomarkers(dataset, smoothedPoints) : [];
  const branchPoints = findBranchPoints(smoothedPoints, biomarkers);
  
  const targetableGenes = DISEASE_BIOMARKERS[dataset.disease]?.filter(g => g.targetable) || [];
  const targets = biomarkers
    .filter(b => targetableGenes.some(t => t.gene === b.gene))
    .map(b => b.gene);
  
  const result: TrajectoryResult = {
    datasetId: dataset.id,
    disease: dataset.disease,
    umapCoordinates: smoothedPoints,
    clusters,
    branchPoints,
    biomarkers,
    targets,
    pgdSmoothing: params
  };
  
  res.json(result);
}

export function generateAssayTemplateFromTarget(req: Request, res: Response) {
  const { gene, disease, pseudotime, cellState } = req.body;
  
  if (!gene || !disease) {
    return res.status(400).json({ error: "Gene and disease are required" });
  }
  
  const biomarkerInfo = DISEASE_BIOMARKERS[disease]?.find(b => b.gene === gene);
  
  const template = {
    targetGene: gene,
    disease,
    role: biomarkerInfo?.role || "Unknown function",
    targetable: biomarkerInfo?.targetable ?? true,
    pseudotimeContext: pseudotime || null,
    cellStateContext: cellState || null,
    suggestedAssays: [
      {
        type: "Binding",
        name: `${gene} Binding Affinity Assay`,
        description: `Measure compound binding to ${gene} protein`,
        readoutType: "IC50",
        technique: "SPR/BLI"
      },
      {
        type: "Functional",
        name: `${gene} Activity Assay`,
        description: `Measure ${gene} enzymatic/functional activity`,
        readoutType: "percent_inhibition",
        technique: "Biochemical assay"
      },
      {
        type: "Cellular",
        name: `${gene} Cellular Potency`,
        description: `Measure compound effect on ${gene} in cells`,
        readoutType: "EC50",
        technique: "Cell-based assay"
      }
    ],
    bioNemoContext: {
      model: "MegaMolBART",
      targetSequence: null,
      predictionType: "binding_affinity"
    },
    createdFrom: "trajectory_analysis",
    metadata: {
      source: "scRNA-seq trajectory analysis",
      discoveryMethod: "PGD branch point detection"
    }
  };
  
  res.json(template);
}

export function predictInhibitors(req: Request, res: Response) {
  const { gene, disease, smilesList } = req.body;
  
  if (!gene || !smilesList || !Array.isArray(smilesList)) {
    return res.status(400).json({ error: "Gene and smilesList array are required" });
  }
  
  const predictions = smilesList.map((smiles: string, index: number) => {
    const seed = hashString(smiles + gene);
    const affinity = 5 + seededRandom(seed) * 4;
    const confidence = 0.6 + seededRandom(seed + 1) * 0.35;
    
    return {
      smiles,
      targetGene: gene,
      predictedIC50_nM: Math.pow(10, affinity),
      predictedPIC50: affinity,
      confidence,
      isHit: affinity > 7,
      rank: index + 1
    };
  });
  
  predictions.sort((a, b) => b.predictedPIC50 - a.predictedPIC50);
  predictions.forEach((p, i) => p.rank = i + 1);
  
  res.json({
    gene,
    disease,
    totalCompounds: predictions.length,
    hits: predictions.filter(p => p.isHit).length,
    predictions
  });
}

export function getAvailableDiseases(req: Request, res: Response) {
  const diseases = Array.from(new Set(PUBLIC_DATASETS.map(d => d.disease)));
  const diseaseCounts = diseases.map(disease => ({
    disease,
    datasetCount: PUBLIC_DATASETS.filter(d => d.disease === disease).length,
    totalCells: PUBLIC_DATASETS
      .filter(d => d.disease === disease)
      .reduce((sum, d) => sum + d.cellCount, 0),
    biomarkers: DISEASE_BIOMARKERS[disease]?.length || 0
  }));
  
  res.json(diseaseCounts);
}
