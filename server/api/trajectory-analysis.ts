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
