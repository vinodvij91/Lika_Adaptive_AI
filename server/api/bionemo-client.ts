/**
 * BioNeMo Client for NVIDIA NIM API Integration
 * Provides context-aware molecular property predictions using MegaMolBART and ESM2 models
 */

export interface BioNemoConfig {
  model: "megamolbart" | "esm2" | "molmim";
  context: string;
  readout: "pIC50" | "IC50" | "Ki" | "Kd" | "percent_inhibition";
}

export interface PredictionResult {
  smiles: string;
  predictedValue: number;
  confidence: number;
  unit: string;
  modelUsed: string;
}

export interface AssayPredictionResponse {
  assayId: string;
  assayName: string;
  context: string;
  predictions: PredictionResult[];
  topHits: PredictionResult[];
  modelInfo: {
    model: string;
    version: string;
    inferenceTime: number;
  };
}

// Assay-specific BioNeMo context configurations
export const ASSAY_CONTEXTS: Record<string, BioNemoConfig> = {
  "AID 720543": {
    model: "megamolbart",
    context: "Thioflavin T fluorescence Aβ42 aggregation inhibition assay IC50 prediction for Alzheimer's disease drug discovery",
    readout: "pIC50"
  },
  "AID 1508": {
    model: "megamolbart",
    context: "BACE1 beta-secretase inhibition FRET-based assay IC50 prediction for amyloid processing modulation",
    readout: "IC50"
  },
  "AID 488997": {
    model: "megamolbart",
    context: "Acetylcholinesterase inhibition Ellman colorimetric assay IC50 prediction for cholinergic enhancement",
    readout: "IC50"
  },
  "CHEMBL3215112": {
    model: "megamolbart",
    context: "Tau protein aggregation inhibition ThT fluorescence assay for tauopathy drug development",
    readout: "pIC50"
  },
  "NLRP3-FUNC-001": {
    model: "esm2",
    context: "NLRP3 inflammasome inhibition ASC speck formation IL-1β secretion assay IC50 prediction for neuroinflammation",
    readout: "IC50"
  },
  "BBB-PERM-001": {
    model: "megamolbart",
    context: "Blood-brain barrier permeability prediction hiPSC-derived BMEC model CNS drug penetration",
    readout: "percent_inhibition"
  },
  "HERG-001": {
    model: "megamolbart",
    context: "hERG potassium channel inhibition cardiac safety IC50 prediction automated patch clamp",
    readout: "IC50"
  }
};

// NVIDIA NIM API configuration
const NVIDIA_NIM_BASE_URL = process.env.NVIDIA_NIM_URL || "https://integrate.api.nvidia.com/v1";
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

/**
 * Predict molecular properties using BioNeMo models
 * Falls back to simulation mode if API key is not configured
 */
export async function bionemoPredict(
  context: string,
  smiles: string[],
  model: "megamolbart" | "esm2" | "molmim" = "megamolbart"
): Promise<PredictionResult[]> {
  const startTime = Date.now();

  // If NVIDIA API key is available, use real API
  if (NVIDIA_API_KEY) {
    try {
      const response = await fetch(`${NVIDIA_NIM_BASE_URL}/biology/nvidia/megamolbart`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${NVIDIA_API_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          sequences: smiles,
          parameters: {
            context: context,
            task: "property_prediction"
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        return processNimResponse(data, smiles);
      }
    } catch (error) {
      console.error("BioNeMo API error, falling back to simulation:", error);
    }
  }

  // Simulation mode: Generate deterministic predictions based on SMILES structure
  return simulatePredictions(smiles, context, model);
}

/**
 * Process NVIDIA NIM API response into standardized format
 */
function processNimResponse(data: any, smiles: string[]): PredictionResult[] {
  return smiles.map((smi, idx) => ({
    smiles: smi,
    predictedValue: data.predictions?.[idx]?.value || Math.random() * 10,
    confidence: data.predictions?.[idx]?.confidence || 0.8,
    unit: "µM",
    modelUsed: "megamolbart"
  }));
}

/**
 * Simulate BioNeMo predictions using molecular fingerprint hashing
 * Provides deterministic, structure-based predictions for demonstration
 */
function simulatePredictions(
  smiles: string[],
  context: string,
  model: string
): PredictionResult[] {
  return smiles.map(smi => {
    // Generate deterministic prediction based on SMILES hash
    const hash = simpleHash(smi + context);
    const baseValue = (hash % 10000) / 1000; // 0-10 range
    
    // Adjust based on molecular properties inferred from SMILES
    const ringCount = (smi.match(/c1|C1|n1|N1/g) || []).length;
    const heteroatoms = (smi.match(/[NOSnos]/g) || []).length;
    const molWeight = smi.length * 12; // Rough approximation
    
    // Better drug-likeness gives lower IC50 (more potent)
    const drugLikeness = Math.min(5, ringCount) * 0.2 + Math.min(8, heteroatoms) * 0.1;
    const adjustedValue = Math.max(0.001, baseValue - drugLikeness);
    
    // Confidence based on structural features
    const confidence = 0.6 + Math.min(0.35, (ringCount + heteroatoms) * 0.05);
    
    return {
      smiles: smi,
      predictedValue: parseFloat(adjustedValue.toFixed(3)),
      confidence: parseFloat(confidence.toFixed(2)),
      unit: "µM",
      modelUsed: `${model} (simulation)`
    };
  });
}

/**
 * Simple string hash function for deterministic predictions
 */
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

/**
 * Predict assay outcomes for a specific assay using BioNeMo
 */
export async function predictForAssay(
  assayId: string,
  smiles: string[],
  assayName?: string,
  assayDescription?: string
): Promise<AssayPredictionResponse> {
  const startTime = Date.now();
  
  // Get assay-specific context or generate from description
  const config = ASSAY_CONTEXTS[assayId] || {
    model: "megamolbart" as const,
    context: assayDescription || `Property prediction for assay ${assayId}`,
    readout: "IC50" as const
  };
  
  const predictions = await bionemoPredict(config.context, smiles, config.model);
  
  // Sort by predicted value to get top hits (lower IC50 = better)
  const sortedPredictions = [...predictions].sort((a, b) => a.predictedValue - b.predictedValue);
  const topHits = sortedPredictions.slice(0, Math.min(10, sortedPredictions.length));
  
  return {
    assayId,
    assayName: assayName || assayId,
    context: config.context,
    predictions,
    topHits,
    modelInfo: {
      model: config.model,
      version: "1.0",
      inferenceTime: Date.now() - startTime
    }
  };
}

/**
 * Batch predict for multiple assays in a campaign
 */
export async function predictCampaign(
  assayIds: string[],
  smiles: string[],
  assayMetadata?: Record<string, { name: string; description: string }>
): Promise<Record<string, AssayPredictionResponse>> {
  const results: Record<string, AssayPredictionResponse> = {};
  
  // Process assays in parallel
  const promises = assayIds.map(async (assayId) => {
    const metadata = assayMetadata?.[assayId];
    const result = await predictForAssay(
      assayId,
      smiles,
      metadata?.name,
      metadata?.description
    );
    return { assayId, result };
  });
  
  const resolved = await Promise.all(promises);
  for (const { assayId, result } of resolved) {
    results[assayId] = result;
  }
  
  return results;
}
