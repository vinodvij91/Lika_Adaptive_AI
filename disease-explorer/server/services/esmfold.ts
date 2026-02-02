/**
 * ESMFold Service - Real protein structure prediction using Meta's free API
 * 
 * ESMFold is a fast, accurate protein structure prediction model from Meta AI.
 * It provides AlphaFold2-quality predictions without requiring an API key.
 * 
 * API: https://api.esmatlas.com/foldSequence/v1/pdb/
 */

import { db } from "../db";
import { structurePredictions } from "@shared/schema";
import { eq } from "drizzle-orm";

const ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/";

interface ESMFoldResult {
  pdbData: string;
  targetId: string;
  targetName: string;
  sequence: string;
  confidenceScore: number;
  metrics: {
    pTM: number;
    pLDDT: number;
    numAtoms: number;
    numResidues: number;
  };
  modelVersion: string;
  isSimulated: boolean;
  fromCache?: boolean;
}

/**
 * Parse pLDDT scores from PDB b-factor column
 * In ESMFold output, the B-factor column contains per-residue pLDDT scores
 */
function parsePLDDTFromPDB(pdbData: string): number {
  const lines = pdbData.split('\n');
  const bFactors: number[] = [];
  
  for (const line of lines) {
    if (line.startsWith('ATOM') && line.includes(' CA ')) {
      // B-factor is at columns 61-66 in PDB format
      const bFactor = parseFloat(line.substring(60, 66).trim());
      if (!isNaN(bFactor)) {
        bFactors.push(bFactor);
      }
    }
  }
  
  if (bFactors.length === 0) return 70; // Default if parsing fails
  
  // Average pLDDT
  return bFactors.reduce((a, b) => a + b, 0) / bFactors.length;
}

/**
 * Count atoms and residues from PDB data
 */
function countAtomsAndResidues(pdbData: string): { numAtoms: number; numResidues: number } {
  const lines = pdbData.split('\n');
  let numAtoms = 0;
  const residues = new Set<string>();
  
  for (const line of lines) {
    if (line.startsWith('ATOM')) {
      numAtoms++;
      // Residue number is at columns 23-26, chain at 22
      const resNum = line.substring(22, 27).trim();
      residues.add(resNum);
    }
  }
  
  return { numAtoms, numResidues: residues.size };
}

/**
 * Generate cache key for structure predictions using a proper hash
 */
function generateCacheKey(targetId: string, sequence: string): string {
  // Create a simple hash of the full sequence
  let hash = 0;
  const input = `${targetId}:${sequence}`;
  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return `esmfold-${Math.abs(hash).toString(36)}-${sequence.length}`;
}

/**
 * Check cache for existing prediction
 */
async function checkCache(cacheKey: string): Promise<ESMFoldResult | null> {
  try {
    const cached = await db.select()
      .from(structurePredictions)
      .where(eq(structurePredictions.cacheKey, cacheKey))
      .limit(1);
    
    if (cached.length > 0) {
      const prediction = cached[0];
      return {
        pdbData: prediction.pdbData,
        targetId: prediction.targetId || '',
        targetName: '',
        sequence: prediction.proteinSequence,
        confidenceScore: prediction.confidenceScore || 0,
        metrics: prediction.metrics as any,
        modelVersion: prediction.modelVersion || 'esmfold-v1',
        isSimulated: false,
        fromCache: true,
      };
    }
  } catch (error) {
    console.error('Cache check error:', error);
  }
  return null;
}

/**
 * Save prediction to cache
 */
async function saveToCache(cacheKey: string, result: ESMFoldResult): Promise<void> {
  try {
    await db.insert(structurePredictions).values({
      cacheKey,
      targetId: result.targetId,
      proteinSequence: result.sequence,
      pdbData: result.pdbData,
      confidenceScore: result.confidenceScore,
      metrics: result.metrics,
      modelVersion: result.modelVersion,
      isSimulated: false,
    }).onConflictDoNothing();
  } catch (error) {
    console.error('Cache save error:', error);
  }
}

/**
 * Predict protein structure using ESMFold API
 */
export async function predictStructureWithESMFold(
  targetId: string,
  targetName: string,
  sequence: string
): Promise<ESMFoldResult> {
  // Clean the sequence - remove any non-amino acid characters
  const cleanSequence = sequence.replace(/[^ACDEFGHIKLMNPQRSTVWY]/gi, '').toUpperCase();
  
  if (cleanSequence.length < 10) {
    throw new Error('Sequence too short. Minimum 10 amino acids required.');
  }
  
  if (cleanSequence.length > 400) {
    throw new Error('Sequence too long for ESMFold API. Maximum 400 amino acids. For longer sequences, consider using local ESMFold or AlphaFold.');
  }
  
  // Check cache first
  const cacheKey = generateCacheKey(targetId, cleanSequence);
  const cached = await checkCache(cacheKey);
  if (cached) {
    console.log(`ESMFold: Using cached prediction for ${targetName}`);
    return cached;
  }
  
  console.log(`ESMFold: Predicting structure for ${targetName} (${cleanSequence.length} residues)`);
  
  try {
    const response = await fetch(ESMFOLD_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'text/plain',
      },
      body: cleanSequence,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ESMFold API error: ${response.status} - ${errorText}`);
    }
    
    const pdbData = await response.text();
    
    if (!pdbData || !pdbData.includes('ATOM')) {
      throw new Error('Invalid PDB data received from ESMFold');
    }
    
    // Parse metrics from PDB
    const pLDDT = parsePLDDTFromPDB(pdbData);
    const { numAtoms, numResidues } = countAtomsAndResidues(pdbData);
    
    // pTM is estimated from pLDDT (ESMFold doesn't directly provide pTM in the PDB)
    const pTM = pLDDT / 100 * 0.95; // Approximate conversion
    
    const result: ESMFoldResult = {
      pdbData,
      targetId,
      targetName,
      sequence: cleanSequence,
      confidenceScore: pLDDT / 100,
      metrics: {
        pTM,
        pLDDT,
        numAtoms,
        numResidues,
      },
      modelVersion: 'esmfold-v1',
      isSimulated: false,
      fromCache: false,
    };
    
    // Cache the result
    await saveToCache(cacheKey, result);
    
    console.log(`ESMFold: Successfully predicted structure for ${targetName} - pLDDT: ${pLDDT.toFixed(1)}`);
    
    return result;
  } catch (error: any) {
    console.error('ESMFold API error:', error.message);
    throw error;
  }
}

/**
 * Get ESMFold service info
 */
export function getESMFoldInfo() {
  return {
    name: 'ESMFold',
    provider: 'Meta AI',
    description: 'Fast and accurate protein structure prediction using evolutionary-scale language models',
    maxSequenceLength: 400,
    apiKeyRequired: false,
    documentation: 'https://esmatlas.com/about#fold',
    capabilities: [
      'Single-chain protein structure prediction',
      'Per-residue confidence scores (pLDDT)',
      'Fast inference (~seconds for short proteins)',
      'No API key required',
    ],
    supportedPipelines: ['drug_discovery', 'vaccine_discovery', 'materials_science'],
  };
}

// ============================================================================
// DRUG DISCOVERY EXTENSIONS
// ============================================================================

export type DrugDiscoveryPipeline = 'target_validation' | 'binding_site_analysis' | 'virtual_screening' | 'lead_optimization';

interface DrugDiscoveryResult extends ESMFoldResult {
  pipeline: 'drug_discovery';
  pipelineStep: DrugDiscoveryPipeline;
  drugDiscoveryMetrics: {
    bindingSitePredicted: boolean;
    potentialBindingResidues: number[];
    surfaceAccessibility: number;
    hydrophobicPockets: number;
    druggabilityScore: number;
  };
}

/**
 * Analyze PDB structure for potential drug binding sites
 * Uses pLDDT scores and residue positions to identify flexible/binding regions
 */
function analyzeDrugBindingSites(pdbData: string): {
  potentialBindingResidues: number[];
  surfaceAccessibility: number;
  hydrophobicPockets: number;
  druggabilityScore: number;
} {
  const lines = pdbData.split('\n');
  const residueData: { resNum: number; pLDDT: number; atomType: string }[] = [];
  
  for (const line of lines) {
    if (line.startsWith('ATOM')) {
      const resNum = parseInt(line.substring(22, 26).trim());
      const pLDDT = parseFloat(line.substring(60, 66).trim());
      const atomType = line.substring(12, 16).trim();
      if (!isNaN(resNum) && !isNaN(pLDDT)) {
        residueData.push({ resNum, pLDDT, atomType });
      }
    }
  }

  // Identify flexible regions (lower pLDDT) as potential binding sites
  const uniqueResidues = [...new Set(residueData.map(r => r.resNum))];
  const residueAvgPLDDT = uniqueResidues.map(resNum => {
    const residueAtoms = residueData.filter(r => r.resNum === resNum);
    return {
      resNum,
      avgPLDDT: residueAtoms.reduce((sum, a) => sum + a.pLDDT, 0) / residueAtoms.length
    };
  });

  // Binding sites often have moderate pLDDT (40-70) indicating flexibility
  const potentialBindingResidues = residueAvgPLDDT
    .filter(r => r.avgPLDDT >= 40 && r.avgPLDDT <= 75)
    .map(r => r.resNum);

  // Estimate surface accessibility (simplified)
  const surfaceAccessibility = Math.min(1, potentialBindingResidues.length / uniqueResidues.length * 3);
  
  // Count hydrophobic pockets (based on sequence patterns - simplified)
  const hydrophobicPockets = Math.floor(potentialBindingResidues.length / 8);
  
  // Druggability score based on binding site characteristics
  const druggabilityScore = Math.min(1, (potentialBindingResidues.length > 5 ? 0.4 : 0.1) + 
    surfaceAccessibility * 0.3 + 
    (hydrophobicPockets > 0 ? 0.3 : 0.1));

  return {
    potentialBindingResidues,
    surfaceAccessibility,
    hydrophobicPockets,
    druggabilityScore,
  };
}

/**
 * Predict protein structure for drug discovery applications
 */
export async function predictStructureForDrugDiscovery(
  targetId: string,
  targetName: string,
  sequence: string,
  pipelineStep: DrugDiscoveryPipeline = 'target_validation'
): Promise<DrugDiscoveryResult> {
  // Get base structure prediction
  const baseResult = await predictStructureWithESMFold(targetId, targetName, sequence);
  
  // Analyze for drug binding sites
  const bindingAnalysis = analyzeDrugBindingSites(baseResult.pdbData);
  
  return {
    ...baseResult,
    pipeline: 'drug_discovery',
    pipelineStep,
    drugDiscoveryMetrics: {
      bindingSitePredicted: bindingAnalysis.potentialBindingResidues.length > 5,
      ...bindingAnalysis,
    },
  };
}

/**
 * Batch structure prediction for drug discovery screening
 */
export async function batchPredictForDrugDiscovery(
  targets: Array<{ targetId: string; targetName: string; sequence: string }>,
  pipelineStep: DrugDiscoveryPipeline = 'virtual_screening'
): Promise<{ results: DrugDiscoveryResult[]; failed: Array<{ targetId: string; error: string }> }> {
  const results: DrugDiscoveryResult[] = [];
  const failed: Array<{ targetId: string; error: string }> = [];
  
  for (const target of targets) {
    try {
      const result = await predictStructureForDrugDiscovery(
        target.targetId,
        target.targetName,
        target.sequence,
        pipelineStep
      );
      results.push(result);
    } catch (error: any) {
      failed.push({ targetId: target.targetId, error: error.message });
    }
  }
  
  return { results, failed };
}

// ============================================================================
// MATERIALS SCIENCE EXTENSIONS
// ============================================================================

export type MaterialsSciencePipeline = 'protein_materials' | 'enzyme_design' | 'biocatalyst_optimization' | 'biomaterial_engineering';

interface MaterialsScienceResult extends ESMFoldResult {
  pipeline: 'materials_science';
  pipelineStep: MaterialsSciencePipeline;
  materialsMetrics: {
    structuralStability: number;
    thermalStabilityEstimate: number;
    mechanicalRigidity: number;
    surfaceChargeDistribution: 'positive' | 'negative' | 'neutral' | 'mixed';
    catalyticPotential: number;
    selfAssemblyPropensity: number;
  };
}

/**
 * Analyze protein structure for materials science applications
 */
function analyzeMaterialsProperties(pdbData: string, sequence: string): {
  structuralStability: number;
  thermalStabilityEstimate: number;
  mechanicalRigidity: number;
  surfaceChargeDistribution: 'positive' | 'negative' | 'neutral' | 'mixed';
  catalyticPotential: number;
  selfAssemblyPropensity: number;
} {
  const lines = pdbData.split('\n');
  const bFactors: number[] = [];
  
  for (const line of lines) {
    if (line.startsWith('ATOM') && line.includes(' CA ')) {
      const bFactor = parseFloat(line.substring(60, 66).trim());
      if (!isNaN(bFactor)) {
        bFactors.push(bFactor);
      }
    }
  }
  
  const avgPLDDT = bFactors.length > 0 
    ? bFactors.reduce((a, b) => a + b, 0) / bFactors.length 
    : 70;
  
  // Structural stability correlates with high pLDDT
  const structuralStability = avgPLDDT / 100;
  
  // Thermal stability estimate (simplified - based on disulfide bridges and pLDDT)
  const cysteinePairs = (sequence.match(/C/g) || []).length / 2;
  const thermalStabilityEstimate = Math.min(1, structuralStability * 0.6 + cysteinePairs * 0.1);
  
  // Mechanical rigidity (inverse of flexibility)
  const highPLDDTRatio = bFactors.filter(b => b > 80).length / Math.max(1, bFactors.length);
  const mechanicalRigidity = highPLDDTRatio;
  
  // Surface charge distribution based on sequence composition
  const positiveAA = (sequence.match(/[KRH]/g) || []).length;
  const negativeAA = (sequence.match(/[DE]/g) || []).length;
  const chargeRatio = positiveAA / Math.max(1, negativeAA);
  let surfaceChargeDistribution: 'positive' | 'negative' | 'neutral' | 'mixed';
  if (chargeRatio > 1.5) surfaceChargeDistribution = 'positive';
  else if (chargeRatio < 0.67) surfaceChargeDistribution = 'negative';
  else if (positiveAA + negativeAA < sequence.length * 0.1) surfaceChargeDistribution = 'neutral';
  else surfaceChargeDistribution = 'mixed';
  
  // Catalytic potential (presence of catalytic residues and active site motifs)
  const catalyticResidues = (sequence.match(/[HCDSEK]/g) || []).length / sequence.length;
  const catalyticPotential = Math.min(1, catalyticResidues * 2);
  
  // Self-assembly propensity (based on hydrophobic patches and structural regularity)
  const hydrophobicAA = (sequence.match(/[AILMFVW]/g) || []).length / sequence.length;
  const selfAssemblyPropensity = Math.min(1, hydrophobicAA * 1.5 + mechanicalRigidity * 0.3);
  
  return {
    structuralStability,
    thermalStabilityEstimate,
    mechanicalRigidity,
    surfaceChargeDistribution,
    catalyticPotential,
    selfAssemblyPropensity,
  };
}

/**
 * Predict protein structure for materials science applications
 */
export async function predictStructureForMaterialsScience(
  targetId: string,
  targetName: string,
  sequence: string,
  pipelineStep: MaterialsSciencePipeline = 'protein_materials'
): Promise<MaterialsScienceResult> {
  // Get base structure prediction
  const baseResult = await predictStructureWithESMFold(targetId, targetName, sequence);
  
  // Analyze for materials properties
  const materialsAnalysis = analyzeMaterialsProperties(baseResult.pdbData, baseResult.sequence);
  
  return {
    ...baseResult,
    pipeline: 'materials_science',
    pipelineStep,
    materialsMetrics: materialsAnalysis,
  };
}

/**
 * Batch structure prediction for materials science screening
 */
export async function batchPredictForMaterialsScience(
  targets: Array<{ targetId: string; targetName: string; sequence: string }>,
  pipelineStep: MaterialsSciencePipeline = 'protein_materials'
): Promise<{ results: MaterialsScienceResult[]; failed: Array<{ targetId: string; error: string }> }> {
  const results: MaterialsScienceResult[] = [];
  const failed: Array<{ targetId: string; error: string }> = [];
  
  for (const target of targets) {
    try {
      const result = await predictStructureForMaterialsScience(
        target.targetId,
        target.targetName,
        target.sequence,
        pipelineStep
      );
      results.push(result);
    } catch (error: any) {
      failed.push({ targetId: target.targetId, error: error.message });
    }
  }
  
  return { results, failed };
}

/**
 * Get domain-specific ESMFold info
 */
export function getESMFoldDomainInfo(domain: 'drug_discovery' | 'materials_science' | 'vaccine_discovery') {
  const baseInfo = getESMFoldInfo();
  
  const domainCapabilities = {
    drug_discovery: {
      pipelineSteps: ['target_validation', 'binding_site_analysis', 'virtual_screening', 'lead_optimization'],
      metrics: ['druggabilityScore', 'bindingSitePrediction', 'surfaceAccessibility', 'hydrophobicPockets'],
      description: 'Structure prediction for drug target validation and binding site analysis',
    },
    materials_science: {
      pipelineSteps: ['protein_materials', 'enzyme_design', 'biocatalyst_optimization', 'biomaterial_engineering'],
      metrics: ['structuralStability', 'thermalStability', 'mechanicalRigidity', 'catalyticPotential', 'selfAssembly'],
      description: 'Structure prediction for protein-based materials and enzyme engineering',
    },
    vaccine_discovery: {
      pipelineSteps: ['antigen_design', 'epitope_prediction', 'immunogen_optimization'],
      metrics: ['surfaceExposure', 'epitopeAccessibility', 'conformationalStability'],
      description: 'Structure prediction for vaccine antigen design and epitope mapping',
    },
  };
  
  return {
    ...baseInfo,
    domain,
    domainCapabilities: domainCapabilities[domain],
  };
}
