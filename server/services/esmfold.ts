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
  };
}
