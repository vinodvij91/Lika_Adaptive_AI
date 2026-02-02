import { classifyAssay, type AssayCategory } from "./assay-classifier";

export interface ChEMBLAssay {
  chemblId: string;
  name: string;
  description: string;
  assayType: string;
  targetChemblId?: string;
  targetName?: string;
  organism?: string;
  standardType?: string;
  category: AssayCategory;
  confidence: number;
}

interface ChEMBLAssayResponse {
  assays?: Array<{
    assay_chembl_id: string;
    assay_type: string;
    description: string;
    assay_organism?: string;
    target_chembl_id?: string;
  }>;
}

interface ChEMBLTargetSearchResponse {
  targets?: Array<{
    target_chembl_id: string;
    pref_name: string;
    organism: string;
    target_type: string;
  }>;
}

const CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data";

export async function searchChEMBLTarget(targetName: string): Promise<string | null> {
  try {
    const searchUrl = `${CHEMBL_BASE_URL}/target/search.json?q=${encodeURIComponent(targetName)}&limit=1`;
    const res = await fetch(searchUrl);
    
    if (!res.ok) return null;
    
    const data = await res.json() as ChEMBLTargetSearchResponse;
    return data.targets?.[0]?.target_chembl_id || null;
  } catch (error) {
    console.error("ChEMBL target search error:", error);
    return null;
  }
}

export async function harvestChEMBLAssays(targetChemblId: string, limit: number = 20): Promise<ChEMBLAssay[]> {
  try {
    const assayUrl = `${CHEMBL_BASE_URL}/assay.json?target_chembl_id=${targetChemblId}&limit=${limit}`;
    const res = await fetch(assayUrl);
    
    if (!res.ok) {
      console.log(`ChEMBL assay fetch for ${targetChemblId} returned ${res.status}`);
      return [];
    }
    
    const data = await res.json() as ChEMBLAssayResponse;
    const assays: ChEMBLAssay[] = [];
    
    for (const assay of data.assays || []) {
      const description = assay.description || "";
      const classification = classifyAssay(description, assay.assay_chembl_id);
      
      assays.push({
        chemblId: assay.assay_chembl_id,
        name: assay.assay_chembl_id,
        description: description.slice(0, 500),
        assayType: assay.assay_type,
        targetChemblId: assay.target_chembl_id,
        organism: assay.assay_organism,
        category: classification.category,
        confidence: classification.confidence
      });
    }
    
    return assays;
  } catch (error) {
    console.error("ChEMBL harvest error:", error);
    return [];
  }
}

export async function searchChEMBLAssays(targetName: string, limit: number = 20): Promise<ChEMBLAssay[]> {
  const targetChemblId = await searchChEMBLTarget(targetName);
  if (!targetChemblId) {
    console.log(`No ChEMBL target found for ${targetName}`);
    return [];
  }
  
  return harvestChEMBLAssays(targetChemblId, limit);
}
