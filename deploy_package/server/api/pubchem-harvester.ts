import { classifyAssay, type AssayCategory } from "./assay-classifier";

export interface PubChemAssay {
  aid: number;
  name: string;
  description: string;
  sourceId: string;
  sourceName: string;
  targetGeneId?: number;
  targetName?: string;
  activityOutcome?: string;
  category: AssayCategory;
  confidence: number;
}

interface PubChemBioAssayResponse {
  Columns: { Column: Array<{ Heading: string }> };
  Row?: Array<{ Cell: Array<{ value?: string | number }> }>;
}

const PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug";

export async function searchPubChemAssays(targetName: string, limit: number = 20): Promise<PubChemAssay[]> {
  try {
    const searchUrl = `${PUBCHEM_BASE_URL}/assay/target/genesymbol/${encodeURIComponent(targetName)}/aids/JSON`;
    const searchRes = await fetch(searchUrl);
    
    if (!searchRes.ok) {
      console.log(`PubChem search for ${targetName} returned ${searchRes.status}`);
      return [];
    }
    
    const searchData = await searchRes.json() as { IdentifierList?: { AID?: number[] } };
    const aids = searchData.IdentifierList?.AID?.slice(0, limit) || [];
    
    if (aids.length === 0) {
      return [];
    }

    const assays: PubChemAssay[] = [];
    
    // Fetch up to the requested limit (capped at 20 for API rate limiting)
    for (const aid of aids.slice(0, Math.min(limit, 20))) {
      try {
        const detailUrl = `${PUBCHEM_BASE_URL}/assay/aid/${aid}/description/JSON`;
        const detailRes = await fetch(detailUrl);
        
        if (!detailRes.ok) continue;
        
        const detailData = await detailRes.json() as { PC_AssayContainer?: Array<{ assay?: { descr?: { name?: string; description?: Array<{ value?: string }>; aid?: { id?: number }; target?: Array<{ name?: string; gi?: number }> } } }> };
        const assayDescr = detailData.PC_AssayContainer?.[0]?.assay?.descr;
        
        if (!assayDescr) continue;
        
        const name = assayDescr.name || `AID ${aid}`;
        const description = assayDescr.description?.map(d => d.value).join(" ") || "";
        const classification = classifyAssay(description, name);
        
        assays.push({
          aid,
          name,
          description: description.slice(0, 500),
          sourceId: `AID ${aid}`,
          sourceName: "PubChem",
          targetName: assayDescr.target?.[0]?.name || targetName,
          targetGeneId: assayDescr.target?.[0]?.gi,
          category: classification.category,
          confidence: classification.confidence
        });
      } catch (err) {
        console.error(`Error fetching AID ${aid}:`, err);
      }
    }
    
    return assays;
  } catch (error) {
    console.error("PubChem harvest error:", error);
    return [];
  }
}

export async function getAssayById(aid: number): Promise<PubChemAssay | null> {
  try {
    const detailUrl = `${PUBCHEM_BASE_URL}/assay/aid/${aid}/description/JSON`;
    const detailRes = await fetch(detailUrl);
    
    if (!detailRes.ok) return null;
    
    const detailData = await detailRes.json() as { PC_AssayContainer?: Array<{ assay?: { descr?: { name?: string; description?: Array<{ value?: string }>; target?: Array<{ name?: string; gi?: number }> } } }> };
    const assayDescr = detailData.PC_AssayContainer?.[0]?.assay?.descr;
    
    if (!assayDescr) return null;
    
    const name = assayDescr.name || `AID ${aid}`;
    const description = assayDescr.description?.map(d => d.value).join(" ") || "";
    const classification = classifyAssay(description, name);
    
    return {
      aid,
      name,
      description: description.slice(0, 500),
      sourceId: `AID ${aid}`,
      sourceName: "PubChem",
      targetName: assayDescr.target?.[0]?.name,
      targetGeneId: assayDescr.target?.[0]?.gi,
      category: classification.category,
      confidence: classification.confidence
    };
  } catch (error) {
    console.error(`Error fetching AID ${aid}:`, error);
    return null;
  }
}
