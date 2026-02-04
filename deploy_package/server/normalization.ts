import { storage } from "./storage";
import type { 
  InsertCanonicalMolecule,
  InsertCanonicalMaterial,
  InsertHitList,
  InsertHitListItem,
  InsertCanonicalAssay,
  InsertCanonicalAssayResult 
} from "@shared/schema";

export interface NormalizationResult {
  domain: "drug" | "materials";
  importType: string;
  counts: {
    inserted: number;
    updated: number;
    skippedDuplicates: number;
    failed: number;
  };
  validationReport: {
    totalRows: number;
    validRows: number;
    invalidRows: number;
    errors: Array<{ row: number; field: string; message: string }>;
  };
  normalizationSummary: {
    canonicalizedSmiles?: number;
    computedInchikeys?: number;
    normalizedUnits?: number;
    strippedSalts?: number;
    computedMaterialHashes?: number;
  };
  createdEntities: {
    type: string;
    ids: string[];
  }[];
}

function canonicalizeSmiles(smiles: string): string {
  return smiles.trim();
}

function computeInchikey(smiles: string): string {
  const hash = smiles.split("").reduce((acc, char) => {
    return ((acc << 5) - acc + char.charCodeAt(0)) | 0;
  }, 0);
  return `INCHIKEY-${Math.abs(hash).toString(16).toUpperCase().padStart(14, "0")}`;
}

function computeMaterialHash(representation: Record<string, unknown>, type: string): string {
  const str = JSON.stringify({ type, ...representation });
  const hash = str.split("").reduce((acc, char) => {
    return ((acc << 5) - acc + char.charCodeAt(0)) | 0;
  }, 0);
  return `MAT-${Math.abs(hash).toString(16).toUpperCase().padStart(12, "0")}`;
}

function normalizeUnits(value: number, fromUnit: string, toUnit: string): number {
  const conversionMap: Record<string, Record<string, number>> = {
    uM: { nM: 1000, M: 0.000001 },
    nM: { uM: 0.001, M: 0.000000001 },
    M: { uM: 1000000, nM: 1000000000 },
  };
  if (fromUnit === toUnit) return value;
  return value * (conversionMap[fromUnit]?.[toUnit] || 1);
}

export async function normalizeDrugCompoundLibrary(
  rows: Array<Record<string, string>>,
  columnMapping: Record<string, string>,
  companyId: string,
  source: "import" | "built_in" | "vendor" = "import"
): Promise<NormalizationResult> {
  const result: NormalizationResult = {
    domain: "drug",
    importType: "compound_library",
    counts: { inserted: 0, updated: 0, skippedDuplicates: 0, failed: 0 },
    validationReport: { totalRows: rows.length, validRows: 0, invalidRows: 0, errors: [] },
    normalizationSummary: { canonicalizedSmiles: 0, computedInchikeys: 0, strippedSalts: 0 },
    createdEntities: []
  };

  const smilesColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "smiles");
  const nameColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "name");

  if (!smilesColumn) {
    result.validationReport.errors.push({ row: 0, field: "smiles", message: "SMILES column not mapped" });
    return result;
  }

  const moleculesToInsert: InsertCanonicalMolecule[] = [];
  const seenInchikeys = new Set<string>();

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const rawSmiles = row[smilesColumn];
    
    if (!rawSmiles || rawSmiles.trim() === "") {
      result.validationReport.errors.push({ row: i + 1, field: "smiles", message: "Empty SMILES" });
      result.validationReport.invalidRows++;
      result.counts.failed++;
      continue;
    }

    const canonicalSmiles = canonicalizeSmiles(rawSmiles);
    result.normalizationSummary.canonicalizedSmiles!++;
    
    const inchikey = computeInchikey(canonicalSmiles);
    result.normalizationSummary.computedInchikeys!++;

    const existingMol = await storage.getCanonicalMoleculeByInchikey(inchikey, companyId);
    if (existingMol || seenInchikeys.has(inchikey)) {
      result.counts.skippedDuplicates++;
      continue;
    }
    seenInchikeys.add(inchikey);

    moleculesToInsert.push({
      companyId,
      name: nameColumn ? row[nameColumn] : null,
      canonicalSmiles,
      inchikey,
      source,
    });
    result.validationReport.validRows++;
  }

  if (moleculesToInsert.length > 0) {
    const inserted = await storage.bulkCreateCanonicalMolecules(moleculesToInsert);
    result.counts.inserted = inserted.length;
    result.createdEntities.push({ type: "canonical_molecules", ids: inserted.map(m => m.id) });
  }

  return result;
}

export async function normalizeMaterialsLibrary(
  rows: Array<Record<string, string>>,
  columnMapping: Record<string, string>,
  companyId: string
): Promise<NormalizationResult> {
  const result: NormalizationResult = {
    domain: "materials",
    importType: "materials_library",
    counts: { inserted: 0, updated: 0, skippedDuplicates: 0, failed: 0 },
    validationReport: { totalRows: rows.length, validRows: 0, invalidRows: 0, errors: [] },
    normalizationSummary: { computedMaterialHashes: 0 },
    createdEntities: []
  };

  const nameColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "name");
  const typeColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "type");
  const representationColumn = Object.keys(columnMapping).find(k => 
    ["smiles", "bigsmiles", "cif", "poscar", "representation"].includes(columnMapping[k])
  );

  if (!nameColumn && !representationColumn) {
    result.validationReport.errors.push({ row: 0, field: "name", message: "Name or representation column required" });
    return result;
  }

  const materialsToInsert: InsertCanonicalMaterial[] = [];
  const seenHashes = new Set<string>();

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const name = nameColumn ? row[nameColumn] : undefined;
    const structureType = typeColumn ? (row[typeColumn] as any) : "polymer";
    
    const representation: Record<string, unknown> = {};
    if (representationColumn) {
      representation.raw = row[representationColumn];
    }
    
    const materialHash = computeMaterialHash(representation, structureType);
    result.normalizationSummary.computedMaterialHashes!++;

    const existingMat = await storage.getCanonicalMaterialByHash(materialHash, companyId);
    if (existingMat || seenHashes.has(materialHash)) {
      result.counts.skippedDuplicates++;
      continue;
    }
    seenHashes.add(materialHash);

    materialsToInsert.push({
      companyId,
      name: name || null,
      structureType,
      canonicalRepresentationJson: representation,
      materialHash,
    });
    result.validationReport.validRows++;
  }

  if (materialsToInsert.length > 0) {
    const inserted = await storage.bulkCreateCanonicalMaterials(materialsToInsert);
    result.counts.inserted = inserted.length;
    result.createdEntities.push({ type: "canonical_materials", ids: inserted.map(m => m.id) });
  }

  return result;
}

export async function normalizeHitList(
  rows: Array<Record<string, string>>,
  columnMapping: Record<string, string>,
  campaignId: string,
  companyId: string,
  hitListName: string
): Promise<NormalizationResult> {
  const result: NormalizationResult = {
    domain: "drug",
    importType: "hit_list",
    counts: { inserted: 0, updated: 0, skippedDuplicates: 0, failed: 0 },
    validationReport: { totalRows: rows.length, validRows: 0, invalidRows: 0, errors: [] },
    normalizationSummary: { canonicalizedSmiles: 0, computedInchikeys: 0 },
    createdEntities: []
  };

  const smilesColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "smiles");
  const scoreColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "score");
  const rankColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "rank");

  if (!smilesColumn) {
    result.validationReport.errors.push({ row: 0, field: "smiles", message: "SMILES column not mapped" });
    return result;
  }

  const hitList = await storage.createHitList({ campaignId, name: hitListName, sourceTool: "import" });
  result.createdEntities.push({ type: "hit_lists", ids: [hitList.id] });

  const hitListItemsToInsert: InsertHitListItem[] = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const rawSmiles = row[smilesColumn];
    
    if (!rawSmiles || rawSmiles.trim() === "") {
      result.validationReport.errors.push({ row: i + 1, field: "smiles", message: "Empty SMILES" });
      result.validationReport.invalidRows++;
      result.counts.failed++;
      continue;
    }

    const canonicalSmiles = canonicalizeSmiles(rawSmiles);
    const inchikey = computeInchikey(canonicalSmiles);
    result.normalizationSummary.canonicalizedSmiles!++;
    result.normalizationSummary.computedInchikeys!++;

    let molecule = await storage.getCanonicalMoleculeByInchikey(inchikey, companyId);
    if (!molecule) {
      molecule = await storage.createCanonicalMolecule({
        companyId,
        canonicalSmiles,
        inchikey,
        source: "import",
      });
      result.counts.inserted++;
    }

    hitListItemsToInsert.push({
      hitListId: hitList.id,
      moleculeId: molecule.id,
      score: scoreColumn && row[scoreColumn] ? parseFloat(row[scoreColumn]) : null,
      rank: rankColumn && row[rankColumn] ? parseInt(row[rankColumn]) : i + 1,
    });
    result.validationReport.validRows++;
  }

  if (hitListItemsToInsert.length > 0) {
    const items = await storage.bulkCreateHitListItems(hitListItemsToInsert);
    result.createdEntities.push({ type: "hit_list_items", ids: items.map(it => it.id) });
  }

  return result;
}

export async function normalizeAssayResults(
  rows: Array<Record<string, string>>,
  columnMapping: Record<string, string>,
  companyId: string,
  assayName: string,
  campaignId?: string
): Promise<NormalizationResult> {
  const result: NormalizationResult = {
    domain: "drug",
    importType: "assay_results",
    counts: { inserted: 0, updated: 0, skippedDuplicates: 0, failed: 0 },
    validationReport: { totalRows: rows.length, validRows: 0, invalidRows: 0, errors: [] },
    normalizationSummary: { canonicalizedSmiles: 0, computedInchikeys: 0, normalizedUnits: 0 },
    createdEntities: []
  };

  const smilesColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "smiles");
  const valueColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "value");
  const unitsColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "units");
  const outcomeColumn = Object.keys(columnMapping).find(k => columnMapping[k] === "outcome");

  if (!smilesColumn || !valueColumn) {
    result.validationReport.errors.push({ row: 0, field: "smiles/value", message: "SMILES and value columns required" });
    return result;
  }

  const assay = await storage.createCanonicalAssay({ companyId, name: assayName });
  result.createdEntities.push({ type: "canonical_assays", ids: [assay.id] });

  const assayResultsToInsert: InsertCanonicalAssayResult[] = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const rawSmiles = row[smilesColumn];
    const rawValue = row[valueColumn];
    
    if (!rawSmiles || !rawValue) {
      result.validationReport.errors.push({ row: i + 1, field: "smiles/value", message: "Missing required fields" });
      result.validationReport.invalidRows++;
      result.counts.failed++;
      continue;
    }

    const canonicalSmiles = canonicalizeSmiles(rawSmiles);
    const inchikey = computeInchikey(canonicalSmiles);
    result.normalizationSummary.canonicalizedSmiles!++;
    result.normalizationSummary.computedInchikeys!++;

    let molecule = await storage.getCanonicalMoleculeByInchikey(inchikey, companyId);
    if (!molecule) {
      molecule = await storage.createCanonicalMolecule({
        companyId,
        canonicalSmiles,
        inchikey,
        source: "import",
      });
    }

    const value = parseFloat(rawValue);
    const units = unitsColumn ? row[unitsColumn] : "nM";
    const normalizedValue = normalizeUnits(value, units, "nM");
    result.normalizationSummary.normalizedUnits!++;

    const outcomeLabel = outcomeColumn && row[outcomeColumn] 
      ? (row[outcomeColumn].toLowerCase() as "active" | "inactive" | "toxic" | "ambiguous")
      : undefined;

    assayResultsToInsert.push({
      assayId: assay.id,
      campaignId: campaignId || null,
      moleculeId: molecule.id,
      value: normalizedValue,
      units: "nM",
      outcomeLabel,
    });
    result.validationReport.validRows++;
  }

  if (assayResultsToInsert.length > 0) {
    const results = await storage.bulkCreateCanonicalAssayResults(assayResultsToInsert);
    result.counts.inserted = results.length;
    result.createdEntities.push({ type: "canonical_assay_results", ids: results.map(r => r.id) });
  }

  return result;
}

export async function runNormalization(
  domain: "drug" | "materials",
  importType: string,
  rows: Array<Record<string, string>>,
  columnMapping: Record<string, string>,
  companyId: string,
  options?: {
    campaignId?: string;
    entityName?: string;
  }
): Promise<NormalizationResult> {
  switch (importType) {
    case "compound_library":
      return normalizeDrugCompoundLibrary(rows, columnMapping, companyId);
    case "hit_list":
      return normalizeHitList(rows, columnMapping, options?.campaignId || "", companyId, options?.entityName || "Imported Hit List");
    case "assay_results":
      return normalizeAssayResults(rows, columnMapping, companyId, options?.entityName || "Imported Assay", options?.campaignId);
    case "materials_library":
    case "polymer_library":
    case "crystal_structures":
    case "property_dataset":
      return normalizeMaterialsLibrary(rows, columnMapping, companyId);
    default:
      return {
        domain,
        importType,
        counts: { inserted: 0, updated: 0, skippedDuplicates: 0, failed: rows.length },
        validationReport: { totalRows: rows.length, validRows: 0, invalidRows: rows.length, errors: [{ row: 0, field: "importType", message: `Unsupported import type: ${importType}` }] },
        normalizationSummary: {},
        createdEntities: []
      };
  }
}
