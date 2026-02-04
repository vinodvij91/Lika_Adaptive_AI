import { db } from "../db";
import { molecules } from "@shared/schema";
import { eq, sql } from "drizzle-orm";
import { v4 as uuidv4 } from "uuid";

const CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data";

interface ImportStats {
  imported: number;
  duplicates: number;
  errors: number;
}

async function importFDAApprovedDrugs(limit: number = 2000): Promise<ImportStats> {
  console.log(`\n📦 Importing FDA-approved drugs from ChEMBL (limit: ${limit})...`);
  const stats: ImportStats = { imported: 0, duplicates: 0, errors: 0 };
  
  try {
    let offset = 0;
    const batchSize = 100;
    
    while (stats.imported + stats.duplicates < limit) {
      const url = `${CHEMBL_BASE_URL}/molecule.json?max_phase=4&limit=${batchSize}&offset=${offset}`;
      const response = await fetch(url, { headers: { "Accept": "application/json" } });
      
      if (!response.ok) {
        console.log(`  API returned ${response.status} at offset ${offset}`);
        break;
      }
      
      const data = await response.json();
      if (!data.molecules || data.molecules.length === 0) break;
      
      for (const mol of data.molecules) {
        const smilesStr = mol.molecule_structures?.canonical_smiles;
        if (!smilesStr) continue;
        
        try {
          const existing = await db.select().from(molecules)
            .where(eq(molecules.smiles, smilesStr)).limit(1);
          
          if (existing.length > 0) {
            stats.duplicates++;
            continue;
          }
          
          const mw = mol.molecule_properties?.full_mwt;
          const logP = mol.molecule_properties?.alogp;
          const hbd = mol.molecule_properties?.hbd;
          const hba = mol.molecule_properties?.hba;
          
          await db.insert(molecules).values({
            id: uuidv4(),
            name: mol.pref_name || mol.molecule_chembl_id,
            smiles: smilesStr,
            source: "screened",
            molecularWeight: mw ? parseFloat(mw) : null,
            logP: logP ? parseFloat(logP) : null,
            numHbondDonors: hbd ? parseInt(hbd) : null,
            numHbondAcceptors: hba ? parseInt(hba) : null,
            isDemo: false,
          });
          stats.imported++;
          
          if (stats.imported % 100 === 0) {
            console.log(`  Imported ${stats.imported} FDA drugs...`);
          }
        } catch (e: any) {
          stats.errors++;
          if (stats.errors < 5) {
            console.log(`  Error: ${e.message?.slice(0, 100)}`);
          }
        }
      }
      
      offset += batchSize;
      await new Promise(r => setTimeout(r, 300));
    }
  } catch (e) {
    console.error("FDA import error:", e);
  }
  
  console.log(`  ✓ FDA drugs: ${stats.imported} imported, ${stats.duplicates} duplicates, ${stats.errors} errors`);
  return stats;
}

async function importDruglikeMolecules(limit: number = 5000): Promise<ImportStats> {
  console.log(`\n📦 Importing drug-like molecules by target from ChEMBL (limit: ${limit})...`);
  const stats: ImportStats = { imported: 0, duplicates: 0, errors: 0 };
  
  const targets = [
    "CHEMBL220",  // ACE
    "CHEMBL1827", // HER2
    "CHEMBL203",  // EGFR
    "CHEMBL240",  // COX-2
    "CHEMBL217",  // DRD2
    "CHEMBL228",  // 5-HT2A
    "CHEMBL2147", // BRAF
    "CHEMBL4036", // PD-L1
    "CHEMBL3474", // JAK2
    "CHEMBL1862", // ABL1
    "CHEMBL4295", // KRAS
    "CHEMBL2185", // PARP1
    "CHEMBL5291", // CDK4
    "CHEMBL4523", // BTK
    "CHEMBL279",  // VEGFR2
    "CHEMBL1824", // PI3K
    "CHEMBL4722", // IDH1
    "CHEMBL3880", // BCL2
    "CHEMBL2111432", // FGFR1
    "CHEMBL4282", // RET
  ];
  
  for (const targetId of targets) {
    if (stats.imported >= limit) break;
    
    console.log(`  Fetching compounds for target ${targetId}...`);
    
    try {
      const url = `${CHEMBL_BASE_URL}/activity.json?target_chembl_id=${targetId}&standard_type=IC50&limit=500`;
      const response = await fetch(url, { headers: { "Accept": "application/json" } });
      
      if (!response.ok) {
        console.log(`    Target ${targetId} returned ${response.status}`);
        continue;
      }
      
      const data = await response.json();
      const seenMols = new Set<string>();
      
      for (const activity of data.activities || []) {
        if (stats.imported >= limit) break;
        
        const molId = activity.molecule_chembl_id;
        const smilesStr = activity.canonical_smiles;
        
        if (!molId || !smilesStr || seenMols.has(smilesStr)) continue;
        seenMols.add(smilesStr);
        
        try {
          const existing = await db.select().from(molecules)
            .where(eq(molecules.smiles, smilesStr)).limit(1);
          
          if (existing.length > 0) {
            stats.duplicates++;
            continue;
          }
          
          await db.insert(molecules).values({
            id: uuidv4(),
            name: activity.molecule_pref_name || molId,
            smiles: smilesStr,
            source: "screened",
            isDemo: false,
          });
          stats.imported++;
          
        } catch (e: any) {
          stats.errors++;
        }
      }
      
      await new Promise(r => setTimeout(r, 300));
    } catch (e) {
      console.error(`Target ${targetId} error:`, e);
    }
  }
  
  console.log(`  ✓ Drug-like: ${stats.imported} imported, ${stats.duplicates} duplicates, ${stats.errors} errors`);
  return stats;
}

async function importNaturalProducts(limit: number = 1500): Promise<ImportStats> {
  console.log(`\n📦 Importing natural products from ChEMBL (limit: ${limit})...`);
  const stats: ImportStats = { imported: 0, duplicates: 0, errors: 0 };
  
  try {
    let offset = 0;
    const batchSize = 100;
    
    while (stats.imported + stats.duplicates < limit) {
      const url = `${CHEMBL_BASE_URL}/molecule.json?natural_product=1&limit=${batchSize}&offset=${offset}`;
      const response = await fetch(url, { headers: { "Accept": "application/json" } });
      
      if (!response.ok) break;
      
      const data = await response.json();
      if (!data.molecules || data.molecules.length === 0) break;
      
      for (const mol of data.molecules) {
        const smilesStr = mol.molecule_structures?.canonical_smiles;
        if (!smilesStr) continue;
        
        try {
          const existing = await db.select().from(molecules)
            .where(eq(molecules.smiles, smilesStr)).limit(1);
          
          if (existing.length > 0) {
            stats.duplicates++;
            continue;
          }
          
          const mw = mol.molecule_properties?.full_mwt;
          const logP = mol.molecule_properties?.alogp;
          
          await db.insert(molecules).values({
            id: uuidv4(),
            name: mol.pref_name || mol.molecule_chembl_id,
            smiles: smilesStr,
            source: "screened",
            molecularWeight: mw ? parseFloat(mw) : null,
            logP: logP ? parseFloat(logP) : null,
            isDemo: false,
          });
          stats.imported++;
          
        } catch (e) {
          stats.errors++;
        }
      }
      
      offset += batchSize;
      await new Promise(r => setTimeout(r, 300));
    }
  } catch (e) {
    console.error("Natural products import error:", e);
  }
  
  console.log(`  ✓ Natural products: ${stats.imported} imported, ${stats.duplicates} duplicates, ${stats.errors} errors`);
  return stats;
}

async function importClinicalTrialCompounds(limit: number = 2500): Promise<ImportStats> {
  console.log(`\n📦 Importing clinical trial compounds from ChEMBL (limit: ${limit})...`);
  const stats: ImportStats = { imported: 0, duplicates: 0, errors: 0 };
  
  for (const phase of [3, 2, 1]) {
    if (stats.imported >= limit) break;
    
    console.log(`  Phase ${phase} compounds...`);
    
    try {
      let offset = 0;
      const batchSize = 100;
      const phaseLimit = Math.ceil(limit / 3);
      let phaseImported = 0;
      
      while (phaseImported < phaseLimit && stats.imported < limit) {
        const url = `${CHEMBL_BASE_URL}/molecule.json?max_phase=${phase}&limit=${batchSize}&offset=${offset}`;
        const response = await fetch(url, { headers: { "Accept": "application/json" } });
        
        if (!response.ok) break;
        
        const data = await response.json();
        if (!data.molecules || data.molecules.length === 0) break;
        
        for (const mol of data.molecules) {
          const smilesStr = mol.molecule_structures?.canonical_smiles;
          if (!smilesStr) continue;
          
          try {
            const existing = await db.select().from(molecules)
              .where(eq(molecules.smiles, smilesStr)).limit(1);
            
            if (existing.length > 0) {
              stats.duplicates++;
              continue;
            }
            
            const mw = mol.molecule_properties?.full_mwt;
            const logP = mol.molecule_properties?.alogp;
            
            await db.insert(molecules).values({
              id: uuidv4(),
              name: mol.pref_name || mol.molecule_chembl_id,
              smiles: smilesStr,
              source: "screened",
              molecularWeight: mw ? parseFloat(mw) : null,
              logP: logP ? parseFloat(logP) : null,
              isDemo: false,
            });
            stats.imported++;
            phaseImported++;
            
          } catch (e) {
            stats.errors++;
          }
        }
        
        offset += batchSize;
        await new Promise(r => setTimeout(r, 300));
      }
    } catch (e) {
      console.error(`Phase ${phase} import error:`, e);
    }
  }
  
  console.log(`  ✓ Clinical: ${stats.imported} imported, ${stats.duplicates} duplicates, ${stats.errors} errors`);
  return stats;
}

async function runBulkImport() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("  LIKA SCIENCES - BULK SMILES IMPORT FROM ChEMBL");
  console.log("═══════════════════════════════════════════════════════════════");
  
  const startCount = await db.select({ count: sql<number>`count(*)` }).from(molecules);
  console.log(`\nStarting molecule count: ${startCount[0].count}`);
  
  const results = {
    fda: await importFDAApprovedDrugs(2500),
    druglike: await importDruglikeMolecules(6000),
    natural: await importNaturalProducts(2000),
    clinical: await importClinicalTrialCompounds(3500),
  };
  
  const endCount = await db.select({ count: sql<number>`count(*)` }).from(molecules);
  
  console.log("\n═══════════════════════════════════════════════════════════════");
  console.log("  IMPORT COMPLETE");
  console.log("═══════════════════════════════════════════════════════════════");
  console.log(`  Starting count: ${startCount[0].count}`);
  console.log(`  Ending count:   ${endCount[0].count}`);
  console.log(`  New molecules:  ${Number(endCount[0].count) - Number(startCount[0].count)}`);
  console.log("═══════════════════════════════════════════════════════════════");
  
  return results;
}

runBulkImport()
  .then(() => {
    console.log("\nImport finished successfully!");
    process.exit(0);
  })
  .catch((e) => {
    console.error("Import failed:", e);
    process.exit(1);
  });
