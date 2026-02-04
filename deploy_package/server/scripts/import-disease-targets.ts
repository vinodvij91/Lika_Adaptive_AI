import { db } from "../db";
import { targets, diseaseTargetMappings } from "@shared/schema";
import { eq } from "drizzle-orm";
import * as fs from "fs";
import * as path from "path";

interface CsvRow {
  Disease: string;
  UniProt_ID: string;
  Protein_Name: string;
  Gene_Name: string;
  Organism: string;
  Sequence: string;
  Sequence_Length: string;
  SMILES: string;
  ChEMBL_ID: string;
  PDB_IDs: string;
  Has_Structure: string;
  Number_of_Structures: string;
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

async function importDiseaseTargets() {
  const csvPath = path.join(process.cwd(), "attached_assets", "disease_targets_1769368229379.csv");
  
  if (!fs.existsSync(csvPath)) {
    console.error("CSV file not found:", csvPath);
    process.exit(1);
  }
  
  const content = fs.readFileSync(csvPath, "utf-8");
  const lines = content.split("\n").filter(line => line.trim());
  const headers = parseCSVLine(lines[0]);
  
  console.log("Headers:", headers);
  console.log(`Total rows to process: ${lines.length - 1}`);
  
  const targetMap = new Map<string, string>();
  const diseaseMappings: { targetId: string; disease: string }[] = [];
  
  let imported = 0;
  let skipped = 0;
  
  for (let i = 1; i < lines.length; i++) {
    try {
      const values = parseCSVLine(lines[i]);
      if (values.length < headers.length) continue;
      
      const row: CsvRow = {
        Disease: values[0] || "",
        UniProt_ID: values[1] || "",
        Protein_Name: values[2] || "",
        Gene_Name: values[3] || "",
        Organism: values[4] || "",
        Sequence: values[5] || "",
        Sequence_Length: values[6] || "",
        SMILES: values[7] || "",
        ChEMBL_ID: values[8] || "",
        PDB_IDs: values[9] || "",
        Has_Structure: values[10] || "",
        Number_of_Structures: values[11] || "",
      };
      
      if (!row.Protein_Name || !row.UniProt_ID) {
        skipped++;
        continue;
      }
      
      const targetKey = row.UniProt_ID;
      let targetId = targetMap.get(targetKey);
      
      if (!targetId) {
        const existing = await db.select().from(targets)
          .where(eq(targets.uniprotId, row.UniProt_ID))
          .limit(1);
        
        if (existing.length > 0) {
          targetId = existing[0].id;
        } else {
          const [newTarget] = await db.insert(targets).values({
            name: row.Protein_Name,
            uniprotId: row.UniProt_ID,
            pdbId: row.PDB_IDs && row.PDB_IDs !== "N/A" ? row.PDB_IDs.split(";")[0] : null,
            sequence: row.Sequence && row.Sequence !== "N/A" ? row.Sequence : null,
            hasStructure: row.Has_Structure === "True",
            geneName: row.Gene_Name || null,
            organism: row.Organism || null,
            chemblId: row.ChEMBL_ID && row.ChEMBL_ID !== "N/A" ? row.ChEMBL_ID : null,
            smiles: row.SMILES && row.SMILES !== "N/A" ? row.SMILES : null,
            sequenceLength: row.Sequence_Length ? parseInt(row.Sequence_Length) || null : null,
            numStructures: row.Number_of_Structures ? parseInt(row.Number_of_Structures) || 0 : 0,
            isDemo: false,
          }).returning();
          targetId = newTarget.id;
          imported++;
        }
        targetMap.set(targetKey, targetId);
      }
      
      if (row.Disease) {
        diseaseMappings.push({ targetId, disease: row.Disease });
      }
      
      if (i % 1000 === 0) {
        console.log(`Processed ${i}/${lines.length - 1} rows (${imported} new targets)`);
      }
    } catch (error) {
      console.error(`Error at row ${i}:`, error);
    }
  }
  
  console.log(`Inserting ${diseaseMappings.length} disease mappings...`);
  
  const batchSize = 500;
  for (let i = 0; i < diseaseMappings.length; i += batchSize) {
    const batch = diseaseMappings.slice(i, i + batchSize);
    await db.insert(diseaseTargetMappings).values(batch);
    console.log(`Inserted disease mappings batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(diseaseMappings.length / batchSize)}`);
  }
  
  console.log(`\nImport complete!`);
  console.log(`- New targets imported: ${imported}`);
  console.log(`- Skipped rows: ${skipped}`);
  console.log(`- Disease mappings created: ${diseaseMappings.length}`);
  console.log(`- Unique targets: ${targetMap.size}`);
}

importDiseaseTargets()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("Import failed:", err);
    process.exit(1);
  });
