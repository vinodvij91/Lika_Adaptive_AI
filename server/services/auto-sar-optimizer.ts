import { db } from "../db";
import { eq, inArray } from "drizzle-orm";
import { molecules, moleculeScores } from "@shared/schema";
import type { Molecule } from "@shared/schema";
import { storage } from "../storage";
import {
  optimizeMoleculeProperties,
  optimizeDoseIndication,
} from "./molecule-optimizer";

export interface AutoSarResult {
  totalProcessed: number;
  totalSkipped: number;
  totalAnalogsInserted: number;
  totalDoseScenarios: number;
}

export async function runAutoSarOptimization(
  campaignId: string,
  diseaseContext: string
): Promise<AutoSarResult> {
  const allScores = await db
    .select({
      moleculeId: moleculeScores.moleculeId,
      rawScores: moleculeScores.rawScores,
    })
    .from(moleculeScores)
    .where(eq(moleculeScores.campaignId, campaignId));

  const alreadyOptimizedFromSmiles = new Set<string>();
  for (const s of allScores) {
    const raw = s.rawScores as Record<string, unknown> | null;
    if (raw && typeof raw === "object" && "optimizedFrom" in raw) {
      alreadyOptimizedFromSmiles.add(raw.optimizedFrom as string);
    }
  }

  const originalMolIds = allScores
    .filter((s) => {
      const raw = s.rawScores as Record<string, unknown> | null;
      return !raw || typeof raw !== "object" || !("optimizedFrom" in raw);
    })
    .map((s) => s.moleculeId)
    .filter(Boolean) as string[];

  if (originalMolIds.length === 0) {
    return { totalProcessed: 0, totalSkipped: 0, totalAnalogsInserted: 0, totalDoseScenarios: 0 };
  }

  const targetMolecules = await db
    .select()
    .from(molecules)
    .where(inArray(molecules.id, originalMolIds));

  const toOptimize = targetMolecules.filter(
    (m) => !alreadyOptimizedFromSmiles.has(m.smiles)
  );

  let totalAnalogsInserted = 0;
  let totalDoseScenarios = 0;
  let totalProcessed = 0;

  for (const mol of toOptimize) {
    try {
      const result = optimizeMoleculeProperties(mol.smiles, diseaseContext, {
        mw: mol.molecularWeight,
        logP: mol.logP,
        hbd: mol.numHBondDonors,
        hba: mol.numHBondAcceptors,
      });

      for (const analog of result.analogs) {
        try {
          const newMol = await storage.createMolecule({
            smiles: analog.smiles,
            name: analog.name,
            seriesId: mol.seriesId || null,
            scaffoldId: mol.scaffoldId || null,
            source: "generated",
            molecularWeight: analog.predictedProperties.molecularWeight,
            logP: analog.predictedProperties.logP,
            numHBondDonors: null,
            numHBondAcceptors: null,
            isDemo: false,
          });
          await storage.createMoleculeScore({
            moleculeId: newMol.id,
            campaignId,
            admetScore:
              analog.admetPredictions.bioavailability,
            oracleScore:
              Math.round(
                ((analog.admetPredictions.bioavailability +
                  analog.admetPredictions.metabolicStability) /
                  2) *
                  100
              ) / 100,
            rawScores: {
              optimizedFrom: mol.smiles,
              modification: analog.modification,
              admet: analog.admetPredictions,
            },
          });
          totalAnalogsInserted++;
        } catch (err) {
          console.error(
            `[AutoSAR] Error inserting analog for molecule ${mol.id}:`,
            err
          );
        }
      }

      const doseResult = optimizeDoseIndication(
        mol.smiles,
        diseaseContext,
        mol.name || undefined
      );
      totalDoseScenarios += doseResult.doseScenarios.length;

      totalProcessed++;
    } catch (err) {
      console.error(
        `[AutoSAR] Error optimizing molecule ${mol.id}:`,
        err
      );
    }
  }

  return {
    totalProcessed,
    totalSkipped: targetMolecules.length - toOptimize.length,
    totalAnalogsInserted,
    totalDoseScenarios,
  };
}
