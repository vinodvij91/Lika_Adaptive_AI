import { storage } from "./storage";
import { bionemoClient, molecularMLClient, dockingClient } from "./clients";
import type { Campaign, PipelineConfig, JobType, InsertMoleculeScore, InsertLearningGraphEntry } from "@shared/schema";

export class JobOrchestrator {
  private runningCampaigns: Set<string> = new Set();

  async startCampaign(campaignId: string): Promise<void> {
    if (this.runningCampaigns.has(campaignId)) {
      throw new Error("Campaign is already running");
    }

    const campaign = await storage.getCampaign(campaignId);
    if (!campaign) {
      throw new Error("Campaign not found");
    }

    this.runningCampaigns.add(campaignId);

    await storage.updateCampaign(campaignId, { status: "running" });

    this.executePipeline(campaign).catch((error) => {
      console.error(`Campaign ${campaignId} failed:`, error);
      storage.updateCampaign(campaignId, { status: "failed" });
      this.runningCampaigns.delete(campaignId);
    });
  }

  private async executePipeline(campaign: Campaign): Promise<void> {
    const config = campaign.pipelineConfig as PipelineConfig | null;
    const campaignId = campaign.id;

    try {
      const generationJob = await storage.createJob({
        campaignId,
        type: "generation",
        status: "running",
      });
      await storage.updateJob(generationJob.id, { startedAt: new Date() });

      let moleculeIds: string[] = [];

      if (config?.generator === "bionemo_molmim") {
        const generated = await bionemoClient.generateMolecules(config.generatorParams?.seedSmiles, config.generatorParams?.n || 50);
        const newMolecules = await storage.bulkCreateMolecules(
          generated.map((g) => ({ smiles: g.smiles, source: "generated" as const }))
        );
        moleculeIds = newMolecules.map((m) => m.id);
      } else {
        const existingMolecules = await storage.getMolecules();
        moleculeIds = existingMolecules.slice(0, 50).map((m) => m.id);
      }

      await storage.updateJob(generationJob.id, { status: "completed", finishedAt: new Date() });

      const filteringJob = await storage.createJob({
        campaignId,
        type: "filtering",
        status: "running",
      });
      await storage.updateJob(filteringJob.id, { startedAt: new Date() });

      const admetPredictions = await molecularMLClient.predictAdmet(moleculeIds);

      const admetScores: Map<string, number> = new Map();
      for (const pred of admetPredictions) {
        admetScores.set(pred.moleculeId, pred.score);
      }

      const filteredMoleculeIds = moleculeIds.filter((id) => {
        const score = admetScores.get(id) || 0;
        return score >= 0.3;
      });

      await storage.updateJob(filteringJob.id, { status: "completed", finishedAt: new Date() });

      const dockingJob = await storage.createJob({
        campaignId,
        type: "docking",
        status: "running",
      });
      await storage.updateJob(dockingJob.id, { startedAt: new Date() });

      let dockingResults;
      if (config?.dockingMethod === "bionemo_diffdock") {
        dockingResults = await bionemoClient.predictDocking(filteredMoleculeIds, config.targetIds?.[0] || "default");
      } else {
        dockingResults = await dockingClient.dock(filteredMoleculeIds, config?.targetIds?.[0] || "default");
      }

      const dockingScores: Map<string, number> = new Map();
      for (const result of dockingResults) {
        dockingScores.set(result.moleculeId, result.score);
      }

      await storage.updateJob(dockingJob.id, { status: "completed", finishedAt: new Date() });

      const scoringJob = await storage.createJob({
        campaignId,
        type: "scoring",
        status: "running",
      });
      await storage.updateJob(scoringJob.id, { startedAt: new Date() });

      const qsarPredictions = await molecularMLClient.predictQsar(filteredMoleculeIds, config?.targetIds?.[0] || "default");
      const qsarScores: Map<string, number> = new Map();
      for (const pred of qsarPredictions) {
        qsarScores.set(pred.moleculeId, pred.score);
      }

      const weights = config?.scoringWeights || { wDocking: 0.4, wAdmet: 0.3, wQsar: 0.3 };

      const moleculeScoresData: InsertMoleculeScore[] = filteredMoleculeIds.map((moleculeId) => {
        const docking = dockingScores.get(moleculeId) || 0;
        const admet = admetScores.get(moleculeId) || 0;
        const qsar = qsarScores.get(moleculeId) || 0;
        const oracle = weights.wDocking * docking + weights.wAdmet * admet + weights.wQsar * qsar;

        return {
          moleculeId,
          campaignId,
          dockingScore: docking,
          admetScore: admet,
          qsarScore: qsar,
          oracleScore: oracle,
          rawScores: { docking, admet, qsar },
        };
      });

      await storage.bulkCreateMoleculeScores(moleculeScoresData);

      const sortedScores = moleculeScoresData.sort((a, b) => (b.oracleScore || 0) - (a.oracleScore || 0));
      const topMolecules = sortedScores.slice(0, Math.min(10, sortedScores.length));

      for (const score of topMolecules) {
        const outcome = (score.oracleScore || 0) >= 0.7 ? "promising" : (score.oracleScore || 0) >= 0.5 ? "hit" : "unknown";
        
        const entry: InsertLearningGraphEntry = {
          moleculeId: score.moleculeId,
          campaignId,
          domainType: campaign.domainType || "Other",
          outcomeLabel: outcome as "promising" | "hit" | "dropped" | "unknown",
          oracleScore: score.oracleScore || 0,
        };
        
        await storage.createLearningGraphEntry(entry);
      }

      await storage.updateJob(scoringJob.id, { status: "completed", finishedAt: new Date() });

      await storage.updateCampaign(campaignId, { status: "completed" });
      this.runningCampaigns.delete(campaignId);

    } catch (error) {
      console.error(`Pipeline execution failed for campaign ${campaignId}:`, error);
      await storage.updateCampaign(campaignId, { status: "failed" });
      this.runningCampaigns.delete(campaignId);
      throw error;
    }
  }

  isRunning(campaignId: string): boolean {
    return this.runningCampaigns.has(campaignId);
  }
}

export const orchestrator = new JobOrchestrator();
