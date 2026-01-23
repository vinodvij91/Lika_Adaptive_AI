import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { orchestrator } from "./orchestrator";
import { registerArtifactsFromManifest } from "./artifact-ingestion";
import { db } from "./db";
import { eq, count } from "drizzle-orm";
import { z } from "zod";
import {
  insertProjectSchema,
  insertTargetSchema,
  insertCampaignSchema,
  insertCommentSchema,
  insertCuratedLibrarySchema,
  insertLibraryMoleculeSchema,
  insertScaffoldSchema,
  insertLibraryAnnotationSchema,
  insertComputeNodeSchema,
  insertSshConfigSchema,
  insertCompanySchema,
  insertUserSshKeySchema,
  insertTargetVariantSchema,
  insertDiseaseContextSignalSchema,
  insertProgramSchema,
  insertOracleVersionSchema,
  insertAssaySchema,
  insertExperimentRecommendationSchema,
  insertAssayResultSchema,
  insertLiteratureAnnotationSchema,
  insertOrganizationSchema,
  insertOrgMemberSchema,
  insertSharedAssetSchema,
  insertMaterialEntitySchema,
  insertMaterialPropertySchema,
  insertMaterialVariantSchema,
  insertMaterialsProgramSchema,
  insertMaterialsCampaignSchema,
  insertMaterialsOracleScoreSchema,
  insertMaterialsLearningGraphSchema,
  insertProcessingJobSchema,
  insertProcessingJobEventSchema,
  insertJobArtifactSchema,
  insertCompoundAssetSchema,
  insertAssayPanelSchema,
  insertAssayPanelTargetSchema,
  insertMoaNodeSchema,
  insertMoaEdgeSchema,
  insertImportTemplateSchema,
  insertImportJobSchema,
  moleculeScores,
} from "@shared/schema";

function requireAuth(req: Request, res: Response, next: NextFunction) {
  // DEV MODE: Skip authentication for exploration
  if (!req.user) {
    (req as any).user = { id: "dev-user", username: "Developer", email: "dev@lika.sciences" };
  }
  next();
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  app.get("/api/dashboard/stats", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const stats = await storage.getDashboardStats(userId);
      res.json(stats);
    } catch (error) {
      console.error("Error fetching dashboard stats:", error);
      res.status(500).json({ error: "Failed to fetch dashboard stats" });
    }
  });

  app.get("/api/projects", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const projects = await storage.getProjects(userId);
      res.json(projects);
    } catch (error) {
      console.error("Error fetching projects:", error);
      res.status(500).json({ error: "Failed to fetch projects" });
    }
  });

  app.get("/api/projects/:id", requireAuth, async (req, res) => {
    try {
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      
      const campaigns = await storage.getCampaigns(project.id);
      const projectComments = await storage.getComments(project.id);
      
      res.json({
        ...project,
        campaigns,
        comments: projectComments,
      });
    } catch (error) {
      console.error("Error fetching project:", error);
      res.status(500).json({ error: "Failed to fetch project" });
    }
  });

  app.post("/api/projects", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertProjectSchema.safeParse({ ...req.body, ownerId: userId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const project = await storage.createProject(parsed.data);
      res.status(201).json(project);
    } catch (error) {
      console.error("Error creating project:", error);
      res.status(500).json({ error: "Failed to create project" });
    }
  });

  app.patch("/api/projects/:id", requireAuth, async (req, res) => {
    try {
      const project = await storage.updateProject(req.params.id, req.body);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      res.json(project);
    } catch (error) {
      console.error("Error updating project:", error);
      res.status(500).json({ error: "Failed to update project" });
    }
  });

  app.delete("/api/projects/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteProject(req.params.id);
      res.status(204).end();
    } catch (error) {
      console.error("Error deleting project:", error);
      res.status(500).json({ error: "Failed to delete project" });
    }
  });

  app.post("/api/projects/:id/comments", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const comment = await storage.createComment({
        projectId: req.params.id,
        userId,
        body: req.body.body,
      });
      res.status(201).json(comment);
    } catch (error) {
      console.error("Error creating comment:", error);
      res.status(500).json({ error: "Failed to create comment" });
    }
  });

  app.get("/api/targets", requireAuth, async (req, res) => {
    try {
      const targets = await storage.getTargets();
      res.json(targets);
    } catch (error) {
      console.error("Error fetching targets:", error);
      res.status(500).json({ error: "Failed to fetch targets" });
    }
  });

  app.get("/api/targets/:id", requireAuth, async (req, res) => {
    try {
      const target = await storage.getTarget(req.params.id);
      if (!target) {
        return res.status(404).json({ error: "Target not found" });
      }
      res.json(target);
    } catch (error) {
      console.error("Error fetching target:", error);
      res.status(500).json({ error: "Failed to fetch target" });
    }
  });

  app.post("/api/targets", requireAuth, async (req, res) => {
    try {
      const parsed = insertTargetSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const target = await storage.createTarget(parsed.data);
      res.status(201).json(target);
    } catch (error) {
      console.error("Error creating target:", error);
      res.status(500).json({ error: "Failed to create target" });
    }
  });

  app.patch("/api/targets/:id", requireAuth, async (req, res) => {
    try {
      const target = await storage.updateTarget(req.params.id, req.body);
      if (!target) {
        return res.status(404).json({ error: "Target not found" });
      }
      res.json(target);
    } catch (error) {
      console.error("Error updating target:", error);
      res.status(500).json({ error: "Failed to update target" });
    }
  });

  app.delete("/api/targets/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteTarget(req.params.id);
      res.status(204).end();
    } catch (error) {
      console.error("Error deleting target:", error);
      res.status(500).json({ error: "Failed to delete target" });
    }
  });

  app.get("/api/molecules", requireAuth, async (req, res) => {
    try {
      const molecules = await storage.getMolecules();
      res.json(molecules);
    } catch (error) {
      console.error("Error fetching molecules:", error);
      res.status(500).json({ error: "Failed to fetch molecules" });
    }
  });

  app.get("/api/molecules/:id", requireAuth, async (req, res) => {
    try {
      const molecule = await storage.getMolecule(req.params.id);
      if (!molecule) {
        return res.status(404).json({ error: "Molecule not found" });
      }
      res.json(molecule);
    } catch (error) {
      console.error("Error fetching molecule:", error);
      res.status(500).json({ error: "Failed to fetch molecule" });
    }
  });

  app.get("/api/campaigns", requireAuth, async (req, res) => {
    try {
      const projectId = req.query.projectId as string | undefined;
      const campaigns = await storage.getCampaigns(projectId);
      res.json(campaigns);
    } catch (error) {
      console.error("Error fetching campaigns:", error);
      res.status(500).json({ error: "Failed to fetch campaigns" });
    }
  });

  app.get("/api/campaigns/:id", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.getCampaign(req.params.id);
      if (!campaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }
      
      const campaignJobs = await storage.getJobs(campaign.id);
      const moleculeScores = await storage.getMoleculeScores(campaign.id);
      
      res.json({
        ...campaign,
        jobs: campaignJobs,
        moleculeScores,
      });
    } catch (error) {
      console.error("Error fetching campaign:", error);
      res.status(500).json({ error: "Failed to fetch campaign" });
    }
  });

  app.post("/api/campaigns", requireAuth, async (req, res) => {
    try {
      const parsed = insertCampaignSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const campaign = await storage.createCampaign(parsed.data);
      res.status(201).json(campaign);
    } catch (error) {
      console.error("Error creating campaign:", error);
      res.status(500).json({ error: "Failed to create campaign" });
    }
  });

  app.post("/api/campaigns/:id/start", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.getCampaign(req.params.id);
      if (!campaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }

      if (campaign.status !== "pending") {
        return res.status(400).json({ error: "Campaign is not in pending status" });
      }

      await orchestrator.startCampaign(campaign.id);
      
      const updatedCampaign = await storage.getCampaign(campaign.id);
      res.json(updatedCampaign);
    } catch (error) {
      console.error("Error starting campaign:", error);
      res.status(500).json({ error: "Failed to start campaign" });
    }
  });

  app.patch("/api/campaigns/:id", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.updateCampaign(req.params.id, req.body);
      if (!campaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }
      res.json(campaign);
    } catch (error) {
      console.error("Error updating campaign:", error);
      res.status(500).json({ error: "Failed to update campaign" });
    }
  });

  app.delete("/api/campaigns/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteCampaign(req.params.id);
      res.status(204).end();
    } catch (error) {
      console.error("Error deleting campaign:", error);
      res.status(500).json({ error: "Failed to delete campaign" });
    }
  });

  app.get("/api/reports", requireAuth, async (req, res) => {
    try {
      const reports = await storage.getReportsData();
      res.json(reports);
    } catch (error) {
      console.error("Error fetching reports:", error);
      res.status(500).json({ error: "Failed to fetch reports" });
    }
  });

  app.get("/api/learning-graph", requireAuth, async (req, res) => {
    try {
      const entries = await storage.getLearningGraphEntries();
      res.json(entries);
    } catch (error) {
      console.error("Error fetching learning graph:", error);
      res.status(500).json({ error: "Failed to fetch learning graph" });
    }
  });

  // ============================================
  // CURATED LIBRARIES API ENDPOINTS
  // Core USP: Domain-aware SMILES libraries as intelligent starting points
  // ============================================

  app.get("/api/libraries", requireAuth, async (req, res) => {
    try {
      const domainType = req.query.domainType as string | undefined;
      const status = req.query.status as string | undefined;
      const isPublic = req.query.isPublic === "true" ? true : req.query.isPublic === "false" ? false : undefined;
      
      const libraries = await storage.getCuratedLibraries({
        domainType: domainType as any,
        status,
        isPublic,
      });
      res.json(libraries);
    } catch (error) {
      console.error("Error fetching libraries:", error);
      res.status(500).json({ error: "Failed to fetch libraries" });
    }
  });

  app.get("/api/libraries/:id", requireAuth, async (req, res) => {
    try {
      const library = await storage.getCuratedLibrary(req.params.id);
      if (!library) {
        return res.status(404).json({ error: "Library not found" });
      }

      const libraryMolecules = await storage.getLibraryMolecules(library.id);
      const libraryScaffolds = await storage.getScaffolds(library.id);
      const libraryAnnotations = await storage.getLibraryAnnotations(library.id);

      res.json({
        ...library,
        molecules: libraryMolecules,
        scaffolds: libraryScaffolds,
        annotations: libraryAnnotations,
      });
    } catch (error) {
      console.error("Error fetching library:", error);
      res.status(500).json({ error: "Failed to fetch library" });
    }
  });

  app.post("/api/libraries", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertCuratedLibrarySchema.safeParse({ ...req.body, ownerId: userId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const library = await storage.createCuratedLibrary(parsed.data);
      res.status(201).json(library);
    } catch (error) {
      console.error("Error creating library:", error);
      res.status(500).json({ error: "Failed to create library" });
    }
  });

  app.patch("/api/libraries/:id", requireAuth, async (req, res) => {
    try {
      const library = await storage.updateCuratedLibrary(req.params.id, req.body);
      if (!library) {
        return res.status(404).json({ error: "Library not found" });
      }
      res.json(library);
    } catch (error) {
      console.error("Error updating library:", error);
      res.status(500).json({ error: "Failed to update library" });
    }
  });

  app.delete("/api/libraries/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteCuratedLibrary(req.params.id);
      res.status(204).end();
    } catch (error) {
      console.error("Error deleting library:", error);
      res.status(500).json({ error: "Failed to delete library" });
    }
  });

  app.get("/api/libraries/:id/molecules", requireAuth, async (req, res) => {
    try {
      const molecules = await storage.getLibraryMolecules(req.params.id);
      res.json(molecules);
    } catch (error) {
      console.error("Error fetching library molecules:", error);
      res.status(500).json({ error: "Failed to fetch library molecules" });
    }
  });

  app.post("/api/libraries/:id/molecules", requireAuth, async (req, res) => {
    try {
      const parsed = insertLibraryMoleculeSchema.safeParse({
        ...req.body,
        libraryId: req.params.id,
      });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const entry = await storage.addLibraryMolecule(parsed.data);
      res.status(201).json(entry);
    } catch (error) {
      console.error("Error adding molecule to library:", error);
      res.status(500).json({ error: "Failed to add molecule to library" });
    }
  });

  const bulkImportSchema = z.object({
    smilesList: z.array(z.string().min(1)),
    tags: z.array(z.string()).optional(),
  });

  app.post("/api/libraries/:id/import-smiles", requireAuth, async (req, res) => {
    try {
      const parsed = bulkImportSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }

      const { smilesList, tags } = parsed.data;
      const libraryId = req.params.id;

      const library = await storage.getCuratedLibrary(libraryId);
      if (!library) {
        return res.status(404).json({ error: "Library not found" });
      }

      const newMolecules = await storage.bulkCreateMolecules(
        smilesList.map((smiles) => ({ smiles, source: "uploaded" as const }))
      );

      const libraryMoleculeEntries = newMolecules.map((mol) => ({
        libraryId,
        moleculeId: mol.id,
        canonicalSmiles: mol.smiles,
        cleaningStatus: "pending" as const,
        tags: tags || [],
      }));

      const added = await storage.bulkAddLibraryMolecules(libraryMoleculeEntries);

      await storage.updateCuratedLibrary(libraryId, {
        moleculeCount: (library.moleculeCount || 0) + added.length,
        status: "processing",
      });

      res.status(201).json({
        imported: added.length,
        duplicates: 0,
        libraryId,
      });
    } catch (error) {
      console.error("Error importing SMILES to library:", error);
      res.status(500).json({ error: "Failed to import SMILES" });
    }
  });

  app.get("/api/libraries/:id/scaffolds", requireAuth, async (req, res) => {
    try {
      const scaffolds = await storage.getScaffolds(req.params.id);
      res.json(scaffolds);
    } catch (error) {
      console.error("Error fetching scaffolds:", error);
      res.status(500).json({ error: "Failed to fetch scaffolds" });
    }
  });

  app.post("/api/libraries/:id/scaffolds", requireAuth, async (req, res) => {
    try {
      const parsed = insertScaffoldSchema.safeParse({
        ...req.body,
        libraryId: req.params.id,
      });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const scaffold = await storage.createScaffold(parsed.data);
      res.status(201).json(scaffold);
    } catch (error) {
      console.error("Error creating scaffold:", error);
      res.status(500).json({ error: "Failed to create scaffold" });
    }
  });

  app.get("/api/libraries/:id/annotations", requireAuth, async (req, res) => {
    try {
      const annotations = await storage.getLibraryAnnotations(req.params.id);
      res.json(annotations);
    } catch (error) {
      console.error("Error fetching annotations:", error);
      res.status(500).json({ error: "Failed to fetch annotations" });
    }
  });

  app.post("/api/libraries/:id/annotations", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertLibraryAnnotationSchema.safeParse({
        ...req.body,
        libraryId: req.params.id,
        userId,
        source: "user",
      });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const annotation = await storage.createLibraryAnnotation(parsed.data);
      res.status(201).json(annotation);
    } catch (error) {
      console.error("Error creating annotation:", error);
      res.status(500).json({ error: "Failed to create annotation" });
    }
  });

  // ============================================
  // AGENT-FRIENDLY API ENDPOINTS
  // These endpoints are designed for AI agents and bots
  // They share the same business logic as UI endpoints
  // ============================================

  app.get("/api/agent/campaigns/pending", requireAuth, async (req, res) => {
    try {
      const pendingCampaigns = await storage.getPendingCampaigns();
      res.json({
        campaigns: pendingCampaigns.map((c) => ({
          id: c.id,
          name: c.name,
          projectId: c.projectId,
          domainType: c.domainType,
          status: c.status,
          createdAt: c.createdAt,
        })),
      });
    } catch (error) {
      console.error("Error fetching pending campaigns for agent:", error);
      res.status(500).json({ error: "Failed to fetch pending campaigns" });
    }
  });

  app.get("/api/agent/campaigns/:id/analytics", requireAuth, async (req, res) => {
    try {
      const analytics = await storage.getCampaignAnalytics(req.params.id);
      res.json(analytics);
    } catch (error) {
      console.error("Error fetching campaign analytics for agent:", error);
      res.status(500).json({ error: "Failed to fetch campaign analytics" });
    }
  });

  app.post("/api/agent/campaigns/:id/start", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.getCampaign(req.params.id);
      if (!campaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }

      if (campaign.status !== "pending") {
        return res.status(400).json({ error: "Campaign is not in pending status" });
      }

      await orchestrator.startCampaign(campaign.id);
      
      const updatedCampaign = await storage.getCampaign(campaign.id);
      res.json({
        campaignId: updatedCampaign?.id,
        status: updatedCampaign?.status,
        message: "Campaign started successfully",
      });
    } catch (error) {
      console.error("Error starting campaign for agent:", error);
      res.status(500).json({ error: "Failed to start campaign" });
    }
  });

  app.get("/api/agent/learning-graph/unlabeled", requireAuth, async (req, res) => {
    try {
      const entries = await storage.getUnlabeledLearningGraphEntries();
      res.json({
        entries: entries.map((e) => ({
          id: e.id,
          moleculeId: e.moleculeId,
          smiles: e.molecule?.smiles,
          campaignId: e.campaignId,
          oracleScore: e.oracleScore,
          domainType: e.domainType,
        })),
      });
    } catch (error) {
      console.error("Error fetching unlabeled entries for agent:", error);
      res.status(500).json({ error: "Failed to fetch unlabeled entries" });
    }
  });

  const agentLabelSchema = z.object({
    entryId: z.string().min(1, "entryId is required"),
    label: z.enum(["promising", "dropped", "hit", "unknown"]),
  });

  app.post("/api/agent/learning-graph/label", requireAuth, async (req, res) => {
    try {
      const parsed = agentLabelSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }

      const { entryId, label } = parsed.data;

      const updated = await storage.updateLearningGraphLabel(entryId, label);
      if (!updated) {
        return res.status(404).json({ error: "Learning graph entry not found" });
      }

      res.json({
        entryId: updated.id,
        moleculeId: updated.moleculeId,
        newLabel: updated.outcomeLabel,
        message: "Label updated successfully",
      });
    } catch (error) {
      console.error("Error labeling learning graph entry for agent:", error);
      res.status(500).json({ error: "Failed to update label" });
    }
  });

  app.post("/api/agent/quantum-recommendation", requireAuth, async (req, res) => {
    try {
      const { campaignId } = req.body;
      
      if (!campaignId) {
        return res.status(400).json({ error: "campaignId is required" });
      }

      const campaign = await storage.getCampaign(campaignId);
      if (!campaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }

      const [moleculeCount] = await db.select({ count: count() }).from(moleculeScores).where(eq(moleculeScores.campaignId, campaignId));
      const totalMolecules = Number(moleculeCount?.count || 0);

      const shouldUseQuantum = totalMolecules > 100;
      const reasoning = shouldUseQuantum
        ? `Campaign has ${totalMolecules} molecules, which is large enough to benefit from quantum optimization for combinatorial selection.`
        : `Campaign has only ${totalMolecules} molecules. Quantum optimization is typically beneficial for larger candidate pools (>100).`;

      res.json({
        shouldUseQuantum,
        suggestedOperation: shouldUseQuantum ? "qaoa_selection" : null,
        reasoning,
        campaignId,
        moleculeCount: totalMolecules,
      });
    } catch (error) {
      console.error("Error generating quantum recommendation:", error);
      res.status(500).json({ error: "Failed to generate quantum recommendation" });
    }
  });

  // ============================================
  // AGENT LIBRARY MANAGEMENT ENDPOINTS
  // Curated SMILES library validation and enrichment
  // ============================================

  app.get("/api/agent/libraries/curated", requireAuth, async (req, res) => {
    try {
      const libraries = await storage.getCuratedLibraries({ status: "curated", isPublic: true });
      res.json({
        libraries: libraries.map((lib) => ({
          id: lib.id,
          name: lib.name,
          domainType: lib.domainType,
          moleculeCount: lib.moleculeCount,
          scaffoldCount: lib.scaffoldCount,
          status: lib.status,
        })),
      });
    } catch (error) {
      console.error("Error fetching curated libraries for agent:", error);
      res.status(500).json({ error: "Failed to fetch curated libraries" });
    }
  });

  app.get("/api/agent/libraries/:id/status", requireAuth, async (req, res) => {
    try {
      const library = await storage.getCuratedLibrary(req.params.id);
      if (!library) {
        return res.status(404).json({ error: "Library not found" });
      }

      const libraryMolecules = await storage.getLibraryMolecules(library.id);
      const pendingCount = libraryMolecules.filter((m) => m.cleaningStatus === "pending").length;
      const validatedCount = libraryMolecules.filter((m) => m.cleaningStatus === "validated").length;
      const failedCount = libraryMolecules.filter((m) => m.cleaningStatus === "failed").length;

      res.json({
        libraryId: library.id,
        name: library.name,
        status: library.status,
        moleculeCount: library.moleculeCount,
        scaffoldCount: library.scaffoldCount,
        cleaningProgress: {
          pending: pendingCount,
          validated: validatedCount,
          failed: failedCount,
          total: libraryMolecules.length,
        },
        isReady: library.status === "curated" && pendingCount === 0,
      });
    } catch (error) {
      console.error("Error fetching library status for agent:", error);
      res.status(500).json({ error: "Failed to fetch library status" });
    }
  });

  const agentValidateSchema = z.object({
    moleculeIds: z.array(z.string()).min(1),
    action: z.enum(["validate", "invalidate", "skip"]),
    reason: z.string().optional(),
  });

  app.post("/api/agent/libraries/:id/validate", requireAuth, async (req, res) => {
    try {
      const parsed = agentValidateSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }

      const library = await storage.getCuratedLibrary(req.params.id);
      if (!library) {
        return res.status(404).json({ error: "Library not found" });
      }

      const { moleculeIds, action, reason } = parsed.data;
      const newStatus = action === "validate" ? "validated" : action === "invalidate" ? "failed" : "pending";

      let updated = 0;
      for (const moleculeId of moleculeIds) {
        const libMols = await storage.getLibraryMolecules(library.id);
        const libMol = libMols.find((lm) => lm.moleculeId === moleculeId);
        if (libMol) {
          await storage.updateLibraryMolecule(libMol.id, { cleaningStatus: newStatus });
          updated++;
        }
      }

      if (reason) {
        await storage.createLibraryAnnotation({
          libraryId: library.id,
          annotationType: "agent_validation",
          annotationValue: reason,
          source: "agent",
          confidence: 1.0,
        });
      }

      res.json({
        libraryId: library.id,
        updated,
        action,
        message: `${updated} molecules marked as ${newStatus}`,
      });
    } catch (error) {
      console.error("Error validating library molecules for agent:", error);
      res.status(500).json({ error: "Failed to validate molecules" });
    }
  });

  const agentClassifySchema = z.object({
    moleculeId: z.string(),
    classification: z.string(),
    confidence: z.number().min(0).max(1).optional(),
  });

  app.post("/api/agent/libraries/:id/classify", requireAuth, async (req, res) => {
    try {
      const parsed = agentClassifySchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }

      const library = await storage.getCuratedLibrary(req.params.id);
      if (!library) {
        return res.status(404).json({ error: "Library not found" });
      }

      const { moleculeId, classification, confidence } = parsed.data;

      const annotation = await storage.createLibraryAnnotation({
        libraryId: library.id,
        moleculeId,
        annotationType: "classification",
        annotationValue: classification,
        source: "agent",
        confidence: confidence || 0.8,
      });

      res.status(201).json({
        libraryId: library.id,
        annotationId: annotation.id,
        moleculeId,
        classification,
        message: "Classification added",
      });
    } catch (error) {
      console.error("Error classifying molecule for agent:", error);
      res.status(500).json({ error: "Failed to classify molecule" });
    }
  });

  // ============================================
  // COMPUTE NODES ENDPOINTS
  // ============================================

  app.get("/api/compute-nodes", requireAuth, async (req, res) => {
    try {
      const { tier } = req.query;
      if (tier && typeof tier === "string") {
        const nodes = await storage.getComputeNodesByTier(tier);
        res.json(nodes);
      } else {
        const nodes = await storage.getComputeNodes();
        res.json(nodes);
      }
    } catch (error) {
      console.error("Error fetching compute nodes:", error);
      res.status(500).json({ error: "Failed to fetch compute nodes" });
    }
  });

  app.get("/api/compute-nodes/default/:tier", requireAuth, async (req, res) => {
    try {
      const node = await storage.getDefaultComputeNode(req.params.tier);
      if (!node) {
        return res.status(404).json({ error: "No default node found for this tier" });
      }
      res.json(node);
    } catch (error) {
      console.error("Error fetching default compute node:", error);
      res.status(500).json({ error: "Failed to fetch default compute node" });
    }
  });

  app.get("/api/compute-nodes/:id", requireAuth, async (req, res) => {
    try {
      const node = await storage.getComputeNode(req.params.id);
      if (!node) {
        return res.status(404).json({ error: "Compute node not found" });
      }
      const keyRegistrations = await storage.getNodeKeyRegistrations(node.id);
      res.json({ ...node, keyRegistrations });
    } catch (error) {
      console.error("Error fetching compute node:", error);
      res.status(500).json({ error: "Failed to fetch compute node" });
    }
  });

  app.post("/api/compute-nodes", requireAuth, async (req, res) => {
    try {
      const parsed = insertComputeNodeSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const node = await storage.createComputeNode(parsed.data);
      res.status(201).json(node);
    } catch (error) {
      console.error("Error creating compute node:", error);
      res.status(500).json({ error: "Failed to create compute node" });
    }
  });

  app.patch("/api/compute-nodes/:id", requireAuth, async (req, res) => {
    try {
      const parsed = insertComputeNodeSchema.partial().safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const node = await storage.updateComputeNode(req.params.id, parsed.data);
      if (!node) {
        return res.status(404).json({ error: "Compute node not found" });
      }
      res.json(node);
    } catch (error) {
      console.error("Error updating compute node:", error);
      res.status(500).json({ error: "Failed to update compute node" });
    }
  });

  app.delete("/api/compute-nodes/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteComputeNode(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting compute node:", error);
      res.status(500).json({ error: "Failed to delete compute node" });
    }
  });

  app.post("/api/compute-nodes/:id/register-key", requireAuth, async (req, res) => {
    try {
      const node = await storage.getComputeNode(req.params.id);
      if (!node) {
        return res.status(404).json({ error: "Compute node not found" });
      }

      const { sshKeyId } = req.body;
      if (!sshKeyId) {
        return res.status(400).json({ error: "sshKeyId is required" });
      }

      const sshKey = await storage.getUserSshKey(sshKeyId);
      if (!sshKey) {
        return res.status(404).json({ error: "SSH key not found" });
      }

      const registration = await storage.createNodeKeyRegistration({
        nodeId: node.id,
        sshKeyId: sshKey.id,
        status: "pending",
      });

      res.status(201).json({
        registration,
        message: "Key registration request created. Provisioning will be handled by automation.",
        sshCommand: `ssh user@${node.ipAddress}`,
      });
    } catch (error) {
      console.error("Error registering key:", error);
      res.status(500).json({ error: "Failed to register key" });
    }
  });

  // ============================================
  // USER SSH KEYS ENDPOINTS
  // ============================================

  app.get("/api/ssh-keys", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const keys = await storage.getUserSshKeys(userId);
      res.json(keys);
    } catch (error) {
      console.error("Error fetching SSH keys:", error);
      res.status(500).json({ error: "Failed to fetch SSH keys" });
    }
  });

  app.post("/api/ssh-keys", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertUserSshKeySchema.safeParse({ ...req.body, userId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const key = await storage.createUserSshKey(parsed.data);
      res.status(201).json(key);
    } catch (error) {
      console.error("Error creating SSH key:", error);
      res.status(500).json({ error: "Failed to create SSH key" });
    }
  });

  app.delete("/api/ssh-keys/:id", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const key = await storage.getUserSshKey(req.params.id);
      if (!key || key.userId !== userId) {
        return res.status(404).json({ error: "SSH key not found" });
      }
      await storage.deleteUserSshKey(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting SSH key:", error);
      res.status(500).json({ error: "Failed to delete SSH key" });
    }
  });

  // ============================================
  // SSH CONFIGS ENDPOINTS
  // ============================================

  app.get("/api/ssh-configs", requireAuth, async (req, res) => {
    try {
      const configs = await storage.getSshConfigs();
      res.json(configs);
    } catch (error) {
      console.error("Error fetching SSH configs:", error);
      res.status(500).json({ error: "Failed to fetch SSH configs" });
    }
  });

  app.get("/api/ssh-configs/:id", requireAuth, async (req, res) => {
    try {
      const config = await storage.getSshConfig(req.params.id);
      if (!config) {
        return res.status(404).json({ error: "SSH config not found" });
      }
      res.json(config);
    } catch (error) {
      console.error("Error fetching SSH config:", error);
      res.status(500).json({ error: "Failed to fetch SSH config" });
    }
  });

  app.post("/api/ssh-configs", requireAuth, async (req, res) => {
    try {
      const parsed = insertSshConfigSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const config = await storage.createSshConfig(parsed.data);
      res.status(201).json(config);
    } catch (error) {
      console.error("Error creating SSH config:", error);
      res.status(500).json({ error: "Failed to create SSH config" });
    }
  });

  app.patch("/api/ssh-configs/:id", requireAuth, async (req, res) => {
    try {
      const parsed = insertSshConfigSchema.partial().safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const config = await storage.updateSshConfig(req.params.id, parsed.data);
      if (!config) {
        return res.status(404).json({ error: "SSH config not found" });
      }
      res.json(config);
    } catch (error) {
      console.error("Error updating SSH config:", error);
      res.status(500).json({ error: "Failed to update SSH config" });
    }
  });

  app.delete("/api/ssh-configs/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteSshConfig(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting SSH config:", error);
      res.status(500).json({ error: "Failed to delete SSH config" });
    }
  });

  app.post("/api/ssh-configs/:id/test", requireAuth, async (req, res) => {
    try {
      const config = await storage.getSshConfig(req.params.id);
      if (!config) {
        return res.status(404).json({ error: "SSH config not found" });
      }
      // Stub: In production, this would actually test the SSH connection
      // For now, we simulate a successful connection
      const updatedConfig = await storage.updateSshConfigStatus(req.params.id, "connected", new Date());
      res.json({ 
        success: true, 
        message: "Connection test successful",
        config: updatedConfig 
      });
    } catch (error) {
      console.error("Error testing SSH connection:", error);
      res.status(500).json({ error: "Failed to test connection" });
    }
  });

  // ============================================
  // COMPANIES ENDPOINTS
  // ============================================

  app.get("/api/companies", requireAuth, async (req, res) => {
    try {
      const companiesList = await storage.getCompanies();
      res.json(companiesList);
    } catch (error) {
      console.error("Error fetching companies:", error);
      res.status(500).json({ error: "Failed to fetch companies" });
    }
  });

  app.get("/api/companies/:id", requireAuth, async (req, res) => {
    try {
      const company = await storage.getCompany(req.params.id);
      if (!company) {
        return res.status(404).json({ error: "Company not found" });
      }
      res.json(company);
    } catch (error) {
      console.error("Error fetching company:", error);
      res.status(500).json({ error: "Failed to fetch company" });
    }
  });

  app.post("/api/companies", requireAuth, async (req, res) => {
    try {
      const parsed = insertCompanySchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const company = await storage.createCompany(parsed.data);
      res.status(201).json(company);
    } catch (error) {
      console.error("Error creating company:", error);
      res.status(500).json({ error: "Failed to create company" });
    }
  });

  app.patch("/api/companies/:id", requireAuth, async (req, res) => {
    try {
      const parsed = insertCompanySchema.partial().safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const company = await storage.updateCompany(req.params.id, parsed.data);
      if (!company) {
        return res.status(404).json({ error: "Company not found" });
      }
      res.json(company);
    } catch (error) {
      console.error("Error updating company:", error);
      res.status(500).json({ error: "Failed to update company" });
    }
  });

  app.delete("/api/companies/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteCompany(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting company:", error);
      res.status(500).json({ error: "Failed to delete company" });
    }
  });

  // ============================================
  // USAGE METERS ENDPOINTS
  // ============================================

  app.get("/api/usage", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const { projectId, campaignId } = req.query;
      
      const userProjects = await storage.getProjects(userId);
      const userProjectIds = userProjects.map(p => p.id);
      
      if (projectId && !userProjectIds.includes(projectId as string)) {
        return res.status(403).json({ error: "Access denied to this project" });
      }
      
      const meters = await storage.getUsageMeters({
        projectId: projectId as string | undefined,
        campaignId: campaignId as string | undefined,
      });
      
      const filteredMeters = meters.filter(m => 
        m.projectId && userProjectIds.includes(m.projectId)
      );
      
      res.json(filteredMeters);
    } catch (error) {
      console.error("Error fetching usage meters:", error);
      res.status(500).json({ error: "Failed to fetch usage" });
    }
  });

  app.get("/api/usage/summary/:projectId", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const project = await storage.getProject(req.params.projectId);
      
      if (!project || project.ownerId !== userId) {
        return res.status(403).json({ error: "Access denied to this project" });
      }
      
      const summary = await storage.getUsageSummary(req.params.projectId);
      res.json(summary);
    } catch (error) {
      console.error("Error fetching usage summary:", error);
      res.status(500).json({ error: "Failed to fetch usage summary" });
    }
  });

  // ============================================
  // CREDITS ENDPOINTS (STUB FOR FUTURE BILLING)
  // ============================================

  app.get("/api/credits/wallet", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      let wallet = await storage.getCreditWallet(userId, "user");
      
      if (!wallet) {
        wallet = await storage.createCreditWallet({
          ownerType: "user",
          ownerId: userId,
          balance: 0,
          currency: "CREDITS",
        });
      }
      
      res.json(wallet);
    } catch (error) {
      console.error("Error fetching credit wallet:", error);
      res.status(500).json({ error: "Failed to fetch wallet" });
    }
  });

  app.get("/api/credits/transactions", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const wallet = await storage.getCreditWallet(userId, "user");
      
      if (!wallet) {
        return res.json([]);
      }
      
      const transactions = await storage.getCreditTransactions(wallet.id);
      res.json(transactions);
    } catch (error) {
      console.error("Error fetching credit transactions:", error);
      res.status(500).json({ error: "Failed to fetch transactions" });
    }
  });

  app.post("/api/credits/purchase", requireAuth, async (req, res) => {
    res.status(501).json({
      error: "Not implemented",
      message: "Credit purchase is not available in v0. This endpoint is a placeholder for future billing integration.",
    });
  });

  app.post("/api/credits/apply", requireAuth, async (req, res) => {
    res.status(501).json({
      error: "Not implemented",
      message: "Credit application is not available in v0. This endpoint is a placeholder for future billing integration.",
    });
  });

  // ============================================
  // TARGET VARIANTS ENDPOINTS
  // ============================================

  app.get("/api/targets/:targetId/variants", requireAuth, async (req, res) => {
    try {
      const variants = await storage.getTargetVariants(req.params.targetId);
      res.json(variants);
    } catch (error) {
      console.error("Error fetching target variants:", error);
      res.status(500).json({ error: "Failed to fetch variants" });
    }
  });

  app.post("/api/targets/:targetId/variants", requireAuth, async (req, res) => {
    try {
      const parsed = insertTargetVariantSchema.safeParse({ ...req.body, targetId: req.params.targetId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const variant = await storage.createTargetVariant(parsed.data);
      res.status(201).json(variant);
    } catch (error) {
      console.error("Error creating target variant:", error);
      res.status(500).json({ error: "Failed to create variant" });
    }
  });

  app.patch("/api/variants/:id", requireAuth, async (req, res) => {
    try {
      const variant = await storage.updateTargetVariant(req.params.id, req.body);
      if (!variant) {
        return res.status(404).json({ error: "Variant not found" });
      }
      res.json(variant);
    } catch (error) {
      console.error("Error updating variant:", error);
      res.status(500).json({ error: "Failed to update variant" });
    }
  });

  app.delete("/api/variants/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteTargetVariant(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting variant:", error);
      res.status(500).json({ error: "Failed to delete variant" });
    }
  });

  // ============================================
  // DISEASE CONTEXT SIGNALS ENDPOINTS
  // ============================================

  app.get("/api/targets/:targetId/signals", requireAuth, async (req, res) => {
    try {
      const signals = await storage.getDiseaseContextSignals(req.params.targetId);
      res.json(signals);
    } catch (error) {
      console.error("Error fetching disease context signals:", error);
      res.status(500).json({ error: "Failed to fetch signals" });
    }
  });

  app.post("/api/targets/:targetId/signals", requireAuth, async (req, res) => {
    try {
      const parsed = insertDiseaseContextSignalSchema.safeParse({ ...req.body, targetId: req.params.targetId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const signal = await storage.createDiseaseContextSignal(parsed.data);
      res.status(201).json(signal);
    } catch (error) {
      console.error("Error creating disease context signal:", error);
      res.status(500).json({ error: "Failed to create signal" });
    }
  });

  // ============================================
  // PROGRAMS ENDPOINTS
  // ============================================

  app.get("/api/programs", requireAuth, async (req, res) => {
    try {
      const { projectId } = req.query;
      const programs = await storage.getPrograms(projectId as string | undefined);
      res.json(programs);
    } catch (error) {
      console.error("Error fetching programs:", error);
      res.status(500).json({ error: "Failed to fetch programs" });
    }
  });

  app.get("/api/programs/:id", requireAuth, async (req, res) => {
    try {
      const program = await storage.getProgram(req.params.id);
      if (!program) {
        return res.status(404).json({ error: "Program not found" });
      }
      res.json(program);
    } catch (error) {
      console.error("Error fetching program:", error);
      res.status(500).json({ error: "Failed to fetch program" });
    }
  });

  app.post("/api/programs", requireAuth, async (req, res) => {
    try {
      const parsed = insertProgramSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const program = await storage.createProgram(parsed.data);
      res.status(201).json(program);
    } catch (error) {
      console.error("Error creating program:", error);
      res.status(500).json({ error: "Failed to create program" });
    }
  });

  app.patch("/api/programs/:id", requireAuth, async (req, res) => {
    try {
      const program = await storage.updateProgram(req.params.id, req.body);
      if (!program) {
        return res.status(404).json({ error: "Program not found" });
      }
      res.json(program);
    } catch (error) {
      console.error("Error updating program:", error);
      res.status(500).json({ error: "Failed to update program" });
    }
  });

  app.delete("/api/programs/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteProgram(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting program:", error);
      res.status(500).json({ error: "Failed to delete program" });
    }
  });

  // ============================================
  // ORACLE VERSIONS ENDPOINTS
  // ============================================

  app.get("/api/oracle-versions", requireAuth, async (req, res) => {
    try {
      const versions = await storage.getOracleVersions();
      res.json(versions);
    } catch (error) {
      console.error("Error fetching oracle versions:", error);
      res.status(500).json({ error: "Failed to fetch oracle versions" });
    }
  });

  app.get("/api/oracle-versions/:id", requireAuth, async (req, res) => {
    try {
      const version = await storage.getOracleVersion(req.params.id);
      if (!version) {
        return res.status(404).json({ error: "Oracle version not found" });
      }
      res.json(version);
    } catch (error) {
      console.error("Error fetching oracle version:", error);
      res.status(500).json({ error: "Failed to fetch oracle version" });
    }
  });

  app.post("/api/oracle-versions", requireAuth, async (req, res) => {
    try {
      const parsed = insertOracleVersionSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const version = await storage.createOracleVersion(parsed.data);
      res.status(201).json(version);
    } catch (error) {
      console.error("Error creating oracle version:", error);
      res.status(500).json({ error: "Failed to create oracle version" });
    }
  });

  // ============================================
  // ASSAYS ENDPOINTS
  // ============================================

  app.get("/api/assays", requireAuth, async (req, res) => {
    try {
      const { targetId, companyId } = req.query;
      const filters: { targetId?: string; companyId?: string } = {};
      if (targetId) filters.targetId = targetId as string;
      if (companyId) filters.companyId = companyId as string;
      const assaysList = await storage.getAssays(Object.keys(filters).length > 0 ? filters : undefined);
      res.json(assaysList);
    } catch (error) {
      console.error("Error fetching assays:", error);
      res.status(500).json({ error: "Failed to fetch assays" });
    }
  });

  app.get("/api/assays/:id", requireAuth, async (req, res) => {
    try {
      const assay = await storage.getAssay(req.params.id);
      if (!assay) {
        return res.status(404).json({ error: "Assay not found" });
      }
      res.json(assay);
    } catch (error) {
      console.error("Error fetching assay:", error);
      res.status(500).json({ error: "Failed to fetch assay" });
    }
  });

  app.post("/api/assays", requireAuth, async (req, res) => {
    try {
      const parsed = insertAssaySchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const assay = await storage.createAssay(parsed.data);
      res.status(201).json(assay);
    } catch (error) {
      console.error("Error creating assay:", error);
      res.status(500).json({ error: "Failed to create assay" });
    }
  });

  app.patch("/api/assays/:id", requireAuth, async (req, res) => {
    try {
      const assay = await storage.updateAssay(req.params.id, req.body);
      if (!assay) {
        return res.status(404).json({ error: "Assay not found" });
      }
      res.json(assay);
    } catch (error) {
      console.error("Error updating assay:", error);
      res.status(500).json({ error: "Failed to update assay" });
    }
  });

  app.delete("/api/assays/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteAssay(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting assay:", error);
      res.status(500).json({ error: "Failed to delete assay" });
    }
  });

  app.get("/api/assays/:id/details", requireAuth, async (req, res) => {
    try {
      const assay = await storage.getAssayWithResultsCount(req.params.id);
      if (!assay) {
        return res.status(404).json({ error: "Assay not found" });
      }
      res.json(assay);
    } catch (error) {
      console.error("Error fetching assay details:", error);
      res.status(500).json({ error: "Failed to fetch assay details" });
    }
  });

  app.get("/api/assays/:assayId/results", requireAuth, async (req, res) => {
    try {
      const { moleculeId } = req.query;
      const results = await storage.getAssayResultsWithMolecules(req.params.assayId, moleculeId as string | undefined);
      res.json(results);
    } catch (error) {
      console.error("Error fetching assay results with molecules:", error);
      res.status(500).json({ error: "Failed to fetch assay results" });
    }
  });

  app.post("/api/assays/:assayId/upload", requireAuth, async (req, res) => {
    try {
      const { rows, campaignId } = req.body;
      
      if (!Array.isArray(rows)) {
        return res.status(400).json({ error: "rows must be an array" });
      }
      
      const assay = await storage.getAssay(req.params.assayId);
      if (!assay) {
        return res.status(404).json({ error: "Assay not found" });
      }
      
      const importResults = {
        imported: 0,
        moleculesUpdated: 0,
        moleculesCreated: 0,
        errors: [] as { row: number; error: string }[],
      };
      
      const resultsToCreate = [];
      
      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        try {
          let moleculeId = row.molecule_id;
          
          if (!moleculeId && row.smiles) {
            const existing = await storage.getMoleculeBySmiles(row.smiles);
            if (existing) {
              moleculeId = existing.id;
              importResults.moleculesUpdated++;
            } else {
              const newMol = await storage.createMolecule({ smiles: row.smiles, name: row.name });
              moleculeId = newMol.id;
              importResults.moleculesCreated++;
            }
          }
          
          if (!moleculeId) {
            importResults.errors.push({ row: i + 1, error: "No molecule_id or smiles provided" });
            continue;
          }
          
          const value = parseFloat(row.value);
          if (isNaN(value)) {
            importResults.errors.push({ row: i + 1, error: "Invalid value" });
            continue;
          }
          
          const { molecule_id, smiles, name, ...extraFields } = row;
          
          resultsToCreate.push({
            assayId: req.params.assayId,
            campaignId: campaignId || null,
            moleculeId,
            value,
            units: row.units || assay.units,
            concentration: row.concentration ? parseFloat(row.concentration) : null,
            outcomeLabel: row.outcome_label || null,
            replicateId: row.replicate_id || null,
            metadata: Object.keys(extraFields).length > 0 ? extraFields : null,
          });
          
        } catch (err: any) {
          importResults.errors.push({ row: i + 1, error: err.message || "Unknown error" });
        }
      }
      
      if (resultsToCreate.length > 0) {
        await storage.bulkCreateAssayResults(resultsToCreate as any);
        importResults.imported = resultsToCreate.length;
      }
      
      res.status(201).json(importResults);
    } catch (error) {
      console.error("Error uploading assay data:", error);
      res.status(500).json({ error: "Failed to upload assay data" });
    }
  });

  // ============================================
  // HIT TRIAGE ENDPOINTS
  // ============================================

  app.get("/api/campaigns/:campaignId/hits", requireAuth, async (req, res) => {
    try {
      const { minOracleScore, maxOracleScore, maxSynthesisComplexity, ipSafeOnly, hasAssayData } = req.query;
      
      const filters: {
        minOracleScore?: number;
        maxOracleScore?: number;
        maxSynthesisComplexity?: number;
        ipSafeOnly?: boolean;
        hasAssayData?: boolean;
      } = {};
      
      if (minOracleScore) filters.minOracleScore = parseFloat(minOracleScore as string);
      if (maxOracleScore) filters.maxOracleScore = parseFloat(maxOracleScore as string);
      if (maxSynthesisComplexity) filters.maxSynthesisComplexity = parseFloat(maxSynthesisComplexity as string);
      if (ipSafeOnly === "true") filters.ipSafeOnly = true;
      if (hasAssayData === "true" || hasAssayData === "false") filters.hasAssayData = hasAssayData === "true";
      
      const hits = await storage.getHitCandidates(req.params.campaignId, Object.keys(filters).length > 0 ? filters : undefined);
      res.json(hits);
    } catch (error) {
      console.error("Error fetching hit candidates:", error);
      res.status(500).json({ error: "Failed to fetch hit candidates" });
    }
  });

  // ============================================
  // SAR (STRUCTURE-ACTIVITY RELATIONSHIP) ENDPOINTS
  // ============================================

  app.get("/api/campaigns/:campaignId/sar/series", requireAuth, async (req, res) => {
    try {
      const series = await storage.getSarSeries(req.params.campaignId);
      res.json(series);
    } catch (error) {
      console.error("Error fetching SAR series:", error);
      res.status(500).json({ error: "Failed to fetch SAR series" });
    }
  });

  app.get("/api/campaigns/:campaignId/sar/molecule/:moleculeId", requireAuth, async (req, res) => {
    try {
      const details = await storage.getSarMoleculeDetails(req.params.campaignId, req.params.moleculeId);
      if (!details) {
        return res.status(404).json({ error: "Molecule not found" });
      }
      res.json(details);
    } catch (error) {
      console.error("Error fetching SAR molecule details:", error);
      res.status(500).json({ error: "Failed to fetch SAR molecule details" });
    }
  });

  // ============================================
  // MULTI-TARGET SAR ENDPOINTS
  // ============================================

  app.get("/api/campaigns/:campaignId/sar/multi-target", requireAuth, async (req, res) => {
    try {
      const data = await storage.getMultiTargetSar(req.params.campaignId);
      res.json(data);
    } catch (error) {
      console.error("Error fetching multi-target SAR:", error);
      res.status(500).json({ error: "Failed to fetch multi-target SAR data" });
    }
  });

  // ============================================
  // ASSAY PANELS ENDPOINTS
  // ============================================

  app.get("/api/assay-panels", requireAuth, async (req, res) => {
    try {
      const { campaignId } = req.query;
      const panels = await storage.getAssayPanels(campaignId as string | undefined);
      res.json(panels);
    } catch (error) {
      console.error("Error fetching assay panels:", error);
      res.status(500).json({ error: "Failed to fetch assay panels" });
    }
  });

  app.get("/api/assay-panels/:id", requireAuth, async (req, res) => {
    try {
      const panel = await storage.getAssayPanel(req.params.id);
      if (!panel) {
        return res.status(404).json({ error: "Assay panel not found" });
      }
      res.json(panel);
    } catch (error) {
      console.error("Error fetching assay panel:", error);
      res.status(500).json({ error: "Failed to fetch assay panel" });
    }
  });

  app.post("/api/assay-panels", requireAuth, async (req, res) => {
    try {
      const { targets, ...panelData } = req.body;
      const parsed = insertAssayPanelSchema.safeParse(panelData);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const panel = await storage.createAssayPanel(parsed.data, targets || []);
      res.status(201).json(panel);
    } catch (error) {
      console.error("Error creating assay panel:", error);
      res.status(500).json({ error: "Failed to create assay panel" });
    }
  });

  app.delete("/api/assay-panels/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteAssayPanel(req.params.id);
      res.status(204).end();
    } catch (error) {
      console.error("Error deleting assay panel:", error);
      res.status(500).json({ error: "Failed to delete assay panel" });
    }
  });

  app.get("/api/assay-panels/:panelId/results", requireAuth, async (req, res) => {
    try {
      const matrix = await storage.getAssayPanelResults(req.params.panelId);
      res.json(matrix);
    } catch (error) {
      console.error("Error fetching assay panel results:", error);
      res.status(500).json({ error: "Failed to fetch assay panel results" });
    }
  });

  app.post("/api/assay-panels/:panelId/upload", requireAuth, async (req, res) => {
    try {
      const { results } = req.body;
      if (!Array.isArray(results)) {
        return res.status(400).json({ error: "Results must be an array" });
      }
      const created = await storage.uploadAssayPanelResults(req.params.panelId, results);
      res.status(201).json(created);
    } catch (error) {
      console.error("Error uploading assay panel results:", error);
      res.status(500).json({ error: "Failed to upload assay panel results" });
    }
  });

  // ============================================
  // MOA (MECHANISM OF ACTION) GRAPH ENDPOINTS
  // ============================================

  app.get("/api/moa/graph", requireAuth, async (req, res) => {
    try {
      const { campaignId } = req.query;
      const graph = await storage.getMoaGraph(campaignId as string | undefined);
      res.json(graph);
    } catch (error) {
      console.error("Error fetching MoA graph:", error);
      res.status(500).json({ error: "Failed to fetch MoA graph" });
    }
  });

  app.get("/api/moa/target/:targetId", requireAuth, async (req, res) => {
    try {
      const subgraph = await storage.getMoaSubgraph(req.params.targetId);
      res.json(subgraph);
    } catch (error) {
      console.error("Error fetching MoA subgraph:", error);
      res.status(500).json({ error: "Failed to fetch MoA subgraph" });
    }
  });

  app.post("/api/moa/nodes", requireAuth, async (req, res) => {
    try {
      const parsed = insertMoaNodeSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const node = await storage.createMoaNode(parsed.data);
      res.status(201).json(node);
    } catch (error) {
      console.error("Error creating MoA node:", error);
      res.status(500).json({ error: "Failed to create MoA node" });
    }
  });

  app.post("/api/moa/edges", requireAuth, async (req, res) => {
    try {
      const parsed = insertMoaEdgeSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const edge = await storage.createMoaEdge(parsed.data);
      res.status(201).json(edge);
    } catch (error) {
      console.error("Error creating MoA edge:", error);
      res.status(500).json({ error: "Failed to create MoA edge" });
    }
  });

  // ============================================
  // PIPELINE TEMPLATES ENDPOINTS
  // ============================================

  app.get("/api/pipeline-templates", requireAuth, async (req, res) => {
    try {
      const { domain } = req.query;
      const templates = await storage.getPipelineTemplates(domain as string | undefined);
      res.json(templates);
    } catch (error) {
      console.error("Error fetching pipeline templates:", error);
      res.status(500).json({ error: "Failed to fetch pipeline templates" });
    }
  });

  app.get("/api/pipeline-templates/:id", requireAuth, async (req, res) => {
    try {
      const template = await storage.getPipelineTemplate(req.params.id);
      if (!template) {
        return res.status(404).json({ error: "Template not found" });
      }
      res.json(template);
    } catch (error) {
      console.error("Error fetching pipeline template:", error);
      res.status(500).json({ error: "Failed to fetch pipeline template" });
    }
  });

  app.post("/api/pipeline-templates", requireAuth, async (req, res) => {
    try {
      const { targets, ...templateData } = req.body;
      const template = await storage.createPipelineTemplate(templateData, targets || []);
      res.status(201).json(template);
    } catch (error) {
      console.error("Error creating pipeline template:", error);
      res.status(500).json({ error: "Failed to create pipeline template" });
    }
  });

  app.delete("/api/pipeline-templates/:id", requireAuth, async (req, res) => {
    try {
      const template = await storage.getPipelineTemplate(req.params.id);
      if (!template) {
        return res.status(404).json({ error: "Template not found" });
      }
      if (template.isBuiltIn) {
        return res.status(403).json({ error: "Cannot delete built-in templates" });
      }
      await storage.deletePipelineTemplate(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting pipeline template:", error);
      res.status(500).json({ error: "Failed to delete pipeline template" });
    }
  });

  app.post("/api/pipeline-templates/seed", requireAuth, async (req, res) => {
    try {
      await storage.seedBuiltInTemplates();
      const templates = await storage.getPipelineTemplates();
      res.json({ message: "Built-in templates seeded successfully", templates });
    } catch (error) {
      console.error("Error seeding templates:", error);
      res.status(500).json({ error: "Failed to seed templates" });
    }
  });

  // ============================================
  // EXPERIMENT RECOMMENDATIONS ENDPOINTS
  // ============================================

  app.get("/api/campaigns/:campaignId/recommendations", requireAuth, async (req, res) => {
    try {
      const recommendations = await storage.getExperimentRecommendations(req.params.campaignId);
      res.json(recommendations);
    } catch (error) {
      console.error("Error fetching experiment recommendations:", error);
      res.status(500).json({ error: "Failed to fetch recommendations" });
    }
  });

  app.post("/api/campaigns/:campaignId/recommendations", requireAuth, async (req, res) => {
    try {
      const parsed = insertExperimentRecommendationSchema.safeParse({ ...req.body, campaignId: req.params.campaignId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const rec = await storage.createExperimentRecommendation(parsed.data);
      res.status(201).json(rec);
    } catch (error) {
      console.error("Error creating experiment recommendation:", error);
      res.status(500).json({ error: "Failed to create recommendation" });
    }
  });

  app.patch("/api/recommendations/:id", requireAuth, async (req, res) => {
    try {
      const rec = await storage.updateExperimentRecommendation(req.params.id, req.body);
      if (!rec) {
        return res.status(404).json({ error: "Recommendation not found" });
      }
      res.json(rec);
    } catch (error) {
      console.error("Error updating recommendation:", error);
      res.status(500).json({ error: "Failed to update recommendation" });
    }
  });

  // ============================================
  // ASSAY RESULTS ENDPOINTS
  // ============================================

  app.get("/api/assay-results", requireAuth, async (req, res) => {
    try {
      const { assayId, campaignId } = req.query;
      const results = await storage.getAssayResults(assayId as string | undefined, campaignId as string | undefined);
      res.json(results);
    } catch (error) {
      console.error("Error fetching assay results:", error);
      res.status(500).json({ error: "Failed to fetch assay results" });
    }
  });

  app.post("/api/assay-results", requireAuth, async (req, res) => {
    try {
      const parsed = insertAssayResultSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const result = await storage.createAssayResult(parsed.data);
      res.status(201).json(result);
    } catch (error) {
      console.error("Error creating assay result:", error);
      res.status(500).json({ error: "Failed to create assay result" });
    }
  });

  app.post("/api/assay-results/bulk", requireAuth, async (req, res) => {
    try {
      const { results } = req.body;
      if (!Array.isArray(results)) {
        return res.status(400).json({ error: "Results must be an array" });
      }
      const created = await storage.bulkCreateAssayResults(results);
      res.status(201).json(created);
    } catch (error) {
      console.error("Error bulk creating assay results:", error);
      res.status(500).json({ error: "Failed to create assay results" });
    }
  });

  // ============================================
  // LITERATURE ANNOTATIONS ENDPOINTS
  // ============================================

  app.get("/api/literature-annotations", requireAuth, async (req, res) => {
    try {
      const { targetId, moleculeId } = req.query;
      const annotations = await storage.getLiteratureAnnotations(targetId as string | undefined, moleculeId as string | undefined);
      res.json(annotations);
    } catch (error) {
      console.error("Error fetching literature annotations:", error);
      res.status(500).json({ error: "Failed to fetch literature annotations" });
    }
  });

  app.post("/api/literature-annotations", requireAuth, async (req, res) => {
    try {
      const parsed = insertLiteratureAnnotationSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const annotation = await storage.createLiteratureAnnotation(parsed.data);
      res.status(201).json(annotation);
    } catch (error) {
      console.error("Error creating literature annotation:", error);
      res.status(500).json({ error: "Failed to create literature annotation" });
    }
  });

  // ============================================
  // ORGANIZATIONS ENDPOINTS
  // ============================================

  app.get("/api/organizations", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const memberships = await storage.getOrgMembersByUser(userId);
      const orgIds = memberships.map(m => m.organizationId);
      const orgs = await storage.getOrganizations();
      const userOrgs = orgs.filter(o => orgIds.includes(o.id));
      res.json(userOrgs);
    } catch (error) {
      console.error("Error fetching organizations:", error);
      res.status(500).json({ error: "Failed to fetch organizations" });
    }
  });

  app.get("/api/organizations/:id", requireAuth, async (req, res) => {
    try {
      const org = await storage.getOrganization(req.params.id);
      if (!org) {
        return res.status(404).json({ error: "Organization not found" });
      }
      const members = await storage.getOrgMembers(org.id);
      res.json({ ...org, members });
    } catch (error) {
      console.error("Error fetching organization:", error);
      res.status(500).json({ error: "Failed to fetch organization" });
    }
  });

  app.post("/api/organizations", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertOrganizationSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const org = await storage.createOrganization(parsed.data);
      await storage.createOrgMember({ userId, organizationId: org.id, role: "admin" });
      res.status(201).json(org);
    } catch (error) {
      console.error("Error creating organization:", error);
      res.status(500).json({ error: "Failed to create organization" });
    }
  });

  app.patch("/api/organizations/:id", requireAuth, async (req, res) => {
    try {
      const org = await storage.updateOrganization(req.params.id, req.body);
      if (!org) {
        return res.status(404).json({ error: "Organization not found" });
      }
      res.json(org);
    } catch (error) {
      console.error("Error updating organization:", error);
      res.status(500).json({ error: "Failed to update organization" });
    }
  });

  // ============================================
  // ORGANIZATION MEMBERS ENDPOINTS
  // ============================================

  app.post("/api/organizations/:orgId/members", requireAuth, async (req, res) => {
    try {
      const parsed = insertOrgMemberSchema.safeParse({ ...req.body, organizationId: req.params.orgId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const member = await storage.createOrgMember(parsed.data);
      res.status(201).json(member);
    } catch (error) {
      console.error("Error adding org member:", error);
      res.status(500).json({ error: "Failed to add member" });
    }
  });

  app.patch("/api/org-members/:id", requireAuth, async (req, res) => {
    try {
      const member = await storage.updateOrgMember(req.params.id, req.body);
      if (!member) {
        return res.status(404).json({ error: "Member not found" });
      }
      res.json(member);
    } catch (error) {
      console.error("Error updating org member:", error);
      res.status(500).json({ error: "Failed to update member" });
    }
  });

  app.delete("/api/org-members/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteOrgMember(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error removing org member:", error);
      res.status(500).json({ error: "Failed to remove member" });
    }
  });

  // ============================================
  // SHARED ASSETS ENDPOINTS
  // ============================================

  app.get("/api/shared-assets", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const memberships = await storage.getOrgMembersByUser(userId);
      const allAssets = [];
      for (const m of memberships) {
        const assets = await storage.getSharedAssets(m.organizationId);
        allAssets.push(...assets);
      }
      res.json(allAssets);
    } catch (error) {
      console.error("Error fetching shared assets:", error);
      res.status(500).json({ error: "Failed to fetch shared assets" });
    }
  });

  app.post("/api/shared-assets", requireAuth, async (req, res) => {
    try {
      const parsed = insertSharedAssetSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const asset = await storage.createSharedAsset(parsed.data);
      res.status(201).json(asset);
    } catch (error) {
      console.error("Error sharing asset:", error);
      res.status(500).json({ error: "Failed to share asset" });
    }
  });

  app.delete("/api/shared-assets/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteSharedAsset(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error removing shared asset:", error);
      res.status(500).json({ error: "Failed to remove shared asset" });
    }
  });

  // ============================================
  // AGENT ENDPOINTS FOR ADVANCED FEATURES
  // ============================================

  app.get("/api/agent/variants/:targetId", async (req, res) => {
    try {
      const variants = await storage.getTargetVariants(req.params.targetId);
      res.json(variants.map(v => ({ id: v.id, variantName: v.variantName, sequence: v.sequence })));
    } catch (error) {
      console.error("Error fetching variants for agent:", error);
      res.status(500).json({ error: "Failed to fetch variants" });
    }
  });

  app.get("/api/agent/programs", async (req, res) => {
    try {
      const programs = await storage.getPrograms();
      res.json(programs.map(p => ({ id: p.id, name: p.name, diseaseArea: p.diseaseArea })));
    } catch (error) {
      console.error("Error fetching programs for agent:", error);
      res.status(500).json({ error: "Failed to fetch programs" });
    }
  });

  app.get("/api/agent/oracle-versions", async (req, res) => {
    try {
      const versions = await storage.getOracleVersions();
      res.json(versions.map(v => ({ id: v.id, name: v.name, componentVersions: v.componentVersions })));
    } catch (error) {
      console.error("Error fetching oracle versions for agent:", error);
      res.status(500).json({ error: "Failed to fetch oracle versions" });
    }
  });

  app.get("/api/agent/assays", async (req, res) => {
    try {
      const assays = await storage.getAssays();
      res.json(assays.map(a => ({ id: a.id, name: a.name, type: a.type, estimatedCost: a.estimatedCost })));
    } catch (error) {
      console.error("Error fetching assays for agent:", error);
      res.status(500).json({ error: "Failed to fetch assays" });
    }
  });

  app.get("/api/agent/campaigns/:campaignId/recommendations", async (req, res) => {
    try {
      const recommendations = await storage.getExperimentRecommendations(req.params.campaignId);
      res.json(recommendations.map(r => ({
        id: r.id,
        assayId: r.assayId,
        priorityScore: r.priorityScore,
        estimatedCost: r.estimatedCost,
        status: r.status,
      })));
    } catch (error) {
      console.error("Error fetching recommendations for agent:", error);
      res.status(500).json({ error: "Failed to fetch recommendations" });
    }
  });

  app.post("/api/agent/assay-results", async (req, res) => {
    try {
      const { results } = req.body;
      if (!Array.isArray(results)) {
        return res.status(400).json({ error: "Results must be an array" });
      }
      const created = await storage.bulkCreateAssayResults(results);
      res.status(201).json({ count: created.length });
    } catch (error) {
      console.error("Error creating assay results from agent:", error);
      res.status(500).json({ error: "Failed to create assay results" });
    }
  });

  app.get("/api/agent/literature/:targetId", async (req, res) => {
    try {
      const annotations = await storage.getLiteratureAnnotations(req.params.targetId);
      res.json(annotations.map(a => ({
        id: a.id,
        source: a.source,
        relevanceScore: a.relevanceScore,
        summary: a.summary,
        url: a.url,
      })));
    } catch (error) {
      console.error("Error fetching literature for agent:", error);
      res.status(500).json({ error: "Failed to fetch literature" });
    }
  });

  app.post("/api/agent/literature", async (req, res) => {
    try {
      const parsed = insertLiteratureAnnotationSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const annotation = await storage.createLiteratureAnnotation(parsed.data);
      res.status(201).json({ id: annotation.id });
    } catch (error) {
      console.error("Error creating literature annotation from agent:", error);
      res.status(500).json({ error: "Failed to create literature annotation" });
    }
  });

  app.post("/api/agent/recommendations/:id/approve", async (req, res) => {
    try {
      const rec = await storage.updateExperimentRecommendation(req.params.id, { status: "approved" });
      if (!rec) {
        return res.status(404).json({ error: "Recommendation not found" });
      }
      res.json({ status: "approved" });
    } catch (error) {
      console.error("Error approving recommendation:", error);
      res.status(500).json({ error: "Failed to approve recommendation" });
    }
  });

  app.post("/api/agent/recommendations/:id/reject", async (req, res) => {
    try {
      const rec = await storage.updateExperimentRecommendation(req.params.id, { status: "rejected" });
      if (!rec) {
        return res.status(404).json({ error: "Recommendation not found" });
      }
      res.json({ status: "rejected" });
    } catch (error) {
      console.error("Error rejecting recommendation:", error);
      res.status(500).json({ error: "Failed to reject recommendation" });
    }
  });

  // ============================================
  // MATERIALS DISCOVERY ENDPOINTS
  // ============================================

  app.get("/api/materials", requireAuth, async (req, res) => {
    try {
      const type = req.query.type as string | undefined;
      const materials = await storage.getMaterialEntities(type);
      res.json(materials);
    } catch (error) {
      console.error("Error fetching materials:", error);
      res.status(500).json({ error: "Failed to fetch materials" });
    }
  });

  app.get("/api/materials/:id", requireAuth, async (req, res) => {
    try {
      const material = await storage.getMaterialEntity(req.params.id);
      if (!material) {
        return res.status(404).json({ error: "Material not found" });
      }
      const properties = await storage.getMaterialProperties(material.id);
      res.json({ ...material, properties });
    } catch (error) {
      console.error("Error fetching material:", error);
      res.status(500).json({ error: "Failed to fetch material" });
    }
  });

  app.post("/api/materials", requireAuth, async (req, res) => {
    try {
      const parsed = insertMaterialEntitySchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const material = await storage.createMaterialEntity(parsed.data);
      res.status(201).json(material);
    } catch (error) {
      console.error("Error creating material:", error);
      res.status(500).json({ error: "Failed to create material" });
    }
  });

  app.patch("/api/materials/:id", requireAuth, async (req, res) => {
    try {
      const material = await storage.updateMaterialEntity(req.params.id, req.body);
      if (!material) {
        return res.status(404).json({ error: "Material not found" });
      }
      res.json(material);
    } catch (error) {
      console.error("Error updating material:", error);
      res.status(500).json({ error: "Failed to update material" });
    }
  });

  app.delete("/api/materials/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteMaterialEntity(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting material:", error);
      res.status(500).json({ error: "Failed to delete material" });
    }
  });

  app.post("/api/materials/:id/properties", requireAuth, async (req, res) => {
    try {
      const parsed = insertMaterialPropertySchema.safeParse({ ...req.body, materialId: req.params.id });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const property = await storage.createMaterialProperty(parsed.data);
      res.status(201).json(property);
    } catch (error) {
      console.error("Error creating material property:", error);
      res.status(500).json({ error: "Failed to create material property" });
    }
  });

  app.get("/api/materials/:id/variants", requireAuth, async (req, res) => {
    try {
      const variants = await storage.getMaterialVariants(req.params.id);
      res.json(variants);
    } catch (error) {
      console.error("Error fetching material variants:", error);
      res.status(500).json({ error: "Failed to fetch material variants" });
    }
  });

  app.get("/api/material-variants/:id", requireAuth, async (req, res) => {
    try {
      const variant = await storage.getMaterialVariant(req.params.id);
      if (!variant) {
        return res.status(404).json({ error: "Material variant not found" });
      }
      res.json(variant);
    } catch (error) {
      console.error("Error fetching material variant:", error);
      res.status(500).json({ error: "Failed to fetch material variant" });
    }
  });

  app.post("/api/materials/:id/variants", requireAuth, async (req, res) => {
    try {
      const parsed = insertMaterialVariantSchema.safeParse({ ...req.body, materialId: req.params.id });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const variant = await storage.createMaterialVariant(parsed.data);
      res.status(201).json(variant);
    } catch (error) {
      console.error("Error creating material variant:", error);
      res.status(500).json({ error: "Failed to create material variant" });
    }
  });

  app.post("/api/materials/:id/variants/batch", requireAuth, async (req, res) => {
    try {
      const { variants } = req.body;
      if (!Array.isArray(variants)) {
        return res.status(400).json({ error: "variants must be an array" });
      }
      const batchSchema = z.array(insertMaterialVariantSchema);
      const variantsWithMaterialId = variants.map((v: any) => ({ ...v, materialId: req.params.id }));
      const parsed = batchSchema.safeParse(variantsWithMaterialId);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const created = await storage.batchCreateMaterialVariants(parsed.data);
      res.status(201).json({ created: created.length, variants: created });
    } catch (error) {
      console.error("Error batch creating material variants:", error);
      res.status(500).json({ error: "Failed to batch create material variants" });
    }
  });

  app.patch("/api/material-variants/:id", requireAuth, async (req, res) => {
    try {
      const variant = await storage.updateMaterialVariant(req.params.id, req.body);
      if (!variant) {
        return res.status(404).json({ error: "Material variant not found" });
      }
      res.json(variant);
    } catch (error) {
      console.error("Error updating material variant:", error);
      res.status(500).json({ error: "Failed to update material variant" });
    }
  });

  app.delete("/api/material-variants/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteMaterialVariant(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting material variant:", error);
      res.status(500).json({ error: "Failed to delete material variant" });
    }
  });

  app.get("/api/materials-programs", requireAuth, async (req, res) => {
    try {
      const programs = await storage.getMaterialsPrograms();
      res.json(programs);
    } catch (error) {
      console.error("Error fetching materials programs:", error);
      res.status(500).json({ error: "Failed to fetch materials programs" });
    }
  });

  app.get("/api/materials-programs/:id", requireAuth, async (req, res) => {
    try {
      const program = await storage.getMaterialsProgram(req.params.id);
      if (!program) {
        return res.status(404).json({ error: "Materials program not found" });
      }
      res.json(program);
    } catch (error) {
      console.error("Error fetching materials program:", error);
      res.status(500).json({ error: "Failed to fetch materials program" });
    }
  });

  app.post("/api/materials-programs", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertMaterialsProgramSchema.safeParse({ ...req.body, ownerId: userId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const program = await storage.createMaterialsProgram(parsed.data);
      res.status(201).json(program);
    } catch (error) {
      console.error("Error creating materials program:", error);
      res.status(500).json({ error: "Failed to create materials program" });
    }
  });

  app.patch("/api/materials-programs/:id", requireAuth, async (req, res) => {
    try {
      const program = await storage.updateMaterialsProgram(req.params.id, req.body);
      if (!program) {
        return res.status(404).json({ error: "Materials program not found" });
      }
      res.json(program);
    } catch (error) {
      console.error("Error updating materials program:", error);
      res.status(500).json({ error: "Failed to update materials program" });
    }
  });

  app.delete("/api/materials-programs/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteMaterialsProgram(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting materials program:", error);
      res.status(500).json({ error: "Failed to delete materials program" });
    }
  });

  app.get("/api/materials-campaigns", requireAuth, async (req, res) => {
    try {
      const programId = req.query.programId as string | undefined;
      const campaigns = await storage.getMaterialsCampaigns(programId);
      res.json(campaigns);
    } catch (error) {
      console.error("Error fetching materials campaigns:", error);
      res.status(500).json({ error: "Failed to fetch materials campaigns" });
    }
  });

  app.get("/api/materials-campaigns/:id", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.getMaterialsCampaign(req.params.id);
      if (!campaign) {
        return res.status(404).json({ error: "Materials campaign not found" });
      }
      const scores = await storage.getMaterialsOracleScores(campaign.id);
      res.json({ ...campaign, scores });
    } catch (error) {
      console.error("Error fetching materials campaign:", error);
      res.status(500).json({ error: "Failed to fetch materials campaign" });
    }
  });

  app.post("/api/materials-campaigns", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertMaterialsCampaignSchema.safeParse({ ...req.body, ownerId: userId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const campaign = await storage.createMaterialsCampaign(parsed.data);
      res.status(201).json(campaign);
    } catch (error) {
      console.error("Error creating materials campaign:", error);
      res.status(500).json({ error: "Failed to create materials campaign" });
    }
  });

  app.patch("/api/materials-campaigns/:id", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.updateMaterialsCampaign(req.params.id, req.body);
      if (!campaign) {
        return res.status(404).json({ error: "Materials campaign not found" });
      }
      res.json(campaign);
    } catch (error) {
      console.error("Error updating materials campaign:", error);
      res.status(500).json({ error: "Failed to update materials campaign" });
    }
  });

  app.delete("/api/materials-campaigns/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteMaterialsCampaign(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting materials campaign:", error);
      res.status(500).json({ error: "Failed to delete materials campaign" });
    }
  });

  app.post("/api/materials-campaigns/:id/scores", requireAuth, async (req, res) => {
    try {
      const parsed = insertMaterialsOracleScoreSchema.safeParse({ ...req.body, campaignId: req.params.id });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const score = await storage.createMaterialsOracleScore(parsed.data);
      res.status(201).json(score);
    } catch (error) {
      console.error("Error creating materials oracle score:", error);
      res.status(500).json({ error: "Failed to create materials oracle score" });
    }
  });

  // Agent endpoints for materials discovery
  app.get("/api/agent/materials", async (req, res) => {
    try {
      const type = req.query.type as string | undefined;
      const materials = await storage.getMaterialEntities(type);
      res.json(materials.map(m => ({ id: m.id, type: m.type, isCurated: m.isCurated })));
    } catch (error) {
      console.error("Error fetching materials for agent:", error);
      res.status(500).json({ error: "Failed to fetch materials" });
    }
  });

  app.get("/api/agent/materials-programs", async (req, res) => {
    try {
      const programs = await storage.getMaterialsPrograms();
      res.json(programs.map(p => ({ id: p.id, name: p.name, materialType: p.materialType })));
    } catch (error) {
      console.error("Error fetching materials programs for agent:", error);
      res.status(500).json({ error: "Failed to fetch materials programs" });
    }
  });

  app.get("/api/agent/materials-campaigns", async (req, res) => {
    try {
      const campaigns = await storage.getMaterialsCampaigns();
      res.json(campaigns.map(c => ({ id: c.id, name: c.name, status: c.status, modality: c.modality })));
    } catch (error) {
      console.error("Error fetching materials campaigns for agent:", error);
      res.status(500).json({ error: "Failed to fetch materials campaigns" });
    }
  });

  app.get("/api/agent/materials-campaigns/:id/scores", async (req, res) => {
    try {
      const scores = await storage.getMaterialsOracleScores(req.params.id);
      res.json(scores.map(s => ({
        id: s.id,
        materialId: s.materialId,
        oracleScore: s.oracleScore,
        synthesisFeasibility: s.synthesisFeasibility,
        manufacturingCostFactor: s.manufacturingCostFactor,
      })));
    } catch (error) {
      console.error("Error fetching materials scores for agent:", error);
      res.status(500).json({ error: "Failed to fetch materials scores" });
    }
  });

  app.get("/api/processing-jobs", requireAuth, async (req, res) => {
    try {
      const { status, type, campaignId, materialsCampaignId, limit, offset } = req.query;
      const filters: any = {};
      if (status) filters.status = status as string;
      if (type) filters.type = type as string;
      if (campaignId) filters.campaignId = campaignId as string;
      if (materialsCampaignId) filters.materialsCampaignId = materialsCampaignId as string;
      if (limit) {
        const parsedLimit = parseInt(limit as string);
        if (!Number.isNaN(parsedLimit)) filters.limit = parsedLimit;
      }
      if (offset) {
        const parsedOffset = parseInt(offset as string);
        if (!Number.isNaN(parsedOffset)) filters.offset = parsedOffset;
      }
      const result = await storage.getProcessingJobs(Object.keys(filters).length > 0 ? filters : undefined);
      res.json(result);
    } catch (error) {
      console.error("Error fetching processing jobs:", error);
      res.status(500).json({ error: "Failed to fetch processing jobs" });
    }
  });

  app.get("/api/processing-jobs/:id", requireAuth, async (req, res) => {
    try {
      const job = await storage.getProcessingJob(req.params.id);
      if (!job) {
        return res.status(404).json({ error: "Processing job not found" });
      }
      const runs = await storage.getProcessingJobRuns(job.id);
      const events = await storage.getProcessingJobEvents(job.id);
      res.json({ ...job, runs, events });
    } catch (error) {
      console.error("Error fetching processing job:", error);
      res.status(500).json({ error: "Failed to fetch processing job" });
    }
  });

  app.post("/api/processing-jobs", requireAuth, async (req, res) => {
    try {
      const parsed = insertProcessingJobSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const job = await storage.createProcessingJob(parsed.data);
      res.status(201).json(job);
    } catch (error) {
      console.error("Error creating processing job:", error);
      res.status(500).json({ error: "Failed to create processing job" });
    }
  });

  app.patch("/api/processing-jobs/:id", requireAuth, async (req, res) => {
    try {
      const job = await storage.updateProcessingJob(req.params.id, req.body);
      if (!job) {
        return res.status(404).json({ error: "Processing job not found" });
      }
      res.json(job);
    } catch (error) {
      console.error("Error updating processing job:", error);
      res.status(500).json({ error: "Failed to update processing job" });
    }
  });

  app.post("/api/processing-jobs/:id/progress", requireAuth, async (req, res) => {
    try {
      const { itemsCompleted, checkpointData } = req.body;
      const job = await storage.updateProcessingJobProgress(req.params.id, itemsCompleted, checkpointData);
      if (!job) {
        return res.status(404).json({ error: "Processing job not found" });
      }
      res.json(job);
    } catch (error) {
      console.error("Error updating processing job progress:", error);
      res.status(500).json({ error: "Failed to update processing job progress" });
    }
  });

  app.post("/api/processing-jobs/:id/events", requireAuth, async (req, res) => {
    try {
      const parsed = insertProcessingJobEventSchema.safeParse({ ...req.body, jobId: req.params.id });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const event = await storage.createProcessingJobEvent(parsed.data);
      res.status(201).json(event);
    } catch (error) {
      console.error("Error creating processing job event:", error);
      res.status(500).json({ error: "Failed to create processing job event" });
    }
  });

  app.get("/api/jobs/:jobId/artifacts", requireAuth, async (req, res) => {
    try {
      const job = await storage.getProcessingJob(req.params.jobId);
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      const artifacts = await storage.getJobArtifacts(req.params.jobId);
      res.json(artifacts);
    } catch (error) {
      console.error("Error fetching job artifacts:", error);
      res.status(500).json({ error: "Failed to fetch job artifacts" });
    }
  });

  app.get("/api/campaigns/:campaignId/artifacts", requireAuth, async (req, res) => {
    try {
      const campaign = await storage.getCampaign(req.params.campaignId);
      if (!campaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }
      const artifacts = await storage.getArtifactsByCampaign(req.params.campaignId, "drug");
      res.json(artifacts);
    } catch (error) {
      console.error("Error fetching campaign artifacts:", error);
      res.status(500).json({ error: "Failed to fetch campaign artifacts" });
    }
  });

  app.get("/api/materials-campaigns/:campaignId/artifacts", requireAuth, async (req, res) => {
    try {
      const materialsCampaign = await storage.getMaterialsCampaign(req.params.campaignId);
      if (!materialsCampaign) {
        return res.status(404).json({ error: "Materials campaign not found" });
      }
      const artifacts = await storage.getArtifactsByCampaign(req.params.campaignId, "materials");
      res.json(artifacts);
    } catch (error) {
      console.error("Error fetching materials campaign artifacts:", error);
      res.status(500).json({ error: "Failed to fetch materials campaign artifacts" });
    }
  });

  app.post("/api/jobs/:jobId/artifacts", requireAuth, async (req, res) => {
    try {
      const parsed = insertJobArtifactSchema.safeParse({ ...req.body, jobId: req.params.jobId });
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.errors });
      }
      const artifact = await storage.createJobArtifact(parsed.data);
      res.status(201).json(artifact);
    } catch (error) {
      console.error("Error creating job artifact:", error);
      res.status(500).json({ error: "Failed to create job artifact" });
    }
  });

  app.post("/api/jobs/:jobId/artifacts/batch", requireAuth, async (req, res) => {
    try {
      const { artifacts } = req.body;
      if (!Array.isArray(artifacts)) {
        return res.status(400).json({ error: "artifacts must be an array" });
      }
      const parsedArtifacts = artifacts.map((a: unknown) => {
        const artifactData = a as Record<string, unknown>;
        const parsed = insertJobArtifactSchema.safeParse({ ...artifactData, jobId: req.params.jobId });
        if (!parsed.success) throw new Error("Invalid artifact");
        return parsed.data;
      });
      const result = await storage.createJobArtifactsBatch(parsedArtifacts);
      res.status(201).json(result);
    } catch (error) {
      console.error("Error creating job artifacts batch:", error);
      res.status(500).json({ error: "Failed to create job artifacts batch" });
    }
  });

  app.post("/api/jobs/:jobId/artifacts/from-manifest", requireAuth, async (req, res) => {
    try {
      const job = await storage.getProcessingJob(req.params.jobId);
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      const { manifest } = req.body;
      if (!manifest || typeof manifest !== "object") {
        return res.status(400).json({ error: "manifest object is required" });
      }
      const result = await registerArtifactsFromManifest(req.params.jobId, manifest);
      if (result.errors.length > 0 && result.registered === 0) {
        return res.status(400).json({ error: "Failed to register artifacts", details: result.errors });
      }
      res.status(201).json(result);
    } catch (error) {
      console.error("Error registering artifacts from manifest:", error);
      res.status(500).json({ error: "Failed to register artifacts from manifest" });
    }
  });

  app.post("/api/materials/:materialId/variants/batch-submit", requireAuth, async (req, res) => {
    try {
      const { variants, priority } = req.body;
      if (!Array.isArray(variants)) {
        return res.status(400).json({ error: "variants must be an array" });
      }
      const job = await storage.createProcessingJob({
        type: "variant_generation",
        status: "queued",
        priority: priority || 0,
        itemsTotal: variants.length,
        inputPayload: { materialId: req.params.materialId, variants }
      });
      res.status(202).json({ jobId: job.id, itemsTotal: variants.length, status: job.status });
    } catch (error) {
      console.error("Error submitting batch variant job:", error);
      res.status(500).json({ error: "Failed to submit batch variant job" });
    }
  });

  app.get("/api/materials-campaigns/:id/aggregates", requireAuth, async (req, res) => {
    try {
      const aggregate = await storage.getMaterialsCampaignAggregate(req.params.id);
      if (!aggregate) {
        return res.status(404).json({ error: "No aggregates found for this campaign" });
      }
      res.json(aggregate);
    } catch (error) {
      console.error("Error fetching campaign aggregates:", error);
      res.status(500).json({ error: "Failed to fetch campaign aggregates" });
    }
  });

  app.post("/api/materials-campaigns/:id/aggregates/refresh", requireAuth, async (req, res) => {
    try {
      const job = await storage.createProcessingJob({
        type: "aggregation",
        status: "queued",
        materialsCampaignId: req.params.id,
        itemsTotal: 1
      });
      res.status(202).json({ jobId: job.id, status: "queued" });
    } catch (error) {
      console.error("Error scheduling aggregation refresh:", error);
      res.status(500).json({ error: "Failed to schedule aggregation refresh" });
    }
  });

  app.get("/api/materials-campaigns/:id/variant-metrics", requireAuth, async (req, res) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : undefined;
      const metrics = await storage.getMaterialVariantMetrics(req.params.id, limit);
      res.json(metrics);
    } catch (error) {
      console.error("Error fetching variant metrics:", error);
      res.status(500).json({ error: "Failed to fetch variant metrics" });
    }
  });

  app.get("/api/import-templates", requireAuth, async (req, res) => {
    try {
      const { domain, importType, organizationId } = req.query;
      const templates = await storage.getImportTemplates(
        domain as string | undefined,
        importType as string | undefined,
        organizationId as string | undefined
      );
      res.json(templates);
    } catch (error) {
      console.error("Error fetching import templates:", error);
      res.status(500).json({ error: "Failed to fetch import templates" });
    }
  });

  app.get("/api/import-templates/:id", requireAuth, async (req, res) => {
    try {
      const template = await storage.getImportTemplate(req.params.id);
      if (!template) {
        return res.status(404).json({ error: "Import template not found" });
      }
      res.json(template);
    } catch (error) {
      console.error("Error fetching import template:", error);
      res.status(500).json({ error: "Failed to fetch import template" });
    }
  });

  app.post("/api/import-templates", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const parsed = insertImportTemplateSchema.safeParse({
        ...req.body,
        createdBy: userId
      });
      if (!parsed.success) {
        return res.status(400).json({ error: "Invalid template data", details: parsed.error.errors });
      }
      const template = await storage.createImportTemplate(parsed.data);
      res.status(201).json(template);
    } catch (error) {
      console.error("Error creating import template:", error);
      res.status(500).json({ error: "Failed to create import template" });
    }
  });

  app.patch("/api/import-templates/:id", requireAuth, async (req, res) => {
    try {
      const template = await storage.updateImportTemplate(req.params.id, req.body);
      if (!template) {
        return res.status(404).json({ error: "Import template not found" });
      }
      res.json(template);
    } catch (error) {
      console.error("Error updating import template:", error);
      res.status(500).json({ error: "Failed to update import template" });
    }
  });

  app.delete("/api/import-templates/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteImportTemplate(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting import template:", error);
      res.status(500).json({ error: "Failed to delete import template" });
    }
  });

  app.get("/api/import-jobs", requireAuth, async (req, res) => {
    try {
      const { domain, importType, status, organizationId } = req.query;
      const jobs = await storage.getImportJobs({
        domain: domain as string | undefined,
        importType: importType as string | undefined,
        status: status as string | undefined,
        organizationId: organizationId as string | undefined
      });
      res.json(jobs);
    } catch (error) {
      console.error("Error fetching import jobs:", error);
      res.status(500).json({ error: "Failed to fetch import jobs" });
    }
  });

  app.get("/api/import-jobs/:id", requireAuth, async (req, res) => {
    try {
      const job = await storage.getImportJob(req.params.id);
      if (!job) {
        return res.status(404).json({ error: "Import job not found" });
      }
      res.json(job);
    } catch (error) {
      console.error("Error fetching import job:", error);
      res.status(500).json({ error: "Failed to fetch import job" });
    }
  });

  app.post("/api/import-jobs", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      
      const processingJob = await storage.createProcessingJob({
        type: "aggregation",
        status: "queued",
        itemsTotal: req.body.validationSummary?.totalRows || 1,
        inputPayload: { importType: req.body.importType, domain: req.body.domain }
      });
      
      const parsed = insertImportJobSchema.safeParse({
        ...req.body,
        processingJobId: processingJob.id,
        createdBy: userId,
        status: "pending"
      });
      if (!parsed.success) {
        return res.status(400).json({ error: "Invalid import job data", details: parsed.error.errors });
      }
      
      const importJob = await storage.createImportJob(parsed.data);
      res.status(201).json(importJob);
    } catch (error) {
      console.error("Error creating import job:", error);
      res.status(500).json({ error: "Failed to create import job" });
    }
  });

  app.patch("/api/import-jobs/:id", requireAuth, async (req, res) => {
    try {
      const job = await storage.updateImportJob(req.params.id, req.body);
      if (!job) {
        return res.status(404).json({ error: "Import job not found" });
      }
      res.json(job);
    } catch (error) {
      console.error("Error updating import job:", error);
      res.status(500).json({ error: "Failed to update import job" });
    }
  });

  app.get("/api/compound-assets/molecule/:moleculeId", requireAuth, async (req, res) => {
    try {
      const assets = await storage.getCompoundAssetsByMolecule(req.params.moleculeId);
      res.json(assets);
    } catch (error) {
      console.error("Error fetching compound assets:", error);
      res.status(500).json({ error: "Failed to fetch compound assets" });
    }
  });

  app.get("/api/compound-assets/molecule/:moleculeId/:assetType", requireAuth, async (req, res) => {
    try {
      const asset = await storage.getCompoundAssetByTypeAndMolecule(
        req.params.moleculeId,
        req.params.assetType
      );
      if (!asset) {
        return res.status(404).json({ error: "Asset not found" });
      }
      res.json(asset);
    } catch (error) {
      console.error("Error fetching compound asset:", error);
      res.status(500).json({ error: "Failed to fetch compound asset" });
    }
  });

  app.post("/api/compound-assets", requireAuth, async (req, res) => {
    try {
      const parsed = insertCompoundAssetSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: "Invalid asset data", details: parsed.error.errors });
      }
      const asset = await storage.createCompoundAsset(parsed.data);
      res.status(201).json(asset);
    } catch (error) {
      console.error("Error creating compound asset:", error);
      res.status(500).json({ error: "Failed to create compound asset" });
    }
  });

  app.post("/api/compound-assets/batch", requireAuth, async (req, res) => {
    try {
      const { assets } = req.body;
      if (!Array.isArray(assets)) {
        return res.status(400).json({ error: "assets must be an array" });
      }
      const parsedAssets = assets.map((a: unknown) => {
        const parsed = insertCompoundAssetSchema.safeParse(a);
        if (!parsed.success) throw new Error("Invalid asset");
        return parsed.data;
      });
      const result = await storage.bulkCreateCompoundAssets(parsedAssets);
      res.status(201).json(result);
    } catch (error) {
      console.error("Error creating compound assets batch:", error);
      res.status(500).json({ error: "Failed to create compound assets batch" });
    }
  });

  app.delete("/api/compound-assets/:id", requireAuth, async (req, res) => {
    try {
      await storage.deleteCompoundAsset(req.params.id);
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting compound asset:", error);
      res.status(500).json({ error: "Failed to delete compound asset" });
    }
  });

  app.get("/api/compound-assets/signed-url/:assetType/:moleculeId", requireAuth, async (req, res) => {
    try {
      const { getSignedDownloadUrl, generateAssetKey, getExtensionForAsset, isSpacesConfigured } = await import("./spaces-storage");
      
      if (!isSpacesConfigured()) {
        return res.status(503).json({ error: "Object storage not configured" });
      }
      
      const extension = getExtensionForAsset(req.params.assetType);
      const key = generateAssetKey(req.params.moleculeId, req.params.assetType, extension);
      const expiresIn = parseInt(req.query.expiresIn as string) || 3600;
      
      const signedUrl = await getSignedDownloadUrl(key, { expiresIn });
      res.json({ signedUrl, key, expiresIn });
    } catch (error) {
      console.error("Error generating signed download URL:", error);
      res.status(500).json({ error: "Failed to generate signed URL" });
    }
  });

  app.post("/api/compound-assets/signed-upload-url", requireAuth, async (req, res) => {
    try {
      const { getSignedUploadUrl, generateAssetKey, getExtensionForAsset, getMimeTypeForAsset, isSpacesConfigured } = await import("./spaces-storage");
      
      if (!isSpacesConfigured()) {
        return res.status(503).json({ error: "Object storage not configured" });
      }
      
      const { moleculeId, assetType, companyId, expiresIn = 3600 } = req.body;
      if (!moleculeId || !assetType) {
        return res.status(400).json({ error: "moleculeId and assetType are required" });
      }
      
      const extension = getExtensionForAsset(assetType);
      const mimeType = getMimeTypeForAsset(assetType);
      const key = generateAssetKey(moleculeId, assetType, extension, companyId);
      
      const signedUrl = await getSignedUploadUrl(key, mimeType, { expiresIn });
      res.json({ signedUrl, key, mimeType, expiresIn });
    } catch (error) {
      console.error("Error generating signed upload URL:", error);
      res.status(500).json({ error: "Failed to generate signed upload URL" });
    }
  });

  app.get("/api/built-in-molecules", requireAuth, async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 100000;
      const offset = parseInt(req.query.offset as string) || 0;
      const molecules = await storage.getBuiltInMolecules(limit, offset);
      const total = await storage.countBuiltInMolecules();
      res.json({ molecules, total, limit, offset });
    } catch (error) {
      console.error("Error fetching built-in molecules:", error);
      res.status(500).json({ error: "Failed to fetch built-in molecules" });
    }
  });

  app.post("/api/built-in-molecules/mark", requireAuth, async (req, res) => {
    try {
      const { moleculeIds } = req.body;
      if (!Array.isArray(moleculeIds)) {
        return res.status(400).json({ error: "moleculeIds must be an array" });
      }
      await storage.markMoleculesAsBuiltIn(moleculeIds);
      res.json({ marked: moleculeIds.length });
    } catch (error) {
      console.error("Error marking molecules as built-in:", error);
      res.status(500).json({ error: "Failed to mark molecules as built-in" });
    }
  });

  app.get("/api/storage/status", requireAuth, async (req, res) => {
    try {
      const { isSpacesConfigured } = await import("./spaces-storage");
      const configured = isSpacesConfigured();
      res.json({
        configured,
        provider: configured ? "do_spaces" : null,
        message: configured 
          ? "DigitalOcean Spaces storage is configured and ready"
          : "Object storage not configured. Set DO_SPACES_* environment variables.",
      });
    } catch (error) {
      console.error("Error checking storage status:", error);
      res.status(500).json({ error: "Failed to check storage status" });
    }
  });

  return httpServer;
}
