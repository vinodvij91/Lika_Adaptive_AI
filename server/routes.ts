import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { orchestrator } from "./orchestrator";
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
  insertUserSshKeySchema,
  moleculeScores,
} from "@shared/schema";

function requireAuth(req: Request, res: Response, next: NextFunction) {
  if (!req.isAuthenticated()) {
    return res.status(401).json({ error: "Unauthorized" });
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
      const nodes = await storage.getComputeNodes();
      res.json(nodes);
    } catch (error) {
      console.error("Error fetching compute nodes:", error);
      res.status(500).json({ error: "Failed to fetch compute nodes" });
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
  // USAGE METERS ENDPOINTS
  // ============================================

  app.get("/api/usage", requireAuth, async (req, res) => {
    try {
      const userId = (req.user as any)?.id || "";
      const { projectId, campaignId } = req.query;
      const meters = await storage.getUsageMeters({
        userId,
        projectId: projectId as string | undefined,
        campaignId: campaignId as string | undefined,
      });
      res.json(meters);
    } catch (error) {
      console.error("Error fetching usage meters:", error);
      res.status(500).json({ error: "Failed to fetch usage" });
    }
  });

  app.get("/api/usage/summary/:projectId", requireAuth, async (req, res) => {
    try {
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

  return httpServer;
}
