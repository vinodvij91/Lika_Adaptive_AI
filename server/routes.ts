import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { orchestrator } from "./orchestrator";
import {
  insertProjectSchema,
  insertTargetSchema,
  insertCampaignSchema,
  insertCommentSchema,
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

  return httpServer;
}
