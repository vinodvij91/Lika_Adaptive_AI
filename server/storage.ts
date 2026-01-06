import { db } from "./db";
import { eq, and, desc, sql, count, avg } from "drizzle-orm";
import {
  projects,
  targets,
  molecules,
  campaigns,
  jobs,
  modelRuns,
  moleculeScores,
  learningGraphEntries,
  comments,
  curatedLibraries,
  libraryMolecules,
  scaffolds,
  libraryAnnotations,
  computeNodes,
  userSshKeys,
  nodeKeyRegistrations,
  type Project,
  type InsertProject,
  type Target,
  type InsertTarget,
  type Molecule,
  type InsertMolecule,
  type Campaign,
  type InsertCampaign,
  type Job,
  type InsertJob,
  type ModelRun,
  type InsertModelRun,
  type MoleculeScore,
  type InsertMoleculeScore,
  type LearningGraphEntry,
  type InsertLearningGraphEntry,
  type Comment,
  type InsertComment,
  type CuratedLibrary,
  type InsertCuratedLibrary,
  type LibraryMolecule,
  type InsertLibraryMolecule,
  type Scaffold,
  type InsertScaffold,
  type LibraryAnnotation,
  type InsertLibraryAnnotation,
  type ComputeNode,
  type InsertComputeNode,
  type UserSshKey,
  type InsertUserSshKey,
  type NodeKeyRegistration,
  type InsertNodeKeyRegistration,
  type DiseaseArea,
} from "@shared/schema";

export interface IStorage {
  getProjects(userId: string): Promise<Project[]>;
  getProject(id: string): Promise<Project | undefined>;
  createProject(project: InsertProject): Promise<Project>;
  updateProject(id: string, project: Partial<InsertProject>): Promise<Project | undefined>;
  deleteProject(id: string): Promise<void>;

  getTargets(): Promise<Target[]>;
  getTarget(id: string): Promise<Target | undefined>;
  createTarget(target: InsertTarget): Promise<Target>;
  updateTarget(id: string, target: Partial<InsertTarget>): Promise<Target | undefined>;
  deleteTarget(id: string): Promise<void>;

  getMolecules(): Promise<Molecule[]>;
  getMolecule(id: string): Promise<Molecule | undefined>;
  createMolecule(molecule: InsertMolecule): Promise<Molecule>;
  bulkCreateMolecules(molecules: InsertMolecule[]): Promise<Molecule[]>;

  getCampaigns(projectId?: string): Promise<Campaign[]>;
  getCampaign(id: string): Promise<Campaign | undefined>;
  createCampaign(campaign: InsertCampaign): Promise<Campaign>;
  updateCampaign(id: string, campaign: Partial<InsertCampaign>): Promise<Campaign | undefined>;
  deleteCampaign(id: string): Promise<void>;

  getJobs(campaignId: string): Promise<Job[]>;
  getJob(id: string): Promise<Job | undefined>;
  createJob(job: InsertJob): Promise<Job>;
  updateJob(id: string, job: Partial<Job>): Promise<Job | undefined>;

  getModelRuns(campaignId: string): Promise<ModelRun[]>;
  createModelRun(run: InsertModelRun): Promise<ModelRun>;
  updateModelRun(id: string, run: Partial<ModelRun>): Promise<ModelRun | undefined>;

  getMoleculeScores(campaignId: string): Promise<(MoleculeScore & { molecule: Molecule | null })[]>;
  createMoleculeScore(score: InsertMoleculeScore): Promise<MoleculeScore>;
  bulkCreateMoleculeScores(scores: InsertMoleculeScore[]): Promise<MoleculeScore[]>;

  getLearningGraphEntries(): Promise<(LearningGraphEntry & { molecule: Molecule | null })[]>;
  createLearningGraphEntry(entry: InsertLearningGraphEntry): Promise<LearningGraphEntry>;

  getComments(projectId?: string, campaignId?: string): Promise<Comment[]>;
  createComment(comment: InsertComment): Promise<Comment>;

  getDashboardStats(userId: string): Promise<{
    totalMolecules: number;
    totalCampaigns: number;
    activeCampaigns: number;
    campaignsThisWeek: number;
    domainBreakdown: Record<DiseaseArea, number>;
  }>;

  getReportsData(): Promise<{
    oracleDistribution: { range: string; count: number }[];
    admetPassRate: { passed: number; failed: number };
    domainBreakdown: Record<DiseaseArea, number>;
    recentCampaigns: { name: string; molecules: number; avgScore: number }[];
  }>;

  getCuratedLibraries(filters?: { domainType?: DiseaseArea; status?: string; isPublic?: boolean }): Promise<CuratedLibrary[]>;
  getCuratedLibrary(id: string): Promise<CuratedLibrary | undefined>;
  createCuratedLibrary(library: InsertCuratedLibrary): Promise<CuratedLibrary>;
  updateCuratedLibrary(id: string, library: Partial<InsertCuratedLibrary>): Promise<CuratedLibrary | undefined>;
  deleteCuratedLibrary(id: string): Promise<void>;

  getLibraryMolecules(libraryId: string): Promise<(LibraryMolecule & { molecule: Molecule | null })[]>;
  addLibraryMolecule(entry: InsertLibraryMolecule): Promise<LibraryMolecule>;
  bulkAddLibraryMolecules(entries: InsertLibraryMolecule[]): Promise<LibraryMolecule[]>;
  updateLibraryMolecule(id: string, entry: Partial<InsertLibraryMolecule>): Promise<LibraryMolecule | undefined>;

  getScaffolds(libraryId: string): Promise<Scaffold[]>;
  createScaffold(scaffold: InsertScaffold): Promise<Scaffold>;
  updateScaffold(id: string, scaffold: Partial<InsertScaffold>): Promise<Scaffold | undefined>;

  getLibraryAnnotations(libraryId: string): Promise<LibraryAnnotation[]>;
  createLibraryAnnotation(annotation: InsertLibraryAnnotation): Promise<LibraryAnnotation>;

  getComputeNodes(): Promise<ComputeNode[]>;
  getComputeNode(id: string): Promise<ComputeNode | undefined>;
  createComputeNode(node: InsertComputeNode): Promise<ComputeNode>;
  updateComputeNode(id: string, node: Partial<InsertComputeNode>): Promise<ComputeNode | undefined>;
  deleteComputeNode(id: string): Promise<void>;

  getUserSshKeys(userId: string): Promise<UserSshKey[]>;
  getUserSshKey(id: string): Promise<UserSshKey | undefined>;
  createUserSshKey(key: InsertUserSshKey): Promise<UserSshKey>;
  deleteUserSshKey(id: string): Promise<void>;

  getNodeKeyRegistrations(nodeId: string): Promise<(NodeKeyRegistration & { sshKey: UserSshKey | null })[]>;
  createNodeKeyRegistration(reg: InsertNodeKeyRegistration): Promise<NodeKeyRegistration>;
}

export class DatabaseStorage implements IStorage {
  async getProjects(userId: string): Promise<Project[]> {
    return db.select().from(projects).where(eq(projects.ownerId, userId)).orderBy(desc(projects.updatedAt));
  }

  async getProject(id: string): Promise<Project | undefined> {
    const result = await db.select().from(projects).where(eq(projects.id, id)).limit(1);
    return result[0];
  }

  async createProject(project: InsertProject): Promise<Project> {
    const result = await db.insert(projects).values(project).returning();
    return result[0];
  }

  async updateProject(id: string, project: Partial<InsertProject>): Promise<Project | undefined> {
    const result = await db.update(projects).set({ ...project, updatedAt: new Date() }).where(eq(projects.id, id)).returning();
    return result[0];
  }

  async deleteProject(id: string): Promise<void> {
    await db.delete(projects).where(eq(projects.id, id));
  }

  async getTargets(): Promise<Target[]> {
    return db.select().from(targets).orderBy(desc(targets.createdAt));
  }

  async getTarget(id: string): Promise<Target | undefined> {
    const result = await db.select().from(targets).where(eq(targets.id, id)).limit(1);
    return result[0];
  }

  async createTarget(target: InsertTarget): Promise<Target> {
    const result = await db.insert(targets).values(target).returning();
    return result[0];
  }

  async updateTarget(id: string, target: Partial<InsertTarget>): Promise<Target | undefined> {
    const result = await db.update(targets).set(target).where(eq(targets.id, id)).returning();
    return result[0];
  }

  async deleteTarget(id: string): Promise<void> {
    await db.delete(targets).where(eq(targets.id, id));
  }

  async getMolecules(): Promise<Molecule[]> {
    return db.select().from(molecules).orderBy(desc(molecules.createdAt)).limit(500);
  }

  async getMolecule(id: string): Promise<Molecule | undefined> {
    const result = await db.select().from(molecules).where(eq(molecules.id, id)).limit(1);
    return result[0];
  }

  async createMolecule(molecule: InsertMolecule): Promise<Molecule> {
    const result = await db.insert(molecules).values(molecule).returning();
    return result[0];
  }

  async bulkCreateMolecules(moleculeData: InsertMolecule[]): Promise<Molecule[]> {
    if (moleculeData.length === 0) return [];
    return db.insert(molecules).values(moleculeData).returning();
  }

  async getCampaigns(projectId?: string): Promise<Campaign[]> {
    if (projectId) {
      return db.select().from(campaigns).where(eq(campaigns.projectId, projectId)).orderBy(desc(campaigns.updatedAt));
    }
    return db.select().from(campaigns).orderBy(desc(campaigns.updatedAt));
  }

  async getCampaign(id: string): Promise<Campaign | undefined> {
    const result = await db.select().from(campaigns).where(eq(campaigns.id, id)).limit(1);
    return result[0];
  }

  async createCampaign(campaign: InsertCampaign): Promise<Campaign> {
    const result = await db.insert(campaigns).values(campaign).returning();
    return result[0];
  }

  async updateCampaign(id: string, campaign: Partial<InsertCampaign>): Promise<Campaign | undefined> {
    const result = await db.update(campaigns).set({ ...campaign, updatedAt: new Date() }).where(eq(campaigns.id, id)).returning();
    return result[0];
  }

  async deleteCampaign(id: string): Promise<void> {
    await db.delete(campaigns).where(eq(campaigns.id, id));
  }

  async getJobs(campaignId: string): Promise<Job[]> {
    return db.select().from(jobs).where(eq(jobs.campaignId, campaignId)).orderBy(jobs.createdAt);
  }

  async getJob(id: string): Promise<Job | undefined> {
    const result = await db.select().from(jobs).where(eq(jobs.id, id)).limit(1);
    return result[0];
  }

  async createJob(job: InsertJob): Promise<Job> {
    const result = await db.insert(jobs).values(job).returning();
    return result[0];
  }

  async updateJob(id: string, job: Partial<Job>): Promise<Job | undefined> {
    const result = await db.update(jobs).set(job).where(eq(jobs.id, id)).returning();
    return result[0];
  }

  async getModelRuns(campaignId: string): Promise<ModelRun[]> {
    return db.select().from(modelRuns).where(eq(modelRuns.campaignId, campaignId));
  }

  async createModelRun(run: InsertModelRun): Promise<ModelRun> {
    const result = await db.insert(modelRuns).values(run).returning();
    return result[0];
  }

  async updateModelRun(id: string, run: Partial<ModelRun>): Promise<ModelRun | undefined> {
    const result = await db.update(modelRuns).set(run).where(eq(modelRuns.id, id)).returning();
    return result[0];
  }

  async getMoleculeScores(campaignId: string): Promise<(MoleculeScore & { molecule: Molecule | null })[]> {
    const scoresWithMolecules = await db
      .select({
        score: moleculeScores,
        molecule: molecules,
      })
      .from(moleculeScores)
      .leftJoin(molecules, eq(moleculeScores.moleculeId, molecules.id))
      .where(eq(moleculeScores.campaignId, campaignId))
      .orderBy(desc(moleculeScores.oracleScore));

    return scoresWithMolecules.map((row) => ({
      ...row.score,
      molecule: row.molecule,
    }));
  }

  async createMoleculeScore(score: InsertMoleculeScore): Promise<MoleculeScore> {
    const result = await db.insert(moleculeScores).values(score).returning();
    return result[0];
  }

  async bulkCreateMoleculeScores(scoreData: InsertMoleculeScore[]): Promise<MoleculeScore[]> {
    if (scoreData.length === 0) return [];
    return db.insert(moleculeScores).values(scoreData).returning();
  }

  async getLearningGraphEntries(): Promise<(LearningGraphEntry & { molecule: Molecule | null })[]> {
    const entriesWithMolecules = await db
      .select({
        entry: learningGraphEntries,
        molecule: molecules,
      })
      .from(learningGraphEntries)
      .leftJoin(molecules, eq(learningGraphEntries.moleculeId, molecules.id))
      .orderBy(desc(learningGraphEntries.createdAt));

    return entriesWithMolecules.map((row) => ({
      ...row.entry,
      molecule: row.molecule,
    }));
  }

  async createLearningGraphEntry(entry: InsertLearningGraphEntry): Promise<LearningGraphEntry> {
    const result = await db.insert(learningGraphEntries).values(entry).returning();
    return result[0];
  }

  async getComments(projectId?: string, campaignId?: string): Promise<Comment[]> {
    if (projectId) {
      return db.select().from(comments).where(eq(comments.projectId, projectId)).orderBy(desc(comments.createdAt));
    }
    if (campaignId) {
      return db.select().from(comments).where(eq(comments.campaignId, campaignId)).orderBy(desc(comments.createdAt));
    }
    return db.select().from(comments).orderBy(desc(comments.createdAt));
  }

  async createComment(comment: InsertComment): Promise<Comment> {
    const result = await db.insert(comments).values(comment).returning();
    return result[0];
  }

  async getDashboardStats(userId: string): Promise<{
    totalMolecules: number;
    totalCampaigns: number;
    activeCampaigns: number;
    campaignsThisWeek: number;
    domainBreakdown: Record<DiseaseArea, number>;
  }> {
    const [moleculeCount] = await db.select({ count: count() }).from(molecules);
    const [campaignCount] = await db.select({ count: count() }).from(campaigns);
    const [activeCount] = await db.select({ count: count() }).from(campaigns).where(eq(campaigns.status, "running"));
    
    const oneWeekAgo = new Date();
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
    const [weekCount] = await db.select({ count: count() }).from(campaigns).where(
      and(eq(campaigns.status, "completed"), sql`${campaigns.updatedAt} > ${oneWeekAgo}`)
    );

    const domainCounts = await db
      .select({ domain: campaigns.domainType, count: count() })
      .from(campaigns)
      .groupBy(campaigns.domainType);

    const domainBreakdown: Record<DiseaseArea, number> = {
      CNS: 0, Oncology: 0, Rare: 0, Infectious: 0,
      Cardiometabolic: 0, Autoimmune: 0, Respiratory: 0, Other: 0,
    };

    for (const row of domainCounts) {
      if (row.domain && row.domain in domainBreakdown) {
        domainBreakdown[row.domain as DiseaseArea] = Number(row.count);
      }
    }

    return {
      totalMolecules: Number(moleculeCount.count),
      totalCampaigns: Number(campaignCount.count),
      activeCampaigns: Number(activeCount.count),
      campaignsThisWeek: Number(weekCount.count),
      domainBreakdown,
    };
  }

  async getReportsData(): Promise<{
    oracleDistribution: { range: string; count: number }[];
    admetPassRate: { passed: number; failed: number };
    domainBreakdown: Record<DiseaseArea, number>;
    recentCampaigns: { name: string; molecules: number; avgScore: number }[];
  }> {
    const ranges = [
      { min: 0, max: 0.2, label: "0.0-0.2" },
      { min: 0.2, max: 0.4, label: "0.2-0.4" },
      { min: 0.4, max: 0.6, label: "0.4-0.6" },
      { min: 0.6, max: 0.8, label: "0.6-0.8" },
      { min: 0.8, max: 1.0, label: "0.8-1.0" },
    ];

    const oracleDistribution: { range: string; count: number }[] = [];
    for (const range of ranges) {
      const [result] = await db
        .select({ count: count() })
        .from(moleculeScores)
        .where(and(
          sql`${moleculeScores.oracleScore} >= ${range.min}`,
          sql`${moleculeScores.oracleScore} < ${range.max}`
        ));
      oracleDistribution.push({ range: range.label, count: Number(result.count) });
    }

    const [passed] = await db.select({ count: count() }).from(moleculeScores).where(sql`${moleculeScores.admetScore} >= 0.5`);
    const [failed] = await db.select({ count: count() }).from(moleculeScores).where(sql`${moleculeScores.admetScore} < 0.5`);

    const domainCounts = await db
      .select({ domain: campaigns.domainType, count: count() })
      .from(campaigns)
      .groupBy(campaigns.domainType);

    const domainBreakdown: Record<DiseaseArea, number> = {
      CNS: 0, Oncology: 0, Rare: 0, Infectious: 0,
      Cardiometabolic: 0, Autoimmune: 0, Respiratory: 0, Other: 0,
    };

    for (const row of domainCounts) {
      if (row.domain && row.domain in domainBreakdown) {
        domainBreakdown[row.domain as DiseaseArea] = Number(row.count);
      }
    }

    const recentCampaignsData = await db
      .select({ name: campaigns.name, id: campaigns.id })
      .from(campaigns)
      .where(eq(campaigns.status, "completed"))
      .orderBy(desc(campaigns.updatedAt))
      .limit(5);

    const recentCampaigns: { name: string; molecules: number; avgScore: number }[] = [];
    for (const campaign of recentCampaignsData) {
      const [stats] = await db
        .select({ count: count(), avg: avg(moleculeScores.oracleScore) })
        .from(moleculeScores)
        .where(eq(moleculeScores.campaignId, campaign.id));

      recentCampaigns.push({
        name: campaign.name,
        molecules: Number(stats.count),
        avgScore: Number(stats.avg) || 0,
      });
    }

    return {
      oracleDistribution,
      admetPassRate: { passed: Number(passed.count), failed: Number(failed.count) },
      domainBreakdown,
      recentCampaigns,
    };
  }

  async getPendingCampaigns(): Promise<Campaign[]> {
    return db.select().from(campaigns).where(eq(campaigns.status, "pending")).orderBy(desc(campaigns.createdAt));
  }

  async getCampaignAnalytics(campaignId: string): Promise<{
    campaignId: string;
    status: string;
    totalMolecules: number;
    avgOracleScore: number;
    topScorers: { moleculeId: string; oracleScore: number }[];
    quantumStepCompleted: boolean;
    jobsSummary: { type: string; status: string }[];
  }> {
    const campaign = await this.getCampaign(campaignId);
    if (!campaign) {
      throw new Error("Campaign not found");
    }

    const [stats] = await db
      .select({ count: count(), avg: avg(moleculeScores.oracleScore) })
      .from(moleculeScores)
      .where(eq(moleculeScores.campaignId, campaignId));

    const topMolecules = await db
      .select({ moleculeId: moleculeScores.moleculeId, oracleScore: moleculeScores.oracleScore })
      .from(moleculeScores)
      .where(eq(moleculeScores.campaignId, campaignId))
      .orderBy(desc(moleculeScores.oracleScore))
      .limit(10);

    const jobsList = await db.select({ type: jobs.type, status: jobs.status }).from(jobs).where(eq(jobs.campaignId, campaignId));

    const quantumJob = jobsList.find((j) => j.type === "quantum_optimization" || j.type === "quantum_scoring");

    return {
      campaignId,
      status: campaign.status || "unknown",
      totalMolecules: Number(stats.count),
      avgOracleScore: Number(stats.avg) || 0,
      topScorers: topMolecules.map((m) => ({ moleculeId: m.moleculeId, oracleScore: m.oracleScore || 0 })),
      quantumStepCompleted: quantumJob?.status === "completed",
      jobsSummary: jobsList.map((j) => ({ type: j.type, status: j.status || "unknown" })),
    };
  }

  async getUnlabeledLearningGraphEntries(): Promise<(LearningGraphEntry & { molecule: Molecule | null })[]> {
    const entries = await db
      .select()
      .from(learningGraphEntries)
      .leftJoin(molecules, eq(learningGraphEntries.moleculeId, molecules.id))
      .where(eq(learningGraphEntries.outcomeLabel, "unknown"))
      .orderBy(desc(learningGraphEntries.createdAt));

    return entries.map((row) => ({
      ...row.learning_graph_entries,
      molecule: row.molecules,
    }));
  }

  async updateLearningGraphLabel(id: string, outcomeLabel: "promising" | "dropped" | "hit" | "unknown"): Promise<LearningGraphEntry | undefined> {
    const result = await db
      .update(learningGraphEntries)
      .set({ outcomeLabel })
      .where(eq(learningGraphEntries.id, id))
      .returning();
    return result[0];
  }

  async getCuratedLibraries(filters?: { domainType?: DiseaseArea; status?: string; isPublic?: boolean }): Promise<CuratedLibrary[]> {
    let query = db.select().from(curatedLibraries).orderBy(desc(curatedLibraries.updatedAt));
    
    if (filters?.domainType) {
      query = query.where(eq(curatedLibraries.domainType, filters.domainType)) as typeof query;
    }
    if (filters?.status) {
      query = query.where(eq(curatedLibraries.status, filters.status as "draft" | "processing" | "curated" | "deprecated")) as typeof query;
    }
    if (filters?.isPublic !== undefined) {
      query = query.where(eq(curatedLibraries.isPublic, filters.isPublic)) as typeof query;
    }
    
    return query;
  }

  async getCuratedLibrary(id: string): Promise<CuratedLibrary | undefined> {
    const result = await db.select().from(curatedLibraries).where(eq(curatedLibraries.id, id)).limit(1);
    return result[0];
  }

  async createCuratedLibrary(library: InsertCuratedLibrary): Promise<CuratedLibrary> {
    const result = await db.insert(curatedLibraries).values(library).returning();
    return result[0];
  }

  async updateCuratedLibrary(id: string, library: Partial<InsertCuratedLibrary>): Promise<CuratedLibrary | undefined> {
    const result = await db.update(curatedLibraries).set({ ...library, updatedAt: new Date() }).where(eq(curatedLibraries.id, id)).returning();
    return result[0];
  }

  async deleteCuratedLibrary(id: string): Promise<void> {
    await db.delete(curatedLibraries).where(eq(curatedLibraries.id, id));
  }

  async getLibraryMolecules(libraryId: string): Promise<(LibraryMolecule & { molecule: Molecule | null })[]> {
    const results = await db
      .select({
        libraryMolecule: libraryMolecules,
        molecule: molecules,
      })
      .from(libraryMolecules)
      .leftJoin(molecules, eq(libraryMolecules.moleculeId, molecules.id))
      .where(eq(libraryMolecules.libraryId, libraryId))
      .orderBy(desc(libraryMolecules.createdAt));

    return results.map((row) => ({
      ...row.libraryMolecule,
      molecule: row.molecule,
    }));
  }

  async addLibraryMolecule(entry: InsertLibraryMolecule): Promise<LibraryMolecule> {
    const result = await db.insert(libraryMolecules).values(entry).returning();
    return result[0];
  }

  async bulkAddLibraryMolecules(entries: InsertLibraryMolecule[]): Promise<LibraryMolecule[]> {
    if (entries.length === 0) return [];
    return db.insert(libraryMolecules).values(entries).returning();
  }

  async updateLibraryMolecule(id: string, entry: Partial<InsertLibraryMolecule>): Promise<LibraryMolecule | undefined> {
    const result = await db.update(libraryMolecules).set(entry).where(eq(libraryMolecules.id, id)).returning();
    return result[0];
  }

  async getScaffolds(libraryId: string): Promise<Scaffold[]> {
    return db.select().from(scaffolds).where(eq(scaffolds.libraryId, libraryId)).orderBy(desc(scaffolds.memberCount));
  }

  async createScaffold(scaffold: InsertScaffold): Promise<Scaffold> {
    const result = await db.insert(scaffolds).values(scaffold).returning();
    return result[0];
  }

  async updateScaffold(id: string, scaffold: Partial<InsertScaffold>): Promise<Scaffold | undefined> {
    const result = await db.update(scaffolds).set(scaffold).where(eq(scaffolds.id, id)).returning();
    return result[0];
  }

  async getLibraryAnnotations(libraryId: string): Promise<LibraryAnnotation[]> {
    return db.select().from(libraryAnnotations).where(eq(libraryAnnotations.libraryId, libraryId)).orderBy(desc(libraryAnnotations.createdAt));
  }

  async createLibraryAnnotation(annotation: InsertLibraryAnnotation): Promise<LibraryAnnotation> {
    const result = await db.insert(libraryAnnotations).values(annotation).returning();
    return result[0];
  }

  async getComputeNodes(): Promise<ComputeNode[]> {
    return db.select().from(computeNodes).orderBy(desc(computeNodes.createdAt));
  }

  async getComputeNode(id: string): Promise<ComputeNode | undefined> {
    const result = await db.select().from(computeNodes).where(eq(computeNodes.id, id)).limit(1);
    return result[0];
  }

  async createComputeNode(node: InsertComputeNode): Promise<ComputeNode> {
    const result = await db.insert(computeNodes).values(node).returning();
    return result[0];
  }

  async updateComputeNode(id: string, node: Partial<InsertComputeNode>): Promise<ComputeNode | undefined> {
    const result = await db.update(computeNodes).set({ ...node, updatedAt: new Date() }).where(eq(computeNodes.id, id)).returning();
    return result[0];
  }

  async deleteComputeNode(id: string): Promise<void> {
    await db.delete(computeNodes).where(eq(computeNodes.id, id));
  }

  async getUserSshKeys(userId: string): Promise<UserSshKey[]> {
    return db.select().from(userSshKeys).where(eq(userSshKeys.userId, userId)).orderBy(desc(userSshKeys.createdAt));
  }

  async getUserSshKey(id: string): Promise<UserSshKey | undefined> {
    const result = await db.select().from(userSshKeys).where(eq(userSshKeys.id, id)).limit(1);
    return result[0];
  }

  async createUserSshKey(key: InsertUserSshKey): Promise<UserSshKey> {
    const result = await db.insert(userSshKeys).values(key).returning();
    return result[0];
  }

  async deleteUserSshKey(id: string): Promise<void> {
    await db.delete(userSshKeys).where(eq(userSshKeys.id, id));
  }

  async getNodeKeyRegistrations(nodeId: string): Promise<(NodeKeyRegistration & { sshKey: UserSshKey | null })[]> {
    const results = await db
      .select({
        registration: nodeKeyRegistrations,
        sshKey: userSshKeys,
      })
      .from(nodeKeyRegistrations)
      .leftJoin(userSshKeys, eq(nodeKeyRegistrations.sshKeyId, userSshKeys.id))
      .where(eq(nodeKeyRegistrations.nodeId, nodeId))
      .orderBy(desc(nodeKeyRegistrations.registeredAt));

    return results.map((row) => ({
      ...row.registration,
      sshKey: row.sshKey,
    }));
  }

  async createNodeKeyRegistration(reg: InsertNodeKeyRegistration): Promise<NodeKeyRegistration> {
    const result = await db.insert(nodeKeyRegistrations).values(reg).returning();
    return result[0];
  }
}

export const storage = new DatabaseStorage();
