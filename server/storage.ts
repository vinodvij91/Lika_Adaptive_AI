import { db } from "./db";
import { eq, and, desc, sql, count, avg, sum } from "drizzle-orm";
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
  usageMeters,
  creditWallets,
  creditTransactions,
  targetVariants,
  diseaseContextSignals,
  programs,
  oracleVersions,
  assays,
  experimentRecommendations,
  assayResults,
  literatureAnnotations,
  organizations,
  orgMembers,
  sharedAssets,
  materialEntities,
  materialProperties,
  materialsPrograms,
  materialsCampaigns,
  materialsOracleScores,
  materialsLearningGraph,
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
  type UsageMeter,
  type InsertUsageMeter,
  type CreditWallet,
  type InsertCreditWallet,
  type CreditTransaction,
  type InsertCreditTransaction,
  type DiseaseArea,
  type TargetVariant,
  type InsertTargetVariant,
  type DiseaseContextSignal,
  type InsertDiseaseContextSignal,
  type Program,
  type InsertProgram,
  type OracleVersion,
  type InsertOracleVersion,
  type Assay,
  type InsertAssay,
  type ExperimentRecommendation,
  type InsertExperimentRecommendation,
  type AssayResult,
  type InsertAssayResult,
  type LiteratureAnnotation,
  type InsertLiteratureAnnotation,
  type Organization,
  type InsertOrganization,
  type OrgMember,
  type InsertOrgMember,
  type SharedAsset,
  type InsertSharedAsset,
  type MaterialEntity,
  type InsertMaterialEntity,
  type MaterialProperty,
  type InsertMaterialProperty,
  type MaterialsProgram,
  type InsertMaterialsProgram,
  type MaterialsCampaign,
  type InsertMaterialsCampaign,
  type MaterialsOracleScore,
  type InsertMaterialsOracleScore,
  type MaterialsLearningGraphEntry,
  type InsertMaterialsLearningGraphEntry,
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

  getUsageMeters(filters?: { userId?: string; projectId?: string; campaignId?: string }): Promise<UsageMeter[]>;
  createUsageMeter(meter: InsertUsageMeter): Promise<UsageMeter>;
  getUsageSummary(projectId: string): Promise<{ resourceType: string; totalAmount: number; unit: string }[]>;

  getCreditWallet(ownerId: string, ownerType: "user" | "org"): Promise<CreditWallet | undefined>;
  createCreditWallet(wallet: InsertCreditWallet): Promise<CreditWallet>;
  updateCreditWallet(id: string, wallet: Partial<InsertCreditWallet>): Promise<CreditWallet | undefined>;

  getCreditTransactions(walletId: string): Promise<CreditTransaction[]>;
  createCreditTransaction(tx: InsertCreditTransaction): Promise<CreditTransaction>;

  getTargetVariants(targetId: string): Promise<TargetVariant[]>;
  getTargetVariant(id: string): Promise<TargetVariant | undefined>;
  createTargetVariant(variant: InsertTargetVariant): Promise<TargetVariant>;
  updateTargetVariant(id: string, variant: Partial<InsertTargetVariant>): Promise<TargetVariant | undefined>;
  deleteTargetVariant(id: string): Promise<void>;

  getDiseaseContextSignals(targetId: string): Promise<DiseaseContextSignal[]>;
  createDiseaseContextSignal(signal: InsertDiseaseContextSignal): Promise<DiseaseContextSignal>;

  getPrograms(ownerId?: string): Promise<Program[]>;
  getProgram(id: string): Promise<Program | undefined>;
  createProgram(program: InsertProgram): Promise<Program>;
  updateProgram(id: string, program: Partial<InsertProgram>): Promise<Program | undefined>;
  deleteProgram(id: string): Promise<void>;

  getOracleVersions(): Promise<OracleVersion[]>;
  getOracleVersion(id: string): Promise<OracleVersion | undefined>;
  createOracleVersion(version: InsertOracleVersion): Promise<OracleVersion>;

  getAssays(targetId?: string): Promise<Assay[]>;
  getAssay(id: string): Promise<Assay | undefined>;
  createAssay(assay: InsertAssay): Promise<Assay>;
  updateAssay(id: string, assay: Partial<InsertAssay>): Promise<Assay | undefined>;

  getExperimentRecommendations(campaignId: string): Promise<ExperimentRecommendation[]>;
  createExperimentRecommendation(rec: InsertExperimentRecommendation): Promise<ExperimentRecommendation>;
  updateExperimentRecommendation(id: string, rec: Partial<InsertExperimentRecommendation>): Promise<ExperimentRecommendation | undefined>;

  getAssayResults(assayId?: string, campaignId?: string): Promise<AssayResult[]>;
  createAssayResult(result: InsertAssayResult): Promise<AssayResult>;
  bulkCreateAssayResults(results: InsertAssayResult[]): Promise<AssayResult[]>;

  getLiteratureAnnotations(targetId?: string, moleculeId?: string): Promise<LiteratureAnnotation[]>;
  createLiteratureAnnotation(annotation: InsertLiteratureAnnotation): Promise<LiteratureAnnotation>;

  getOrganizations(): Promise<Organization[]>;
  getOrganization(id: string): Promise<Organization | undefined>;
  createOrganization(org: InsertOrganization): Promise<Organization>;
  updateOrganization(id: string, org: Partial<InsertOrganization>): Promise<Organization | undefined>;

  getOrgMembers(organizationId: string): Promise<OrgMember[]>;
  getOrgMembersByUser(userId: string): Promise<OrgMember[]>;
  createOrgMember(member: InsertOrgMember): Promise<OrgMember>;
  updateOrgMember(id: string, member: Partial<InsertOrgMember>): Promise<OrgMember | undefined>;
  deleteOrgMember(id: string): Promise<void>;

  getSharedAssets(sharedWithOrgId: string): Promise<SharedAsset[]>;
  createSharedAsset(asset: InsertSharedAsset): Promise<SharedAsset>;
  deleteSharedAsset(id: string): Promise<void>;

  getMaterialEntities(type?: string): Promise<MaterialEntity[]>;
  getMaterialEntity(id: string): Promise<MaterialEntity | undefined>;
  createMaterialEntity(entity: InsertMaterialEntity): Promise<MaterialEntity>;
  updateMaterialEntity(id: string, entity: Partial<InsertMaterialEntity>): Promise<MaterialEntity | undefined>;
  deleteMaterialEntity(id: string): Promise<void>;

  getMaterialProperties(materialId: string): Promise<MaterialProperty[]>;
  createMaterialProperty(property: InsertMaterialProperty): Promise<MaterialProperty>;

  getMaterialsPrograms(): Promise<MaterialsProgram[]>;
  getMaterialsProgram(id: string): Promise<MaterialsProgram | undefined>;
  createMaterialsProgram(program: InsertMaterialsProgram): Promise<MaterialsProgram>;
  updateMaterialsProgram(id: string, program: Partial<InsertMaterialsProgram>): Promise<MaterialsProgram | undefined>;
  deleteMaterialsProgram(id: string): Promise<void>;

  getMaterialsCampaigns(programId?: string): Promise<MaterialsCampaign[]>;
  getMaterialsCampaign(id: string): Promise<MaterialsCampaign | undefined>;
  createMaterialsCampaign(campaign: InsertMaterialsCampaign): Promise<MaterialsCampaign>;
  updateMaterialsCampaign(id: string, campaign: Partial<InsertMaterialsCampaign>): Promise<MaterialsCampaign | undefined>;
  deleteMaterialsCampaign(id: string): Promise<void>;

  getMaterialsOracleScores(campaignId: string): Promise<MaterialsOracleScore[]>;
  createMaterialsOracleScore(score: InsertMaterialsOracleScore): Promise<MaterialsOracleScore>;

  getMaterialsLearningGraph(campaignId?: string): Promise<MaterialsLearningGraphEntry[]>;
  createMaterialsLearningGraphEntry(entry: InsertMaterialsLearningGraphEntry): Promise<MaterialsLearningGraphEntry>;
  labelMaterialsLearningGraphEntry(id: string, label: string): Promise<MaterialsLearningGraphEntry | undefined>;
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

  async getUsageMeters(filters?: { userId?: string; projectId?: string; campaignId?: string }): Promise<UsageMeter[]> {
    const conditions = [];
    if (filters?.userId) conditions.push(eq(usageMeters.userId, filters.userId));
    if (filters?.projectId) conditions.push(eq(usageMeters.projectId, filters.projectId));
    if (filters?.campaignId) conditions.push(eq(usageMeters.campaignId, filters.campaignId));

    if (conditions.length === 0) {
      return db.select().from(usageMeters).orderBy(desc(usageMeters.createdAt)).limit(500);
    }
    return db.select().from(usageMeters).where(and(...conditions)).orderBy(desc(usageMeters.createdAt));
  }

  async createUsageMeter(meter: InsertUsageMeter): Promise<UsageMeter> {
    const result = await db.insert(usageMeters).values(meter).returning();
    return result[0];
  }

  async getUsageSummary(projectId: string): Promise<{ resourceType: string; totalAmount: number; unit: string }[]> {
    const result = await db
      .select({
        resourceType: usageMeters.resourceType,
        totalAmount: sum(usageMeters.amount),
        unit: usageMeters.unit,
      })
      .from(usageMeters)
      .where(eq(usageMeters.projectId, projectId))
      .groupBy(usageMeters.resourceType, usageMeters.unit);

    return result.map((row) => ({
      resourceType: row.resourceType,
      totalAmount: Number(row.totalAmount) || 0,
      unit: row.unit,
    }));
  }

  async getCreditWallet(ownerId: string, ownerType: "user" | "org"): Promise<CreditWallet | undefined> {
    const result = await db
      .select()
      .from(creditWallets)
      .where(and(eq(creditWallets.ownerId, ownerId), eq(creditWallets.ownerType, ownerType)))
      .limit(1);
    return result[0];
  }

  async createCreditWallet(wallet: InsertCreditWallet): Promise<CreditWallet> {
    const result = await db.insert(creditWallets).values(wallet).returning();
    return result[0];
  }

  async updateCreditWallet(id: string, wallet: Partial<InsertCreditWallet>): Promise<CreditWallet | undefined> {
    const result = await db
      .update(creditWallets)
      .set({ ...wallet, updatedAt: new Date() })
      .where(eq(creditWallets.id, id))
      .returning();
    return result[0];
  }

  async getCreditTransactions(walletId: string): Promise<CreditTransaction[]> {
    return db.select().from(creditTransactions).where(eq(creditTransactions.walletId, walletId)).orderBy(desc(creditTransactions.createdAt));
  }

  async createCreditTransaction(tx: InsertCreditTransaction): Promise<CreditTransaction> {
    const result = await db.insert(creditTransactions).values(tx).returning();
    return result[0];
  }

  async getTargetVariants(targetId: string): Promise<TargetVariant[]> {
    return db.select().from(targetVariants).where(eq(targetVariants.targetId, targetId)).orderBy(desc(targetVariants.createdAt));
  }

  async getTargetVariant(id: string): Promise<TargetVariant | undefined> {
    const result = await db.select().from(targetVariants).where(eq(targetVariants.id, id)).limit(1);
    return result[0];
  }

  async createTargetVariant(variant: InsertTargetVariant): Promise<TargetVariant> {
    const result = await db.insert(targetVariants).values(variant).returning();
    return result[0];
  }

  async updateTargetVariant(id: string, variant: Partial<InsertTargetVariant>): Promise<TargetVariant | undefined> {
    const result = await db.update(targetVariants).set(variant).where(eq(targetVariants.id, id)).returning();
    return result[0];
  }

  async deleteTargetVariant(id: string): Promise<void> {
    await db.delete(targetVariants).where(eq(targetVariants.id, id));
  }

  async getDiseaseContextSignals(targetId: string): Promise<DiseaseContextSignal[]> {
    return db.select().from(diseaseContextSignals).where(eq(diseaseContextSignals.targetId, targetId)).orderBy(desc(diseaseContextSignals.createdAt));
  }

  async createDiseaseContextSignal(signal: InsertDiseaseContextSignal): Promise<DiseaseContextSignal> {
    const result = await db.insert(diseaseContextSignals).values(signal).returning();
    return result[0];
  }

  async getPrograms(ownerId?: string): Promise<Program[]> {
    if (ownerId) {
      return db.select().from(programs).where(eq(programs.ownerId, ownerId)).orderBy(desc(programs.updatedAt));
    }
    return db.select().from(programs).orderBy(desc(programs.updatedAt));
  }

  async getProgram(id: string): Promise<Program | undefined> {
    const result = await db.select().from(programs).where(eq(programs.id, id)).limit(1);
    return result[0];
  }

  async createProgram(program: InsertProgram): Promise<Program> {
    const result = await db.insert(programs).values(program).returning();
    return result[0];
  }

  async updateProgram(id: string, program: Partial<InsertProgram>): Promise<Program | undefined> {
    const result = await db.update(programs).set({ ...program, updatedAt: new Date() }).where(eq(programs.id, id)).returning();
    return result[0];
  }

  async deleteProgram(id: string): Promise<void> {
    await db.delete(programs).where(eq(programs.id, id));
  }

  async getOracleVersions(): Promise<OracleVersion[]> {
    return db.select().from(oracleVersions).orderBy(desc(oracleVersions.createdAt));
  }

  async getOracleVersion(id: string): Promise<OracleVersion | undefined> {
    const result = await db.select().from(oracleVersions).where(eq(oracleVersions.id, id)).limit(1);
    return result[0];
  }

  async createOracleVersion(version: InsertOracleVersion): Promise<OracleVersion> {
    const result = await db.insert(oracleVersions).values(version).returning();
    return result[0];
  }

  async getAssays(targetId?: string): Promise<Assay[]> {
    if (targetId) {
      return db.select().from(assays).where(eq(assays.targetId, targetId)).orderBy(desc(assays.createdAt));
    }
    return db.select().from(assays).orderBy(desc(assays.createdAt));
  }

  async getAssay(id: string): Promise<Assay | undefined> {
    const result = await db.select().from(assays).where(eq(assays.id, id)).limit(1);
    return result[0];
  }

  async createAssay(assay: InsertAssay): Promise<Assay> {
    const result = await db.insert(assays).values(assay).returning();
    return result[0];
  }

  async updateAssay(id: string, assay: Partial<InsertAssay>): Promise<Assay | undefined> {
    const result = await db.update(assays).set(assay).where(eq(assays.id, id)).returning();
    return result[0];
  }

  async getExperimentRecommendations(campaignId: string): Promise<ExperimentRecommendation[]> {
    return db.select().from(experimentRecommendations).where(eq(experimentRecommendations.campaignId, campaignId)).orderBy(desc(experimentRecommendations.createdAt));
  }

  async createExperimentRecommendation(rec: InsertExperimentRecommendation): Promise<ExperimentRecommendation> {
    const result = await db.insert(experimentRecommendations).values(rec).returning();
    return result[0];
  }

  async updateExperimentRecommendation(id: string, rec: Partial<InsertExperimentRecommendation>): Promise<ExperimentRecommendation | undefined> {
    const result = await db.update(experimentRecommendations).set(rec).where(eq(experimentRecommendations.id, id)).returning();
    return result[0];
  }

  async getAssayResults(assayId?: string, campaignId?: string): Promise<AssayResult[]> {
    const conditions = [];
    if (assayId) conditions.push(eq(assayResults.assayId, assayId));
    if (campaignId) conditions.push(eq(assayResults.campaignId, campaignId));

    if (conditions.length === 0) {
      return db.select().from(assayResults).orderBy(desc(assayResults.createdAt)).limit(500);
    }
    return db.select().from(assayResults).where(and(...conditions)).orderBy(desc(assayResults.createdAt));
  }

  async createAssayResult(result: InsertAssayResult): Promise<AssayResult> {
    const res = await db.insert(assayResults).values(result).returning();
    return res[0];
  }

  async bulkCreateAssayResults(results: InsertAssayResult[]): Promise<AssayResult[]> {
    if (results.length === 0) return [];
    return db.insert(assayResults).values(results).returning();
  }

  async getLiteratureAnnotations(targetId?: string, moleculeId?: string): Promise<LiteratureAnnotation[]> {
    const conditions = [];
    if (targetId) conditions.push(eq(literatureAnnotations.targetId, targetId));
    if (moleculeId) conditions.push(eq(literatureAnnotations.moleculeId, moleculeId));

    if (conditions.length === 0) {
      return db.select().from(literatureAnnotations).orderBy(desc(literatureAnnotations.createdAt)).limit(500);
    }
    return db.select().from(literatureAnnotations).where(and(...conditions)).orderBy(desc(literatureAnnotations.createdAt));
  }

  async createLiteratureAnnotation(annotation: InsertLiteratureAnnotation): Promise<LiteratureAnnotation> {
    const result = await db.insert(literatureAnnotations).values(annotation).returning();
    return result[0];
  }

  async getOrganizations(): Promise<Organization[]> {
    return db.select().from(organizations).orderBy(desc(organizations.createdAt));
  }

  async getOrganization(id: string): Promise<Organization | undefined> {
    const result = await db.select().from(organizations).where(eq(organizations.id, id)).limit(1);
    return result[0];
  }

  async createOrganization(org: InsertOrganization): Promise<Organization> {
    const result = await db.insert(organizations).values(org).returning();
    return result[0];
  }

  async updateOrganization(id: string, org: Partial<InsertOrganization>): Promise<Organization | undefined> {
    const result = await db.update(organizations).set(org).where(eq(organizations.id, id)).returning();
    return result[0];
  }

  async getOrgMembers(organizationId: string): Promise<OrgMember[]> {
    return db.select().from(orgMembers).where(eq(orgMembers.organizationId, organizationId)).orderBy(desc(orgMembers.createdAt));
  }

  async getOrgMembersByUser(userId: string): Promise<OrgMember[]> {
    return db.select().from(orgMembers).where(eq(orgMembers.userId, userId)).orderBy(desc(orgMembers.createdAt));
  }

  async createOrgMember(member: InsertOrgMember): Promise<OrgMember> {
    const result = await db.insert(orgMembers).values(member).returning();
    return result[0];
  }

  async updateOrgMember(id: string, member: Partial<InsertOrgMember>): Promise<OrgMember | undefined> {
    const result = await db.update(orgMembers).set(member).where(eq(orgMembers.id, id)).returning();
    return result[0];
  }

  async deleteOrgMember(id: string): Promise<void> {
    await db.delete(orgMembers).where(eq(orgMembers.id, id));
  }

  async getSharedAssets(sharedWithOrgId: string): Promise<SharedAsset[]> {
    return db.select().from(sharedAssets).where(eq(sharedAssets.sharedWithOrgId, sharedWithOrgId)).orderBy(desc(sharedAssets.createdAt));
  }

  async createSharedAsset(asset: InsertSharedAsset): Promise<SharedAsset> {
    const result = await db.insert(sharedAssets).values(asset).returning();
    return result[0];
  }

  async deleteSharedAsset(id: string): Promise<void> {
    await db.delete(sharedAssets).where(eq(sharedAssets.id, id));
  }

  async getMaterialEntities(type?: string): Promise<MaterialEntity[]> {
    if (type) {
      return db.select().from(materialEntities).where(eq(materialEntities.type, type as any)).orderBy(desc(materialEntities.createdAt));
    }
    return db.select().from(materialEntities).orderBy(desc(materialEntities.createdAt)).limit(500);
  }

  async getMaterialEntity(id: string): Promise<MaterialEntity | undefined> {
    const result = await db.select().from(materialEntities).where(eq(materialEntities.id, id)).limit(1);
    return result[0];
  }

  async createMaterialEntity(entity: InsertMaterialEntity): Promise<MaterialEntity> {
    const result = await db.insert(materialEntities).values(entity).returning();
    return result[0];
  }

  async updateMaterialEntity(id: string, entity: Partial<InsertMaterialEntity>): Promise<MaterialEntity | undefined> {
    const result = await db.update(materialEntities).set(entity).where(eq(materialEntities.id, id)).returning();
    return result[0];
  }

  async deleteMaterialEntity(id: string): Promise<void> {
    await db.delete(materialEntities).where(eq(materialEntities.id, id));
  }

  async getMaterialProperties(materialId: string): Promise<MaterialProperty[]> {
    return db.select().from(materialProperties).where(eq(materialProperties.materialId, materialId)).orderBy(desc(materialProperties.createdAt));
  }

  async createMaterialProperty(property: InsertMaterialProperty): Promise<MaterialProperty> {
    const result = await db.insert(materialProperties).values(property).returning();
    return result[0];
  }

  async getMaterialsPrograms(): Promise<MaterialsProgram[]> {
    return db.select().from(materialsPrograms).orderBy(desc(materialsPrograms.createdAt));
  }

  async getMaterialsProgram(id: string): Promise<MaterialsProgram | undefined> {
    const result = await db.select().from(materialsPrograms).where(eq(materialsPrograms.id, id)).limit(1);
    return result[0];
  }

  async createMaterialsProgram(program: InsertMaterialsProgram): Promise<MaterialsProgram> {
    const result = await db.insert(materialsPrograms).values(program).returning();
    return result[0];
  }

  async updateMaterialsProgram(id: string, program: Partial<InsertMaterialsProgram>): Promise<MaterialsProgram | undefined> {
    const result = await db.update(materialsPrograms).set({ ...program, updatedAt: new Date() }).where(eq(materialsPrograms.id, id)).returning();
    return result[0];
  }

  async deleteMaterialsProgram(id: string): Promise<void> {
    await db.delete(materialsPrograms).where(eq(materialsPrograms.id, id));
  }

  async getMaterialsCampaigns(programId?: string): Promise<MaterialsCampaign[]> {
    if (programId) {
      return db.select().from(materialsCampaigns).where(eq(materialsCampaigns.programId, programId)).orderBy(desc(materialsCampaigns.updatedAt));
    }
    return db.select().from(materialsCampaigns).orderBy(desc(materialsCampaigns.updatedAt));
  }

  async getMaterialsCampaign(id: string): Promise<MaterialsCampaign | undefined> {
    const result = await db.select().from(materialsCampaigns).where(eq(materialsCampaigns.id, id)).limit(1);
    return result[0];
  }

  async createMaterialsCampaign(campaign: InsertMaterialsCampaign): Promise<MaterialsCampaign> {
    const result = await db.insert(materialsCampaigns).values(campaign).returning();
    return result[0];
  }

  async updateMaterialsCampaign(id: string, campaign: Partial<InsertMaterialsCampaign>): Promise<MaterialsCampaign | undefined> {
    const result = await db.update(materialsCampaigns).set({ ...campaign, updatedAt: new Date() }).where(eq(materialsCampaigns.id, id)).returning();
    return result[0];
  }

  async deleteMaterialsCampaign(id: string): Promise<void> {
    await db.delete(materialsCampaigns).where(eq(materialsCampaigns.id, id));
  }

  async getMaterialsOracleScores(campaignId: string): Promise<MaterialsOracleScore[]> {
    return db.select().from(materialsOracleScores).where(eq(materialsOracleScores.campaignId, campaignId)).orderBy(desc(materialsOracleScores.createdAt));
  }

  async createMaterialsOracleScore(score: InsertMaterialsOracleScore): Promise<MaterialsOracleScore> {
    const result = await db.insert(materialsOracleScores).values(score).returning();
    return result[0];
  }

  async getMaterialsLearningGraph(campaignId?: string): Promise<MaterialsLearningGraphEntry[]> {
    if (campaignId) {
      return db.select().from(materialsLearningGraph).where(eq(materialsLearningGraph.campaignId, campaignId)).orderBy(desc(materialsLearningGraph.createdAt));
    }
    return db.select().from(materialsLearningGraph).orderBy(desc(materialsLearningGraph.createdAt)).limit(500);
  }

  async createMaterialsLearningGraphEntry(entry: InsertMaterialsLearningGraphEntry): Promise<MaterialsLearningGraphEntry> {
    const result = await db.insert(materialsLearningGraph).values(entry).returning();
    return result[0];
  }

  async labelMaterialsLearningGraphEntry(id: string, label: string): Promise<MaterialsLearningGraphEntry | undefined> {
    const result = await db.update(materialsLearningGraph).set({ label, labeledAt: new Date() }).where(eq(materialsLearningGraph.id, id)).returning();
    return result[0];
  }
}

export const storage = new DatabaseStorage();
