import { db } from "./db";
import { supabaseSyncService } from "./services/supabase-sync";
import { eq, and, or, desc, sql, count, avg, sum, inArray } from "drizzle-orm";
import {
  projects,
  projectMolecules,
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
  sshConfigs,
  companies,
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
  assayTargets,
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
  assayPanels,
  assayPanelTargets,
  moaNodes,
  moaEdges,
  pipelineTemplates,
  pipelineTemplateTargets,
  compoundAssets,
  diseaseTargetMappings,
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
  type SshConfig,
  type InsertSshConfig,
  type Company,
  type InsertCompany,
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
  type MaterialVariant,
  type InsertMaterialVariant,
  materialVariants,
  type MaterialsProgram,
  type InsertMaterialsProgram,
  type MaterialsCampaign,
  type InsertMaterialsCampaign,
  type MaterialsOracleScore,
  type InsertMaterialsOracleScore,
  type MaterialsLearningGraphEntry,
  type InsertMaterialsLearningGraphEntry,
  type ProcessingJob,
  type InsertProcessingJob,
  type ProcessingJobRun,
  type InsertProcessingJobRun,
  type ProcessingJobEvent,
  type InsertProcessingJobEvent,
  type MaterialsCampaignAggregate,
  type InsertMaterialsCampaignAggregate,
  type MaterialVariantMetric,
  type InsertMaterialVariantMetric,
  processingJobs,
  processingJobRuns,
  processingJobEvents,
  materialsCampaignAggregates,
  materialVariantMetrics,
  jobArtifacts,
  importTemplates,
  importJobs,
  type JobArtifact,
  type InsertJobArtifact,
  type ImportTemplate,
  type InsertImportTemplate,
  type ImportJob,
  type InsertImportJob,
  canonicalMolecules,
  moleculeDescriptors,
  moleculeFingerprints,
  hitLists,
  hitListItems,
  canonicalAssays,
  canonicalAssayResults,
  targetAssets,
  canonicalMaterials,
  canonicalMaterialVariants,
  canonicalMaterialProperties,
  simulationRuns,
  manufacturabilityScores,
  type CanonicalMolecule,
  type InsertCanonicalMolecule,
  type MoleculeDescriptor,
  type InsertMoleculeDescriptor,
  type MoleculeFingerprint,
  type InsertMoleculeFingerprint,
  type HitList,
  type InsertHitList,
  type HitListItem,
  type InsertHitListItem,
  type CanonicalAssay,
  type InsertCanonicalAssay,
  type CanonicalAssayResult,
  type InsertCanonicalAssayResult,
  type TargetAsset,
  type InsertTargetAsset,
  type CanonicalMaterial,
  type InsertCanonicalMaterial,
  type CanonicalMaterialVariant,
  type InsertCanonicalMaterialVariant,
  type CanonicalMaterialProperty,
  type InsertCanonicalMaterialProperty,
  type SimulationRun,
  type InsertSimulationRun,
  type ManufacturabilityScore,
  type InsertManufacturabilityScore,
  type CompoundAsset,
  type InsertCompoundAsset,
  type AssayPanel,
  type InsertAssayPanel,
  type AssayPanelTarget,
  type InsertAssayPanelTarget,
  type PipelineTemplate,
  type InsertPipelineTemplate,
  type PipelineTemplateTarget,
  type InsertPipelineTemplateTarget,
  type MoaNode,
  type InsertMoaNode,
  type MoaEdge,
  type InsertMoaEdge,
  activityLogs,
  type ActivityLog,
  type InsertActivityLog,
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
  getDiseases(): Promise<{ disease: string; count: number }[]>;
  getTargetsWithDiseases(disease?: string): Promise<(Target & { diseases: string[] })[]>;

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
  getMoleculeScoresByMolecule(moleculeId: string): Promise<MoleculeScore[]>;
  getAssayResultsByMolecule(moleculeId: string): Promise<(AssayResult & { assayName?: string })[]>;
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

  getDrugDashboardStats(): Promise<{
    moleculesScreened: number;
    hitsIdentified: number;
    activeTargets: number;
    assaysUploaded: number;
    activeCampaigns: number;
  }>;

  getMaterialsDashboardStats(): Promise<{
    materialVariantsEvaluated: number;
    propertiesPredicted: number;
    manufacturableCandidates: number;
    activePipelines: number;
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
  getComputeNodesByTier(tier: string): Promise<ComputeNode[]>;
  getDefaultComputeNode(tier: string): Promise<ComputeNode | undefined>;
  createComputeNode(node: InsertComputeNode): Promise<ComputeNode>;
  updateComputeNode(id: string, node: Partial<InsertComputeNode>): Promise<ComputeNode | undefined>;
  deleteComputeNode(id: string): Promise<void>;

  getSshConfigs(): Promise<SshConfig[]>;
  getSshConfig(id: string): Promise<SshConfig | undefined>;
  createSshConfig(config: InsertSshConfig): Promise<SshConfig>;
  updateSshConfig(id: string, config: Partial<InsertSshConfig>): Promise<SshConfig | undefined>;
  deleteSshConfig(id: string): Promise<void>;
  updateSshConfigStatus(id: string, status: string, lastConnected?: Date): Promise<SshConfig | undefined>;

  getCompanies(): Promise<Company[]>;
  getCompany(id: string): Promise<Company | undefined>;
  createCompany(company: InsertCompany): Promise<Company>;
  updateCompany(id: string, company: Partial<InsertCompany>): Promise<Company | undefined>;
  deleteCompany(id: string): Promise<void>;

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

  getAssays(filters?: { targetId?: string; companyId?: string; projectId?: string; category?: string }): Promise<Assay[]>;
  getAssay(id: string): Promise<Assay | undefined>;
  getAssayWithDetails(id: string): Promise<(Assay & { resultsCount: number; targets: { targetId: string; targetName: string; weight: number; role: string }[] }) | undefined>;
  getAssayWithResultsCount(id: string): Promise<(Assay & { resultsCount: number }) | undefined>;
  createAssay(assay: InsertAssay): Promise<Assay>;
  createAssayWithTargets(assay: InsertAssay, targetMappings: { targetId: string; weight: number; role: string }[]): Promise<Assay>;
  updateAssay(id: string, assay: Partial<InsertAssay>): Promise<Assay | undefined>;
  deleteAssay(id: string): Promise<void>;
  getAssayTargets(assayId: string): Promise<{ targetId: string; targetName: string; weight: number; role: string }[]>;

  getExperimentRecommendations(campaignId: string): Promise<ExperimentRecommendation[]>;
  createExperimentRecommendation(rec: InsertExperimentRecommendation): Promise<ExperimentRecommendation>;
  updateExperimentRecommendation(id: string, rec: Partial<InsertExperimentRecommendation>): Promise<ExperimentRecommendation | undefined>;

  getAssayResults(assayId?: string, campaignId?: string): Promise<AssayResult[]>;
  getAssayResultsWithMolecules(assayId: string, moleculeId?: string): Promise<(AssayResult & { molecule: Molecule | null })[]>;
  createAssayResult(result: InsertAssayResult): Promise<AssayResult>;
  bulkCreateAssayResults(results: InsertAssayResult[]): Promise<AssayResult[]>;
  getMoleculeBySmiles(smiles: string): Promise<Molecule | undefined>;
  addMoleculeToProject(projectId: string, moleculeId: string): Promise<boolean>;
  getHitCandidates(campaignId: string, filters?: { minOracleScore?: number; maxOracleScore?: number; maxSynthesisComplexity?: number; ipSafeOnly?: boolean; hasAssayData?: boolean }): Promise<(MoleculeScore & { molecule: Molecule | null; lastAssayOutcome?: string; bestAssayValue?: number })[]>;

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

  getMaterialEntities(type?: string, limit?: number, offset?: number): Promise<{ materials: MaterialEntity[], total: number }>;
  getMaterialEntitiesByTypes(types?: string[], limit?: number, offset?: number): Promise<{ materials: MaterialEntity[], total: number }>;
  getMaterialEntity(id: string): Promise<MaterialEntity | undefined>;
  createMaterialEntity(entity: InsertMaterialEntity): Promise<MaterialEntity>;
  updateMaterialEntity(id: string, entity: Partial<InsertMaterialEntity>): Promise<MaterialEntity | undefined>;
  deleteMaterialEntity(id: string): Promise<void>;

  getMaterialProperties(materialId: string): Promise<MaterialProperty[]>;
  createMaterialProperty(property: InsertMaterialProperty): Promise<MaterialProperty>;

  getMaterialVariants(materialId: string): Promise<MaterialVariant[]>;
  getAllMaterialVariants(limit?: number, offset?: number): Promise<{ variants: MaterialVariant[], total: number }>;
  getMaterialVariant(id: string): Promise<MaterialVariant | undefined>;
  createMaterialVariant(variant: InsertMaterialVariant): Promise<MaterialVariant>;
  updateMaterialVariant(id: string, variant: Partial<InsertMaterialVariant>): Promise<MaterialVariant | undefined>;
  deleteMaterialVariant(id: string): Promise<void>;
  batchCreateMaterialVariants(variants: InsertMaterialVariant[]): Promise<MaterialVariant[]>;

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

  getSarSeries(campaignId: string): Promise<{ seriesId: string | null; scaffoldId: string | null; molecules: Molecule[]; assaySummary: { count: number; meanValue: number | null; bestValue: number | null }; scoreRanges: { minOracle: number | null; maxOracle: number | null } }[]>;
  getSarMoleculeDetails(campaignId: string, moleculeId: string): Promise<{ molecule: Molecule; analogs: Molecule[]; assayValues: { assayId: string; assayName: string; value: number; outcome: string | null }[]; predictedVsExperimental: { predictedScore: number | null; experimentalValue: number | null } } | null>;

  getMultiTargetSar(campaignId: string): Promise<{
    molecules: {
      id: string;
      smiles: string;
      seriesId: string | null;
      scaffoldId: string | null;
      oracleScore: number | null;
      targetScores: { targetId: string; targetName: string; predictedScore: number | null; experimentalValue: number | null; safetyFlag: boolean }[];
      compositeScore: number | null;
    }[];
    targets: { id: string; name: string; role: string }[];
    series: { seriesId: string; improvesMultipleTargets: boolean; degradesSafety: boolean }[];
  }>;

  getAssayPanels(campaignId?: string): Promise<AssayPanel[]>;
  getAssayPanel(id: string): Promise<(AssayPanel & { targets: (AssayPanelTarget & { target: Target })[] }) | undefined>;
  createAssayPanel(panel: InsertAssayPanel, targetIds: { targetId: string; role: string }[]): Promise<AssayPanel>;
  deleteAssayPanel(id: string): Promise<void>;
  getAssayPanelResults(panelId: string): Promise<{
    molecules: { id: string; smiles: string; name: string | null }[];
    targets: { id: string; name: string; role: string }[];
    matrix: { moleculeId: string; targetId: string; value: number | null; outcomeLabel: string | null }[];
    summary: { totalMolecules: number; balancedActives: number; selectiveCompounds: number };
  }>;
  uploadAssayPanelResults(panelId: string, results: { moleculeId?: string; smiles?: string; targetId: string; value: number; concentration?: number; outcomeLabel?: string }[]): Promise<{ imported: number; warnings: string[] }>;

  getMoaGraph(campaignId?: string): Promise<{ nodes: MoaNode[]; edges: MoaEdge[] }>;
  getMoaSubgraph(targetId: string): Promise<{ nodes: MoaNode[]; edges: MoaEdge[] }>;
  createMoaNode(node: InsertMoaNode): Promise<MoaNode>;
  createMoaEdge(edge: InsertMoaEdge): Promise<MoaEdge>;

  getPipelineTemplates(domain?: string): Promise<PipelineTemplate[]>;
  getPipelineTemplate(id: string): Promise<(PipelineTemplate & { targets: PipelineTemplateTarget[] }) | undefined>;
  createPipelineTemplate(template: InsertPipelineTemplate, targets?: InsertPipelineTemplateTarget[]): Promise<PipelineTemplate>;
  deletePipelineTemplate(id: string): Promise<void>;
  seedBuiltInTemplates(): Promise<void>;

  getProcessingJobs(filters?: { status?: string; type?: string; campaignId?: string; materialsCampaignId?: string; limit?: number; offset?: number }): Promise<{ jobs: ProcessingJob[]; total: number }>;
  getProcessingJob(id: string): Promise<ProcessingJob | undefined>;
  createProcessingJob(job: InsertProcessingJob): Promise<ProcessingJob>;
  updateProcessingJob(id: string, job: Partial<InsertProcessingJob>): Promise<ProcessingJob | undefined>;
  updateProcessingJobProgress(id: string, itemsCompleted: number, checkpointData?: unknown): Promise<ProcessingJob | undefined>;
  getProcessingJobRuns(jobId: string): Promise<ProcessingJobRun[]>;
  createProcessingJobRun(run: InsertProcessingJobRun): Promise<ProcessingJobRun>;
  getProcessingJobEvents(jobId: string): Promise<ProcessingJobEvent[]>;
  createProcessingJobEvent(event: InsertProcessingJobEvent): Promise<ProcessingJobEvent>;

  getJobArtifacts(jobId: string): Promise<JobArtifact[]>;
  getArtifactsByCampaign(campaignId: string, domain: "drug" | "materials"): Promise<JobArtifact[]>;
  createJobArtifact(artifact: InsertJobArtifact): Promise<JobArtifact>;
  createJobArtifactsBatch(artifacts: InsertJobArtifact[]): Promise<JobArtifact[]>;

  getMaterialsCampaignAggregate(campaignId: string): Promise<MaterialsCampaignAggregate | undefined>;
  upsertMaterialsCampaignAggregate(aggregate: InsertMaterialsCampaignAggregate): Promise<MaterialsCampaignAggregate>;
  getMaterialVariantMetrics(campaignId: string, limit?: number): Promise<MaterialVariantMetric[]>;
  upsertMaterialVariantMetric(metric: InsertMaterialVariantMetric): Promise<MaterialVariantMetric>;

  getImportTemplates(domain?: string, importType?: string, organizationId?: string): Promise<ImportTemplate[]>;
  getImportTemplate(id: string): Promise<ImportTemplate | undefined>;
  createImportTemplate(template: InsertImportTemplate): Promise<ImportTemplate>;
  updateImportTemplate(id: string, template: Partial<InsertImportTemplate>): Promise<ImportTemplate | undefined>;
  deleteImportTemplate(id: string): Promise<void>;

  getImportJobs(filters?: { domain?: string; importType?: string; status?: string; organizationId?: string }): Promise<ImportJob[]>;
  getImportJob(id: string): Promise<ImportJob | undefined>;
  createImportJob(job: InsertImportJob): Promise<ImportJob>;
  updateImportJob(id: string, job: Partial<InsertImportJob>): Promise<ImportJob | undefined>;

  getCanonicalMolecules(companyId?: string): Promise<CanonicalMolecule[]>;
  getCanonicalMoleculeByInchikey(inchikey: string, companyId?: string): Promise<CanonicalMolecule | undefined>;
  createCanonicalMolecule(molecule: InsertCanonicalMolecule): Promise<CanonicalMolecule>;
  bulkCreateCanonicalMolecules(molecules: InsertCanonicalMolecule[]): Promise<CanonicalMolecule[]>;

  getCanonicalMaterials(companyId?: string): Promise<CanonicalMaterial[]>;
  getCanonicalMaterialByHash(materialHash: string, companyId?: string): Promise<CanonicalMaterial | undefined>;
  createCanonicalMaterial(material: InsertCanonicalMaterial): Promise<CanonicalMaterial>;
  bulkCreateCanonicalMaterials(materials: InsertCanonicalMaterial[]): Promise<CanonicalMaterial[]>;

  createHitList(hitList: InsertHitList): Promise<HitList>;
  bulkCreateHitListItems(items: InsertHitListItem[]): Promise<HitListItem[]>;

  createCanonicalAssay(assay: InsertCanonicalAssay): Promise<CanonicalAssay>;
  bulkCreateCanonicalAssayResults(results: InsertCanonicalAssayResult[]): Promise<CanonicalAssayResult[]>;

  createMoleculeDescriptor(descriptor: InsertMoleculeDescriptor): Promise<MoleculeDescriptor>;
  createMoleculeFingerprint(fingerprint: InsertMoleculeFingerprint): Promise<MoleculeFingerprint>;
  createTargetAsset(asset: InsertTargetAsset): Promise<TargetAsset>;

  createCanonicalMaterialVariant(variant: InsertCanonicalMaterialVariant): Promise<CanonicalMaterialVariant>;
  bulkCreateCanonicalMaterialVariants(variants: InsertCanonicalMaterialVariant[]): Promise<CanonicalMaterialVariant[]>;
  getCanonicalMaterialVariantByHash(variantHash: string): Promise<CanonicalMaterialVariant | undefined>;

  createCanonicalMaterialProperty(property: InsertCanonicalMaterialProperty): Promise<CanonicalMaterialProperty>;
  bulkCreateCanonicalMaterialProperties(properties: InsertCanonicalMaterialProperty[]): Promise<CanonicalMaterialProperty[]>;

  createSimulationRun(run: InsertSimulationRun): Promise<SimulationRun>;
  createManufacturabilityScore(score: InsertManufacturabilityScore): Promise<ManufacturabilityScore>;

  createCompoundAsset(asset: InsertCompoundAsset): Promise<CompoundAsset>;
  bulkCreateCompoundAssets(assets: InsertCompoundAsset[]): Promise<CompoundAsset[]>;
  getCompoundAsset(id: string): Promise<CompoundAsset | undefined>;
  getCompoundAssetsByMolecule(moleculeId: string): Promise<CompoundAsset[]>;
  getCompoundAssetByTypeAndMolecule(moleculeId: string, assetType: string): Promise<CompoundAsset | undefined>;
  deleteCompoundAsset(id: string): Promise<void>;

  getBuiltInMolecules(limit?: number, offset?: number): Promise<{ id: string; inchikey: string; canonicalSmiles: string }[]>;
  countBuiltInMolecules(): Promise<number>;
  markMoleculesAsBuiltIn(moleculeIds: string[]): Promise<void>;
}

export class DatabaseStorage implements IStorage {
  async getProjects(userId: string): Promise<Project[]> {
    return db.select().from(projects).where(
      or(eq(projects.ownerId, userId), eq(projects.isDemo, true))
    ).orderBy(desc(projects.updatedAt));
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

  async getDiseases(): Promise<{ disease: string; count: number }[]> {
    const result = await db
      .select({
        disease: diseaseTargetMappings.disease,
        count: count(diseaseTargetMappings.id),
      })
      .from(diseaseTargetMappings)
      .groupBy(diseaseTargetMappings.disease)
      .orderBy(desc(count(diseaseTargetMappings.id)));
    return result.map(r => ({ disease: r.disease, count: Number(r.count) }));
  }

  async getTargetsWithDiseases(disease?: string): Promise<(Target & { diseases: string[] })[]> {
    let targetIds: string[] = [];
    
    if (disease) {
      const mappings = await db
        .select({ targetId: diseaseTargetMappings.targetId })
        .from(diseaseTargetMappings)
        .where(eq(diseaseTargetMappings.disease, disease));
      targetIds = mappings.map(m => m.targetId);
      if (targetIds.length === 0) return [];
    }
    
    const targetsData = disease && targetIds.length > 0
      ? await db.select().from(targets).where(inArray(targets.id, targetIds)).orderBy(desc(targets.createdAt)).limit(500)
      : await db.select().from(targets).orderBy(desc(targets.createdAt)).limit(500);
    
    const targetIdsList = targetsData.map(t => t.id);
    if (targetIdsList.length === 0) return [];
    
    const allMappings = await db
      .select()
      .from(diseaseTargetMappings)
      .where(inArray(diseaseTargetMappings.targetId, targetIdsList));
    
    const diseaseMap = new Map<string, string[]>();
    for (const m of allMappings) {
      if (!diseaseMap.has(m.targetId)) {
        diseaseMap.set(m.targetId, []);
      }
      if (!diseaseMap.get(m.targetId)!.includes(m.disease)) {
        diseaseMap.get(m.targetId)!.push(m.disease);
      }
    }
    
    return targetsData.map(t => ({
      ...t,
      diseases: diseaseMap.get(t.id) || [],
    }));
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

  async getMoleculeScoresByMolecule(moleculeId: string): Promise<MoleculeScore[]> {
    return db.select().from(moleculeScores).where(eq(moleculeScores.moleculeId, moleculeId)).orderBy(desc(moleculeScores.createdAt));
  }

  async getAssayResultsByMolecule(moleculeId: string): Promise<(AssayResult & { assayName?: string })[]> {
    const resultsWithAssays = await db
      .select({
        result: assayResults,
        assay: assays,
      })
      .from(assayResults)
      .leftJoin(assays, eq(assayResults.assayId, assays.id))
      .where(eq(assayResults.moleculeId, moleculeId))
      .orderBy(desc(assayResults.createdAt));

    return resultsWithAssays.map((row) => ({
      ...row.result,
      assayName: row.assay?.name,
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

  async getDrugDashboardStats(): Promise<{
    moleculesScreened: number;
    hitsIdentified: number;
    activeTargets: number;
    assaysUploaded: number;
    activeCampaigns: number;
  }> {
    let externalSmilesCount = 0;
    try {
      const doStats = await supabaseSyncService.getDigitalOceanSmilesStats();
      if (doStats.success && doStats.totalRecords > 0) {
        externalSmilesCount = doStats.totalRecords;
      }
    } catch (err) {
      console.error("Failed to fetch DigitalOcean SMILES count:", err);
    }

    const [internalMoleculeCount] = await db.select({ count: count() }).from(molecules);
    const moleculesScreened = externalSmilesCount > 0 ? externalSmilesCount : Number(internalMoleculeCount.count);

    const [hitsCount] = await db
      .select({ count: count() })
      .from(moleculeScores)
      .where(sql`${moleculeScores.oracleScore} >= 0.7`);
    
    const [targetCount] = await db.select({ count: count() }).from(targets);
    
    const [assayCount] = await db.select({ count: count() }).from(assays);
    
    const [activeCampaignCount] = await db
      .select({ count: count() })
      .from(campaigns)
      .where(eq(campaigns.status, "running"));

    return {
      moleculesScreened,
      hitsIdentified: Number(hitsCount.count),
      activeTargets: Number(targetCount.count),
      assaysUploaded: Number(assayCount.count),
      activeCampaigns: Number(activeCampaignCount.count),
    };
  }

  async getMaterialsDashboardStats(): Promise<{
    materialVariantsEvaluated: number;
    propertiesPredicted: number;
    manufacturableCandidates: number;
    activePipelines: number;
  }> {
    const [materialCount] = await db.select({ count: count() }).from(materialEntities);
    
    const [propertyCount] = await db.select({ count: count() }).from(materialProperties);
    
    const [manufacturableCount] = await db
      .select({ count: count() })
      .from(materialsOracleScores)
      .where(sql`${materialsOracleScores.synthesisFeasibility} >= 0.7`);
    
    const [pipelineCount] = await db
      .select({ count: count() })
      .from(materialsCampaigns)
      .where(eq(materialsCampaigns.status, "running"));

    return {
      materialVariantsEvaluated: Number(materialCount.count),
      propertiesPredicted: Number(propertyCount.count),
      manufacturableCandidates: Number(manufacturableCount.count),
      activePipelines: Number(pipelineCount.count),
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

  async getComputeNodesByTier(tier: string): Promise<ComputeNode[]> {
    return db.select().from(computeNodes)
      .where(and(eq(computeNodes.tier, tier as any), eq(computeNodes.status, "active")))
      .orderBy(desc(computeNodes.isDefault), desc(computeNodes.createdAt));
  }

  async getDefaultComputeNode(tier: string): Promise<ComputeNode | undefined> {
    const result = await db.select().from(computeNodes)
      .where(and(
        eq(computeNodes.tier, tier as any),
        eq(computeNodes.isDefault, true),
        eq(computeNodes.status, "active")
      ))
      .limit(1);
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

  async getSshConfigs(): Promise<SshConfig[]> {
    return db.select().from(sshConfigs).orderBy(desc(sshConfigs.createdAt));
  }

  async getSshConfig(id: string): Promise<SshConfig | undefined> {
    const result = await db.select().from(sshConfigs).where(eq(sshConfigs.id, id)).limit(1);
    return result[0];
  }

  async createSshConfig(config: InsertSshConfig): Promise<SshConfig> {
    const result = await db.insert(sshConfigs).values(config).returning();
    return result[0];
  }

  async updateSshConfig(id: string, config: Partial<InsertSshConfig>): Promise<SshConfig | undefined> {
    const result = await db.update(sshConfigs).set({ ...config, updatedAt: new Date() }).where(eq(sshConfigs.id, id)).returning();
    return result[0];
  }

  async deleteSshConfig(id: string): Promise<void> {
    await db.delete(sshConfigs).where(eq(sshConfigs.id, id));
  }

  async updateSshConfigStatus(id: string, status: string, lastConnected?: Date): Promise<SshConfig | undefined> {
    const updates: any = { status: status as any, updatedAt: new Date() };
    if (lastConnected) updates.lastConnected = lastConnected;
    const result = await db.update(sshConfigs).set(updates).where(eq(sshConfigs.id, id)).returning();
    return result[0];
  }

  async getCompanies(): Promise<Company[]> {
    return db.select().from(companies).orderBy(desc(companies.createdAt));
  }

  async getCompany(id: string): Promise<Company | undefined> {
    const result = await db.select().from(companies).where(eq(companies.id, id)).limit(1);
    return result[0];
  }

  async createCompany(company: InsertCompany): Promise<Company> {
    const result = await db.insert(companies).values(company).returning();
    return result[0];
  }

  async updateCompany(id: string, company: Partial<InsertCompany>): Promise<Company | undefined> {
    const result = await db.update(companies).set({ ...company, updatedAt: new Date() }).where(eq(companies.id, id)).returning();
    return result[0];
  }

  async deleteCompany(id: string): Promise<void> {
    await db.delete(companies).where(eq(companies.id, id));
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

  async getAssays(filters?: { targetId?: string; companyId?: string; projectId?: string; category?: string }): Promise<Assay[]> {
    const conditions = [];
    if (filters?.targetId) conditions.push(eq(assays.targetId, filters.targetId));
    if (filters?.companyId) conditions.push(eq(assays.companyId, filters.companyId));
    if (filters?.projectId) conditions.push(eq(assays.projectId, filters.projectId));
    if (filters?.category) conditions.push(eq(assays.category, filters.category as any));
    
    if (conditions.length === 0) {
      return db.select().from(assays).orderBy(desc(assays.createdAt));
    }
    return db.select().from(assays).where(and(...conditions)).orderBy(desc(assays.createdAt));
  }

  async getAssay(id: string): Promise<Assay | undefined> {
    const result = await db.select().from(assays).where(eq(assays.id, id)).limit(1);
    return result[0];
  }

  async getAssayWithResultsCount(id: string): Promise<(Assay & { resultsCount: number }) | undefined> {
    const assay = await this.getAssay(id);
    if (!assay) return undefined;
    
    const countResult = await db.select({ count: count() }).from(assayResults).where(eq(assayResults.assayId, id));
    const resultsCount = countResult[0]?.count ?? 0;
    
    return { ...assay, resultsCount };
  }

  async getAssayWithDetails(id: string): Promise<(Assay & { resultsCount: number; targets: { targetId: string; targetName: string; weight: number; role: string }[] }) | undefined> {
    const assay = await this.getAssayWithResultsCount(id);
    if (!assay) return undefined;
    
    const targetMappings = await this.getAssayTargets(id);
    return { ...assay, targets: targetMappings };
  }

  async getAssayTargets(assayId: string): Promise<{ targetId: string; targetName: string; weight: number; role: string }[]> {
    const result = await db
      .select({
        targetId: assayTargets.targetId,
        targetName: targets.name,
        weight: assayTargets.weight,
        role: assayTargets.role,
      })
      .from(assayTargets)
      .innerJoin(targets, eq(assayTargets.targetId, targets.id))
      .where(eq(assayTargets.assayId, assayId));
    
    return result.map(r => ({
      targetId: r.targetId,
      targetName: r.targetName,
      weight: r.weight ?? 1.0,
      role: r.role ?? 'primary',
    }));
  }

  async createAssay(assay: InsertAssay): Promise<Assay> {
    const result = await db.insert(assays).values(assay).returning();
    return result[0];
  }

  async createAssayWithTargets(assay: InsertAssay, targetMappings: { targetId: string; weight: number; role: string }[]): Promise<Assay> {
    const createdAssay = await this.createAssay(assay);
    
    if (targetMappings.length > 0) {
      await db.insert(assayTargets).values(
        targetMappings.map(tm => ({
          assayId: createdAssay.id,
          targetId: tm.targetId,
          weight: tm.weight,
          role: tm.role,
        }))
      );
    }
    
    return createdAssay;
  }

  async updateAssay(id: string, assay: Partial<InsertAssay>): Promise<Assay | undefined> {
    const result = await db.update(assays).set({ ...assay, updatedAt: new Date() }).where(eq(assays.id, id)).returning();
    return result[0];
  }

  async deleteAssay(id: string): Promise<void> {
    await db.delete(assays).where(eq(assays.id, id));
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

  async getAssayResultsWithMolecules(assayId: string, moleculeId?: string): Promise<(AssayResult & { molecule: Molecule | null })[]> {
    const conditions = [eq(assayResults.assayId, assayId)];
    if (moleculeId) conditions.push(eq(assayResults.moleculeId, moleculeId));
    
    const results = await db
      .select({
        assayResult: assayResults,
        molecule: molecules,
      })
      .from(assayResults)
      .leftJoin(molecules, eq(assayResults.moleculeId, molecules.id))
      .where(and(...conditions))
      .orderBy(desc(assayResults.createdAt));
    
    return results.map(r => ({ ...r.assayResult, molecule: r.molecule }));
  }

  async getMoleculeBySmiles(smiles: string): Promise<Molecule | undefined> {
    const result = await db.select().from(molecules).where(eq(molecules.smiles, smiles)).limit(1);
    return result[0];
  }

  async addMoleculeToProject(projectId: string, moleculeId: string): Promise<boolean> {
    // Check if already linked
    const existing = await db.select()
      .from(projectMolecules)
      .where(and(
        eq(projectMolecules.projectId, projectId),
        eq(projectMolecules.moleculeId, moleculeId)
      ))
      .limit(1);
    
    if (existing.length === 0) {
      await db.insert(projectMolecules).values({
        projectId,
        moleculeId
      });
      return true; // Newly linked
    }
    return false; // Already linked
  }

  async getHitCandidates(campaignId: string, filters?: { minOracleScore?: number; maxOracleScore?: number; maxSynthesisComplexity?: number; ipSafeOnly?: boolean; hasAssayData?: boolean }): Promise<(MoleculeScore & { molecule: Molecule | null; lastAssayOutcome?: string; bestAssayValue?: number })[]> {
    const baseQuery = db
      .select({
        score: moleculeScores,
        molecule: molecules,
      })
      .from(moleculeScores)
      .leftJoin(molecules, eq(moleculeScores.moleculeId, molecules.id))
      .where(eq(moleculeScores.campaignId, campaignId))
      .orderBy(desc(moleculeScores.oracleScore))
      .limit(500);
    
    const results = await baseQuery;
    
    let filtered = results.map(r => ({ ...r.score, molecule: r.molecule, lastAssayOutcome: undefined as string | undefined, bestAssayValue: undefined as number | undefined }));
    
    if (filters?.minOracleScore !== undefined) {
      filtered = filtered.filter(r => (r.oracleScore ?? 0) >= filters.minOracleScore!);
    }
    if (filters?.maxOracleScore !== undefined) {
      filtered = filtered.filter(r => (r.oracleScore ?? 0) <= filters.maxOracleScore!);
    }
    if (filters?.maxSynthesisComplexity !== undefined) {
      filtered = filtered.filter(r => (r.synthesisComplexity ?? 0) <= filters.maxSynthesisComplexity!);
    }
    if (filters?.ipSafeOnly) {
      filtered = filtered.filter(r => !r.ipRiskFlag);
    }
    
    return filtered;
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

  async getMaterialEntities(type?: string, limit: number = 5000, offset: number = 0): Promise<{ materials: MaterialEntity[], total: number }> {
    const countResult = await db.select({ count: sql<number>`count(*)` }).from(materialEntities);
    const total = Number(countResult[0]?.count || 0);
    
    let query;
    if (type) {
      query = db.select().from(materialEntities).where(eq(materialEntities.type, type as any)).orderBy(desc(materialEntities.createdAt)).limit(limit).offset(offset);
    } else {
      query = db.select().from(materialEntities).orderBy(desc(materialEntities.createdAt)).limit(limit).offset(offset);
    }
    
    const materials = await query;
    return { materials, total };
  }

  async getMaterialEntitiesByTypes(types?: string[], limit: number = 5000, offset: number = 0): Promise<{ materials: MaterialEntity[], total: number }> {
    if (!types || types.length === 0) {
      return this.getMaterialEntities(undefined, limit, offset);
    }
    
    const countResult = await db.select({ count: sql<number>`count(*)` })
      .from(materialEntities)
      .where(inArray(materialEntities.type, types as any));
    const total = Number(countResult[0]?.count || 0);
    
    const materials = await db.select()
      .from(materialEntities)
      .where(inArray(materialEntities.type, types as any))
      .orderBy(desc(materialEntities.createdAt))
      .limit(limit)
      .offset(offset);
    
    return { materials, total };
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

  async getMaterialVariants(materialId: string): Promise<MaterialVariant[]> {
    return db.select().from(materialVariants).where(eq(materialVariants.materialId, materialId)).orderBy(desc(materialVariants.createdAt));
  }

  async getAllMaterialVariants(limit: number = 100, offset: number = 0): Promise<{ variants: MaterialVariant[], total: number }> {
    const [variants, countResult] = await Promise.all([
      db.select().from(materialVariants).orderBy(desc(materialVariants.createdAt)).limit(limit).offset(offset),
      db.select({ count: sql<number>`count(*)` }).from(materialVariants)
    ]);
    return { variants, total: Number(countResult[0]?.count || 0) };
  }

  async getMaterialVariant(id: string): Promise<MaterialVariant | undefined> {
    const result = await db.select().from(materialVariants).where(eq(materialVariants.id, id)).limit(1);
    return result[0];
  }

  async createMaterialVariant(variant: InsertMaterialVariant): Promise<MaterialVariant> {
    const result = await db.insert(materialVariants).values(variant).returning();
    return result[0];
  }

  async updateMaterialVariant(id: string, variant: Partial<InsertMaterialVariant>): Promise<MaterialVariant | undefined> {
    const result = await db.update(materialVariants).set(variant).where(eq(materialVariants.id, id)).returning();
    return result[0];
  }

  async deleteMaterialVariant(id: string): Promise<void> {
    await db.delete(materialVariants).where(eq(materialVariants.id, id));
  }

  async batchCreateMaterialVariants(variants: InsertMaterialVariant[]): Promise<MaterialVariant[]> {
    if (variants.length === 0) return [];
    const result = await db.insert(materialVariants).values(variants).returning();
    return result;
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

  async getSarSeries(campaignId: string): Promise<{ seriesId: string | null; scaffoldId: string | null; molecules: Molecule[]; assaySummary: { count: number; meanValue: number | null; bestValue: number | null }; scoreRanges: { minOracle: number | null; maxOracle: number | null } }[]> {
    const scoresData = await db
      .select()
      .from(moleculeScores)
      .where(eq(moleculeScores.campaignId, campaignId));

    if (scoresData.length === 0) return [];

    const moleculeIds = scoresData.map(s => s.moleculeId).filter((id): id is string => id !== null);
    if (moleculeIds.length === 0) return [];

    const moleculesData = await db
      .select()
      .from(molecules)
      .where(inArray(molecules.id, moleculeIds));

    const assayResultsData = await db
      .select()
      .from(assayResults)
      .where(inArray(assayResults.moleculeId, moleculeIds));

    const moleculeScoreMap = new Map(scoresData.map(s => [s.moleculeId, s]));
    const moleculeAssayMap = new Map<string, typeof assayResultsData>();
    for (const ar of assayResultsData) {
      if (ar.moleculeId) {
        if (!moleculeAssayMap.has(ar.moleculeId)) {
          moleculeAssayMap.set(ar.moleculeId, []);
        }
        moleculeAssayMap.get(ar.moleculeId)!.push(ar);
      }
    }

    const groupMap = new Map<string, { seriesId: string | null; scaffoldId: string | null; molecules: Molecule[]; oracleScores: number[]; assayValues: number[] }>();

    for (const mol of moleculesData) {
      const groupKey = mol.seriesId || mol.scaffoldId || `single-${mol.id}`;
      if (!groupMap.has(groupKey)) {
        groupMap.set(groupKey, { seriesId: mol.seriesId, scaffoldId: mol.scaffoldId, molecules: [], oracleScores: [], assayValues: [] });
      }
      const group = groupMap.get(groupKey)!;
      group.molecules.push(mol);

      const score = moleculeScoreMap.get(mol.id);
      if (score?.oracleScore) {
        group.oracleScores.push(Number(score.oracleScore));
      }

      const molAssays = moleculeAssayMap.get(mol.id) || [];
      for (const ar of molAssays) {
        if (ar.value !== null) {
          group.assayValues.push(Number(ar.value));
        }
      }
    }

    return Array.from(groupMap.values()).map(g => ({
      seriesId: g.seriesId,
      scaffoldId: g.scaffoldId,
      molecules: g.molecules,
      assaySummary: {
        count: g.assayValues.length,
        meanValue: g.assayValues.length > 0 ? g.assayValues.reduce((a, b) => a + b, 0) / g.assayValues.length : null,
        bestValue: g.assayValues.length > 0 ? Math.min(...g.assayValues) : null,
      },
      scoreRanges: {
        minOracle: g.oracleScores.length > 0 ? Math.min(...g.oracleScores) : null,
        maxOracle: g.oracleScores.length > 0 ? Math.max(...g.oracleScores) : null,
      },
    }));
  }

  async getSarMoleculeDetails(campaignId: string, moleculeId: string): Promise<{ molecule: Molecule; analogs: Molecule[]; assayValues: { assayId: string; assayName: string; value: number; outcome: string | null }[]; predictedVsExperimental: { predictedScore: number | null; experimentalValue: number | null } } | null> {
    if (!moleculeId) return null;

    const molResult = await db.select().from(molecules).where(eq(molecules.id, moleculeId)).limit(1);
    if (molResult.length === 0) return null;
    const molecule = molResult[0];

    let analogs: Molecule[] = [];
    if (molecule.seriesId) {
      analogs = await db.select().from(molecules)
        .where(and(eq(molecules.seriesId, molecule.seriesId), sql`${molecules.id} != ${moleculeId}`))
        .limit(20);
    } else if (molecule.scaffoldId) {
      analogs = await db.select().from(molecules)
        .where(and(eq(molecules.scaffoldId, molecule.scaffoldId), sql`${molecules.id} != ${moleculeId}`))
        .limit(20);
    }

    const assayResultsData = await db
      .select({
        assayId: assays.id,
        assayName: assays.name,
        value: assayResults.value,
        outcomeLabel: assayResults.outcomeLabel,
      })
      .from(assayResults)
      .innerJoin(assays, eq(assayResults.assayId, assays.id))
      .where(eq(assayResults.moleculeId, moleculeId));

    const assayValues = assayResultsData.map(ar => ({
      assayId: ar.assayId,
      assayName: ar.assayName,
      value: Number(ar.value),
      outcome: ar.outcomeLabel as string | null,
    }));

    const scoreResult = await db.select().from(moleculeScores)
      .where(and(eq(moleculeScores.moleculeId, moleculeId), eq(moleculeScores.campaignId, campaignId)))
      .limit(1);

    const predictedScore = scoreResult.length > 0 && scoreResult[0].oracleScore ? Number(scoreResult[0].oracleScore) : null;
    const experimentalValue = assayValues.length > 0 ? Math.min(...assayValues.map(v => v.value)) : null;

    return {
      molecule,
      analogs,
      assayValues,
      predictedVsExperimental: { predictedScore, experimentalValue },
    };
  }

  async getMultiTargetSar(campaignId: string): Promise<{
    molecules: {
      id: string;
      smiles: string;
      seriesId: string | null;
      scaffoldId: string | null;
      oracleScore: number | null;
      targetScores: { targetId: string; targetName: string; predictedScore: number | null; experimentalValue: number | null; safetyFlag: boolean }[];
      compositeScore: number | null;
    }[];
    targets: { id: string; name: string; role: string }[];
    series: { seriesId: string; improvesMultipleTargets: boolean; degradesSafety: boolean }[];
  }> {
    const campaign = await this.getCampaign(campaignId);
    if (!campaign) {
      return { molecules: [], targets: [], series: [] };
    }

    const scores = await db.select().from(moleculeScores).where(eq(moleculeScores.campaignId, campaignId));
    if (scores.length === 0) {
      return { molecules: [], targets: [], series: [] };
    }

    const molIds = Array.from(new Set(scores.map(s => s.moleculeId).filter((id): id is string => id !== null)));
    const molsData = molIds.length > 0 
      ? await db.select().from(molecules).where(inArray(molecules.id, molIds))
      : [];

    const panelData = await db.select().from(assayPanels).where(eq(assayPanels.campaignId, campaignId));
    const panelTargets: { id: string; name: string; role: string }[] = [];
    
    for (const panel of panelData) {
      const pts = await db
        .select({
          targetId: assayPanelTargets.targetId,
          role: assayPanelTargets.role,
          targetName: targets.name,
        })
        .from(assayPanelTargets)
        .innerJoin(targets, eq(assayPanelTargets.targetId, targets.id))
        .where(eq(assayPanelTargets.assayPanelId, panel.id));
      
      for (const pt of pts) {
        if (!panelTargets.find(t => t.id === pt.targetId)) {
          panelTargets.push({ id: pt.targetId, name: pt.targetName, role: pt.role || "primary" });
        }
      }
    }

    if (panelTargets.length === 0) {
      const allTargets = await db.select().from(targets).limit(5);
      for (const t of allTargets) {
        panelTargets.push({ id: t.id, name: t.name, role: "primary" });
      }
    }

    const assayResultsData = await db
      .select()
      .from(assayResults)
      .where(eq(assayResults.campaignId, campaignId));

    const resultsByMolecule = new Map<string, Map<string, { value: number; isSafety: boolean }>>();
    for (const ar of assayResultsData) {
      const assay = await db.select().from(assays).where(eq(assays.id, ar.assayId)).limit(1);
      if (assay.length === 0) continue;
      const targetId = assay[0].targetId;
      if (!targetId) continue;
      
      if (!resultsByMolecule.has(ar.moleculeId)) {
        resultsByMolecule.set(ar.moleculeId, new Map());
      }
      const targetRole = panelTargets.find(pt => pt.id === targetId)?.role || "primary";
      resultsByMolecule.get(ar.moleculeId)!.set(targetId, {
        value: ar.value,
        isSafety: targetRole === "safety",
      });
    }

    const moleculesResult: {
      id: string;
      smiles: string;
      seriesId: string | null;
      scaffoldId: string | null;
      oracleScore: number | null;
      targetScores: { targetId: string; targetName: string; predictedScore: number | null; experimentalValue: number | null; safetyFlag: boolean }[];
      compositeScore: number | null;
    }[] = [];

    const seriesMap = new Map<string, { improvesCount: number; degradesSafety: boolean }>();

    for (const mol of molsData) {
      const score = scores.find(s => s.moleculeId === mol.id);
      const oracleScore = score?.oracleScore ?? null;

      const targetScores: { targetId: string; targetName: string; predictedScore: number | null; experimentalValue: number | null; safetyFlag: boolean }[] = [];
      let compositeSum = 0;
      let compositeCount = 0;
      
      for (const pt of panelTargets) {
        const expData = resultsByMolecule.get(mol.id)?.get(pt.id);
        const experimentalValue = expData?.value ?? null;
        const predictedScore = oracleScore;
        const isSafety = pt.role === "safety";
        
        targetScores.push({
          targetId: pt.id,
          targetName: pt.name,
          predictedScore,
          experimentalValue,
          safetyFlag: isSafety,
        });

        if (experimentalValue !== null) {
          compositeSum += experimentalValue;
          compositeCount++;
        }
      }

      const compositeScore = compositeCount > 0 ? compositeSum / compositeCount : oracleScore;

      moleculesResult.push({
        id: mol.id,
        smiles: mol.smiles,
        seriesId: mol.seriesId,
        scaffoldId: mol.scaffoldId,
        oracleScore,
        targetScores,
        compositeScore,
      });

      if (mol.seriesId) {
        if (!seriesMap.has(mol.seriesId)) {
          seriesMap.set(mol.seriesId, { improvesCount: 0, degradesSafety: false });
        }
        const seriesData = seriesMap.get(mol.seriesId)!;
        const activeTargets = targetScores.filter(ts => ts.experimentalValue !== null && ts.experimentalValue < 1000);
        if (activeTargets.length > 1) {
          seriesData.improvesCount++;
        }
        const safetyIssue = targetScores.find(ts => ts.safetyFlag && ts.experimentalValue !== null && ts.experimentalValue < 100);
        if (safetyIssue) {
          seriesData.degradesSafety = true;
        }
      }
    }

    const series = Array.from(seriesMap.entries()).map(([seriesId, data]) => ({
      seriesId,
      improvesMultipleTargets: data.improvesCount > 0,
      degradesSafety: data.degradesSafety,
    }));

    return {
      molecules: moleculesResult,
      targets: panelTargets,
      series,
    };
  }

  async getAssayPanels(campaignId?: string): Promise<AssayPanel[]> {
    if (campaignId) {
      return db.select().from(assayPanels).where(eq(assayPanels.campaignId, campaignId)).orderBy(desc(assayPanels.createdAt));
    }
    return db.select().from(assayPanels).orderBy(desc(assayPanels.createdAt));
  }

  async getAssayPanel(id: string): Promise<(AssayPanel & { targets: (AssayPanelTarget & { target: Target })[] }) | undefined> {
    const panelResult = await db.select().from(assayPanels).where(eq(assayPanels.id, id)).limit(1);
    if (panelResult.length === 0) return undefined;
    
    const panel = panelResult[0];
    const panelTargetsData = await db
      .select()
      .from(assayPanelTargets)
      .where(eq(assayPanelTargets.assayPanelId, id));
    
    const targetsWithInfo: (AssayPanelTarget & { target: Target })[] = [];
    for (const pt of panelTargetsData) {
      const targetData = await db.select().from(targets).where(eq(targets.id, pt.targetId)).limit(1);
      if (targetData.length > 0) {
        targetsWithInfo.push({ ...pt, target: targetData[0] });
      }
    }

    return { ...panel, targets: targetsWithInfo };
  }

  async createAssayPanel(panel: InsertAssayPanel, targetIds: { targetId: string; role: string }[]): Promise<AssayPanel> {
    const result = await db.insert(assayPanels).values(panel).returning();
    const createdPanel = result[0];

    for (const t of targetIds) {
      await db.insert(assayPanelTargets).values({
        assayPanelId: createdPanel.id,
        targetId: t.targetId,
        role: t.role as "primary" | "secondary" | "safety",
      });
    }

    return createdPanel;
  }

  async deleteAssayPanel(id: string): Promise<void> {
    await db.delete(assayPanels).where(eq(assayPanels.id, id));
  }

  async getAssayPanelResults(panelId: string): Promise<{
    molecules: { id: string; smiles: string; name: string | null }[];
    targets: { id: string; name: string; role: string }[];
    matrix: { moleculeId: string; targetId: string; value: number | null; outcomeLabel: string | null }[];
    summary: { totalMolecules: number; balancedActives: number; selectiveCompounds: number };
  }> {
    const panel = await this.getAssayPanel(panelId);
    if (!panel) {
      return { molecules: [], targets: [], matrix: [], summary: { totalMolecules: 0, balancedActives: 0, selectiveCompounds: 0 } };
    }

    const targetsList = panel.targets.map(t => ({
      id: t.targetId,
      name: t.target.name,
      role: t.role || "primary",
    }));

    const targetIdSet = new Set(targetsList.map(t => t.id));

    const assayList = await db.select().from(assays).where(sql`${assays.targetId} = ANY(${Array.from(targetIdSet)})`);
    const assayIdToTarget = new Map<string, string>();
    for (const a of assayList) {
      if (a.targetId) assayIdToTarget.set(a.id, a.targetId);
    }

    const arData = await db
      .select()
      .from(assayResults)
      .where(eq(assayResults.campaignId, panel.campaignId));

    const moleculeIds = new Set<string>();
    const matrix: { moleculeId: string; targetId: string; value: number | null; outcomeLabel: string | null }[] = [];

    for (const ar of arData) {
      const targetId = assayIdToTarget.get(ar.assayId);
      if (targetId && targetIdSet.has(targetId)) {
        moleculeIds.add(ar.moleculeId);
        matrix.push({
          moleculeId: ar.moleculeId,
          targetId,
          value: ar.value,
          outcomeLabel: ar.outcomeLabel,
        });
      }
    }

    const molData = moleculeIds.size > 0 
      ? await db.select().from(molecules).where(sql`${molecules.id} = ANY(${Array.from(moleculeIds)})`)
      : [];

    const moleculesFormatted = molData.map(m => ({ id: m.id, smiles: m.smiles, name: m.name }));

    const molTargetActive = new Map<string, Set<string>>();
    for (const entry of matrix) {
      if (entry.value !== null && entry.value < 1000) {
        if (!molTargetActive.has(entry.moleculeId)) {
          molTargetActive.set(entry.moleculeId, new Set());
        }
        molTargetActive.get(entry.moleculeId)!.add(entry.targetId);
      }
    }

    let balancedActives = 0;
    let selectiveCompounds = 0;
    for (const activeTargets of Array.from(molTargetActive.values())) {
      if (activeTargets.size >= 2) {
        balancedActives++;
      } else if (activeTargets.size === 1) {
        selectiveCompounds++;
      }
    }

    return {
      molecules: moleculesFormatted,
      targets: targetsList,
      matrix,
      summary: {
        totalMolecules: moleculesFormatted.length,
        balancedActives,
        selectiveCompounds,
      },
    };
  }

  async uploadAssayPanelResults(panelId: string, results: { moleculeId?: string; smiles?: string; targetId: string; value: number; concentration?: number; outcomeLabel?: string }[]): Promise<{ imported: number; warnings: string[] }> {
    const panel = await this.getAssayPanel(panelId);
    if (!panel) {
      return { imported: 0, warnings: ["Panel not found"] };
    }

    const panelTargetIds = new Set(panel.targets.map(t => t.targetId));
    const warnings: string[] = [];
    let imported = 0;

    for (const row of results) {
      if (!panelTargetIds.has(row.targetId)) {
        warnings.push(`Target ${row.targetId} not in panel`);
        continue;
      }

      let moleculeId = row.moleculeId;
      if (!moleculeId && row.smiles) {
        const existingMol = await db.select().from(molecules).where(eq(molecules.smiles, row.smiles)).limit(1);
        if (existingMol.length > 0) {
          moleculeId = existingMol[0].id;
        } else {
          const newMol = await db.insert(molecules).values({ smiles: row.smiles }).returning();
          moleculeId = newMol[0].id;
        }
      }

      if (!moleculeId) {
        warnings.push("Row missing moleculeId and smiles");
        continue;
      }

      const targetAssays = await db.select().from(assays).where(eq(assays.targetId, row.targetId)).limit(1);
      let assayId: string;
      if (targetAssays.length > 0) {
        assayId = targetAssays[0].id;
      } else {
        const newAssay = await db.insert(assays).values({ 
          name: `Panel Assay - ${row.targetId}`,
          targetId: row.targetId,
          type: "binding",
        }).returning();
        assayId = newAssay[0].id;
      }

      await db.insert(assayResults).values({
        assayId,
        campaignId: panel.campaignId,
        moleculeId,
        value: row.value,
        concentration: row.concentration,
        outcomeLabel: row.outcomeLabel as any,
      });
      imported++;
    }

    return { imported, warnings };
  }

  async getMoaGraph(campaignId?: string): Promise<{ nodes: MoaNode[]; edges: MoaEdge[] }> {
    const allNodes = await db.select().from(moaNodes).orderBy(desc(moaNodes.createdAt));
    const allEdges = await db.select().from(moaEdges).orderBy(desc(moaEdges.createdAt));
    return { nodes: allNodes, edges: allEdges };
  }

  async getMoaSubgraph(targetId: string): Promise<{ nodes: MoaNode[]; edges: MoaEdge[] }> {
    const targetNode = await db.select().from(moaNodes).where(
      and(eq(moaNodes.type, "target"), sql`${moaNodes.metadata}->>'targetId' = ${targetId}`)
    ).limit(1);

    if (targetNode.length === 0) {
      return { nodes: [], edges: [] };
    }

    const nodeId = targetNode[0].id;
    const edgesFrom = await db.select().from(moaEdges).where(eq(moaEdges.fromNodeId, nodeId));
    const edgesTo = await db.select().from(moaEdges).where(eq(moaEdges.toNodeId, nodeId));
    
    const allEdges = [...edgesFrom, ...edgesTo];
    const connectedNodeIds = new Set<string>([nodeId]);
    for (const e of allEdges) {
      connectedNodeIds.add(e.fromNodeId);
      connectedNodeIds.add(e.toNodeId);
    }

    const nodes = await db.select().from(moaNodes).where(sql`${moaNodes.id} = ANY(${Array.from(connectedNodeIds)})`);
    return { nodes, edges: allEdges };
  }

  async createMoaNode(node: InsertMoaNode): Promise<MoaNode> {
    const result = await db.insert(moaNodes).values(node).returning();
    return result[0];
  }

  async createMoaEdge(edge: InsertMoaEdge): Promise<MoaEdge> {
    const result = await db.insert(moaEdges).values(edge).returning();
    return result[0];
  }

  async getPipelineTemplates(domain?: string): Promise<PipelineTemplate[]> {
    if (domain) {
      return db.select().from(pipelineTemplates).where(eq(pipelineTemplates.domain, domain as any)).orderBy(desc(pipelineTemplates.createdAt));
    }
    return db.select().from(pipelineTemplates).orderBy(desc(pipelineTemplates.createdAt));
  }

  async getPipelineTemplate(id: string): Promise<(PipelineTemplate & { targets: PipelineTemplateTarget[] }) | undefined> {
    const template = await db.select().from(pipelineTemplates).where(eq(pipelineTemplates.id, id)).limit(1);
    if (template.length === 0) return undefined;

    const targets = await db.select().from(pipelineTemplateTargets).where(eq(pipelineTemplateTargets.templateId, id)).orderBy(pipelineTemplateTargets.sortOrder);
    return { ...template[0], targets };
  }

  async createPipelineTemplate(template: InsertPipelineTemplate, templateTargets?: InsertPipelineTemplateTarget[]): Promise<PipelineTemplate> {
    const result = await db.insert(pipelineTemplates).values(template).returning();
    const created = result[0];

    if (templateTargets && templateTargets.length > 0) {
      await db.insert(pipelineTemplateTargets).values(
        templateTargets.map((t, i) => ({
          ...t,
          templateId: created.id,
          sortOrder: t.sortOrder ?? i,
        }))
      );
    }

    return created;
  }

  async deletePipelineTemplate(id: string): Promise<void> {
    await db.delete(pipelineTemplates).where(eq(pipelineTemplates.id, id));
  }

  async seedBuiltInTemplates(): Promise<void> {
    const existing = await db.select().from(pipelineTemplates).where(eq(pipelineTemplates.isBuiltIn, true)).limit(1);
    if (existing.length > 0) return;

    const alzheimersTemplate = await this.createPipelineTemplate({
      name: "Alzheimer's Disease Pipeline",
      description: "Multi-target approach for Alzheimer's disease focusing on amyloid, tau, and neuroinflammation pathways. Optimized for blood-brain barrier penetration and CNS safety.",
      domain: "alzheimers",
      isBuiltIn: true,
      modality: "small_molecule",
      pipelineConfig: {
        steps: ["hit_generation", "screening_cascade", "virtual_docking", "admet_prediction", "bbb_prediction", "hit_prioritization"],
        defaultLibrary: "cns_focused",
        dockingProtocol: "multi_target_ensemble"
      },
      scoringWeights: {
        efficacy: 0.35,
        selectivity: 0.25,
        bbbPenetration: 0.20,
        safety: 0.20
      },
      assayPanelConfig: {
        primaryAssays: ["amyloid_binding", "tau_aggregation", "cholinesterase"],
        secondaryAssays: ["microglial_activation", "synaptic_function"],
        safetyAssays: ["herg", "cyp_inhibition", "hepatotoxicity"]
      },
      visualizationPresets: {
        defaultView: "multi_target_radar",
        highlightSafetyFlags: true,
        bbbThreshold: 0.5
      }
    }, [
      { name: "Amyloid Beta", description: "Primary Alzheimer's target for amyloid pathway modulation", role: "primary", category: "amyloid_pathway", templateId: "", sortOrder: 0 },
      { name: "Tau Protein", description: "Secondary target for tau aggregation inhibition", role: "secondary", category: "tau_pathway", templateId: "", sortOrder: 1 },
      { name: "AChE", description: "Acetylcholinesterase for symptomatic treatment", role: "secondary", category: "cholinergic", templateId: "", sortOrder: 2 },
      { name: "hERG", description: "Cardiac safety target", role: "safety", category: "safety", templateId: "", sortOrder: 3 }
    ]);

    const oncologyTemplate = await this.createPipelineTemplate({
      name: "Oncology Multi-Target Pipeline",
      description: "Comprehensive oncology pipeline for kinase inhibitors and targeted therapies. Includes resistance profiling and selectivity optimization.",
      domain: "oncology",
      isBuiltIn: true,
      modality: "small_molecule",
      pipelineConfig: {
        steps: ["hit_generation", "kinase_selectivity", "virtual_docking", "resistance_prediction", "metabolic_stability", "hit_prioritization"],
        defaultLibrary: "kinase_focused",
        dockingProtocol: "kinase_ensemble"
      },
      scoringWeights: {
        efficacy: 0.40,
        selectivity: 0.30,
        resistance: 0.15,
        safety: 0.15
      },
      assayPanelConfig: {
        primaryAssays: ["target_kinase", "cell_viability"],
        secondaryAssays: ["kinase_panel", "apoptosis", "cell_cycle"],
        safetyAssays: ["herg", "mitotoxicity", "genotoxicity"]
      },
      visualizationPresets: {
        defaultView: "selectivity_matrix",
        highlightResistance: true,
        kinasePanelView: true
      }
    }, [
      { name: "Primary Kinase", description: "Main oncology target kinase", role: "primary", category: "kinase", templateId: "", sortOrder: 0 },
      { name: "Resistance Mutant", description: "Common resistance mutation variant", role: "primary", category: "kinase", templateId: "", sortOrder: 1 },
      { name: "Off-Target Kinase 1", description: "Selectivity counter-screen", role: "secondary", category: "kinase", templateId: "", sortOrder: 2 },
      { name: "hERG", description: "Cardiac safety target", role: "safety", category: "safety", templateId: "", sortOrder: 3 }
    ]);

    const neuroinflammationTemplate = await this.createPipelineTemplate({
      name: "Neuroinflammation Pipeline",
      description: "Focused on microglial modulation and neuroinflammatory pathways. Combines anti-inflammatory and neuroprotective screening.",
      domain: "neuroinflammation",
      isBuiltIn: true,
      modality: "small_molecule",
      pipelineConfig: {
        steps: ["hit_generation", "inflammation_panel", "microglial_assay", "bbb_prediction", "hit_prioritization"],
        defaultLibrary: "cns_focused",
        dockingProtocol: "inflammatory_targets"
      },
      scoringWeights: {
        antiInflammatory: 0.35,
        neuroprotection: 0.25,
        bbbPenetration: 0.20,
        safety: 0.20
      },
      assayPanelConfig: {
        primaryAssays: ["microglial_activation", "cytokine_release"],
        secondaryAssays: ["neuronal_survival", "oxidative_stress"],
        safetyAssays: ["herg", "immunosuppression"]
      },
      visualizationPresets: {
        defaultView: "inflammation_heatmap",
        highlightSafetyFlags: true,
        bbbThreshold: 0.6
      }
    }, [
      { name: "NLRP3", description: "Inflammasome modulation target", role: "primary", category: "inflammasome", templateId: "", sortOrder: 0 },
      { name: "TNF-alpha", description: "Pro-inflammatory cytokine pathway", role: "secondary", category: "cytokine", templateId: "", sortOrder: 1 },
      { name: "IL-1beta", description: "Interleukin signaling", role: "secondary", category: "cytokine", templateId: "", sortOrder: 2 },
      { name: "hERG", description: "Cardiac safety target", role: "safety", category: "safety", templateId: "", sortOrder: 3 }
    ]);

    const metabolicTemplate = await this.createPipelineTemplate({
      name: "Metabolic Disease Pipeline",
      description: "Multi-pathway approach for diabetes and metabolic disorders. Includes glucose homeostasis, lipid metabolism, and NASH-related targets.",
      domain: "metabolic_disease",
      isBuiltIn: true,
      modality: "small_molecule",
      pipelineConfig: {
        steps: ["hit_generation", "metabolic_panel", "liver_safety", "pk_prediction", "hit_prioritization"],
        defaultLibrary: "metabolic_focused",
        dockingProtocol: "gpcr_nuclear_receptor"
      },
      scoringWeights: {
        glucoseControl: 0.30,
        lipidModulation: 0.25,
        liverSafety: 0.25,
        cardioSafety: 0.20
      },
      assayPanelConfig: {
        primaryAssays: ["glucose_uptake", "insulin_sensitivity"],
        secondaryAssays: ["lipogenesis", "fatty_acid_oxidation", "hepatocyte_health"],
        safetyAssays: ["herg", "hepatotoxicity", "steatosis"]
      },
      visualizationPresets: {
        defaultView: "metabolic_radar",
        highlightLiverFlags: true,
        glucoseThreshold: 0.7
      }
    }, [
      { name: "GLP-1R", description: "Glucagon-like peptide-1 receptor", role: "primary", category: "glucose", templateId: "", sortOrder: 0 },
      { name: "PPAR-gamma", description: "Nuclear receptor for lipid metabolism", role: "secondary", category: "lipid", templateId: "", sortOrder: 1 },
      { name: "AMPK", description: "Energy sensing kinase", role: "secondary", category: "energy", templateId: "", sortOrder: 2 },
      { name: "hERG", description: "Cardiac safety target", role: "safety", category: "safety", templateId: "", sortOrder: 3 }
    ]);
  }

  async getProcessingJobs(filters?: { status?: string; type?: string; campaignId?: string; materialsCampaignId?: string; limit?: number; offset?: number }): Promise<{ jobs: ProcessingJob[]; total: number }> {
    const conditions: any[] = [];
    const pageLimit = Math.min(filters?.limit || 100, 1000);
    const pageOffset = filters?.offset || 0;
    
    if (filters?.status) {
      conditions.push(eq(processingJobs.status, filters.status as any));
    }
    if (filters?.type) {
      conditions.push(eq(processingJobs.type, filters.type as any));
    }
    if (filters?.campaignId) {
      conditions.push(eq(processingJobs.campaignId, filters.campaignId));
    }
    if (filters?.materialsCampaignId) {
      conditions.push(eq(processingJobs.materialsCampaignId, filters.materialsCampaignId));
    }
    
    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;
    
    const countResult = await db.select({ count: sql<number>`count(*)::int` })
      .from(processingJobs)
      .where(whereClause);
    const total = countResult[0]?.count || 0;
    
    const jobs = await db.select()
      .from(processingJobs)
      .where(whereClause)
      .orderBy(desc(processingJobs.createdAt))
      .limit(pageLimit)
      .offset(pageOffset);
    
    return { jobs, total };
  }

  async getProcessingJob(id: string): Promise<ProcessingJob | undefined> {
    const result = await db.select().from(processingJobs).where(eq(processingJobs.id, id)).limit(1);
    return result[0];
  }

  async createProcessingJob(job: InsertProcessingJob): Promise<ProcessingJob> {
    const result = await db.insert(processingJobs).values(job).returning();
    return result[0];
  }

  async updateProcessingJob(id: string, job: Partial<InsertProcessingJob>): Promise<ProcessingJob | undefined> {
    const result = await db.update(processingJobs).set({ ...job, updatedAt: new Date() }).where(eq(processingJobs.id, id)).returning();
    return result[0];
  }

  async updateProcessingJobProgress(id: string, itemsCompleted: number, checkpointData?: unknown): Promise<ProcessingJob | undefined> {
    const job = await this.getProcessingJob(id);
    if (!job) return undefined;
    
    const total = job.itemsTotal || 1;
    const progressPercent = (itemsCompleted / total) * 100;
    
    const result = await db.update(processingJobs).set({
      itemsCompleted,
      progressPercent,
      checkpointData: checkpointData || job.checkpointData,
      heartbeatAt: new Date(),
      updatedAt: new Date()
    }).where(eq(processingJobs.id, id)).returning();
    return result[0];
  }

  async getProcessingJobRuns(jobId: string): Promise<ProcessingJobRun[]> {
    return db.select().from(processingJobRuns).where(eq(processingJobRuns.jobId, jobId)).orderBy(desc(processingJobRuns.startedAt));
  }

  async createProcessingJobRun(run: InsertProcessingJobRun): Promise<ProcessingJobRun> {
    const result = await db.insert(processingJobRuns).values(run).returning();
    return result[0];
  }

  async getProcessingJobEvents(jobId: string): Promise<ProcessingJobEvent[]> {
    return db.select().from(processingJobEvents).where(eq(processingJobEvents.jobId, jobId)).orderBy(desc(processingJobEvents.createdAt));
  }

  async createProcessingJobEvent(event: InsertProcessingJobEvent): Promise<ProcessingJobEvent> {
    const result = await db.insert(processingJobEvents).values(event).returning();
    return result[0];
  }

  async getJobArtifacts(jobId: string): Promise<JobArtifact[]> {
    return db.select().from(jobArtifacts).where(eq(jobArtifacts.jobId, jobId)).orderBy(desc(jobArtifacts.createdAt));
  }

  async getArtifactsByCampaign(campaignId: string, domain: "drug" | "materials"): Promise<JobArtifact[]> {
    if (domain === "materials") {
      return db.select().from(jobArtifacts).where(
        and(
          eq(jobArtifacts.materialsCampaignId, campaignId),
          eq(jobArtifacts.domain, "materials")
        )
      ).orderBy(desc(jobArtifacts.createdAt));
    }
    return db.select().from(jobArtifacts).where(
      and(
        eq(jobArtifacts.campaignId, campaignId),
        eq(jobArtifacts.domain, "drug")
      )
    ).orderBy(desc(jobArtifacts.createdAt));
  }

  async createJobArtifact(artifact: InsertJobArtifact): Promise<JobArtifact> {
    const result = await db.insert(jobArtifacts).values(artifact).returning();
    return result[0];
  }

  async createJobArtifactsBatch(artifacts: InsertJobArtifact[]): Promise<JobArtifact[]> {
    if (artifacts.length === 0) return [];
    const result = await db.insert(jobArtifacts).values(artifacts).returning();
    return result;
  }

  async getMaterialsCampaignAggregate(campaignId: string): Promise<MaterialsCampaignAggregate | undefined> {
    const result = await db.select().from(materialsCampaignAggregates).where(eq(materialsCampaignAggregates.campaignId, campaignId)).limit(1);
    return result[0];
  }

  async upsertMaterialsCampaignAggregate(aggregate: InsertMaterialsCampaignAggregate): Promise<MaterialsCampaignAggregate> {
    const existing = await this.getMaterialsCampaignAggregate(aggregate.campaignId);
    if (existing) {
      const result = await db.update(materialsCampaignAggregates)
        .set({ ...aggregate, lastRefreshedAt: new Date() })
        .where(eq(materialsCampaignAggregates.id, existing.id))
        .returning();
      return result[0];
    }
    const result = await db.insert(materialsCampaignAggregates).values(aggregate).returning();
    return result[0];
  }

  async getMaterialVariantMetrics(campaignId: string, limit?: number): Promise<MaterialVariantMetric[]> {
    if (limit) {
      return db.select().from(materialVariantMetrics)
        .where(eq(materialVariantMetrics.campaignId, campaignId))
        .orderBy(desc(materialVariantMetrics.aggregateScore))
        .limit(limit);
    }
    return db.select().from(materialVariantMetrics)
      .where(eq(materialVariantMetrics.campaignId, campaignId))
      .orderBy(desc(materialVariantMetrics.aggregateScore));
  }

  async upsertMaterialVariantMetric(metric: InsertMaterialVariantMetric): Promise<MaterialVariantMetric> {
    const existing = await db.select().from(materialVariantMetrics)
      .where(and(
        eq(materialVariantMetrics.variantId, metric.variantId),
        eq(materialVariantMetrics.campaignId, metric.campaignId || "")
      )).limit(1);
    
    if (existing.length > 0) {
      const result = await db.update(materialVariantMetrics)
        .set({ ...metric, lastComputedAt: new Date() })
        .where(eq(materialVariantMetrics.id, existing[0].id))
        .returning();
      return result[0];
    }
    const result = await db.insert(materialVariantMetrics).values(metric).returning();
    return result[0];
  }

  async getImportTemplates(domain?: string, importType?: string, organizationId?: string): Promise<ImportTemplate[]> {
    const conditions = [];
    if (domain) conditions.push(eq(importTemplates.domain, domain as any));
    if (importType) conditions.push(eq(importTemplates.importType, importType as any));
    if (organizationId) conditions.push(eq(importTemplates.organizationId, organizationId));
    
    if (conditions.length === 0) {
      return db.select().from(importTemplates).orderBy(desc(importTemplates.createdAt));
    }
    return db.select().from(importTemplates)
      .where(and(...conditions))
      .orderBy(desc(importTemplates.createdAt));
  }

  async getImportTemplate(id: string): Promise<ImportTemplate | undefined> {
    const result = await db.select().from(importTemplates).where(eq(importTemplates.id, id)).limit(1);
    return result[0];
  }

  async createImportTemplate(template: InsertImportTemplate): Promise<ImportTemplate> {
    const result = await db.insert(importTemplates).values(template).returning();
    return result[0];
  }

  async updateImportTemplate(id: string, template: Partial<InsertImportTemplate>): Promise<ImportTemplate | undefined> {
    const result = await db.update(importTemplates)
      .set({ ...template, updatedAt: new Date() })
      .where(eq(importTemplates.id, id))
      .returning();
    return result[0];
  }

  async deleteImportTemplate(id: string): Promise<void> {
    await db.delete(importTemplates).where(eq(importTemplates.id, id));
  }

  async getImportJobs(filters?: { domain?: string; importType?: string; status?: string; organizationId?: string }): Promise<ImportJob[]> {
    const conditions = [];
    if (filters?.domain) conditions.push(eq(importJobs.domain, filters.domain as any));
    if (filters?.importType) conditions.push(eq(importJobs.importType, filters.importType as any));
    if (filters?.status) conditions.push(eq(importJobs.status, filters.status as any));
    if (filters?.organizationId) conditions.push(eq(importJobs.organizationId, filters.organizationId));
    
    if (conditions.length === 0) {
      return db.select().from(importJobs).orderBy(desc(importJobs.createdAt));
    }
    return db.select().from(importJobs)
      .where(and(...conditions))
      .orderBy(desc(importJobs.createdAt));
  }

  async getImportJob(id: string): Promise<ImportJob | undefined> {
    const result = await db.select().from(importJobs).where(eq(importJobs.id, id)).limit(1);
    return result[0];
  }

  async createImportJob(job: InsertImportJob): Promise<ImportJob> {
    const result = await db.insert(importJobs).values(job).returning();
    return result[0];
  }

  async updateImportJob(id: string, job: Partial<InsertImportJob>): Promise<ImportJob | undefined> {
    const result = await db.update(importJobs)
      .set(job)
      .where(eq(importJobs.id, id))
      .returning();
    return result[0];
  }

  async getCanonicalMolecules(companyId?: string): Promise<CanonicalMolecule[]> {
    if (companyId) {
      return db.select().from(canonicalMolecules).where(eq(canonicalMolecules.companyId, companyId)).orderBy(desc(canonicalMolecules.createdAt));
    }
    return db.select().from(canonicalMolecules).orderBy(desc(canonicalMolecules.createdAt));
  }

  async getCanonicalMoleculeByInchikey(inchikey: string, companyId?: string): Promise<CanonicalMolecule | undefined> {
    const conditions = companyId 
      ? and(eq(canonicalMolecules.inchikey, inchikey), eq(canonicalMolecules.companyId, companyId))
      : eq(canonicalMolecules.inchikey, inchikey);
    const result = await db.select().from(canonicalMolecules).where(conditions).limit(1);
    return result[0];
  }

  async createCanonicalMolecule(molecule: InsertCanonicalMolecule): Promise<CanonicalMolecule> {
    const result = await db.insert(canonicalMolecules).values(molecule).returning();
    return result[0];
  }

  async bulkCreateCanonicalMolecules(molecules: InsertCanonicalMolecule[]): Promise<CanonicalMolecule[]> {
    if (molecules.length === 0) return [];
    return db.insert(canonicalMolecules).values(molecules).returning();
  }

  async getCanonicalMaterials(companyId?: string): Promise<CanonicalMaterial[]> {
    if (companyId) {
      return db.select().from(canonicalMaterials).where(eq(canonicalMaterials.companyId, companyId)).orderBy(desc(canonicalMaterials.createdAt));
    }
    return db.select().from(canonicalMaterials).orderBy(desc(canonicalMaterials.createdAt));
  }

  async getCanonicalMaterialByHash(materialHash: string, companyId?: string): Promise<CanonicalMaterial | undefined> {
    const conditions = companyId
      ? and(eq(canonicalMaterials.materialHash, materialHash), eq(canonicalMaterials.companyId, companyId))
      : eq(canonicalMaterials.materialHash, materialHash);
    const result = await db.select().from(canonicalMaterials).where(conditions).limit(1);
    return result[0];
  }

  async createCanonicalMaterial(material: InsertCanonicalMaterial): Promise<CanonicalMaterial> {
    const result = await db.insert(canonicalMaterials).values(material).returning();
    return result[0];
  }

  async bulkCreateCanonicalMaterials(materials: InsertCanonicalMaterial[]): Promise<CanonicalMaterial[]> {
    if (materials.length === 0) return [];
    return db.insert(canonicalMaterials).values(materials).returning();
  }

  async createHitList(hitList: InsertHitList): Promise<HitList> {
    const result = await db.insert(hitLists).values(hitList).returning();
    return result[0];
  }

  async bulkCreateHitListItems(items: InsertHitListItem[]): Promise<HitListItem[]> {
    if (items.length === 0) return [];
    return db.insert(hitListItems).values(items).returning();
  }

  async createCanonicalAssay(assay: InsertCanonicalAssay): Promise<CanonicalAssay> {
    const result = await db.insert(canonicalAssays).values(assay).returning();
    return result[0];
  }

  async bulkCreateCanonicalAssayResults(results: InsertCanonicalAssayResult[]): Promise<CanonicalAssayResult[]> {
    if (results.length === 0) return [];
    return db.insert(canonicalAssayResults).values(results).returning();
  }

  async createMoleculeDescriptor(descriptor: InsertMoleculeDescriptor): Promise<MoleculeDescriptor> {
    const result = await db.insert(moleculeDescriptors).values(descriptor).returning();
    return result[0];
  }

  async createMoleculeFingerprint(fingerprint: InsertMoleculeFingerprint): Promise<MoleculeFingerprint> {
    const result = await db.insert(moleculeFingerprints).values(fingerprint).returning();
    return result[0];
  }

  async createTargetAsset(asset: InsertTargetAsset): Promise<TargetAsset> {
    const result = await db.insert(targetAssets).values(asset).returning();
    return result[0];
  }

  async createCanonicalMaterialVariant(variant: InsertCanonicalMaterialVariant): Promise<CanonicalMaterialVariant> {
    const result = await db.insert(canonicalMaterialVariants).values(variant).returning();
    return result[0];
  }

  async bulkCreateCanonicalMaterialVariants(variants: InsertCanonicalMaterialVariant[]): Promise<CanonicalMaterialVariant[]> {
    if (variants.length === 0) return [];
    return db.insert(canonicalMaterialVariants).values(variants).returning();
  }

  async getCanonicalMaterialVariantByHash(variantHash: string): Promise<CanonicalMaterialVariant | undefined> {
    const result = await db.select().from(canonicalMaterialVariants).where(eq(canonicalMaterialVariants.variantHash, variantHash)).limit(1);
    return result[0];
  }

  async createCanonicalMaterialProperty(property: InsertCanonicalMaterialProperty): Promise<CanonicalMaterialProperty> {
    const result = await db.insert(canonicalMaterialProperties).values(property).returning();
    return result[0];
  }

  async bulkCreateCanonicalMaterialProperties(properties: InsertCanonicalMaterialProperty[]): Promise<CanonicalMaterialProperty[]> {
    if (properties.length === 0) return [];
    return db.insert(canonicalMaterialProperties).values(properties).returning();
  }

  async createSimulationRun(run: InsertSimulationRun): Promise<SimulationRun> {
    const result = await db.insert(simulationRuns).values(run).returning();
    return result[0];
  }

  async createManufacturabilityScore(score: InsertManufacturabilityScore): Promise<ManufacturabilityScore> {
    const result = await db.insert(manufacturabilityScores).values(score).returning();
    return result[0];
  }

  async createCompoundAsset(asset: InsertCompoundAsset): Promise<CompoundAsset> {
    const result = await db.insert(compoundAssets).values(asset).returning();
    return result[0];
  }

  async bulkCreateCompoundAssets(assets: InsertCompoundAsset[]): Promise<CompoundAsset[]> {
    if (assets.length === 0) return [];
    return db.insert(compoundAssets).values(assets).returning();
  }

  async getCompoundAsset(id: string): Promise<CompoundAsset | undefined> {
    const result = await db.select().from(compoundAssets).where(eq(compoundAssets.id, id)).limit(1);
    return result[0];
  }

  async getCompoundAssetsByMolecule(moleculeId: string): Promise<CompoundAsset[]> {
    return db.select().from(compoundAssets).where(eq(compoundAssets.moleculeId, moleculeId));
  }

  async getCompoundAssetByTypeAndMolecule(moleculeId: string, assetType: string): Promise<CompoundAsset | undefined> {
    const result = await db.select().from(compoundAssets)
      .where(and(
        eq(compoundAssets.moleculeId, moleculeId),
        eq(compoundAssets.assetType, assetType as any)
      ))
      .limit(1);
    return result[0];
  }

  async deleteCompoundAsset(id: string): Promise<void> {
    await db.delete(compoundAssets).where(eq(compoundAssets.id, id));
  }

  async getBuiltInMolecules(limit: number = 100000, offset: number = 0): Promise<{ id: string; inchikey: string; canonicalSmiles: string }[]> {
    return db.select({
      id: canonicalMolecules.id,
      inchikey: canonicalMolecules.inchikey,
      canonicalSmiles: canonicalMolecules.canonicalSmiles,
    })
      .from(canonicalMolecules)
      .where(eq(canonicalMolecules.isBuiltIn, true))
      .limit(limit)
      .offset(offset);
  }

  async countBuiltInMolecules(): Promise<number> {
    const result = await db.select({ count: count() })
      .from(canonicalMolecules)
      .where(eq(canonicalMolecules.isBuiltIn, true));
    return result[0]?.count || 0;
  }

  async markMoleculesAsBuiltIn(moleculeIds: string[]): Promise<void> {
    if (moleculeIds.length === 0) return;
    for (const id of moleculeIds) {
      await db.update(canonicalMolecules)
        .set({ isBuiltIn: true })
        .where(eq(canonicalMolecules.id, id));
    }
  }

  async getActivityLogs(
    userId: string, 
    options: { limit: number; offset: number; activityType?: string }
  ): Promise<ActivityLog[]> {
    const { limit, offset, activityType } = options;
    
    const conditions = [eq(activityLogs.userId, userId)];
    if (activityType) {
      conditions.push(eq(activityLogs.activityType, activityType as any));
    }
    
    return db.select()
      .from(activityLogs)
      .where(and(...conditions))
      .orderBy(desc(activityLogs.createdAt))
      .limit(limit)
      .offset(offset);
  }

  async getActivityLogCount(userId: string, activityType?: string): Promise<number> {
    const conditions = [eq(activityLogs.userId, userId)];
    if (activityType) {
      conditions.push(eq(activityLogs.activityType, activityType as any));
    }
    
    const result = await db.select({ count: count() })
      .from(activityLogs)
      .where(and(...conditions));
    return result[0]?.count || 0;
  }

  async createActivityLog(log: InsertActivityLog): Promise<ActivityLog> {
    const result = await db.insert(activityLogs).values(log).returning();
    return result[0];
  }

  async getActivityLogStats(userId: string): Promise<{ type: string; count: number }[]> {
    return db.select({
      type: activityLogs.activityType,
      count: count(),
    })
      .from(activityLogs)
      .where(eq(activityLogs.userId, userId))
      .groupBy(activityLogs.activityType);
  }
}

export const storage = new DatabaseStorage();
