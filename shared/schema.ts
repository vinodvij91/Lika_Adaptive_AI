import { sql, relations } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, boolean, real, jsonb, pgEnum, index, integer } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export * from "./models/auth";

export const diseaseAreaEnum = pgEnum("disease_area", ["CNS", "Oncology", "Rare", "Infectious", "Cardiometabolic", "Autoimmune", "Respiratory", "Other"]);
export const moleculeSourceEnum = pgEnum("molecule_source", ["generated", "uploaded", "screened"]);
export const structureSourceEnum = pgEnum("structure_source", ["uploaded", "bionemo_predicted", "other"]);
export const campaignStatusEnum = pgEnum("campaign_status", ["pending", "running", "completed", "failed"]);
export const jobTypeEnum = pgEnum("job_type", ["generation", "filtering", "docking", "scoring", "quantum_optimization", "quantum_scoring", "other"]);
export const jobStatusEnum = pgEnum("job_status", ["pending", "running", "completed", "failed"]);
export const providerTypeEnum = pgEnum("provider_type", ["bionemo", "ml", "docking", "quantum", "ip", "literature", "smiles_library", "agent", "materials_library", "simulation", "oracle", "selection"]);
export const outcomeEnum = pgEnum("outcome_label", ["promising", "dropped", "hit", "unknown"]);
export const collaboratorRoleEnum = pgEnum("collaborator_role", ["owner", "editor", "viewer"]);
export const libraryTypeEnum = pgEnum("library_type", ["internal", "uploaded", "generated"]);
export const libraryStatusEnum = pgEnum("library_status", ["draft", "processing", "curated", "deprecated"]);
export const cleaningStatusEnum = pgEnum("cleaning_status", ["pending", "cleaning", "validated", "failed"]);
export const computeProviderEnum = pgEnum("compute_provider", ["hetzner", "vast", "onprem", "aws", "azure", "gcp", "other"]);
export const computePurposeEnum = pgEnum("compute_purpose", ["ml", "bionemo", "docking", "quantum", "agents", "general"]);
export const computeStatusEnum = pgEnum("compute_status", ["active", "offline", "degraded"]);
export const connectionTypeEnum = pgEnum("connection_type", ["ssh", "cloud_api"]);
export const gpuTypeEnum = pgEnum("gpu_type", ["none", "T4", "A40", "A100", "H100", "H200", "MI300", "other"]);
export const gpuTierEnum = pgEnum("gpu_tier", ["shared-low", "shared-mid", "shared-high", "dedicated-A100", "dedicated-H100", "dedicated-H200", "enterprise"]);
export const modalityEnum = pgEnum("modality", ["small_molecule", "fragment", "protac", "peptide", "other"]);
export const assayTypeEnum = pgEnum("assay_type", ["binding", "functional", "in_vivo", "pk", "admet", "other"]);
export const assayOutcomeEnum = pgEnum("assay_outcome", ["active", "inactive", "toxic", "no_effect", "inconclusive"]);
export const orgRoleEnum = pgEnum("org_role", ["admin", "member", "viewer"]);
export const assetTypeEnum = pgEnum("asset_type", ["smiles_library", "pipeline_template", "program"]);
export const sharePermissionEnum = pgEnum("share_permission", ["read", "fork"]);
export const discoveryDomainEnum = pgEnum("discovery_domain", ["drug", "materials"]);
export const materialTypeEnum = pgEnum("material_type", ["polymer", "crystal", "composite", "surface", "membrane", "catalyst"]);
export const materialPropertySourceEnum = pgEnum("material_property_source", ["ml", "simulation", "experiment"]);

export const projects = pgTable("projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  diseaseArea: diseaseAreaEnum("disease_area").default("Other"),
  ownerId: varchar("owner_id").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const projectsRelations = relations(projects, ({ many }) => ({
  collaborators: many(projectCollaborators),
  campaigns: many(campaigns),
  molecules: many(projectMolecules),
  targets: many(projectTargets),
  comments: many(comments),
}));

export const projectCollaborators = pgTable("project_collaborators", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull(),
  role: collaboratorRoleEnum("role").default("viewer"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const projectCollaboratorsRelations = relations(projectCollaborators, ({ one }) => ({
  project: one(projects, { fields: [projectCollaborators.projectId], references: [projects.id] }),
}));

export const targets = pgTable("targets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  uniprotId: text("uniprot_id"),
  sequence: text("sequence"),
  hasStructure: boolean("has_structure").default(false),
  structureSource: structureSourceEnum("structure_source"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const targetVariants = pgTable("target_variants", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  targetId: varchar("target_id").notNull().references(() => targets.id, { onDelete: "cascade" }),
  variantName: text("variant_name").notNull(),
  sequence: text("sequence"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const targetVariantsRelations = relations(targetVariants, ({ one }) => ({
  target: one(targets, { fields: [targetVariants.targetId], references: [targets.id] }),
}));

export const diseaseContextSignals = pgTable("disease_context_signals", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  targetId: varchar("target_id").references(() => targets.id, { onDelete: "cascade" }),
  diseaseArea: diseaseAreaEnum("disease_area"),
  evidenceSource: text("evidence_source"),
  evidenceStrength: real("evidence_strength"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const diseaseContextSignalsRelations = relations(diseaseContextSignals, ({ one }) => ({
  target: one(targets, { fields: [diseaseContextSignals.targetId], references: [targets.id] }),
}));

export const programs = pgTable("programs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  targetId: varchar("target_id").references(() => targets.id, { onDelete: "set null" }),
  diseaseArea: diseaseAreaEnum("disease_area"),
  description: text("description"),
  ownerId: varchar("owner_id").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const programsRelations = relations(programs, ({ one, many }) => ({
  target: one(targets, { fields: [programs.targetId], references: [targets.id] }),
  campaigns: many(campaigns),
}));

export const oracleVersions = pgTable("oracle_versions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  componentVersions: jsonb("component_versions"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const projectTargets = pgTable("project_targets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  targetId: varchar("target_id").notNull().references(() => targets.id, { onDelete: "cascade" }),
});

export const projectTargetsRelations = relations(projectTargets, ({ one }) => ({
  project: one(projects, { fields: [projectTargets.projectId], references: [projects.id] }),
  target: one(targets, { fields: [projectTargets.targetId], references: [targets.id] }),
}));

export const molecules = pgTable("molecules", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  smiles: text("smiles").notNull(),
  source: moleculeSourceEnum("source").default("generated"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const projectMolecules = pgTable("project_molecules", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  moleculeId: varchar("molecule_id").notNull().references(() => molecules.id, { onDelete: "cascade" }),
});

export const projectMoleculesRelations = relations(projectMolecules, ({ one }) => ({
  project: one(projects, { fields: [projectMolecules.projectId], references: [projects.id] }),
  molecule: one(molecules, { fields: [projectMolecules.moleculeId], references: [molecules.id] }),
}));

export const campaigns = pgTable("campaigns", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  programId: varchar("program_id").references(() => programs.id, { onDelete: "set null" }),
  name: text("name").notNull(),
  domainType: diseaseAreaEnum("domain_type").default("Other"),
  modality: modalityEnum("modality").default("small_molecule"),
  oracleVersionId: varchar("oracle_version_id").references(() => oracleVersions.id, { onDelete: "set null" }),
  pipelineConfig: jsonb("pipeline_config"),
  status: campaignStatusEnum("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const campaignsRelations = relations(campaigns, ({ one, many }) => ({
  project: one(projects, { fields: [campaigns.projectId], references: [projects.id] }),
  program: one(programs, { fields: [campaigns.programId], references: [programs.id] }),
  oracleVersion: one(oracleVersions, { fields: [campaigns.oracleVersionId], references: [oracleVersions.id] }),
  jobs: many(jobs),
  modelRuns: many(modelRuns),
  moleculeScores: many(moleculeScores),
  learningGraphEntries: many(learningGraphEntries),
  experimentRecommendations: many(experimentRecommendations),
  assayResults: many(assayResults),
  comments: many(comments),
}));

export const jobs = pgTable("jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  campaignId: varchar("campaign_id").notNull().references(() => campaigns.id, { onDelete: "cascade" }),
  type: jobTypeEnum("type").notNull(),
  status: jobStatusEnum("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
  startedAt: timestamp("started_at"),
  finishedAt: timestamp("finished_at"),
  externalJobId: text("external_job_id"),
});

export const jobsRelations = relations(jobs, ({ one }) => ({
  campaign: one(campaigns, { fields: [jobs.campaignId], references: [campaigns.id] }),
}));

export const modelRuns = pgTable("model_runs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  campaignId: varchar("campaign_id").notNull().references(() => campaigns.id, { onDelete: "cascade" }),
  stepName: text("step_name").notNull(),
  providerType: providerTypeEnum("provider_type").notNull(),
  modelVersion: text("model_version"),
  status: jobStatusEnum("status").default("pending"),
  startedAt: timestamp("started_at"),
  finishedAt: timestamp("finished_at"),
  requestPayload: jsonb("request_payload"),
  responsePayload: jsonb("response_payload"),
});

export const modelRunsRelations = relations(modelRuns, ({ one }) => ({
  campaign: one(campaigns, { fields: [modelRuns.campaignId], references: [campaigns.id] }),
}));

export const moleculeScores = pgTable("molecule_scores", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  moleculeId: varchar("molecule_id").notNull().references(() => molecules.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").notNull().references(() => campaigns.id, { onDelete: "cascade" }),
  dockingScore: real("docking_score"),
  admetScore: real("admet_score"),
  qsarScore: real("qsar_score"),
  oracleScore: real("oracle_score"),
  translationalScore: real("translational_score"),
  translationalConfidence: real("translational_confidence"),
  translationalMetadata: jsonb("translational_metadata"),
  variantScores: jsonb("variant_scores"),
  variantRobustnessScore: real("variant_robustness_score"),
  synthesisScore: real("synthesis_score"),
  synthesisComplexity: real("synthesis_complexity"),
  synthesisMetadata: jsonb("synthesis_metadata"),
  dockingUncertainty: real("docking_uncertainty"),
  admetUncertainty: real("admet_uncertainty"),
  qsarUncertainty: real("qsar_uncertainty"),
  translationalUncertainty: real("translational_uncertainty"),
  applicabilityDomainFlag: boolean("applicability_domain_flag"),
  ipSimilarityScore: real("ip_similarity_score"),
  ipRiskFlag: boolean("ip_risk_flag"),
  ipMetadata: jsonb("ip_metadata"),
  rawScores: jsonb("raw_scores"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_molecule_scores_campaign").on(table.campaignId),
  index("idx_molecule_scores_oracle").on(table.oracleScore),
]);

export const moleculeScoresRelations = relations(moleculeScores, ({ one }) => ({
  molecule: one(molecules, { fields: [moleculeScores.moleculeId], references: [molecules.id] }),
  campaign: one(campaigns, { fields: [moleculeScores.campaignId], references: [campaigns.id] }),
}));

export const learningGraphEntries = pgTable("learning_graph_entries", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  moleculeId: varchar("molecule_id").notNull().references(() => molecules.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").notNull().references(() => campaigns.id, { onDelete: "cascade" }),
  domainType: diseaseAreaEnum("domain_type"),
  outcomeLabel: outcomeEnum("outcome_label").default("unknown"),
  oracleScore: real("oracle_score"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const learningGraphEntriesRelations = relations(learningGraphEntries, ({ one }) => ({
  molecule: one(molecules, { fields: [learningGraphEntries.moleculeId], references: [molecules.id] }),
  campaign: one(campaigns, { fields: [learningGraphEntries.campaignId], references: [campaigns.id] }),
}));

export const comments = pgTable("comments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").references(() => campaigns.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull(),
  body: text("body").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const commentsRelations = relations(comments, ({ one }) => ({
  project: one(projects, { fields: [comments.projectId], references: [projects.id] }),
  campaign: one(campaigns, { fields: [comments.campaignId], references: [campaigns.id] }),
}));

export const curatedLibraries = pgTable("curated_libraries", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  domainType: diseaseAreaEnum("domain_type").default("Other"),
  libraryType: libraryTypeEnum("library_type").default("uploaded"),
  status: libraryStatusEnum("status").default("draft"),
  ownerId: varchar("owner_id").notNull(),
  isPublic: boolean("is_public").default(false),
  moleculeCount: real("molecule_count").default(0),
  scaffoldCount: real("scaffold_count").default(0),
  version: real("version").default(1),
  tags: text("tags").array(),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  index("idx_curated_libraries_domain").on(table.domainType),
  index("idx_curated_libraries_status").on(table.status),
]);

export const curatedLibrariesRelations = relations(curatedLibraries, ({ many }) => ({
  molecules: many(libraryMolecules),
  scaffolds: many(scaffolds),
  annotations: many(libraryAnnotations),
  collaborators: many(libraryCollaborators),
}));

export const libraryMolecules = pgTable("library_molecules", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  libraryId: varchar("library_id").notNull().references(() => curatedLibraries.id, { onDelete: "cascade" }),
  moleculeId: varchar("molecule_id").notNull().references(() => molecules.id, { onDelete: "cascade" }),
  canonicalSmiles: text("canonical_smiles"),
  canonicalHash: text("canonical_hash"),
  cleaningStatus: cleaningStatusEnum("cleaning_status").default("pending"),
  scaffoldId: varchar("scaffold_id"),
  tags: text("tags").array(),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_library_molecules_library").on(table.libraryId),
  index("idx_library_molecules_scaffold").on(table.scaffoldId),
  index("idx_library_molecules_hash").on(table.canonicalHash),
]);

export const libraryMoleculesRelations = relations(libraryMolecules, ({ one }) => ({
  library: one(curatedLibraries, { fields: [libraryMolecules.libraryId], references: [curatedLibraries.id] }),
  molecule: one(molecules, { fields: [libraryMolecules.moleculeId], references: [molecules.id] }),
}));

export const scaffolds = pgTable("scaffolds", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  libraryId: varchar("library_id").notNull().references(() => curatedLibraries.id, { onDelete: "cascade" }),
  name: text("name"),
  coreSmiles: text("core_smiles").notNull(),
  scaffoldType: text("scaffold_type").default("murcko"),
  memberCount: real("member_count").default(0),
  properties: jsonb("properties"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_scaffolds_library").on(table.libraryId),
]);

export const scaffoldsRelations = relations(scaffolds, ({ one, many }) => ({
  library: one(curatedLibraries, { fields: [scaffolds.libraryId], references: [curatedLibraries.id] }),
  members: many(libraryMolecules),
}));

export const libraryAnnotations = pgTable("library_annotations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  libraryId: varchar("library_id").notNull().references(() => curatedLibraries.id, { onDelete: "cascade" }),
  moleculeId: varchar("molecule_id").references(() => molecules.id, { onDelete: "cascade" }),
  annotationType: text("annotation_type").notNull(),
  annotationValue: text("annotation_value"),
  confidence: real("confidence"),
  source: text("source").default("user"),
  userId: varchar("user_id"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const libraryAnnotationsRelations = relations(libraryAnnotations, ({ one }) => ({
  library: one(curatedLibraries, { fields: [libraryAnnotations.libraryId], references: [curatedLibraries.id] }),
  molecule: one(molecules, { fields: [libraryAnnotations.moleculeId], references: [molecules.id] }),
}));

export const libraryCollaborators = pgTable("library_collaborators", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  libraryId: varchar("library_id").notNull().references(() => curatedLibraries.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull(),
  role: collaboratorRoleEnum("role").default("viewer"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const libraryCollaboratorsRelations = relations(libraryCollaborators, ({ one }) => ({
  library: one(curatedLibraries, { fields: [libraryCollaborators.libraryId], references: [curatedLibraries.id] }),
}));

export const assays = pgTable("assays", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  targetId: varchar("target_id").references(() => targets.id, { onDelete: "set null" }),
  type: assayTypeEnum("type").default("binding"),
  estimatedCost: real("estimated_cost"),
  estimatedDurationDays: real("estimated_duration_days"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const assaysRelations = relations(assays, ({ one, many }) => ({
  target: one(targets, { fields: [assays.targetId], references: [targets.id] }),
  results: many(assayResults),
  recommendations: many(experimentRecommendations),
}));

export const experimentRecommendations = pgTable("experiment_recommendations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  campaignId: varchar("campaign_id").notNull().references(() => campaigns.id, { onDelete: "cascade" }),
  assayId: varchar("assay_id").references(() => assays.id, { onDelete: "set null" }),
  moleculeIds: jsonb("molecule_ids"),
  priorityScore: real("priority_score"),
  estimatedCost: real("estimated_cost"),
  rationale: text("rationale"),
  status: text("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const experimentRecommendationsRelations = relations(experimentRecommendations, ({ one }) => ({
  campaign: one(campaigns, { fields: [experimentRecommendations.campaignId], references: [campaigns.id] }),
  assay: one(assays, { fields: [experimentRecommendations.assayId], references: [assays.id] }),
}));

export const assayResults = pgTable("assay_results", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assayId: varchar("assay_id").notNull().references(() => assays.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").references(() => campaigns.id, { onDelete: "set null" }),
  moleculeId: varchar("molecule_id").notNull().references(() => molecules.id, { onDelete: "cascade" }),
  value: real("value"),
  units: text("units"),
  outcomeLabel: assayOutcomeEnum("outcome_label"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const assayResultsRelations = relations(assayResults, ({ one }) => ({
  assay: one(assays, { fields: [assayResults.assayId], references: [assays.id] }),
  campaign: one(campaigns, { fields: [assayResults.campaignId], references: [campaigns.id] }),
  molecule: one(molecules, { fields: [assayResults.moleculeId], references: [molecules.id] }),
}));

export const literatureAnnotations = pgTable("literature_annotations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  targetId: varchar("target_id").references(() => targets.id, { onDelete: "cascade" }),
  moleculeId: varchar("molecule_id").references(() => molecules.id, { onDelete: "cascade" }),
  source: text("source"),
  relevanceScore: real("relevance_score"),
  confidence: real("confidence"),
  summary: text("summary"),
  url: text("url"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const literatureAnnotationsRelations = relations(literatureAnnotations, ({ one }) => ({
  target: one(targets, { fields: [literatureAnnotations.targetId], references: [targets.id] }),
  molecule: one(molecules, { fields: [literatureAnnotations.moleculeId], references: [molecules.id] }),
}));

export const organizations = pgTable("organizations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const organizationsRelations = relations(organizations, ({ many }) => ({
  members: many(orgMembers),
  sharedByAssets: many(sharedAssets),
}));

export const orgMembers = pgTable("org_members", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull(),
  organizationId: varchar("organization_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  role: orgRoleEnum("role").default("member"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const orgMembersRelations = relations(orgMembers, ({ one }) => ({
  organization: one(organizations, { fields: [orgMembers.organizationId], references: [organizations.id] }),
}));

export const sharedAssets = pgTable("shared_assets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assetType: assetTypeEnum("asset_type").notNull(),
  assetId: varchar("asset_id").notNull(),
  sharedByOrgId: varchar("shared_by_org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  sharedWithOrgId: varchar("shared_with_org_id").notNull().references(() => organizations.id, { onDelete: "cascade" }),
  permissions: sharePermissionEnum("permissions").default("read"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const sharedAssetsRelations = relations(sharedAssets, ({ one }) => ({
  sharedByOrg: one(organizations, { fields: [sharedAssets.sharedByOrgId], references: [organizations.id] }),
  sharedWithOrg: one(organizations, { fields: [sharedAssets.sharedWithOrgId], references: [organizations.id] }),
}));

export const insertProjectSchema = createInsertSchema(projects).omit({ id: true, createdAt: true, updatedAt: true });
export const insertTargetSchema = createInsertSchema(targets).omit({ id: true, createdAt: true });
export const insertMoleculeSchema = createInsertSchema(molecules).omit({ id: true, createdAt: true });
export const insertCampaignSchema = createInsertSchema(campaigns).omit({ id: true, createdAt: true, updatedAt: true });
export const insertJobSchema = createInsertSchema(jobs).omit({ id: true, createdAt: true });
export const insertModelRunSchema = createInsertSchema(modelRuns).omit({ id: true });
export const insertMoleculeScoreSchema = createInsertSchema(moleculeScores).omit({ id: true, createdAt: true });
export const insertLearningGraphEntrySchema = createInsertSchema(learningGraphEntries).omit({ id: true, createdAt: true });
export const insertCommentSchema = createInsertSchema(comments).omit({ id: true, createdAt: true });
export const insertCuratedLibrarySchema = createInsertSchema(curatedLibraries).omit({ id: true, createdAt: true, updatedAt: true });
export const insertLibraryMoleculeSchema = createInsertSchema(libraryMolecules).omit({ id: true, createdAt: true });
export const insertScaffoldSchema = createInsertSchema(scaffolds).omit({ id: true, createdAt: true });
export const insertLibraryAnnotationSchema = createInsertSchema(libraryAnnotations).omit({ id: true, createdAt: true });
export const insertTargetVariantSchema = createInsertSchema(targetVariants).omit({ id: true, createdAt: true });
export const insertDiseaseContextSignalSchema = createInsertSchema(diseaseContextSignals).omit({ id: true, createdAt: true });
export const insertProgramSchema = createInsertSchema(programs).omit({ id: true, createdAt: true, updatedAt: true });
export const insertOracleVersionSchema = createInsertSchema(oracleVersions).omit({ id: true, createdAt: true });
export const insertAssaySchema = createInsertSchema(assays).omit({ id: true, createdAt: true });
export const insertExperimentRecommendationSchema = createInsertSchema(experimentRecommendations).omit({ id: true, createdAt: true });
export const insertAssayResultSchema = createInsertSchema(assayResults).omit({ id: true, createdAt: true });
export const insertLiteratureAnnotationSchema = createInsertSchema(literatureAnnotations).omit({ id: true, createdAt: true });
export const insertOrganizationSchema = createInsertSchema(organizations).omit({ id: true, createdAt: true });
export const insertOrgMemberSchema = createInsertSchema(orgMembers).omit({ id: true, createdAt: true });
export const insertSharedAssetSchema = createInsertSchema(sharedAssets).omit({ id: true, createdAt: true });

export type Project = typeof projects.$inferSelect;
export type InsertProject = z.infer<typeof insertProjectSchema>;
export type Target = typeof targets.$inferSelect;
export type InsertTarget = z.infer<typeof insertTargetSchema>;
export type Molecule = typeof molecules.$inferSelect;
export type InsertMolecule = z.infer<typeof insertMoleculeSchema>;
export type Campaign = typeof campaigns.$inferSelect;
export type InsertCampaign = z.infer<typeof insertCampaignSchema>;
export type Job = typeof jobs.$inferSelect;
export type InsertJob = z.infer<typeof insertJobSchema>;
export type ModelRun = typeof modelRuns.$inferSelect;
export type InsertModelRun = z.infer<typeof insertModelRunSchema>;
export type MoleculeScore = typeof moleculeScores.$inferSelect;
export type InsertMoleculeScore = z.infer<typeof insertMoleculeScoreSchema>;
export type LearningGraphEntry = typeof learningGraphEntries.$inferSelect;
export type InsertLearningGraphEntry = z.infer<typeof insertLearningGraphEntrySchema>;
export type Comment = typeof comments.$inferSelect;
export type InsertComment = z.infer<typeof insertCommentSchema>;
export type CuratedLibrary = typeof curatedLibraries.$inferSelect;
export type InsertCuratedLibrary = z.infer<typeof insertCuratedLibrarySchema>;
export type LibraryMolecule = typeof libraryMolecules.$inferSelect;
export type InsertLibraryMolecule = z.infer<typeof insertLibraryMoleculeSchema>;
export type Scaffold = typeof scaffolds.$inferSelect;
export type InsertScaffold = z.infer<typeof insertScaffoldSchema>;
export type LibraryAnnotation = typeof libraryAnnotations.$inferSelect;
export type InsertLibraryAnnotation = z.infer<typeof insertLibraryAnnotationSchema>;
export type TargetVariant = typeof targetVariants.$inferSelect;
export type InsertTargetVariant = z.infer<typeof insertTargetVariantSchema>;
export type DiseaseContextSignal = typeof diseaseContextSignals.$inferSelect;
export type InsertDiseaseContextSignal = z.infer<typeof insertDiseaseContextSignalSchema>;
export type Program = typeof programs.$inferSelect;
export type InsertProgram = z.infer<typeof insertProgramSchema>;
export type OracleVersion = typeof oracleVersions.$inferSelect;
export type InsertOracleVersion = z.infer<typeof insertOracleVersionSchema>;
export type Assay = typeof assays.$inferSelect;
export type InsertAssay = z.infer<typeof insertAssaySchema>;
export type ExperimentRecommendation = typeof experimentRecommendations.$inferSelect;
export type InsertExperimentRecommendation = z.infer<typeof insertExperimentRecommendationSchema>;
export type AssayResult = typeof assayResults.$inferSelect;
export type InsertAssayResult = z.infer<typeof insertAssayResultSchema>;
export type LiteratureAnnotation = typeof literatureAnnotations.$inferSelect;
export type InsertLiteratureAnnotation = z.infer<typeof insertLiteratureAnnotationSchema>;
export type Organization = typeof organizations.$inferSelect;
export type InsertOrganization = z.infer<typeof insertOrganizationSchema>;
export type OrgMember = typeof orgMembers.$inferSelect;
export type InsertOrgMember = z.infer<typeof insertOrgMemberSchema>;
export type SharedAsset = typeof sharedAssets.$inferSelect;
export type InsertSharedAsset = z.infer<typeof insertSharedAssetSchema>;

export const computeNodes = pgTable("compute_nodes", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  provider: computeProviderEnum("provider").default("other"),
  connectionType: connectionTypeEnum("connection_type").default("ssh"),
  gpuType: gpuTypeEnum("gpu_type").default("none"),
  tier: gpuTierEnum("tier").default("shared-low"),
  purpose: computePurposeEnum("purpose").default("general"),
  ipAddress: text("ip_address"),
  sshHost: text("ssh_host"),
  sshPort: text("ssh_port").default("22"),
  sshUsername: text("ssh_username"),
  sshConfigId: varchar("ssh_config_id"),
  companyId: varchar("company_id"),
  region: text("region"),
  isDefault: boolean("is_default").default(false),
  status: computeStatusEnum("status").default("active"),
  specs: jsonb("specs"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertComputeNodeSchema = createInsertSchema(computeNodes).omit({ id: true, createdAt: true, updatedAt: true });
export type ComputeNode = typeof computeNodes.$inferSelect;
export type InsertComputeNode = z.infer<typeof insertComputeNodeSchema>;

export type ConnectionType = "ssh" | "cloud_api";
export type GpuType = "none" | "T4" | "A40" | "A100" | "H100" | "H200" | "MI300" | "other";
export type GpuTier = "shared-low" | "shared-mid" | "shared-high" | "dedicated-A100" | "dedicated-H100" | "dedicated-H200" | "enterprise";
export type ComputeProvider = "hetzner" | "vast" | "onprem" | "aws" | "azure" | "gcp" | "other";

export const sshConfigStatusEnum = pgEnum("ssh_config_status", ["connected", "disconnected", "error", "unknown"]);

export const sshAuthMethodEnum = pgEnum("ssh_auth_method", ["key", "password"]);

export const sshConfigs = pgTable("ssh_configs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  host: text("host").notNull(),
  port: integer("port").default(22),
  username: text("username").notNull(),
  authMethod: sshAuthMethodEnum("auth_method").default("key"),
  fingerprint: text("fingerprint"),
  serviceLabel: text("service_label"),
  companyId: varchar("company_id"),
  status: sshConfigStatusEnum("status").default("unknown"),
  lastConnected: timestamp("last_connected"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertSshConfigSchema = createInsertSchema(sshConfigs).omit({ id: true, createdAt: true, updatedAt: true, lastConnected: true });
export type SshConfig = typeof sshConfigs.$inferSelect;
export type InsertSshConfig = z.infer<typeof insertSshConfigSchema>;

export const companies = pgTable("companies", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  slug: text("slug").notNull(),
  gpuTier: gpuTierEnum("gpu_tier"),
  defaultComputeNodeId: varchar("default_compute_node_id"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertCompanySchema = createInsertSchema(companies).omit({ id: true, createdAt: true, updatedAt: true });
export type Company = typeof companies.$inferSelect;
export type InsertCompany = z.infer<typeof insertCompanySchema>;

export const userSshKeys = pgTable("user_ssh_keys", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull(),
  publicKey: text("public_key").notNull(),
  label: text("label"),
  fingerprint: text("fingerprint"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertUserSshKeySchema = createInsertSchema(userSshKeys).omit({ id: true, createdAt: true });
export type UserSshKey = typeof userSshKeys.$inferSelect;
export type InsertUserSshKey = z.infer<typeof insertUserSshKeySchema>;

export const nodeKeyRegistrations = pgTable("node_key_registrations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  nodeId: varchar("node_id").notNull().references(() => computeNodes.id, { onDelete: "cascade" }),
  sshKeyId: varchar("ssh_key_id").notNull().references(() => userSshKeys.id, { onDelete: "cascade" }),
  status: text("status").default("pending"),
  registeredAt: timestamp("registered_at").defaultNow(),
});

export const insertNodeKeyRegistrationSchema = createInsertSchema(nodeKeyRegistrations).omit({ id: true, registeredAt: true });
export type NodeKeyRegistration = typeof nodeKeyRegistrations.$inferSelect;
export type InsertNodeKeyRegistration = z.infer<typeof insertNodeKeyRegistrationSchema>;

export const resourceTypeEnum = pgEnum("resource_type", ["cpu_time", "gpu_time", "storage_gb"]);
export const usageUnitEnum = pgEnum("usage_unit", ["seconds", "hours", "gb"]);
export const usageSourceEnum = pgEnum("usage_source", ["hetzner", "vastai", "internal"]);
export const ownerTypeEnum = pgEnum("owner_type", ["user", "org"]);
export const currencyTypeEnum = pgEnum("currency_type", ["USD", "EUR", "CREDITS"]);

export const usageMeters = pgTable("usage_meters", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id"),
  projectId: varchar("project_id").references(() => projects.id, { onDelete: "set null" }),
  campaignId: varchar("campaign_id").references(() => campaigns.id, { onDelete: "set null" }),
  resourceType: resourceTypeEnum("resource_type").notNull(),
  amount: real("amount").notNull(),
  unit: usageUnitEnum("unit").notNull(),
  source: usageSourceEnum("source").default("internal"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertUsageMeterSchema = createInsertSchema(usageMeters).omit({ id: true, createdAt: true });
export type UsageMeter = typeof usageMeters.$inferSelect;
export type InsertUsageMeter = z.infer<typeof insertUsageMeterSchema>;

export const creditWallets = pgTable("credit_wallets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  ownerType: ownerTypeEnum("owner_type").notNull(),
  ownerId: varchar("owner_id").notNull(),
  balance: real("balance").default(0).notNull(),
  currency: currencyTypeEnum("currency").default("CREDITS"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertCreditWalletSchema = createInsertSchema(creditWallets).omit({ id: true, createdAt: true, updatedAt: true });
export type CreditWallet = typeof creditWallets.$inferSelect;
export type InsertCreditWallet = z.infer<typeof insertCreditWalletSchema>;

export const creditTransactions = pgTable("credit_transactions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  walletId: varchar("wallet_id").notNull().references(() => creditWallets.id, { onDelete: "cascade" }),
  delta: real("delta").notNull(),
  reason: text("reason"),
  usageMeterId: varchar("usage_meter_id").references(() => usageMeters.id, { onDelete: "set null" }),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertCreditTransactionSchema = createInsertSchema(creditTransactions).omit({ id: true, createdAt: true });
export type CreditTransaction = typeof creditTransactions.$inferSelect;
export type InsertCreditTransaction = z.infer<typeof insertCreditTransactionSchema>;

export const materialEntities = pgTable("material_entities", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: materialTypeEnum("type").notNull(),
  representation: jsonb("representation"),
  metadata: jsonb("metadata"),
  isCurated: boolean("is_curated").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMaterialEntitySchema = createInsertSchema(materialEntities).omit({ id: true, createdAt: true });
export type MaterialEntity = typeof materialEntities.$inferSelect;
export type InsertMaterialEntity = z.infer<typeof insertMaterialEntitySchema>;

export const materialProperties = pgTable("material_properties", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialId: varchar("material_id").notNull().references(() => materialEntities.id, { onDelete: "cascade" }),
  propertyName: text("property_name").notNull(),
  value: real("value"),
  units: text("units"),
  confidence: real("confidence"),
  source: materialPropertySourceEnum("source").default("ml"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const materialPropertiesRelations = relations(materialProperties, ({ one }) => ({
  material: one(materialEntities, { fields: [materialProperties.materialId], references: [materialEntities.id] }),
}));

export const insertMaterialPropertySchema = createInsertSchema(materialProperties).omit({ id: true, createdAt: true });
export type MaterialProperty = typeof materialProperties.$inferSelect;
export type InsertMaterialProperty = z.infer<typeof insertMaterialPropertySchema>;

export const materialsPrograms = pgTable("materials_programs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  materialType: materialTypeEnum("material_type"),
  ownerId: varchar("owner_id").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMaterialsProgramSchema = createInsertSchema(materialsPrograms).omit({ id: true, createdAt: true, updatedAt: true });
export type MaterialsProgram = typeof materialsPrograms.$inferSelect;
export type InsertMaterialsProgram = z.infer<typeof insertMaterialsProgramSchema>;

export const materialsCampaigns = pgTable("materials_campaigns", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  programId: varchar("program_id").references(() => materialsPrograms.id, { onDelete: "set null" }),
  name: text("name").notNull(),
  domain: discoveryDomainEnum("domain").default("materials"),
  modality: materialTypeEnum("modality"),
  pipelineConfig: jsonb("pipeline_config"),
  oracleVersionId: varchar("oracle_version_id").references(() => oracleVersions.id, { onDelete: "set null" }),
  status: campaignStatusEnum("status").default("pending"),
  ownerId: varchar("owner_id").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const materialsCampaignsRelations = relations(materialsCampaigns, ({ one }) => ({
  program: one(materialsPrograms, { fields: [materialsCampaigns.programId], references: [materialsPrograms.id] }),
  oracleVersion: one(oracleVersions, { fields: [materialsCampaigns.oracleVersionId], references: [oracleVersions.id] }),
}));

export const insertMaterialsCampaignSchema = createInsertSchema(materialsCampaigns).omit({ id: true, createdAt: true, updatedAt: true });
export type MaterialsCampaign = typeof materialsCampaigns.$inferSelect;
export type InsertMaterialsCampaign = z.infer<typeof insertMaterialsCampaignSchema>;

export const materialsOracleScores = pgTable("materials_oracle_scores", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialId: varchar("material_id").notNull().references(() => materialEntities.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").notNull().references(() => materialsCampaigns.id, { onDelete: "cascade" }),
  oracleScore: real("oracle_score"),
  propertyBreakdown: jsonb("property_breakdown"),
  synthesisFeasibility: real("synthesis_feasibility"),
  manufacturingCostFactor: real("manufacturing_cost_factor"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const materialsOracleScoresRelations = relations(materialsOracleScores, ({ one }) => ({
  material: one(materialEntities, { fields: [materialsOracleScores.materialId], references: [materialEntities.id] }),
  campaign: one(materialsCampaigns, { fields: [materialsOracleScores.campaignId], references: [materialsCampaigns.id] }),
}));

export const insertMaterialsOracleScoreSchema = createInsertSchema(materialsOracleScores).omit({ id: true, createdAt: true });
export type MaterialsOracleScore = typeof materialsOracleScores.$inferSelect;
export type InsertMaterialsOracleScore = z.infer<typeof insertMaterialsOracleScoreSchema>;

export const materialsLearningGraph = pgTable("materials_learning_graph", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  campaignId: varchar("campaign_id").references(() => materialsCampaigns.id, { onDelete: "cascade" }),
  materialId: varchar("material_id").references(() => materialEntities.id, { onDelete: "cascade" }),
  stepName: text("step_name"),
  inputPayload: jsonb("input_payload"),
  outputPayload: jsonb("output_payload"),
  label: text("label"),
  labeledAt: timestamp("labeled_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMaterialsLearningGraphSchema = createInsertSchema(materialsLearningGraph).omit({ id: true, createdAt: true });
export type MaterialsLearningGraphEntry = typeof materialsLearningGraph.$inferSelect;
export type InsertMaterialsLearningGraphEntry = z.infer<typeof insertMaterialsLearningGraphSchema>;

export type DiseaseArea = "CNS" | "Oncology" | "Rare" | "Infectious" | "Cardiometabolic" | "Autoimmune" | "Respiratory" | "Other";
export type LibraryType = "internal" | "uploaded" | "generated";
export type LibraryStatus = "draft" | "processing" | "curated" | "deprecated";
export type CleaningStatus = "pending" | "cleaning" | "validated" | "failed";
export type CampaignStatus = "pending" | "running" | "completed" | "failed";
export type JobStatus = "pending" | "running" | "completed" | "failed";
export type JobType = "generation" | "filtering" | "docking" | "scoring" | "quantum_optimization" | "quantum_scoring" | "other";
export type OutcomeLabel = "promising" | "dropped" | "hit" | "unknown";
export type ProviderType = "bionemo" | "ml" | "docking" | "quantum" | "ip" | "literature" | "smiles_library" | "agent" | "materials_library" | "simulation" | "oracle" | "selection";
export type DiscoveryDomain = "drug" | "materials";
export type MaterialType = "polymer" | "crystal" | "composite" | "surface" | "membrane" | "catalyst";
export type MaterialPropertySource = "ml" | "simulation" | "experiment";
export type Modality = "small_molecule" | "fragment" | "protac" | "peptide" | "other";
export type AssayType = "binding" | "functional" | "in_vivo" | "pk" | "admet" | "other";
export type AssayOutcome = "active" | "inactive" | "toxic" | "no_effect" | "inconclusive";
export type OrgRole = "admin" | "member" | "viewer";
export type AssetType = "smiles_library" | "pipeline_template" | "program";
export type SharePermission = "read" | "fork";
export type ComputePurpose = "ml" | "bionemo" | "docking" | "quantum" | "agents" | "general";
export type ComputeStatus = "active" | "offline" | "degraded";
export type ResourceType = "cpu_time" | "gpu_time" | "storage_gb";
export type UsageUnit = "seconds" | "hours" | "gb";
export type UsageSource = "hetzner" | "vastai" | "internal";
export type OwnerType = "user" | "org";
export type CurrencyType = "USD" | "EUR" | "CREDITS";

export type ServiceAccountRole = "agent_pipeline_copilot" | "agent_operator" | "agent_readonly";

export interface ServiceAccount {
  id: string;
  name: string;
  role: ServiceAccountRole;
  allowedActions: string[];
  apiKey?: string;
  createdAt?: Date;
}

export interface PipelineStep {
  name: string;
  provider: ProviderType;
  operation: string;
  params?: Record<string, unknown>;
}

export type SeedSourceType = "curated_library" | "uploaded_set" | "generated";

export interface SeedSource {
  type: SeedSourceType;
  libraryId?: string;
  libraryVersion?: number;
  filters?: {
    domainType?: DiseaseArea;
    scaffoldIds?: string[];
    cleaningStatus?: CleaningStatus;
    tags?: string[];
  };
}

export interface PipelineConfig {
  generator: "bionemo_molmim" | "upload_library" | "curated_library";
  generatorParams?: {
    seedSmiles?: string[];
    n?: number;
  };
  modality?: Modality;
  seedSource?: SeedSource;
  filteringRules?: string[];
  dockingMethod: "bionemo_diffdock" | "external_docking";
  scoringWeights: {
    wDocking: number;
    wAdmet: number;
    wQsar: number;
    wTranslational?: number;
    wSynthesis?: number;
    wVariantRobustness?: number;
    wIpRisk?: number;
  };
  targetIds: string[];
  variantIds?: string[];
  variantScoringStrategy?: "min_score" | "average" | "max_penalty";
  steps?: PipelineStep[];
  enableQuantumOptimization?: boolean;
  quantumParams?: {
    objective?: string;
    maxMolecules?: number;
    constraints?: Record<string, unknown>;
  };
  diseaseArea?: DiseaseArea;
}

export interface MaterialsPipelineStep {
  name?: string;
  provider: "materials_library" | "ml" | "simulation" | "quantum" | "oracle" | "selection";
  operation: string;
  params?: Record<string, unknown>;
}

export interface MaterialsPipelineConfig {
  domain: "materials";
  modality: MaterialType;
  steps: MaterialsPipelineStep[];
  targetProperties?: string[];
  scoringWeights?: {
    wPropertyMatch?: number;
    wSynthesisFeasibility?: number;
    wManufacturingCost?: number;
    wStability?: number;
  };
  enableQuantumOptimization?: boolean;
  quantumParams?: {
    objective?: string;
    maxMaterials?: number;
    constraints?: Record<string, unknown>;
  };
}
