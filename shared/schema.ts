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
export const gpuTypeEnum = pgEnum("gpu_type", ["none", "T4", "A40", "A100", "H100", "H200", "MI300", "RTX3090", "RTX4090", "other"]);
export const gpuTierEnum = pgEnum("gpu_tier", ["shared-low", "shared-mid", "shared-high", "dedicated-A100", "dedicated-H100", "dedicated-H200", "enterprise"]);
export const modalityEnum = pgEnum("modality", ["small_molecule", "fragment", "protac", "peptide", "other"]);
export const assayTypeEnum = pgEnum("assay_type", ["binding", "functional", "in_vivo", "pk", "admet", "other"]);
export const assayCategoryEnum = pgEnum("assay_category", ["target_engagement", "functional_cellular", "adme_pk", "safety_selectivity", "advanced_in_vivo"]);
export const assaySourceEnum = pgEnum("assay_source", ["experimental", "predicted", "literature"]);
export const assayDirectionEnum = pgEnum("assay_direction", ["lower_is_better", "higher_is_better"]);
export const assayOutcomeEnum = pgEnum("assay_outcome", ["active", "inactive", "toxic", "no_effect", "inconclusive"]);
export const orgRoleEnum = pgEnum("org_role", ["admin", "member", "viewer"]);
export const assetTypeEnum = pgEnum("asset_type", ["smiles_library", "pipeline_template", "program"]);
export const sharePermissionEnum = pgEnum("share_permission", ["read", "fork"]);
export const discoveryDomainEnum = pgEnum("discovery_domain", ["drug", "materials"]);
export const materialTypeEnum = pgEnum("material_type", [
  "polymer", "crystal", "composite", "surface", "membrane", "catalyst", "coating",
  "thin_film", "doped_semiconductor", "binary_oxide", "binary_chalcogenide", "binary_pnictide",
  "binary_alloy", "ternary_alloy", "high_entropy_alloy", "perovskite", "double_perovskite",
  "battery_cathode", "battery_anode", "solid_electrolyte", "spinel", "mxene_2d", "tmd_2d",
  "homopolymer", "copolymer", "2d_material"
]);
export const materialPropertySourceEnum = pgEnum("material_property_source", ["ml", "simulation", "experiment"]);
export const variantGeneratedByEnum = pgEnum("variant_generated_by", ["human", "ml", "genetic", "quantum"]);
export const processingJobStatusEnum = pgEnum("processing_job_status", ["queued", "dispatched", "running", "succeeded", "failed", "cancelled", "paused"]);
export const processingJobTypeEnum = pgEnum("processing_job_type", [
  "property_prediction", "simulation", "variant_generation", "optimization", "screening", "aggregation",
  "docking", "fingerprint_generation", "ml_training", "distributed_prediction", "full_pipeline",
  "mat_battery", "mat_solar", "mat_superconductor", "mat_catalyst", "mat_thermoelectric",
  "mat_pfas_replacement", "mat_aerospace", "mat_biomedical", "mat_semiconductor",
  "mat_construction", "mat_transparent", "mat_magnet", "mat_electrolyte", "mat_water", "mat_carbon_capture",
  "alzheimers_multitarget", "vaccine_discovery", "oncology_multitarget"
]);

export const canonicalMoleculeSourceEnum = pgEnum("canonical_molecule_source", ["import", "built_in", "vendor", "generated"]);
export const canonicalAssayOutcomeEnum = pgEnum("canonical_assay_outcome", ["active", "inactive", "toxic", "ambiguous"]);
export const descriptorMethodEnum = pgEnum("descriptor_method", ["rdkit", "mordred", "custom"]);
export const fingerprintTypeEnum = pgEnum("fingerprint_type", ["morgan", "ecfp4", "maccs", "topological", "custom"]);
export const targetAssetTypeEnum = pgEnum("target_asset_type", ["pdb", "mmcif", "pocket", "surface", "other"]);
export const materialPropertyMethodEnum = pgEnum("material_property_method", ["ML", "MD", "DFT", "FEM", "experimental"]);
export const simulationTypeEnum = pgEnum("simulation_type", ["MD", "DFT", "FEM", "MC", "other"]);
export const simulationStatusEnum = pgEnum("simulation_status", ["queued", "running", "succeeded", "failed", "cancelled"]);
export const scaleRiskEnum = pgEnum("scale_risk", ["low", "medium", "high"]);

export const projects = pgTable("projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  diseaseArea: diseaseAreaEnum("disease_area").default("Other"),
  ownerId: varchar("owner_id").notNull(),
  isDemo: boolean("is_demo").default(false),
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
  pdbId: text("pdb_id"),
  sequence: text("sequence"),
  hasStructure: boolean("has_structure").default(false),
  structureSource: structureSourceEnum("structure_source"),
  isDemo: boolean("is_demo").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  geneName: text("gene_name"),
  organism: text("organism"),
  chemblId: text("chembl_id"),
  smiles: text("smiles"),
  sequenceLength: integer("sequence_length"),
  numStructures: integer("num_structures"),
});

export const diseaseTargetMappings = pgTable("disease_target_mappings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  targetId: varchar("target_id").notNull().references(() => targets.id, { onDelete: "cascade" }),
  disease: text("disease").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const diseaseTargetMappingsRelations = relations(diseaseTargetMappings, ({ one }) => ({
  target: one(targets, { fields: [diseaseTargetMappings.targetId], references: [targets.id] }),
}));

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
  name: text("name"),
  seriesId: text("series_id"),
  scaffoldId: text("scaffold_id"),
  source: moleculeSourceEnum("source").default("generated"),
  molecularWeight: real("molecular_weight"),
  logP: real("log_p"),
  numHBondDonors: integer("num_hbond_donors"),
  numHBondAcceptors: integer("num_hbond_acceptors"),
  isDemo: boolean("is_demo").default(false),
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
  isDemo: boolean("is_demo").default(false),
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

export const assayReadoutTypeEnum = pgEnum("assay_readout_type", ["IC50", "EC50", "percent_inhibition", "AUC", "Ki", "Kd", "other"]);

export const assays = pgTable("assays", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  projectId: varchar("project_id").references(() => projects.id, { onDelete: "cascade" }),
  targetId: varchar("target_id").references(() => targets.id, { onDelete: "set null" }),
  diseaseId: varchar("disease_id"),
  companyId: varchar("company_id"),
  type: assayTypeEnum("type").default("binding"),
  category: assayCategoryEnum("category").default("target_engagement"),
  readoutType: assayReadoutTypeEnum("readout_type").default("IC50"),
  units: text("units"),
  direction: assayDirectionEnum("direction").default("lower_is_better"),
  source: assaySourceEnum("source").default("predicted"),
  description: text("description"),
  estimatedCost: real("estimated_cost"),
  estimatedDurationDays: real("estimated_duration_days"),
  isDefault: boolean("is_default").default(false),
  isPredicted: boolean("is_predicted").default(true),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  index("idx_assays_project").on(table.projectId),
  index("idx_assays_target").on(table.targetId),
  index("idx_assays_category").on(table.category),
]);

export const assaysRelations = relations(assays, ({ one, many }) => ({
  project: one(projects, { fields: [assays.projectId], references: [projects.id] }),
  target: one(targets, { fields: [assays.targetId], references: [targets.id] }),
  results: many(assayResults),
  recommendations: many(experimentRecommendations),
  assayTargets: many(assayTargets),
}));

export const assayTargets = pgTable("assay_targets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assayId: varchar("assay_id").notNull().references(() => assays.id, { onDelete: "cascade" }),
  targetId: varchar("target_id").notNull().references(() => targets.id, { onDelete: "cascade" }),
  weight: real("weight").default(1.0),
  role: text("role").default("primary"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_assay_targets_assay").on(table.assayId),
  index("idx_assay_targets_target").on(table.targetId),
]);

export const assayTargetsRelations = relations(assayTargets, ({ one }) => ({
  assay: one(assays, { fields: [assayTargets.assayId], references: [assays.id] }),
  target: one(targets, { fields: [assayTargets.targetId], references: [targets.id] }),
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
  concentration: real("concentration"),
  value: real("value").notNull(),
  units: text("units"),
  source: assaySourceEnum("source").default("predicted"),
  confidence: real("confidence").default(0.5),
  outcomeLabel: assayOutcomeEnum("outcome_label"),
  replicateId: text("replicate_id"),
  notes: text("notes"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_assay_results_assay").on(table.assayId),
  index("idx_assay_results_molecule").on(table.moleculeId),
  index("idx_assay_results_campaign").on(table.campaignId),
  index("idx_assay_results_source").on(table.source),
]);

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
export const insertAssayTargetSchema = createInsertSchema(assayTargets).omit({ id: true, createdAt: true });
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
export type AssayTarget = typeof assayTargets.$inferSelect;
export type InsertAssayTarget = z.infer<typeof insertAssayTargetSchema>;
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
export type GpuType = "none" | "T4" | "A40" | "A100" | "H100" | "H200" | "MI300" | "RTX3090" | "RTX4090" | "other";
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
  name: text("name"),
  type: materialTypeEnum("type").notNull(),
  representation: jsonb("representation"),
  baseFamily: text("base_family"),
  metadata: jsonb("metadata"),
  isCurated: boolean("is_curated").default(false),
  companyId: varchar("company_id"),
  isDemo: boolean("is_demo").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMaterialEntitySchema = createInsertSchema(materialEntities).omit({ id: true, createdAt: true });
export type MaterialEntity = typeof materialEntities.$inferSelect;
export type InsertMaterialEntity = z.infer<typeof insertMaterialEntitySchema>;

export const materialVariants = pgTable("material_variants", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialId: varchar("material_id").notNull().references(() => materialEntities.id, { onDelete: "cascade" }),
  variantParams: jsonb("variant_params"),
  generatedBy: variantGeneratedByEnum("generated_by").default("human"),
  simulationState: text("simulation_state"),
  manufacturabilityScore: real("manufacturability_score"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const materialVariantsRelations = relations(materialVariants, ({ one }) => ({
  material: one(materialEntities, { fields: [materialVariants.materialId], references: [materialEntities.id] }),
}));

export const insertMaterialVariantSchema = createInsertSchema(materialVariants).omit({ id: true, createdAt: true });
export type MaterialVariant = typeof materialVariants.$inferSelect;
export type InsertMaterialVariant = z.infer<typeof insertMaterialVariantSchema>;

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
  isDemo: boolean("is_demo").default(false),
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

export const processingJobs = pgTable("processing_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: processingJobTypeEnum("type").notNull(),
  status: processingJobStatusEnum("status").default("queued"),
  priority: integer("priority").default(0),
  campaignId: varchar("campaign_id"),
  materialsCampaignId: varchar("materials_campaign_id"),
  organizationId: varchar("organization_id"),
  computeNodeId: varchar("compute_node_id"),
  itemsTotal: integer("items_total").default(0),
  itemsCompleted: integer("items_completed").default(0),
  progressPercent: real("progress_percent").default(0),
  checkpointData: jsonb("checkpoint_data"),
  inputPayload: jsonb("input_payload"),
  outputPayload: jsonb("output_payload"),
  errorMessage: text("error_message"),
  retryCount: integer("retry_count").default(0),
  maxRetries: integer("max_retries").default(3),
  heartbeatAt: timestamp("heartbeat_at"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertProcessingJobSchema = createInsertSchema(processingJobs).omit({ id: true, createdAt: true, updatedAt: true });
export type ProcessingJob = typeof processingJobs.$inferSelect;
export type InsertProcessingJob = z.infer<typeof insertProcessingJobSchema>;

export const processingJobRuns = pgTable("processing_job_runs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  jobId: varchar("job_id").notNull().references(() => processingJobs.id, { onDelete: "cascade" }),
  runNumber: integer("run_number").default(1),
  computeNodeId: varchar("compute_node_id"),
  status: processingJobStatusEnum("status").default("running"),
  checkpointData: jsonb("checkpoint_data"),
  startedAt: timestamp("started_at").defaultNow(),
  completedAt: timestamp("completed_at"),
  errorMessage: text("error_message"),
});

export const processingJobRunsRelations = relations(processingJobRuns, ({ one }) => ({
  job: one(processingJobs, { fields: [processingJobRuns.jobId], references: [processingJobs.id] }),
}));

export const insertProcessingJobRunSchema = createInsertSchema(processingJobRuns).omit({ id: true });
export type ProcessingJobRun = typeof processingJobRuns.$inferSelect;
export type InsertProcessingJobRun = z.infer<typeof insertProcessingJobRunSchema>;

export const processingJobEvents = pgTable("processing_job_events", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  jobId: varchar("job_id").notNull().references(() => processingJobs.id, { onDelete: "cascade" }),
  eventType: text("event_type").notNull(),
  payload: jsonb("payload"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const processingJobEventsRelations = relations(processingJobEvents, ({ one }) => ({
  job: one(processingJobs, { fields: [processingJobEvents.jobId], references: [processingJobs.id] }),
}));

export const insertProcessingJobEventSchema = createInsertSchema(processingJobEvents).omit({ id: true, createdAt: true });
export type ProcessingJobEvent = typeof processingJobEvents.$inferSelect;
export type InsertProcessingJobEvent = z.infer<typeof insertProcessingJobEventSchema>;

export const artifactDomainEnum = pgEnum("artifact_domain", ["drug", "materials"]);
export const artifactTypeEnum = pgEnum("artifact_type", ["json", "table", "image", "model3d", "report"]);

export const jobArtifacts = pgTable("job_artifacts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  jobId: varchar("job_id").notNull().references(() => processingJobs.id, { onDelete: "cascade" }),
  companyId: varchar("company_id"),
  campaignId: varchar("campaign_id"),
  materialsCampaignId: varchar("materials_campaign_id"),
  domain: artifactDomainEnum("domain").notNull(),
  artifactType: artifactTypeEnum("artifact_type").notNull(),
  name: text("name").notNull(),
  uri: text("uri").notNull(),
  mimeType: text("mime_type"),
  summaryJson: jsonb("summary_json"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const jobArtifactsRelations = relations(jobArtifacts, ({ one }) => ({
  job: one(processingJobs, { fields: [jobArtifacts.jobId], references: [processingJobs.id] }),
}));

export const insertJobArtifactSchema = createInsertSchema(jobArtifacts).omit({ id: true, createdAt: true });
export type JobArtifact = typeof jobArtifacts.$inferSelect;
export type InsertJobArtifact = z.infer<typeof insertJobArtifactSchema>;

export const materialsCampaignAggregates = pgTable("materials_campaign_aggregates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  campaignId: varchar("campaign_id").notNull().references(() => materialsCampaigns.id, { onDelete: "cascade" }),
  totalMaterials: integer("total_materials").default(0),
  totalVariants: integer("total_variants").default(0),
  variantsByType: jsonb("variants_by_type"),
  avgOracleScore: real("avg_oracle_score"),
  scoreDistribution: jsonb("score_distribution"),
  topVariantIds: jsonb("top_variant_ids"),
  propertyCorrelations: jsonb("property_correlations"),
  lastRefreshedAt: timestamp("last_refreshed_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const materialsCampaignAggregatesRelations = relations(materialsCampaignAggregates, ({ one }) => ({
  campaign: one(materialsCampaigns, { fields: [materialsCampaignAggregates.campaignId], references: [materialsCampaigns.id] }),
}));

export const insertMaterialsCampaignAggregateSchema = createInsertSchema(materialsCampaignAggregates).omit({ id: true, createdAt: true });
export type MaterialsCampaignAggregate = typeof materialsCampaignAggregates.$inferSelect;
export type InsertMaterialsCampaignAggregate = z.infer<typeof insertMaterialsCampaignAggregateSchema>;

export const materialVariantMetrics = pgTable("material_variant_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  variantId: varchar("variant_id").notNull().references(() => materialVariants.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").references(() => materialsCampaigns.id, { onDelete: "set null" }),
  propertyScores: jsonb("property_scores"),
  aggregateScore: real("aggregate_score"),
  rank: integer("rank"),
  percentile: real("percentile"),
  flags: jsonb("flags"),
  lastComputedAt: timestamp("last_computed_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const materialVariantMetricsRelations = relations(materialVariantMetrics, ({ one }) => ({
  variant: one(materialVariants, { fields: [materialVariantMetrics.variantId], references: [materialVariants.id] }),
  campaign: one(materialsCampaigns, { fields: [materialVariantMetrics.campaignId], references: [materialsCampaigns.id] }),
}));

export const insertMaterialVariantMetricSchema = createInsertSchema(materialVariantMetrics).omit({ id: true, createdAt: true });
export type MaterialVariantMetric = typeof materialVariantMetrics.$inferSelect;
export type InsertMaterialVariantMetric = z.infer<typeof insertMaterialVariantMetricSchema>;

export const importDomainEnum = pgEnum("import_domain", ["drug", "materials"]);
export const importTypeEnum = pgEnum("import_type", [
  "compound_library", "hit_list", "assay_results", "target_structures", "sar_annotation",
  "materials_library", "material_variants", "properties_dataset", "simulation_summaries", "imaging_spectroscopy"
]);
export const importStatusEnum = pgEnum("import_status", ["pending", "parsing", "validating", "ingesting", "succeeded", "failed", "cancelled"]);

export const importTemplates = pgTable("import_templates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  domain: importDomainEnum("domain").notNull(),
  importType: importTypeEnum("import_type").notNull(),
  organizationId: varchar("organization_id"),
  columnMapping: jsonb("column_mapping").notNull(),
  isDefault: boolean("is_default").default(false),
  createdBy: varchar("created_by"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertImportTemplateSchema = createInsertSchema(importTemplates).omit({ id: true, createdAt: true, updatedAt: true });
export type ImportTemplate = typeof importTemplates.$inferSelect;
export type InsertImportTemplate = z.infer<typeof insertImportTemplateSchema>;

export const importJobs = pgTable("import_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  domain: importDomainEnum("domain").notNull(),
  importType: importTypeEnum("import_type").notNull(),
  fileName: text("file_name").notNull(),
  fileType: text("file_type"),
  fileSize: integer("file_size"),
  status: importStatusEnum("status").default("pending"),
  processingJobId: varchar("processing_job_id").references(() => processingJobs.id, { onDelete: "set null" }),
  templateId: varchar("template_id").references(() => importTemplates.id, { onDelete: "set null" }),
  columnMapping: jsonb("column_mapping"),
  validationSummary: jsonb("validation_summary"),
  createdObjects: jsonb("created_objects"),
  organizationId: varchar("organization_id"),
  createdBy: varchar("created_by"),
  errorMessage: text("error_message"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const importJobsRelations = relations(importJobs, ({ one }) => ({
  processingJob: one(processingJobs, { fields: [importJobs.processingJobId], references: [processingJobs.id] }),
  template: one(importTemplates, { fields: [importJobs.templateId], references: [importTemplates.id] }),
}));

export const insertImportJobSchema = createInsertSchema(importJobs).omit({ id: true, createdAt: true });
export type ImportJob = typeof importJobs.$inferSelect;
export type InsertImportJob = z.infer<typeof insertImportJobSchema>;

export const targetRoleEnum = pgEnum("target_role", ["primary", "secondary", "safety"]);
export const moaNodeTypeEnum = pgEnum("moa_node_type", ["target", "pathway", "process", "phenotype"]);
export const moaRelationEnum = pgEnum("moa_relation", ["activates", "inhibits", "associated_with"]);

export const assayPanels = pgTable("assay_panels", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  campaignId: varchar("campaign_id").notNull().references(() => campaigns.id, { onDelete: "cascade" }),
  description: text("description"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const assayPanelsRelations = relations(assayPanels, ({ one, many }) => ({
  campaign: one(campaigns, { fields: [assayPanels.campaignId], references: [campaigns.id] }),
  targets: many(assayPanelTargets),
}));

export const assayPanelTargets = pgTable("assay_panel_targets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assayPanelId: varchar("assay_panel_id").notNull().references(() => assayPanels.id, { onDelete: "cascade" }),
  targetId: varchar("target_id").notNull().references(() => targets.id, { onDelete: "cascade" }),
  role: targetRoleEnum("role").default("primary"),
});

export const assayPanelTargetsRelations = relations(assayPanelTargets, ({ one }) => ({
  panel: one(assayPanels, { fields: [assayPanelTargets.assayPanelId], references: [assayPanels.id] }),
  target: one(targets, { fields: [assayPanelTargets.targetId], references: [targets.id] }),
}));

export const moaNodes = pgTable("moa_nodes", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: moaNodeTypeEnum("type").notNull(),
  name: text("name").notNull(),
  description: text("description"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const moaEdges = pgTable("moa_edges", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  fromNodeId: varchar("from_node_id").notNull().references(() => moaNodes.id, { onDelete: "cascade" }),
  toNodeId: varchar("to_node_id").notNull().references(() => moaNodes.id, { onDelete: "cascade" }),
  relation: moaRelationEnum("relation").notNull(),
  confidence: real("confidence").default(1.0),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const moaEdgesRelations = relations(moaEdges, ({ one }) => ({
  fromNode: one(moaNodes, { fields: [moaEdges.fromNodeId], references: [moaNodes.id] }),
  toNode: one(moaNodes, { fields: [moaEdges.toNodeId], references: [moaNodes.id] }),
}));

export const insertAssayPanelSchema = createInsertSchema(assayPanels).omit({ id: true, createdAt: true });
export const insertAssayPanelTargetSchema = createInsertSchema(assayPanelTargets).omit({ id: true });
export const insertMoaNodeSchema = createInsertSchema(moaNodes).omit({ id: true, createdAt: true });
export const insertMoaEdgeSchema = createInsertSchema(moaEdges).omit({ id: true, createdAt: true });

export type AssayPanel = typeof assayPanels.$inferSelect;
export type InsertAssayPanel = z.infer<typeof insertAssayPanelSchema>;
export type AssayPanelTarget = typeof assayPanelTargets.$inferSelect;
export type InsertAssayPanelTarget = z.infer<typeof insertAssayPanelTargetSchema>;
export type MoaNode = typeof moaNodes.$inferSelect;
export type InsertMoaNode = z.infer<typeof insertMoaNodeSchema>;
export type MoaEdge = typeof moaEdges.$inferSelect;
export type InsertMoaEdge = z.infer<typeof insertMoaEdgeSchema>;

export type TargetRole = "primary" | "secondary" | "safety";
export type MoaNodeType = "target" | "pathway" | "process" | "phenotype";
export type MoaRelation = "activates" | "inhibits" | "associated_with";

export const pipelineTemplateDomainEnum = pgEnum("pipeline_template_domain", [
  "alzheimers", "oncology", "neuroinflammation", "metabolic_disease", "immunology", "infectious_disease", "custom"
]);

export const pipelineTemplates = pgTable("pipeline_templates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  domain: pipelineTemplateDomainEnum("domain").notNull(),
  isBuiltIn: boolean("is_built_in").default(false),
  pipelineConfig: jsonb("pipeline_config"),
  targetConfigs: jsonb("target_configs"),
  assayPanelConfig: jsonb("assay_panel_config"),
  scoringWeights: jsonb("scoring_weights"),
  visualizationPresets: jsonb("visualization_presets"),
  modality: modalityEnum("modality").default("small_molecule"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const pipelineTemplateTargets = pgTable("pipeline_template_targets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  templateId: varchar("template_id").notNull().references(() => pipelineTemplates.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  description: text("description"),
  role: targetRoleEnum("role").default("primary"),
  category: text("category"),
  sortOrder: integer("sort_order").default(0),
  createdAt: timestamp("created_at").defaultNow(),
});

export const pipelineTemplatesRelations = relations(pipelineTemplates, ({ many }) => ({
  targets: many(pipelineTemplateTargets),
}));

export const pipelineTemplateTargetsRelations = relations(pipelineTemplateTargets, ({ one }) => ({
  template: one(pipelineTemplates, { fields: [pipelineTemplateTargets.templateId], references: [pipelineTemplates.id] }),
}));

export const insertPipelineTemplateSchema = createInsertSchema(pipelineTemplates).omit({ id: true, createdAt: true, updatedAt: true });
export const insertPipelineTemplateTargetSchema = createInsertSchema(pipelineTemplateTargets).omit({ id: true, createdAt: true });

export type PipelineTemplate = typeof pipelineTemplates.$inferSelect;
export type InsertPipelineTemplate = z.infer<typeof insertPipelineTemplateSchema>;
export type PipelineTemplateTarget = typeof pipelineTemplateTargets.$inferSelect;
export type InsertPipelineTemplateTarget = z.infer<typeof insertPipelineTemplateTargetSchema>;
export type PipelineTemplateDomain = "alzheimers" | "oncology" | "neuroinflammation" | "metabolic_disease" | "immunology" | "infectious_disease" | "custom";

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

export type ImportDomain = "drug" | "materials";
export type ImportType = "compound_library" | "hit_list" | "assay_results" | "target_structures" | "sar_annotation" | "materials_library" | "material_variants" | "properties_dataset" | "simulation_summaries" | "imaging_spectroscopy";
export type ImportStatus = "pending" | "parsing" | "validating" | "ingesting" | "succeeded" | "failed" | "cancelled";

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
  templateId?: string;
  templateTargets?: { name: string; role: TargetRole | null; category: string | null }[];
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

export const canonicalMolecules = pgTable("canonical_molecules", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  companyId: varchar("company_id"),
  name: text("name"),
  canonicalSmiles: text("canonical_smiles").notNull(),
  inchikey: text("inchikey").notNull(),
  inchi: text("inchi"),
  source: canonicalMoleculeSourceEnum("source").default("import"),
  isBuiltIn: boolean("is_built_in").default(false),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_canonical_molecules_inchikey").on(table.inchikey),
  index("idx_canonical_molecules_company").on(table.companyId),
  index("idx_canonical_molecules_built_in").on(table.isBuiltIn),
]);

export const compoundAssetTypeEnum = pgEnum("compound_asset_type", [
  "thumbnail_2d",
  "thumbnail_3d",
  "conformer_sdf",
  "conformer_pdb",
  "descriptors_json",
  "fingerprint_json",
]);

export const compoundAssets = pgTable("compound_assets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  moleculeId: varchar("molecule_id").notNull().references(() => canonicalMolecules.id, { onDelete: "cascade" }),
  companyId: varchar("company_id"),
  assetType: compoundAssetTypeEnum("asset_type").notNull(),
  storageKey: text("storage_key").notNull(),
  storageBucket: text("storage_bucket").notNull(),
  storageProvider: text("storage_provider").default("do_spaces"),
  mimeType: text("mime_type"),
  sizeBytes: integer("size_bytes"),
  cdnUrl: text("cdn_url"),
  metadata: jsonb("metadata"),
  generatedByJobId: varchar("generated_by_job_id"),
  computeNodeId: varchar("compute_node_id"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_compound_assets_molecule").on(table.moleculeId),
  index("idx_compound_assets_type").on(table.assetType),
  index("idx_compound_assets_company").on(table.companyId),
]);

export const insertCompoundAssetSchema = createInsertSchema(compoundAssets).omit({ id: true, createdAt: true });
export type InsertCompoundAsset = z.infer<typeof insertCompoundAssetSchema>;
export type CompoundAsset = typeof compoundAssets.$inferSelect;

export const moleculeDescriptors = pgTable("molecule_descriptors", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  moleculeId: varchar("molecule_id").notNull().references(() => canonicalMolecules.id, { onDelete: "cascade" }),
  mw: real("mw"),
  logp: real("logp"),
  tpsa: real("tpsa"),
  hbd: integer("hbd"),
  hba: integer("hba"),
  rotb: integer("rotb"),
  rings: integer("rings"),
  computedAt: timestamp("computed_at").defaultNow(),
  method: descriptorMethodEnum("method").default("rdkit"),
});

export const moleculeFingerprints = pgTable("molecule_fingerprints", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  moleculeId: varchar("molecule_id").notNull().references(() => canonicalMolecules.id, { onDelete: "cascade" }),
  fpType: fingerprintTypeEnum("fp_type").default("morgan"),
  fpBits: text("fp_bits"),
  radius: integer("radius"),
  nbits: integer("nbits"),
  computedAt: timestamp("computed_at").defaultNow(),
});

export const hitLists = pgTable("hit_lists", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  campaignId: varchar("campaign_id").references(() => campaigns.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  sourceTool: text("source_tool"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const hitListItems = pgTable("hit_list_items", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  hitListId: varchar("hit_list_id").notNull().references(() => hitLists.id, { onDelete: "cascade" }),
  moleculeId: varchar("molecule_id").notNull().references(() => canonicalMolecules.id, { onDelete: "cascade" }),
  score: real("score"),
  rank: integer("rank"),
  metadataJson: jsonb("metadata_json"),
});

export const canonicalAssays = pgTable("canonical_assays", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  companyId: varchar("company_id"),
  name: text("name").notNull(),
  targetId: varchar("target_id").references(() => targets.id),
  diseaseId: varchar("disease_id"),
  readoutType: assayReadoutTypeEnum("readout_type").default("IC50"),
  units: text("units"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const canonicalAssayResults = pgTable("canonical_assay_results", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assayId: varchar("assay_id").notNull().references(() => canonicalAssays.id, { onDelete: "cascade" }),
  campaignId: varchar("campaign_id").references(() => campaigns.id),
  moleculeId: varchar("molecule_id").notNull().references(() => canonicalMolecules.id, { onDelete: "cascade" }),
  concentration: real("concentration"),
  value: real("value").notNull(),
  units: text("units"),
  outcomeLabel: canonicalAssayOutcomeEnum("outcome_label"),
  metadataJson: jsonb("metadata_json"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const targetAssets = pgTable("target_assets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  targetId: varchar("target_id").notNull().references(() => targets.id, { onDelete: "cascade" }),
  assetType: targetAssetTypeEnum("asset_type").default("pdb"),
  uri: text("uri"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const canonicalMaterials = pgTable("canonical_materials", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  companyId: varchar("company_id"),
  name: text("name"),
  structureType: materialTypeEnum("structure_type").default("polymer"),
  canonicalRepresentationJson: jsonb("canonical_representation_json"),
  materialHash: text("material_hash").notNull(),
  baseFamily: text("base_family"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_canonical_materials_hash").on(table.materialHash),
  index("idx_canonical_materials_company").on(table.companyId),
]);

export const canonicalMaterialVariants = pgTable("canonical_material_variants", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialId: varchar("material_id").notNull().references(() => canonicalMaterials.id, { onDelete: "cascade" }),
  variantParamsJson: jsonb("variant_params_json"),
  variantHash: text("variant_hash"),
  generatedBy: variantGeneratedByEnum("generated_by").default("human"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("idx_canonical_material_variants_hash").on(table.variantHash),
]);

export const canonicalMaterialProperties = pgTable("canonical_material_properties", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialVariantId: varchar("material_variant_id").notNull().references(() => canonicalMaterialVariants.id, { onDelete: "cascade" }),
  propertyName: text("property_name").notNull(),
  value: real("value"),
  units: text("units"),
  method: materialPropertyMethodEnum("method").default("ML"),
  confidence: real("confidence"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const simulationRuns = pgTable("simulation_runs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialVariantId: varchar("material_variant_id").notNull().references(() => canonicalMaterialVariants.id, { onDelete: "cascade" }),
  simulationType: simulationTypeEnum("simulation_type").default("MD"),
  parametersJson: jsonb("parameters_json"),
  status: simulationStatusEnum("status").default("queued"),
  resultsJson: jsonb("results_json"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const manufacturabilityScores = pgTable("manufacturability_scores", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  materialVariantId: varchar("material_variant_id").notNull().references(() => canonicalMaterialVariants.id, { onDelete: "cascade" }),
  feasibilityScore: real("feasibility_score"),
  costIndex: real("cost_index"),
  scaleRisk: scaleRiskEnum("scale_risk").default("medium"),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertCanonicalMoleculeSchema = createInsertSchema(canonicalMolecules).omit({ id: true, createdAt: true });
export const insertMoleculeDescriptorSchema = createInsertSchema(moleculeDescriptors).omit({ id: true, computedAt: true });
export const insertMoleculeFingerprintSchema = createInsertSchema(moleculeFingerprints).omit({ id: true, computedAt: true });
export const insertHitListSchema = createInsertSchema(hitLists).omit({ id: true, createdAt: true });
export const insertHitListItemSchema = createInsertSchema(hitListItems).omit({ id: true });
export const insertCanonicalAssaySchema = createInsertSchema(canonicalAssays).omit({ id: true, createdAt: true });
export const insertCanonicalAssayResultSchema = createInsertSchema(canonicalAssayResults).omit({ id: true, createdAt: true });
export const insertTargetAssetSchema = createInsertSchema(targetAssets).omit({ id: true, createdAt: true });
export const insertCanonicalMaterialSchema = createInsertSchema(canonicalMaterials).omit({ id: true, createdAt: true });
export const insertCanonicalMaterialVariantSchema = createInsertSchema(canonicalMaterialVariants).omit({ id: true, createdAt: true });
export const insertCanonicalMaterialPropertySchema = createInsertSchema(canonicalMaterialProperties).omit({ id: true, createdAt: true });
export const insertSimulationRunSchema = createInsertSchema(simulationRuns).omit({ id: true, createdAt: true });
export const insertManufacturabilityScoreSchema = createInsertSchema(manufacturabilityScores).omit({ id: true, createdAt: true });

export type InsertCanonicalMolecule = z.infer<typeof insertCanonicalMoleculeSchema>;
export type CanonicalMolecule = typeof canonicalMolecules.$inferSelect;
export type InsertMoleculeDescriptor = z.infer<typeof insertMoleculeDescriptorSchema>;
export type MoleculeDescriptor = typeof moleculeDescriptors.$inferSelect;
export type InsertMoleculeFingerprint = z.infer<typeof insertMoleculeFingerprintSchema>;
export type MoleculeFingerprint = typeof moleculeFingerprints.$inferSelect;
export type InsertHitList = z.infer<typeof insertHitListSchema>;
export type HitList = typeof hitLists.$inferSelect;
export type InsertHitListItem = z.infer<typeof insertHitListItemSchema>;
export type HitListItem = typeof hitListItems.$inferSelect;
export type InsertCanonicalAssay = z.infer<typeof insertCanonicalAssaySchema>;
export type CanonicalAssay = typeof canonicalAssays.$inferSelect;
export type InsertCanonicalAssayResult = z.infer<typeof insertCanonicalAssayResultSchema>;
export type CanonicalAssayResult = typeof canonicalAssayResults.$inferSelect;
export type InsertTargetAsset = z.infer<typeof insertTargetAssetSchema>;
export type TargetAsset = typeof targetAssets.$inferSelect;
export type InsertCanonicalMaterial = z.infer<typeof insertCanonicalMaterialSchema>;
export type CanonicalMaterial = typeof canonicalMaterials.$inferSelect;
export type InsertCanonicalMaterialVariant = z.infer<typeof insertCanonicalMaterialVariantSchema>;
export type CanonicalMaterialVariant = typeof canonicalMaterialVariants.$inferSelect;
export type InsertCanonicalMaterialProperty = z.infer<typeof insertCanonicalMaterialPropertySchema>;
export type CanonicalMaterialProperty = typeof canonicalMaterialProperties.$inferSelect;
export type InsertSimulationRun = z.infer<typeof insertSimulationRunSchema>;
export type SimulationRun = typeof simulationRuns.$inferSelect;
export type InsertManufacturabilityScore = z.infer<typeof insertManufacturabilityScoreSchema>;
export type ManufacturabilityScore = typeof manufacturabilityScores.$inferSelect;

// Pipeline Configuration Types
export const pipelineConfigSchema = z.object({
  // General settings
  name: z.string().min(1),
  description: z.string().optional(),
  jobType: z.enum([
    "docking", "fingerprint_generation", "ml_training", "distributed_prediction", "full_pipeline",
    "mat_battery", "mat_solar", "mat_superconductor", "mat_catalyst", "mat_thermoelectric",
    "mat_pfas_replacement", "mat_aerospace", "mat_biomedical", "mat_semiconductor",
    "mat_construction", "mat_transparent", "mat_magnet", "mat_electrolyte", "mat_water", "mat_carbon_capture"
  ]),
  
  // Compute settings
  useGpu: z.boolean().default(true),
  useRapids: z.boolean().default(false),
  useMixedPrecision: z.boolean().default(true),
  preferredNodeId: z.string().optional(),
  
  // Dask distributed settings
  nWorkers: z.number().min(1).max(64).default(4),
  threadsPerWorker: z.number().min(1).max(8).default(2),
  memoryPerWorker: z.string().default("8GB"),
  clusterAddress: z.string().optional(),
  
  // Docking settings
  docking: z.object({
    vinaExecutable: z.string().default("vina"),
    exhaustiveness: z.number().min(1).max(32).default(8),
    numModes: z.number().min(1).max(20).default(9),
    parallelJobs: z.number().min(1).max(32).default(4),
    boxSize: z.tuple([z.number(), z.number(), z.number()]).default([20, 20, 20]),
    targetId: z.string().optional(),
    bindingSiteCenter: z.tuple([z.number(), z.number(), z.number()]).optional(),
  }).optional(),
  
  // Fingerprint settings
  fingerprint: z.object({
    type: z.enum(["morgan", "ecfp4", "maccs", "topological"]).default("morgan"),
    radius: z.number().min(1).max(4).default(2),
    nBits: z.number().min(512).max(4096).default(2048),
  }).optional(),
  
  // ML model settings
  mlModel: z.object({
    type: z.enum(["neural_network", "random_forest", "xgboost", "rapids_rf"]).default("neural_network"),
    epochs: z.number().min(1).max(500).default(50),
    batchSize: z.number().min(32).max(65536).default(20000),
    learningRate: z.number().min(0.0001).max(0.1).default(0.001),
    hiddenDims: z.array(z.number()).default([1024, 512, 256, 128]),
  }).optional(),
  
  // Input data
  campaignId: z.string().optional(),
  moleculeIds: z.array(z.string()).optional(),
  smilesFile: z.string().optional(),
});

export type ComputePipelineConfig = z.infer<typeof pipelineConfigSchema>;

export const pipelineJobResultSchema = z.object({
  jobId: z.string(),
  status: z.enum(["queued", "dispatched", "running", "succeeded", "failed", "cancelled", "paused"]),
  progress: z.number().min(0).max(100),
  itemsTotal: z.number(),
  itemsCompleted: z.number(),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  computeNodeId: z.string().optional(),
  artifacts: z.array(z.object({
    name: z.string(),
    type: z.string(),
    uri: z.string(),
  })).optional(),
  errorMessage: z.string().optional(),
});

export type PipelineJobResult = z.infer<typeof pipelineJobResultSchema>;

export const conversations = pgTable("conversations", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  title: text("title").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const messages = pgTable("messages", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  conversationId: integer("conversation_id").notNull().references(() => conversations.id, { onDelete: "cascade" }),
  role: text("role").notNull(),
  content: text("content").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertConversationSchema = z.object({
  title: z.string().min(1),
});

export const insertMessageSchema = z.object({
  conversationId: z.number(),
  role: z.string(),
  content: z.string(),
});

export type Conversation = typeof conversations.$inferSelect;
export type InsertConversation = z.infer<typeof insertConversationSchema>;
export type Message = typeof messages.$inferSelect;
export type InsertMessage = z.infer<typeof insertMessageSchema>;
