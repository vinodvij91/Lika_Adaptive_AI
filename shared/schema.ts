import { sql, relations } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, boolean, real, jsonb, pgEnum, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export * from "./models/auth";

export const diseaseAreaEnum = pgEnum("disease_area", ["CNS", "Oncology", "Rare", "Infectious", "Cardiometabolic", "Autoimmune", "Respiratory", "Other"]);
export const moleculeSourceEnum = pgEnum("molecule_source", ["generated", "uploaded", "screened"]);
export const structureSourceEnum = pgEnum("structure_source", ["uploaded", "bionemo_predicted", "other"]);
export const campaignStatusEnum = pgEnum("campaign_status", ["pending", "running", "completed", "failed"]);
export const jobTypeEnum = pgEnum("job_type", ["generation", "filtering", "docking", "scoring", "quantum_optimization", "quantum_scoring", "other"]);
export const jobStatusEnum = pgEnum("job_status", ["pending", "running", "completed", "failed"]);
export const providerTypeEnum = pgEnum("provider_type", ["bionemo", "ml", "docking", "quantum"]);
export const outcomeEnum = pgEnum("outcome_label", ["promising", "dropped", "hit", "unknown"]);
export const collaboratorRoleEnum = pgEnum("collaborator_role", ["owner", "editor", "viewer"]);

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
  name: text("name").notNull(),
  domainType: diseaseAreaEnum("domain_type").default("Other"),
  pipelineConfig: jsonb("pipeline_config"),
  status: campaignStatusEnum("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const campaignsRelations = relations(campaigns, ({ one, many }) => ({
  project: one(projects, { fields: [campaigns.projectId], references: [projects.id] }),
  jobs: many(jobs),
  modelRuns: many(modelRuns),
  moleculeScores: many(moleculeScores),
  learningGraphEntries: many(learningGraphEntries),
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

export const insertProjectSchema = createInsertSchema(projects).omit({ id: true, createdAt: true, updatedAt: true });
export const insertTargetSchema = createInsertSchema(targets).omit({ id: true, createdAt: true });
export const insertMoleculeSchema = createInsertSchema(molecules).omit({ id: true, createdAt: true });
export const insertCampaignSchema = createInsertSchema(campaigns).omit({ id: true, createdAt: true, updatedAt: true });
export const insertJobSchema = createInsertSchema(jobs).omit({ id: true, createdAt: true });
export const insertModelRunSchema = createInsertSchema(modelRuns).omit({ id: true });
export const insertMoleculeScoreSchema = createInsertSchema(moleculeScores).omit({ id: true, createdAt: true });
export const insertLearningGraphEntrySchema = createInsertSchema(learningGraphEntries).omit({ id: true, createdAt: true });
export const insertCommentSchema = createInsertSchema(comments).omit({ id: true, createdAt: true });

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

export type DiseaseArea = "CNS" | "Oncology" | "Rare" | "Infectious" | "Cardiometabolic" | "Autoimmune" | "Respiratory" | "Other";
export type CampaignStatus = "pending" | "running" | "completed" | "failed";
export type JobStatus = "pending" | "running" | "completed" | "failed";
export type JobType = "generation" | "filtering" | "docking" | "scoring" | "quantum_optimization" | "quantum_scoring" | "other";
export type OutcomeLabel = "promising" | "dropped" | "hit" | "unknown";
export type ProviderType = "bionemo" | "ml" | "docking" | "quantum";

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

export interface PipelineConfig {
  generator: "bionemo_molmim" | "upload_library";
  generatorParams?: {
    seedSmiles?: string[];
    n?: number;
  };
  filteringRules?: string[];
  dockingMethod: "bionemo_diffdock" | "external_docking";
  scoringWeights: {
    wDocking: number;
    wAdmet: number;
    wQsar: number;
  };
  targetIds: string[];
  steps?: PipelineStep[];
  enableQuantumOptimization?: boolean;
  quantumParams?: {
    objective?: string;
    maxMolecules?: number;
    constraints?: Record<string, unknown>;
  };
}
