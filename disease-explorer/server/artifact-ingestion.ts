import { storage } from "./storage";
import { z } from "zod";
import type { InsertJobArtifact } from "@shared/schema";

const manifestArtifactSchema = z.object({
  name: z.string().min(1),
  type: z.enum(["json", "table", "image", "model3d", "report"]),
  uri: z.string().min(1),
  mimeType: z.string().optional(),
  summary: z.record(z.unknown()).optional(),
});

const manifestSchema = z.object({
  domain: z.enum(["drug", "materials"]),
  artifacts: z.array(manifestArtifactSchema).min(1),
});

type ManifestInput = z.infer<typeof manifestSchema>;

export async function registerArtifactsFromManifest(
  jobId: string,
  manifestInput: unknown
): Promise<{ registered: number; errors: string[] }> {
  const errors: string[] = [];
  
  const parseResult = manifestSchema.safeParse(manifestInput);
  if (!parseResult.success) {
    return { 
      registered: 0, 
      errors: [`Invalid manifest: ${parseResult.error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')}`] 
    };
  }
  const manifest = parseResult.data;
  
  const job = await storage.getProcessingJob(jobId);
  if (!job) {
    return { registered: 0, errors: [`Job ${jobId} not found`] };
  }
  
  const campaignId = job.campaignId || undefined;
  const materialsCampaignId = job.materialsCampaignId || undefined;
  const organizationId = job.organizationId || undefined;
  
  const artifactsToInsert: InsertJobArtifact[] = manifest.artifacts.map((artifact) => ({
    jobId,
    companyId: organizationId,
    campaignId,
    materialsCampaignId,
    domain: manifest.domain,
    artifactType: artifact.type,
    name: artifact.name,
    uri: artifact.uri,
    mimeType: artifact.mimeType || getMimeTypeFromUri(artifact.uri),
    summaryJson: artifact.summary,
  }));
  
  try {
    await storage.createJobArtifactsBatch(artifactsToInsert);
    return { registered: artifactsToInsert.length, errors };
  } catch (error) {
    errors.push(`Failed to insert artifacts: ${error}`);
    return { registered: 0, errors };
  }
}

function getMimeTypeFromUri(uri: string): string {
  const ext = uri.split('.').pop()?.toLowerCase();
  const mimeTypes: Record<string, string> = {
    json: "application/json",
    csv: "text/csv",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    gif: "image/gif",
    svg: "image/svg+xml",
    webp: "image/webp",
    glb: "model/gltf-binary",
    gltf: "model/gltf+json",
    obj: "model/obj",
    stl: "model/stl",
    pdb: "chemical/x-pdb",
    xyz: "chemical/x-xyz",
    html: "text/html",
    pdf: "application/pdf",
    md: "text/markdown",
  };
  return mimeTypes[ext || ""] || "application/octet-stream";
}
