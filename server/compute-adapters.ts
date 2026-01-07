import type { ComputeNode, ComputeProvider, ConnectionType } from "@shared/schema";

export interface ComputeJob {
  id: string;
  type: string;
  command: string;
  params?: Record<string, unknown>;
}

export interface ComputeJobResult {
  success: boolean;
  output?: string;
  error?: string;
  exitCode?: number;
}

export interface ComputeAdapter {
  provider: ComputeProvider;
  connectionType: ConnectionType;
  
  runJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult>;
  checkHealth(node: ComputeNode): Promise<boolean>;
  getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded">;
}

export class SshComputeAdapter implements ComputeAdapter {
  provider: ComputeProvider;
  connectionType: ConnectionType = "ssh";

  constructor(provider: ComputeProvider = "hetzner") {
    this.provider = provider;
  }

  async runJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    console.log(`[SshComputeAdapter] Running job ${job.id} on node ${node.name}`);
    console.log(`  Host: ${node.sshHost || node.ipAddress}:${node.sshPort}`);
    console.log(`  User: ${node.sshUsername}`);
    console.log(`  Command: ${job.command}`);
    
    return {
      success: true,
      output: `[STUB] SSH job ${job.id} would execute: ${job.command}`,
    };
  }

  async checkHealth(node: ComputeNode): Promise<boolean> {
    console.log(`[SshComputeAdapter] Checking health of ${node.name} (${node.sshHost || node.ipAddress})`);
    return true;
  }

  async getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded"> {
    const isHealthy = await this.checkHealth(node);
    return isHealthy ? "active" : "offline";
  }
}

export class CloudApiComputeAdapter implements ComputeAdapter {
  provider: ComputeProvider;
  connectionType: ConnectionType = "cloud_api";

  constructor(provider: ComputeProvider) {
    this.provider = provider;
  }

  async runJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    console.log(`[CloudApiComputeAdapter] Running job ${job.id} on ${this.provider} node ${node.name}`);
    console.log(`  Region: ${node.region}`);
    console.log(`  GPU Type: ${node.gpuType}`);
    console.log(`  Command: ${job.command}`);

    switch (this.provider) {
      case "aws":
        return this.runAwsJob(node, job);
      case "azure":
        return this.runAzureJob(node, job);
      case "gcp":
        return this.runGcpJob(node, job);
      default:
        return {
          success: false,
          error: `Unsupported cloud provider: ${this.provider}`,
        };
    }
  }

  private async runAwsJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    console.log(`[AWS] Would use AWS_ACCESS_KEY_ID/SECRET to run job on ${node.region}`);
    return {
      success: true,
      output: `[STUB] AWS job ${job.id} would execute on ${node.region}`,
    };
  }

  private async runAzureJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    console.log(`[Azure] Would use AZURE_CLIENT_ID/SECRET to run job on ${node.region}`);
    return {
      success: true,
      output: `[STUB] Azure job ${job.id} would execute on ${node.region}`,
    };
  }

  private async runGcpJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    console.log(`[GCP] Would use GCP credentials to run job on ${node.region}`);
    return {
      success: true,
      output: `[STUB] GCP job ${job.id} would execute on ${node.region}`,
    };
  }

  async checkHealth(node: ComputeNode): Promise<boolean> {
    console.log(`[CloudApiComputeAdapter] Checking health of ${this.provider} node ${node.name}`);
    return true;
  }

  async getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded"> {
    const isHealthy = await this.checkHealth(node);
    return isHealthy ? "active" : "offline";
  }
}

export function getComputeAdapter(node: ComputeNode): ComputeAdapter {
  if (node.connectionType === "cloud_api") {
    return new CloudApiComputeAdapter(node.provider as ComputeProvider);
  }
  return new SshComputeAdapter(node.provider as ComputeProvider);
}

export async function selectComputeNodeForTier(
  nodes: ComputeNode[],
  tier: string,
  preferredNodeId?: string
): Promise<ComputeNode | null> {
  if (preferredNodeId) {
    const preferred = nodes.find(n => n.id === preferredNodeId && n.status === "active");
    if (preferred) {
      return preferred;
    }
  }

  const defaultNode = nodes.find(
    n => n.tier === tier && n.isDefault === true && n.status === "active"
  );
  if (defaultNode) {
    return defaultNode;
  }

  const anyMatchingNode = nodes.find(
    n => n.tier === tier && n.status === "active"
  );
  return anyMatchingNode || null;
}

export async function seedDefaultComputeNode(
  createNode: (data: Partial<ComputeNode>) => Promise<ComputeNode>,
  existingNodes: ComputeNode[]
): Promise<ComputeNode | null> {
  if (existingNodes.length > 0) {
    console.log("[Compute] Nodes already exist, skipping seed");
    return null;
  }

  const sharedHost = process.env.SHARED_GPU_HOST;
  const sharedUser = process.env.SHARED_GPU_USER;
  const sharedPort = process.env.SHARED_GPU_PORT || "22";

  if (!sharedHost || !sharedUser) {
    console.log("[Compute] No SHARED_GPU_* env vars set, skipping default node seed");
    return null;
  }

  console.log(`[Compute] Seeding default shared-low compute node from env vars`);
  
  const node = await createNode({
    name: "Default Shared GPU",
    provider: "hetzner",
    connectionType: "ssh",
    gpuType: "T4",
    tier: "shared-low",
    purpose: "general",
    sshHost: sharedHost,
    sshPort: sharedPort,
    sshUsername: sharedUser,
    isDefault: true,
    status: "active",
  });

  console.log(`[Compute] Created default node: ${node.id}`);
  return node;
}
