import { storage } from "./storage";
import {
  getComputeAdapter,
  selectNodeByCapability,
  buildPipelineCommand,
  getCpuVsGpuSteps,
  type ComputeJob,
  type ComputeJobResult,
} from "./compute-adapters";
import type { ComputeNode, ProcessingJob } from "@shared/schema";

export interface PipelineExecutionConfig {
  pipelineScript: string;
  workingDir: string;
  environment?: Record<string, string>;
  timeout?: number;
}

export interface StepResult {
  stepName: string;
  success: boolean;
  output?: string;
  error?: string;
  nodeUsed: string;
  gpuUsed: boolean;
  durationSeconds: number;
}

export interface PipelineExecutionResult {
  success: boolean;
  steps: StepResult[];
  totalCpuTimeSeconds: number;
  totalGpuTimeSeconds: number;
  outputData?: any;
  error?: string;
}

const DEFAULT_CONFIG: PipelineExecutionConfig = {
  pipelineScript: "drug_discovery_pipeline.py",
  workingDir: "/opt/lika-compute",
  environment: {
    PYTHONUNBUFFERED: "1",
    CUDA_VISIBLE_DEVICES: "0,1",
  },
  timeout: 3600000,
};

interface StagingResult {
  success: boolean;
  remoteInputFile?: string;
}

async function stageFilesToNode(
  node: ComputeNode,
  adapter: any,
  pipelineScript: string,
  inputData?: Record<string, unknown>
): Promise<StagingResult> {
  if (!adapter.uploadFile) {
    console.log(`[ComputeExecutor] Node ${node.name} adapter doesn't support file upload, assuming pre-provisioned`);
    return { success: true };
  }

  try {
    const fs = await import("fs");
    const path = await import("path");
    const os = await import("os");
    
    const localPipelinePath = `./compute/${pipelineScript}`;
    const remotePipelinePath = `/opt/lika-compute/${pipelineScript}`;
    
    if (fs.existsSync(localPipelinePath)) {
      console.log(`[ComputeExecutor] Staging pipeline to ${node.name}: ${remotePipelinePath}`);
      await adapter.uploadFile(node, localPipelinePath, remotePipelinePath);
      console.log(`[ComputeExecutor] Successfully staged pipeline to ${node.name}`);
    }
    
    if (inputData) {
      const tempDir = os.tmpdir();
      const inputFileName = `input_${Date.now()}.json`;
      const localInputPath = path.join(tempDir, inputFileName);
      const remoteInputPath = `/opt/lika-compute/${inputFileName}`;
      
      fs.writeFileSync(localInputPath, JSON.stringify(inputData, null, 2));
      
      console.log(`[ComputeExecutor] Staging input data to ${node.name}: ${remoteInputPath}`);
      const inputUploaded = await adapter.uploadFile(node, localInputPath, remoteInputPath);
      
      fs.unlinkSync(localInputPath);
      
      if (inputUploaded) {
        console.log(`[ComputeExecutor] Successfully staged input data to ${node.name}`);
        return { success: true, remoteInputFile: remoteInputPath };
      }
      
      return { success: false };
    }
    
    return { success: true };
  } catch (err: any) {
    console.error(`[ComputeExecutor] Failed to stage files to ${node.name}:`, err.message);
    return { success: false };
  }
}

export class ComputeExecutor {
  private nodes: ComputeNode[] = [];
  private config: PipelineExecutionConfig;

  constructor(config: Partial<PipelineExecutionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async refreshNodes(): Promise<void> {
    this.nodes = await storage.getComputeNodes();
  }

  async executePipelineJob(
    processingJob: ProcessingJob,
    params: Record<string, unknown>
  ): Promise<PipelineExecutionResult> {
    await this.refreshNodes();

    const jobType = processingJob.type;
    const { cpuSteps, gpuSteps } = getCpuVsGpuSteps(jobType);

    console.log(`[ComputeExecutor] Executing job ${processingJob.id} (${jobType})`);
    console.log(`  CPU steps: ${cpuSteps.join(", ") || "none"}`);
    console.log(`  GPU steps: ${gpuSteps.join(", ") || "none"}`);

    const results: StepResult[] = [];
    let totalCpuTime = 0;
    let totalGpuTime = 0;

    try {
      for (const step of cpuSteps) {
        const cpuNode = await selectNodeByCapability(this.nodes, false, "hetzner");
        if (!cpuNode) {
          return {
            success: false,
            steps: results,
            totalCpuTimeSeconds: totalCpuTime,
            totalGpuTimeSeconds: totalGpuTime,
            error: `No CPU node available for step: ${step}`,
          };
        }

        const stepResult = await this.executeStep(cpuNode, step, params);
        results.push(stepResult);

        if (!stepResult.success) {
          return {
            success: false,
            steps: results,
            totalCpuTimeSeconds: totalCpuTime,
            totalGpuTimeSeconds: totalGpuTime,
            error: `CPU step "${step}" failed: ${stepResult.error}`,
          };
        }

        totalCpuTime += stepResult.durationSeconds;
      }

      for (const step of gpuSteps) {
        const gpuNode = await selectNodeByCapability(this.nodes, true, "vast");
        if (!gpuNode) {
          return {
            success: false,
            steps: results,
            totalCpuTimeSeconds: totalCpuTime,
            totalGpuTimeSeconds: totalGpuTime,
            error: `No GPU node available for step: ${step}`,
          };
        }

        const stepResult = await this.executeStep(gpuNode, step, params);
        results.push(stepResult);

        if (!stepResult.success) {
          return {
            success: false,
            steps: results,
            totalCpuTimeSeconds: totalCpuTime,
            totalGpuTimeSeconds: totalGpuTime,
            error: `GPU step "${step}" failed: ${stepResult.error}`,
          };
        }

        totalGpuTime += stepResult.durationSeconds;
      }

      return {
        success: true,
        steps: results,
        totalCpuTimeSeconds: totalCpuTime,
        totalGpuTimeSeconds: totalGpuTime,
        outputData: this.extractOutputData(results),
      };
    } catch (err: any) {
      return {
        success: false,
        steps: results,
        totalCpuTimeSeconds: totalCpuTime,
        totalGpuTimeSeconds: totalGpuTime,
        error: err.message,
      };
    }
  }

  private async executeStep(
    node: ComputeNode,
    stepName: string,
    params: Record<string, unknown>
  ): Promise<StepResult> {
    const startTime = Date.now();
    const adapter = getComputeAdapter(node);

    const staging = await stageFilesToNode(node, adapter, this.config.pipelineScript, params);

    const command = buildPipelineCommand(
      this.config.pipelineScript,
      stepName,
      params,
      staging.remoteInputFile
    );

    const job: ComputeJob = {
      id: `${stepName}-${Date.now()}`,
      type: stepName,
      command,
      params,
      workingDir: this.config.workingDir,
      environment: this.config.environment,
      timeout: this.config.timeout,
    };

    console.log(`[ComputeExecutor] Running step "${stepName}" on ${node.name} (${node.provider})`);

    const result = await adapter.runJob(node, job);
    const durationSeconds = (Date.now() - startTime) / 1000;

    return {
      stepName,
      success: result.success,
      output: result.output,
      error: result.error,
      nodeUsed: node.name,
      gpuUsed: result.gpuUsed || false,
      durationSeconds,
    };
  }

  private extractOutputData(results: StepResult[]): any {
    const lastSuccessfulResult = [...results]
      .reverse()
      .find((r) => r.success && r.output);

    if (!lastSuccessfulResult?.output) {
      return null;
    }

    try {
      const jsonMatch = lastSuccessfulResult.output.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch {
    }

    return { rawOutput: lastSuccessfulResult.output };
  }

  async executeFullPipeline(
    campaignId: string,
    moleculeIds: string[],
    targetId?: string
  ): Promise<PipelineExecutionResult> {
    await this.refreshNodes();

    const params = {
      campaignId,
      moleculeIds,
      targetId: targetId || "default",
      useGpu: true,
      useDask: true,
      nWorkers: 4,
    };

    const allSteps = [
      { name: "smiles_validation", requiresGpu: false },
      { name: "fingerprint_generation", requiresGpu: false },
      { name: "property_calculation", requiresGpu: false },
      { name: "ml_prediction", requiresGpu: true },
      { name: "docking_preparation", requiresGpu: false },
      { name: "vina_docking", requiresGpu: true },
      { name: "scoring", requiresGpu: true },
    ];

    const results: StepResult[] = [];
    let totalCpuTime = 0;
    let totalGpuTime = 0;

    console.log(`[ComputeExecutor] Starting full pipeline for campaign ${campaignId}`);
    console.log(`  Molecules: ${moleculeIds.length}`);
    console.log(`  Target: ${targetId || "default"}`);

    for (const step of allSteps) {
      const node = await selectNodeByCapability(
        this.nodes,
        step.requiresGpu,
        step.requiresGpu ? "vast" : "hetzner"
      );

      if (!node) {
        return {
          success: false,
          steps: results,
          totalCpuTimeSeconds: totalCpuTime,
          totalGpuTimeSeconds: totalGpuTime,
          error: `No suitable node for step: ${step.name} (GPU required: ${step.requiresGpu})`,
        };
      }

      const stepResult = await this.executeStep(node, step.name, {
        ...params,
        step: step.name,
      });
      results.push(stepResult);

      if (!stepResult.success) {
        console.error(`[ComputeExecutor] Step "${step.name}" failed on ${node.name}`);
        return {
          success: false,
          steps: results,
          totalCpuTimeSeconds: totalCpuTime,
          totalGpuTimeSeconds: totalGpuTime,
          error: `Pipeline failed at step "${step.name}": ${stepResult.error}`,
        };
      }

      if (step.requiresGpu) {
        totalGpuTime += stepResult.durationSeconds;
      } else {
        totalCpuTime += stepResult.durationSeconds;
      }

      console.log(`[ComputeExecutor] Step "${step.name}" completed in ${stepResult.durationSeconds.toFixed(1)}s`);
    }

    console.log(`[ComputeExecutor] Full pipeline completed: CPU=${totalCpuTime.toFixed(1)}s, GPU=${totalGpuTime.toFixed(1)}s`);

    return {
      success: true,
      steps: results,
      totalCpuTimeSeconds: totalCpuTime,
      totalGpuTimeSeconds: totalGpuTime,
      outputData: this.extractOutputData(results),
    };
  }

  async checkNodeHealth(): Promise<Map<string, boolean>> {
    await this.refreshNodes();
    const healthMap = new Map<string, boolean>();

    for (const node of this.nodes) {
      const adapter = getComputeAdapter(node);
      const isHealthy = await adapter.checkHealth(node);
      healthMap.set(node.id, isHealthy);

      if (node.status === "active" && !isHealthy) {
        await storage.updateComputeNode(node.id, { status: "offline" });
      } else if (node.status === "offline" && isHealthy) {
        await storage.updateComputeNode(node.id, { status: "active" });
      }
    }

    return healthMap;
  }

  async getAvailableCapacity(): Promise<{
    cpuNodes: number;
    gpuNodes: number;
    totalGpus: number;
  }> {
    await this.refreshNodes();

    const activeNodes = this.nodes.filter((n) => n.status === "active");
    const cpuNodes = activeNodes.filter((n) => n.gpuType === "none").length;
    const gpuNodes = activeNodes.filter((n) => n.gpuType !== "none");

    let totalGpus = 0;
    for (const node of gpuNodes) {
      const specs = node.specs as any;
      totalGpus += specs?.numGpus || 1;
    }

    return {
      cpuNodes,
      gpuNodes: gpuNodes.length,
      totalGpus,
    };
  }
}

export const computeExecutor = new ComputeExecutor();

export async function setupDefaultComputeNodes(): Promise<void> {
  const existingNodes = await storage.getComputeNodes();

  if (existingNodes.length > 0) {
    console.log("[ComputeExecutor] Compute nodes already configured");
    return;
  }

  console.log("[ComputeExecutor] Setting up default compute node configurations...");

  const hetznerHost = process.env.HETZNER_SSH_HOST;
  const hetznerUser = process.env.HETZNER_SSH_USER || "root";
  const hetznerPort = process.env.HETZNER_SSH_PORT || "22";

  if (hetznerHost) {
    await storage.createComputeNode({
      name: "Hetzner CPU Node",
      provider: "hetzner",
      connectionType: "ssh",
      gpuType: "none",
      tier: "shared-low",
      purpose: "general",
      sshHost: hetznerHost,
      sshPort: hetznerPort,
      sshUsername: hetznerUser,
      isDefault: true,
      status: "active",
      specs: {
        cpuCores: 8,
        memoryGb: 32,
        storageGb: 200,
      },
    });
    console.log("[ComputeExecutor] Created Hetzner CPU node");
  }

  const vastApiKey = process.env.VAST_AI_API_KEY;
  if (vastApiKey) {
    await storage.createComputeNode({
      name: "Vast.ai GPU Node (2x RTX 3090)",
      provider: "vast",
      connectionType: "cloud_api",
      gpuType: "RTX3090",
      tier: "shared-high",
      purpose: "ml",
      isDefault: true,
      status: "active",
      specs: {
        gpuName: "RTX 3090",
        numGpus: 2,
        gpuMemoryGb: 24,
        cpuCores: 16,
        memoryGb: 64,
      },
    });
    console.log("[ComputeExecutor] Created Vast.ai GPU node");
  }

  const gcpHost = process.env.GCP_SSH_HOST;
  const gcpUser = process.env.GCP_SSH_USER || "replit";
  const gcpPort = process.env.GCP_SSH_PORT || "22";
  if (gcpHost) {
    await storage.createComputeNode({
      name: "GCP GPU Node (A100)",
      provider: "gcp",
      connectionType: "ssh",
      gpuType: "A100",
      tier: "dedicated-A100",
      purpose: "ml",
      sshHost: gcpHost,
      sshPort: gcpPort,
      sshUsername: gcpUser,
      isDefault: true,
      status: "active",
      specs: {
        gpuName: "NVIDIA A100",
        numGpus: 1,
        gpuMemoryGb: 40,
        cpuCores: 12,
        memoryGb: 85,
        storageGb: 200,
      },
    });
    console.log("[ComputeExecutor] Created GCP GPU node");
  }
}
