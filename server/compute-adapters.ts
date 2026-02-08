import type { ComputeNode, ComputeProvider, ConnectionType } from "@shared/schema";
import { Client as SshClient } from "ssh2";
import { storage } from "./storage";
import sshpk from "sshpk";

export interface ComputeJob {
  id: string;
  type: string;
  command: string;
  params?: Record<string, unknown>;
  workingDir?: string;
  environment?: Record<string, string>;
  timeout?: number;
}

export interface ComputeJobResult {
  success: boolean;
  output?: string;
  error?: string;
  exitCode?: number;
  gpuUsed?: boolean;
  cpuTimeSeconds?: number;
  gpuTimeSeconds?: number;
}

export interface ComputeAdapter {
  provider: ComputeProvider;
  connectionType: ConnectionType;
  
  runJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult>;
  checkHealth(node: ComputeNode): Promise<boolean>;
  getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded">;
  uploadFile?(node: ComputeNode, localPath: string, remotePath: string): Promise<boolean>;
  downloadFile?(node: ComputeNode, remotePath: string, localPath: string): Promise<boolean>;
}

function formatPrivateKey(key: string | undefined): string | null {
  if (!key) return null;
  
  let formattedKey = key.trim();
  
  // Replace escaped newlines with actual newlines first
  formattedKey = formattedKey.replace(/\\n/g, '\n');
  
  // Fix common issue: key pasted as single line with spaces instead of newlines
  // Pattern: -----BEGIN...KEY----- <base64> -----END...KEY-----
  if (formattedKey.includes('-----BEGIN') && formattedKey.includes('-----END')) {
    // Check if it's all on one line (no real newlines except possibly at ends)
    const lines = formattedKey.split('\n').filter(l => l.trim());
    if (lines.length === 1) {
      // Single line key - need to reformat
      const beginMatch = formattedKey.match(/(-----BEGIN [A-Z ]+ KEY-----)/);
      const endMatch = formattedKey.match(/(-----END [A-Z ]+ KEY-----)/);
      if (beginMatch && endMatch) {
        const beginHeader = beginMatch[1];
        const endHeader = endMatch[1];
        // Extract the base64 content between headers
        const startIdx = formattedKey.indexOf(beginHeader) + beginHeader.length;
        const endIdx = formattedKey.indexOf(endHeader);
        let base64Content = formattedKey.substring(startIdx, endIdx).trim();
        // Remove any spaces that might be in the base64
        base64Content = base64Content.replace(/\s+/g, '');
        // Chunk the base64 content into 70-char lines
        const chunkedKey = base64Content.match(/.{1,70}/g)?.join('\n') || base64Content;
        formattedKey = `${beginHeader}\n${chunkedKey}\n${endHeader}\n`;
        console.log(`[SSH] Reformatted single-line PEM key with proper line breaks`);
      }
    }
  }
  // Check if key is base64 encoded (doesn't start with -----)
  else if (!formattedKey.startsWith('-----')) {
    try {
      // Try to decode from base64
      const keyBuffer = Buffer.from(formattedKey, 'base64');
      const decoded = keyBuffer.toString('utf8');
      
      // Check if the decoded content looks like text (OpenSSH key header)
      if (decoded.includes('openssh-key-v1')) {
        // It's raw binary OpenSSH key, wrap it with proper PEM headers
        const base64Key = keyBuffer.toString('base64');
        const chunkedKey = base64Key.match(/.{1,70}/g)?.join('\n') || base64Key;
        formattedKey = `-----BEGIN OPENSSH PRIVATE KEY-----\n${chunkedKey}\n-----END OPENSSH PRIVATE KEY-----\n`;
        console.log(`[SSH] Wrapped raw OpenSSH key with PEM headers`);
      } else if (decoded.startsWith('-----')) {
        // It was base64-encoded PEM text
        formattedKey = decoded;
        console.log(`[SSH] Decoded base64 PEM key`);
      } else {
        // Unknown format, might be raw binary - try wrapping as OpenSSH
        const base64Key = keyBuffer.toString('base64');
        const chunkedKey = base64Key.match(/.{1,70}/g)?.join('\n') || base64Key;
        formattedKey = `-----BEGIN OPENSSH PRIVATE KEY-----\n${chunkedKey}\n-----END OPENSSH PRIVATE KEY-----\n`;
        console.log(`[SSH] Wrapped binary key with PEM headers`);
      }
    } catch (e) {
      console.log(`[SSH] Key is not base64 encoded, using as-is`);
    }
  }
  
  // Ensure key ends with a newline
  if (!formattedKey.endsWith('\n')) {
    formattedKey += '\n';
  }
  
  // Log first 50 chars but show actual newlines
  const logPreview = formattedKey.substring(0, 80).replace(/\n/g, '\\n');
  console.log(`[SSH] Final key format: ${logPreview}...`);
  console.log(`[SSH] Key length: ${formattedKey.length} chars, lines: ${formattedKey.split('\n').length}`);
  
  return formattedKey;
}

async function getSshPrivateKey(node: ComputeNode): Promise<string | null> {
  if (!node.sshConfigId) {
    return formatPrivateKey(process.env.SSH_PRIVATE_KEY);
  }
  
  const sshConfig = await storage.getSshConfig(node.sshConfigId);
  if (sshConfig) {
    const secretKey = `SSH_KEY_${sshConfig.id}`;
    return formatPrivateKey(process.env[secretKey]) || formatPrivateKey(process.env.SSH_PRIVATE_KEY);
  }
  
  return formatPrivateKey(process.env.SSH_PRIVATE_KEY);
}

export class SshComputeAdapter implements ComputeAdapter {
  provider: ComputeProvider;
  connectionType: ConnectionType = "ssh";

  constructor(provider: ComputeProvider = "hetzner") {
    this.provider = provider;
  }

  private async createConnection(node: ComputeNode): Promise<SshClient> {
    return new Promise(async (resolve, reject) => {
      const conn = new SshClient();
      const host = node.sshHost || node.ipAddress;
      const port = parseInt(node.sshPort || "22", 10);
      const username = node.sshUsername || "root";
      
      const privateKey = await getSshPrivateKey(node);
      
      if (!host) {
        reject(new Error(`No SSH host configured for node ${node.name}`));
        return;
      }

      const config: any = {
        host,
        port,
        username,
        readyTimeout: 30000,
        keepaliveInterval: 10000,
        debug: (msg: string) => {
          if (msg.includes("error") || msg.includes("fail") || msg.includes("auth")) {
            console.log(`[SSH-DEBUG] ${msg}`);
          }
        },
        algorithms: {
          serverHostKey: ['ssh-ed25519', 'ecdsa-sha2-nistp256', 'ecdsa-sha2-nistp384', 'ecdsa-sha2-nistp521', 'rsa-sha2-512', 'rsa-sha2-256', 'ssh-rsa'],
        },
        tryKeyboard: true,
      };

      const nodePassword = node.provider === "hetzner" 
        ? process.env.HETZNER_SSH_PASSWORD 
        : process.env.SSH_PASSWORD;
      
      if (nodePassword) {
        config.password = nodePassword;
        console.log(`[SSH] Using password authentication for ${node.name}`);
      } else if (privateKey) {
        try {
          config.privateKey = privateKey;
          // Check if passphrase is available for encrypted keys
          const passphrase = process.env.SSH_KEY_PASSPHRASE;
          if (passphrase) {
            config.passphrase = passphrase;
            console.log(`[SSH] Using private key with passphrase for ${node.name}`);
          } else {
            console.log(`[SSH] Using private key authentication for ${node.name}`);
          }
        } catch (keyErr: any) {
          console.error(`[SSH] Private key parse error:`, keyErr.message);
          reject(new Error(`Cannot parse SSH private key for node ${node.name}`));
          return;
        }
      } else {
        reject(new Error(`No SSH credentials configured for node ${node.name}. Set HETZNER_SSH_PASSWORD or SSH_PRIVATE_KEY.`));
        return;
      }

      conn.on("ready", () => {
        console.log(`[SSH] Connected to ${host}:${port} as ${username}`);
        resolve(conn);
      });

      conn.on("error", (err) => {
        console.error(`[SSH] Connection error to ${host}:`, err.message);
        if (err.message.includes("handshake") || err.message.includes("parse")) {
          console.error(`[SSH] This may be due to an encrypted Ed25519 key. Consider using an unencrypted key or RSA format.`);
        }
        reject(err);
      });
      
      conn.on("keyboard-interactive", (name: string, instructions: string, _: string, prompts: any[], finish: (responses: string[]) => void) => {
        console.log(`[SSH] Keyboard-interactive auth requested: ${prompts.length} prompts`);
        const responses = prompts.map(() => process.env.SSH_KEY_PASSPHRASE || "");
        finish(responses);
      });

      try {
        conn.connect(config);
      } catch (connectErr: any) {
        console.error(`[SSH] Connection failed:`, connectErr.message);
        if (connectErr.message.includes('privateKey')) {
          console.error(`[SSH] Private key format issue. Key starts with: ${privateKey?.substring(0, 40)}...`);
        }
        reject(connectErr);
      }
    });
  }

  async runJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    const startTime = Date.now();
    console.log(`[SshComputeAdapter] Running job ${job.id} on node ${node.name}`);
    console.log(`  Host: ${node.sshHost || node.ipAddress}:${node.sshPort}`);
    console.log(`  Command: ${job.command}`);

    try {
      const conn = await this.createConnection(node);
      
      return new Promise((resolve) => {
        let command = job.command;
        
        if (job.workingDir) {
          command = `cd ${job.workingDir} && ${command}`;
        }
        
        if (job.environment) {
          const envVars = Object.entries(job.environment)
            .map(([k, v]) => `export ${k}="${v}"`)
            .join(" && ");
          command = `${envVars} && ${command}`;
        }

        let stdout = "";
        let stderr = "";
        let timedOut = false;
        
        const timeout = job.timeout || 3600000;
        const timeoutHandle = setTimeout(() => {
          timedOut = true;
          conn.end();
        }, timeout);

        conn.exec(command, (err, stream) => {
          if (err) {
            clearTimeout(timeoutHandle);
            conn.end();
            resolve({
              success: false,
              error: err.message,
              exitCode: -1,
            });
            return;
          }

          stream.on("close", (code: number) => {
            clearTimeout(timeoutHandle);
            const elapsed = (Date.now() - startTime) / 1000;
            conn.end();
            
            if (timedOut) {
              resolve({
                success: false,
                error: `Job timed out after ${timeout}ms`,
                exitCode: -1,
                cpuTimeSeconds: elapsed,
              });
              return;
            }

            const gpuUsed = node.gpuType !== "none";
            
            resolve({
              success: code === 0,
              output: stdout,
              error: stderr || undefined,
              exitCode: code,
              gpuUsed,
              cpuTimeSeconds: gpuUsed ? 0 : elapsed,
              gpuTimeSeconds: gpuUsed ? elapsed : 0,
            });
          });

          stream.on("data", (data: Buffer) => {
            stdout += data.toString();
          });

          stream.stderr.on("data", (data: Buffer) => {
            stderr += data.toString();
          });
        });
      });
    } catch (err: any) {
      return {
        success: false,
        error: err.message,
        exitCode: -1,
      };
    }
  }

  async checkHealth(node: ComputeNode): Promise<boolean> {
    try {
      const conn = await this.createConnection(node);
      
      return new Promise((resolve) => {
        conn.exec("echo ok && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no-gpu'", (err, stream) => {
          if (err) {
            conn.end();
            resolve(false);
            return;
          }

          let output = "";
          stream.on("close", () => {
            conn.end();
            console.log(`[SSH] Health check for ${node.name}: ${output.trim()}`);
            resolve(output.includes("ok"));
          });

          stream.on("data", (data: Buffer) => {
            output += data.toString();
          });
        });
      });
    } catch {
      return false;
    }
  }

  async getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded"> {
    const isHealthy = await this.checkHealth(node);
    return isHealthy ? "active" : "offline";
  }

  async uploadFile(node: ComputeNode, localPath: string, remotePath: string): Promise<boolean> {
    try {
      const conn = await this.createConnection(node);
      const fs = await import("fs");
      const path = await import("path");
      
      const remoteDir = path.dirname(remotePath);
      
      await new Promise<void>((resolve, reject) => {
        conn.exec(`mkdir -p ${remoteDir}`, (err, stream) => {
          if (err) {
            reject(err);
            return;
          }
          stream.on("close", () => resolve());
          stream.on("error", reject);
          stream.on("data", () => {});
          stream.stderr.on("data", () => {});
        });
      });
      console.log(`[SSH] Created remote directory: ${remoteDir}`);
      
      return new Promise((resolve) => {
        conn.sftp((err, sftp) => {
          if (err) {
            conn.end();
            console.error(`[SSH] SFTP error:`, err.message);
            resolve(false);
            return;
          }

          const readStream = fs.createReadStream(localPath);
          const writeStream = sftp.createWriteStream(remotePath);

          writeStream.on("close", () => {
            console.log(`[SSH] Uploaded ${localPath} to ${remotePath}`);
            conn.end();
            resolve(true);
          });

          writeStream.on("error", (uploadErr: any) => {
            console.error(`[SSH] Upload error:`, uploadErr.message);
            conn.end();
            resolve(false);
          });

          readStream.pipe(writeStream);
        });
      });
    } catch (err: any) {
      console.error(`[SSH] uploadFile error:`, err.message);
      return false;
    }
  }

  async downloadFile(node: ComputeNode, remotePath: string, localPath: string): Promise<boolean> {
    try {
      const conn = await this.createConnection(node);
      const fs = await import("fs");
      
      return new Promise((resolve) => {
        conn.sftp((err, sftp) => {
          if (err) {
            conn.end();
            resolve(false);
            return;
          }

          const readStream = sftp.createReadStream(remotePath);
          const writeStream = fs.createWriteStream(localPath);

          readStream.on("error", (readErr: any) => {
            console.error(`[SSH] Read stream error:`, readErr.message);
            try { writeStream.destroy(); } catch {}
            try { conn.end(); } catch {}
            resolve(false);
          });

          writeStream.on("close", () => {
            try { conn.end(); } catch {}
            resolve(true);
          });

          writeStream.on("error", (writeErr: any) => {
            console.error(`[SSH] Write stream error:`, writeErr.message);
            try { readStream.destroy(); } catch {}
            try { conn.end(); } catch {}
            resolve(false);
          });

          readStream.pipe(writeStream);
        });
      });
    } catch {
      return false;
    }
  }
}

export class VastAiComputeAdapter implements ComputeAdapter {
  provider: ComputeProvider = "vast";
  connectionType: ConnectionType = "cloud_api";
  private apiKey: string;
  private baseUrl = "https://console.vast.ai/api/v0";

  constructor() {
    this.apiKey = process.env.VAST_AI_API_KEY || "";
  }

  private async apiRequest(endpoint: string, method: string = "GET", body?: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`Vast.ai API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async runJob(node: ComputeNode, job: ComputeJob): Promise<ComputeJobResult> {
    const startTime = Date.now();
    console.log(`[VastAiAdapter] Running job ${job.id} on Vast.ai instance`);
    console.log(`  GPU Type: ${node.gpuType}`);
    console.log(`  Command: ${job.command}`);

    if (!this.apiKey) {
      return {
        success: false,
        error: "VAST_AI_API_KEY not configured",
      };
    }

    try {
      const instances = await this.apiRequest("/instances");
      
      let instanceId: string | null = null;
      let instanceInfo: any = null;
      
      if (node.specs && typeof node.specs === "object" && "vastInstanceId" in node.specs) {
        const storedId = (node.specs as any).vastInstanceId;
        const storedInstance = instances.instances?.find((i: any) => i.id === storedId);
        if (storedInstance?.actual_status === "running") {
          instanceId = storedId;
          instanceInfo = storedInstance;
          console.log(`[VastAiAdapter] Reusing stored instance ${instanceId}`);
        }
      }
      
      if (!instanceId) {
        console.log(`[VastAiAdapter] No stored instance ID found for node ${node.name}, will create new instance if needed`);
      }

      const sshPublicKey = process.env.SSH_PUBLIC_KEY || "";
      
      if (!instanceId) {
        console.log("[VastAiAdapter] No running instance found, creating one...");
        
        const offers = await this.apiRequest("/bundles?q=" + encodeURIComponent(JSON.stringify({
          gpu_name: { eq: "RTX 3090" },
          num_gpus: { gte: 2 },
          rentable: { eq: true },
          order: [["dph_total", "asc"]],
          limit: 5,
        })));

        if (!offers.offers?.length) {
          return {
            success: false,
            error: "No suitable Vast.ai instances available",
          };
        }

        const offer = offers.offers[0];
        
        const createPayload: any = {
          client_id: "api",
          image: "nvidia/cuda:12.1.0-devel-ubuntu22.04",
          disk: 50,
          onstart: "mkdir -p /opt/lika-compute && pip install rdkit dask distributed torch xgboost scikit-learn numpy pandas",
        };
        
        if (sshPublicKey) {
          createPayload.ssh_key = sshPublicKey;
        }
        
        const createResult = await this.apiRequest("/asks/" + offer.id, "PUT", createPayload);

        instanceId = createResult.new_contract;
        console.log(`[VastAiAdapter] Created instance ${instanceId}`);
        
        if (node.id) {
          try {
            const { storage } = await import("./storage");
            const currentSpecs = (node.specs as Record<string, unknown>) || {};
            await storage.updateComputeNode(node.id, {
              specs: { ...currentSpecs, vastInstanceId: instanceId },
            });
            console.log(`[VastAiAdapter] Persisted instance ID to node ${node.id}`);
          } catch (err) {
            console.error("[VastAiAdapter] Failed to persist instance ID:", err);
          }
        }
        
        console.log("[VastAiAdapter] Waiting for instance to start...");
        for (let i = 0; i < 12; i++) {
          await new Promise(resolve => setTimeout(resolve, 10000));
          try {
            instanceInfo = await this.apiRequest(`/instances/${instanceId}`);
            if (instanceInfo.actual_status === "running") {
              console.log("[VastAiAdapter] Instance is running");
              break;
            }
          } catch {
          }
        }
      }

      if (!instanceInfo) {
        instanceInfo = await this.apiRequest(`/instances/${instanceId}`);
      }
      
      // Use node's configured SSH settings if available, otherwise fall back to API info
      // Prefer ssh_host (proxy like ssh6.vast.ai) over public_ipaddr for proper SSH routing
      const sshHost = node.sshHost || instanceInfo.ssh_host || instanceInfo.public_ipaddr;
      const sshPort = node.sshPort || instanceInfo.ssh_port || instanceInfo.ports?.["22/tcp"]?.[0]?.HostPort || 22;

      if (!sshHost) {
        return {
          success: false,
          error: "Failed to get SSH connection info for Vast.ai instance",
        };
      }

      console.log(`[VastAiAdapter] Using SSH: ${sshHost}:${sshPort}`);
      
      const sshAdapter = new SshComputeAdapter("vast");
      const sshNode: ComputeNode = {
        ...node,
        sshHost,
        sshPort: String(sshPort),
        sshUsername: "root",
      };

      const result = await sshAdapter.runJob(sshNode, job);
      
      const elapsed = (Date.now() - startTime) / 1000;
      
      return {
        ...result,
        gpuUsed: true,
        gpuTimeSeconds: elapsed,
      };
    } catch (err: any) {
      return {
        success: false,
        error: err.message,
      };
    }
  }

  async checkHealth(node: ComputeNode): Promise<boolean> {
    if (!this.apiKey) {
      return false;
    }

    try {
      const instances = await this.apiRequest("/instances");
      
      if (node.specs && typeof node.specs === "object" && "vastInstanceId" in node.specs) {
        const instanceId = (node.specs as any).vastInstanceId;
        const instance = instances.instances?.find((i: any) => i.id === instanceId);
        return instance?.actual_status === "running";
      }
      
      return instances.instances?.some((i: any) => i.actual_status === "running") || false;
    } catch {
      return false;
    }
  }

  async getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded"> {
    const isHealthy = await this.checkHealth(node);
    return isHealthy ? "active" : "offline";
  }

  async listAvailableGpus(): Promise<any[]> {
    if (!this.apiKey) {
      return [];
    }

    try {
      const offers = await this.apiRequest("/bundles?q=" + encodeURIComponent(JSON.stringify({
        rentable: { eq: true },
        order: [["dph_total", "asc"]],
        limit: 50,
      })));

      return offers.offers || [];
    } catch {
      return [];
    }
  }

  private async getSshNode(node: ComputeNode): Promise<ComputeNode> {
    const sshHost = node.sshHost;
    const sshPort = node.sshPort || "22";
    
    if (sshHost) {
      return {
        ...node,
        sshHost,
        sshPort: String(sshPort),
        sshUsername: node.sshUsername || "root",
      };
    }
    
    // Get SSH info from Vast.ai API if not configured
    const instances = await this.apiRequest("/instances");
    const instanceId = (node.specs as any)?.vastInstanceId;
    const instanceInfo = instances.instances?.find((i: any) => i.id === instanceId);
    
    if (instanceInfo) {
      return {
        ...node,
        sshHost: instanceInfo.ssh_host || instanceInfo.public_ipaddr,
        sshPort: String(instanceInfo.ssh_port || "22"),
        sshUsername: "root",
      };
    }
    
    return node;
  }

  async uploadFile(node: ComputeNode, localPath: string, remotePath: string): Promise<boolean> {
    try {
      const sshNode = await this.getSshNode(node);
      const sshAdapter = new SshComputeAdapter("vast");
      return sshAdapter.uploadFile(sshNode, localPath, remotePath);
    } catch (err: any) {
      console.error(`[VastAiAdapter] uploadFile error:`, err.message);
      return false;
    }
  }

  async downloadFile(node: ComputeNode, remotePath: string, localPath: string): Promise<boolean> {
    try {
      const sshNode = await this.getSshNode(node);
      const sshAdapter = new SshComputeAdapter("vast");
      return sshAdapter.downloadFile(sshNode, remotePath, localPath);
    } catch (err: any) {
      console.error(`[VastAiAdapter] downloadFile error:`, err.message);
      return false;
    }
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

    switch (this.provider) {
      case "vast":
        const vastAdapter = new VastAiComputeAdapter();
        return vastAdapter.runJob(node, job);
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
    if (node.sshHost) {
      console.log(`[GCP] Routing job ${job.id} via SSH to ${node.sshHost}:${node.sshPort || 22}`);
      const sshAdapter = new SshComputeAdapter("gcp");
      return sshAdapter.runJob(node, job);
    }
    console.log(`[GCP] No SSH host configured for node ${node.name}, cannot execute`);
    return {
      success: false,
      error: `GCP node ${node.name} has no SSH host configured`,
    };
  }

  async checkHealth(node: ComputeNode): Promise<boolean> {
    if (this.provider === "vast") {
      const vastAdapter = new VastAiComputeAdapter();
      return vastAdapter.checkHealth(node);
    }
    if (this.provider === "gcp" && node.sshHost) {
      const sshAdapter = new SshComputeAdapter("gcp");
      return sshAdapter.checkHealth(node);
    }
    console.log(`[CloudApiComputeAdapter] Checking health of ${this.provider} node ${node.name}`);
    return true;
  }

  async getStatus(node: ComputeNode): Promise<"active" | "offline" | "degraded"> {
    const isHealthy = await this.checkHealth(node);
    return isHealthy ? "active" : "offline";
  }
}

export function getComputeAdapter(node: ComputeNode): ComputeAdapter {
  if (node.provider === "vast") {
    return new VastAiComputeAdapter();
  }
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

export async function selectNodeByCapability(
  nodes: ComputeNode[],
  requireGpu: boolean,
  preferredProvider?: ComputeProvider
): Promise<ComputeNode | null> {
  const activeNodes = nodes.filter(n => n.status === "active");
  
  if (requireGpu) {
    const gpuNodes = activeNodes.filter(n => n.gpuType !== "none");
    
    if (preferredProvider) {
      const preferred = gpuNodes.find(n => n.provider === preferredProvider);
      if (preferred) return preferred;
    }
    
    const vastNode = gpuNodes.find(n => n.provider === "vast");
    if (vastNode) return vastNode;
    
    return gpuNodes[0] || null;
  }
  
  const cpuNodes = activeNodes.filter(n => n.gpuType === "none" || n.tier?.includes("shared"));
  
  if (preferredProvider) {
    const preferred = cpuNodes.find(n => n.provider === preferredProvider);
    if (preferred) return preferred;
  }
  
  const hetznerNode = cpuNodes.find(n => n.provider === "hetzner");
  if (hetznerNode) return hetznerNode;
  
  return cpuNodes[0] || activeNodes[0] || null;
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

function toSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
}

function normalizeParamsToSnakeCase(params: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    const snakeKey = toSnakeCase(key);
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      normalized[snakeKey] = normalizeParamsToSnakeCase(value as Record<string, unknown>);
    } else {
      normalized[snakeKey] = value;
    }
  }
  return normalized;
}

export function buildPipelineCommand(
  pipelineScript: string,
  jobType: string,
  params: Record<string, unknown>,
  useParamsFile?: string
): string {
  const normalizedParams = normalizeParamsToSnakeCase(params);
  
  if (useParamsFile) {
    return `mkdir -p /opt/lika-compute && cd /opt/lika-compute && python3 ${pipelineScript} --job-type ${jobType} --params-file ${useParamsFile}`;
  }
  
  const paramsJson = JSON.stringify(normalizedParams).replace(/'/g, "\\'").replace(/"/g, '\\"');
  const paramSize = paramsJson.length;
  
  if (paramSize > 32000) {
    console.log(`[buildPipelineCommand] Params too large (${paramSize} chars), consider using --params-file`);
  }
  
  return `mkdir -p /opt/lika-compute && cd /opt/lika-compute && python3 ${pipelineScript} --job-type ${jobType} --params '${paramsJson}'`;
}

export function getCpuVsGpuSteps(jobType: string): { cpuSteps: string[], gpuSteps: string[] } {
  const stepMapping: Record<string, { cpuSteps: string[], gpuSteps: string[] }> = {
    generation: {
      cpuSteps: ["smiles_validation", "fingerprint_generation"],
      gpuSteps: [],
    },
    filtering: {
      cpuSteps: ["property_calculation", "rule_filtering"],
      gpuSteps: [],
    },
    docking: {
      cpuSteps: ["ligand_preparation", "receptor_preparation"],
      gpuSteps: ["vina_docking"],
    },
    scoring: {
      cpuSteps: ["descriptor_calculation"],
      gpuSteps: ["ml_prediction", "neural_network_inference"],
    },
    quantum_optimization: {
      cpuSteps: ["input_preparation"],
      gpuSteps: ["quantum_circuit_simulation", "optimization"],
    },
    ml_training: {
      cpuSteps: ["data_preprocessing", "feature_engineering"],
      gpuSteps: ["model_training", "hyperparameter_tuning"],
    },
  };

  return stepMapping[jobType] || { cpuSteps: [jobType], gpuSteps: [] };
}
