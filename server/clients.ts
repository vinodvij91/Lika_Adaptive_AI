export interface GeneratedMolecule {
  smiles: string;
}

export interface DockingResult {
  moleculeId: string;
  score: number;
  pose?: string;
}

export interface AdmetPrediction {
  moleculeId: string;
  score: number;
  properties: Record<string, number>;
}

export interface QsarPrediction {
  moleculeId: string;
  score: number;
  activityPrediction: number;
}

const randomFloat = (min: number, max: number) => Math.random() * (max - min) + min;

const sampleSmiles = [
  "CC(=O)Oc1ccccc1C(=O)O",
  "CCN(CC)c1ccc2c(c1)c(C)c1cc(C)ccc1c2",
  "CN1CCC[C@H]1c2cccnc2",
  "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O",
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
  "c1ccc2c(c1)nc(o2)C3=Nc4ccccc4O3",
  "COc1ccc2cc(ccc2c1)[C@H](C)C(=O)O",
  "CC(C)NCC(O)c1ccc(O)c(O)c1",
  "CN1CCC[C@@H]1c2ccccc2O",
  "COc1cc2nc(nc(N)c2cc1OC)N3CCN(CC3)C(=O)c4occc4",
  "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  "CC1=C(C(=O)Nc2ccccc2)c3ccccc3N1C",
  "c1ccc(cc1)c2ccc(s2)c3cccs3",
  "CC(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
];

export class BioNemoClient {
  async generateMolecules(seedSmiles?: string[], n = 50): Promise<GeneratedMolecule[]> {
    await this.simulateDelay(1000, 3000);
    
    const generated: GeneratedMolecule[] = [];
    const count = Math.min(n, 100);
    
    for (let i = 0; i < count; i++) {
      const baseSmi = sampleSmiles[Math.floor(Math.random() * sampleSmiles.length)];
      const variation = Math.random() > 0.5 ? "C" : "O";
      const smiles = baseSmi.replace(/c1/g, `c1${variation}`).slice(0, 50 + Math.floor(Math.random() * 30));
      generated.push({ smiles });
    }
    
    return generated;
  }

  async predictDocking(moleculeIds: string[], targetId: string): Promise<DockingResult[]> {
    await this.simulateDelay(500, 2000);
    
    return moleculeIds.map((moleculeId) => ({
      moleculeId,
      score: randomFloat(0.1, 0.95),
      pose: "mock_pose_data",
    }));
  }

  private simulateDelay(min: number, max: number): Promise<void> {
    const delay = Math.floor(Math.random() * (max - min) + min);
    return new Promise((resolve) => setTimeout(resolve, delay));
  }
}

export class MolecularMLClient {
  async predictAdmet(moleculeIds: string[]): Promise<AdmetPrediction[]> {
    await this.simulateDelay(300, 1500);
    
    return moleculeIds.map((moleculeId) => ({
      moleculeId,
      score: randomFloat(0.2, 0.9),
      properties: {
        logP: randomFloat(-2, 5),
        mw: randomFloat(150, 600),
        tpsa: randomFloat(20, 150),
        hba: Math.floor(randomFloat(0, 12)),
        hbd: Math.floor(randomFloat(0, 6)),
        rotatable_bonds: Math.floor(randomFloat(0, 10)),
      },
    }));
  }

  async predictQsar(moleculeIds: string[], targetId: string): Promise<QsarPrediction[]> {
    await this.simulateDelay(400, 1800);
    
    return moleculeIds.map((moleculeId) => ({
      moleculeId,
      score: randomFloat(0.15, 0.85),
      activityPrediction: randomFloat(4, 9),
    }));
  }

  private simulateDelay(min: number, max: number): Promise<void> {
    const delay = Math.floor(Math.random() * (max - min) + min);
    return new Promise((resolve) => setTimeout(resolve, delay));
  }
}

export class DockingClient {
  async dock(moleculeIds: string[], targetId: string): Promise<DockingResult[]> {
    await this.simulateDelay(600, 2500);
    
    return moleculeIds.map((moleculeId) => ({
      moleculeId,
      score: randomFloat(0.2, 0.9),
    }));
  }

  private simulateDelay(min: number, max: number): Promise<void> {
    const delay = Math.floor(Math.random() * (max - min) + min);
    return new Promise((resolve) => setTimeout(resolve, delay));
  }
}

export interface QuantumOptimizationParams {
  campaignId: string;
  moleculeIds: string[];
  objective: string;
  constraints?: Record<string, unknown>;
}

export interface QuantumJobResult {
  selectedMoleculeIds: string[];
  metadata?: Record<string, unknown>;
}

export class QuantumClient {
  private baseUrl: string;
  private apiKey: string;

  constructor() {
    this.baseUrl = process.env.QUANTUM_API_BASE_URL || "http://localhost:8000";
    this.apiKey = process.env.QUANTUM_API_KEY || "";
  }

  async submitOptimizationJob(params: QuantumOptimizationParams): Promise<{ quantumJobId: string }> {
    await this.simulateDelay(200, 500);
    
    const quantumJobId = `qjob_${Date.now()}_${Math.random().toString(36).substring(7)}`;
    console.log(`[QuantumClient] Submitted optimization job ${quantumJobId} for campaign ${params.campaignId}`);
    
    return { quantumJobId };
  }

  async getJobStatus(params: { quantumJobId: string }): Promise<{ status: "queued" | "running" | "completed" | "failed" }> {
    await this.simulateDelay(50, 150);
    
    return { status: "completed" };
  }

  async getJobResult(params: { quantumJobId: string; moleculeIds: string[] }): Promise<QuantumJobResult> {
    await this.simulateDelay(300, 800);
    
    const selectedCount = Math.max(1, Math.floor(params.moleculeIds.length * 0.6));
    const shuffled = [...params.moleculeIds].sort(() => Math.random() - 0.5);
    const selectedMoleculeIds = shuffled.slice(0, selectedCount);
    
    return {
      selectedMoleculeIds,
      metadata: {
        algorithm: "qaoa_mock",
        iterations: 100,
        circuitDepth: 3,
        optimizationValue: Math.random() * 0.5 + 0.5,
      },
    };
  }

  private simulateDelay(min: number, max: number): Promise<void> {
    const delay = Math.floor(Math.random() * (max - min) + min);
    return new Promise((resolve) => setTimeout(resolve, delay));
  }
}

export const bionemoClient = new BioNemoClient();
export const molecularMLClient = new MolecularMLClient();
export const dockingClient = new DockingClient();
export const quantumClient = new QuantumClient();
