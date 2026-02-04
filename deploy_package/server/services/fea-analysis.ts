import OpenAI from "openai";

// Use Replit AI Integrations (no API key needed)
const openai = new OpenAI({
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY || "dummy-key",
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

export interface FEASimulationResult {
  jobId: number;
  simulationType: "structural" | "thermal" | "cfd";
  name: string;
  status: string;
  material: {
    name: string;
    youngsModulus?: number;
    density?: number;
    thermalConductivity?: number;
    poissonsRatio?: number;
    yieldStrength?: number;
    specificHeat?: number;
  };
  meshQuality: string;
  results?: {
    maxStress?: number;
    maxDeformation?: number;
    safetyFactor?: number;
    maxTemperature?: number;
    minTemperature?: number;
    heatFlux?: number;
    maxVelocity?: number;
    pressureDrop?: number;
    reynoldsNumber?: number;
  };
  isAssembly?: boolean;
  components?: Array<{
    name: string;
    material: string;
  }>;
}

export interface FEAAnalysisResult {
  summary: string;
  keyFindings: string[];
  safetyAssessment: {
    status: "safe" | "marginal" | "critical";
    explanation: string;
  };
  materialPerformance: {
    rating: "excellent" | "good" | "adequate" | "poor";
    strengths: string[];
    limitations: string[];
  };
  recommendations: string[];
  optimizationSuggestions: string[];
  alternativeMaterials?: string[];
  designModifications?: string[];
  confidenceLevel: "high" | "medium" | "low";
  technicalNotes: string;
}

export interface BioNeMoAnalysisResult {
  molecularInsights?: string;
  structuralRecommendations?: string[];
  biocompatibilityNotes?: string;
  degradationPredictions?: string;
}

const FEA_ANALYSIS_SYSTEM_PROMPT = `You are an expert finite element analysis (FEA) engineer and materials scientist. Your role is to analyze FEA simulation results and provide actionable engineering insights.

When analyzing simulation results, consider:
1. Safety factors and failure modes
2. Material behavior under the given conditions
3. Design optimization opportunities
4. Alternative materials that could improve performance
5. Real-world manufacturing and cost considerations

Provide your analysis in a structured JSON format with the following fields:
- summary: A 2-3 sentence executive summary of the analysis
- keyFindings: Array of 3-5 key findings from the simulation
- safetyAssessment: Object with status ("safe", "marginal", "critical") and explanation
- materialPerformance: Object with rating ("excellent", "good", "adequate", "poor"), strengths array, and limitations array
- recommendations: Array of 3-5 actionable recommendations
- optimizationSuggestions: Array of 2-3 optimization suggestions
- alternativeMaterials: Array of 2-3 alternative materials to consider
- designModifications: Array of 2-3 potential design modifications
- confidenceLevel: "high", "medium", or "low" based on data quality
- technicalNotes: Any additional technical notes or caveats

Be specific with numbers and values when available. Reference industry standards like ASME, ISO, or ASTM when relevant.`;

export async function analyzeFeaResults(
  simulation: FEASimulationResult
): Promise<FEAAnalysisResult> {
  const simulationContext = buildSimulationContext(simulation);

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: FEA_ANALYSIS_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Analyze the following FEA simulation results:\n\n${simulationContext}\n\nProvide a comprehensive engineering analysis in JSON format.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 2000,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response from AI model");
    }

    return JSON.parse(content) as FEAAnalysisResult;
  } catch (error) {
    console.error("FEA Analysis error:", error);
    throw new Error("Failed to analyze FEA results");
  }
}

export async function compareFeaMaterials(
  baselineSimulation: FEASimulationResult,
  alternativeSimulation: FEASimulationResult
): Promise<{
  comparison: string;
  winner: string;
  tradeoffs: string[];
  recommendation: string;
}> {
  const baseContext = buildSimulationContext(baselineSimulation);
  const altContext = buildSimulationContext(alternativeSimulation);

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are an expert materials engineer comparing FEA simulation results for different materials. Provide a comprehensive comparison with clear recommendations. Return JSON with: comparison (summary), winner (material name), tradeoffs (array of key tradeoffs), recommendation (final recommendation).`,
        },
        {
          role: "user",
          content: `Compare these two FEA simulations:\n\nBASELINE MATERIAL:\n${baseContext}\n\nALTERNATIVE MATERIAL:\n${altContext}\n\nProvide a detailed comparison in JSON format.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response from AI model");
    }

    return JSON.parse(content);
  } catch (error) {
    console.error("Material comparison error:", error);
    throw new Error("Failed to compare materials");
  }
}

export async function analyzeAssemblyInteractions(
  simulation: FEASimulationResult
): Promise<{
  interfaceAnalysis: string;
  stressConcentrations: string[];
  thermalMismatch: string;
  recommendations: string[];
}> {
  if (!simulation.isAssembly || !simulation.components) {
    throw new Error("Not an assembly simulation");
  }

  const context = buildSimulationContext(simulation);

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are an expert in multi-material assembly analysis. Analyze the interactions between different materials in an assembly, focusing on interface stresses, thermal expansion mismatches, and potential failure points. Return JSON with: interfaceAnalysis (summary), stressConcentrations (array of areas of concern), thermalMismatch (analysis of thermal expansion differences), recommendations (array of recommendations).`,
        },
        {
          role: "user",
          content: `Analyze this multi-component assembly simulation:\n\n${context}\n\nProvide interface and interaction analysis in JSON format.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response from AI model");
    }

    return JSON.parse(content);
  } catch (error) {
    console.error("Assembly analysis error:", error);
    throw new Error("Failed to analyze assembly interactions");
  }
}

export async function getBioNeMoInsights(
  simulation: FEASimulationResult,
  context: "biomedical" | "pharmaceutical" | "general"
): Promise<BioNeMoAnalysisResult> {
  // BioNeMo integration for biomedical/pharmaceutical applications
  const systemPrompt =
    context === "biomedical"
      ? `You are an expert in biomaterials and medical device engineering, with knowledge of BioNeMo molecular analysis capabilities. Analyze FEA results for biomedical applications, considering biocompatibility, sterilization effects, fatigue life in physiological conditions, and regulatory requirements (FDA, ISO 10993). Return JSON with: molecularInsights, structuralRecommendations (array), biocompatibilityNotes, degradationPredictions.`
      : context === "pharmaceutical"
        ? `You are an expert in pharmaceutical engineering and drug delivery systems. Analyze FEA results for pharmaceutical applications like tablet presses, packaging, or delivery devices. Consider material interactions with APIs, temperature sensitivity, and regulatory compliance (FDA, EMA). Return JSON with: molecularInsights, structuralRecommendations (array), biocompatibilityNotes, degradationPredictions.`
        : `You are an expert materials scientist with molecular modeling expertise. Analyze FEA results and provide insights on molecular-level behavior, polymer chain dynamics, or crystalline structure effects. Return JSON with: molecularInsights, structuralRecommendations (array), biocompatibilityNotes (if applicable), degradationPredictions.`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content: `Analyze this simulation for ${context} applications:\n\n${buildSimulationContext(simulation)}\n\nProvide molecular and structural insights in JSON format.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response from AI model");
    }

    return JSON.parse(content) as BioNeMoAnalysisResult;
  } catch (error) {
    console.error("BioNeMo analysis error:", error);
    throw new Error("Failed to get BioNeMo insights");
  }
}

function buildSimulationContext(simulation: FEASimulationResult): string {
  const lines: string[] = [
    `Simulation Name: ${simulation.name}`,
    `Simulation Type: ${simulation.simulationType.toUpperCase()}`,
    `Status: ${simulation.status}`,
    `Mesh Quality: ${simulation.meshQuality}`,
    "",
    "MATERIAL PROPERTIES:",
    `  Name: ${simulation.material.name}`,
  ];

  if (simulation.material.youngsModulus) {
    lines.push(`  Young's Modulus: ${simulation.material.youngsModulus} GPa`);
  }
  if (simulation.material.density) {
    lines.push(`  Density: ${simulation.material.density} kg/m³`);
  }
  if (simulation.material.thermalConductivity) {
    lines.push(
      `  Thermal Conductivity: ${simulation.material.thermalConductivity} W/m·K`
    );
  }
  if (simulation.material.poissonsRatio) {
    lines.push(`  Poisson's Ratio: ${simulation.material.poissonsRatio}`);
  }
  if (simulation.material.yieldStrength) {
    lines.push(`  Yield Strength: ${simulation.material.yieldStrength} MPa`);
  }
  if (simulation.material.specificHeat) {
    lines.push(`  Specific Heat: ${simulation.material.specificHeat} J/kg·K`);
  }

  if (simulation.results) {
    lines.push("", "SIMULATION RESULTS:");
    const r = simulation.results;
    if (r.maxStress !== undefined)
      lines.push(`  Maximum Stress: ${r.maxStress} MPa`);
    if (r.maxDeformation !== undefined)
      lines.push(`  Maximum Deformation: ${r.maxDeformation} mm`);
    if (r.safetyFactor !== undefined)
      lines.push(`  Safety Factor: ${r.safetyFactor}`);
    if (r.maxTemperature !== undefined)
      lines.push(`  Maximum Temperature: ${r.maxTemperature} °C`);
    if (r.minTemperature !== undefined)
      lines.push(`  Minimum Temperature: ${r.minTemperature} °C`);
    if (r.heatFlux !== undefined)
      lines.push(`  Heat Flux: ${r.heatFlux} W/m²`);
    if (r.maxVelocity !== undefined)
      lines.push(`  Maximum Velocity: ${r.maxVelocity} m/s`);
    if (r.pressureDrop !== undefined)
      lines.push(`  Pressure Drop: ${r.pressureDrop} Pa`);
    if (r.reynoldsNumber !== undefined)
      lines.push(`  Reynolds Number: ${r.reynoldsNumber}`);
  }

  if (simulation.isAssembly && simulation.components) {
    lines.push("", "ASSEMBLY COMPONENTS:");
    simulation.components.forEach((c, i) => {
      lines.push(`  ${i + 1}. ${c.name} - Material: ${c.material}`);
    });
  }

  return lines.join("\n");
}

export function isAIConfigured(): boolean {
  return !!(
    process.env.AI_INTEGRATIONS_OPENAI_API_KEY &&
    process.env.AI_INTEGRATIONS_OPENAI_BASE_URL
  );
}
