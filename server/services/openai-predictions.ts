import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export interface MoleculePrediction {
  smiles: string;
  predictions: {
    drugLikeness: {
      score: number;
      lipinskiViolations: number;
      molecularWeight: string;
      logP: string;
      hbdCount: number;
      hbaCount: number;
      verdict: string;
    };
    admet: {
      absorption: { score: number; details: string };
      distribution: { score: number; details: string };
      metabolism: { score: number; details: string };
      excretion: { score: number; details: string };
      toxicity: { score: number; details: string; alerts: string[] };
    };
    targetPredictions: Array<{
      targetName: string;
      confidence: number;
      mechanism: string;
    }>;
    synthesizability: {
      score: number;
      complexity: string;
      estimatedSteps: number;
    };
    summary: string;
  };
  confidence: number;
  generatedAt: string;
}

export interface MaterialPrediction {
  representation: string;
  materialType: string;
  predictions: {
    mechanicalProperties: {
      tensileStrength: { value: number; unit: string; confidence: number };
      elasticModulus: { value: number; unit: string; confidence: number };
      hardness: { value: number; unit: string; confidence: number };
    };
    thermalProperties: {
      glassTempC: { value: number; confidence: number };
      meltingTempC: { value: number; confidence: number };
      thermalConductivity: { value: number; unit: string; confidence: number };
    };
    electricalProperties: {
      conductivity: { value: number; unit: string; confidence: number };
      dielectricConstant: { value: number; confidence: number };
    };
    stability: {
      chemicalResistance: string;
      uvResistance: string;
      oxidationResistance: string;
    };
    applications: string[];
    summary: string;
  };
  confidence: number;
  generatedAt: string;
}

const MOLECULE_PREDICTION_PROMPT = `You are an expert computational chemist and pharmacologist. Analyze the following SMILES structure and provide detailed predictions.

SMILES: {smiles}

Provide your analysis as a JSON object with this exact structure:
{
  "drugLikeness": {
    "score": <0-100>,
    "lipinskiViolations": <0-4>,
    "molecularWeight": "<estimated MW range>",
    "logP": "<estimated logP range>",
    "hbdCount": <hydrogen bond donors>,
    "hbaCount": <hydrogen bond acceptors>,
    "verdict": "<Good/Moderate/Poor drug-likeness>"
  },
  "admet": {
    "absorption": { "score": <0-100>, "details": "<brief assessment>" },
    "distribution": { "score": <0-100>, "details": "<brief assessment>" },
    "metabolism": { "score": <0-100>, "details": "<brief assessment>" },
    "excretion": { "score": <0-100>, "details": "<brief assessment>" },
    "toxicity": { "score": <0-100>, "details": "<brief assessment>", "alerts": ["<any structural alerts>"] }
  },
  "targetPredictions": [
    { "targetName": "<predicted target>", "confidence": <0-100>, "mechanism": "<mechanism of action>" }
  ],
  "synthesizability": {
    "score": <0-100>,
    "complexity": "<Low/Medium/High>",
    "estimatedSteps": <estimated synthetic steps>
  },
  "summary": "<2-3 sentence overall assessment>"
}

Be scientifically accurate and base predictions on structural features you can identify from the SMILES.`;

const MATERIAL_PREDICTION_PROMPT = `You are an expert materials scientist. Analyze the following material representation and provide detailed property predictions.

Material Type: {materialType}
Representation: {representation}

Provide your analysis as a JSON object with this exact structure:
{
  "mechanicalProperties": {
    "tensileStrength": { "value": <MPa>, "unit": "MPa", "confidence": <0-100> },
    "elasticModulus": { "value": <GPa>, "unit": "GPa", "confidence": <0-100> },
    "hardness": { "value": <Shore/Rockwell>, "unit": "Shore A or Rockwell", "confidence": <0-100> }
  },
  "thermalProperties": {
    "glassTempC": { "value": <Celsius>, "confidence": <0-100> },
    "meltingTempC": { "value": <Celsius>, "confidence": <0-100> },
    "thermalConductivity": { "value": <W/mK>, "unit": "W/m·K", "confidence": <0-100> }
  },
  "electricalProperties": {
    "conductivity": { "value": <S/m>, "unit": "S/m", "confidence": <0-100> },
    "dielectricConstant": { "value": <number>, "confidence": <0-100> }
  },
  "stability": {
    "chemicalResistance": "<Excellent/Good/Moderate/Poor>",
    "uvResistance": "<Excellent/Good/Moderate/Poor>",
    "oxidationResistance": "<Excellent/Good/Moderate/Poor>"
  },
  "applications": ["<suggested application 1>", "<suggested application 2>", "<suggested application 3>"],
  "summary": "<2-3 sentence overall assessment>"
}

Base predictions on the material type and structural representation provided.`;

export async function predictMoleculeProperties(smiles: string): Promise<MoleculePrediction> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured");
  }

  const prompt = MOLECULE_PREDICTION_PROMPT.replace("{smiles}", smiles);

  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "You are a computational chemistry expert. Always respond with valid JSON only, no markdown formatting.",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    temperature: 0.3,
    max_tokens: 2000,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    throw new Error("No response from OpenAI");
  }

  try {
    const predictions = JSON.parse(content.replace(/```json\n?|\n?```/g, "").trim());
    return {
      smiles,
      predictions,
      confidence: 75,
      generatedAt: new Date().toISOString(),
    };
  } catch (error) {
    throw new Error(`Failed to parse prediction response: ${error}`);
  }
}

export async function predictMaterialProperties(
  representation: string,
  materialType: string
): Promise<MaterialPrediction> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured");
  }

  const prompt = MATERIAL_PREDICTION_PROMPT
    .replace("{representation}", representation)
    .replace("{materialType}", materialType);

  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "You are a materials science expert. Always respond with valid JSON only, no markdown formatting.",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    temperature: 0.3,
    max_tokens: 2000,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    throw new Error("No response from OpenAI");
  }

  try {
    const predictions = JSON.parse(content.replace(/```json\n?|\n?```/g, "").trim());
    return {
      representation,
      materialType,
      predictions,
      confidence: 70,
      generatedAt: new Date().toISOString(),
    };
  } catch (error) {
    throw new Error(`Failed to parse prediction response: ${error}`);
  }
}

export function isOpenAIConfigured(): boolean {
  return !!process.env.OPENAI_API_KEY;
}
