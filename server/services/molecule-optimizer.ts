interface MoleculeDescriptors {
  molecularWeight: number;
  logP: number;
  tpsa: number;
  rotatableBonds: number;
  numHBondDonors: number;
  numHBondAcceptors: number;
  functionalGroups: Record<string, number>;
}

export interface OptimizationSuggestion {
  category: "solubility" | "permeability" | "safety" | "metabolic_stability" | "dose_indication";
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
  modification?: string;
}

export interface OptimizedAnalog {
  smiles: string;
  name: string;
  modification: string;
  parentSmiles: string;
  predictedProperties: {
    molecularWeight: number;
    logP: number;
    tpsa: number;
    rotatableBonds: number;
  };
  admetPredictions: {
    caco2Permeability: string;
    intestinalAbsorption: string;
    bioavailability: number;
    bbbPenetration: string;
    metabolicStability: number;
    halfLife: number;
    hergInhibition: string;
  };
}

export interface DoseScenario {
  scenario: string;
  currentDose: string;
  suggestedDose: string;
  rationale: string;
  indication: string;
  targetReceptor?: string;
  safetyNote?: string;
}

export interface OptimizationResult {
  moleculeId: string;
  smiles: string;
  properties: MoleculeDescriptors;
  suggestions: OptimizationSuggestion[];
  analogs: OptimizedAnalog[];
}

export interface DoseOptimizationResult {
  moleculeId: string;
  smiles: string;
  doseScenarios: DoseScenario[];
  repurposingHints: string[];
}

function detectFunctionalGroups(smiles: string): Record<string, number> {
  const groups: Record<string, number> = {};
  const patterns: [string, RegExp][] = [
    ["hydroxyl", /O(?=[^=])/g],
    ["amine", /N(?=[^=])/g],
    ["ester", /C\(=O\)O/g],
    ["amide", /C\(=O\)N/g],
    ["carboxyl", /C\(=O\)O[H]?(?![A-Z])/g],
    ["ketone", /C\(=O\)C/g],
    ["aldehyde", /C=O/g],
    ["nitro", /\[N\+\]\(=O\)\[O-\]/g],
    ["sulfone", /S\(=O\)\(=O\)/g],
    ["halogen", /[FClBr]/g],
    ["aromatic_ring", /c1.*?c.*?1/g],
    ["methoxy", /COC/g],
  ];
  for (const [name, pattern] of patterns) {
    const matches = smiles.match(pattern);
    if (matches && matches.length > 0) {
      groups[name] = matches.length;
    }
  }
  return groups;
}

function estimateDescriptors(smiles: string, existingProps?: { mw?: number | null; logP?: number | null; hbd?: number | null; hba?: number | null }): MoleculeDescriptors {
  const atomCount = smiles.replace(/[^A-Z]/gi, "").length;
  const mw = existingProps?.mw ?? atomCount * 12.5 + Math.random() * 50;
  const logP = existingProps?.logP ?? (atomCount * 0.15 - 0.5 + Math.random() * 1.5);
  const hbd = existingProps?.hbd ?? Math.max(0, Math.floor((smiles.match(/O|N/g)?.length || 0) * 0.4));
  const hba = existingProps?.hba ?? Math.max(0, Math.floor((smiles.match(/O|N|S/g)?.length || 0) * 0.6));
  const tpsa = hbd * 20.23 + hba * 9.23 + Math.random() * 15;
  const rotatableBonds = Math.max(0, (smiles.match(/-|C(?=[^=])/g)?.length || 0) - 2);
  const functionalGroups = detectFunctionalGroups(smiles);

  return {
    molecularWeight: Math.round(mw * 100) / 100,
    logP: Math.round(logP * 100) / 100,
    tpsa: Math.round(tpsa * 100) / 100,
    rotatableBonds,
    numHBondDonors: hbd,
    numHBondAcceptors: hba,
    functionalGroups,
  };
}

export function generateOptimizationSuggestions(
  descriptors: MoleculeDescriptors,
  diseaseContext: string = ""
): OptimizationSuggestion[] {
  const suggestions: OptimizationSuggestion[] = [];
  const disease = diseaseContext.toLowerCase();
  const fg = descriptors.functionalGroups;

  if (descriptors.tpsa < 40) {
    suggestions.push({
      category: "solubility",
      title: "Low Polar Surface Area",
      description: "TPSA < 40 indicates poor aqueous solubility. Add H-bond donors/acceptors (hydroxyl, amine) to improve solubility and oral exposure.",
      priority: "high",
      modification: "Add -OH or -NH2 group",
    });
  }

  if (descriptors.molecularWeight > 500) {
    suggestions.push({
      category: "solubility",
      title: "High Molecular Weight",
      description: `MW = ${descriptors.molecularWeight.toFixed(0)} Da exceeds Lipinski rule (>500). Reduce by removing bulky groups or simplifying structure.`,
      priority: "high",
      modification: "Remove bulky substituents",
    });
  }

  if (descriptors.logP > 5) {
    suggestions.push({
      category: "solubility",
      title: "High Lipophilicity",
      description: `logP = ${descriptors.logP.toFixed(2)} is too high (>5). Add polar groups (hydroxyl, amine) or replace lipophilic groups with more polar alternatives.`,
      priority: "high",
      modification: "Add polar groups or replace -CH3 with -OH",
    });
  } else if (descriptors.logP < 0) {
    suggestions.push({
      category: "solubility",
      title: "Low Lipophilicity",
      description: `logP = ${descriptors.logP.toFixed(2)} is too low (<0). Add methyl groups or aromatic rings to improve membrane permeability.`,
      priority: "medium",
      modification: "Add -CH3 or aromatic ring",
    });
  }

  if (descriptors.rotatableBonds > 10) {
    suggestions.push({
      category: "permeability",
      title: "High Conformational Flexibility",
      description: `${descriptors.rotatableBonds} rotatable bonds exceed recommended limit (>10). Introduce ring constraints or rigid linkers.`,
      priority: "medium",
      modification: "Introduce cyclic constraint or rigid linker",
    });
  }

  const isNeuro = disease.includes("neurological") || disease.includes("alzheimer") ||
    disease.includes("parkinson") || disease.includes("psp") || disease.includes("supranuclear") ||
    disease.includes("huntington") || disease.includes("epilep") || disease.includes("cns") ||
    disease.includes("brain") || disease.includes("neuro");

  if (isNeuro) {
    if (descriptors.molecularWeight > 450) {
      suggestions.push({
        category: "permeability",
        title: "CNS Drug: High MW",
        description: "For CNS drugs, reduce MW to < 450 Da for better blood-brain barrier penetration.",
        priority: "high",
        modification: "Reduce MW below 450 Da",
      });
    }
    if (descriptors.tpsa > 90) {
      suggestions.push({
        category: "permeability",
        title: "CNS Drug: High TPSA",
        description: "For CNS drugs, reduce TPSA to < 90 \u00c5\u00b2 for better BBB penetration.",
        priority: "high",
        modification: "Reduce polar surface area",
      });
    }
  }

  if ((fg.hydroxyl || 0) > 3) {
    suggestions.push({
      category: "safety",
      title: "Cardiotoxicity Risk",
      description: `${fg.hydroxyl} hydroxyl groups detected. Multiple hydroxyls increase hERG channel interaction risk. Consider replacing some with methoxy (-OCH3).`,
      priority: "high",
      modification: "-OH \u2192 -OCH3 swap",
    });
  }

  if (fg.ester && fg.ester > 0) {
    suggestions.push({
      category: "metabolic_stability",
      title: "Ester Group Present",
      description: "Ester groups are rapidly hydrolyzed in vivo by esterases. Consider replacing ester with amide for improved metabolic stability.",
      priority: "medium",
      modification: "Ester \u2192 Amide replacement",
    });
  }

  const isObesity = disease.includes("obesity") || disease.includes("metabolic") || disease.includes("weight");
  if (isObesity) {
    suggestions.push({
      category: "dose_indication",
      title: "Dose Repurposing Opportunity",
      description: "Consider lower dose (10\u201320\u00d7 reduction) for epilepsy indication (Dravet syndrome / Lennox-Gastaut). Fenfluramine-style approach.",
      priority: "medium",
    });
    suggestions.push({
      category: "dose_indication",
      title: "Target Selectivity",
      description: "Target 5-HT2C receptor agonism while minimizing 5-HT2B agonism to reduce cardiac valvulopathy risk.",
      priority: "high",
    });
  }

  const isCancer = disease.includes("cancer") || disease.includes("oncol") || disease.includes("tumor") || disease.includes("leukemia") || disease.includes("lymphoma");
  if (isCancer) {
    suggestions.push({
      category: "dose_indication",
      title: "Dose Optimization",
      description: "Evaluate lower-dose regimen for chronic administration. Consider combination therapy to reduce single-agent toxicity.",
      priority: "medium",
    });
  }

  if (suggestions.filter(s => s.category === "dose_indication").length === 0) {
    if (descriptors.molecularWeight < 300 && descriptors.logP > 1 && descriptors.logP < 3) {
      suggestions.push({
        category: "dose_indication",
        title: "Fragment-like Profile",
        description: "Molecule has fragment-like properties. May serve as starting point for multiple therapeutic indications through fragment growing/merging.",
        priority: "low",
      });
    }
  }

  return suggestions;
}

function applyModification(smiles: string, modification: string, descriptors: MoleculeDescriptors): OptimizedAnalog | null {
  let newSmiles = smiles;
  let modDesc = modification;
  const props = { ...descriptors };

  if (modification.includes("Ester") && modification.includes("Amide")) {
    newSmiles = smiles.replace(/C\(=O\)O/, "C(=O)N");
    if (newSmiles === smiles) return null;
    modDesc = "Ester\u2192Amide: improved metabolic stability";
    props.molecularWeight = props.molecularWeight - 1;
    props.tpsa = props.tpsa + 5;
  } else if (modification.includes("OH") && modification.includes("OCH3")) {
    newSmiles = smiles.replace(/O(?=[^=C(])/, "OC");
    if (newSmiles === smiles) newSmiles = smiles + "(OC)";
    modDesc = "Hydroxyl\u2192Methoxy: reduced cardiotoxicity risk";
    props.molecularWeight = props.molecularWeight + 14;
    props.logP = props.logP + 0.5;
    props.tpsa = Math.max(0, props.tpsa - 10);
  } else if (modification.includes("polar") || modification.includes("-OH") || modification.includes("Add -OH")) {
    newSmiles = smiles + "O";
    modDesc = "Added hydroxyl: improved solubility";
    props.molecularWeight = props.molecularWeight + 17;
    props.logP = props.logP - 0.7;
    props.tpsa = props.tpsa + 20;
    props.numHBondDonors = props.numHBondDonors + 1;
  } else if (modification.includes("CH3") || modification.includes("methyl") || modification.includes("aromatic")) {
    newSmiles = smiles + "C";
    modDesc = "Added methyl: increased lipophilicity";
    props.molecularWeight = props.molecularWeight + 15;
    props.logP = props.logP + 0.5;
  } else if (modification.includes("cyclic") || modification.includes("rigid")) {
    const idx = Math.floor(smiles.length / 2);
    newSmiles = smiles.slice(0, idx) + "C1CC1" + smiles.slice(idx);
    modDesc = "Ring constraint: reduced flexibility";
    props.molecularWeight = props.molecularWeight + 42;
    props.rotatableBonds = Math.max(0, props.rotatableBonds - 2);
  } else if (modification.includes("Remove bulky") || modification.includes("Reduce MW")) {
    if (smiles.length > 10) {
      newSmiles = smiles.slice(0, Math.floor(smiles.length * 0.8));
    }
    modDesc = "Simplified structure: reduced MW";
    props.molecularWeight = props.molecularWeight * 0.85;
  } else if (modification.includes("Reduce polar")) {
    newSmiles = smiles.replace(/O(?=[^=])/, "C");
    if (newSmiles === smiles) newSmiles = smiles.replace(/N(?=[^=])/, "C");
    modDesc = "Reduced PSA for BBB penetration";
    props.tpsa = Math.max(0, props.tpsa - 20);
    props.logP = props.logP + 0.3;
  } else {
    return null;
  }

  if (newSmiles === smiles) return null;

  const bioavailability = Math.min(0.95, Math.max(0.1, 0.8 - (props.molecularWeight - 300) / 1000 - Math.max(0, props.logP - 3) / 10));
  const metabolicStability = props.tpsa > 50 ? 0.7 + Math.random() * 0.2 : 0.4 + Math.random() * 0.3;

  return {
    smiles: newSmiles,
    name: `Opt-${modDesc.split(":")[0].trim().replace(/\s+/g, "_")}`,
    modification: modDesc,
    parentSmiles: smiles,
    predictedProperties: {
      molecularWeight: Math.round(props.molecularWeight * 100) / 100,
      logP: Math.round(props.logP * 100) / 100,
      tpsa: Math.round(props.tpsa * 100) / 100,
      rotatableBonds: props.rotatableBonds,
    },
    admetPredictions: {
      caco2Permeability: props.logP > 1 && props.tpsa < 100 ? "High" : props.tpsa < 140 ? "Moderate" : "Low",
      intestinalAbsorption: props.tpsa < 140 ? "High" : "Low",
      bioavailability: Math.round(bioavailability * 1000) / 1000,
      bbbPenetration: props.molecularWeight < 450 && props.tpsa < 90 ? "Yes" : "No",
      metabolicStability: Math.round(metabolicStability * 1000) / 1000,
      halfLife: Math.round((4 + metabolicStability * 12) * 10) / 10,
      hergInhibition: props.logP > 4 ? "Risk" : "Low",
    },
  };
}

export function optimizeMoleculeProperties(
  smiles: string,
  diseaseContext: string,
  existingProps?: { mw?: number | null; logP?: number | null; hbd?: number | null; hba?: number | null }
): OptimizationResult {
  const descriptors = estimateDescriptors(smiles, existingProps);
  const suggestions = generateOptimizationSuggestions(descriptors, diseaseContext);
  const analogs: OptimizedAnalog[] = [];

  for (const suggestion of suggestions) {
    if (suggestion.modification) {
      const analog = applyModification(smiles, suggestion.modification, descriptors);
      if (analog) {
        analogs.push(analog);
      }
    }
  }

  return {
    moleculeId: "",
    smiles,
    properties: descriptors,
    suggestions,
    analogs,
  };
}

export function optimizeDoseIndication(
  smiles: string,
  diseaseContext: string,
  moleculeName?: string
): DoseOptimizationResult {
  const disease = diseaseContext.toLowerCase();
  const doseScenarios: DoseScenario[] = [];
  const repurposingHints: string[] = [];

  if (disease.includes("obesity") || disease.includes("weight") || disease.includes("metabolic")) {
    doseScenarios.push({
      scenario: "Epilepsy Repurposing (Dravet Syndrome)",
      currentDose: "60\u2013120 mg/day (obesity)",
      suggestedDose: "0.2\u20130.7 mg/kg/day",
      rationale: "Fenfluramine-style 10\u201320\u00d7 dose reduction effective for Dravet syndrome seizure control",
      indication: "Dravet Syndrome / Lennox-Gastaut Syndrome",
      targetReceptor: "5-HT2C agonist",
      safetyNote: "Minimize 5-HT2B agonism to reduce cardiac valvulopathy risk",
    });
    repurposingHints.push(
      "Low-dose serotonergic activity may benefit seizure control",
      "Consider sigma-1 receptor activity for neuroprotection"
    );
  }

  if (disease.includes("cancer") || disease.includes("oncol") || disease.includes("tumor")) {
    doseScenarios.push({
      scenario: "Low-Dose Metronomic Therapy",
      currentDose: "Standard MTD-based dosing",
      suggestedDose: "1/10 to 1/3 of MTD, continuous",
      rationale: "Metronomic dosing targets tumor vasculature with reduced toxicity, improved tolerability for chronic administration",
      indication: "Anti-angiogenic / maintenance therapy",
      safetyNote: "Monitor for myelosuppression at reduced frequency",
    });
    doseScenarios.push({
      scenario: "Combination Synergy",
      currentDose: "Single-agent MTD",
      suggestedDose: "50\u201375% of single-agent dose",
      rationale: "Dose reduction in combination allows synergistic efficacy while reducing overlapping toxicities",
      indication: "Combination therapy",
    });
    repurposingHints.push(
      "Evaluate for anti-inflammatory indications at sub-therapeutic oncology doses",
      "Consider immunomodulatory properties for autoimmune conditions"
    );
  }

  if (disease.includes("alzheimer") || disease.includes("psp") || disease.includes("supranuclear") ||
      disease.includes("parkinson") || disease.includes("neuro") || disease.includes("huntington")) {
    doseScenarios.push({
      scenario: "CNS-Optimized Dosing",
      currentDose: "Standard peripheral dose",
      suggestedDose: "Adjusted for BBB penetration (lower if high permeability, higher if low)",
      rationale: "CNS drugs require dose optimization based on BBB penetration and CSF/plasma ratio",
      indication: diseaseContext,
      safetyNote: "Monitor for CNS side effects: somnolence, cognitive changes",
    });
    doseScenarios.push({
      scenario: "Neuroprotective Low-Dose",
      currentDose: "Symptomatic treatment dose",
      suggestedDose: "25\u201350% of symptomatic dose",
      rationale: "Sub-symptomatic doses may provide neuroprotection via anti-inflammatory / antioxidant mechanisms",
      indication: "Disease modification / neuroprotection",
    });
    repurposingHints.push(
      "Evaluate tau aggregation inhibition at modified doses",
      "Consider anti-neuroinflammatory properties for related tauopathies"
    );
  }

  if (disease.includes("inflamm") || disease.includes("arthritis") || disease.includes("autoimmune")) {
    doseScenarios.push({
      scenario: "Low-Dose Anti-inflammatory",
      currentDose: "Standard anti-inflammatory dose",
      suggestedDose: "30\u201350% dose reduction",
      rationale: "Lower doses maintain anti-inflammatory efficacy while reducing GI and cardiovascular side effects",
      indication: "Chronic inflammatory conditions",
      safetyNote: "Monitor GI tolerability and cardiovascular markers",
    });
    repurposingHints.push("Consider for fibrotic conditions at modified doses");
  }

  if (doseScenarios.length === 0) {
    doseScenarios.push({
      scenario: "Standard Dose Optimization",
      currentDose: "Initial therapeutic dose",
      suggestedDose: "Titrate based on PK/PD modeling",
      rationale: "Optimize dose-response relationship using exposure-response analysis",
      indication: diseaseContext || "Primary indication",
      safetyNote: "Monitor therapeutic window with biomarker-guided dosing",
    });
    repurposingHints.push(
      "Screen for activity in related disease pathways",
      "Evaluate polypharmacology for multi-indication potential"
    );
  }

  return {
    moleculeId: "",
    smiles,
    doseScenarios,
    repurposingHints,
  };
}
