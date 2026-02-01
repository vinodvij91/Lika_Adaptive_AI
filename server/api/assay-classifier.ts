export type AssayCategory = "binding" | "functional" | "adme" | "safety" | "physicochemical";

interface ClassificationResult {
  category: AssayCategory;
  confidence: number;
  keywords: string[];
}

const CATEGORY_KEYWORDS: Record<AssayCategory, string[]> = {
  binding: [
    "kd", "ic50", "ki", "ec50", "spr", "binding", "affinity", "dissociation",
    "inhibition", "potency", "dose-response", "competition", "displacement",
    "radioligand", "fluorescence polarization", "fitc", "tr-fret", "alphascreen"
  ],
  functional: [
    "aggregation", "pathway", "cell-based", "cellular", "phosphorylation",
    "activation", "inflammasome", "cytokine", "secretion", "proliferation",
    "apoptosis", "migration", "invasion", "colony", "reporter", "luciferase",
    "western", "elisa", "flow cytometry", "immunofluorescence"
  ],
  adme: [
    "permeability", "metabolism", "solubility", "caco-2", "mdck", "pampa",
    "microsomal", "hepatocyte", "cyp", "clearance", "half-life", "bbb",
    "blood-brain", "p-gp", "efflux", "transport", "absorption", "distribution",
    "plasma protein", "bioavailability"
  ],
  safety: [
    "cytotox", "cytotoxicity", "herg", "cardiac", "genotox", "mutagenicity",
    "ames", "micronucleus", "chromosome", "toxicity", "viability", "ldh",
    "atp", "mtt", "resazurin", "off-target", "selectivity", "safety"
  ],
  physicochemical: [
    "logd", "logp", "psa", "polar surface", "molecular weight", "solubility",
    "stability", "pka", "lipophilicity", "hydrophilicity", "crystallinity",
    "melting point", "partition", "dissolution", "formulation"
  ]
};

export function classifyAssay(description: string, assayName?: string): ClassificationResult {
  const text = `${description} ${assayName || ""}`.toLowerCase();
  const scores: Record<AssayCategory, { score: number; matches: string[] }> = {
    binding: { score: 0, matches: [] },
    functional: { score: 0, matches: [] },
    adme: { score: 0, matches: [] },
    safety: { score: 0, matches: [] },
    physicochemical: { score: 0, matches: [] }
  };

  for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    for (const keyword of keywords) {
      if (text.includes(keyword)) {
        scores[category as AssayCategory].score += 1;
        scores[category as AssayCategory].matches.push(keyword);
      }
    }
  }

  let bestCategory: AssayCategory = "binding";
  let bestScore = 0;

  for (const [category, data] of Object.entries(scores)) {
    if (data.score > bestScore) {
      bestScore = data.score;
      bestCategory = category as AssayCategory;
    }
  }

  const totalMatches = Object.values(scores).reduce((sum, d) => sum + d.score, 0);
  
  // If no keywords matched, return low confidence and default to "functional" as generic category
  if (totalMatches === 0) {
    return {
      category: "functional" as AssayCategory,
      confidence: 0.1,
      keywords: []
    };
  }
  
  const confidence = bestScore / totalMatches;

  return {
    category: bestCategory,
    confidence: Math.max(confidence, 0.3), // Minimum 30% confidence if any keywords matched
    keywords: scores[bestCategory].matches
  };
}

export function categorizeAssays(assays: Array<{ id: string; name: string; description: string }>): Record<AssayCategory, typeof assays> {
  const result: Record<AssayCategory, typeof assays> = {
    binding: [],
    functional: [],
    adme: [],
    safety: [],
    physicochemical: []
  };

  for (const assay of assays) {
    const classification = classifyAssay(assay.description, assay.name);
    result[classification.category].push(assay);
  }

  return result;
}
