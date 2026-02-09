import { useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
} from "@/components/ui/sheet";
import { useToast } from "@/hooks/use-toast";
import {
  X,
  Copy,
  Download,
  ExternalLink,
  Beaker,
  FlaskConical,
  Sparkles,
  Target,
  Layers,
  Factory,
  BarChart3,
  Building2,
  Rocket,
  AlertTriangle,
} from "lucide-react";

function formatNumber(num: number): string {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
  if (num >= 1000) return (num / 1000).toFixed(0) + "K";
  return num.toLocaleString();
}

export interface MaterialDetail {
  id: string;
  smiles: string;
  name: string;
  family: string;
  properties: {
    thermalStability: number;
    tensileStrength: number;
    conductivity: number;
    flexibility: number;
    molecularWeight: number;
    glassTransition: number;
    meltingPoint: number;
  };
  confidenceScores: {
    thermalStability: number;
    tensileStrength: number;
    conductivity: number;
    flexibility: number;
  };
  synthesis: {
    feasibility: number;
    complexity: string;
    estimatedCost: string;
    recommendedRoute: string;
    precursors: string[];
  };
  similarMaterials: {
    id: string;
    name: string;
    similarity: number;
    smiles: string;
  }[];
  overallScore: number;
  tier?: string;
  manufacturabilityScore?: number;
}

function PropertyRow({ label, value, confidence }: { label: string; value: string; confidence: number }) {
  const confidenceBadge = confidence >= 0.85
    ? { label: "High", className: "bg-green-500/10 text-green-700 dark:text-green-400" }
    : confidence >= 0.7
    ? { label: "Med", className: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400" }
    : { label: "Low", className: "bg-red-500/10 text-red-700 dark:text-red-400" };

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-sm">{value}</span>
        <Badge variant="outline" className={`text-xs ${confidenceBadge.className}`}>
          {confidenceBadge.label} ({(confidence * 100).toFixed(0)}%)
        </Badge>
      </div>
    </div>
  );
}

const TIER_CONFIG: Record<string, { label: string; icon: typeof FlaskConical; color: string }> = {
  "lab-only": { label: "Lab-Only", icon: FlaskConical, color: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30" },
  "pilot-ready": { label: "Pilot-Ready", icon: Rocket, color: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30" },
  "production-viable": { label: "Production-Viable", icon: Building2, color: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30" },
};

interface MaterialDetailPanelContentProps {
  material: MaterialDetail;
  onClose: () => void;
  onSelectMaterial?: (material: MaterialDetail) => void;
  showNavButtons?: boolean;
}

function MaterialDetailPanelContent({ material, onClose, onSelectMaterial, showNavButtons = true }: MaterialDetailPanelContentProps) {
  const { toast } = useToast();
  const [, navigate] = useLocation();

  const copySmiles = () => {
    navigator.clipboard.writeText(material.smiles);
    toast({ title: "SMILES Copied", description: "Structure copied to clipboard" });
  };

  const getFeasibilityColor = (score: number) => {
    if (score >= 0.7) return "text-green-600 dark:text-green-400";
    if (score >= 0.4) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const tierKey = material.tier || (material.synthesis.feasibility >= 0.7 ? "production-viable" : material.synthesis.feasibility >= 0.4 ? "pilot-ready" : "lab-only");
  const tierConfig = TIER_CONFIG[tierKey];
  const TierIcon = tierConfig?.icon || FlaskConical;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <div>
          <h2 className="text-lg font-semibold">{material.name}</h2>
          <div className="flex items-center gap-2 mt-1">
            <p className="text-sm text-muted-foreground font-mono">{material.id}</p>
            <Badge variant="secondary">{material.family}</Badge>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} data-testid="button-close-material-panel">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div className="text-center flex-1">
              <p className="text-sm text-muted-foreground mb-1">Overall Score</p>
              <p className="text-4xl font-bold">{(material.overallScore * 100).toFixed(1)}%</p>
            </div>
            {tierConfig && (
              <div className="text-center">
                <Badge variant="outline" className={`${tierConfig.color}`}>
                  <TierIcon className="h-3 w-3 mr-1" />
                  {tierConfig.label}
                </Badge>
                {material.manufacturabilityScore !== undefined && (
                  <p className="text-xs text-muted-foreground mt-1 font-mono">
                    Mfg: {(material.manufacturabilityScore * 100).toFixed(0)}
                  </p>
                )}
              </div>
            )}
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <Beaker className="h-4 w-4" />
              Chemical Structure (SMILES)
            </h3>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-xs bg-muted p-3 rounded-md font-mono break-all" data-testid="text-smiles">
                {material.smiles}
              </code>
              <Button variant="outline" size="icon" onClick={copySmiles} data-testid="button-copy-smiles">
                <Copy className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <Separator />

          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Predicted Properties
            </h3>
            <div className="space-y-3">
              <PropertyRow label="Thermal Stability" value={`${material.properties.thermalStability.toFixed(1)}\u00b0C`} confidence={material.confidenceScores.thermalStability} />
              <PropertyRow label="Tensile Strength" value={`${material.properties.tensileStrength.toFixed(1)} MPa`} confidence={material.confidenceScores.tensileStrength} />
              <PropertyRow label="Conductivity" value={`${material.properties.conductivity.toFixed(2)} S/m`} confidence={material.confidenceScores.conductivity} />
              <PropertyRow label="Flexibility Index" value={material.properties.flexibility.toFixed(3)} confidence={material.confidenceScores.flexibility} />
              <div className="pt-2 border-t">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Molecular Weight</span>
                  <span className="font-mono">{formatNumber(material.properties.molecularWeight)} Da</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-muted-foreground">Glass Transition (Tg)</span>
                  <span className="font-mono">{material.properties.glassTransition.toFixed(1)}\u00b0C</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-muted-foreground">Melting Point (Tm)</span>
                  <span className="font-mono">{material.properties.meltingPoint.toFixed(1)}\u00b0C</span>
                </div>
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <FlaskConical className="h-4 w-4" />
              Synthesis Feasibility
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Feasibility Score</span>
                <div className="flex items-center gap-2">
                  <Progress value={material.synthesis.feasibility * 100} className="w-24 h-2" />
                  <span className={`text-sm font-semibold ${getFeasibilityColor(material.synthesis.feasibility)}`}>
                    {(material.synthesis.feasibility * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Complexity</span>
                <Badge variant={material.synthesis.complexity === "Low" ? "secondary" : material.synthesis.complexity === "Medium" ? "outline" : "destructive"}>
                  {material.synthesis.complexity}
                </Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Estimated Cost</span>
                <span className="font-mono">{material.synthesis.estimatedCost}</span>
              </div>
              <div className="text-sm">
                <span className="text-muted-foreground">Recommended Route:</span>
                <p className="mt-1 font-medium">{material.synthesis.recommendedRoute}</p>
              </div>
              <div className="text-sm">
                <span className="text-muted-foreground">Key Precursors:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {material.synthesis.precursors.map((p, i) => (
                    <Badge key={i} variant="outline" className="text-xs">{p}</Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <Separator />

          {material.similarMaterials.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Target className="h-4 w-4" />
                Similar Materials
              </h3>
              <div className="space-y-2">
                {material.similarMaterials.map((sim) => (
                  <div
                    key={sim.id}
                    className="p-3 border rounded-lg cursor-pointer hover-elevate"
                    onClick={() => onSelectMaterial && onSelectMaterial({
                      ...material,
                      id: sim.id,
                      name: sim.name,
                      smiles: sim.smiles,
                      overallScore: sim.similarity,
                    })}
                    data-testid={`similar-material-${sim.id}`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-sm">{sim.name}</p>
                        <p className="text-xs text-muted-foreground font-mono">{sim.id}</p>
                      </div>
                      <Badge variant="secondary">{(sim.similarity * 100).toFixed(0)}% similar</Badge>
                    </div>
                    <code className="text-xs text-muted-foreground mt-1 block truncate">{sim.smiles}</code>
                  </div>
                ))}
              </div>
            </div>
          )}

          {showNavButtons && (
            <>
              <Separator />
              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <ExternalLink className="h-4 w-4" />
                  Navigate To
                </h3>
                <div className="space-y-2">
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => navigate(`/structure-property?family=${encodeURIComponent(material.family)}`)}
                    data-testid="button-nav-structure-property"
                  >
                    <BarChart3 className="h-4 w-4 mr-2" />
                    View in Structure-Property Analytics
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => navigate(`/manufacturability-scoring?family=${encodeURIComponent(material.family)}`)}
                    data-testid="button-nav-manufacturability"
                  >
                    <Factory className="h-4 w-4 mr-2" />
                    View in Manufacturability Scoring
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => navigate(`/multi-scale-representations?family=${encodeURIComponent(material.family)}`)}
                    data-testid="button-nav-multi-scale"
                  >
                    <Layers className="h-4 w-4 mr-2" />
                    View Multi-Scale Representations
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => navigate("/materials")}
                    data-testid="button-nav-materials-library"
                  >
                    <Target className="h-4 w-4 mr-2" />
                    Open in Materials Library
                  </Button>
                </div>
              </div>
            </>
          )}
        </div>
      </ScrollArea>

      <div className="p-4 border-t flex gap-2">
        <Button className="flex-1" variant="outline" onClick={copySmiles} data-testid="button-export-structure">
          <Download className="h-4 w-4 mr-2" />
          Export Structure
        </Button>
        <Button className="flex-1" data-testid="button-add-to-pipeline">
          <ExternalLink className="h-4 w-4 mr-2" />
          Add to Pipeline
        </Button>
      </div>
    </div>
  );
}

interface MaterialDetailSheetProps {
  material: MaterialDetail | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectMaterial?: (material: MaterialDetail) => void;
  showNavButtons?: boolean;
}

export function MaterialDetailSheet({ material, open, onOpenChange, onSelectMaterial, showNavButtons = true }: MaterialDetailSheetProps) {
  if (!material) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-lg p-0" side="right">
        <MaterialDetailPanelContent
          material={material}
          onClose={() => onOpenChange(false)}
          onSelectMaterial={onSelectMaterial}
          showNavButtons={showNavButtons}
        />
      </SheetContent>
    </Sheet>
  );
}

export function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

const SMILES_TEMPLATES: Record<string, string[]> = {
  "Polyamide": ["CC(=O)NCCCCCCNC(=O)CCCCC", "CC(=O)NC1CCC(NC(=O)C2CCC2)CC1", "O=C(NCCCCCCNC(=O)CCCCCC)CCCCCC"],
  "Polyester": ["CC(=O)OCCCO[C@@H](C)C(=O)O", "O=C(OCCCO)c1ccc(C(=O)OCCCO)cc1", "CC(C)(C)OC(=O)c1ccc(C(=O)OC(C)(C)C)cc1"],
  "Polyethylene": ["CCCCCCCCCCCCCCCCCCCC", "CC(C)CCCCCCCCCCCCCCCC", "CCCCC(CC)CCCCCCCCCCCC"],
  "Polypropylene": ["CC(C)CC(C)CC(C)CC(C)C", "C[C@@H](CC(C)C)CC(C)CC(C)C", "CC(C)C[C@H](C)CC(C)CC(C)C"],
  "PEEK": ["Oc1ccc(Oc2ccc(C(=O)c3ccc(O)cc3)cc2)cc1", "c1cc(Oc2ccc(C(=O)c3ccc(Oc4ccccc4)cc3)cc2)ccc1O"],
  "PPS": ["c1ccc(Sc2ccccc2)cc1", "c1cc(Sc2ccc(Sc3ccccc3)cc2)ccc1S"],
  "PTFE": ["FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "FC(F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F"],
};

export function generateMaterialDetail(family: string, index: number, seed: number): MaterialDetail {
  const rand = seededRandom(seed + index * 1337);
  const templates = SMILES_TEMPLATES[family] || SMILES_TEMPLATES["Polyethylene"];
  const smiles = templates[index % templates.length];
  const baseScore = 0.5 + rand() * 0.4;
  const feasibility = 0.4 + rand() * 0.55;

  return {
    id: `MAT-${family.substring(0, 3).toUpperCase()}-${String(index + 1).padStart(5, "0")}`,
    smiles,
    name: `${family} Variant ${index + 1}`,
    family,
    properties: {
      thermalStability: 150 + rand() * 300,
      tensileStrength: 20 + rand() * 180,
      conductivity: rand() * 10,
      flexibility: 0.1 + rand() * 0.8,
      molecularWeight: 5000 + rand() * 95000,
      glassTransition: 50 + rand() * 200,
      meltingPoint: 100 + rand() * 300,
    },
    confidenceScores: {
      thermalStability: 0.7 + rand() * 0.25,
      tensileStrength: 0.65 + rand() * 0.3,
      conductivity: 0.6 + rand() * 0.35,
      flexibility: 0.75 + rand() * 0.2,
    },
    synthesis: {
      feasibility,
      complexity: rand() > 0.6 ? "High" : rand() > 0.3 ? "Medium" : "Low",
      estimatedCost: rand() > 0.5 ? "$$$" : rand() > 0.25 ? "$$" : "$",
      recommendedRoute: rand() > 0.5 ? "Condensation polymerization" : "Ring-opening polymerization",
      precursors: [
        rand() > 0.5 ? "Adipic acid" : "Terephthalic acid",
        rand() > 0.5 ? "Hexamethylenediamine" : "Ethylene glycol",
        rand() > 0.7 ? "Catalyst (Ti-based)" : "Catalyst (Sb-based)",
      ],
    },
    similarMaterials: Array.from({ length: 3 }, (_, i) => ({
      id: `MAT-${family.substring(0, 3).toUpperCase()}-${String(i + 100 + index).padStart(5, "0")}`,
      name: `${family} Variant ${i + 100 + index}`,
      similarity: 0.75 + rand() * 0.2,
      smiles: templates[(i + index + 1) % templates.length],
    })),
    overallScore: baseScore,
    tier: feasibility >= 0.7 ? "production-viable" : feasibility >= 0.4 ? "pilot-ready" : "lab-only",
  };
}

export function generateMaterialsForFamily(family: string, count: number, seed: number): MaterialDetail[] {
  return Array.from({ length: count }, (_, i) => generateMaterialDetail(family, i, seed));
}
