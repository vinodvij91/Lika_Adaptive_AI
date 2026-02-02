import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
  BarChart,
  Bar,
  LineChart,
  Line,
  Legend,
} from "recharts";
import {
  Layers,
  TrendingUp,
  BarChart3,
  Grid3X3,
  Hexagon,
  Target,
  Filter,
  Download,
  ZoomIn,
  Activity,
  X,
  Copy,
  ExternalLink,
  Beaker,
  FlaskConical,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Info,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

const GROUP_CONFIGS: Record<string, { labels: string[]; paramLabel: string }> = {
  family: { 
    labels: ["Polyamide", "Polyester", "Polyethylene", "Polypropylene", "PEEK", "PPS", "PTFE"],
    paramLabel: "Material Family"
  },
  scaffold: { 
    labels: ["Linear Chain", "Branched", "Cross-linked", "Dendritic", "Star", "Ladder"],
    paramLabel: "Scaffold Type"
  },
  chain_length: { 
    labels: ["10-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"],
    paramLabel: "Chain Length Bin"
  },
  dopant: { 
    labels: ["None", "Carbon", "Silicon", "Nitrogen", "Fluorine", "Metal Oxide"],
    paramLabel: "Dopant Type"
  },
};

const PROPERTY_OFFSETS: Record<string, number> = {
  thermal_stability: 0,
  tensile_strength: 0.1,
  conductivity: 0.2,
  flexibility: 0.15,
};

// Individual material types for drill-down
interface IndividualMaterial {
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
}

// SMILES templates for different polymer families
const SMILES_TEMPLATES: Record<string, string[]> = {
  "Polyamide": [
    "CC(=O)NCCCCCCNC(=O)CCCCC",
    "CC(=O)NC1CCC(NC(=O)C2CCC2)CC1",
    "O=C(NCCCCCCNC(=O)CCCCCC)CCCCCC"
  ],
  "Polyester": [
    "CC(=O)OCCCO[C@@H](C)C(=O)O",
    "O=C(OCCCO)c1ccc(C(=O)OCCCO)cc1",
    "CC(C)(C)OC(=O)c1ccc(C(=O)OC(C)(C)C)cc1"
  ],
  "Polyethylene": [
    "CCCCCCCCCCCCCCCCCCCC",
    "CC(C)CCCCCCCCCCCCCCCC",
    "CCCCC(CC)CCCCCCCCCCCC"
  ],
  "Polypropylene": [
    "CC(C)CC(C)CC(C)CC(C)C",
    "C[C@@H](CC(C)C)CC(C)CC(C)C",
    "CC(C)C[C@H](C)CC(C)CC(C)C"
  ],
  "PEEK": [
    "Oc1ccc(Oc2ccc(C(=O)c3ccc(O)cc3)cc2)cc1",
    "c1cc(Oc2ccc(C(=O)c3ccc(Oc4ccccc4)cc3)cc2)ccc1O",
    "O=C(c1ccc(Oc2ccccc2)cc1)c1ccc(Oc2ccccc2)cc1"
  ],
  "PPS": [
    "c1ccc(Sc2ccccc2)cc1",
    "c1cc(Sc2ccc(Sc3ccccc3)cc2)ccc1S",
    "Sc1ccc(Sc2ccc(Sc3ccccc3)cc2)cc1"
  ],
  "PTFE": [
    "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "FC(F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F",
    "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
  ],
};

function generateIndividualMaterial(family: string, index: number, seed: number): IndividualMaterial {
  const rand = seededRandom(seed + index * 1337);
  const templates = SMILES_TEMPLATES[family] || SMILES_TEMPLATES["Polyethylene"];
  const smiles = templates[index % templates.length];
  
  const baseScore = 0.5 + rand() * 0.4;
  
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
      feasibility: 0.4 + rand() * 0.55,
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
  };
}

function generateMaterialsFromGroup(family: string, count: number, seed: number): IndividualMaterial[] {
  return Array.from({ length: count }, (_, i) => generateIndividualMaterial(family, i, seed));
}

// Material Detail Panel Component
interface MaterialDetailPanelProps {
  material: IndividualMaterial | null;
  onClose: () => void;
  onSelectMaterial: (material: IndividualMaterial) => void;
}

function MaterialDetailPanel({ material, onClose, onSelectMaterial }: MaterialDetailPanelProps) {
  const { toast } = useToast();
  
  if (!material) return null;
  
  const copySmiles = () => {
    navigator.clipboard.writeText(material.smiles);
    toast({
      title: "SMILES Copied",
      description: "Structure copied to clipboard",
    });
  };
  
  const getFeasibilityColor = (score: number) => {
    if (score >= 0.7) return "text-green-600 dark:text-green-400";
    if (score >= 0.4) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };
  
  const getConfidenceBadge = (score: number) => {
    if (score >= 0.85) return { label: "High", variant: "default" as const };
    if (score >= 0.7) return { label: "Medium", variant: "secondary" as const };
    return { label: "Low", variant: "outline" as const };
  };
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <div>
          <h2 className="text-lg font-semibold">{material.name}</h2>
          <p className="text-sm text-muted-foreground font-mono">{material.id}</p>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} data-testid="button-close-material-panel">
          <X className="h-4 w-4" />
        </Button>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          {/* Overall Score */}
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <p className="text-sm text-muted-foreground mb-1">Overall Score</p>
            <p className="text-4xl font-bold">{(material.overallScore * 100).toFixed(1)}%</p>
            <Badge variant="secondary" className="mt-2">{material.family}</Badge>
          </div>
          
          {/* SMILES Structure */}
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
            {/* Structure Schematic */}
            <div className="mt-3 p-3 bg-muted/30 rounded-md border">
              <p className="text-xs text-muted-foreground mb-2">Repeat Unit Schematic</p>
              <div className="font-mono text-xs leading-relaxed text-center" data-testid="structure-preview">
                {material.family === "Polyamide" && (
                  <pre className="inline-block text-left">{`
     O       O
     ||      ||
 ~N-C-(CH2)6-C-N-(CH2)6~
  H             H
                  `}</pre>
                )}
                {material.family === "Polyester" && (
                  <pre className="inline-block text-left">{`
     O           O
     ||          ||
 ~O-C-[Ar]-C-O-(CH2)2~
                  `}</pre>
                )}
                {material.family === "Polyethylene" && (
                  <pre className="inline-block text-left">{`
 ~(CH2-CH2)n~
                  `}</pre>
                )}
                {material.family === "PEEK" && (
                  <pre className="inline-block text-left">{`
       O
       ||
 ~[Ar]-O-[Ar]-C-[Ar]-O~
                  `}</pre>
                )}
                {!["Polyamide", "Polyester", "Polyethylene", "PEEK"].includes(material.family) && (
                  <pre className="inline-block text-left">{`
 ~[Repeat Unit]n~
                  `}</pre>
                )}
              </div>
            </div>
          </div>
          
          <Separator />
          
          {/* Properties with Confidence */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Predicted Properties
            </h3>
            <div className="space-y-3">
              <PropertyRow 
                label="Thermal Stability" 
                value={`${material.properties.thermalStability.toFixed(1)}°C`}
                confidence={material.confidenceScores.thermalStability}
              />
              <PropertyRow 
                label="Tensile Strength" 
                value={`${material.properties.tensileStrength.toFixed(1)} MPa`}
                confidence={material.confidenceScores.tensileStrength}
              />
              <PropertyRow 
                label="Conductivity" 
                value={`${material.properties.conductivity.toFixed(2)} S/m`}
                confidence={material.confidenceScores.conductivity}
              />
              <PropertyRow 
                label="Flexibility Index" 
                value={material.properties.flexibility.toFixed(3)}
                confidence={material.confidenceScores.flexibility}
              />
              <div className="pt-2 border-t">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Molecular Weight</span>
                  <span className="font-mono">{formatNumber(material.properties.molecularWeight)} Da</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-muted-foreground">Glass Transition (Tg)</span>
                  <span className="font-mono">{material.properties.glassTransition.toFixed(1)}°C</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-muted-foreground">Melting Point (Tm)</span>
                  <span className="font-mono">{material.properties.meltingPoint.toFixed(1)}°C</span>
                </div>
              </div>
            </div>
          </div>
          
          <Separator />
          
          {/* Synthesis Feasibility */}
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
          
          {/* Similar Materials */}
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
                  onClick={() => onSelectMaterial(generateIndividualMaterial(material.family, parseInt(sim.id.split("-")[2]), 42))}
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

// Property Row Component with Confidence
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
};

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function generateMockPercentileData(totalVariants: number, groupBy: string, seed: number) {
  const rand = seededRandom(seed);
  const percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99];
  const groupOffset = Object.keys(GROUP_CONFIGS).indexOf(groupBy) * 0.05;
  return percentiles.map(p => ({
    percentile: p,
    score: rand() * 0.2 + (100 - p) / 100 * 0.7 + groupOffset,
    count: Math.floor(totalVariants * (p / 100)),
    label: `P${p}`,
  }));
}

function generateMockDensityData(groupBy: string, propertyX: string, seed: number) {
  const rand = seededRandom(seed);
  const bins = 20;
  const data = [];
  const propOffset = PROPERTY_OFFSETS[propertyX] || 0;
  const groupSeed = Object.keys(GROUP_CONFIGS).indexOf(groupBy) * 0.1;
  const peakPos = 0.4 + groupSeed + propOffset;
  
  for (let i = 0; i < bins; i++) {
    const x = i / bins;
    const density = Math.exp(-Math.pow((x - peakPos) * 3, 2)) * 0.7 + 
                   Math.exp(-Math.pow((x - 0.8 - propOffset) * 5, 2)) * 0.3;
    data.push({
      bin: (x * 100).toFixed(0),
      density: density * 1000 + rand() * 100,
      topPerformers: density > 0.5 ? density * 500 : 0,
    });
  }
  return data;
}

function generateMockBinnedScatter(propertyX: string, propertyY: string, seed: number) {
  const rand = seededRandom(seed);
  const data = [];
  const xOffset = PROPERTY_OFFSETS[propertyX] || 0;
  const yOffset = PROPERTY_OFFSETS[propertyY] || 0;
  const centerX = 0.5 + xOffset * 0.5;
  const centerY = 0.6 + yOffset * 0.5;
  
  for (let i = 0; i < 15; i++) {
    for (let j = 0; j < 15; j++) {
      const x = i / 15;
      const y = j / 15;
      const count = Math.floor(Math.exp(-Math.pow(x - centerX, 2) * 4 - Math.pow(y - centerY, 2) * 4) * 5000 + rand() * 200);
      if (count > 100) {
        data.push({
          x: x * 100,
          y: y * 100,
          count,
          avgScore: 0.3 + (x + y) / 4 + rand() * 0.1,
        });
      }
    }
  }
  return data;
}

function generateMockGroupData(groupBy: string, seed: number) {
  const rand = seededRandom(seed);
  const config = GROUP_CONFIGS[groupBy] || GROUP_CONFIGS.family;
  return config.labels.map((label, idx) => ({
    group: label,
    variantCount: Math.floor(rand() * 50000) + 5000 + idx * 3000,
    avgScore: rand() * 0.3 + 0.45 + idx * 0.02,
    topPerformers: Math.floor(rand() * 2000) + 500 + idx * 100,
    p95Score: rand() * 0.15 + 0.78 + idx * 0.015,
  }));
}

function generateMockTrendData(groupBy: string, propertyX: string, seed: number) {
  const rand = seededRandom(seed);
  const data = [];
  const propOffset = PROPERTY_OFFSETS[propertyX] || 0;
  const config = GROUP_CONFIGS[groupBy] || GROUP_CONFIGS.family;
  
  for (let i = 0; i < 10; i++) {
    const param = (i + 1) * 10;
    const baseP50 = 0.4 + i * 0.03 + propOffset;
    data.push({
      parameter: param,
      parameterLabel: groupBy === "chain_length" ? `${param}` : config.labels[i % config.labels.length],
      p50: baseP50 + rand() * 0.05,
      p75: baseP50 + 0.1 + rand() * 0.05,
      p95: baseP50 + 0.25 + rand() * 0.05,
      count: Math.floor(rand() * 15000) + 5000,
    });
  }
  return data;
}

interface InsightBadgeProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  variant?: "default" | "success" | "warning";
}

function InsightBadge({ icon, label, value, variant = "default" }: InsightBadgeProps) {
  const bgClass = variant === "success" 
    ? "bg-green-500/10 border-green-500/30 text-green-700 dark:text-green-400"
    : variant === "warning"
    ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-700 dark:text-yellow-400"
    : "bg-muted border-border";

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md border ${bgClass}`}>
      {icon}
      <span className="text-sm font-medium">{label}:</span>
      <span className="text-sm font-mono">{value}</span>
    </div>
  );
}

function DensityHeatmap({ 
  data, 
  onCellClick,
  xLabel = "X-Axis",
  yLabel = "Y-Axis"
}: { 
  data: any[]; 
  onCellClick?: (cellIndex: number) => void;
  xLabel?: string;
  yLabel?: string;
}) {
  const maxCount = Math.max(...data.map(d => d.count));
  const gridSize = 8; // 8x8 grid for better visibility
  const xLabels = ["Low", "", "", "Med", "", "", "", "High"];
  const yLabels = ["High", "", "", "Med", "", "", "", "Low"];
  
  return (
    <div className="space-y-3">
      <div className="flex items-stretch gap-2">
        {/* Y-axis label */}
        <div className="flex items-center justify-center w-6">
          <span className="text-xs text-muted-foreground font-medium -rotate-90 whitespace-nowrap">
            {yLabel}
          </span>
        </div>
        
        {/* Y-axis ticks */}
        <div className="flex flex-col justify-between text-xs text-muted-foreground py-1" style={{ width: '32px' }}>
          {yLabels.map((label, i) => (
            <span key={i} className="text-right pr-1 h-6 flex items-center justify-end">{label}</span>
          ))}
        </div>
        
        {/* Heatmap grid */}
        <div className="flex-1 max-w-md">
          <div className="grid grid-cols-8 gap-1">
            {data.slice(0, gridSize * gridSize).map((cell, i) => {
              const intensity = cell.count / maxCount;
              const hue = 220 - intensity * 160;
              return (
                <div
                  key={i}
                  className="aspect-square rounded cursor-pointer hover-elevate"
                  style={{
                    backgroundColor: `hsl(${hue}, 70%, ${40 + intensity * 30}%)`,
                    opacity: 0.3 + intensity * 0.7,
                    minHeight: '24px',
                  }}
                  title={`Count: ${formatNumber(cell.count)}\nAvg Score: ${cell.avgScore.toFixed(2)}\nClick to view materials`}
                  onClick={() => onCellClick?.(i)}
                  data-testid={`heatmap-cell-${i}`}
                />
              );
            })}
          </div>
          
          {/* X-axis ticks */}
          <div className="flex justify-between text-xs text-muted-foreground mt-1 px-0.5">
            {xLabels.map((label, i) => (
              <span key={i} className="w-6 text-center">{label}</span>
            ))}
          </div>
          
          {/* X-axis label */}
          <div className="text-center mt-1">
            <span className="text-xs text-muted-foreground font-medium">{xLabel}</span>
          </div>
        </div>
        
        {/* Color legend */}
        <div className="flex flex-col items-center justify-center gap-1 ml-4">
          <span className="text-xs text-muted-foreground">High</span>
          <div className="flex flex-col gap-0.5">
            {[1, 0.8, 0.6, 0.4, 0.2].map((v, i) => (
              <div
                key={i}
                className="w-4 h-4 rounded-sm"
                style={{
                  backgroundColor: `hsl(${220 - v * 160}, 70%, ${40 + v * 30}%)`,
                }}
              />
            ))}
          </div>
          <span className="text-xs text-muted-foreground">Low</span>
          <span className="text-xs text-muted-foreground mt-1">Density</span>
        </div>
      </div>
    </div>
  );
}

export default function StructurePropertyPage() {
  const { toast } = useToast();
  const [groupBy, setGroupBy] = useState<string>("family");
  const [propertyX, setPropertyX] = useState<string>("thermal_stability");
  const [propertyY, setPropertyY] = useState<string>("tensile_strength");
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false);
  const [analysisSeed, setAnalysisSeed] = useState(0);
  const [selectedMaterial, setSelectedMaterial] = useState<IndividualMaterial | null>(null);
  const [materialPanelOpen, setMaterialPanelOpen] = useState(false);

  const handleRunAnalysis = () => {
    setIsRunningAnalysis(true);
    toast({
      title: "Running Analysis",
      description: "Processing 127K+ material variants...",
    });
    // Simulate analysis with new random seed
    setTimeout(() => {
      setAnalysisSeed(prev => prev + 1);
      setIsRunningAnalysis(false);
      toast({
        title: "Analysis Complete",
        description: "Updated structure-property correlations with latest data.",
      });
    }, 1500);
  };

  const handleMaterialClick = (family: string, index: number) => {
    const material = generateIndividualMaterial(family, index, seed);
    setSelectedMaterial(material);
    setMaterialPanelOpen(true);
  };

  const handleSelectMaterial = (material: IndividualMaterial) => {
    setSelectedMaterial(material);
    setMaterialPanelOpen(true);
  };

  const totalVariants = 127450 + analysisSeed * 150;
  const topPerformersCount = 12450 + analysisSeed * 25;
  const topPercentile = 5;

  const groupConfig = GROUP_CONFIGS[groupBy] || GROUP_CONFIGS.family;
  const seed = groupBy.length + propertyX.length + propertyY.length + analysisSeed;

  const percentileData = useMemo(() => generateMockPercentileData(totalVariants, groupBy, seed), [groupBy, seed]);
  const densityData = useMemo(() => generateMockDensityData(groupBy, propertyX, seed), [groupBy, propertyX, seed]);
  const binnedScatterData = useMemo(() => generateMockBinnedScatter(propertyX, propertyY, seed), [propertyX, propertyY, seed]);
  const groupData = useMemo(() => generateMockGroupData(groupBy, seed), [groupBy, seed]);
  const trendData = useMemo(() => generateMockTrendData(groupBy, propertyX, seed), [groupBy, propertyX, seed]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Structure-Property Analytics" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" data-testid="button-export-analytics">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button onClick={handleRunAnalysis} disabled={isRunningAnalysis} data-testid="button-run-analysis">
              {isRunningAnalysis ? (
                <>
                  <div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4 mr-2" />
                  Run Analysis
                </>
              )}
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <InsightBadge
              icon={<Layers className="h-4 w-4" />}
              label="Total Variants"
              value={formatNumber(totalVariants)}
            />
            <InsightBadge
              icon={<Target className="h-4 w-4" />}
              label={`Top ${topPercentile}% performers`}
              value={`${formatNumber(topPerformersCount)} variants`}
              variant="success"
            />
            <InsightBadge
              icon={<TrendingUp className="h-4 w-4" />}
              label="Performance plateau"
              value="detected at scale"
              variant="warning"
            />
          </div>

          <Card>
            <CardHeader className="pb-2">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Filter className="h-5 w-5" />
                  Analysis Controls
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Group By</Label>
                  <Select value={groupBy} onValueChange={setGroupBy}>
                    <SelectTrigger data-testid="select-group-by">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="family">Material Family</SelectItem>
                      <SelectItem value="scaffold">Scaffold / Lattice</SelectItem>
                      <SelectItem value="chain_length">Chain Length Bins</SelectItem>
                      <SelectItem value="dopant">Dopant Type</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>X-Axis Property</Label>
                  <Select value={propertyX} onValueChange={setPropertyX}>
                    <SelectTrigger data-testid="select-property-x">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="thermal_stability">Thermal Stability</SelectItem>
                      <SelectItem value="tensile_strength">Tensile Strength</SelectItem>
                      <SelectItem value="conductivity">Conductivity</SelectItem>
                      <SelectItem value="flexibility">Flexibility</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Y-Axis Property</Label>
                  <Select value={propertyY} onValueChange={setPropertyY}>
                    <SelectTrigger data-testid="select-property-y">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="thermal_stability">Thermal Stability</SelectItem>
                      <SelectItem value="tensile_strength">Tensile Strength</SelectItem>
                      <SelectItem value="conductivity">Conductivity</SelectItem>
                      <SelectItem value="flexibility">Flexibility</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="density" className="space-y-4">
            <TabsList>
              <TabsTrigger value="density" data-testid="tab-density">
                <Grid3X3 className="h-4 w-4 mr-2" />
                Density Plot
              </TabsTrigger>
              <TabsTrigger value="percentiles" data-testid="tab-percentiles">
                <TrendingUp className="h-4 w-4 mr-2" />
                Percentile Curves
              </TabsTrigger>
              <TabsTrigger value="heatmap" data-testid="tab-heatmap">
                <Hexagon className="h-4 w-4 mr-2" />
                Property Heatmap
              </TabsTrigger>
              <TabsTrigger value="families" data-testid="tab-families">
                <BarChart3 className="h-4 w-4 mr-2" />
                Family Analysis
              </TabsTrigger>
            </TabsList>

            <TabsContent value="density">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Score Distribution Density</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Aggregated view of {formatNumber(totalVariants)} variants - density shows concentration of results
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={densityData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis 
                          dataKey="bin" 
                          className="text-xs"
                          label={{ value: 'Score Percentile', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          className="text-xs"
                          label={{ value: 'Variant Count', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px',
                          }}
                          formatter={(value: number) => [formatNumber(value), 'Variants']}
                        />
                        <Area
                          type="monotone"
                          dataKey="density"
                          stroke="hsl(var(--primary))"
                          fill="hsl(var(--primary))"
                          fillOpacity={0.3}
                          name="All Variants"
                        />
                        <Area
                          type="monotone"
                          dataKey="topPerformers"
                          stroke="hsl(142, 76%, 36%)"
                          fill="hsl(142, 76%, 36%)"
                          fillOpacity={0.5}
                          name="Top Performers"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 flex items-center justify-center gap-6 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm bg-primary/50" />
                      <span>All Variants</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(142, 76%, 36%)' }} />
                      <span>Top 10% Performers</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="percentiles">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Performance Envelope by Parameter</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Percentile curves showing performance distribution across chain length parameter
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis 
                          dataKey="parameter" 
                          className="text-xs"
                          label={{ value: 'Chain Length', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          className="text-xs"
                          domain={[0.3, 1]}
                          label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px',
                          }}
                          formatter={(value: number, name: string) => [
                            value.toFixed(3),
                            name === 'p50' ? 'Median' : name === 'p75' ? '75th Percentile' : '95th Percentile'
                          ]}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="p95"
                          stroke="hsl(142, 76%, 36%)"
                          strokeWidth={2}
                          dot={false}
                          name="P95 (Top 5%)"
                        />
                        <Line
                          type="monotone"
                          dataKey="p75"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                          name="P75"
                        />
                        <Line
                          type="monotone"
                          dataKey="p50"
                          stroke="hsl(var(--muted-foreground))"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                          name="Median"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 p-3 rounded-md bg-muted/50 text-sm">
                    <span className="font-medium">Insight:</span> Performance plateau detected at chain length 70-80. 
                    Top 5% performers ({formatNumber(topPerformersCount)} variants) concentrate in this range.
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="heatmap">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Binned Property Scatter</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    {propertyX.replace(/_/g, ' ')} vs {propertyY.replace(/_/g, ' ')} - color intensity shows variant density
                  </p>
                </CardHeader>
                <CardContent>
                  <DensityHeatmap 
                    data={binnedScatterData} 
                    xLabel={propertyX.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    yLabel={propertyY.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    onCellClick={(cellIndex) => {
                      const families = GROUP_CONFIGS.family.labels;
                      const family = families[cellIndex % families.length];
                      handleMaterialClick(family, cellIndex);
                    }}
                  />
                  <div className="mt-6 grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-md bg-muted/50">
                      <div className="text-sm font-medium mb-1">High-Density Region</div>
                      <div className="text-xs text-muted-foreground">
                        X: 40-60, Y: 50-70 contains ~{formatNumber(45000)} variants (35%)
                      </div>
                    </div>
                    <div className="p-3 rounded-md bg-green-500/10 border border-green-500/30">
                      <div className="text-sm font-medium mb-1 text-green-700 dark:text-green-400">Optimal Zone</div>
                      <div className="text-xs text-muted-foreground">
                        X: 70-90, Y: 60-80 - highest avg scores (0.82)
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="families">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Performance by {groupConfig.paramLabel}</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Aggregated metrics across {groupData.length} {groupConfig.paramLabel.toLowerCase()}s
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={groupData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis type="number" domain={[0, 1]} className="text-xs" />
                        <YAxis type="category" dataKey="group" className="text-xs" width={100} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px',
                          }}
                          formatter={(value: number, name: string) => [
                            name === 'variantCount' ? formatNumber(value) : value.toFixed(3),
                            name === 'avgScore' ? 'Avg Score' : name === 'p95Score' ? 'P95 Score' : 'Variant Count'
                          ]}
                        />
                        <Legend />
                        <Bar dataKey="avgScore" fill="hsl(var(--primary))" name="Avg Score" />
                        <Bar dataKey="p95Score" fill="hsl(142, 76%, 36%)" name="P95 Score" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 font-medium">{groupConfig.paramLabel}</th>
                          <th className="text-right py-2 font-medium">Variants</th>
                          <th className="text-right py-2 font-medium">Top Performers</th>
                          <th className="text-right py-2 font-medium">Avg Score</th>
                          <th className="text-right py-2 font-medium">P95 Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {groupData.sort((a, b) => b.p95Score - a.p95Score).map((row, idx) => (
                          <tr 
                            key={row.group} 
                            className="border-b border-muted hover-elevate cursor-pointer"
                            onClick={() => handleMaterialClick(row.group, idx)}
                            data-testid={`family-row-${row.group.toLowerCase().replace(/\s+/g, '-')}`}
                          >
                            <td className="py-2 font-medium">{row.group}</td>
                            <td className="text-right py-2 font-mono">{formatNumber(row.variantCount)}</td>
                            <td className="text-right py-2 font-mono text-green-600 dark:text-green-400">
                              {formatNumber(row.topPerformers)}
                            </td>
                            <td className="text-right py-2 font-mono">{row.avgScore.toFixed(3)}</td>
                            <td className="text-right py-2 font-mono font-medium">{row.p95Score.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>

      {/* Material Detail Sheet Panel */}
      <Sheet open={materialPanelOpen} onOpenChange={setMaterialPanelOpen}>
        <SheetContent className="w-full sm:max-w-lg p-0" side="right">
          <MaterialDetailPanel
            material={selectedMaterial}
            onClose={() => setMaterialPanelOpen(false)}
            onSelectMaterial={handleSelectMaterial}
          />
        </SheetContent>
      </Sheet>
    </div>
  );
}
