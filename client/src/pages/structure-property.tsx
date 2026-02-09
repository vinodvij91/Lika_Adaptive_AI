import { useState, useMemo, useEffect } from "react";
import { useLocation } from "wouter";
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
  ChevronLeft,
  ArrowRight,
  Table,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  MaterialDetailSheet,
  generateMaterialDetail,
  generateMaterialsForFamily,
  seededRandom,
  type MaterialDetail,
} from "@/components/material-detail-panel";

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
  material_type: {
    labels: ["Polymer", "Crystal", "Metal", "Ceramic", "Composite", "Semiconductor", "Perovskite", "2D Material"],
    paramLabel: "Material Type"
  },
  crystal_system: {
    labels: ["Cubic", "Tetragonal", "Orthorhombic", "Hexagonal", "Monoclinic", "Triclinic", "Trigonal"],
    paramLabel: "Crystal System"
  },
  bonding_type: {
    labels: ["Covalent", "Ionic", "Metallic", "Van der Waals", "Hydrogen Bond", "Mixed"],
    paramLabel: "Bonding Type"
  },
  synthesis_method: {
    labels: ["Solution", "Melt", "Vapor Deposition", "Sol-Gel", "Electrochemical", "Sintering", "Extrusion", "Spin Coating"],
    paramLabel: "Synthesis Method"
  },
  application_domain: {
    labels: ["Structural", "Electronic", "Optical", "Energy Storage", "Catalysis", "Biomedical", "Thermal Management", "Sensing"],
    paramLabel: "Application Domain"
  },
  molecular_weight_bin: {
    labels: ["<1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", "100K-500K", "500K+"],
    paramLabel: "Molecular Weight Range"
  },
  bandgap_range: {
    labels: ["0-0.5 eV", "0.5-1.0 eV", "1.0-1.5 eV", "1.5-2.0 eV", "2.0-3.0 eV", "3.0-5.0 eV", "5.0+ eV"],
    paramLabel: "Bandgap Range"
  },
  density_bin: {
    labels: ["<1.0", "1.0-2.0", "2.0-4.0", "4.0-6.0", "6.0-8.0", "8.0-12.0", "12.0+"],
    paramLabel: "Density Range (g/cm3)"
  },
  processing_temp: {
    labels: ["<100C", "100-300C", "300-600C", "600-1000C", "1000-1500C", "1500-2000C", "2000C+"],
    paramLabel: "Processing Temperature"
  },
};

const ALL_PROPERTIES: Record<string, string> = {
  thermal_stability: "Thermal Stability",
  tensile_strength: "Tensile Strength",
  conductivity: "Conductivity",
  flexibility: "Flexibility",
  youngs_modulus: "Young's Modulus",
  hardness: "Hardness",
  density: "Density",
  bandgap: "Bandgap",
  thermal_conductivity: "Thermal Conductivity",
  glass_transition: "Glass Transition (Tg)",
  melting_point: "Melting Point",
  dielectric_constant: "Dielectric Constant",
  refractive_index: "Refractive Index",
  corrosion_resistance: "Corrosion Resistance",
  fatigue_life: "Fatigue Life",
  fracture_toughness: "Fracture Toughness",
  ionic_conductivity: "Ionic Conductivity",
  magnetization: "Magnetization",
  poisson_ratio: "Poisson's Ratio",
  specific_heat: "Specific Heat",
  surface_energy: "Surface Energy",
  oxidation_resistance: "Oxidation Resistance",
};

const PROPERTY_OFFSETS: Record<string, number> = {
  thermal_stability: 0,
  tensile_strength: 0.1,
  conductivity: 0.2,
  flexibility: 0.15,
  youngs_modulus: 0.05,
  hardness: 0.12,
  density: 0.08,
  bandgap: 0.18,
  thermal_conductivity: 0.03,
  glass_transition: 0.07,
  melting_point: 0.13,
  dielectric_constant: 0.16,
  refractive_index: 0.09,
  corrosion_resistance: 0.11,
  fatigue_life: 0.14,
  fracture_toughness: 0.06,
  ionic_conductivity: 0.19,
  magnetization: 0.04,
  poisson_ratio: 0.17,
  specific_heat: 0.02,
  surface_energy: 0.1,
  oxidation_resistance: 0.08,
};

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
  const gridSize = 8;
  const xLabels = ["Low", "", "", "Med", "", "", "", "High"];
  const yLabels = ["High", "", "", "Med", "", "", "", "Low"];
  
  return (
    <div className="space-y-3">
      <div className="flex items-stretch gap-2">
        <div className="flex items-center justify-center w-6">
          <span className="text-xs text-muted-foreground font-medium -rotate-90 whitespace-nowrap">
            {yLabel}
          </span>
        </div>
        
        <div className="flex flex-col justify-between text-xs text-muted-foreground py-1" style={{ width: '32px' }}>
          {yLabels.map((label, i) => (
            <span key={i} className="text-right">{label}</span>
          ))}
        </div>
        
        <div className="flex-1">
          <div 
            className="grid gap-[2px]"
            style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)`, gridTemplateRows: `repeat(${gridSize}, 1fr)` }}
          >
            {Array.from({ length: gridSize * gridSize }).map((_, idx) => {
              const dataIdx = idx % data.length;
              const item = data[dataIdx] || { count: 0, avgScore: 0 };
              const intensity = Math.min(item.count / maxCount, 1);
              const hue = 142;
              const alpha = Math.max(0.05, intensity * 0.8);
              
              return (
                <div
                  key={idx}
                  className="aspect-square rounded-sm cursor-pointer transition-transform"
                  style={{ 
                    backgroundColor: `hsla(${hue}, 76%, 36%, ${alpha})`,
                    minHeight: '20px',
                  }}
                  onClick={() => onCellClick?.(idx)}
                  title={`Count: ${formatNumber(item.count)} | Avg Score: ${item.avgScore?.toFixed(3) || 'N/A'}`}
                  data-testid={`heatmap-cell-${idx}`}
                />
              );
            })}
          </div>
          
          <div className="flex justify-between text-xs text-muted-foreground mt-1 px-1">
            {xLabels.map((label, i) => (
              <span key={i}>{label}</span>
            ))}
          </div>
          <div className="text-center text-xs text-muted-foreground mt-1 font-medium">{xLabel}</div>
        </div>
      </div>
      
      <div className="flex items-center gap-2 text-xs text-muted-foreground justify-center">
        <span>Low density</span>
        <div className="flex gap-px">
          {[0.1, 0.3, 0.5, 0.7, 0.9].map((alpha, i) => (
            <div
              key={i}
              className="w-5 h-3 rounded-sm"
              style={{ backgroundColor: `hsla(142, 76%, 36%, ${alpha})` }}
            />
          ))}
        </div>
        <span>High density</span>
      </div>
    </div>
  );
}

interface GroupDrillDownProps {
  groupLabel: string;
  groupBy: string;
  seed: number;
  onClose: () => void;
  onSelectVariant: (material: MaterialDetail) => void;
}

function GroupDrillDown({ groupLabel, groupBy, seed, onClose, onSelectVariant }: GroupDrillDownProps) {
  const familyName = groupBy === "family" ? groupLabel : "Polyethylene";
  const variants = useMemo(() => generateMaterialsForFamily(familyName, 20, seed), [familyName, seed]);

  const getTierBadge = (feasibility: number) => {
    if (feasibility >= 0.7) return <Badge variant="outline" className="bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30 text-xs">Production-Viable</Badge>;
    if (feasibility >= 0.4) return <Badge variant="outline" className="bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30 text-xs">Pilot-Ready</Badge>;
    return <Badge variant="outline" className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30 text-xs">Lab-Only</Badge>;
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onClose} data-testid="button-close-drilldown">
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <div>
            <h2 className="text-lg font-semibold">{groupLabel} Variants</h2>
            <p className="text-sm text-muted-foreground">{variants.length} variants in this {GROUP_CONFIGS[groupBy]?.paramLabel?.toLowerCase() || "group"}</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} data-testid="button-close-drilldown-x">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 font-medium">Variant</th>
                <th className="text-right py-2 font-medium">Score</th>
                <th className="text-right py-2 font-medium">Thermal</th>
                <th className="text-right py-2 font-medium">Tensile</th>
                <th className="text-right py-2 font-medium">Tier</th>
              </tr>
            </thead>
            <tbody>
              {variants.map((v) => (
                <tr
                  key={v.id}
                  className="border-b border-muted hover-elevate cursor-pointer"
                  onClick={() => onSelectVariant(v)}
                  data-testid={`variant-row-${v.id}`}
                >
                  <td className="py-2">
                    <div className="font-medium">{v.name}</div>
                    <div className="text-xs text-muted-foreground font-mono">{v.id}</div>
                  </td>
                  <td className="text-right py-2 font-mono font-medium">{(v.overallScore * 100).toFixed(1)}%</td>
                  <td className="text-right py-2 font-mono">{v.properties.thermalStability.toFixed(0)}&deg;C</td>
                  <td className="text-right py-2 font-mono">{v.properties.tensileStrength.toFixed(0)} MPa</td>
                  <td className="text-right py-2">{getTierBadge(v.synthesis.feasibility)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ScrollArea>
    </div>
  );
}

export default function StructurePropertyPage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const [groupBy, setGroupBy] = useState<string>("family");
  const [propertyX, setPropertyX] = useState<string>("thermal_stability");
  const [propertyY, setPropertyY] = useState<string>("tensile_strength");
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false);
  const [analysisSeed, setAnalysisSeed] = useState(0);
  const [selectedMaterial, setSelectedMaterial] = useState<MaterialDetail | null>(null);
  const [materialPanelOpen, setMaterialPanelOpen] = useState(false);
  const [drillDownGroup, setDrillDownGroup] = useState<string | null>(null);
  const [drillDownOpen, setDrillDownOpen] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const family = params.get("family");
    if (family) {
      setGroupBy("family");
      setDrillDownGroup(family);
      setDrillDownOpen(true);
    }
  }, []);

  const handleRunAnalysis = () => {
    setIsRunningAnalysis(true);
    toast({
      title: "Running Analysis",
      description: "Processing 127K+ material variants...",
    });
    setTimeout(() => {
      setAnalysisSeed(prev => prev + 1);
      setIsRunningAnalysis(false);
      toast({
        title: "Analysis Complete",
        description: "Updated structure-property correlations with latest data.",
      });
    }, 1500);
  };

  const handleGroupClick = (groupLabel: string) => {
    setDrillDownGroup(groupLabel);
    setDrillDownOpen(true);
  };

  const handleVariantClick = (material: MaterialDetail) => {
    setSelectedMaterial(material);
    setMaterialPanelOpen(true);
  };

  const handleMaterialClick = (family: string, index: number) => {
    const material = generateMaterialDetail(family, index, seed);
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
                    <SelectContent className="max-h-72">
                      <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">Structural</div>
                      <SelectItem value="family">Material Family</SelectItem>
                      <SelectItem value="scaffold">Scaffold / Lattice</SelectItem>
                      <SelectItem value="material_type">Material Type</SelectItem>
                      <SelectItem value="crystal_system">Crystal System</SelectItem>
                      <SelectItem value="bonding_type">Bonding Type</SelectItem>
                      <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">Quantitative Bins</div>
                      <SelectItem value="chain_length">Chain Length Bins</SelectItem>
                      <SelectItem value="molecular_weight_bin">Molecular Weight Range</SelectItem>
                      <SelectItem value="bandgap_range">Bandgap Range</SelectItem>
                      <SelectItem value="density_bin">Density Range</SelectItem>
                      <SelectItem value="processing_temp">Processing Temperature</SelectItem>
                      <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">Functional</div>
                      <SelectItem value="dopant">Dopant Type</SelectItem>
                      <SelectItem value="synthesis_method">Synthesis Method</SelectItem>
                      <SelectItem value="application_domain">Application Domain</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>X-Axis Property</Label>
                  <Select value={propertyX} onValueChange={setPropertyX}>
                    <SelectTrigger data-testid="select-property-x">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="max-h-72">
                      {Object.entries(ALL_PROPERTIES).map(([key, label]) => (
                        <SelectItem key={key} value={key}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Y-Axis Property</Label>
                  <Select value={propertyY} onValueChange={setPropertyY}>
                    <SelectTrigger data-testid="select-property-y">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="max-h-72">
                      {Object.entries(ALL_PROPERTIES).map(([key, label]) => (
                        <SelectItem key={key} value={key}>{label}</SelectItem>
                      ))}
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
                    {propertyX.replace(/_/g, ' ')} vs {propertyY.replace(/_/g, ' ')} - click any cell to drill down to underlying variants
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
                      handleGroupClick(family);
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
                    Click any row to see underlying variants for that {groupConfig.paramLabel.toLowerCase()}
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
                          <th className="text-right py-2 font-medium"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {groupData.sort((a, b) => b.p95Score - a.p95Score).map((row, idx) => (
                          <tr 
                            key={row.group} 
                            className="border-b border-muted hover-elevate cursor-pointer"
                            onClick={() => handleGroupClick(row.group)}
                            data-testid={`family-row-${row.group.toLowerCase().replace(/\s+/g, '-')}`}
                          >
                            <td className="py-2 font-medium">{row.group}</td>
                            <td className="text-right py-2 font-mono">{formatNumber(row.variantCount)}</td>
                            <td className="text-right py-2 font-mono text-green-600 dark:text-green-400">
                              {formatNumber(row.topPerformers)}
                            </td>
                            <td className="text-right py-2 font-mono">{row.avgScore.toFixed(3)}</td>
                            <td className="text-right py-2 font-mono font-medium">{row.p95Score.toFixed(3)}</td>
                            <td className="text-right py-2">
                              <Badge variant="outline" className="text-xs">
                                <Table className="h-3 w-3 mr-1" />
                                View
                              </Badge>
                            </td>
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

      <Sheet open={drillDownOpen} onOpenChange={setDrillDownOpen}>
        <SheetContent className="w-full sm:max-w-2xl p-0" side="right">
          {drillDownGroup && (
            <GroupDrillDown
              groupLabel={drillDownGroup}
              groupBy={groupBy}
              seed={seed}
              onClose={() => setDrillDownOpen(false)}
              onSelectVariant={(material) => {
                setDrillDownOpen(false);
                handleVariantClick(material);
              }}
            />
          )}
        </SheetContent>
      </Sheet>

      <MaterialDetailSheet
        material={selectedMaterial}
        open={materialPanelOpen}
        onOpenChange={setMaterialPanelOpen}
        onSelectMaterial={(m) => setSelectedMaterial(m)}
      />
    </div>
  );
}
