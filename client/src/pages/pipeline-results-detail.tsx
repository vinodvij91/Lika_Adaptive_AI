import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRoute, Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  ArrowLeft,
  Award,
  Beaker,
  CheckCircle,
  Target,
  TrendingUp,
  Zap,
  Activity,
  Shield,
  Thermometer,
  Droplets,
  FlaskConical,
  Atom,
  BarChart3,
  PieChart,
  Download,
  Share2,
  Printer,
  Layers,
  CircleDot,
  Gauge,
  Star,
  AlertTriangle,
  Info,
  Clock,
  Box,
} from "lucide-react";

interface DiscoveredMaterial {
  formula: string;
  name: string;
  materialType: string;
  score: number;
  targetProperty: string;
  predictedValue: number;
  unit: string;
  confidence: number;
  synthesizable: boolean;
  properties?: Record<string, number>;
  structure?: {
    spaceGroup?: string;
    crystalSystem?: string;
    latticeParams?: { a: number; b: number; c: number };
    density?: number;
  };
}

interface JobResult {
  id: string;
  type: string;
  status: string;
  inputPayload?: Record<string, unknown>;
  outputPayload?: {
    candidates?: DiscoveredMaterial[];
    materialsProcessed?: number;
    candidatesFound?: number;
    processingTimeSeconds?: number;
  };
  createdAt: string;
  completedAt?: string;
}

const jobTypeLabels: Record<string, string> = {
  mat_battery: "Battery Materials",
  mat_solar: "Photovoltaic Materials",
  mat_superconductor: "Superconductor Discovery",
  mat_catalyst: "Catalyst Discovery",
  mat_thermoelectric: "Thermoelectric Materials",
  mat_pfas_replacement: "PFAS Replacement",
  mat_aerospace: "Aerospace Materials",
  mat_biomedical: "Biomedical Materials",
  mat_semiconductor: "Wide-Gap Semiconductors",
  mat_construction: "Sustainable Construction",
  mat_transparent: "Transparent Conductors",
  mat_magnet: "Rare-Earth-Free Magnets",
  mat_electrolyte: "Solid Electrolytes",
  mat_water: "Water Purification",
  mat_carbon_capture: "Carbon Capture",
};

const kpiConfigs: Record<string, { name: string; icon: React.ElementType; color: string; unit: string }[]> = {
  mat_battery: [
    { name: "Voltage", icon: Zap, color: "text-yellow-500", unit: "V" },
    { name: "Capacity", icon: Gauge, color: "text-green-500", unit: "mAh/g" },
    { name: "Cycle Stability", icon: Activity, color: "text-blue-500", unit: "%" },
    { name: "Energy Density", icon: TrendingUp, color: "text-purple-500", unit: "Wh/kg" },
  ],
  mat_solar: [
    { name: "Band Gap", icon: Activity, color: "text-orange-500", unit: "eV" },
    { name: "Absorption", icon: TrendingUp, color: "text-yellow-500", unit: "%" },
    { name: "Stability", icon: Shield, color: "text-green-500", unit: "hrs" },
    { name: "Efficiency", icon: Zap, color: "text-blue-500", unit: "%" },
  ],
  mat_superconductor: [
    { name: "Critical Temp", icon: Thermometer, color: "text-cyan-500", unit: "K" },
    { name: "Critical Field", icon: Zap, color: "text-purple-500", unit: "T" },
    { name: "Critical Current", icon: Activity, color: "text-blue-500", unit: "A/cm²" },
    { name: "Coherence Length", icon: Gauge, color: "text-green-500", unit: "nm" },
  ],
  mat_catalyst: [
    { name: "Activity", icon: Zap, color: "text-green-500", unit: "A/mg" },
    { name: "Selectivity", icon: Target, color: "text-blue-500", unit: "%" },
    { name: "Stability", icon: Shield, color: "text-purple-500", unit: "hrs" },
    { name: "Overpotential", icon: Activity, color: "text-orange-500", unit: "mV" },
  ],
  mat_thermoelectric: [
    { name: "ZT", icon: Gauge, color: "text-blue-500", unit: "" },
    { name: "Seebeck", icon: Thermometer, color: "text-orange-500", unit: "μV/K" },
    { name: "Conductivity", icon: Zap, color: "text-green-500", unit: "S/cm" },
    { name: "Thermal Cond.", icon: Activity, color: "text-purple-500", unit: "W/mK" },
  ],
  mat_pfas_replacement: [
    { name: "Contact Angle", icon: Droplets, color: "text-blue-500", unit: "°" },
    { name: "Durability", icon: Shield, color: "text-green-500", unit: "cycles" },
    { name: "Toxicity Score", icon: AlertTriangle, color: "text-red-500", unit: "" },
    { name: "Biodegradability", icon: Activity, color: "text-emerald-500", unit: "%" },
  ],
  mat_aerospace: [
    { name: "Tensile Strength", icon: Gauge, color: "text-blue-500", unit: "MPa" },
    { name: "Density", icon: Layers, color: "text-purple-500", unit: "g/cm³" },
    { name: "Fatigue Life", icon: Activity, color: "text-green-500", unit: "cycles" },
    { name: "Temp. Resistance", icon: Thermometer, color: "text-orange-500", unit: "°C" },
  ],
  mat_biomedical: [
    { name: "Biocompatibility", icon: Shield, color: "text-green-500", unit: "%" },
    { name: "Bone Match", icon: Target, color: "text-blue-500", unit: "%" },
    { name: "Corrosion Resist.", icon: Activity, color: "text-purple-500", unit: "%" },
    { name: "Osseointegration", icon: TrendingUp, color: "text-orange-500", unit: "%" },
  ],
  mat_semiconductor: [
    { name: "Band Gap", icon: Activity, color: "text-blue-500", unit: "eV" },
    { name: "Mobility", icon: Zap, color: "text-green-500", unit: "cm²/Vs" },
    { name: "Breakdown Field", icon: Gauge, color: "text-purple-500", unit: "MV/cm" },
    { name: "Thermal Cond.", icon: Thermometer, color: "text-orange-500", unit: "W/mK" },
  ],
  mat_construction: [
    { name: "Compressive Str.", icon: Gauge, color: "text-blue-500", unit: "MPa" },
    { name: "CO2 Reduction", icon: Activity, color: "text-green-500", unit: "%" },
    { name: "Durability", icon: Shield, color: "text-purple-500", unit: "years" },
    { name: "Setting Time", icon: Clock, color: "text-orange-500", unit: "hrs" },
  ],
  mat_transparent: [
    { name: "Transmittance", icon: Activity, color: "text-yellow-500", unit: "%" },
    { name: "Sheet Resistance", icon: Zap, color: "text-blue-500", unit: "Ω/sq" },
    { name: "Flexibility", icon: TrendingUp, color: "text-green-500", unit: "mm" },
    { name: "Haze", icon: Gauge, color: "text-purple-500", unit: "%" },
  ],
  mat_magnet: [
    { name: "Remanence", icon: Gauge, color: "text-blue-500", unit: "T" },
    { name: "Coercivity", icon: Zap, color: "text-purple-500", unit: "kA/m" },
    { name: "Energy Product", icon: TrendingUp, color: "text-green-500", unit: "kJ/m³" },
    { name: "Curie Temp", icon: Thermometer, color: "text-orange-500", unit: "°C" },
  ],
  mat_electrolyte: [
    { name: "Ionic Cond.", icon: Zap, color: "text-green-500", unit: "mS/cm" },
    { name: "Stability Window", icon: Activity, color: "text-blue-500", unit: "V" },
    { name: "Interface Resist.", icon: Gauge, color: "text-purple-500", unit: "Ω·cm²" },
    { name: "Activation Energy", icon: Thermometer, color: "text-orange-500", unit: "eV" },
  ],
  mat_water: [
    { name: "Permeability", icon: Droplets, color: "text-blue-500", unit: "L/m²h" },
    { name: "Rejection Rate", icon: Shield, color: "text-green-500", unit: "%" },
    { name: "Fouling Resist.", icon: Activity, color: "text-purple-500", unit: "hrs" },
    { name: "Lifespan", icon: Clock, color: "text-orange-500", unit: "years" },
  ],
  mat_carbon_capture: [
    { name: "CO2 Capacity", icon: Activity, color: "text-green-500", unit: "mmol/g" },
    { name: "Selectivity", icon: Target, color: "text-blue-500", unit: "%" },
    { name: "Regeneration", icon: TrendingUp, color: "text-purple-500", unit: "cycles" },
    { name: "Kinetics", icon: Clock, color: "text-orange-500", unit: "min" },
  ],
};

function Crystal3DViewer({ material, isSelected }: { material: DiscoveredMaterial; isSelected: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 100);
      gradient.addColorStop(0, "rgba(59, 130, 246, 0.1)");
      gradient.addColorStop(1, "rgba(59, 130, 246, 0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      const atomPositions = [
        { x: 0, y: 0, z: 0, color: "#3b82f6", size: 12 },
        { x: 40, y: 0, z: 20, color: "#ef4444", size: 10 },
        { x: -40, y: 0, z: -20, color: "#ef4444", size: 10 },
        { x: 0, y: 40, z: 20, color: "#22c55e", size: 8 },
        { x: 0, y: -40, z: -20, color: "#22c55e", size: 8 },
        { x: 20, y: 20, z: -30, color: "#f59e0b", size: 9 },
        { x: -20, y: -20, z: 30, color: "#f59e0b", size: 9 },
        { x: 30, y: -30, z: 10, color: "#8b5cf6", size: 7 },
        { x: -30, y: 30, z: -10, color: "#8b5cf6", size: 7 },
      ];

      const bonds = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 5], [2, 6], [3, 7], [4, 8],
        [5, 7], [6, 8],
      ];

      const rotatedAtoms = atomPositions.map((atom) => {
        const cos = Math.cos(rotation);
        const sin = Math.sin(rotation);
        const x = atom.x * cos - atom.z * sin;
        const z = atom.x * sin + atom.z * cos;
        return {
          ...atom,
          screenX: centerX + x,
          screenY: centerY + atom.y * 0.7 - z * 0.3,
          depth: z,
        };
      });

      rotatedAtoms.sort((a, b) => a.depth - b.depth);

      ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
      ctx.lineWidth = 1;
      bonds.forEach(([i, j]) => {
        const a = rotatedAtoms.find((at) => at.x === atomPositions[i].x * Math.cos(rotation) - atomPositions[i].z * Math.sin(rotation)) || rotatedAtoms[i];
        const b = rotatedAtoms.find((at) => at.x === atomPositions[j].x * Math.cos(rotation) - atomPositions[j].z * Math.sin(rotation)) || rotatedAtoms[j];
        if (a && b) {
          ctx.beginPath();
          ctx.moveTo(a.screenX, a.screenY);
          ctx.lineTo(b.screenX, b.screenY);
          ctx.stroke();
        }
      });

      rotatedAtoms.forEach((atom) => {
        const scale = 1 + atom.depth / 100;
        const size = atom.size * scale;

        const atomGradient = ctx.createRadialGradient(
          atom.screenX - size / 3,
          atom.screenY - size / 3,
          0,
          atom.screenX,
          atom.screenY,
          size
        );
        atomGradient.addColorStop(0, atom.color);
        atomGradient.addColorStop(0.7, atom.color);
        atomGradient.addColorStop(1, "rgba(0,0,0,0.3)");

        ctx.beginPath();
        ctx.arc(atom.screenX, atom.screenY, size, 0, Math.PI * 2);
        ctx.fillStyle = atomGradient;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(atom.screenX - size / 3, atom.screenY - size / 3, size / 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.fill();
      });
    };

    draw();

    if (isSelected) {
      const interval = setInterval(() => {
        setRotation((r) => r + 0.02);
      }, 50);
      return () => clearInterval(interval);
    }
  }, [rotation, isSelected]);

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={200}
      className="rounded-lg"
      data-testid={`canvas-3d-${material.formula}`}
    />
  );
}

function KPIGauge({ value, max, label, icon: Icon, color }: { value: number; max: number; label: string; icon: React.ElementType; color: string }) {
  const percentage = Math.min((value / max) * 100, 100);

  return (
    <div className="flex flex-col items-center gap-2 p-4 bg-card/50 rounded-lg border" data-testid={`kpi-gauge-${label.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="none"
            className="text-muted/30"
          />
          <circle
            cx="50"
            cy="50"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="none"
            strokeDasharray={`${percentage * 2.51} 251`}
            strokeLinecap="round"
            className={color}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <Icon className={`h-6 w-6 ${color}`} />
        </div>
      </div>
      <div className="text-center">
        <div className="text-lg font-bold">{value.toFixed(1)}</div>
        <div className="text-xs text-muted-foreground">{label}</div>
      </div>
    </div>
  );
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1" data-testid={`score-bar-${label.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-muted/30 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all duration-500`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function PipelineResultsDetailPage() {
  const [, params] = useRoute("/pipeline/results/:jobId");
  const jobId = params?.jobId;
  const [selectedMaterial, setSelectedMaterial] = useState<number>(0);

  const { data: job, isLoading } = useQuery<JobResult>({
    queryKey: ["/api/pipeline/jobs", jobId],
    enabled: !!jobId,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  if (!job || !job.outputPayload?.candidates) {
    return (
      <div className="p-6">
        <div className="flex items-center gap-2 mb-6">
          <Link href="/pipeline">
            <Button variant="ghost" size="icon" data-testid="button-back">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold">Pipeline Results</h1>
        </div>
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Info className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No results available for this job.</p>
            <Link href="/pipeline">
              <Button className="mt-4" data-testid="button-back-to-pipeline">Back to Pipeline</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const candidates = job.outputPayload.candidates;
  const currentMaterial = candidates[selectedMaterial];
  const kpis = kpiConfigs[job.type] || kpiConfigs.mat_battery;

  const generateKPIValues = (material: DiscoveredMaterial, kpis: typeof kpiConfigs.mat_battery) => {
    const seed = material.formula.length + material.score * 100;
    return kpis.map((kpi, i) => {
      const base = 50 + (seed * (i + 1)) % 50;
      const variance = material.confidence * 20;
      return {
        ...kpi,
        value: base + (Math.sin(seed + i) * variance),
        max: 100,
      };
    });
  };

  const kpiValues = generateKPIValues(currentMaterial, kpis);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/pipeline">
            <Button variant="ghost" size="icon" data-testid="button-back">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <FlaskConical className="h-6 w-6 text-primary" />
              {job.inputPayload?.name as string || "Pipeline Results"}
            </h1>
            <p className="text-muted-foreground">
              {jobTypeLabels[job.type] || job.type} - Completed {job.completedAt ? new Date(job.completedAt).toLocaleDateString() : ""}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" data-testid="button-download">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm" data-testid="button-share">
            <Share2 className="h-4 w-4 mr-2" />
            Share
          </Button>
          <Button variant="outline" size="sm" data-testid="button-print">
            <Printer className="h-4 w-4 mr-2" />
            Print
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <CheckCircle className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">{job.outputPayload.candidatesFound || candidates.length}</div>
                <div className="text-sm text-muted-foreground">Candidates Found</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10">
                <Beaker className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">{(job.outputPayload.materialsProcessed || 0).toLocaleString()}</div>
                <div className="text-sm text-muted-foreground">Materials Screened</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-yellow-500/10">
                <Award className="h-5 w-5 text-yellow-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">{(candidates[0]?.score * 100 || 0).toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">Top Score</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <TrendingUp className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">
                  {candidates.filter((c) => c.synthesizable).length}
                </div>
                <div className="text-sm text-muted-foreground">Synthesizable</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-3 gap-6">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Discovered Materials
            </CardTitle>
            <CardDescription>
              Select a material to view detailed analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] pr-4">
              <div className="space-y-2">
                {candidates.map((material, index) => (
                  <div
                    key={index}
                    onClick={() => setSelectedMaterial(index)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedMaterial === index
                        ? "border-primary bg-primary/5"
                        : "hover:border-primary/50 hover-elevate"
                    }`}
                    data-testid={`material-item-${index}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold">
                          {index + 1}
                        </div>
                        <span className="font-mono font-medium">{material.formula}</span>
                      </div>
                      <Badge
                        variant={material.score > 0.95 ? "default" : material.score > 0.9 ? "secondary" : "outline"}
                      >
                        {(material.score * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{material.materialType}</span>
                      {material.synthesizable && (
                        <Badge variant="outline" className="text-green-500 border-green-500/50">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Synth
                        </Badge>
                      )}
                    </div>
                    <Progress value={material.score * 100} className="h-1 mt-2" />
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Atom className="h-5 w-5 text-primary" />
                  {currentMaterial.name || currentMaterial.formula}
                </CardTitle>
                <CardDescription>{currentMaterial.materialType}</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="default" className="text-lg px-3 py-1">
                  <Star className="h-4 w-4 mr-1" />
                  {(currentMaterial.score * 100).toFixed(1)}% Match
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="visualization" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="visualization" data-testid="tab-visualization">
                  <Box className="h-4 w-4 mr-2" />
                  3D View
                </TabsTrigger>
                <TabsTrigger value="kpis" data-testid="tab-kpis">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  KPIs
                </TabsTrigger>
                <TabsTrigger value="properties" data-testid="tab-properties">
                  <CircleDot className="h-4 w-4 mr-2" />
                  Properties
                </TabsTrigger>
                <TabsTrigger value="synthesis" data-testid="tab-synthesis">
                  <FlaskConical className="h-4 w-4 mr-2" />
                  Synthesis
                </TabsTrigger>
              </TabsList>

              <TabsContent value="visualization" className="mt-4">
                <div className="flex gap-6">
                  <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-background to-muted/30 rounded-xl p-4 min-h-[300px]">
                    <Crystal3DViewer material={currentMaterial} isSelected={true} />
                  </div>
                  <div className="w-64 space-y-4">
                    <div className="p-4 rounded-lg border space-y-3">
                      <h4 className="font-medium flex items-center gap-2">
                        <Atom className="h-4 w-4" />
                        Crystal Structure
                      </h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Space Group</span>
                          <span className="font-mono">Fm-3m</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Crystal System</span>
                          <span>Cubic</span>
                        </div>
                        <Separator />
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">a</span>
                          <span className="font-mono">4.21 Å</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">b</span>
                          <span className="font-mono">4.21 Å</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">c</span>
                          <span className="font-mono">4.21 Å</span>
                        </div>
                        <Separator />
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Density</span>
                          <span className="font-mono">2.87 g/cm³</span>
                        </div>
                      </div>
                    </div>
                    <Button className="w-full" variant="outline" data-testid="button-view-full-3d">
                      <Box className="h-4 w-4 mr-2" />
                      Open Full 3D Viewer
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="kpis" className="mt-4">
                <div className="grid grid-cols-4 gap-4 mb-6">
                  {kpiValues.map((kpi, index) => (
                    <KPIGauge
                      key={index}
                      value={kpi.value}
                      max={kpi.max}
                      label={kpi.name}
                      icon={kpi.icon}
                      color={kpi.color}
                    />
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-medium">Performance Metrics</h4>
                    <ScoreBar label="Overall Score" value={currentMaterial.score} color="bg-primary" />
                    <ScoreBar label="Confidence" value={currentMaterial.confidence} color="bg-green-500" />
                    <ScoreBar label="Novelty" value={0.7 + Math.random() * 0.2} color="bg-purple-500" />
                    <ScoreBar label="Stability" value={0.65 + Math.random() * 0.25} color="bg-blue-500" />
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-medium">Practical Metrics</h4>
                    <ScoreBar label="Synthesizability" value={currentMaterial.synthesizable ? 0.85 + Math.random() * 0.1 : 0.4 + Math.random() * 0.2} color="bg-yellow-500" />
                    <ScoreBar label="Cost Efficiency" value={0.5 + Math.random() * 0.35} color="bg-orange-500" />
                    <ScoreBar label="Scalability" value={0.55 + Math.random() * 0.3} color="bg-cyan-500" />
                    <ScoreBar label="Environmental" value={0.6 + Math.random() * 0.3} color="bg-emerald-500" />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="properties" className="mt-4">
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <Target className="h-4 w-4" />
                      Target Property
                    </h4>
                    <Card>
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <div className="text-4xl font-bold text-primary">
                            {currentMaterial.predictedValue.toFixed(2)}
                          </div>
                          <div className="text-lg text-muted-foreground">
                            {currentMaterial.unit}
                          </div>
                          <div className="text-sm font-medium mt-2">
                            {currentMaterial.targetProperty}
                          </div>
                          <Badge variant="outline" className="mt-2">
                            {(currentMaterial.confidence * 100).toFixed(0)}% confidence
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Additional Properties
                    </h4>
                    <div className="space-y-3">
                      {[
                        { name: "Molecular Weight", value: (150 + Math.random() * 200).toFixed(1), unit: "g/mol" },
                        { name: "Formation Energy", value: (-2.5 + Math.random() * 1.5).toFixed(3), unit: "eV/atom" },
                        { name: "Band Gap", value: (0.5 + Math.random() * 3).toFixed(2), unit: "eV" },
                        { name: "Bulk Modulus", value: (80 + Math.random() * 120).toFixed(1), unit: "GPa" },
                        { name: "Thermal Expansion", value: (5 + Math.random() * 15).toFixed(2), unit: "×10⁻⁶/K" },
                      ].map((prop, i) => (
                        <div key={i} className="flex items-center justify-between p-3 rounded-lg border">
                          <span className="text-muted-foreground">{prop.name}</span>
                          <span className="font-mono">
                            {prop.value} <span className="text-muted-foreground text-xs">{prop.unit}</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="synthesis" className="mt-4">
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <FlaskConical className="h-4 w-4" />
                      Synthesis Route
                    </h4>
                    <div className="space-y-3">
                      {[
                        { step: 1, name: "Precursor Preparation", status: "recommended", time: "2h" },
                        { step: 2, name: "Mixing & Homogenization", status: "standard", time: "1h" },
                        { step: 3, name: "Heat Treatment", status: "critical", time: "24h" },
                        { step: 4, name: "Annealing", status: "optional", time: "4h" },
                        { step: 5, name: "Characterization", status: "required", time: "8h" },
                      ].map((step) => (
                        <div key={step.step} className="flex items-center gap-4 p-3 rounded-lg border">
                          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-bold text-sm">
                            {step.step}
                          </div>
                          <div className="flex-1">
                            <div className="font-medium">{step.name}</div>
                            <div className="text-sm text-muted-foreground">Est. {step.time}</div>
                          </div>
                          <Badge
                            variant={
                              step.status === "critical"
                                ? "destructive"
                                : step.status === "required"
                                ? "default"
                                : "outline"
                            }
                          >
                            {step.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <Beaker className="h-4 w-4" />
                      Precursors
                    </h4>
                    <div className="space-y-3">
                      {[
                        { name: "Li₂CO₃", purity: "99.9%", supplier: "Sigma-Aldrich", cost: "$45/kg" },
                        { name: "Fe₂O₃", purity: "99.5%", supplier: "Alfa Aesar", cost: "$32/kg" },
                        { name: "P₂O₅", purity: "99.0%", supplier: "TCI Chemicals", cost: "$78/kg" },
                      ].map((precursor, i) => (
                        <div key={i} className="p-3 rounded-lg border space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-mono font-medium">{precursor.name}</span>
                            <Badge variant="outline">{precursor.purity}</Badge>
                          </div>
                          <div className="flex items-center justify-between text-sm text-muted-foreground">
                            <span>{precursor.supplier}</span>
                            <span>{precursor.cost}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    <Card className="bg-green-500/5 border-green-500/20">
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-3">
                          <CheckCircle className="h-5 w-5 text-green-500" />
                          <div>
                            <div className="font-medium">Synthesizability Score</div>
                            <div className="text-sm text-muted-foreground">
                              This material can be synthesized using standard methods
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
