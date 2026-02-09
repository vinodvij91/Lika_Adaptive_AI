import { useState, useMemo, useRef } from "react";
import { useLocation } from "wouter";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { PageHeader } from "@/components/page-header";
import { ResultsPanel } from "@/components/results-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import {
  Atom,
  Layers,
  Hexagon,
  Box,
  ArrowRight,
  CheckCircle,
  Activity,
  Zap,
  Settings,
  Play,
  Target,
  BarChart3,
  RefreshCw,
  Database,
  Info,
  Factory,
  ExternalLink,
} from "lucide-react";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

type ScaleLevel = "molecular" | "chain" | "lattice" | "bulk";

interface ScaleRepresentation {
  scale: ScaleLevel;
  title: string;
  description: string;
  icon: typeof Atom;
  descriptorCount: number;
  variantsDerived: number;
  computeTime: string;
  status: "computed" | "computing" | "pending";
  progress?: number;
  descriptors: string[];
  enabled: boolean;
}

const FAMILY_DESCRIPTORS: Record<string, { families: string[]; note: string }> = {
  molecular: {
    families: ["Polymer", "Crystal", "Composite", "Membrane", "Catalyst"],
    note: "Morgan fingerprints, MACCS keys, atom pair descriptors, and functional group counts for each material family.",
  },
  chain: {
    families: ["Polymer", "Composite", "Membrane"],
    note: "Chain length distributions, sequence patterns, persistence length, branching degree, and cross-link density.",
  },
  lattice: {
    families: ["Crystal", "Catalyst"],
    note: "Unit cell parameters, space group encoding, coordination numbers, bond angle distributions, and defect site markers.",
  },
  bulk: {
    families: ["Polymer", "Crystal", "Composite", "Membrane", "Catalyst"],
    note: "Density estimates, thermal conductivity proxy, mechanical modulus indicators, dielectric response features.",
  },
};

function generateMockRepresentations(): ScaleRepresentation[] {
  return [
    {
      scale: "molecular",
      title: "Molecular / Repeat Unit",
      description: "Fingerprints, atom counts, bond topology, functional groups from monomer or repeat unit structure.",
      icon: Atom,
      descriptorCount: 2048,
      variantsDerived: 120000,
      computeTime: "2.3s avg",
      status: "computed",
      descriptors: [
        "Morgan Fingerprints (2048-bit)",
        "MACCS Keys (166-bit)",
        "Atom Pair Fingerprints",
        "Topological Torsions",
        "Functional Group Counts",
        "Ring System Analysis",
      ],
      enabled: true,
    },
    {
      scale: "chain",
      title: "Polymer Chain / Lattice",
      description: "Chain length distributions, sequence patterns, persistence length, crystallinity indicators.",
      icon: Layers,
      descriptorCount: 512,
      variantsDerived: 120000,
      computeTime: "4.7s avg",
      status: "computed",
      descriptors: [
        "Chain Length Distribution",
        "Sequence Pattern Encoding",
        "Persistence Length Estimate",
        "Branching Degree",
        "Tacticity Indicators",
        "Cross-link Density",
      ],
      enabled: true,
    },
    {
      scale: "lattice",
      title: "Crystal / Lattice Structure",
      description: "Unit cell parameters, space group features, coordination environments, defect descriptors.",
      icon: Hexagon,
      descriptorCount: 256,
      variantsDerived: 85000,
      computeTime: "8.2s avg",
      status: "computing",
      progress: 72,
      descriptors: [
        "Unit Cell Parameters",
        "Space Group Encoding",
        "Coordination Numbers",
        "Bond Angle Distribution",
        "Voronoi Tessellation",
        "Defect Site Markers",
      ],
      enabled: true,
    },
    {
      scale: "bulk",
      title: "Bulk / Effective Material",
      description: "Aggregated properties, effective medium approximations, homogenized descriptors for ML models.",
      icon: Box,
      descriptorCount: 128,
      variantsDerived: 120000,
      computeTime: "12.4s avg",
      status: "pending",
      descriptors: [
        "Density Estimates",
        "Thermal Conductivity Proxy",
        "Mechanical Modulus Indicators",
        "Dielectric Response Features",
        "Surface Energy Markers",
        "Porosity Descriptors",
      ],
      enabled: false,
    },
  ];
}

interface ScaleCardProps {
  representation: ScaleRepresentation;
  onToggle: (scale: ScaleLevel, enabled: boolean) => void;
  selectedForPrediction: boolean;
  onSelectForPrediction: () => void;
}

function ScaleCard({ representation, onToggle, selectedForPrediction, onSelectForPrediction }: ScaleCardProps) {
  const Icon = representation.icon;
  const familyInfo = FAMILY_DESCRIPTORS[representation.scale];
  
  const statusColors = {
    computed: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30",
    computing: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30",
    pending: "bg-muted text-muted-foreground border-border",
  };

  const statusLabels = {
    computed: "Computed",
    computing: "Computing...",
    pending: "Pending",
  };

  return (
    <Card 
      className={`relative overflow-hidden transition-all ${selectedForPrediction ? "ring-2 ring-primary" : ""}`}
      data-testid={`card-scale-${representation.scale}`}
    >
      {selectedForPrediction && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-primary" />
      )}
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center border border-primary/30">
              <Icon className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">{representation.title}</CardTitle>
              <Badge variant="outline" className={`mt-1 ${statusColors[representation.status]}`}>
                {representation.status === "computing" && <RefreshCw className="h-3 w-3 mr-1 animate-spin" />}
                {representation.status === "computed" && <CheckCircle className="h-3 w-3 mr-1" />}
                {statusLabels[representation.status]}
              </Badge>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Switch 
              checked={representation.enabled}
              onCheckedChange={(checked) => onToggle(representation.scale, checked)}
              data-testid={`switch-enable-${representation.scale}`}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{representation.description}</p>

        {representation.status === "computing" && representation.progress !== undefined && (
          <div className="space-y-1">
            <Progress value={representation.progress} className="h-2" />
            <div className="text-xs text-muted-foreground text-right">{representation.progress}% complete</div>
          </div>
        )}

        <div className="grid grid-cols-3 gap-3">
          <div className="p-2 rounded-md bg-muted/50 text-center">
            <div className="text-lg font-bold font-mono">{representation.descriptorCount}</div>
            <div className="text-xs text-muted-foreground">Descriptors</div>
          </div>
          <div className="p-2 rounded-md bg-muted/50 text-center">
            <div className="text-lg font-bold font-mono">{formatNumber(representation.variantsDerived)}</div>
            <div className="text-xs text-muted-foreground">Variants</div>
          </div>
          <div className="p-2 rounded-md bg-muted/50 text-center">
            <div className="text-lg font-bold font-mono">{representation.computeTime}</div>
            <div className="text-xs text-muted-foreground">Per Variant</div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Descriptor Types</div>
          <div className="flex flex-wrap gap-1.5">
            {representation.descriptors.slice(0, 4).map((desc, i) => (
              <Badge key={i} variant="secondary" className="text-xs font-normal">
                {desc}
              </Badge>
            ))}
            {representation.descriptors.length > 4 && (
              <Badge variant="outline" className="text-xs font-normal">
                +{representation.descriptors.length - 4} more
              </Badge>
            )}
          </div>
        </div>

        {familyInfo && (
          <div className="p-3 rounded-md bg-primary/5 border border-primary/10">
            <div className="flex items-start gap-2">
              <Info className="h-4 w-4 text-primary mt-0.5 shrink-0" />
              <div>
                <div className="text-xs font-medium mb-1">Feeds into prediction models for:</div>
                <div className="flex flex-wrap gap-1">
                  {familyInfo.families.map((f) => (
                    <Badge key={f} variant="outline" className="text-xs">{f}</Badge>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-1">{familyInfo.note}</p>
              </div>
            </div>
          </div>
        )}

        <div className="pt-2 border-t flex items-center justify-between">
          <div className="text-xs text-muted-foreground flex items-center gap-1">
            <Database className="h-3 w-3" />
            Derived from {formatNumber(representation.variantsDerived)} variants
          </div>
          <Button 
            variant={selectedForPrediction ? "default" : "outline"} 
            size="sm"
            onClick={onSelectForPrediction}
            disabled={representation.status !== "computed"}
            data-testid={`button-select-${representation.scale}`}
          >
            {selectedForPrediction ? (
              <>
                <CheckCircle className="h-3.5 w-3.5 mr-1" />
                Selected
              </>
            ) : (
              <>
                <Target className="h-3.5 w-3.5 mr-1" />
                Use for Prediction
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function PipelineConfig({ selectedScales, onRun, isRunning, progress, result }: { 
  selectedScales: ScaleLevel[];
  onRun: (target: string, model: string) => void;
  isRunning: boolean;
  progress: number;
  result: { variantCount: number; r2Score: number; maeScore: number; target: string } | null;
}) {
  const [predictionTarget, setPredictionTarget] = useState<string>("thermal_stability");
  const [modelArch, setModelArch] = useState<string>("gnn");

  return (
    <Card className="bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 border-primary/20">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Zap className="h-5 w-5 text-primary" />
          Property Prediction Pipeline
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm text-muted-foreground">Selected scales:</span>
          {selectedScales.length === 0 ? (
            <span className="text-sm text-muted-foreground italic">None selected</span>
          ) : (
            selectedScales.map((scale) => (
              <Badge key={scale} className="capitalize">{scale}</Badge>
            ))
          )}
        </div>

        <div className="flex items-center gap-3 p-3 rounded-md bg-background border">
          <div className="flex items-center gap-2">
            {selectedScales.map((scale, i) => (
              <div key={scale} className="flex items-center gap-2">
                <Badge variant="outline" className="capitalize">{scale}</Badge>
                {i < selectedScales.length - 1 && <ArrowRight className="h-4 w-4 text-muted-foreground" />}
              </div>
            ))}
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">ML Property Prediction</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Prediction Target</Label>
            <Select value={predictionTarget} onValueChange={setPredictionTarget}>
              <SelectTrigger data-testid="select-prediction-target">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="thermal_stability">Thermal Stability</SelectItem>
                <SelectItem value="tensile_strength">Tensile Strength</SelectItem>
                <SelectItem value="conductivity">Ionic Conductivity</SelectItem>
                <SelectItem value="permeability">Gas Permeability</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Model Architecture</Label>
            <Select value={modelArch} onValueChange={setModelArch}>
              <SelectTrigger data-testid="select-model-arch">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gnn">Graph Neural Network</SelectItem>
                <SelectItem value="transformer">Transformer</SelectItem>
                <SelectItem value="ensemble">Ensemble (GNN + XGBoost)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {isRunning && (
          <div className="space-y-2 p-3 rounded-md bg-background border">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">
                {progress < 30 ? "Concatenating descriptor matrices..." : 
                 progress < 60 ? "Training prediction model..." :
                 progress < 90 ? "Running inference on variants..." :
                 "Finalizing results..."}
              </span>
              <span className="font-mono font-medium">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {result && !isRunning && (
          <div className="p-3 rounded-md bg-green-500/10 border border-green-500/20">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-green-700 dark:text-green-400">Pipeline Complete</span>
            </div>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <div className="text-lg font-bold font-mono">{formatNumber(result.variantCount)}</div>
                <div className="text-xs text-muted-foreground">Variants Predicted</div>
              </div>
              <div>
                <div className="text-lg font-bold font-mono text-green-600 dark:text-green-400">{result.r2Score.toFixed(3)}</div>
                <div className="text-xs text-muted-foreground">R² Score</div>
              </div>
              <div>
                <div className="text-lg font-bold font-mono">{result.maeScore.toFixed(2)}</div>
                <div className="text-xs text-muted-foreground">MAE</div>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Predicted {result.target.replace(/_/g, ' ')} for {formatNumber(result.variantCount)} variants using {selectedScales.length} scale representation(s).
            </p>
          </div>
        )}

        <div className="flex items-center justify-between pt-2">
          <div className="text-sm text-muted-foreground">
            {isRunning 
              ? "Pipeline is running..."
              : selectedScales.length > 0 
              ? `${selectedScales.length} scale(s) will feed into prediction pipeline`
              : "Select at least one scale to enable predictions"
            }
          </div>
          <Button 
            disabled={selectedScales.length === 0 || isRunning} 
            onClick={() => onRun(predictionTarget, modelArch)}
            data-testid="button-run-prediction"
          >
            {isRunning ? (
              <>
                <div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Running...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Run Prediction Pipeline
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function MultiScaleRepresentationsPage() {
  const [, navigate] = useLocation();
  const [representations, setRepresentations] = useState<ScaleRepresentation[]>(() => generateMockRepresentations());
  const [selectedScales, setSelectedScales] = useState<ScaleLevel[]>(["molecular", "chain"]);
  const [materialType, setMaterialType] = useState<string>("polymer");

  const totalDescriptors = useMemo(() => 
    representations.filter(r => r.enabled && r.status === "computed").reduce((sum, r) => sum + r.descriptorCount, 0),
    [representations]
  );

  const totalVariants = useMemo(() => 
    Math.max(...representations.map(r => r.variantsDerived)),
    [representations]
  );

  const handleToggle = (scale: ScaleLevel, enabled: boolean) => {
    setRepresentations(prev => prev.map(r => 
      r.scale === scale ? { ...r, enabled } : r
    ));
    if (!enabled) {
      setSelectedScales(prev => prev.filter(s => s !== scale));
    }
  };

  const { toast } = useToast();
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [pipelineResult, setPipelineResult] = useState<{ variantCount: number; r2Score: number; maeScore: number; target: string } | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const predictionMutation = useMutation({
    mutationFn: async (params: { target: string; model: string }) => {
      const res = await apiRequest("POST", "/api/compute/materials/multi-scale-predict", {
        predictionTarget: params.target,
        modelArchitecture: params.model,
        materialType,
        selectedScales,
      });
      return res.json();
    },
    onSuccess: (data) => {
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
      setPipelineProgress(100);
      setTimeout(() => {
        setPipelineRunning(false);
        setPipelineResult({
          variantCount: data.variantCount,
          r2Score: data.r2Score,
          maeScore: data.maeScore,
          target: data.target,
        });
        toast({
          title: "Pipeline Complete",
          description: `Predicted ${data.target.replace(/_/g, ' ')} for ${data.variantCount} variants.${data.fallback ? " (CPU fallback mode)" : ""}`,
        });
      }, 300);
    },
    onError: (error: any) => {
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
      setPipelineRunning(false);
      setPipelineProgress(0);
      toast({
        title: "Pipeline Failed",
        description: error.message || "Prediction pipeline encountered an error.",
        variant: "destructive",
      });
    },
  });

  const handleSelectForPrediction = (scale: ScaleLevel) => {
    setSelectedScales(prev => 
      prev.includes(scale) 
        ? prev.filter(s => s !== scale)
        : [...prev, scale]
    );
  };

  const handleRunPipeline = (target: string, model: string) => {
    setPipelineRunning(true);
    setPipelineProgress(0);
    setPipelineResult(null);

    let step = 0;
    progressIntervalRef.current = setInterval(() => {
      step++;
      setPipelineProgress(prev => Math.min(prev + 3 + Math.random() * 2, 90));
    }, 500);

    predictionMutation.mutate({ target, model });
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Multi-Scale Representations" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" data-testid="button-refresh-all">
              <RefreshCw className="h-4 w-4 mr-2" />
              Recompute All
            </Button>
            <Button variant="outline" data-testid="button-settings">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6 rounded-lg border border-primary/20">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/30">
                <Layers className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-1">Multi-Scale Representation Engine</h2>
                <p className="text-muted-foreground">
                  Multi-Scale Representation Engine computes <strong className="text-foreground">molecular</strong>, 
                  <strong className="text-foreground"> chain/lattice</strong>, and <strong className="text-foreground">bulk</strong> descriptors 
                  for each material family. These representations are used by property prediction and manufacturability scoring.
                </p>
              </div>
            </div>
          </div>

          <Card className="border-blue-500/20 bg-blue-500/5">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 shrink-0" />
                <div className="flex-1">
                  <p className="text-sm">
                    Computed descriptors from this engine feed directly into the <strong>Structure-Property Analytics</strong> models 
                    and the <strong>Manufacturability Scoring Engine</strong>. Changes to scale selection or recomputation 
                    will update predictions and scores across both modules.
                  </p>
                  <div className="flex items-center gap-2 mt-3 flex-wrap">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate(`/structure-property?family=${encodeURIComponent(materialType)}`)}
                      data-testid="button-goto-structure-property"
                    >
                      <BarChart3 className="h-3.5 w-3.5 mr-1" />
                      View Structure-Property Analytics
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate("/manufacturability-scoring")}
                      data-testid="button-goto-manufacturability"
                    >
                      <Factory className="h-3.5 w-3.5 mr-1" />
                      View Manufacturability Scores
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono">{representations.length}</div>
                <div className="text-xs text-muted-foreground">Scale Levels</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono">{formatNumber(totalDescriptors)}</div>
                <div className="text-xs text-muted-foreground">Total Descriptors</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">{formatNumber(totalVariants)}</div>
                <div className="text-xs text-muted-foreground">Variants Processed</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono text-primary">{selectedScales.length}</div>
                <div className="text-xs text-muted-foreground">Scales for Prediction</div>
              </CardContent>
            </Card>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Atom className="h-5 w-5" />
                Scale-Level Representations
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <Label className="text-sm text-muted-foreground">Material Type:</Label>
              <Select value={materialType} onValueChange={setMaterialType}>
                <SelectTrigger className="w-40" data-testid="select-material-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="polymer">Polymer</SelectItem>
                  <SelectItem value="crystal">Crystal</SelectItem>
                  <SelectItem value="composite">Composite</SelectItem>
                  <SelectItem value="membrane">Membrane</SelectItem>
                  <SelectItem value="catalyst">Catalyst</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {representations.map((rep) => (
              <ScaleCard
                key={rep.scale}
                representation={rep}
                onToggle={handleToggle}
                selectedForPrediction={selectedScales.includes(rep.scale)}
                onSelectForPrediction={() => handleSelectForPrediction(rep.scale)}
              />
            ))}
          </div>

          <PipelineConfig 
            selectedScales={selectedScales}
            onRun={handleRunPipeline}
            isRunning={pipelineRunning}
            progress={pipelineProgress}
            result={pipelineResult}
          />

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Computation Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                  <div className="flex items-center gap-3">
                    <Atom className="h-5 w-5 text-green-500" />
                    <div>
                      <div className="font-medium">Molecular Fingerprints</div>
                      <div className="text-xs text-muted-foreground">Morgan, MACCS, Atom Pairs</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{formatNumber(120000)} variants</div>
                    <div className="text-xs text-muted-foreground">2.3s avg per variant</div>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                  <div className="flex items-center gap-3">
                    <Layers className="h-5 w-5 text-green-500" />
                    <div>
                      <div className="font-medium">Chain-Level Embeddings</div>
                      <div className="text-xs text-muted-foreground">Sequence, persistence, branching</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{formatNumber(120000)} variants</div>
                    <div className="text-xs text-muted-foreground">4.7s avg per variant</div>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 rounded-md bg-blue-500/10 border border-blue-500/20">
                  <div className="flex items-center gap-3">
                    <Hexagon className="h-5 w-5 text-blue-500" />
                    <div>
                      <div className="font-medium">Lattice Descriptors</div>
                      <div className="text-xs text-muted-foreground">Unit cell, coordination, defects</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">72% complete</div>
                    <div className="text-xs text-muted-foreground">{formatNumber(61200)} / {formatNumber(85000)}</div>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 rounded-md bg-muted/30 opacity-60">
                  <div className="flex items-center gap-3">
                    <Box className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium">Bulk Properties</div>
                      <div className="text-xs text-muted-foreground">Effective medium, homogenized</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm text-muted-foreground">Pending</div>
                    <div className="text-xs text-muted-foreground">Requires chain + lattice</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <ExternalLink className="h-5 w-5" />
                Representation Artifacts
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div
                  className="p-4 rounded-md border hover-elevate cursor-pointer"
                  onClick={() => navigate(`/structure-property?family=${encodeURIComponent(materialType)}`)}
                  data-testid="artifact-property-predictions"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <span className="font-medium text-sm">Property Predictions</span>
                  </div>
                  <p className="text-xs text-muted-foreground">View predicted properties and structure-property correlations for the current dataset.</p>
                  <div className="mt-2 flex items-center gap-1 text-xs text-primary">
                    <ArrowRight className="h-3 w-3" />
                    Open Structure-Property
                  </div>
                </div>
                <div
                  className="p-4 rounded-md border hover-elevate cursor-pointer"
                  onClick={() => navigate("/manufacturability-scoring")}
                  data-testid="artifact-score-distribution"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Factory className="h-5 w-5 text-primary" />
                    <span className="font-medium text-sm">Score Distribution</span>
                  </div>
                  <p className="text-xs text-muted-foreground">View manufacturability scores and production readiness tiers for scored materials.</p>
                  <div className="mt-2 flex items-center gap-1 text-xs text-primary">
                    <ArrowRight className="h-3 w-3" />
                    Open Manufacturability
                  </div>
                </div>
                <div className="p-4 rounded-md border">
                  <div className="flex items-center gap-2 mb-2">
                    <Database className="h-5 w-5 text-muted-foreground" />
                    <span className="font-medium text-sm">Descriptor Matrices</span>
                  </div>
                  <p className="text-xs text-muted-foreground">Raw descriptor vectors for all computed scales, available for export.</p>
                  <Badge variant="secondary" className="mt-2 text-xs">Available</Badge>
                </div>
                <div className="p-4 rounded-md border opacity-60">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="h-5 w-5 text-muted-foreground" />
                    <span className="font-medium text-sm">Model Weights</span>
                  </div>
                  <p className="text-xs text-muted-foreground">Trained model weights from the prediction pipeline.</p>
                  <Badge variant="outline" className="mt-2 text-xs">Pending</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
