import { useState, useMemo } from "react";
import { PageHeader } from "@/components/page-header";
import { ResultsPanel } from "@/components/results-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Calculator,
  Zap,
  Activity,
  Play,
  CheckCircle,
  Clock,
  Filter,
  BarChart3,
  Cpu,
  Atom,
  Layers,
  Box,
  Download,
} from "lucide-react";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

type PredictionMethod = "ml" | "md" | "dft" | "fem" | "hybrid";

interface PropertyPrediction {
  id: string;
  name: string;
  value: number;
  unit: string;
  confidence: number;
  percentile: number;
  method: PredictionMethod;
  computeTime: string;
}

interface BatchJob {
  id: string;
  property: string;
  method: PredictionMethod;
  total: number;
  completed: number;
  status: "running" | "completed" | "queued";
  startTime: string;
  eta?: string;
}

const METHOD_LABELS: Record<PredictionMethod, string> = {
  ml: "ML Model",
  md: "MD Simulation",
  dft: "DFT Calculation",
  fem: "FEM Analysis",
  hybrid: "Hybrid Pipeline",
};

const METHOD_COLORS: Record<PredictionMethod, string> = {
  ml: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30",
  md: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/30",
  dft: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30",
  fem: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30",
  hybrid: "bg-primary/10 text-primary border-primary/30",
};

function generateMockPredictions(count: number): PropertyPrediction[] {
  const properties = [
    { name: "Thermal Conductivity", unit: "W/(m·K)", range: [0.1, 400] },
    { name: "Tensile Strength", unit: "MPa", range: [10, 1500] },
    { name: "Glass Transition", unit: "°C", range: [-50, 350] },
    { name: "Ionic Conductivity", unit: "S/cm", range: [1e-8, 1e-1] },
    { name: "Young's Modulus", unit: "GPa", range: [0.5, 300] },
    { name: "Thermal Stability", unit: "°C", range: [100, 600] },
    { name: "Dielectric Constant", unit: "ε", range: [2, 100] },
    { name: "Density", unit: "g/cm³", range: [0.9, 8.5] },
  ];
  const methods: PredictionMethod[] = ["ml", "md", "dft", "fem", "hybrid"];

  return Array.from({ length: count }, (_, i) => {
    const prop = properties[i % properties.length];
    const seed = i * 137;
    const value = prop.range[0] + seededRandom(seed) * (prop.range[1] - prop.range[0]);
    return {
      id: `pred-${i}`,
      name: prop.name,
      value: prop.name === "Ionic Conductivity" ? value : Math.round(value * 100) / 100,
      unit: prop.unit,
      confidence: 0.7 + seededRandom(seed + 1) * 0.25,
      percentile: Math.round(seededRandom(seed + 2) * 100),
      method: methods[Math.floor(seededRandom(seed + 3) * methods.length)],
      computeTime: `${(0.1 + seededRandom(seed + 4) * 5).toFixed(1)}s`,
    };
  });
}

function generateMockBatchJobs(): BatchJob[] {
  return [
    { id: "job-1", property: "Thermal Conductivity", method: "ml", total: 420000, completed: 387500, status: "running", startTime: "2h 14m ago", eta: "23m" },
    { id: "job-2", property: "Tensile Strength", method: "hybrid", total: 420000, completed: 420000, status: "completed", startTime: "4h 02m ago" },
    { id: "job-3", property: "Glass Transition", method: "md", total: 85000, completed: 12400, status: "running", startTime: "38m ago", eta: "2h 45m" },
    { id: "job-4", property: "Ionic Conductivity", method: "dft", total: 50000, completed: 0, status: "queued", startTime: "Queued" },
  ];
}

interface PropertyCardProps {
  prediction: PropertyPrediction;
}

function PropertyCard({ prediction }: PropertyCardProps) {
  const formatValue = (val: number, name: string) => {
    if (name === "Ionic Conductivity") {
      return val.toExponential(2);
    }
    return val.toLocaleString();
  };

  return (
    <Card className="hover-elevate" data-testid={`card-property-${prediction.id}`}>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className="font-medium text-sm">{prediction.name}</div>
          <Badge variant="outline" className={`text-xs ${METHOD_COLORS[prediction.method]}`}>
            {METHOD_LABELS[prediction.method]}
          </Badge>
        </div>

        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono">{formatValue(prediction.value, prediction.name)}</span>
          <span className="text-sm text-muted-foreground">{prediction.unit}</span>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 rounded-md bg-muted/50">
            <div className="text-sm font-mono font-medium">{Math.round(prediction.confidence * 100)}%</div>
            <div className="text-xs text-muted-foreground">Confidence</div>
          </div>
          <div className="p-2 rounded-md bg-muted/50">
            <div className="text-sm font-mono font-medium">P{prediction.percentile}</div>
            <div className="text-xs text-muted-foreground">Percentile</div>
          </div>
          <div className="p-2 rounded-md bg-muted/50">
            <div className="text-sm font-mono font-medium">{prediction.computeTime}</div>
            <div className="text-xs text-muted-foreground">Compute</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface BatchJobCardProps {
  job: BatchJob;
}

function BatchJobCard({ job }: BatchJobCardProps) {
  const progress = Math.round((job.completed / job.total) * 100);

  const statusColors = {
    running: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
    completed: "bg-green-500/10 text-green-600 dark:text-green-400",
    queued: "bg-muted text-muted-foreground",
  };

  return (
    <div className="p-4 rounded-lg border bg-card" data-testid={`card-batch-job-${job.id}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-md flex items-center justify-center ${statusColors[job.status]}`}>
            {job.status === "running" && <Activity className="h-4 w-4 animate-pulse" />}
            {job.status === "completed" && <CheckCircle className="h-4 w-4" />}
            {job.status === "queued" && <Clock className="h-4 w-4" />}
          </div>
          <div>
            <div className="font-medium">{job.property}</div>
            <div className="text-xs text-muted-foreground flex items-center gap-2">
              <Badge variant="outline" className={`text-xs ${METHOD_COLORS[job.method]}`}>
                {METHOD_LABELS[job.method]}
              </Badge>
              <span>{job.startTime}</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-sm font-medium">{formatNumber(job.completed)} / {formatNumber(job.total)}</div>
          {job.eta && <div className="text-xs text-muted-foreground">ETA: {job.eta}</div>}
        </div>
      </div>
      <Progress value={progress} className="h-2" />
    </div>
  );
}

export default function PropertyPredictionPage() {
  const [selectedScale, setSelectedScale] = useState<string>("molecular");
  const [selectedMethod, setSelectedMethod] = useState<PredictionMethod>("ml");
  const [variantCount, setVariantCount] = useState<number>(420000);
  const [topPercentile, setTopPercentile] = useState<number[]>([100]);
  const [showOnlyHighConfidence, setShowOnlyHighConfidence] = useState(false);

  const predictions = useMemo(() => generateMockPredictions(16), []);
  const batchJobs = useMemo(() => generateMockBatchJobs(), []);

  const todayPredictions = 847000;
  const avgThroughput = 12400;

  const filteredPredictions = useMemo(() => {
    let result = predictions;
    if (topPercentile[0] < 100) {
      result = result.filter(p => p.percentile >= (100 - topPercentile[0]));
    }
    if (showOnlyHighConfidence) {
      result = result.filter(p => p.confidence >= 0.9);
    }
    return result;
  }, [predictions, topPercentile, showOnlyHighConfidence]);

  const completedJobs = batchJobs.filter(j => j.status === "completed").length;
  const runningJobs = batchJobs.filter(j => j.status === "running").length;

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Property Prediction Engine" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" data-testid="button-export">
              <Download className="h-4 w-4 mr-2" />
              Export Results
            </Button>
            <Button data-testid="button-new-prediction">
              <Calculator className="h-4 w-4 mr-2" />
              New Prediction
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6 rounded-lg border border-primary/20">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/30">
                <Calculator className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-1">Property Prediction Engine</h2>
                <p className="text-muted-foreground">
                  Calculate material properties at scale using <strong className="text-foreground">ML models</strong>,
                  <strong className="text-foreground"> physics-based simulations</strong> (MD, DFT, FEM), 
                  and <strong className="text-foreground">hybrid pipelines</strong>.
                </p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold font-mono text-primary">{formatNumber(todayPredictions)}</div>
                <div className="text-sm text-muted-foreground">predictions today</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono">{formatNumber(variantCount)}</div>
                <div className="text-xs text-muted-foreground">Variants Selected</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">{formatNumber(avgThroughput)}/hr</div>
                <div className="text-xs text-muted-foreground">Avg Throughput</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono">{runningJobs}</div>
                <div className="text-xs text-muted-foreground">Running Jobs</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono text-primary">{completedJobs}</div>
                <div className="text-xs text-muted-foreground">Completed Today</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                Prediction Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Representation Scale</Label>
                  <Select value={selectedScale} onValueChange={setSelectedScale}>
                    <SelectTrigger data-testid="select-scale">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="molecular">
                        <div className="flex items-center gap-2">
                          <Atom className="h-4 w-4" />
                          Molecular / Repeat Unit
                        </div>
                      </SelectItem>
                      <SelectItem value="chain">
                        <div className="flex items-center gap-2">
                          <Layers className="h-4 w-4" />
                          Chain / Lattice
                        </div>
                      </SelectItem>
                      <SelectItem value="bulk">
                        <div className="flex items-center gap-2">
                          <Box className="h-4 w-4" />
                          Bulk / Effective
                        </div>
                      </SelectItem>
                      <SelectItem value="multi">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4" />
                          Multi-Scale (All)
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Prediction Method</Label>
                  <Select value={selectedMethod} onValueChange={(v) => setSelectedMethod(v as PredictionMethod)}>
                    <SelectTrigger data-testid="select-method">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ml">ML Model (Fast)</SelectItem>
                      <SelectItem value="md">MD Simulation</SelectItem>
                      <SelectItem value="dft">DFT Calculation</SelectItem>
                      <SelectItem value="fem">FEM Analysis</SelectItem>
                      <SelectItem value="hybrid">Hybrid Pipeline</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Target Properties</Label>
                  <Select defaultValue="all">
                    <SelectTrigger data-testid="select-properties">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Properties (8)</SelectItem>
                      <SelectItem value="thermal">Thermal Properties</SelectItem>
                      <SelectItem value="mechanical">Mechanical Properties</SelectItem>
                      <SelectItem value="electrical">Electrical Properties</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex items-center justify-between pt-2 border-t">
                <div className="text-sm text-muted-foreground">
                  Ready to predict <strong>{formatNumber(variantCount)}</strong> variants using <strong>{METHOD_LABELS[selectedMethod]}</strong>
                </div>
                <Button data-testid="button-run-prediction">
                  <Play className="h-4 w-4 mr-2" />
                  Run Prediction
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Batch Processing Queue
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {batchJobs.map(job => (
                <BatchJobCard key={job.id} job={job} />
              ))}
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Prediction Results
            </h3>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground">Top</Label>
                <Slider
                  value={topPercentile}
                  onValueChange={setTopPercentile}
                  max={100}
                  min={1}
                  step={1}
                  className="w-24"
                />
                <span className="text-sm font-mono w-10">{topPercentile[0]}%</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={showOnlyHighConfidence}
                  onCheckedChange={setShowOnlyHighConfidence}
                  data-testid="switch-high-confidence"
                />
                <Label className="text-sm">High confidence only</Label>
              </div>
              <Button variant="outline" size="sm" data-testid="button-filter">
                <Filter className="h-4 w-4 mr-2" />
                More Filters
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {filteredPredictions.map(prediction => (
              <PropertyCard key={prediction.id} prediction={prediction} />
            ))}
          </div>

          {filteredPredictions.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No predictions match current filters. Try adjusting the percentile or confidence threshold.
            </div>
          )}

          <ResultsPanel
            materialsCampaignId="demo-campaign"
            title="Computation Artifacts"
            collapsible={true}
            defaultExpanded={false}
          />
        </div>
      </main>
    </div>
  );
}
