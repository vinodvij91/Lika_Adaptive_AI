import { useState } from "react";
import { Link } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  Rocket,
  Play,
  Pause,
  XCircle,
  RefreshCw,
  Cpu,
  Zap,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  Settings,
  Activity,
  Target,
  Layers,
  Database,
  Server,
  ChevronDown,
  ChevronUp,
  Beaker,
  FlaskConical,
  Sparkles,
  TrendingUp,
  Award,
} from "lucide-react";

interface PipelineJob {
  id: string;
  type: string;
  status: string;
  progressPercent: number;
  itemsTotal: number;
  itemsCompleted: number;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  errorMessage?: string;
  inputPayload?: Record<string, unknown>;
  outputPayload?: Record<string, unknown>;
}

interface ComputeNode {
  id: string;
  name: string;
  provider: string;
  gpuType: string;
  tier: string;
  purpose: string;
  status: string;
}

interface PipelineStats {
  total: number;
  queued: number;
  running: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  byType: Record<string, number>;
}

const drugDiscoveryJobs = [
  { value: "alzheimers_multitarget", label: "Alzheimer's 12-Target Pipeline", icon: Sparkles, description: "Dedicated multi-target algorithm for Alzheimer's disease (Tau, APP, NLRP3, PINK1, ULK1, TFEB, Sigma-1, and more)", isSpecialized: true },
  { value: "docking", label: "Molecular Docking", icon: Target, description: "Run AutoDock Vina docking simulations" },
  { value: "fingerprint_generation", label: "Fingerprint Generation", icon: Layers, description: "Generate ECFP/MACCS molecular fingerprints" },
  { value: "ml_training", label: "ML Model Training", icon: Zap, description: "Train QSAR/ADMET prediction models" },
  { value: "distributed_prediction", label: "Distributed Prediction", icon: Activity, description: "Run predictions on large molecule sets" },
  { value: "full_pipeline", label: "Full Pipeline", icon: Rocket, description: "Complete discovery pipeline with all steps" },
];

const materialsJobTypes = [
  { value: "mat_battery", label: "Battery Materials", icon: Zap, description: "Cathode/anode discovery for Li-ion and solid-state batteries" },
  { value: "mat_solar", label: "Photovoltaic Materials", icon: Activity, description: "Solar absorber discovery with band gap optimization" },
  { value: "mat_superconductor", label: "Superconductor Discovery", icon: Zap, description: "High-Tc superconductor discovery with DFT validation" },
  { value: "mat_catalyst", label: "Catalyst Discovery", icon: Target, description: "HER/ORR catalyst discovery for fuel cells" },
  { value: "mat_thermoelectric", label: "Thermoelectric Materials", icon: Activity, description: "High-ZT thermoelectric materials discovery" },
  { value: "mat_pfas_replacement", label: "PFAS Replacement", icon: Layers, description: "Fluorine-free alternatives for coatings (EPA compliant)" },
  { value: "mat_aerospace", label: "Aerospace Materials", icon: Rocket, description: "Lightweight alloys and composites (Ti-Al, SiC)" },
  { value: "mat_biomedical", label: "Biomedical Materials", icon: Target, description: "Biocompatible implant materials with bone matching" },
  { value: "mat_semiconductor", label: "Wide-Gap Semiconductors", icon: Zap, description: "SiC/GaN alternatives for power electronics" },
  { value: "mat_construction", label: "Sustainable Construction", icon: Layers, description: "Low-carbon cement alternatives (geopolymers)" },
  { value: "mat_transparent", label: "Transparent Conductors", icon: Activity, description: "ITO-free transparent electrodes (graphene, AgNW)" },
  { value: "mat_magnet", label: "Rare-Earth-Free Magnets", icon: Target, description: "RE-free permanent magnets for EVs and wind turbines" },
  { value: "mat_electrolyte", label: "Solid Electrolytes", icon: Zap, description: "Solid-state battery electrolytes (LGPS, LLZO)" },
  { value: "mat_water", label: "Water Purification", icon: Layers, description: "Membrane materials for desalination and filtration" },
  { value: "mat_carbon_capture", label: "Carbon Capture", icon: Activity, description: "DAC and flue gas CO2 capture sorbents (MOFs, zeolites)" },
];

const jobTypeOptions = [...drugDiscoveryJobs, ...materialsJobTypes];

const statusColors: Record<string, { bg: string; text: string; icon: typeof CheckCircle }> = {
  queued: { bg: "bg-blue-500/10", text: "text-blue-500", icon: Clock },
  dispatched: { bg: "bg-cyan-500/10", text: "text-cyan-500", icon: RefreshCw },
  running: { bg: "bg-amber-500/10", text: "text-amber-500", icon: Loader2 },
  succeeded: { bg: "bg-emerald-500/10", text: "text-emerald-500", icon: CheckCircle },
  failed: { bg: "bg-red-500/10", text: "text-red-500", icon: AlertCircle },
  cancelled: { bg: "bg-gray-500/10", text: "text-gray-500", icon: XCircle },
  paused: { bg: "bg-violet-500/10", text: "text-violet-500", icon: Pause },
};

export default function PipelineLauncherPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("launch");
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  
  const [config, setConfig] = useState({
    name: "",
    jobType: "docking",
    useGpu: true,
    mixedPrecision: true,
    daskWorkers: 4,
    chunkSize: 10000,
    dockingExhaustiveness: 8,
    fpRadius: 2,
    fpBits: 2048,
    preferredNodeId: "",
    targetPdb: "",
    centerX: 0,
    centerY: 0,
    centerZ: 0,
    boxSizeX: 20,
    boxSizeY: 20,
    boxSizeZ: 20,
  });

  // Alzheimer's 12-target configuration with toggleable targets
  const alzheimersTargets = [
    { key: "tau", name: "Tau (MAPT)", pathway: "Protein Aggregation", active: true },
    { key: "app", name: "APP", pathway: "Protein Aggregation", active: true },
    { key: "snca", name: "Alpha-Synuclein", pathway: "Protein Aggregation", active: true },
    { key: "nlrp3", name: "NLRP3", pathway: "Neuroinflammation", active: true },
    { key: "rock2", name: "ROCK2", pathway: "Neuroprotection", active: true },
    { key: "pink1", name: "PINK1", pathway: "Autophagy", active: true },
    { key: "ulk1", name: "ULK1", pathway: "Autophagy", active: true },
    { key: "tfeb", name: "TFEB", pathway: "Autophagy", active: true },
    { key: "sigma1", name: "Sigma-1 Receptor", pathway: "Neuroprotection", active: true },
    { key: "nsmase2", name: "nSMase2", pathway: "Exosome Biogenesis", active: true },
    { key: "aqp4", name: "AQP4", pathway: "Glymphatic Clearance", active: true },
    { key: "lrp1", name: "LRP1", pathway: "Receptor-mediated Clearance", active: true },
  ];

  const [activeTargets, setActiveTargets] = useState<string[]>(alzheimersTargets.map(t => t.key));

  const toggleAlzheimersTarget = (targetKey: string) => {
    setActiveTargets(prev => 
      prev.includes(targetKey) 
        ? prev.filter(k => k !== targetKey)
        : [...prev, targetKey]
    );
  };

  const { data: stats, isLoading: statsLoading } = useQuery<PipelineStats>({
    queryKey: ["/api/pipeline/stats"],
    refetchInterval: 10000,
  });

  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery<{ jobs: PipelineJob[] }>({
    queryKey: ["/api/pipeline/jobs"],
    refetchInterval: 5000,
  });

  const { data: nodesData } = useQuery<{ nodes: ComputeNode[] }>({
    queryKey: ["/api/pipeline/compute-nodes"],
  });

  const launchMutation = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiRequest("POST", "/api/pipeline/launch", payload);
    },
    onSuccess: (data: any) => {
      toast({ title: "Pipeline Launched", description: data.message || "Job queued successfully" });
      queryClient.invalidateQueries({ queryKey: ["/api/pipeline/jobs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/pipeline/stats"] });
      setActiveTab("jobs");
    },
    onError: (error: any) => {
      toast({ title: "Launch Failed", description: error.message || "Failed to launch pipeline", variant: "destructive" });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: async (jobId: string) => {
      return apiRequest("POST", `/api/pipeline/jobs/${jobId}/cancel`);
    },
    onSuccess: () => {
      toast({ title: "Job Cancelled" });
      refetchJobs();
    },
  });

  const retryMutation = useMutation({
    mutationFn: async (jobId: string) => {
      return apiRequest("POST", `/api/pipeline/jobs/${jobId}/retry`);
    },
    onSuccess: () => {
      toast({ title: "Job Queued for Retry" });
      refetchJobs();
    },
  });

  const simulateMutation = useMutation({
    mutationFn: async (jobId: string) => {
      return apiRequest("POST", `/api/pipeline/jobs/${jobId}/simulate`);
    },
    onSuccess: (data: any) => {
      toast({ title: "Simulation Complete", description: `Found ${data.candidatesFound} candidate materials` });
      refetchJobs();
      queryClient.invalidateQueries({ queryKey: ["/api/pipeline/stats"] });
    },
    onError: (error: any) => {
      toast({ title: "Simulation Failed", description: error.message, variant: "destructive" });
    },
  });

  const alzheimersLaunchMutation = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      return apiRequest("POST", "/api/alzheimers/launch", payload);
    },
    onSuccess: (data: any) => {
      toast({ 
        title: "Alzheimer's Pipeline Launched", 
        description: `${data.message} - Estimated time: ${data.executionPlan?.totalTimeHours?.toFixed(1) || 'N/A'}h` 
      });
      queryClient.invalidateQueries({ queryKey: ["/api/pipeline/jobs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/pipeline/stats"] });
      setActiveTab("jobs");
    },
    onError: (error: any) => {
      toast({ title: "Launch Failed", description: error.message || "Failed to launch Alzheimer's pipeline", variant: "destructive" });
    },
  });

  const handleLaunch = () => {
    if (!config.name.trim()) {
      toast({ title: "Validation Error", description: "Pipeline name is required", variant: "destructive" });
      return;
    }

    if (config.jobType === "alzheimers_multitarget") {
      const alzheimersPayload = {
        campaignId: `alzheimers-${Date.now()}`,
        moleculeIds: [],
        config: {
          enableGpuAcceleration: config.useGpu,
          prioritizeBbbPenetration: true,
          maxCandidates: 100,
          diversityClustering: true,
          activeTargets: activeTargets,
        },
        preferredNodeId: config.preferredNodeId || undefined,
        useGpu: config.useGpu,
        name: config.name || `Alzheimer's ${activeTargets.length}-Target Pipeline`,
      };
      alzheimersLaunchMutation.mutate(alzheimersPayload);
      return;
    }

    const payload = {
      name: config.name,
      jobType: config.jobType,
      useGpu: config.useGpu,
      enableMixedPrecision: config.mixedPrecision,
      daskWorkers: config.daskWorkers,
      chunkSize: config.chunkSize,
      dockingConfig: config.jobType === "docking" || config.jobType === "full_pipeline" ? {
        exhaustiveness: config.dockingExhaustiveness,
        targetPdb: config.targetPdb || undefined,
        boxCenter: [config.centerX, config.centerY, config.centerZ],
        boxSize: [config.boxSizeX, config.boxSizeY, config.boxSizeZ],
      } : undefined,
      fingerprintConfig: config.jobType === "fingerprint_generation" || config.jobType === "full_pipeline" ? {
        fpType: "ecfp4",
        radius: config.fpRadius,
        nBits: config.fpBits,
      } : undefined,
      preferredNodeId: config.preferredNodeId || undefined,
    };

    launchMutation.mutate(payload);
  };

  const jobs = jobsData?.jobs || [];
  const nodes = nodesData?.nodes || [];

  const formatDuration = (start?: string, end?: string) => {
    if (!start) return "-";
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const diffMs = endDate.getTime() - startDate.getTime();
    const seconds = Math.floor(diffMs / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-orange-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-orange-600 via-amber-500 to-yellow-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxMGwyMCAxMHYyMEwzMCA1MCAxMCA0MFYyMEwzMCAxMHoiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjE1KSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9nPjwvc3ZnPg==')] opacity-40" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Rocket className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">Pipeline Launcher</h1>
                  <p className="text-orange-100">Configure and launch high-throughput compute pipelines</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-orange-100">
                <Zap className="h-4 w-4" />
                <span>Dask distributed computing with GPU acceleration and mixed precision</span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-blue-500/10 flex items-center justify-center">
                    <Clock className="h-7 w-7 text-blue-500" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold">{stats?.queued || 0}</p>
                    <p className="text-sm text-muted-foreground">Queued</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-amber-500/10 flex items-center justify-center">
                    <Activity className="h-7 w-7 text-amber-500" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold">{stats?.running || 0}</p>
                    <p className="text-sm text-muted-foreground">Running</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 flex items-center justify-center">
                    <CheckCircle className="h-7 w-7 text-emerald-500" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold">{stats?.succeeded || 0}</p>
                    <p className="text-sm text-muted-foreground">Succeeded</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-red-500/10 flex items-center justify-center">
                    <AlertCircle className="h-7 w-7 text-red-500" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold">{stats?.failed || 0}</p>
                    <p className="text-sm text-muted-foreground">Failed</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="shadow-lg">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <CardHeader className="border-b pb-0">
                <TabsList className="w-full justify-start gap-2 bg-transparent p-0">
                  <TabsTrigger value="launch" className="data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-600">
                    <Rocket className="h-4 w-4 mr-2" />
                    Launch Pipeline
                  </TabsTrigger>
                  <TabsTrigger value="jobs" className="data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-600">
                    <Activity className="h-4 w-4 mr-2" />
                    Job Queue ({jobs.length})
                  </TabsTrigger>
                  <TabsTrigger value="compute" className="data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-600">
                    <Server className="h-4 w-4 mr-2" />
                    Compute Nodes ({nodes.length})
                  </TabsTrigger>
                  <TabsTrigger value="results" className="data-[state=active]:bg-emerald-500/10 data-[state=active]:text-emerald-600">
                    <FlaskConical className="h-4 w-4 mr-2" />
                    Results ({jobs.filter(j => j.status === "succeeded").length})
                  </TabsTrigger>
                </TabsList>
              </CardHeader>

              <TabsContent value="launch" className="p-0 m-0">
                <CardContent className="p-6 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="name">Pipeline Name</Label>
                        <Input
                          id="name"
                          data-testid="input-pipeline-name"
                          placeholder="e.g., EGFR Inhibitor Screen"
                          value={config.name}
                          onChange={(e) => setConfig({ ...config, name: e.target.value })}
                          className="mt-1.5"
                        />
                      </div>

                      <div>
                        <Label>Job Type</Label>
                        <Select value={config.jobType} onValueChange={(v) => setConfig({ ...config, jobType: v })}>
                          <SelectTrigger data-testid="select-job-type" className="mt-1.5">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="max-h-80">
                            <SelectGroup>
                              <SelectLabel className="text-orange-600 font-semibold">Drug Discovery</SelectLabel>
                              {drugDiscoveryJobs.map((opt) => (
                                <SelectItem key={opt.value} value={opt.value}>
                                  <div className="flex items-center gap-2">
                                    <opt.icon className="h-4 w-4 text-muted-foreground" />
                                    <span>{opt.label}</span>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectGroup>
                            <SelectGroup>
                              <SelectLabel className="text-teal-600 font-semibold">Materials Science</SelectLabel>
                              {materialsJobTypes.map((opt) => (
                                <SelectItem key={opt.value} value={opt.value}>
                                  <div className="flex items-center gap-2">
                                    <opt.icon className="h-4 w-4 text-muted-foreground" />
                                    <span>{opt.label}</span>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          {jobTypeOptions.find((o) => o.value === config.jobType)?.description}
                        </p>
                      </div>

                      <div>
                        <Label>Compute Node</Label>
                        <Select value={config.preferredNodeId || "auto"} onValueChange={(v) => setConfig({ ...config, preferredNodeId: v === "auto" ? "" : v })}>
                          <SelectTrigger data-testid="select-compute-node" className="mt-1.5">
                            <SelectValue placeholder="Auto-select best node" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="auto">Auto-select</SelectItem>
                            {nodes.map((node) => (
                              <SelectItem key={node.id} value={node.id}>
                                <div className="flex items-center gap-2">
                                  <Cpu className="h-4 w-4 text-muted-foreground" />
                                  <span>{node.name}</span>
                                  <Badge variant="outline" className="ml-2">{node.gpuType}</Badge>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>GPU Acceleration</Label>
                          <p className="text-xs text-muted-foreground">Use CUDA for 10-50x speedup</p>
                        </div>
                        <Switch
                          data-testid="switch-gpu"
                          checked={config.useGpu}
                          onCheckedChange={(v) => setConfig({ ...config, useGpu: v })}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Mixed Precision (FP16)</Label>
                          <p className="text-xs text-muted-foreground">2x memory efficiency and speed</p>
                        </div>
                        <Switch
                          data-testid="switch-mixed-precision"
                          checked={config.mixedPrecision}
                          onCheckedChange={(v) => setConfig({ ...config, mixedPrecision: v })}
                        />
                      </div>

                      <div>
                        <Label>Dask Workers: {config.daskWorkers}</Label>
                        <Slider
                          data-testid="slider-dask-workers"
                          value={[config.daskWorkers]}
                          onValueChange={([v]) => setConfig({ ...config, daskWorkers: v })}
                          min={1}
                          max={32}
                          step={1}
                          className="mt-2"
                        />
                      </div>

                      <div>
                        <Label>Chunk Size: {config.chunkSize.toLocaleString()}</Label>
                        <Slider
                          data-testid="slider-chunk-size"
                          value={[config.chunkSize]}
                          onValueChange={([v]) => setConfig({ ...config, chunkSize: v })}
                          min={1000}
                          max={100000}
                          step={1000}
                          className="mt-2"
                        />
                        <p className="text-xs text-muted-foreground mt-1">Molecules per worker batch</p>
                      </div>
                    </div>
                  </div>

                  {(config.jobType === "docking" || config.jobType === "full_pipeline") && (
                    <div className="border rounded-lg p-4 bg-muted/30">
                      <h4 className="font-medium mb-4 flex items-center gap-2">
                        <Target className="h-4 w-4" />
                        Docking Configuration
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <Label>Exhaustiveness</Label>
                          <Input
                            data-testid="input-exhaustiveness"
                            type="number"
                            value={config.dockingExhaustiveness}
                            onChange={(e) => setConfig({ ...config, dockingExhaustiveness: parseInt(e.target.value) || 8 })}
                            min={1}
                            max={64}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Center X</Label>
                          <Input
                            data-testid="input-center-x"
                            type="number"
                            value={config.centerX}
                            onChange={(e) => setConfig({ ...config, centerX: parseFloat(e.target.value) || 0 })}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Center Y</Label>
                          <Input
                            data-testid="input-center-y"
                            type="number"
                            value={config.centerY}
                            onChange={(e) => setConfig({ ...config, centerY: parseFloat(e.target.value) || 0 })}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Center Z</Label>
                          <Input
                            data-testid="input-center-z"
                            type="number"
                            value={config.centerZ}
                            onChange={(e) => setConfig({ ...config, centerZ: parseFloat(e.target.value) || 0 })}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Box Size X</Label>
                          <Input
                            data-testid="input-box-x"
                            type="number"
                            value={config.boxSizeX}
                            onChange={(e) => setConfig({ ...config, boxSizeX: parseFloat(e.target.value) || 20 })}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Box Size Y</Label>
                          <Input
                            data-testid="input-box-y"
                            type="number"
                            value={config.boxSizeY}
                            onChange={(e) => setConfig({ ...config, boxSizeY: parseFloat(e.target.value) || 20 })}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Box Size Z</Label>
                          <Input
                            data-testid="input-box-z"
                            type="number"
                            value={config.boxSizeZ}
                            onChange={(e) => setConfig({ ...config, boxSizeZ: parseFloat(e.target.value) || 20 })}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Target PDB</Label>
                          <Input
                            data-testid="input-target-pdb"
                            placeholder="e.g., 1UKL"
                            value={config.targetPdb}
                            onChange={(e) => setConfig({ ...config, targetPdb: e.target.value })}
                            className="mt-1.5"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {(config.jobType === "fingerprint_generation" || config.jobType === "full_pipeline") && (
                    <div className="border rounded-lg p-4 bg-muted/30">
                      <h4 className="font-medium mb-4 flex items-center gap-2">
                        <Layers className="h-4 w-4" />
                        Fingerprint Configuration
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label>Radius (ECFP)</Label>
                          <Input
                            data-testid="input-fp-radius"
                            type="number"
                            value={config.fpRadius}
                            onChange={(e) => setConfig({ ...config, fpRadius: parseInt(e.target.value) || 2 })}
                            min={1}
                            max={6}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label>Bits</Label>
                          <Select value={config.fpBits.toString()} onValueChange={(v) => setConfig({ ...config, fpBits: parseInt(v) })}>
                            <SelectTrigger data-testid="select-fp-bits" className="mt-1.5">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="1024">1024</SelectItem>
                              <SelectItem value="2048">2048</SelectItem>
                              <SelectItem value="4096">4096</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </div>
                  )}

                  {config.jobType === "alzheimers_multitarget" && (
                    <div className="border rounded-lg p-4 bg-gradient-to-r from-purple-500/10 to-pink-500/10">
                      <h4 className="font-medium mb-4 flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-purple-500" />
                        Alzheimer's Multi-Target Algorithm Configuration
                      </h4>
                      <p className="text-sm text-muted-foreground mb-4">
                        Dedicated multi-target algorithm optimized for Alzheimer's disease drug discovery.
                        Select which targets to include in the screening pipeline.
                      </p>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div className="bg-background/60 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-purple-500">{activeTargets.length}</p>
                          <p className="text-xs text-muted-foreground">Active Targets</p>
                        </div>
                        <div className="bg-background/60 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-pink-500">{new Set(alzheimersTargets.filter(t => activeTargets.includes(t.key)).map(t => t.pathway)).size}</p>
                          <p className="text-xs text-muted-foreground">Pathways</p>
                        </div>
                        <div className="bg-background/60 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-amber-500">3</p>
                          <p className="text-xs text-muted-foreground">Phases</p>
                        </div>
                        <div className="bg-background/60 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-emerald-500">16</p>
                          <p className="text-xs text-muted-foreground">Tasks</p>
                        </div>
                      </div>
                      
                      <div className="bg-background/60 rounded-lg p-3 mb-4">
                        <div className="flex items-center justify-between mb-3">
                          <p className="text-sm font-medium text-purple-500">Select Targets</p>
                          <div className="flex gap-2">
                            <Button 
                              size="sm" 
                              variant="outline" 
                              className="text-xs h-7"
                              onClick={() => setActiveTargets(alzheimersTargets.map(t => t.key))}
                              data-testid="button-select-all-targets"
                            >
                              Select All
                            </Button>
                            <Button 
                              size="sm" 
                              variant="outline" 
                              className="text-xs h-7"
                              onClick={() => setActiveTargets([])}
                              data-testid="button-deselect-all-targets"
                            >
                              Clear
                            </Button>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {alzheimersTargets.map((target) => (
                            <div 
                              key={target.key}
                              className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
                                activeTargets.includes(target.key) 
                                  ? "bg-purple-500/20 border border-purple-500/30" 
                                  : "bg-muted/30 border border-transparent hover:bg-muted/50"
                              }`}
                              onClick={() => toggleAlzheimersTarget(target.key)}
                              data-testid={`target-toggle-${target.key}`}
                            >
                              <div className={`w-4 h-4 rounded flex items-center justify-center ${
                                activeTargets.includes(target.key) 
                                  ? "bg-purple-500 text-white" 
                                  : "border border-muted-foreground"
                              }`}>
                                {activeTargets.includes(target.key) && <CheckCircle className="h-3 w-3" />}
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-medium truncate">{target.name}</p>
                                <p className="text-[10px] text-muted-foreground truncate">{target.pathway}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="p-3 bg-amber-500/10 rounded-lg border border-amber-500/20">
                        <div className="flex items-center gap-2 text-amber-600">
                          <TrendingUp className="h-4 w-4" />
                          <span className="text-sm font-medium">BBB Penetration Priority</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          This algorithm prioritizes blood-brain barrier penetration for CNS drug candidates
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-end gap-3 pt-4 border-t">
                    <Button variant="outline" onClick={() => setConfig({ ...config, name: "" })}>
                      Reset
                    </Button>
                    <Button
                      data-testid="button-launch-pipeline"
                      className={config.jobType === "alzheimers_multitarget" 
                        ? "bg-gradient-to-r from-purple-500 to-pink-500 gap-2" 
                        : "bg-gradient-to-r from-orange-500 to-amber-500 gap-2"}
                      onClick={handleLaunch}
                      disabled={launchMutation.isPending || alzheimersLaunchMutation.isPending}
                    >
                      {(launchMutation.isPending || alzheimersLaunchMutation.isPending) ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      {config.jobType === "alzheimers_multitarget" 
                        ? "Launch Alzheimer's Pipeline" 
                        : "Launch Pipeline"}
                    </Button>
                  </div>
                </CardContent>
              </TabsContent>

              <TabsContent value="jobs" className="p-0 m-0">
                <CardContent className="p-0">
                  {jobsLoading ? (
                    <div className="p-12 text-center">
                      <Loader2 className="h-8 w-8 animate-spin mx-auto text-orange-500" />
                      <p className="text-muted-foreground mt-4">Loading jobs...</p>
                    </div>
                  ) : jobs.length === 0 ? (
                    <div className="p-12 text-center">
                      <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-orange-500/10 to-amber-500/10 flex items-center justify-center mx-auto mb-5">
                        <Rocket className="h-9 w-9 text-orange-500" />
                      </div>
                      <p className="font-semibold text-lg mb-2">No pipeline jobs yet</p>
                      <p className="text-muted-foreground mb-6">Configure and launch your first compute pipeline</p>
                      <Button onClick={() => setActiveTab("launch")} className="gap-2 bg-gradient-to-r from-orange-500 to-amber-500">
                        <Play className="h-4 w-4" />
                        Launch Pipeline
                      </Button>
                    </div>
                  ) : (
                    <div className="divide-y">
                      {jobs.map((job) => {
                        const statusInfo = statusColors[job.status] || statusColors.queued;
                        const StatusIcon = statusInfo.icon;
                        const isExpanded = expandedJob === job.id;

                        return (
                          <div key={job.id} className="p-4 hover:bg-muted/30 transition-colors">
                            <div
                              className="flex items-center gap-4 cursor-pointer"
                              onClick={() => setExpandedJob(isExpanded ? null : job.id)}
                              data-testid={`job-row-${job.id}`}
                            >
                              <div className={`w-10 h-10 rounded-xl ${statusInfo.bg} flex items-center justify-center`}>
                                <StatusIcon className={`h-5 w-5 ${statusInfo.text} ${job.status === "running" ? "animate-spin" : ""}`} />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <span className="font-medium truncate">
                                    {(job.inputPayload as any)?.name || job.type}
                                  </span>
                                  <Badge variant="outline" className="text-xs">{job.type}</Badge>
                                </div>
                                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                                  <span>{new Date(job.createdAt).toLocaleString()}</span>
                                  <span>{formatDuration(job.startedAt, job.completedAt)}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-3">
                                {job.status === "running" && (
                                  <div className="w-32">
                                    <Progress value={job.progressPercent} className="h-2" />
                                    <p className="text-xs text-muted-foreground text-right mt-1">
                                      {job.itemsCompleted}/{job.itemsTotal}
                                    </p>
                                  </div>
                                )}
                                <Badge className={`${statusInfo.bg} ${statusInfo.text} border-0`}>
                                  {job.status}
                                </Badge>
                                {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                              </div>
                            </div>

                            {isExpanded && (
                              <div className="mt-4 pt-4 border-t space-y-4">
                                {/* Pipeline Configuration Details */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                  <div className="bg-muted/30 rounded-lg p-3">
                                    <p className="text-muted-foreground text-xs mb-1">Job Type</p>
                                    <p className="font-medium">{jobTypeOptions.find(j => j.value === job.type)?.label || job.type}</p>
                                  </div>
                                  <div className="bg-muted/30 rounded-lg p-3">
                                    <p className="text-muted-foreground text-xs mb-1">GPU Acceleration</p>
                                    <p className="font-medium">{(job.inputPayload as any)?.useGpu ? "Enabled" : "Disabled"}</p>
                                  </div>
                                  <div className="bg-muted/30 rounded-lg p-3">
                                    <p className="text-muted-foreground text-xs mb-1">Dask Workers</p>
                                    <p className="font-medium">{(job.inputPayload as any)?.nWorkers || 4}</p>
                                  </div>
                                  <div className="bg-muted/30 rounded-lg p-3">
                                    <p className="text-muted-foreground text-xs mb-1">Chunk Size</p>
                                    <p className="font-medium">{((job.inputPayload as any)?.chunkSize || 10000).toLocaleString()}</p>
                                  </div>
                                </div>
                                
                                {/* Materials Being Processed */}
                                <div className="bg-orange-500/5 border border-orange-500/20 rounded-lg p-4">
                                  <p className="text-sm font-medium text-orange-600 mb-2">Materials Discovery Pipeline</p>
                                  <p className="text-sm text-muted-foreground mb-3">
                                    {jobTypeOptions.find(j => j.value === job.type)?.description || "Processing materials from the database"}
                                  </p>
                                  <div className="grid grid-cols-2 gap-3 text-sm">
                                    <div>
                                      <span className="text-muted-foreground">Source:</span>
                                      <span className="ml-2 font-medium">Materials Library (53,923 materials)</span>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">Target Properties:</span>
                                      <span className="ml-2 font-medium">
                                        {job.type.includes('battery') ? 'Voltage, Capacity, Stability' :
                                         job.type.includes('solar') ? 'Band Gap, Absorption, Efficiency' :
                                         job.type.includes('pfas') ? 'Fluorine-free, EPA Compliant' :
                                         job.type.includes('catalyst') ? 'Activity, Selectivity, Durability' :
                                         job.type.includes('superconductor') ? 'Critical Temperature (Tc)' :
                                         job.type.includes('thermoelectric') ? 'Figure of Merit (ZT)' :
                                         job.type.includes('aerospace') ? 'Strength-to-Weight Ratio' :
                                         job.type.includes('biomedical') ? 'Biocompatibility, Osseointegration' :
                                         job.type.includes('semiconductor') ? 'Band Gap, Mobility' :
                                         job.type.includes('construction') ? 'CO2 Reduction, Strength' :
                                         job.type.includes('transparent') ? 'Conductivity, Transparency' :
                                         job.type.includes('magnet') ? 'Coercivity, Remanence' :
                                         job.type.includes('electrolyte') ? 'Ionic Conductivity, Stability' :
                                         job.type.includes('water') ? 'Permeability, Selectivity' :
                                         job.type.includes('carbon') ? 'Adsorption Capacity, Selectivity' :
                                         'Multiple Properties'}
                                      </span>
                                    </div>
                                  </div>
                                </div>

                                {/* Output Payload if succeeded */}
                                {job.status === "succeeded" && job.outputPayload && (
                                  <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-4">
                                    <p className="text-sm font-medium text-emerald-600 mb-2">Results</p>
                                    <div className="text-sm space-y-1">
                                      <p><span className="text-muted-foreground">Materials Processed:</span> <span className="font-medium">{(job.outputPayload as any)?.materialsProcessed || job.itemsCompleted}</span></p>
                                      <p><span className="text-muted-foreground">Candidates Found:</span> <span className="font-medium">{(job.outputPayload as any)?.candidatesFound || 0}</span></p>
                                    </div>
                                  </div>
                                )}

                                {job.errorMessage && (
                                  <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-sm text-red-600">
                                    {job.errorMessage}
                                  </div>
                                )}
                                <div className="flex gap-2">
                                  {job.status === "queued" && (
                                    <Button
                                      size="sm"
                                      className="bg-gradient-to-r from-emerald-500 to-teal-500"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        simulateMutation.mutate(job.id);
                                      }}
                                      disabled={simulateMutation.isPending}
                                      data-testid={`button-simulate-${job.id}`}
                                    >
                                      {simulateMutation.isPending ? (
                                        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                      ) : (
                                        <Sparkles className="h-4 w-4 mr-1" />
                                      )}
                                      Simulate Completion
                                    </Button>
                                  )}
                                  {(job.status === "queued" || job.status === "running") && (
                                    <Button
                                      size="sm"
                                      variant="destructive"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        cancelMutation.mutate(job.id);
                                      }}
                                      disabled={cancelMutation.isPending}
                                    >
                                      <XCircle className="h-4 w-4 mr-1" />
                                      Cancel
                                    </Button>
                                  )}
                                  {job.status === "failed" && (
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        retryMutation.mutate(job.id);
                                      }}
                                      disabled={retryMutation.isPending}
                                    >
                                      <RefreshCw className="h-4 w-4 mr-1" />
                                      Retry
                                    </Button>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </TabsContent>

              <TabsContent value="compute" className="p-0 m-0">
                <CardContent className="p-0">
                  {nodes.length === 0 ? (
                    <div className="p-12 text-center">
                      <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-orange-500/10 to-amber-500/10 flex items-center justify-center mx-auto mb-5">
                        <Server className="h-9 w-9 text-orange-500" />
                      </div>
                      <p className="font-semibold text-lg mb-2">No compute nodes available</p>
                      <p className="text-muted-foreground">Configure compute nodes in the Infrastructure section</p>
                    </div>
                  ) : (
                    <div className="divide-y">
                      {nodes.map((node) => (
                        <div key={node.id} className="p-4 flex items-center gap-4">
                          <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center">
                            <Cpu className="h-5 w-5 text-orange-500" />
                          </div>
                          <div className="flex-1">
                            <p className="font-medium">{node.name}</p>
                            <p className="text-sm text-muted-foreground">{node.provider} - {node.purpose}</p>
                          </div>
                          <Badge variant="outline">{node.gpuType}</Badge>
                          <Badge className={node.status === "active" ? "bg-emerald-500/10 text-emerald-500 border-0" : "bg-gray-500/10 text-gray-500 border-0"}>
                            {node.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </TabsContent>

              <TabsContent value="results" className="p-0 m-0">
                <CardContent className="p-0">
                  {(() => {
                    const completedJobs = jobs.filter(j => j.status === "succeeded");
                    if (completedJobs.length === 0) {
                      return (
                        <div className="p-12 text-center">
                          <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-5">
                            <FlaskConical className="h-9 w-9 text-emerald-500" />
                          </div>
                          <p className="font-semibold text-lg mb-2">No completed pipelines yet</p>
                          <p className="text-muted-foreground mb-6">Launch a pipeline and simulate completion to see discovered materials</p>
                          <Button onClick={() => setActiveTab("jobs")} className="gap-2 bg-gradient-to-r from-emerald-500 to-teal-500">
                            <Activity className="h-4 w-4" />
                            View Job Queue
                          </Button>
                        </div>
                      );
                    }
                    return (
                      <div className="divide-y">
                        {completedJobs.map((job) => {
                          const output = job.outputPayload as any || {};
                          const candidates = output.candidates || [];
                          const jobTypeInfo = jobTypeOptions.find(j => j.value === job.type);
                          
                          return (
                            <div key={job.id} className="p-6 space-y-4">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                                    <CheckCircle className="h-6 w-6 text-emerald-500" />
                                  </div>
                                  <div>
                                    <h3 className="font-semibold text-lg">{(job.inputPayload as any)?.name || job.type}</h3>
                                    <p className="text-sm text-muted-foreground">{jobTypeInfo?.label} - Completed {new Date(job.completedAt || job.createdAt).toLocaleString()}</p>
                                  </div>
                                </div>
                                <div className="flex items-center gap-4">
                                  <div className="text-right">
                                    <p className="text-2xl font-bold text-emerald-600">{output.candidatesFound || candidates.length}</p>
                                    <p className="text-xs text-muted-foreground">Candidates Found</p>
                                  </div>
                                  <div className="text-right">
                                    <p className="text-2xl font-bold">{output.materialsProcessed || job.itemsCompleted}</p>
                                    <p className="text-xs text-muted-foreground">Materials Screened</p>
                                  </div>
                                  <Link href={`/pipeline/results/${job.id}`}>
                                    <Button data-testid={`button-view-report-${job.id}`}>
                                      View Full Report
                                    </Button>
                                  </Link>
                                </div>
                              </div>
                              
                              {candidates.length > 0 && (
                                <div className="space-y-3">
                                  <div className="flex items-center gap-2">
                                    <Award className="h-4 w-4 text-amber-500" />
                                    <h4 className="font-medium">Top Discovered Materials</h4>
                                  </div>
                                  <div className="grid gap-3">
                                    {candidates.slice(0, 5).map((candidate: any, idx: number) => (
                                      <div key={idx} className="bg-muted/30 rounded-lg p-4 flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center font-bold text-amber-600">
                                          #{idx + 1}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                          <p className="font-medium truncate">{candidate.formula || candidate.name}</p>
                                          <p className="text-sm text-muted-foreground">{candidate.materialType || "Crystal"}</p>
                                        </div>
                                        <div className="grid grid-cols-3 gap-4 text-sm">
                                          {candidate.score && (
                                            <div className="text-center">
                                              <p className="font-semibold text-emerald-600">{(candidate.score * 100).toFixed(1)}%</p>
                                              <p className="text-xs text-muted-foreground">Score</p>
                                            </div>
                                          )}
                                          {candidate.targetProperty && (
                                            <div className="text-center">
                                              <p className="font-semibold">{candidate.predictedValue?.toFixed(2) || "N/A"}</p>
                                              <p className="text-xs text-muted-foreground">{candidate.targetProperty}</p>
                                            </div>
                                          )}
                                          {candidate.confidence && (
                                            <div className="text-center">
                                              <p className="font-semibold">{(candidate.confidence * 100).toFixed(0)}%</p>
                                              <p className="text-xs text-muted-foreground">Confidence</p>
                                            </div>
                                          )}
                                        </div>
                                        <Badge className="bg-emerald-500/10 text-emerald-600 border-0">
                                          <TrendingUp className="h-3 w-3 mr-1" />
                                          Promising
                                        </Badge>
                                      </div>
                                    ))}
                                  </div>
                                  {candidates.length > 5 && (
                                    <Link href={`/pipeline/results/${job.id}`}>
                                      <p className="text-sm text-muted-foreground text-center cursor-pointer hover:underline" data-testid={`link-more-candidates-${job.id}`}>
                                        + {candidates.length - 5} more candidates in full results →
                                      </p>
                                    </Link>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                </CardContent>
              </TabsContent>
            </Tabs>
          </Card>
        </div>
      </div>
    </div>
  );
}
