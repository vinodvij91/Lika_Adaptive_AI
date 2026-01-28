import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  Box, 
  Upload, 
  Play, 
  Settings2, 
  Layers,
  Thermometer,
  Wind,
  Activity,
  ChevronRight,
  Clock,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  FileBox,
  Cpu,
  Zap,
  BarChart3,
  Download,
  Eye,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Scale,
  Beaker,
  FlaskConical
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface FEAJob {
  id: string;
  name: string;
  simulationType: "structural" | "thermal" | "cfd";
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  fileName: string;
  createdAt: string;
  completedAt?: string;
  results?: {
    maxStress?: number;
    maxDisplacement?: number;
    maxTemperature?: number;
    convergence?: boolean;
  };
}

const SIMULATION_TYPES = [
  { 
    id: "structural", 
    name: "Structural Mechanics", 
    icon: Box, 
    description: "Stress, strain, and displacement analysis",
    color: "from-blue-500 to-cyan-500",
    available: true
  },
  { 
    id: "thermal", 
    name: "Thermal Analysis", 
    icon: Thermometer, 
    description: "Heat transfer and temperature distribution",
    color: "from-orange-500 to-red-500",
    available: true
  },
  { 
    id: "cfd", 
    name: "Fluid Dynamics (CFD)", 
    icon: Wind, 
    description: "Flow simulation and pressure analysis",
    color: "from-emerald-500 to-teal-500",
    available: true
  },
];

const MATERIAL_PRESETS = [
  // Metals
  { name: "Steel (AISI 1045)", youngsModulus: 200, poissonsRatio: 0.29, density: 7850, thermalConductivity: 51.9, specificHeat: 486, category: "metal" },
  { name: "Aluminum 6061-T6", youngsModulus: 68.9, poissonsRatio: 0.33, density: 2700, thermalConductivity: 167, specificHeat: 896, category: "metal" },
  { name: "Titanium Ti-6Al-4V", youngsModulus: 113.8, poissonsRatio: 0.34, density: 4430, thermalConductivity: 6.7, specificHeat: 526, category: "metal" },
  { name: "Copper C11000", youngsModulus: 117, poissonsRatio: 0.34, density: 8940, thermalConductivity: 388, specificHeat: 385, category: "metal" },
  { name: "Stainless Steel 316L", youngsModulus: 193, poissonsRatio: 0.27, density: 8000, thermalConductivity: 16.3, specificHeat: 500, category: "metal" },
  // Composites
  { name: "Carbon Fiber Composite", youngsModulus: 135, poissonsRatio: 0.30, density: 1600, thermalConductivity: 7, specificHeat: 1000, category: "composite" },
  // High-Performance Polymers
  { name: "PEEK", youngsModulus: 3.6, poissonsRatio: 0.38, density: 1320, thermalConductivity: 0.25, specificHeat: 2000, category: "polymer" },
  { name: "PTFE (Teflon)", youngsModulus: 0.5, poissonsRatio: 0.46, density: 2200, thermalConductivity: 0.25, specificHeat: 1000, category: "polymer" },
  { name: "PCTFE (Kel-F)", youngsModulus: 1.3, poissonsRatio: 0.36, density: 2100, thermalConductivity: 0.22, specificHeat: 900, category: "polymer" },
  { name: "PVDF (Kynar)", youngsModulus: 2.0, poissonsRatio: 0.34, density: 1780, thermalConductivity: 0.19, specificHeat: 1200, category: "polymer" },
  { name: "ECTFE (Halar)", youngsModulus: 1.7, poissonsRatio: 0.35, density: 1680, thermalConductivity: 0.16, specificHeat: 1100, category: "polymer" },
  { name: "POM (Delrin)", youngsModulus: 2.9, poissonsRatio: 0.35, density: 1410, thermalConductivity: 0.31, specificHeat: 1500, category: "polymer" },
  { name: "PPS (Ryton)", youngsModulus: 3.8, poissonsRatio: 0.36, density: 1350, thermalConductivity: 0.29, specificHeat: 1090, category: "polymer" },
  { name: "LCP (Vectra)", youngsModulus: 12.0, poissonsRatio: 0.35, density: 1400, thermalConductivity: 0.20, specificHeat: 1000, category: "polymer" },
  { name: "PES/PSU (Udel)", youngsModulus: 2.5, poissonsRatio: 0.37, density: 1370, thermalConductivity: 0.22, specificHeat: 1130, category: "polymer" },
  { name: "Polyimide (Kapton)", youngsModulus: 3.2, poissonsRatio: 0.34, density: 1420, thermalConductivity: 0.12, specificHeat: 1090, category: "polymer" },
  { name: "PDMS (Silicone)", youngsModulus: 0.002, poissonsRatio: 0.49, density: 970, thermalConductivity: 0.15, specificHeat: 1460, category: "polymer" },
  // Custom
  { name: "Custom", youngsModulus: 0, poissonsRatio: 0, density: 0, thermalConductivity: 0, specificHeat: 0, category: "custom" },
];

const MESH_QUALITY_OPTIONS = [
  { value: "coarse", label: "Coarse", description: "Fast computation, lower accuracy", elements: "~1K-5K" },
  { value: "medium", label: "Medium", description: "Balanced speed and accuracy", elements: "~10K-50K" },
  { value: "fine", label: "Fine", description: "High accuracy, longer computation", elements: "~100K-500K" },
  { value: "ultra", label: "Ultra Fine", description: "Maximum accuracy, GPU recommended", elements: "~1M+" },
];

export default function FEASimulationsPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("new");
  const [simulationType, setSimulationType] = useState<string>("structural");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [jobName, setJobName] = useState("");
  const [selectedMaterial, setSelectedMaterial] = useState(MATERIAL_PRESETS[0]);
  const [meshQuality, setMeshQuality] = useState("medium");

  // Structural mechanics parameters
  const [fixedFaces, setFixedFaces] = useState("bottom");
  const [loadType, setLoadType] = useState("force");
  const [loadMagnitude, setLoadMagnitude] = useState("1000");
  const [loadDirection, setLoadDirection] = useState("-z");

  // Thermal parameters
  const [heatSource, setHeatSource] = useState("1000");
  const [ambientTemp, setAmbientTemp] = useState("25");
  const [convectionCoeff, setConvectionCoeff] = useState("10");

  // CFD parameters
  const [inletVelocity, setInletVelocity] = useState("1.0");
  const [fluidDensity, setFluidDensity] = useState("1.225");
  const [fluidViscosity, setFluidViscosity] = useState("1.81e-5");

  // Material comparison state
  const [baselineMaterial, setBaselineMaterial] = useState(MATERIAL_PRESETS[0]);
  const [alternativeMaterial, setAlternativeMaterial] = useState(MATERIAL_PRESETS[7]); // PTFE by default

  // Calculate comparison metrics
  const calculateComparison = (baseline: typeof MATERIAL_PRESETS[0], alternative: typeof MATERIAL_PRESETS[0]) => {
    const metrics = [
      {
        property: "Young's Modulus",
        unit: "GPa",
        baseline: baseline.youngsModulus,
        alternative: alternative.youngsModulus,
        higherIsBetter: true,
        description: "Stiffness - higher means more rigid"
      },
      {
        property: "Density",
        unit: "kg/m³",
        baseline: baseline.density,
        alternative: alternative.density,
        higherIsBetter: false,
        description: "Weight - lower is often better for lightweight designs"
      },
      {
        property: "Thermal Conductivity",
        unit: "W/m·K",
        baseline: baseline.thermalConductivity,
        alternative: alternative.thermalConductivity,
        higherIsBetter: null, // Depends on application
        description: "Heat transfer - depends on application"
      },
      {
        property: "Specific Heat",
        unit: "J/kg·K",
        baseline: baseline.specificHeat,
        alternative: alternative.specificHeat,
        higherIsBetter: null,
        description: "Heat capacity - depends on application"
      },
      {
        property: "Poisson's Ratio",
        unit: "",
        baseline: baseline.poissonsRatio,
        alternative: alternative.poissonsRatio,
        higherIsBetter: null,
        description: "Lateral vs axial strain ratio"
      }
    ];

    return metrics.map(m => {
      const change = m.baseline > 0 ? ((m.alternative - m.baseline) / m.baseline) * 100 : 0;
      let status: "better" | "worse" | "neutral" = "neutral";
      if (m.higherIsBetter !== null && Math.abs(change) > 5) {
        if (m.higherIsBetter) {
          status = change > 0 ? "better" : "worse";
        } else {
          status = change < 0 ? "better" : "worse";
        }
      }
      return { ...m, change, status };
    });
  };

  const comparisonMetrics = calculateComparison(baselineMaterial, alternativeMaterial);

  // Fetch FEA jobs from API with polling for status updates
  const { data: jobs = [], isLoading: isLoadingJobs, refetch: refetchJobs } = useQuery<FEAJob[]>({
    queryKey: ["/api/fea/jobs"],
    refetchInterval: 5000, // Poll every 5 seconds to catch status updates
  });

  // Submit job mutation
  const submitJobMutation = useMutation({
    mutationFn: async (jobData: {
      name: string;
      simulationType: string;
      fileName: string;
      meshQuality: string;
      material: typeof MATERIAL_PRESETS[0];
      parameters: Record<string, string>;
    }) => {
      const response = await apiRequest("POST", "/api/fea/jobs", jobData);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Simulation queued",
        description: `${jobName} has been submitted to the compute cluster`
      });
      queryClient.invalidateQueries({ queryKey: ["/api/fea/jobs"] });
      setActiveTab("history");
      setUploadedFile(null);
      setJobName("");
    },
    onError: (error: Error) => {
      toast({
        title: "Submission failed",
        description: error.message || "Failed to submit simulation job",
        variant: "destructive"
      });
    }
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const validExtensions = ['.stl', '.step', '.stp', '.obj', '.iges', '.igs'];
      const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      if (!validExtensions.includes(ext)) {
        toast({
          title: "Invalid file type",
          description: "Please upload STL, STEP, OBJ, or IGES files",
          variant: "destructive"
        });
        return;
      }
      setUploadedFile(file);
      if (!jobName) {
        setJobName(file.name.replace(/\.[^/.]+$/, "") + " Analysis");
      }
      toast({
        title: "File uploaded",
        description: `${file.name} ready for simulation`
      });
    }
  };

  const handleMaterialChange = (materialName: string) => {
    const material = MATERIAL_PRESETS.find(m => m.name === materialName);
    if (material) {
      setSelectedMaterial(material);
    }
  };

  const handleSubmitJob = async () => {
    if (!uploadedFile) {
      toast({ title: "No file uploaded", description: "Please upload a CAD file first", variant: "destructive" });
      return;
    }
    if (!jobName.trim()) {
      toast({ title: "Job name required", description: "Please enter a name for this simulation", variant: "destructive" });
      return;
    }

    // Build parameters based on simulation type
    let parameters: Record<string, string> = {};
    if (simulationType === "structural") {
      parameters = { fixedFaces, loadType, loadMagnitude, loadDirection };
    } else if (simulationType === "thermal") {
      parameters = { heatSource, ambientTemp, convectionCoeff };
    } else if (simulationType === "cfd") {
      parameters = { inletVelocity, fluidDensity, fluidViscosity };
    }

    submitJobMutation.mutate({
      name: jobName,
      simulationType,
      fileName: uploadedFile.name,
      meshQuality,
      material: selectedMaterial,
      parameters
    });
  };

  const getStatusBadge = (status: FEAJob["status"]) => {
    switch (status) {
      case "completed":
        return <Badge className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30"><CheckCircle2 className="h-3 w-3 mr-1" />Completed</Badge>;
      case "running":
        return <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/30"><RefreshCw className="h-3 w-3 mr-1 animate-spin" />Running</Badge>;
      case "queued":
        return <Badge className="bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/30"><Clock className="h-3 w-3 mr-1" />Queued</Badge>;
      case "failed":
        return <Badge className="bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/30"><AlertCircle className="h-3 w-3 mr-1" />Failed</Badge>;
    }
  };

  const getSimulationIcon = (type: string) => {
    switch (type) {
      case "structural": return <Box className="h-4 w-4" />;
      case "thermal": return <Thermometer className="h-4 w-4" />;
      case "cfd": return <Wind className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-blue-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          {/* Header */}
          <header className="relative overflow-hidden rounded-md bg-gradient-to-r from-blue-600 via-indigo-500 to-violet-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0xMCAzMGgyME0zMCAxMHYyMCIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-md bg-white/20 backdrop-blur flex items-center justify-center">
                    <Activity className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">FEA Simulations</h1>
                    <p className="text-blue-100">Finite Element Analysis for structural, thermal, and fluid dynamics</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-blue-100">
                  <Cpu className="h-4 w-4" />
                  <span>Powered by FEniCS, OpenFOAM, and GPU acceleration</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Badge className="bg-white/20 text-white border-white/30">
                  <Zap className="h-3 w-3 mr-1" />
                  2x RTX 3090 Available
                </Badge>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full max-w-lg grid-cols-3">
              <TabsTrigger value="new" className="gap-2" data-testid="tab-new-simulation">
                <Play className="h-4 w-4" />
                New Simulation
              </TabsTrigger>
              <TabsTrigger value="compare" className="gap-2" data-testid="tab-material-compare">
                <Scale className="h-4 w-4" />
                Material Compare
              </TabsTrigger>
              <TabsTrigger value="history" className="gap-2" data-testid="tab-history">
                <Clock className="h-4 w-4" />
                Job History
              </TabsTrigger>
            </TabsList>

            {/* New Simulation Tab */}
            <TabsContent value="new" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - File Upload & Preview */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Simulation Type Selection */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Simulation Type</CardTitle>
                      <CardDescription>Select the type of analysis to perform</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {SIMULATION_TYPES.map((type) => (
                          <div
                            key={type.id}
                            onClick={() => setSimulationType(type.id)}
                            className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all ${
                              simulationType === type.id
                                ? "border-primary bg-primary/5"
                                : "border-border hover-elevate"
                            }`}
                            data-testid={`simulation-type-${type.id}`}
                          >
                            <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${type.color} flex items-center justify-center text-white mb-3`}>
                              <type.icon className="h-5 w-5" />
                            </div>
                            <h3 className="font-medium mb-1">{type.name}</h3>
                            <p className="text-xs text-muted-foreground">{type.description}</p>
                            {simulationType === type.id && (
                              <div className="absolute top-2 right-2">
                                <CheckCircle2 className="h-5 w-5 text-primary" />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* File Upload */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Upload Geometry</CardTitle>
                      <CardDescription>Upload your CAD file (STL, STEP, OBJ, IGES)</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {!uploadedFile ? (
                        <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-md cursor-pointer hover-elevate transition-colors" data-testid="label-file-upload">
                          <div className="flex flex-col items-center justify-center pt-5 pb-6">
                            <Upload className="h-10 w-10 text-muted-foreground mb-3" />
                            <p className="mb-2 text-sm">
                              <span className="font-semibold">Click to upload</span> or drag and drop
                            </p>
                            <p className="text-xs text-muted-foreground">STL, STEP, OBJ, IGES (max 100MB)</p>
                          </div>
                          <input
                            type="file"
                            className="hidden"
                            accept=".stl,.step,.stp,.obj,.iges,.igs"
                            onChange={handleFileUpload}
                            data-testid="input-file-upload"
                          />
                        </label>
                      ) : (
                        <div className="space-y-4">
                          <div className="flex items-center gap-4 p-4 bg-muted/50 rounded-lg">
                            <FileBox className="h-10 w-10 text-primary" />
                            <div className="flex-1">
                              <p className="font-medium">{uploadedFile.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                              </p>
                            </div>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => setUploadedFile(null)}
                              data-testid="button-remove-file"
                            >
                              Remove
                            </Button>
                          </div>
                          
                          {/* 3D Preview Placeholder */}
                          <div className="h-64 bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg flex items-center justify-center">
                            <div className="text-center text-slate-400">
                              <Box className="h-16 w-16 mx-auto mb-3 opacity-50" />
                              <p className="text-sm">3D Preview</p>
                              <p className="text-xs opacity-75">WebGL visualization loading...</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Simulation Parameters */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Simulation Parameters</CardTitle>
                      <CardDescription>Configure analysis settings</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Job Name */}
                      <div className="space-y-2">
                        <Label htmlFor="job-name">Job Name</Label>
                        <Input
                          id="job-name"
                          placeholder="e.g., Bracket Stress Analysis"
                          value={jobName}
                          onChange={(e) => setJobName(e.target.value)}
                          data-testid="input-job-name"
                        />
                      </div>

                      {/* Mesh Quality */}
                      <div className="space-y-2">
                        <Label>Mesh Quality</Label>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {MESH_QUALITY_OPTIONS.map((option) => (
                            <div
                              key={option.value}
                              onClick={() => setMeshQuality(option.value)}
                              className={`p-3 rounded-lg border cursor-pointer transition-all text-center ${
                                meshQuality === option.value
                                  ? "border-primary bg-primary/5"
                                  : "border-border hover-elevate"
                              }`}
                              data-testid={`mesh-quality-${option.value}`}
                            >
                              <p className="font-medium text-sm">{option.label}</p>
                              <p className="text-xs text-muted-foreground">{option.elements}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <Separator />

                      {/* Structural Mechanics Parameters */}
                      {simulationType === "structural" && (
                        <div className="space-y-4">
                          <h4 className="font-medium flex items-center gap-2">
                            <Box className="h-4 w-4" />
                            Structural Analysis Settings
                          </h4>
                          
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label>Fixed Boundary</Label>
                              <Select value={fixedFaces} onValueChange={setFixedFaces}>
                                <SelectTrigger data-testid="select-fixed-faces">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="bottom">Bottom Face</SelectItem>
                                  <SelectItem value="top">Top Face</SelectItem>
                                  <SelectItem value="left">Left Face</SelectItem>
                                  <SelectItem value="right">Right Face</SelectItem>
                                  <SelectItem value="auto">Auto-detect</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-2">
                              <Label>Load Type</Label>
                              <Select value={loadType} onValueChange={setLoadType}>
                                <SelectTrigger data-testid="select-load-type">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="force">Point Force (N)</SelectItem>
                                  <SelectItem value="pressure">Pressure (Pa)</SelectItem>
                                  <SelectItem value="displacement">Displacement (mm)</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-2">
                              <Label>Load Magnitude</Label>
                              <Input
                                type="number"
                                value={loadMagnitude}
                                onChange={(e) => setLoadMagnitude(e.target.value)}
                                placeholder="1000"
                                data-testid="input-load-magnitude"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label>Load Direction</Label>
                              <Select value={loadDirection} onValueChange={setLoadDirection}>
                                <SelectTrigger data-testid="select-load-direction">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="-z">-Z (Down)</SelectItem>
                                  <SelectItem value="+z">+Z (Up)</SelectItem>
                                  <SelectItem value="-y">-Y</SelectItem>
                                  <SelectItem value="+y">+Y</SelectItem>
                                  <SelectItem value="-x">-X</SelectItem>
                                  <SelectItem value="+x">+X</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Thermal Parameters */}
                      {simulationType === "thermal" && (
                        <div className="space-y-4">
                          <h4 className="font-medium flex items-center gap-2">
                            <Thermometer className="h-4 w-4" />
                            Thermal Analysis Settings
                          </h4>
                          
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="space-y-2">
                              <Label>Heat Source (W)</Label>
                              <Input
                                type="number"
                                value={heatSource}
                                onChange={(e) => setHeatSource(e.target.value)}
                                data-testid="input-heat-source"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label>Ambient Temp (C)</Label>
                              <Input
                                type="number"
                                value={ambientTemp}
                                onChange={(e) => setAmbientTemp(e.target.value)}
                                data-testid="input-ambient-temp"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label>Convection Coeff (W/m2K)</Label>
                              <Input
                                type="number"
                                value={convectionCoeff}
                                onChange={(e) => setConvectionCoeff(e.target.value)}
                                data-testid="input-convection-coeff"
                              />
                            </div>
                          </div>
                        </div>
                      )}

                      {/* CFD Parameters */}
                      {simulationType === "cfd" && (
                        <div className="space-y-4">
                          <h4 className="font-medium flex items-center gap-2">
                            <Wind className="h-4 w-4" />
                            CFD Analysis Settings
                          </h4>
                          
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="space-y-2">
                              <Label>Inlet Velocity (m/s)</Label>
                              <Input
                                type="number"
                                value={inletVelocity}
                                onChange={(e) => setInletVelocity(e.target.value)}
                                data-testid="input-inlet-velocity"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label>Fluid Density (kg/m3)</Label>
                              <Input
                                type="number"
                                value={fluidDensity}
                                onChange={(e) => setFluidDensity(e.target.value)}
                                data-testid="input-fluid-density"
                              />
                            </div>

                            <div className="space-y-2">
                              <Label>Dynamic Viscosity (Pa.s)</Label>
                              <Input
                                type="text"
                                value={fluidViscosity}
                                onChange={(e) => setFluidViscosity(e.target.value)}
                                data-testid="input-fluid-viscosity"
                              />
                            </div>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>

                {/* Right Column - Material & Submit */}
                <div className="space-y-6">
                  {/* Material Selection */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Material Properties</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label>Material Preset</Label>
                        <Select 
                          value={selectedMaterial.name} 
                          onValueChange={handleMaterialChange}
                        >
                          <SelectTrigger data-testid="select-material">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {MATERIAL_PRESETS.map((mat) => (
                              <SelectItem key={mat.name} value={mat.name}>
                                {mat.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <Separator />

                      <div className="space-y-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Young's Modulus</span>
                          <span className="font-mono">{selectedMaterial.youngsModulus} GPa</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Poisson's Ratio</span>
                          <span className="font-mono">{selectedMaterial.poissonsRatio}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Density</span>
                          <span className="font-mono">{selectedMaterial.density} kg/m3</span>
                        </div>
                        {simulationType === "thermal" && (
                          <>
                            <Separator />
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Thermal Conductivity</span>
                              <span className="font-mono">{selectedMaterial.thermalConductivity} W/mK</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Specific Heat</span>
                              <span className="font-mono">{selectedMaterial.specificHeat} J/kgK</span>
                            </div>
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Compute Resources */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Compute Resources</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="p-3 bg-muted/50 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Cpu className="h-4 w-4" />
                          <span className="font-medium text-sm">Hetzner CPU Node</span>
                        </div>
                        <p className="text-xs text-muted-foreground">8 cores, 32GB RAM - Best for coarse/medium meshes</p>
                      </div>
                      <div className="p-3 bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-lg border border-emerald-500/30">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="h-4 w-4 text-emerald-600" />
                          <span className="font-medium text-sm">Vast.ai GPU Node</span>
                          <Badge variant="secondary" className="text-xs">Recommended</Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">2x RTX 3090, 48 cores - Best for fine/ultra meshes</p>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Submit Button */}
                  <Button
                    size="lg"
                    className="w-full gap-2 bg-gradient-to-r from-blue-500 to-indigo-500"
                    onClick={handleSubmitJob}
                    disabled={!uploadedFile || !jobName || submitJobMutation.isPending}
                    data-testid="button-submit-simulation"
                  >
                    {submitJobMutation.isPending ? (
                      <>
                        <RefreshCw className="h-5 w-5 animate-spin" />
                        Submitting...
                      </>
                    ) : (
                      <>
                        <Play className="h-5 w-5" />
                        Run Simulation
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </TabsContent>

            {/* Material Comparison Tab */}
            <TabsContent value="compare" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Baseline Material */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Box className="h-5 w-5 text-blue-500" />
                      Current Material (Baseline)
                    </CardTitle>
                    <CardDescription>Select the material currently used in your component</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Select 
                      value={baselineMaterial.name} 
                      onValueChange={(name) => {
                        const mat = MATERIAL_PRESETS.find(m => m.name === name);
                        if (mat) setBaselineMaterial(mat);
                      }}
                    >
                      <SelectTrigger data-testid="select-baseline-material">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {MATERIAL_PRESETS.filter(m => m.name !== "Custom").map((mat) => (
                          <SelectItem key={mat.name} value={mat.name}>
                            {mat.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <div className="p-4 rounded-md bg-muted/50 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Young's Modulus</span>
                        <span className="font-mono">{baselineMaterial.youngsModulus} GPa</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Density</span>
                        <span className="font-mono">{baselineMaterial.density} kg/m³</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Thermal Conductivity</span>
                        <span className="font-mono">{baselineMaterial.thermalConductivity} W/m·K</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Specific Heat</span>
                        <span className="font-mono">{baselineMaterial.specificHeat} J/kg·K</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Alternative Material */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <FlaskConical className="h-5 w-5 text-green-500" />
                      Alternative Material
                    </CardTitle>
                    <CardDescription>Select a discovered or alternative material to compare</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Select 
                      value={alternativeMaterial.name} 
                      onValueChange={(name) => {
                        const mat = MATERIAL_PRESETS.find(m => m.name === name);
                        if (mat) setAlternativeMaterial(mat);
                      }}
                    >
                      <SelectTrigger data-testid="select-alternative-material">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {MATERIAL_PRESETS.filter(m => m.name !== "Custom").map((mat) => (
                          <SelectItem key={mat.name} value={mat.name}>
                            {mat.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <div className="p-4 rounded-md bg-muted/50 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Young's Modulus</span>
                        <span className="font-mono">{alternativeMaterial.youngsModulus} GPa</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Density</span>
                        <span className="font-mono">{alternativeMaterial.density} kg/m³</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Thermal Conductivity</span>
                        <span className="font-mono">{alternativeMaterial.thermalConductivity} W/m·K</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Specific Heat</span>
                        <span className="font-mono">{alternativeMaterial.specificHeat} J/kg·K</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Comparison Results */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Property Comparison
                  </CardTitle>
                  <CardDescription>
                    Compare {baselineMaterial.name} vs {alternativeMaterial.name}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {comparisonMetrics.map((metric) => (
                      <div key={metric.property} className="space-y-2">
                        <div className="flex items-center justify-between gap-4 flex-wrap">
                          <div className="flex-1 min-w-[200px]">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{metric.property}</span>
                              {metric.unit && <span className="text-xs text-muted-foreground">({metric.unit})</span>}
                            </div>
                            <p className="text-xs text-muted-foreground">{metric.description}</p>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="text-right">
                              <div className="text-xs text-muted-foreground">Baseline</div>
                              <div className="font-mono text-sm">{metric.baseline.toLocaleString()}</div>
                            </div>
                            <ArrowRight className="h-4 w-4 text-muted-foreground" />
                            <div className="text-right">
                              <div className="text-xs text-muted-foreground">Alternative</div>
                              <div className="font-mono text-sm">{metric.alternative.toLocaleString()}</div>
                            </div>
                            <div className="w-24 text-right">
                              <Badge 
                                className={`gap-1 ${
                                  metric.status === "better" 
                                    ? "bg-green-500/10 text-green-600 border-green-500/20" 
                                    : metric.status === "worse"
                                    ? "bg-red-500/10 text-red-600 border-red-500/20"
                                    : "bg-muted text-muted-foreground"
                                }`}
                                data-testid={`badge-comparison-${metric.property.toLowerCase().replace(/[^a-z]/g, '-')}`}
                              >
                                {metric.status === "better" && <TrendingUp className="h-3 w-3" />}
                                {metric.status === "worse" && <TrendingDown className="h-3 w-3" />}
                                {metric.status === "neutral" && <Minus className="h-3 w-3" />}
                                {metric.change > 0 ? "+" : ""}{metric.change.toFixed(1)}%
                              </Badge>
                            </div>
                          </div>
                        </div>
                        <div className="relative h-2 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="absolute inset-y-0 left-0 bg-blue-500 rounded-full"
                            style={{ 
                              width: `${Math.min(100, (metric.baseline / Math.max(metric.baseline, metric.alternative)) * 100)}%` 
                            }}
                          />
                          <div 
                            className="absolute inset-y-0 right-0 bg-green-500 rounded-full"
                            style={{ 
                              width: `${Math.min(100, (metric.alternative / Math.max(metric.baseline, metric.alternative)) * 100)}%`,
                              opacity: 0.5
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <Separator className="my-6" />

                  {/* Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-md bg-green-500/10 border border-green-500/20">
                      <div className="text-2xl font-bold text-green-600">
                        {comparisonMetrics.filter(m => m.status === "better").length}
                      </div>
                      <div className="text-sm text-muted-foreground">Properties Improved</div>
                    </div>
                    <div className="p-4 rounded-md bg-red-500/10 border border-red-500/20">
                      <div className="text-2xl font-bold text-red-600">
                        {comparisonMetrics.filter(m => m.status === "worse").length}
                      </div>
                      <div className="text-sm text-muted-foreground">Properties Reduced</div>
                    </div>
                    <div className="p-4 rounded-md bg-muted">
                      <div className="text-2xl font-bold">
                        {comparisonMetrics.filter(m => m.status === "neutral").length}
                      </div>
                      <div className="text-sm text-muted-foreground">Application Dependent</div>
                    </div>
                  </div>

                  <div className="mt-6 flex items-center gap-3 flex-wrap">
                    <Button 
                      className="gap-2"
                      onClick={() => {
                        setSelectedMaterial(alternativeMaterial);
                        setActiveTab("new");
                        toast({
                          title: "Material selected",
                          description: `${alternativeMaterial.name} is now selected for simulation`
                        });
                      }}
                      data-testid="button-use-alternative-material"
                    >
                      <Play className="h-4 w-4" />
                      Run Simulation with {alternativeMaterial.name}
                    </Button>
                    <Button variant="outline" className="gap-2" data-testid="button-export-comparison">
                      <Download className="h-4 w-4" />
                      Export Comparison Report
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Job History Tab */}
            <TabsContent value="history">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-4">
                  <div>
                    <CardTitle className="text-lg">Simulation History</CardTitle>
                    <CardDescription>View and manage your FEA jobs</CardDescription>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => refetchJobs()} data-testid="button-refresh-jobs">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <Table className="min-w-[700px]">
                      <TableHeader>
                        <TableRow>
                          <TableHead>Job Name</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>File</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Progress</TableHead>
                          <TableHead>Results</TableHead>
                          <TableHead className="w-[100px]">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {isLoadingJobs ? (
                          [...Array(3)].map((_, i) => (
                            <TableRow key={i}>
                              <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                              <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                              <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                              <TableCell><Skeleton className="h-6 w-20" /></TableCell>
                              <TableCell><Skeleton className="h-2 w-24" /></TableCell>
                              <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                              <TableCell><Skeleton className="h-8 w-16" /></TableCell>
                            </TableRow>
                          ))
                        ) : jobs.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                              No simulation jobs yet. Create your first simulation above.
                            </TableCell>
                          </TableRow>
                        ) : jobs.map((job) => (
                          <TableRow key={job.id} data-testid={`row-job-${job.id}`}>
                            <TableCell className="font-medium">{job.name}</TableCell>
                            <TableCell>
                              <div className="flex items-center gap-2">
                                {getSimulationIcon(job.simulationType)}
                                <span className="capitalize">{job.simulationType}</span>
                              </div>
                            </TableCell>
                            <TableCell className="text-muted-foreground">{job.fileName}</TableCell>
                            <TableCell>{getStatusBadge(job.status)}</TableCell>
                            <TableCell>
                              <div className="flex items-center gap-2 min-w-[100px]">
                                <Progress value={job.progress} className="h-2 flex-1" />
                                <span className="text-xs text-muted-foreground">{job.progress}%</span>
                              </div>
                            </TableCell>
                            <TableCell>
                              {job.results ? (
                                <div className="text-xs space-y-0.5">
                                  {job.results.maxStress && <div>Max Stress: {job.results.maxStress} MPa</div>}
                                  {job.results.maxDisplacement && <div>Max Disp: {job.results.maxDisplacement} mm</div>}
                                  {job.results.maxTemperature && <div>Max Temp: {job.results.maxTemperature} C</div>}
                                </div>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-1">
                                {job.status === "completed" && (
                                  <>
                                    <Button variant="ghost" size="icon" data-testid={`button-view-${job.id}`}>
                                      <Eye className="h-4 w-4" />
                                    </Button>
                                    <Button variant="ghost" size="icon" data-testid={`button-download-${job.id}`}>
                                      <Download className="h-4 w-4" />
                                    </Button>
                                  </>
                                )}
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
