import { useState, useEffect, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import { useActivityLog } from "@/hooks/use-activity-log";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  GitBranch,
  Database,
  Play,
  Download,
  RefreshCw,
  Target,
  Activity,
  TrendingUp,
  TrendingDown,
  Atom,
  Microscope,
  FlaskConical,
  Loader2,
  ChevronRight,
  Zap,
  Brain,
  Dna,
  ClipboardCopy,
  Check,
  ArrowRight,
  Settings2,
  Plus,
  FolderPlus,
} from "lucide-react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useLocation } from "wouter";

interface ScRNADataset {
  id: string;
  name: string;
  geoAccession: string;
  disease: string;
  tissue: string;
  cellCount: number;
  description: string;
  source: "GEO" | "ArrayExpress" | "HCA";
  species: "human" | "mouse";
  publicationYear: number;
  hasTrajectory: boolean;
}

interface TrajectoryResult {
  datasetId: string;
  disease: string;
  umapCoordinates: Array<{ x: number; y: number; cluster: string; pseudotime: number }>;
  clusters: Array<{ id: string; name: string; cellCount: number; color: string }>;
  branchPoints: Array<{ pseudotime: number; genes: string[]; significance: number }>;
  biomarkers: Array<{
    gene: string;
    pseudotimeExpression: number;
    direction: "up" | "down";
    foldChange: number;
    pValue: number;
    cluster: string;
  }>;
  targets: string[];
  pgdSmoothing: {
    alpha: number;
    iterations: number;
    convergence: number;
  };
}

interface AssayTemplate {
  targetGene: string;
  disease: string;
  role: string;
  targetable: boolean;
  suggestedAssays: Array<{
    type: string;
    name: string;
    description: string;
    readoutType: string;
    technique: string;
  }>;
}

interface DiseaseInfo {
  disease: string;
  datasetCount: number;
  totalCells: number;
  biomarkers: number;
}

interface InhibitorPrediction {
  smiles: string;
  targetGene: string;
  predictedIC50_nM: number;
  predictedPIC50: number;
  confidence: number;
  isHit: boolean;
  rank: number;
}

interface InhibitorResult {
  gene: string;
  disease: string;
  totalCompounds: number;
  hits: number;
  predictions: InhibitorPrediction[];
}

interface Campaign {
  id: string;
  name: string;
  projectId: string;
  domainType?: string;
  status?: string;
  createdAt?: string;
  pipelineConfig?: Record<string, unknown> | null;
}

interface Project {
  id: string;
  name: string;
}

export default function TrajectoryAnalysisPage() {
  const { toast } = useToast();
  const [, navigate] = useLocation();
  const { logAnalysisRun, logDataImport } = useActivityLog();
  const [selectedDisease, setSelectedDisease] = useState<string>("");
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [trajectoryResult, setTrajectoryResult] = useState<TrajectoryResult | null>(null);
  const [selectedBiomarker, setSelectedBiomarker] = useState<string | null>(null);
  const [assayTemplate, setAssayTemplate] = useState<AssayTemplate | null>(null);
  const [showAssayDialog, setShowAssayDialog] = useState(false);
  const [smoothingAlpha, setSmoothingAlpha] = useState([0.1]);
  const [copiedGene, setCopiedGene] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("datasets");
  const [showPredictionDialog, setShowPredictionDialog] = useState(false);
  const [selectedTargetForPrediction, setSelectedTargetForPrediction] = useState<string | null>(null);
  const [inhibitorResult, setInhibitorResult] = useState<InhibitorResult | null>(null);
  const [smilesInput, setSmilesInput] = useState("CCO\nCC(=O)O\nC1=CC=CC=C1\nCCCCCC\nC1=CC=C(C=C1)O\nCC(C)CC(C(=O)O)N\nC1CCCCC1\nC(C(=O)O)N\nCCCCCCCC\nC1=CC=C(C=C1)N");
  
  const [showCampaignDialog, setShowCampaignDialog] = useState(false);
  const [campaignMode, setCampaignMode] = useState<"new" | "existing">("new");
  const [selectedCampaignId, setSelectedCampaignId] = useState<string>("");
  const [newCampaignName, setNewCampaignName] = useState("");
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [pipelineType, setPipelineType] = useState<"drug" | "vaccine">("drug");
  const [selectedReadouts, setSelectedReadouts] = useState<Set<string>>(new Set());
  
  const isVaccineMode = pipelineType === "vaccine";

  const IMMUNE_READOUTS = {
    humoral: [
      { id: "neutralizing_titer", name: "Neutralizing antibody titer (MNA/PNA)", description: "Microneutralization or pseudovirus neutralization assay vs target strain", biomarkers: ["IGHG1", "IGHG3", "IGHA1"], timepoints: ["D0", "D14", "D28", "D56", "D180"] },
      { id: "spike_igg_elisa", name: "Spike/RBD IgG ELISA titer", description: "Binding antibody levels against spike or receptor-binding domain", biomarkers: ["IGHG1", "CD27", "PRDM1"], timepoints: ["D0", "D7", "D14", "D28", "D90"] },
      { id: "antibody_avidity", name: "Antibody avidity index", description: "Measure of antibody binding strength/maturation", biomarkers: ["AICDA", "BCL6", "CD38"], timepoints: ["D28", "D56", "D180"] },
      { id: "memory_bcell", name: "Antigen-specific memory B cells", description: "Flow cytometry for memory B cell frequencies", biomarkers: ["CD27", "CD38", "PAX5"], timepoints: ["D14", "D28", "D90", "D180"] },
      { id: "systems_serology", name: "Systems serology panel", description: "Comprehensive antibody profiling (Fc effector functions, subclasses)", biomarkers: ["FCGR3A", "IGHG1", "IGHG3", "IGHA1"], timepoints: ["D0", "D14", "D28", "D56"] },
    ],
    cellular: [
      { id: "ifng_elispot", name: "IFN-γ ELISpot (CD4/CD8)", description: "T cell IFN-γ secretion upon antigen stimulation", biomarkers: ["IFNG", "CD4", "CD8A"], timepoints: ["D0", "D14", "D28", "D56"] },
      { id: "polyfunctional_cd4", name: "Polyfunctional CD4 T cells (ICS)", description: "Intracellular cytokine staining for multi-cytokine producers", biomarkers: ["IL2", "IFNG", "TNF", "CD4"], timepoints: ["D14", "D28", "D56"] },
      { id: "cytotoxic_cd8", name: "Cytotoxic CD8+ T cell response", description: "Granzyme/perforin-expressing CD8 T cells", biomarkers: ["GZMB", "PRF1", "CD8A"], timepoints: ["D14", "D28", "D56"] },
      { id: "tfh_cells", name: "Circulating Tfh cell frequencies", description: "CXCR5+ PD-1+ helper T cells in blood", biomarkers: ["CXCR5", "PDCD1", "BCL6", "IL21"], timepoints: ["D7", "D14", "D28"] },
    ],
    innate: [
      { id: "ifn_signature", name: "Innate IFN signature score (D1-D3)", description: "Type I/III interferon gene expression from scRNA trajectories", biomarkers: ["IFITM1", "ISG15", "MX1", "OAS1"], timepoints: ["D1", "D2", "D3", "D7"] },
      { id: "cytokine_panel", name: "Serum IL-6, CRP, IP-10 panel", description: "Pro-inflammatory cytokine and chemokine levels", biomarkers: ["IL6", "CRP", "CXCL10"], timepoints: ["D0", "D1", "D3", "D7"] },
      { id: "monocyte_activation", name: "Monocyte/DC activation score", description: "CD14+ monocyte and dendritic cell activation markers", biomarkers: ["CD14", "CD80", "CD86", "HLA-DRA"], timepoints: ["D1", "D3", "D7"] },
    ],
  };

  const toggleReadout = (id: string) => {
    setSelectedReadouts(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const allReadouts = [...IMMUNE_READOUTS.humoral, ...IMMUNE_READOUTS.cellular, ...IMMUNE_READOUTS.innate];

  const { data: diseases, isLoading: diseasesLoading, refetch: refetchDiseases } = useQuery<DiseaseInfo[]>({
    queryKey: ["/api/trajectory/diseases", { pipelineType }],
    queryFn: async () => {
      const res = await fetch(`/api/trajectory/diseases?pipelineType=${pipelineType}`, {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to fetch diseases");
      return res.json();
    },
  });

  const { data: datasetsData, isLoading: datasetsLoading } = useQuery<{
    datasets: ScRNADataset[];
    totalCount: number;
    diseases: string[];
  }>({
    queryKey: ["/api/trajectory/datasets", { disease: selectedDisease, pipelineType }],
    queryFn: async () => {
      const res = await fetch(`/api/trajectory/datasets?disease=${encodeURIComponent(selectedDisease)}&pipelineType=${pipelineType}`, {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to fetch datasets");
      return res.json();
    },
    enabled: !!selectedDisease,
  });

  const datasets = datasetsData?.datasets || [];

  const analyzeMutation = useMutation({
    mutationFn: async (datasetId: string) => {
      const response = await apiRequest("POST", `/api/trajectory/analyze/${datasetId}`, { 
        smoothingAlpha: smoothingAlpha[0], 
        detectBiomarkers: true,
        pipelineType 
      });
      return await response.json() as TrajectoryResult;
    },
    onSuccess: (data) => {
      setTrajectoryResult(data);
      setActiveTab("trajectory");
      const biomarkerCount = (data.biomarkers || []).length;
      const targetCount = (data.targets || []).length;
      logAnalysisRun(
        "Trajectory Analysis Complete",
        `Found ${biomarkerCount} biomarkers and ${targetCount} druggable targets`,
        "scRNA_dataset",
        selectedDataset,
        { disease: selectedDisease, biomarkerCount, targetCount, smoothingAlpha: smoothingAlpha[0] }
      );
      toast({
        title: "Analysis Complete",
        description: `Found ${biomarkerCount} biomarkers and ${targetCount} druggable targets`,
      });
    },
    onError: () => {
      toast({
        title: "Analysis Failed",
        description: "Could not run trajectory analysis",
        variant: "destructive",
      });
    },
  });

  const generateTemplateMutation = useMutation({
    mutationFn: async (params: { gene: string; disease: string; pseudotime?: number; cellState?: string }) => {
      const response = await apiRequest("POST", "/api/trajectory/assay-template", { ...params, pipelineType });
      return await response.json() as AssayTemplate;
    },
    onSuccess: (data) => {
      setAssayTemplate(data);
      setShowAssayDialog(true);
    },
    onError: () => {
      toast({
        title: "Template Generation Failed",
        description: "Could not generate assay template",
        variant: "destructive",
      });
    },
  });

  const predictInhibitorsMutation = useMutation({
    mutationFn: async (params: { gene: string; disease: string; smilesList: string[] }) => {
      const response = await apiRequest("POST", "/api/trajectory/predict-inhibitors", { ...params, pipelineType });
      return await response.json() as InhibitorResult;
    },
    onSuccess: (data) => {
      setInhibitorResult(data);
      logAnalysisRun(
        "Inhibitor Prediction Complete",
        `Found ${data.hits} potential hits out of ${data.totalCompounds} compounds`,
        "target",
        selectedTargetForPrediction || undefined,
        { hits: data.hits, totalCompounds: data.totalCompounds, disease: selectedDisease }
      );
      toast({
        title: "Prediction Complete",
        description: `Found ${data.hits} potential hits out of ${data.totalCompounds} compounds`,
      });
    },
    onError: () => {
      toast({
        title: "Prediction Failed",
        description: "Could not run inhibitor predictions",
        variant: "destructive",
      });
    },
  });

  const { data: campaignsData } = useQuery<Campaign[]>({
    queryKey: ["/api/campaigns"],
    enabled: showCampaignDialog,
  });

  const { data: projectsData } = useQuery<Project[]>({
    queryKey: ["/api/projects"],
    enabled: showCampaignDialog,
  });

  const campaigns = campaignsData || [];
  const projects = projectsData || [];

  const createCampaignMutation = useMutation({
    mutationFn: async (params: { 
      name: string; 
      projectId: string; 
      pipelineConfig: Record<string, unknown>;
      domainType: string;
    }) => {
      const response = await apiRequest("POST", "/api/campaigns", params);
      return await response.json() as Campaign;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns"] });
      setShowCampaignDialog(false);
      setShowAssayDialog(false);
      toast({
        title: "Campaign Created",
        description: `${data.name} has been created with ${assayTemplate?.suggestedAssays?.length || 0} assays`,
      });
      navigate(`/campaigns/${data.id}`);
    },
    onError: () => {
      toast({
        title: "Campaign Creation Failed",
        description: "Could not create campaign",
        variant: "destructive",
      });
    },
  });

  const updateCampaignMutation = useMutation({
    mutationFn: async (params: { campaignId: string; assayConfig: Record<string, unknown>; existingConfig: Record<string, unknown> | null }) => {
      const mergedConfig = {
        ...params.existingConfig,
        trajectoryAssays: [
          ...((params.existingConfig as Record<string, unknown>)?.trajectoryAssays as Array<unknown> || []),
          params.assayConfig,
        ],
      };
      const response = await apiRequest("PATCH", `/api/campaigns/${params.campaignId}`, { 
        pipelineConfig: mergedConfig 
      });
      return await response.json() as Campaign;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns"] });
      setShowCampaignDialog(false);
      setShowAssayDialog(false);
      toast({
        title: "Campaign Updated",
        description: `Assays added to ${data.name}`,
      });
      navigate(`/campaigns/${data.id}`);
    },
    onError: () => {
      toast({
        title: "Update Failed",
        description: "Could not add assays to campaign",
        variant: "destructive",
      });
    },
  });

  const handleAddToCampaign = () => {
    if (!assayTemplate) return;

    const assayConfig = {
      trajectorySource: {
        datasetId: selectedDataset,
        disease: selectedDisease,
        analysisTimestamp: new Date().toISOString(),
      },
      targetGene: assayTemplate.targetGene,
      role: assayTemplate.role,
      druggable: assayTemplate.targetable,
      assays: assayTemplate.suggestedAssays,
    };

    if (campaignMode === "new") {
      if (!newCampaignName.trim()) {
        toast({
          title: "Name Required",
          description: "Please enter a campaign name",
          variant: "destructive",
        });
        return;
      }
      
      if (!selectedProjectId) {
        toast({
          title: "No Project Selected",
          description: "Please select a project for this campaign",
          variant: "destructive",
        });
        return;
      }

      createCampaignMutation.mutate({
        name: newCampaignName,
        projectId: selectedProjectId,
        pipelineConfig: { trajectoryAssays: [assayConfig] },
        domainType: "Neurology",
      });
    } else {
      if (!selectedCampaignId) {
        toast({
          title: "No Campaign Selected",
          description: "Please select an existing campaign",
          variant: "destructive",
        });
        return;
      }

      const existingCampaign = campaigns.find(c => c.id === selectedCampaignId);
      updateCampaignMutation.mutate({
        campaignId: selectedCampaignId,
        assayConfig,
        existingConfig: existingCampaign?.pipelineConfig || null,
      });
    }
  };

  const openCampaignDialog = () => {
    setNewCampaignName(`${assayTemplate?.targetGene || "Target"} ${assayTemplate?.disease || "Discovery"} Campaign`);
    setCampaignMode("new");
    setSelectedCampaignId("");
    setSelectedProjectId(projects[0]?.id || "");
    setShowCampaignDialog(true);
  };

  const handleRunAnalysis = () => {
    if (!selectedDataset) {
      toast({
        title: "No Dataset Selected",
        description: "Please select a dataset to analyze",
        variant: "destructive",
      });
      return;
    }
    analyzeMutation.mutate(selectedDataset);
  };

  const handleGenerateTemplate = (gene: string, pseudotime?: number, cluster?: string) => {
    if (!trajectoryResult) return;
    setSelectedBiomarker(gene);
    generateTemplateMutation.mutate({
      gene,
      disease: trajectoryResult.disease,
      pseudotime,
      cellState: cluster,
    });
  };

  const copyToClipboard = (text: string, gene: string) => {
    navigator.clipboard.writeText(text);
    setCopiedGene(gene);
    setTimeout(() => setCopiedGene(null), 2000);
  };

  const handleOpenPredictionDialog = (gene: string) => {
    setSelectedTargetForPrediction(gene);
    setInhibitorResult(null);
    setShowPredictionDialog(true);
  };

  const handleRunPrediction = () => {
    if (!selectedTargetForPrediction || !trajectoryResult) return;
    const smilesList = smilesInput.split("\n").map(s => s.trim()).filter(s => s.length > 0);
    if (smilesList.length === 0) {
      toast({
        title: "No SMILES",
        description: "Please enter at least one SMILES string",
        variant: "destructive",
      });
      return;
    }
    predictInhibitorsMutation.mutate({
      gene: selectedTargetForPrediction,
      disease: trajectoryResult.disease,
      smilesList,
    });
  };

  const umapData = useMemo(() => {
    if (!trajectoryResult || !trajectoryResult.umapCoordinates) return [];
    return trajectoryResult.umapCoordinates.slice(0, 500);
  }, [trajectoryResult]);

  const selectedDatasetInfo = datasets.find(d => d.id === selectedDataset);

  useEffect(() => {
    setSelectedDisease("");
    setSelectedDataset("");
    setTrajectoryResult(null);
    setActiveTab("datasets");
  }, [pipelineType]);

  return (
    <div className="flex flex-col h-full" data-testid="page-trajectory-analysis">
      <div className="border-b bg-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <GitBranch className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Trajectory Analysis</h1>
              <p className="text-sm text-muted-foreground">
                {isVaccineMode 
                  ? "Systems vaccinology for immune response discovery" 
                  : "scRNA + PGD for automated biomarker discovery"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-muted/50 rounded-lg p-1">
              <Button
                variant={pipelineType === "drug" ? "default" : "ghost"}
                size="sm"
                onClick={() => setPipelineType("drug")}
                data-testid="button-mode-drug"
              >
                <FlaskConical className="w-4 h-4 mr-1" />
                Drug Discovery
              </Button>
              <Button
                variant={pipelineType === "vaccine" ? "default" : "ghost"}
                size="sm"
                onClick={() => setPipelineType("vaccine")}
                data-testid="button-mode-vaccine"
              >
                <Dna className="w-4 h-4 mr-1" />
                Vaccine Discovery
              </Button>
            </div>
            {trajectoryResult && trajectoryResult.targets && (
              <Badge variant="secondary" className="gap-1">
                <Target className="w-3 h-3" />
                {trajectoryResult.targets.length} {isVaccineMode ? "Readouts" : "Targets"}
              </Badge>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 overflow-auto">
        <Tabs value={activeTab} onValueChange={(value) => {
            if (!trajectoryResult && value !== "datasets") {
              toast({
                title: "Analysis Required",
                description: "Please select and load a dataset first, then run PGD analysis",
                variant: "destructive",
              });
              return;
            }
            setActiveTab(value);
          }}>
          <TabsList className="mb-4">
            <TabsTrigger value="datasets" data-testid="tab-datasets">
              <Database className="w-4 h-4 mr-2" />
              Datasets
            </TabsTrigger>
            <TabsTrigger 
              value="trajectory" 
              className={!trajectoryResult ? "opacity-50" : ""} 
              data-testid="tab-trajectory"
            >
              <GitBranch className="w-4 h-4 mr-2" />
              Trajectory
            </TabsTrigger>
            <TabsTrigger 
              value="biomarkers" 
              className={!trajectoryResult ? "opacity-50" : ""} 
              data-testid="tab-biomarkers"
            >
              <Dna className="w-4 h-4 mr-2" />
              Biomarkers
            </TabsTrigger>
            <TabsTrigger 
              value="targets" 
              className={!trajectoryResult ? "opacity-50" : ""} 
              data-testid="tab-targets"
            >
              <Target className="w-4 h-4 mr-2" />
              {isVaccineMode ? "Immune Readouts" : "Targets"}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="datasets" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <Card className="lg:col-span-1">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Brain className="w-4 h-4" />
                    Select Disease
                  </CardTitle>
                  <CardDescription>Choose a disease area to explore</CardDescription>
                </CardHeader>
                <CardContent>
                  {diseasesLoading ? (
                    <Skeleton className="h-10 w-full" />
                  ) : (
                    <Select value={selectedDisease} onValueChange={setSelectedDisease}>
                      <SelectTrigger data-testid="select-disease">
                        <SelectValue placeholder="Select disease..." />
                      </SelectTrigger>
                      <SelectContent>
                        {diseases?.map((d) => (
                          <SelectItem key={d.disease} value={d.disease}>
                            <div className="flex items-center justify-between w-full gap-4">
                              <span>{d.disease}</span>
                              <Badge variant="outline" className="text-xs">
                                {d.datasetCount} datasets
                              </Badge>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}

                  {selectedDisease && (
                    <div className="mt-4 space-y-3">
                      <div className="p-3 bg-muted/50 rounded-lg space-y-2">
                        <button 
                          className="flex justify-between text-sm w-full hover:text-primary transition-colors cursor-pointer"
                          onClick={() => document.getElementById('datasets-panel')?.scrollIntoView({ behavior: 'smooth' })}
                          data-testid="link-view-datasets"
                        >
                          <span className="text-muted-foreground">Datasets</span>
                          <span className="font-medium text-primary underline underline-offset-2">
                            {diseases?.find(d => d.disease === selectedDisease)?.datasetCount} available →
                          </span>
                        </button>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Total Cells</span>
                          <span className="font-medium">
                            {diseases?.find(d => d.disease === selectedDisease)?.totalCells.toLocaleString()}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Known Biomarkers</span>
                          <span className="font-medium">{diseases?.find(d => d.disease === selectedDisease)?.biomarkers}</span>
                        </div>
                      </div>
                      
                      {selectedDataset && selectedDatasetInfo && (
                        <div className="p-3 bg-primary/10 border border-primary/30 rounded-lg">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <Check className="w-4 h-4 text-primary" />
                              <span className="text-sm font-medium">Active: {selectedDatasetInfo.name}</span>
                            </div>
                            <Button 
                              size="sm" 
                              onClick={handleRunAnalysis}
                              disabled={analyzeMutation.isPending}
                              data-testid="button-proceed-trajectory"
                            >
                              {analyzeMutation.isPending ? (
                                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                              ) : (
                                <ArrowRight className="w-3 h-3 mr-1" />
                              )}
                              Run Analysis
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="lg:col-span-2" id="datasets-panel">
                <CardHeader className="pb-3 flex flex-row items-center justify-between gap-2">
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      <Database className="w-4 h-4" />
                      Public Datasets
                    </CardTitle>
                    <CardDescription>
                      {selectedDisease ? `${datasets.length} datasets available for ${selectedDisease}` : "Select a disease to view datasets"}
                    </CardDescription>
                  </div>
                  {selectedDataset && (
                    <Button
                      onClick={handleRunAnalysis}
                      disabled={analyzeMutation.isPending}
                      data-testid="button-run-analysis"
                    >
                      {analyzeMutation.isPending ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <Play className="w-4 h-4 mr-2" />
                      )}
                      Run PGD Analysis
                    </Button>
                  )}
                </CardHeader>
                <CardContent>
                  {!selectedDisease ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>Select a disease to view available scRNA-seq datasets</p>
                    </div>
                  ) : datasetsLoading ? (
                    <div className="space-y-2">
                      {[1, 2, 3].map((i) => (
                        <Skeleton key={i} className="h-16 w-full" />
                      ))}
                    </div>
                  ) : datasets.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>No datasets found for {selectedDisease}</p>
                    </div>
                  ) : (
                    <ScrollArea className="h-[400px]">
                      <div className="space-y-2">
                        {datasets.map((dataset) => (
                          <div
                            key={dataset.id}
                            className={`p-3 rounded-lg border transition-colors ${
                              selectedDataset === dataset.id
                                ? "border-primary bg-primary/5"
                                : "border-border hover-elevate"
                            }`}
                            data-testid={`dataset-${dataset.id}`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <h4 className="font-medium truncate">{dataset.name}</h4>
                                  <Badge variant="outline" className="text-xs shrink-0">
                                    {dataset.geoAccession}
                                  </Badge>
                                </div>
                                <p className="text-sm text-muted-foreground mt-1 line-clamp-1">
                                  {dataset.description}
                                </p>
                                <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                                  <span className="flex items-center gap-1">
                                    <Microscope className="w-3 h-3" />
                                    {dataset.tissue}
                                  </span>
                                  <span>{dataset.cellCount.toLocaleString()} cells</span>
                                  <span>{dataset.publicationYear}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2 shrink-0">
                                {selectedDataset === dataset.id ? (
                                  <Badge variant="default" className="gap-1">
                                    <Check className="w-3 h-3" />
                                    Active
                                  </Badge>
                                ) : (
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => {
                                      setSelectedDataset(dataset.id);
                                      toast({
                                        title: "Dataset Selected",
                                        description: `${dataset.name} is now active. Click "Run PGD Analysis" to proceed.`,
                                      });
                                    }}
                                    data-testid={`button-load-${dataset.id}`}
                                  >
                                    <Database className="w-3 h-3 mr-1" />
                                    Load
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </div>

            {selectedDatasetInfo && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Settings2 className="w-4 h-4" />
                    PGD Settings
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <Label>Smoothing Alpha ({smoothingAlpha[0].toFixed(2)})</Label>
                      <Slider
                        value={smoothingAlpha}
                        onValueChange={setSmoothingAlpha}
                        min={0.01}
                        max={0.5}
                        step={0.01}
                        data-testid="slider-alpha"
                      />
                      <p className="text-xs text-muted-foreground">
                        Lower values preserve more local structure, higher values smooth more aggressively
                      </p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg space-y-2">
                      <h4 className="font-medium text-sm">Analysis Pipeline</h4>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Badge variant="outline" className="text-xs">1</Badge>
                        Load Dataset
                        <ArrowRight className="w-3 h-3" />
                        <Badge variant="outline" className="text-xs">2</Badge>
                        UMAP Embedding
                        <ArrowRight className="w-3 h-3" />
                        <Badge variant="outline" className="text-xs">3</Badge>
                        PGD Smoothing
                      </div>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Badge variant="outline" className="text-xs">4</Badge>
                        Trajectory Inference
                        <ArrowRight className="w-3 h-3" />
                        <Badge variant="outline" className="text-xs">5</Badge>
                        Branch Detection
                        <ArrowRight className="w-3 h-3" />
                        <Badge variant="outline" className="text-xs">6</Badge>
                        Biomarkers
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="trajectory" className="space-y-4">
            {trajectoryResult && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <Card className="lg:col-span-2">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">UMAP + Pseudotime</CardTitle>
                      <CardDescription>
                        Cells colored by pseudotime (0 = start, 1 = end of trajectory)
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="relative w-full h-[400px] bg-muted/30 rounded-lg border overflow-hidden">
                        <svg viewBox="-15 -15 30 30" className="w-full h-full">
                          {umapData.map((point, i) => (
                            <circle
                              key={i}
                              cx={point.x}
                              cy={point.y}
                              r={0.15}
                              fill={`hsl(${(1 - point.pseudotime) * 240}, 70%, 50%)`}
                              opacity={0.7}
                            />
                          ))}
                          {(trajectoryResult.branchPoints || []).map((bp, i) => {
                            const matchingPoints = umapData.filter(
                              p => Math.abs(p.pseudotime - bp.pseudotime) < 0.1
                            );
                            if (matchingPoints.length === 0) return null;
                            const avgX = matchingPoints.reduce((s, p) => s + p.x, 0) / matchingPoints.length;
                            const avgY = matchingPoints.reduce((s, p) => s + p.y, 0) / matchingPoints.length;
                            return (
                              <g key={i}>
                                <circle cx={avgX} cy={avgY} r={0.8} fill="none" stroke="#ef4444" strokeWidth={0.1} />
                                <text x={avgX + 1} y={avgY} fontSize={0.8} fill="#ef4444">
                                  {bp.genes.slice(0, 2).join(", ")}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                        <div className="absolute bottom-4 left-4 flex items-center gap-2 bg-background/80 backdrop-blur rounded-md px-3 py-2">
                          <div className="w-20 h-2 rounded-full" style={{
                            background: "linear-gradient(to right, hsl(240, 70%, 50%), hsl(0, 70%, 50%))"
                          }} />
                          <span className="text-xs text-muted-foreground">Pseudotime 0 → 1</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">Clusters</CardTitle>
                      <CardDescription>{(trajectoryResult.clusters || []).length} cell populations</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ScrollArea className="h-[360px]">
                        <div className="space-y-2">
                          {(trajectoryResult.clusters || []).map((cluster) => (
                            <div
                              key={cluster.id}
                              className="flex items-center justify-between p-2 rounded-lg border"
                            >
                              <div className="flex items-center gap-2">
                                <div
                                  className="w-3 h-3 rounded-full"
                                  style={{ backgroundColor: cluster.color }}
                                />
                                <span className="text-sm font-medium">{cluster.name}</span>
                              </div>
                              <Badge variant="secondary" className="text-xs">
                                {cluster.cellCount.toLocaleString()}
                              </Badge>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </Card>
                </div>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      <GitBranch className="w-4 h-4" />
                      Branch Points
                    </CardTitle>
                    <CardDescription>Key decision points in the trajectory</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative">
                      <div className="absolute top-4 left-0 right-0 h-1 bg-muted rounded-full" />
                      <div className="flex justify-between relative">
                        {(trajectoryResult.branchPoints || []).map((bp, i) => (
                          <div
                            key={i}
                            className="flex flex-col items-center"
                            style={{ left: `${bp.pseudotime * 100}%`, position: "absolute", transform: "translateX(-50%)" }}
                          >
                            <div className="w-4 h-4 rounded-full bg-destructive border-2 border-background z-10" />
                            <div className="mt-2 p-2 bg-muted/50 rounded-lg text-center">
                              <div className="text-xs font-medium">t = {bp.pseudotime.toFixed(1)}</div>
                              <div className="text-xs text-muted-foreground mt-1">
                                {bp.genes.slice(0, 2).join(", ")}
                                {bp.genes.length > 2 && ` +${bp.genes.length - 2}`}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="h-24" />
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          <TabsContent value="biomarkers" className="space-y-4">
            {trajectoryResult && (
              <Card>
                <CardHeader className="pb-3 flex flex-row items-center justify-between gap-2">
                  <div>
                    <CardTitle className="text-base">Detected Biomarkers</CardTitle>
                    <CardDescription>
                      {isVaccineMode 
                        ? "Genes associated with differential vaccine-induced immune responses over pseudotime"
                        : "Genes with significant expression changes along the trajectory"}
                    </CardDescription>
                  </div>
                  <Badge variant="secondary">{(trajectoryResult.biomarkers || []).length} genes</Badge>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Gene</TableHead>
                        <TableHead>Pseudotime</TableHead>
                        <TableHead>Direction</TableHead>
                        <TableHead>Fold Change</TableHead>
                        <TableHead>p-value</TableHead>
                        <TableHead>Cell Type</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(trajectoryResult.biomarkers || []).map((marker) => (
                        <TableRow key={marker.gene}>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <span className="font-mono font-medium">{marker.gene}</span>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0"
                                onClick={() => copyToClipboard(marker.gene, marker.gene)}
                                data-testid={`button-copy-${marker.gene}`}
                              >
                                {copiedGene === marker.gene ? (
                                  <Check className="w-3 h-3 text-green-500" />
                                ) : (
                                  <ClipboardCopy className="w-3 h-3" />
                                )}
                              </Button>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Progress value={marker.pseudotimeExpression * 100} className="w-16 h-2" />
                              <span className="text-xs text-muted-foreground">
                                {marker.pseudotimeExpression.toFixed(2)}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            {marker.direction === "up" ? (
                              <Badge variant="default" className="gap-1 bg-green-500 hover:bg-green-600">
                                <TrendingUp className="w-3 h-3" />
                                Up
                              </Badge>
                            ) : (
                              <Badge variant="default" className="gap-1 bg-red-500 hover:bg-red-600">
                                <TrendingDown className="w-3 h-3" />
                                Down
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <span className="font-mono">{marker.foldChange.toFixed(2)}x</span>
                          </TableCell>
                          <TableCell>
                            <span className="font-mono text-xs">
                              {marker.pValue.toExponential(2)}
                            </span>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {marker.cluster}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleGenerateTemplate(marker.gene, marker.pseudotimeExpression, marker.cluster)}
                              disabled={generateTemplateMutation.isPending}
                              data-testid={`button-template-${marker.gene}`}
                            >
                              {generateTemplateMutation.isPending && selectedBiomarker === marker.gene ? (
                                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                              ) : (
                                <FlaskConical className="w-3 h-3 mr-1" />
                              )}
                              {isVaccineMode ? "Readout" : "Assay"}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="targets" className="space-y-4">
            {isVaccineMode ? (
              <div className="space-y-6">
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-lg flex items-center gap-2">
                          <Dna className="w-5 h-5" />
                          Immune Readouts
                        </CardTitle>
                        <CardDescription className="mt-1">
                          Immune readouts link your scRNA biomarkers to actual vaccine efficacy measurements (antibody titers, T cell responses, innate signatures).
                        </CardDescription>
                      </div>
                      <Badge variant="secondary" className="text-sm">
                        {selectedReadouts.size}/{allReadouts.length} readouts selected
                      </Badge>
                    </div>
                  </CardHeader>
                </Card>

                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3 flex items-center gap-2">
                      <FlaskConical className="w-4 h-4" />
                      Humoral Responses
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {IMMUNE_READOUTS.humoral.map((readout) => (
                        <Card 
                          key={readout.id} 
                          className={`cursor-pointer transition-all ${selectedReadouts.has(readout.id) ? "ring-2 ring-primary bg-primary/5" : "hover-elevate"}`}
                          onClick={() => toggleReadout(readout.id)}
                          data-testid={`readout-${readout.id}`}
                        >
                          <CardHeader className="pb-2">
                            <div className="flex items-start gap-3">
                              <div className={`mt-0.5 w-5 h-5 rounded border-2 flex items-center justify-center ${selectedReadouts.has(readout.id) ? "bg-primary border-primary" : "border-muted-foreground/30"}`}>
                                {selectedReadouts.has(readout.id) && <Check className="w-3 h-3 text-primary-foreground" />}
                              </div>
                              <div className="flex-1">
                                <CardTitle className="text-sm font-medium">{readout.name}</CardTitle>
                                <CardDescription className="text-xs mt-1">{readout.description}</CardDescription>
                              </div>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <div className="flex flex-wrap gap-1 mb-2">
                              {readout.biomarkers.slice(0, 3).map((b) => (
                                <Badge key={b} variant="outline" className="text-xs font-mono">{b}</Badge>
                              ))}
                              {readout.biomarkers.length > 3 && (
                                <Badge variant="outline" className="text-xs">+{readout.biomarkers.length - 3}</Badge>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Timepoints: {readout.timepoints.join(", ")}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3 flex items-center gap-2">
                      <Target className="w-4 h-4" />
                      Cellular Responses
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {IMMUNE_READOUTS.cellular.map((readout) => (
                        <Card 
                          key={readout.id} 
                          className={`cursor-pointer transition-all ${selectedReadouts.has(readout.id) ? "ring-2 ring-primary bg-primary/5" : "hover-elevate"}`}
                          onClick={() => toggleReadout(readout.id)}
                          data-testid={`readout-${readout.id}`}
                        >
                          <CardHeader className="pb-2">
                            <div className="flex items-start gap-3">
                              <div className={`mt-0.5 w-5 h-5 rounded border-2 flex items-center justify-center ${selectedReadouts.has(readout.id) ? "bg-primary border-primary" : "border-muted-foreground/30"}`}>
                                {selectedReadouts.has(readout.id) && <Check className="w-3 h-3 text-primary-foreground" />}
                              </div>
                              <div className="flex-1">
                                <CardTitle className="text-sm font-medium">{readout.name}</CardTitle>
                                <CardDescription className="text-xs mt-1">{readout.description}</CardDescription>
                              </div>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <div className="flex flex-wrap gap-1 mb-2">
                              {readout.biomarkers.slice(0, 3).map((b) => (
                                <Badge key={b} variant="outline" className="text-xs font-mono">{b}</Badge>
                              ))}
                              {readout.biomarkers.length > 3 && (
                                <Badge variant="outline" className="text-xs">+{readout.biomarkers.length - 3}</Badge>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Timepoints: {readout.timepoints.join(", ")}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3 flex items-center gap-2">
                      <Zap className="w-4 h-4" />
                      Innate Responses
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {IMMUNE_READOUTS.innate.map((readout) => (
                        <Card 
                          key={readout.id} 
                          className={`cursor-pointer transition-all ${selectedReadouts.has(readout.id) ? "ring-2 ring-primary bg-primary/5" : "hover-elevate"}`}
                          onClick={() => toggleReadout(readout.id)}
                          data-testid={`readout-${readout.id}`}
                        >
                          <CardHeader className="pb-2">
                            <div className="flex items-start gap-3">
                              <div className={`mt-0.5 w-5 h-5 rounded border-2 flex items-center justify-center ${selectedReadouts.has(readout.id) ? "bg-primary border-primary" : "border-muted-foreground/30"}`}>
                                {selectedReadouts.has(readout.id) && <Check className="w-3 h-3 text-primary-foreground" />}
                              </div>
                              <div className="flex-1">
                                <CardTitle className="text-sm font-medium">{readout.name}</CardTitle>
                                <CardDescription className="text-xs mt-1">{readout.description}</CardDescription>
                              </div>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <div className="flex flex-wrap gap-1 mb-2">
                              {readout.biomarkers.slice(0, 3).map((b) => (
                                <Badge key={b} variant="outline" className="text-xs font-mono">{b}</Badge>
                              ))}
                              {readout.biomarkers.length > 3 && (
                                <Badge variant="outline" className="text-xs">+{readout.biomarkers.length - 3}</Badge>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Timepoints: {readout.timepoints.join(", ")}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                </div>

                <Card className="bg-muted/30">
                  <CardContent className="py-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{selectedReadouts.size} readouts selected</p>
                        <p className="text-sm text-muted-foreground">
                          {selectedReadouts.size > 0 
                            ? "Ready to create vaccine campaign with chosen readouts and biomarkers"
                            : "Select readouts above to build your vaccine efficacy panel"}
                        </p>
                      </div>
                      <Button
                        disabled={selectedReadouts.size === 0}
                        onClick={() => {
                          setShowCampaignDialog(true);
                          setCampaignMode("new");
                        }}
                        data-testid="button-add-readouts-campaign"
                      >
                        <Plus className="w-4 h-4 mr-2" />
                        Add Selected to Campaign
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              trajectoryResult && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(trajectoryResult.targets || []).map((target) => {
                    const biomarker = (trajectoryResult.biomarkers || []).find(b => b.gene === target);
                    return (
                      <Card key={target} className="hover-elevate">
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-lg font-mono">{target}</CardTitle>
                            <Badge variant="default" className="gap-1">
                              <Target className="w-3 h-3" />
                              Druggable
                            </Badge>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          {biomarker && (
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Expression Peak</span>
                                <span>t = {biomarker.pseudotimeExpression.toFixed(2)}</span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Fold Change</span>
                                <span className="flex items-center gap-1">
                                  {biomarker.direction === "up" ? (
                                    <TrendingUp className="w-3 h-3 text-green-500" />
                                  ) : (
                                    <TrendingDown className="w-3 h-3 text-red-500" />
                                  )}
                                  {biomarker.foldChange.toFixed(1)}x
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Cell Type</span>
                                <Badge variant="outline" className="text-xs">{biomarker.cluster}</Badge>
                              </div>
                            </div>
                          )}
                          <div className="flex gap-2 pt-2">
                            <Button
                              size="sm"
                              className="flex-1"
                              onClick={() => handleGenerateTemplate(target)}
                              data-testid={`button-create-assay-${target}`}
                            >
                              <FlaskConical className="w-3 h-3 mr-1" />
                              Create Assay
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleOpenPredictionDialog(target)}
                              data-testid={`button-predict-${target}`}
                            >
                              <Zap className="w-3 h-3 mr-1" />
                              BioNeMo
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )
            )}
          </TabsContent>
        </Tabs>
      </div>

      <Dialog open={showAssayDialog} onOpenChange={setShowAssayDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <FlaskConical className="w-5 h-5" />
              {isVaccineMode ? "Immune Readout" : "Assay Template"} for {assayTemplate?.targetGene}
            </DialogTitle>
            <DialogDescription>
              {isVaccineMode 
                ? "Auto-generated immune correlate readout from vaccine response trajectory"
                : "Auto-generated from trajectory analysis biomarker detection"}
            </DialogDescription>
          </DialogHeader>
          {assayTemplate && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-muted/50 rounded-lg">
                  <Label className="text-xs text-muted-foreground">Target Gene</Label>
                  <p className="font-mono font-medium">{assayTemplate.targetGene}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <Label className="text-xs text-muted-foreground">Disease</Label>
                  <p className="font-medium">{assayTemplate.disease}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <Label className="text-xs text-muted-foreground">Role</Label>
                  <p className="text-sm">{assayTemplate.role}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <Label className="text-xs text-muted-foreground">{isVaccineMode ? "Predictive Value" : "Druggability"}</Label>
                  <Badge variant={assayTemplate.targetable ? "default" : "secondary"}>
                    {assayTemplate.targetable 
                      ? (isVaccineMode ? "Predictive" : "Druggable") 
                      : "Challenging"}
                  </Badge>
                </div>
              </div>

              <div>
                <Label className="text-sm font-medium">Suggested Assays</Label>
                <div className="mt-2 space-y-2">
                  {assayTemplate.suggestedAssays.map((assay, i) => (
                    <div key={i} className="p-3 border rounded-lg">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">{assay.name}</h4>
                        <Badge variant="outline">{assay.type}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">{assay.description}</p>
                      <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                        <span>Readout: {assay.readoutType}</span>
                        <span>Technique: {assay.technique}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setShowAssayDialog(false)} data-testid="button-close-assay">
                  Close
                </Button>
                <Button onClick={openCampaignDialog} data-testid="button-add-to-campaign">
                  <FolderPlus className="w-4 h-4 mr-2" />
                  Add to Campaign
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={showPredictionDialog} onOpenChange={setShowPredictionDialog}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              BioNeMo Inhibitor Prediction: {selectedTargetForPrediction}
            </DialogTitle>
            <DialogDescription>
              Predict binding affinity of compounds against the identified target
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>SMILES (one per line)</Label>
              <textarea
                value={smilesInput}
                onChange={(e) => setSmilesInput(e.target.value)}
                className="w-full h-32 mt-2 p-3 font-mono text-sm border rounded-lg bg-background resize-none"
                placeholder="Enter SMILES strings, one per line..."
                data-testid="textarea-smiles"
              />
              <p className="text-xs text-muted-foreground mt-1">
                {smilesInput.split("\n").filter(s => s.trim()).length} compounds
              </p>
            </div>

            <Button
              onClick={handleRunPrediction}
              disabled={predictInhibitorsMutation.isPending}
              className="w-full"
              data-testid="button-run-prediction"
            >
              {predictInhibitorsMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              Run Prediction
            </Button>

            {inhibitorResult && (
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Badge variant="secondary">
                    {inhibitorResult.totalCompounds} Compounds
                  </Badge>
                  <Badge variant="default" className="bg-green-500">
                    {inhibitorResult.hits} Hits
                  </Badge>
                  <Badge variant="outline">
                    Target: {inhibitorResult.gene}
                  </Badge>
                </div>

                <div className="border rounded-lg overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Rank</TableHead>
                        <TableHead>SMILES</TableHead>
                        <TableHead>pIC50</TableHead>
                        <TableHead>IC50 (nM)</TableHead>
                        <TableHead>Confidence</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {inhibitorResult.predictions.slice(0, 10).map((pred) => (
                        <TableRow key={pred.rank}>
                          <TableCell className="font-medium">{pred.rank}</TableCell>
                          <TableCell>
                            <span className="font-mono text-xs truncate max-w-[200px] block">
                              {pred.smiles}
                            </span>
                          </TableCell>
                          <TableCell>
                            <span className="font-mono">{pred.predictedPIC50.toFixed(2)}</span>
                          </TableCell>
                          <TableCell>
                            <span className="font-mono">{pred.predictedIC50_nM.toFixed(0)}</span>
                          </TableCell>
                          <TableCell>
                            <Progress value={pred.confidence * 100} className="w-16 h-2" />
                          </TableCell>
                          <TableCell>
                            {pred.isHit ? (
                              <Badge variant="default" className="bg-green-500">Hit</Badge>
                            ) : (
                              <Badge variant="secondary">Low</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowPredictionDialog(false)} data-testid="button-close-prediction">
                Close
              </Button>
              {inhibitorResult && inhibitorResult.hits > 0 && (
                <Button data-testid="button-export-hits">
                  <Download className="w-4 h-4 mr-2" />
                  Export Hits
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={showCampaignDialog} onOpenChange={setShowCampaignDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <FolderPlus className="w-5 h-5" />
              Add to Campaign
            </DialogTitle>
            <DialogDescription>
              Create a new campaign or add assays to an existing one
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <RadioGroup value={campaignMode} onValueChange={(v) => setCampaignMode(v as "new" | "existing")}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="new" id="new-campaign" data-testid="radio-new-campaign" />
                <Label htmlFor="new-campaign">Create New Campaign</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="existing" id="existing-campaign" data-testid="radio-existing-campaign" />
                <Label htmlFor="existing-campaign">Add to Existing Campaign</Label>
              </div>
            </RadioGroup>

            {campaignMode === "new" ? (
              <div className="space-y-3">
                <div>
                  <Label htmlFor="campaign-name">Campaign Name</Label>
                  <Input
                    id="campaign-name"
                    value={newCampaignName}
                    onChange={(e) => setNewCampaignName(e.target.value)}
                    placeholder="Enter campaign name"
                    data-testid="input-campaign-name"
                  />
                </div>
                <div>
                  <Label htmlFor="select-project">Project</Label>
                  <Select value={selectedProjectId} onValueChange={setSelectedProjectId}>
                    <SelectTrigger data-testid="select-project-trigger">
                      <SelectValue placeholder="Select a project" />
                    </SelectTrigger>
                    <SelectContent>
                      {projects.map((project) => (
                        <SelectItem key={project.id} value={project.id} data-testid={`select-project-${project.id}`}>
                          {project.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg text-sm">
                  <p className="font-medium mb-1">Will include:</p>
                  <ul className="space-y-1 text-muted-foreground">
                    <li className="flex items-center gap-2">
                      <Target className="w-3 h-3" />
                      Target: {assayTemplate?.targetGene}
                    </li>
                    <li className="flex items-center gap-2">
                      <FlaskConical className="w-3 h-3" />
                      {assayTemplate?.suggestedAssays?.length || 0} assays
                    </li>
                    <li className="flex items-center gap-2">
                      <Microscope className="w-3 h-3" />
                      Disease: {assayTemplate?.disease}
                    </li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {campaigns.length > 0 ? (
                  <div>
                    <Label htmlFor="select-campaign">Select Campaign</Label>
                    <Select value={selectedCampaignId} onValueChange={setSelectedCampaignId}>
                      <SelectTrigger data-testid="select-campaign-trigger">
                        <SelectValue placeholder="Select a campaign" />
                      </SelectTrigger>
                      <SelectContent>
                        {campaigns.map((campaign) => (
                          <SelectItem key={campaign.id} value={campaign.id} data-testid={`select-campaign-${campaign.id}`}>
                            {campaign.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ) : (
                  <div className="p-4 bg-muted/50 rounded-lg text-center">
                    <p className="text-sm text-muted-foreground">No existing campaigns found</p>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => setCampaignMode("new")}
                      data-testid="button-switch-to-new"
                    >
                      Create a new campaign instead
                    </Button>
                  </div>
                )}
              </div>
            )}

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowCampaignDialog(false)} data-testid="button-cancel-campaign">
                Cancel
              </Button>
              <Button 
                onClick={handleAddToCampaign}
                disabled={createCampaignMutation.isPending || updateCampaignMutation.isPending}
                data-testid="button-confirm-campaign"
              >
                {(createCampaignMutation.isPending || updateCampaignMutation.isPending) && (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                )}
                {campaignMode === "new" ? "Create Campaign" : "Add to Campaign"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
