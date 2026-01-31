import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  Syringe, 
  Dna, 
  Atom,
  Activity,
  Loader2,
  Play,
  CheckCircle2,
  AlertCircle,
  Server,
  Cpu,
  Zap,
  FileCode,
  Beaker,
  Grid3X3,
  Clock,
  DollarSign,
  ChevronDown,
  ChevronRight,
  Upload,
  FileText,
  X
} from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { queryClient } from "@/lib/queryClient";

interface PdbUpload {
  id: string;
  fileName: string;
  storedPath: string;
  description: string;
  uploadedBy: string;
  uploadedAt: string;
  fileSize: number;
  purpose?: string;
  extractedSequence?: string;
  sequenceLength?: number;
}

interface HardwareReport {
  totalNodes: number;
  activeNodes: number;
  gpuNodes: number;
  taskRouting: {
    gpu_intensive: string[];
    gpu_preferred: string[];
    cpu_intensive: string[];
    cpu_only: string[];
  };
  estimatedSpeedups: Record<string, string>;
}

interface VaccineResult {
  step: string;
  success: boolean;
  result?: any;
  error?: string;
  nodeUsed?: string;
}

interface TaskInfo {
  type: string;
  reason: string;
  cpuCores?: number;
  memoryGb?: number;
  gpuMemoryGb?: number;
  estimatedTimeMinutes?: number;
  speedup?: string;
  tools?: string[];
  note?: string;
}

interface CategoryInfo {
  stage: number;
  stageName: string;
  tasks: Record<string, TaskInfo>;
}

interface TaskMatrixData {
  taskClassification: Record<string, CategoryInfo>;
  hardwareRequirements: Record<string, any>;
  costAnalysis: Record<string, any>;
  summary: {
    totalCategories: number;
    totalTasks: number;
    typeCounts: Record<string, number>;
    stages: Array<{ stage: number; name: string }>;
  };
}

export default function VaccineDiscoveryPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("complete");
  
  const [pathogenName, setPathogenName] = useState("Novel Coronavirus");
  const [proteinSequences, setProteinSequences] = useState("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF");
  const [mhcAlleles, setMhcAlleles] = useState(["HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02"]);
  const [runMD, setRunMD] = useState(false);
  const [organism, setOrganism] = useState("human");
  
  const [structureSequence, setStructureSequence] = useState("MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLFRPGQKNSL");
  const [structureMethod, setStructureMethod] = useState("esmfold");
  
  const [epitopeSequence, setEpitopeSequence] = useState("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFF");
  const [peptideLength, setPeptideLength] = useState(9);
  
  const [codonSequence, setCodonSequence] = useState("MFVFLVLLPLVSS");
  const [codonOrganism, setCodonOrganism] = useState("human");
  
  const [mrnaSequence, setMrnaSequence] = useState("MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNS");
  const [utrType, setUtrType] = useState("optimized");
  const [capType, setCapType] = useState("cap1");
  const [polyALength, setPolyALength] = useState(120);

  const [pipelineResult, setPipelineResult] = useState<VaccineResult | null>(null);
  const [structureResult, setStructureResult] = useState<VaccineResult | null>(null);
  const [epitopeResult, setEpitopeResult] = useState<VaccineResult | null>(null);
  const [codonResult, setCodonResult] = useState<VaccineResult | null>(null);
  const [mrnaResult, setMrnaResult] = useState<VaccineResult | null>(null);
  const [completePipelineResult, setCompletePipelineResult] = useState<any | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());

  const [completeSequence, setCompleteSequence] = useState("");
  const [completeVaccineType, setCompleteVaccineType] = useState("protein_subunit");
  const [completePdbFileId, setCompletePdbFileId] = useState<string | null>(null);
  
  const [selectedPdbFileId, setSelectedPdbFileId] = useState<string | null>(null);
  const [structurePdbFileId, setStructurePdbFileId] = useState<string | null>(null);
  const [pdbDescription, setPdbDescription] = useState("");
  const [uploadingPdb, setUploadingPdb] = useState(false);

  const { data: hardwareReport, isLoading: hardwareLoading } = useQuery<HardwareReport>({
    queryKey: ["/api/compute/vaccine/hardware"],
  });

  const { data: taskMatrix, isLoading: taskMatrixLoading } = useQuery<TaskMatrixData>({
    queryKey: ["/api/compute/vaccine/task-matrix"],
  });

  const { data: pdbUploads, isLoading: pdbLoading } = useQuery<PdbUpload[]>({
    queryKey: ["/api/compute/vaccine/pdb-uploads"],
  });

  const handlePdbUpload = async (file: File) => {
    setUploadingPdb(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("description", pdbDescription || "Vaccine discovery structure");
      formData.append("purpose", "vaccine_pipeline");

      const res = await fetch("/api/compute/vaccine/upload-pdb", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Upload failed");
      }

      const data = await res.json();
      queryClient.invalidateQueries({ queryKey: ["/api/compute/vaccine/pdb-uploads"] });
      setSelectedPdbFileId(data.id);
      
      if (data.extractedSequence) {
        setProteinSequences(data.extractedSequence);
      }
      
      toast({ 
        title: "PDB file uploaded", 
        description: `${data.fileName} - ${data.sequenceLength || 0} residues extracted` 
      });
      setPdbDescription("");
    } catch (error: any) {
      toast({ title: "Upload failed", description: error.message, variant: "destructive" });
    } finally {
      setUploadingPdb(false);
    }
  };

  const toggleCategory = (category: string) => {
    const newSet = new Set(expandedCategories);
    if (newSet.has(category)) {
      newSet.delete(category);
    } else {
      newSet.add(category);
    }
    setExpandedCategories(newSet);
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "GPU_INTENSIVE": return "bg-purple-500/10 text-purple-600 border-purple-500/30";
      case "GPU_PREFERRED": return "bg-blue-500/10 text-blue-600 border-blue-500/30";
      case "CPU_INTENSIVE": return "bg-amber-500/10 text-amber-600 border-amber-500/30";
      case "CPU_ONLY": return "bg-slate-500/10 text-slate-600 border-slate-500/30";
      case "HYBRID": return "bg-green-500/10 text-green-600 border-green-500/30";
      default: return "bg-muted";
    }
  };

  const pipelineMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/pipeline", {
        pathogenName,
        proteinSequences: proteinSequences.split("\n").filter(s => s.trim()),
        pdbFileId: selectedPdbFileId,
        mhcAlleles,
        runMD,
        organism,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setPipelineResult(data);
      toast({ title: "Pipeline Complete", description: `Processed on ${data.nodeUsed || "compute node"}` });
    },
    onError: (error: any) => {
      toast({ title: "Pipeline Error", description: error.message, variant: "destructive" });
    },
  });

  const structureMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/structure", {
        sequence: structureSequence,
        pdbFileId: structurePdbFileId,
        method: structureMethod,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setStructureResult(data);
      toast({ title: "Structure Prediction Complete", description: "GPU-accelerated prediction finished" });
    },
    onError: (error: any) => {
      toast({ title: "Structure Error", description: error.message, variant: "destructive" });
    },
  });

  const epitopeMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/epitopes", {
        sequence: epitopeSequence,
        mhcAlleles,
        peptideLength,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setEpitopeResult(data);
      toast({ title: "Epitope Prediction Complete", description: `Found ${data.result?.epitopes?.length || 0} epitopes` });
    },
    onError: (error: any) => {
      toast({ title: "Epitope Error", description: error.message, variant: "destructive" });
    },
  });

  const codonMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/codon-optimize", {
        sequence: codonSequence,
        organism: codonOrganism,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setCodonResult(data);
      toast({ title: "Codon Optimization Complete", description: `Optimized for ${codonOrganism} expression` });
    },
    onError: (error: any) => {
      toast({ title: "Codon Error", description: error.message, variant: "destructive" });
    },
  });

  const mrnaMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/mrna-design", {
        sequence: mrnaSequence,
        utrType,
        capType,
        polyALength,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setMrnaResult(data);
      toast({ title: "mRNA Design Complete", description: "Construct designed successfully" });
    },
    onError: (error: any) => {
      toast({ title: "mRNA Error", description: error.message, variant: "destructive" });
    },
  });

  const completePipelineMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/complete-pipeline", {
        sequence: completeSequence,
        vaccineType: completeVaccineType,
        pdbFileId: completePdbFileId,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setCompletePipelineResult(data);
      toast({ title: "Complete Pipeline Finished", description: "All bioinformatics analyses complete" });
    },
    onError: (error: any) => {
      toast({ title: "Pipeline Error", description: error.message, variant: "destructive" });
    },
  });

  return (
    <div className="flex flex-col h-full overflow-auto">
      <div className="border-b bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-600/20">
                <Syringe className="h-6 w-6 text-purple-500" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Vaccine Discovery</h1>
                <p className="text-sm text-muted-foreground">GPU-Agnostic Pipeline for Antigen & mRNA Design</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {hardwareReport?.activeNodes ? (
                <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/30">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  {hardwareReport.activeNodes} Node{hardwareReport.activeNodes > 1 ? "s" : ""} Active
                </Badge>
              ) : (
                <Badge variant="outline" className="bg-amber-500/10 text-amber-600 border-amber-500/30">
                  <AlertCircle className="h-3 w-3 mr-1" />
                  No Nodes Available
                </Badge>
              )}
              {hardwareReport?.gpuNodes ? (
                <Badge variant="outline" className="bg-purple-500/10 text-purple-600 border-purple-500/30">
                  <Zap className="h-3 w-3 mr-1" />
                  {hardwareReport.gpuNodes} GPU Node{hardwareReport.gpuNodes > 1 ? "s" : ""}
                </Badge>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6 flex-1">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-8">
            <TabsTrigger value="complete" data-testid="tab-complete">
              <Syringe className="h-4 w-4 mr-2" />
              Complete
            </TabsTrigger>
            <TabsTrigger value="pipeline" data-testid="tab-pipeline">
              <Activity className="h-4 w-4 mr-2" />
              Pipeline
            </TabsTrigger>
            <TabsTrigger value="structure" data-testid="tab-structure">
              <Atom className="h-4 w-4 mr-2" />
              Structure
            </TabsTrigger>
            <TabsTrigger value="epitopes" data-testid="tab-epitopes">
              <Beaker className="h-4 w-4 mr-2" />
              Epitopes
            </TabsTrigger>
            <TabsTrigger value="codon" data-testid="tab-codon">
              <FileCode className="h-4 w-4 mr-2" />
              Codon
            </TabsTrigger>
            <TabsTrigger value="mrna" data-testid="tab-mrna">
              <Dna className="h-4 w-4 mr-2" />
              mRNA
            </TabsTrigger>
            <TabsTrigger value="task-matrix" data-testid="tab-task-matrix">
              <Grid3X3 className="h-4 w-4 mr-2" />
              Matrix
            </TabsTrigger>
            <TabsTrigger value="hardware" data-testid="tab-hardware">
              <Server className="h-4 w-4 mr-2" />
              Hardware
            </TabsTrigger>
          </TabsList>

          <TabsContent value="complete" className="space-y-6">
            {/* AQAffinity Integration Banner */}
            <div className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 rounded-lg p-4" data-testid="aqaffinity-banner">
              <div className="flex items-center gap-3">
                <div className="bg-amber-500/20 p-2 rounded-lg">
                  <Zap className="h-5 w-5 text-amber-500" />
                </div>
                <div>
                  <h3 className="font-semibold text-amber-600 dark:text-amber-400 flex items-center gap-2">
                    SandboxAQ AQAffinity Integration
                    <Badge variant="outline" className="text-xs border-amber-500/50 text-amber-600">Step 4</Badge>
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    AI-powered epitope ranking using OpenFold3 for protein-antibody binding affinity prediction
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Syringe className="h-5 w-5 text-purple-500" />
                    Complete Vaccine Pipeline
                  </CardTitle>
                  <CardDescription>
                    End-to-end vaccine design with DSSP, DiscoTope, NetMHCpan, MAFFT, JCat codon optimization, and ViennaRNA
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>PDB Structure File (Optional)</Label>
                    <div className="border rounded-lg p-3 space-y-3 bg-muted/30">
                      {completePdbFileId && pdbUploads ? (
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 text-purple-500" />
                            <span className="text-sm font-medium">
                              {pdbUploads.find(p => p.id === completePdbFileId)?.fileName || "Selected PDB"}
                            </span>
                            <Badge variant="outline" className="text-xs">
                              {pdbUploads.find(p => p.id === completePdbFileId)?.sequenceLength || 0} residues
                            </Badge>
                          </div>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            onClick={() => setCompletePdbFileId(null)}
                            data-testid="button-remove-complete-pdb"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <label className="flex-1">
                            <input
                              type="file"
                              accept=".pdb"
                              className="hidden"
                              onChange={async (e) => {
                                const file = e.target.files?.[0];
                                if (file) {
                                  setUploadingPdb(true);
                                  try {
                                    const formData = new FormData();
                                    formData.append("file", file);
                                    formData.append("description", "Complete pipeline structure");
                                    formData.append("purpose", "vaccine_complete_pipeline");
                                    const res = await fetch("/api/compute/vaccine/upload-pdb", {
                                      method: "POST",
                                      body: formData,
                                      credentials: "include",
                                    });
                                    if (!res.ok) throw new Error("Upload failed");
                                    const data = await res.json();
                                    queryClient.invalidateQueries({ queryKey: ["/api/compute/vaccine/pdb-uploads"] });
                                    setCompletePdbFileId(data.id);
                                    if (data.extractedSequence) {
                                      setCompleteSequence(data.extractedSequence);
                                    }
                                    toast({ title: "PDB Uploaded", description: `Extracted ${data.sequenceLength || 0} residues` });
                                  } catch (err: any) {
                                    toast({ title: "Upload Failed", description: err.message, variant: "destructive" });
                                  } finally {
                                    setUploadingPdb(false);
                                  }
                                }
                                e.target.value = "";
                              }}
                              disabled={uploadingPdb}
                              data-testid="input-complete-pdb-file"
                            />
                            <Button 
                              variant="outline" 
                              className="w-full cursor-pointer" 
                              disabled={uploadingPdb}
                              asChild
                            >
                              <span>
                                {uploadingPdb ? (
                                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                ) : (
                                  <Upload className="h-4 w-4 mr-2" />
                                )}
                                {uploadingPdb ? "Uploading..." : "Upload PDB File"}
                              </span>
                            </Button>
                          </label>
                          {pdbUploads && pdbUploads.length > 0 && (
                            <Select value={completePdbFileId || ""} onValueChange={(v) => {
                              setCompletePdbFileId(v || null);
                              if (v) {
                                const pdb = pdbUploads.find(p => p.id === v);
                                if (pdb?.extractedSequence) {
                                  setCompleteSequence(pdb.extractedSequence);
                                }
                              }
                            }}>
                              <SelectTrigger className="w-[180px]" data-testid="select-complete-existing-pdb">
                                <SelectValue placeholder="Or select existing" />
                              </SelectTrigger>
                              <SelectContent>
                                {pdbUploads.map((pdb) => (
                                  <SelectItem key={pdb.id} value={pdb.id}>
                                    {pdb.fileName}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          )}
                        </div>
                      )}
                      <p className="text-xs text-muted-foreground">
                        Upload a PDB file to automatically extract protein sequences and enable structure-based analysis (DSSP, DiscoTope)
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Protein Sequence</Label>
                    <Textarea
                      value={completeSequence}
                      onChange={(e) => setCompleteSequence(e.target.value)}
                      placeholder="Enter amino acid sequence or select PDB file above..."
                      rows={4}
                      className="font-mono text-xs"
                      data-testid="textarea-complete-sequence"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Vaccine Type</Label>
                    <Select value={completeVaccineType} onValueChange={setCompleteVaccineType}>
                      <SelectTrigger data-testid="select-complete-vaccine-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="protein_subunit">Protein Subunit</SelectItem>
                        <SelectItem value="mrna">mRNA Vaccine</SelectItem>
                        <SelectItem value="multi_epitope">Multi-Epitope</SelectItem>
                        <SelectItem value="peptide">Peptide Vaccine</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="border rounded-lg p-3 bg-muted/30">
                    <div className="text-sm font-medium mb-2">Pipeline Stages:</div>
                    <div className="space-y-1 text-xs">
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-purple-500/20 flex items-center justify-center text-[10px] font-bold text-purple-600">1</span>
                        Input PDB/Sequence Processing
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-purple-500/20 flex items-center justify-center text-[10px] font-bold text-purple-600">2</span>
                        Epitope Prediction (MHCflurry, DiscoTope)
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-purple-500/20 flex items-center justify-center text-[10px] font-bold text-purple-600">3</span>
                        Conservation Analysis (MAFFT)
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-amber-500/20 flex items-center justify-center text-[10px] font-bold text-amber-600">4</span>
                        <Zap className="h-3 w-3 text-amber-500" />
                        <span className="font-medium text-amber-600">AQAffinity Epitope Ranking</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-purple-500/20 flex items-center justify-center text-[10px] font-bold text-purple-600">5</span>
                        Multi-Epitope Construct Assembly
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-purple-500/20 flex items-center justify-center text-[10px] font-bold text-purple-600">6</span>
                        JCat Codon Optimization
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-4 h-4 rounded-full bg-purple-500/20 flex items-center justify-center text-[10px] font-bold text-purple-600">7</span>
                        mRNA Design (ViennaRNA)
                      </div>
                    </div>
                  </div>
                  <Button
                    onClick={() => completePipelineMutation.mutate()}
                    disabled={completePipelineMutation.isPending || (!completeSequence.trim() && !completePdbFileId)}
                    className="w-full"
                    data-testid="button-run-complete-pipeline"
                  >
                    {completePipelineMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Run Complete Pipeline
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Pipeline Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {completePipelineResult ? (
                    <div className="space-y-4">
                      {completePipelineResult.stages && (
                        <div className="space-y-3">
                          {Object.entries(completePipelineResult.stages).map(([stage, data]: [string, any]) => (
                            <Collapsible key={stage}>
                              <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-lg bg-muted/50 hover:bg-muted">
                                <span className="text-sm font-medium capitalize">{stage.replace(/_/g, ' ')}</span>
                                <ChevronRight className="h-4 w-4" />
                              </CollapsibleTrigger>
                              <CollapsibleContent className="pt-2">
                                <pre className="bg-muted p-3 rounded-lg text-xs overflow-auto max-h-48 font-mono">
                                  {JSON.stringify(data, null, 2)}
                                </pre>
                              </CollapsibleContent>
                            </Collapsible>
                          ))}
                        </div>
                      )}
                      {completePipelineResult.nodeUsed && (
                        <Badge variant="outline" className="text-xs">
                          <Server className="h-3 w-3 mr-1" />
                          {completePipelineResult.nodeUsed}
                        </Badge>
                      )}
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground py-12">
                      Run the complete pipeline to see comprehensive vaccine design results
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="pipeline" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-purple-500" />
                    Full Vaccine Pipeline
                  </CardTitle>
                  <CardDescription>
                    Complete workflow: antigen analysis, epitope prediction, codon optimization, and mRNA design
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Pathogen Name</Label>
                    <Input
                      value={pathogenName}
                      onChange={(e) => setPathogenName(e.target.value)}
                      placeholder="e.g., Novel Coronavirus"
                      data-testid="input-pathogen-name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>PDB Structure File (Optional)</Label>
                    <div className="border rounded-lg p-3 space-y-3 bg-muted/30">
                      {selectedPdbFileId && pdbUploads ? (
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 text-purple-500" />
                            <span className="text-sm font-medium">
                              {pdbUploads.find(p => p.id === selectedPdbFileId)?.fileName || "Selected PDB"}
                            </span>
                            <Badge variant="outline" className="text-xs">
                              {pdbUploads.find(p => p.id === selectedPdbFileId)?.sequenceLength || 0} residues
                            </Badge>
                          </div>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            onClick={() => setSelectedPdbFileId(null)}
                            data-testid="button-remove-pdb"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <label className="flex-1">
                            <input
                              type="file"
                              accept=".pdb"
                              className="hidden"
                              onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) handlePdbUpload(file);
                                e.target.value = "";
                              }}
                              disabled={uploadingPdb}
                              data-testid="input-pdb-file"
                            />
                            <Button 
                              variant="outline" 
                              className="w-full cursor-pointer" 
                              disabled={uploadingPdb}
                              asChild
                            >
                              <span>
                                {uploadingPdb ? (
                                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                ) : (
                                  <Upload className="h-4 w-4 mr-2" />
                                )}
                                {uploadingPdb ? "Uploading..." : "Upload PDB File"}
                              </span>
                            </Button>
                          </label>
                          {pdbUploads && pdbUploads.length > 0 && (
                            <Select value={selectedPdbFileId || ""} onValueChange={setSelectedPdbFileId}>
                              <SelectTrigger className="w-[180px]" data-testid="select-existing-pdb">
                                <SelectValue placeholder="Or select existing" />
                              </SelectTrigger>
                              <SelectContent>
                                {pdbUploads.map((pdb) => (
                                  <SelectItem key={pdb.id} value={pdb.id}>
                                    {pdb.fileName}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          )}
                        </div>
                      )}
                      <p className="text-xs text-muted-foreground">
                        Upload a PDB file to automatically extract protein sequences for analysis
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Protein Sequence(s)</Label>
                    <Textarea
                      value={proteinSequences}
                      onChange={(e) => setProteinSequences(e.target.value)}
                      placeholder="Enter protein sequence(s), one per line, or upload a PDB file above..."
                      rows={6}
                      className="font-mono text-xs"
                      data-testid="textarea-protein-sequences"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Target Organism</Label>
                    <Select value={organism} onValueChange={setOrganism}>
                      <SelectTrigger data-testid="select-organism">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="human">Human</SelectItem>
                        <SelectItem value="mouse">Mouse</SelectItem>
                        <SelectItem value="macaque">Macaque</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Run MD Simulation</Label>
                      <p className="text-xs text-muted-foreground">GPU-intensive stability assessment</p>
                    </div>
                    <Switch
                      checked={runMD}
                      onCheckedChange={setRunMD}
                      data-testid="switch-run-md"
                    />
                  </div>
                  <Button
                    onClick={() => pipelineMutation.mutate()}
                    disabled={pipelineMutation.isPending || !proteinSequences.trim()}
                    className="w-full"
                    data-testid="button-run-pipeline"
                  >
                    {pipelineMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Run Full Pipeline
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Pipeline Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {pipelineResult ? (
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        {pipelineResult.success ? (
                          <Badge className="bg-green-500">Success</Badge>
                        ) : (
                          <Badge variant="destructive">Failed</Badge>
                        )}
                        {pipelineResult.nodeUsed && (
                          <Badge variant="outline">{pipelineResult.nodeUsed}</Badge>
                        )}
                      </div>
                      <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-96 font-mono">
                        {JSON.stringify(pipelineResult, null, 2)}
                      </pre>
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground py-12">
                      Run the pipeline to see results
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="structure" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Atom className="h-5 w-5 text-blue-500" />
                    Structure Prediction
                    <Badge variant="outline" className="ml-2 text-xs">GPU-Intensive</Badge>
                  </CardTitle>
                  <CardDescription>
                    Predict 3D protein structure using ESMFold or AlphaFold2 (15-20x GPU speedup)
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {pdbUploads && pdbUploads.length > 0 && (
                    <div className="space-y-2">
                      <Label>Use Existing PDB Structure (Optional)</Label>
                      <div className="flex items-center gap-2">
                        <Select value={structurePdbFileId || ""} onValueChange={(v) => {
                          setStructurePdbFileId(v || null);
                          if (v) {
                            const pdb = pdbUploads.find(p => p.id === v);
                            if (pdb?.extractedSequence) {
                              setStructureSequence(pdb.extractedSequence);
                            }
                          }
                        }}>
                          <SelectTrigger className="flex-1" data-testid="select-structure-pdb">
                            <SelectValue placeholder="Select uploaded PDB file..." />
                          </SelectTrigger>
                          <SelectContent>
                            {pdbUploads.map((pdb) => (
                              <SelectItem key={pdb.id} value={pdb.id}>
                                {pdb.fileName} ({pdb.sequenceLength || 0} residues)
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        {structurePdbFileId && (
                          <Button variant="ghost" size="icon" onClick={() => setStructurePdbFileId(null)}>
                            <X className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                  )}
                  <div className="space-y-2">
                    <Label>Protein Sequence</Label>
                    <Textarea
                      value={structureSequence}
                      onChange={(e) => setStructureSequence(e.target.value)}
                      placeholder="Enter amino acid sequence or select PDB file above..."
                      rows={4}
                      className="font-mono text-xs"
                      data-testid="textarea-structure-sequence"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Prediction Method</Label>
                    <Select value={structureMethod} onValueChange={setStructureMethod}>
                      <SelectTrigger data-testid="select-structure-method">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="esmfold">ESMFold (Fast)</SelectItem>
                        <SelectItem value="alphafold2">AlphaFold2 (Accurate)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    onClick={() => structureMutation.mutate()}
                    disabled={structureMutation.isPending || (!structureSequence.trim() && !structurePdbFileId)}
                    className="w-full"
                    data-testid="button-predict-structure"
                  >
                    {structureMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Zap className="h-4 w-4 mr-2" />
                    )}
                    Predict Structure
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Structure Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {structureResult ? (
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-96 font-mono">
                      {JSON.stringify(structureResult, null, 2)}
                    </pre>
                  ) : (
                    <div className="text-center text-muted-foreground py-12">
                      Run prediction to see results
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="epitopes" className="space-y-6">
            {/* Pipeline Step Indicator */}
            <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted/30 rounded-lg p-3" data-testid="epitope-pipeline-steps">
              <span className="flex items-center gap-1">
                <span className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center text-xs font-bold text-green-600">2</span>
                Epitope Prediction
              </span>
              <ChevronRight className="h-4 w-4" />
              <span className="flex items-center gap-1">
                <span className="w-5 h-5 rounded-full bg-purple-500/20 flex items-center justify-center text-xs font-bold text-purple-600">3</span>
                Conservation
              </span>
              <ChevronRight className="h-4 w-4" />
              <span className="flex items-center gap-1 bg-amber-500/10 px-2 py-1 rounded-md border border-amber-500/30">
                <span className="w-5 h-5 rounded-full bg-amber-500/20 flex items-center justify-center text-xs font-bold text-amber-600">4</span>
                <Zap className="h-3 w-3 text-amber-500" />
                <span className="font-medium text-amber-600">AQAffinity Ranking</span>
              </span>
              <ChevronRight className="h-4 w-4" />
              <span className="flex items-center gap-1">
                <span className="w-5 h-5 rounded-full bg-purple-500/20 flex items-center justify-center text-xs font-bold text-purple-600">5</span>
                Assembly
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Beaker className="h-5 w-5 text-green-500" />
                    Epitope Prediction
                    <Badge variant="outline" className="ml-2 text-xs">CPU-Intensive</Badge>
                  </CardTitle>
                  <CardDescription>
                    Predict MHC-I/II binding epitopes for T-cell response
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Protein Sequence</Label>
                    <Textarea
                      value={epitopeSequence}
                      onChange={(e) => setEpitopeSequence(e.target.value)}
                      placeholder="Enter amino acid sequence..."
                      rows={4}
                      className="font-mono text-xs"
                      data-testid="textarea-epitope-sequence"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Peptide Length: {peptideLength}</Label>
                    <Slider
                      value={[peptideLength]}
                      onValueChange={(v) => setPeptideLength(v[0])}
                      min={8}
                      max={15}
                      step={1}
                      data-testid="slider-peptide-length"
                    />
                  </div>
                  <Button
                    onClick={() => epitopeMutation.mutate()}
                    disabled={epitopeMutation.isPending || !epitopeSequence.trim()}
                    className="w-full"
                    data-testid="button-predict-epitopes"
                  >
                    {epitopeMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Predict Epitopes
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Epitope Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {epitopeResult ? (
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-96 font-mono">
                      {JSON.stringify(epitopeResult, null, 2)}
                    </pre>
                  ) : (
                    <div className="text-center text-muted-foreground py-12">
                      Run prediction to see results
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="codon" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileCode className="h-5 w-5 text-amber-500" />
                    Codon Optimization
                    <Badge variant="outline" className="ml-2 text-xs">CPU-Only</Badge>
                  </CardTitle>
                  <CardDescription>
                    Optimize codons for expression in target organism
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Protein Sequence</Label>
                    <Textarea
                      value={codonSequence}
                      onChange={(e) => setCodonSequence(e.target.value)}
                      placeholder="Enter amino acid sequence..."
                      rows={4}
                      className="font-mono text-xs"
                      data-testid="textarea-codon-sequence"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Target Organism</Label>
                    <Select value={codonOrganism} onValueChange={setCodonOrganism}>
                      <SelectTrigger data-testid="select-codon-organism">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="human">Human (Homo sapiens)</SelectItem>
                        <SelectItem value="mouse">Mouse (Mus musculus)</SelectItem>
                        <SelectItem value="ecoli">E. coli</SelectItem>
                        <SelectItem value="yeast">Yeast (S. cerevisiae)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    onClick={() => codonMutation.mutate()}
                    disabled={codonMutation.isPending || !codonSequence.trim()}
                    className="w-full"
                    data-testid="button-optimize-codons"
                  >
                    {codonMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Optimize Codons
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Optimized Sequence</CardTitle>
                </CardHeader>
                <CardContent>
                  {codonResult ? (
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-96 font-mono">
                      {JSON.stringify(codonResult, null, 2)}
                    </pre>
                  ) : (
                    <div className="text-center text-muted-foreground py-12">
                      Run optimization to see results
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="mrna" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Dna className="h-5 w-5 text-pink-500" />
                    mRNA Construct Design
                    <Badge variant="outline" className="ml-2 text-xs">CPU-Intensive</Badge>
                  </CardTitle>
                  <CardDescription>
                    Design complete mRNA construct with UTRs, cap, and poly(A) tail
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Protein Sequence</Label>
                    <Textarea
                      value={mrnaSequence}
                      onChange={(e) => setMrnaSequence(e.target.value)}
                      placeholder="Enter amino acid sequence..."
                      rows={4}
                      className="font-mono text-xs"
                      data-testid="textarea-mrna-sequence"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>UTR Type</Label>
                      <Select value={utrType} onValueChange={setUtrType}>
                        <SelectTrigger data-testid="select-utr-type">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="optimized">Optimized</SelectItem>
                          <SelectItem value="alpha_globin">Alpha Globin</SelectItem>
                          <SelectItem value="beta_globin">Beta Globin</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Cap Type</Label>
                      <Select value={capType} onValueChange={setCapType}>
                        <SelectTrigger data-testid="select-cap-type">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="cap1">Cap1 (Recommended)</SelectItem>
                          <SelectItem value="cap0">Cap0</SelectItem>
                          <SelectItem value="arca">ARCA</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Poly(A) Length: {polyALength} nt</Label>
                    <Slider
                      value={[polyALength]}
                      onValueChange={(v) => setPolyALength(v[0])}
                      min={50}
                      max={200}
                      step={10}
                      data-testid="slider-poly-a-length"
                    />
                  </div>
                  <Button
                    onClick={() => mrnaMutation.mutate()}
                    disabled={mrnaMutation.isPending || !mrnaSequence.trim()}
                    className="w-full"
                    data-testid="button-design-mrna"
                  >
                    {mrnaMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Dna className="h-4 w-4 mr-2" />
                    )}
                    Design mRNA Construct
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>mRNA Construct</CardTitle>
                </CardHeader>
                <CardContent>
                  {mrnaResult ? (
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-96 font-mono">
                      {JSON.stringify(mrnaResult, null, 2)}
                    </pre>
                  ) : (
                    <div className="text-center text-muted-foreground py-12">
                      Run design to see results
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="task-matrix" className="space-y-6">
            {taskMatrixLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : taskMatrix ? (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                  <Card className="col-span-1">
                    <CardContent className="pt-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold">{taskMatrix.summary.totalTasks}</div>
                        <div className="text-sm text-muted-foreground">Total Tasks</div>
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="col-span-1">
                    <CardContent className="pt-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-purple-500">{taskMatrix.summary.typeCounts.GPU_INTENSIVE}</div>
                        <div className="text-sm text-muted-foreground">GPU-Intensive</div>
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="col-span-1">
                    <CardContent className="pt-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-blue-500">{taskMatrix.summary.typeCounts.GPU_PREFERRED}</div>
                        <div className="text-sm text-muted-foreground">GPU-Preferred</div>
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="col-span-1">
                    <CardContent className="pt-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-amber-500">{taskMatrix.summary.typeCounts.CPU_INTENSIVE}</div>
                        <div className="text-sm text-muted-foreground">CPU-Intensive</div>
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="col-span-1">
                    <CardContent className="pt-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-slate-500">{taskMatrix.summary.typeCounts.CPU_ONLY}</div>
                        <div className="text-sm text-muted-foreground">CPU-Only</div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Grid3X3 className="h-5 w-5 text-purple-500" />
                      Task Classification Matrix
                    </CardTitle>
                    <CardDescription>
                      Hardware routing map showing optimal execution environment for each task
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {Object.entries(taskMatrix.taskClassification).map(([categoryKey, category]) => (
                        <Collapsible
                          key={categoryKey}
                          open={expandedCategories.has(categoryKey)}
                          onOpenChange={() => toggleCategory(categoryKey)}
                        >
                          <CollapsibleTrigger className="flex items-center justify-between w-full p-3 bg-muted rounded-lg hover-elevate" data-testid={`category-${categoryKey}`}>
                            <div className="flex items-center gap-3">
                              {expandedCategories.has(categoryKey) ? (
                                <ChevronDown className="h-4 w-4" />
                              ) : (
                                <ChevronRight className="h-4 w-4" />
                              )}
                              <div className="text-left">
                                <div className="font-medium">{categoryKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                                <div className="text-xs text-muted-foreground">Stage {category.stage}: {category.stageName}</div>
                              </div>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {Object.keys(category.tasks).length} tasks
                            </Badge>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <Table className="mt-2">
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="w-[200px]">Task</TableHead>
                                  <TableHead className="w-[120px]">Type</TableHead>
                                  <TableHead>Reason</TableHead>
                                  <TableHead className="w-[100px]">Resources</TableHead>
                                  <TableHead className="w-[80px]">Speedup</TableHead>
                                  <TableHead className="w-[150px]">Tools</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {Object.entries(category.tasks).map(([taskKey, task]) => (
                                  <TableRow key={taskKey}>
                                    <TableCell className="font-medium text-sm">
                                      {taskKey.replace(/_/g, ' ')}
                                    </TableCell>
                                    <TableCell>
                                      <Badge variant="outline" className={`text-xs ${getTypeColor(task.type)}`}>
                                        {task.type.replace(/_/g, ' ')}
                                      </Badge>
                                    </TableCell>
                                    <TableCell className="text-sm text-muted-foreground">
                                      {task.reason}
                                    </TableCell>
                                    <TableCell className="text-xs">
                                      {task.cpuCores && <div>{task.cpuCores} CPU</div>}
                                      {task.memoryGb && <div>{task.memoryGb}GB RAM</div>}
                                      {task.gpuMemoryGb && <div>{task.gpuMemoryGb}GB GPU</div>}
                                    </TableCell>
                                    <TableCell>
                                      {task.speedup && (
                                        <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/30 text-xs">
                                          {task.speedup}
                                        </Badge>
                                      )}
                                    </TableCell>
                                    <TableCell className="text-xs text-muted-foreground">
                                      {task.tools?.join(', ')}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </CollapsibleContent>
                        </Collapsible>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Server className="h-5 w-5 text-blue-500" />
                        Hardware Requirements
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {Object.entries(taskMatrix.hardwareRequirements).map(([key, config]: [string, any]) => (
                          <div key={key} className="p-4 bg-muted rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-medium capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                              <Badge variant="outline" className="text-xs">
                                <Clock className="h-3 w-3 mr-1" />
                                {config.estimatedHours}h
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mb-2">{config.description}</p>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              <div className="text-center p-2 bg-background rounded">
                                <div className="font-medium">{config.cpuCores}</div>
                                <div className="text-muted-foreground">CPU Cores</div>
                              </div>
                              <div className="text-center p-2 bg-background rounded">
                                <div className="font-medium">{config.memoryGb}GB</div>
                                <div className="text-muted-foreground">Memory</div>
                              </div>
                              <div className="text-center p-2 bg-background rounded">
                                <div className="font-medium text-xs">{typeof config.gpu === 'string' ? config.gpu : config.gpu}</div>
                                <div className="text-muted-foreground">GPU</div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <DollarSign className="h-5 w-5 text-green-500" />
                        Cost Analysis
                      </CardTitle>
                      <CardDescription>
                        GPU vs CPU cost comparison for common workloads
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Workload</TableHead>
                            <TableHead className="text-right">GPU Cost</TableHead>
                            <TableHead className="text-right">CPU Cost</TableHead>
                            <TableHead className="text-right">Savings</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(taskMatrix.costAnalysis).map(([key, data]: [string, any]) => (
                            <TableRow key={key}>
                              <TableCell className="font-medium text-sm">
                                {key.replace(/([A-Z])/g, ' $1').replace(/(\d+)/g, ' $1').trim()}
                              </TableCell>
                              <TableCell className="text-right text-green-600">${data.gpuCostUsd}</TableCell>
                              <TableCell className="text-right text-red-600">${data.cpuCostUsd}</TableCell>
                              <TableCell className="text-right">
                                <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/30">
                                  {data.gpuSavingsPct}%
                                </Badge>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-12">
                No task matrix data available
              </div>
            )}
          </TabsContent>

          <TabsContent value="hardware" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Server className="h-5 w-5 text-slate-500" />
                    Compute Infrastructure
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {hardwareLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  ) : hardwareReport ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center p-4 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">{hardwareReport.totalNodes}</div>
                          <div className="text-sm text-muted-foreground">Total Nodes</div>
                        </div>
                        <div className="text-center p-4 bg-muted rounded-lg">
                          <div className="text-2xl font-bold text-green-500">{hardwareReport.activeNodes}</div>
                          <div className="text-sm text-muted-foreground">Active</div>
                        </div>
                        <div className="text-center p-4 bg-muted rounded-lg">
                          <div className="text-2xl font-bold text-purple-500">{hardwareReport.gpuNodes}</div>
                          <div className="text-sm text-muted-foreground">GPU Nodes</div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      No hardware information available
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="h-5 w-5 text-blue-500" />
                    Task Routing
                  </CardTitle>
                  <CardDescription>
                    Intelligent routing based on workload characteristics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {hardwareReport?.taskRouting ? (
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="h-4 w-4 text-purple-500" />
                          <span className="font-medium text-sm">GPU-Intensive (15-100x speedup)</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {hardwareReport.taskRouting.gpu_intensive.map((task) => (
                            <Badge key={task} variant="outline" className="text-xs">{task}</Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Activity className="h-4 w-4 text-blue-500" />
                          <span className="font-medium text-sm">GPU-Preferred (2-5x speedup)</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {hardwareReport.taskRouting.gpu_preferred.map((task) => (
                            <Badge key={task} variant="outline" className="text-xs">{task}</Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Cpu className="h-4 w-4 text-amber-500" />
                          <span className="font-medium text-sm">CPU-Intensive</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {hardwareReport.taskRouting.cpu_intensive.map((task) => (
                            <Badge key={task} variant="outline" className="text-xs">{task}</Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Server className="h-4 w-4 text-slate-500" />
                          <span className="font-medium text-sm">CPU-Only</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {hardwareReport.taskRouting.cpu_only.map((task) => (
                            <Badge key={task} variant="outline" className="text-xs">{task}</Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      No routing information available
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
