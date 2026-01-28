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
  Beaker
} from "lucide-react";

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

export default function VaccineDiscoveryPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("pipeline");
  
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

  const { data: hardwareReport, isLoading: hardwareLoading } = useQuery<HardwareReport>({
    queryKey: ["/api/compute/vaccine/hardware"],
  });

  const pipelineMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/compute/vaccine/pipeline", {
        pathogenName,
        proteinSequences: proteinSequences.split("\n").filter(s => s.trim()),
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
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="pipeline" data-testid="tab-pipeline">
              <Activity className="h-4 w-4 mr-2" />
              Full Pipeline
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
              Codon Opt
            </TabsTrigger>
            <TabsTrigger value="mrna" data-testid="tab-mrna">
              <Dna className="h-4 w-4 mr-2" />
              mRNA Design
            </TabsTrigger>
            <TabsTrigger value="hardware" data-testid="tab-hardware">
              <Server className="h-4 w-4 mr-2" />
              Hardware
            </TabsTrigger>
          </TabsList>

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
                    <Label>Protein Sequence(s)</Label>
                    <Textarea
                      value={proteinSequences}
                      onChange={(e) => setProteinSequences(e.target.value)}
                      placeholder="Enter protein sequence(s), one per line..."
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
                  <div className="space-y-2">
                    <Label>Protein Sequence</Label>
                    <Textarea
                      value={structureSequence}
                      onChange={(e) => setStructureSequence(e.target.value)}
                      placeholder="Enter amino acid sequence..."
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
                    disabled={structureMutation.isPending || !structureSequence.trim()}
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
