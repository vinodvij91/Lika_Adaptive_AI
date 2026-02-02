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
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { 
  Atom, 
  Dna, 
  Microscope, 
  Beaker,
  ChevronRight, 
  CheckCircle2,
  AlertCircle,
  Loader2,
  Play,
  FileText,
  Sparkles,
  Activity
} from "lucide-react";

interface BioNemoStatus {
  configured: boolean;
  provider: string;
  capabilities: string[];
  modules: {
    molmim: { name: string; description: string; available: boolean };
    genmol: { name: string; description: string; available: boolean };
    alphafold2: { name: string; description: string; available: boolean };
    spectroscopy: { name: string; description: string; available: boolean };
  };
}

interface MolMIMResult {
  molecules: Array<{ smiles: string; score: number; similarity: number }>;
}

interface GenMolResult {
  status: string;
  molecules: Array<{ smiles: string; score: number }>;
}

interface AlphaFold2Result {
  pdbData: string;
  confidenceScore: number;
  structureMetrics: { pLDDT: number; pTM: number; numResidues: number };
}

interface SpectroscopyResult {
  functionalGroups: string[];
  molecularFingerprint: string;
  suggestedStructures: string[];
  confidence: number;
  interpretation: string;
}

export default function BioNemoPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("molmim");
  
  const [molmimSmiles, setMolmimSmiles] = useState("CCO");
  const [molmimNumMolecules, setMolmimNumMolecules] = useState(5);
  const [molmimProperty, setMolmimProperty] = useState<"QED" | "plogP">("QED");
  const [molmimResults, setMolmimResults] = useState<MolMIMResult | null>(null);
  
  const [genmolSmiles, setGenmolSmiles] = useState("c1ccccc1");
  const [genmolNumMolecules, setGenmolNumMolecules] = useState(5);
  const [genmolTemperature, setGenmolTemperature] = useState(2.0);
  const [genmolScoring, setGenmolScoring] = useState<"QED" | "plogP" | "SA">("QED");
  const [genmolResults, setGenmolResults] = useState<GenMolResult | null>(null);
  
  const [alphaSequence, setAlphaSequence] = useState("MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLFRPGQKNSL");
  const [alphaDatabases, setAlphaDatabases] = useState<string[]>(["uniref90", "small_bfd"]);
  const [alphaResults, setAlphaResults] = useState<AlphaFold2Result | null>(null);
  
  const [spectroType, setSpectroType] = useState<"FTIR" | "Raman" | "NMR" | "UV-Vis" | "Mass">("FTIR");
  const [spectroPeaks, setSpectroPeaks] = useState("3400, 0.8\n2950, 0.6\n1720, 0.9\n1200, 0.5");
  const [spectroResults, setSpectroResults] = useState<SpectroscopyResult | null>(null);

  const { data: status, isLoading: statusLoading } = useQuery<BioNemoStatus>({
    queryKey: ["/api/bionemo/status"],
  });

  const molmimMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/bionemo/generate", {
        smiles: molmimSmiles,
        propertyName: molmimProperty,
        numMolecules: molmimNumMolecules,
        minSimilarity: 0.4,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setMolmimResults(data);
      toast({ title: "MolMIM Complete", description: `Generated ${data.molecules?.length || 0} molecules` });
    },
    onError: (error: any) => {
      toast({ title: "MolMIM Error", description: error.message, variant: "destructive" });
    },
  });

  const genmolMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/bionemo/genmol/generate", {
        smiles: genmolSmiles,
        numMolecules: genmolNumMolecules,
        temperature: genmolTemperature,
        scoring: genmolScoring,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setGenmolResults(data);
      toast({ title: "GenMol Complete", description: `Generated ${data.molecules?.length || 0} molecules` });
    },
    onError: (error: any) => {
      toast({ title: "GenMol Error", description: error.message, variant: "destructive" });
    },
  });

  const alphafoldMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/bionemo/alphafold2/predict", {
        sequences: [alphaSequence],
        databases: alphaDatabases,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setAlphaResults(data);
      toast({ title: "AlphaFold2 Complete", description: `pLDDT: ${data.structureMetrics?.pLDDT?.toFixed(1) || "N/A"}` });
    },
    onError: (error: any) => {
      toast({ title: "AlphaFold2 Error", description: error.message, variant: "destructive" });
    },
  });

  const spectroMutation = useMutation({
    mutationFn: async () => {
      const peaks = spectroPeaks.split("\n").filter(l => l.trim()).map(line => {
        const [pos, int] = line.split(",").map(s => parseFloat(s.trim()));
        return { position: pos, intensity: int || 1 };
      });
      const res = await apiRequest("POST", "/api/bionemo/spectroscopy/analyze", {
        type: spectroType,
        peaks,
        metadata: {},
      });
      return res.json();
    },
    onSuccess: (data) => {
      setSpectroResults(data);
      toast({ title: "Analysis Complete", description: `Found ${data.functionalGroups?.length || 0} functional groups` });
    },
    onError: (error: any) => {
      toast({ title: "Spectroscopy Error", description: error.message, variant: "destructive" });
    },
  });

  if (statusLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-auto">
      <div className="border-b bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-green-500/20 to-emerald-600/20">
                <Atom className="h-6 w-6 text-green-500" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">BioNeMo AI</h1>
                <p className="text-sm text-muted-foreground">NVIDIA Drug Discovery & Molecular Analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {status?.configured ? (
                <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/30">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="outline" className="bg-amber-500/10 text-amber-600 border-amber-500/30">
                  <AlertCircle className="h-3 w-3 mr-1" />
                  Not Configured
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {status?.modules && Object.entries(status.modules).map(([key, module]) => (
            <Card 
              key={key} 
              data-testid={`card-module-${key}`}
              className={`cursor-pointer transition-all ${activeTab === key ? 'ring-2 ring-primary' : 'hover-elevate'}`}
              onClick={() => setActiveTab(key)}
            >
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  {key === "molmim" && <Beaker className="h-5 w-5 text-blue-500" />}
                  {key === "genmol" && <Sparkles className="h-5 w-5 text-purple-500" />}
                  {key === "alphafold2" && <Dna className="h-5 w-5 text-green-500" />}
                  {key === "spectroscopy" && <Activity className="h-5 w-5 text-orange-500" />}
                  <div>
                    <h3 className="font-medium">{module.name}</h3>
                    <p className="text-xs text-muted-foreground">{module.description}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="molmim" data-testid="tab-molmim">
              <Beaker className="h-4 w-4 mr-2" />
              MolMIM
            </TabsTrigger>
            <TabsTrigger value="genmol" data-testid="tab-genmol">
              <Sparkles className="h-4 w-4 mr-2" />
              GenMol
            </TabsTrigger>
            <TabsTrigger value="alphafold2" data-testid="tab-alphafold2">
              <Dna className="h-4 w-4 mr-2" />
              AlphaFold2
            </TabsTrigger>
            <TabsTrigger value="spectroscopy" data-testid="tab-spectroscopy">
              <Activity className="h-4 w-4 mr-2" />
              Spectroscopy
            </TabsTrigger>
          </TabsList>

          <TabsContent value="molmim">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Beaker className="h-5 w-5 text-blue-500" />
                    MolMIM Optimization
                  </CardTitle>
                  <CardDescription>
                    Optimize molecules using Masked Image Modeling
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Seed SMILES</Label>
                    <Input
                      data-testid="input-molmim-smiles"
                      value={molmimSmiles}
                      onChange={(e) => setMolmimSmiles(e.target.value)}
                      placeholder="Enter SMILES string..."
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Property to Optimize</Label>
                      <Select value={molmimProperty} onValueChange={(v) => setMolmimProperty(v as any)}>
                        <SelectTrigger data-testid="select-molmim-property">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="QED">QED (Drug-likeness)</SelectItem>
                          <SelectItem value="plogP">pLogP (Lipophilicity)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Number of Molecules: {molmimNumMolecules}</Label>
                      <Slider
                        value={[molmimNumMolecules]}
                        onValueChange={([v]) => setMolmimNumMolecules(v)}
                        min={1}
                        max={20}
                        step={1}
                      />
                    </div>
                  </div>
                  <Button 
                    data-testid="button-molmim-run"
                    onClick={() => molmimMutation.mutate()}
                    disabled={molmimMutation.isPending || !status?.configured}
                    className="w-full"
                  >
                    {molmimMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Run MolMIM Optimization
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {molmimResults?.molecules && molmimResults.molecules.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>SMILES</TableHead>
                          <TableHead className="text-right">Score</TableHead>
                          <TableHead className="text-right">Similarity</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {molmimResults.molecules.map((mol, i) => (
                          <TableRow key={i}>
                            <TableCell className="font-mono text-xs max-w-[200px] truncate">
                              {mol.smiles}
                            </TableCell>
                            <TableCell className="text-right">{mol.score?.toFixed(3)}</TableCell>
                            <TableCell className="text-right">{mol.similarity?.toFixed(3)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Beaker className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>Run MolMIM to see optimized molecules</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="genmol">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-purple-500" />
                    GenMol Generation
                  </CardTitle>
                  <CardDescription>
                    Generate novel molecules with target properties
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Seed SMILES</Label>
                    <Input
                      data-testid="input-genmol-smiles"
                      value={genmolSmiles}
                      onChange={(e) => setGenmolSmiles(e.target.value)}
                      placeholder="Enter seed SMILES..."
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Scoring Function</Label>
                      <Select value={genmolScoring} onValueChange={(v) => setGenmolScoring(v as any)}>
                        <SelectTrigger data-testid="select-genmol-scoring">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="QED">QED</SelectItem>
                          <SelectItem value="plogP">pLogP</SelectItem>
                          <SelectItem value="SA">SA Score</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Molecules: {genmolNumMolecules}</Label>
                      <Slider
                        value={[genmolNumMolecules]}
                        onValueChange={([v]) => setGenmolNumMolecules(v)}
                        min={1}
                        max={20}
                        step={1}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Temperature: {genmolTemperature.toFixed(1)}</Label>
                    <Slider
                      value={[genmolTemperature]}
                      onValueChange={([v]) => setGenmolTemperature(v)}
                      min={0.5}
                      max={5.0}
                      step={0.1}
                    />
                  </div>
                  <Button 
                    data-testid="button-genmol-run"
                    onClick={() => genmolMutation.mutate()}
                    disabled={genmolMutation.isPending || !status?.configured}
                    className="w-full"
                  >
                    {genmolMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    Generate Molecules
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Generated Molecules</CardTitle>
                </CardHeader>
                <CardContent>
                  {genmolResults?.molecules && genmolResults.molecules.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>SMILES</TableHead>
                          <TableHead className="text-right">Score</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {genmolResults.molecules.map((mol, i) => (
                          <TableRow key={i}>
                            <TableCell className="font-mono text-xs max-w-[250px] truncate">
                              {mol.smiles}
                            </TableCell>
                            <TableCell className="text-right">{mol.score?.toFixed(3)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Sparkles className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>Generate molecules to see results</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="alphafold2">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Dna className="h-5 w-5 text-green-500" />
                    AlphaFold2 Structure Prediction
                  </CardTitle>
                  <CardDescription>
                    Predict 3D protein structure from amino acid sequence
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Protein Sequence (Amino Acids)</Label>
                    <Textarea
                      data-testid="input-alphafold-sequence"
                      value={alphaSequence}
                      onChange={(e) => setAlphaSequence(e.target.value.toUpperCase().replace(/[^A-Z]/g, ""))}
                      placeholder="Enter amino acid sequence (e.g., MKFLILLFNILCLFPVLAAD...)"
                      rows={4}
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-muted-foreground">
                      {alphaSequence.length} residues
                    </p>
                  </div>
                  <Button 
                    data-testid="button-alphafold-run"
                    onClick={() => alphafoldMutation.mutate()}
                    disabled={alphafoldMutation.isPending || !status?.configured || alphaSequence.length < 10}
                    className="w-full"
                  >
                    {alphafoldMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Dna className="h-4 w-4 mr-2" />
                    )}
                    Predict Structure
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Prediction Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {alphaResults ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center p-3 rounded-lg bg-muted/50">
                          <div className="text-2xl font-bold text-green-500">
                            {alphaResults.structureMetrics.pLDDT.toFixed(1)}
                          </div>
                          <div className="text-xs text-muted-foreground">pLDDT Score</div>
                        </div>
                        <div className="text-center p-3 rounded-lg bg-muted/50">
                          <div className="text-2xl font-bold text-blue-500">
                            {alphaResults.structureMetrics.pTM.toFixed(2)}
                          </div>
                          <div className="text-xs text-muted-foreground">pTM Score</div>
                        </div>
                        <div className="text-center p-3 rounded-lg bg-muted/50">
                          <div className="text-2xl font-bold text-purple-500">
                            {alphaResults.structureMetrics.numResidues}
                          </div>
                          <div className="text-xs text-muted-foreground">Residues</div>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>PDB Data Preview</Label>
                        <pre className="p-3 rounded-lg bg-muted/50 text-xs font-mono overflow-auto max-h-48">
                          {alphaResults.pdbData.slice(0, 500)}
                          {alphaResults.pdbData.length > 500 && "..."}
                        </pre>
                      </div>
                      <Badge variant="outline" className="bg-green-500/10 text-green-600">
                        Confidence: {(alphaResults.confidenceScore * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Dna className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>Run AlphaFold2 to predict protein structure</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="spectroscopy">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-orange-500" />
                    Spectroscopy Analysis
                  </CardTitle>
                  <CardDescription>
                    Analyze FTIR, Raman, NMR, UV-Vis, or Mass spectrometry data
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Spectroscopy Type</Label>
                    <Select value={spectroType} onValueChange={(v) => setSpectroType(v as any)}>
                      <SelectTrigger data-testid="select-spectro-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="FTIR">FTIR (Infrared)</SelectItem>
                        <SelectItem value="Raman">Raman</SelectItem>
                        <SelectItem value="NMR">NMR</SelectItem>
                        <SelectItem value="UV-Vis">UV-Vis</SelectItem>
                        <SelectItem value="Mass">Mass Spectrometry</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Peak Data (position, intensity per line)</Label>
                    <Textarea
                      data-testid="input-spectro-peaks"
                      value={spectroPeaks}
                      onChange={(e) => setSpectroPeaks(e.target.value)}
                      placeholder="3400, 0.8&#10;2950, 0.6&#10;1720, 0.9"
                      rows={6}
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-muted-foreground">
                      Format: wavenumber/position, intensity (one per line)
                    </p>
                  </div>
                  <Button 
                    data-testid="button-spectro-run"
                    onClick={() => spectroMutation.mutate()}
                    disabled={spectroMutation.isPending}
                    className="w-full"
                  >
                    {spectroMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Microscope className="h-4 w-4 mr-2" />
                    )}
                    Analyze Spectrum
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                </CardHeader>
                <CardContent>
                  {spectroResults ? (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label>Functional Groups Detected</Label>
                        <div className="flex flex-wrap gap-2">
                          {spectroResults.functionalGroups.map((group, i) => (
                            <Badge key={i} variant="secondary">{group}</Badge>
                          ))}
                        </div>
                      </div>
                      {spectroResults.suggestedStructures.length > 0 && (
                        <div className="space-y-2">
                          <Label>Suggested Structure Classes</Label>
                          <div className="flex flex-wrap gap-2">
                            {spectroResults.suggestedStructures.map((s, i) => (
                              <Badge key={i} className="bg-purple-500/10 text-purple-600">{s}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="space-y-2">
                        <Label>Interpretation</Label>
                        <p className="text-sm text-muted-foreground p-3 rounded-lg bg-muted/50">
                          {spectroResults.interpretation}
                        </p>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Spectral Fingerprint</span>
                        <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
                          {spectroResults.molecularFingerprint}
                        </code>
                      </div>
                      <Badge variant="outline" className="bg-orange-500/10 text-orange-600">
                        Confidence: {(spectroResults.confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Activity className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>Run analysis to interpret spectroscopy data</p>
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
