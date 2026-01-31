import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { 
  Loader2, 
  Zap, 
  FlaskConical, 
  Target, 
  Activity,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Beaker
} from "lucide-react";

type Pipeline = "drug_discovery" | "vaccine_discovery" | "materials_discovery";

interface SinglePrediction {
  proteinSequence: string;
  proteinLength: number;
  ligandSmiles: string;
  predictedAffinity: number;
  affinityUnit: string;
  confidenceScore: number;
  predictionMethod: string;
  modelVersion: string;
  pipelineType: string;
  isStrongBinder: boolean;
  metadata: Record<string, any>;
}

interface BatchResult {
  totalCount: number;
  successfulCount: number;
  failedCount: number;
  averageAffinity: number;
  topBinders: Array<{
    index: number;
    ligandSmiles: string;
    predictedAffinity: number;
    affinityUnit: string;
    confidenceScore: number;
    isStrongBinder: boolean;
  }>;
  allPredictions: Array<{
    index: number;
    ligandSmiles: string;
    predictedAffinity: number;
    affinityUnit: string;
    confidenceScore: number;
    isStrongBinder: boolean;
  }>;
  pipelineType: string;
}

interface ScreeningResult {
  totalScreened: number;
  totalHits: number;
  hitRate: number;
  thresholdNm: number;
  hits: Array<{
    name: string;
    smiles: string;
    predictedIc50Nm: number;
    confidence: number;
    isHit: boolean;
  }>;
  allResults: Array<{
    name: string;
    smiles: string;
    predictedIc50Nm: number;
    confidence: number;
    isHit: boolean;
  }>;
  top10: Array<{
    name: string;
    smiles: string;
    predictedIc50Nm: number;
    confidence: number;
    isHit: boolean;
  }>;
  pipelineType: string;
  recommendations: string[];
}

export default function AQAffinityPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("single");
  
  const [proteinSequence, setProteinSequence] = useState("");
  const [ligandSmiles, setLigandSmiles] = useState("");
  const [pipeline, setPipeline] = useState<Pipeline>("drug_discovery");
  
  const [batchSmiles, setBatchSmiles] = useState("");
  const [topN, setTopN] = useState("10");
  
  const [libraryInput, setLibraryInput] = useState("");
  const [threshold, setThreshold] = useState("100");
  
  const [singleResult, setSingleResult] = useState<SinglePrediction | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [screeningResult, setScreeningResult] = useState<ScreeningResult | null>(null);

  const singlePredictMutation = useMutation({
    mutationFn: async (data: { proteinSequence: string; ligandSmiles: string; pipeline: Pipeline }) => {
      const response = await apiRequest("POST", "/api/compute/aqaffinity/predict", data);
      return response.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        setSingleResult(data.prediction);
        toast({ title: "Prediction Complete", description: "Binding affinity calculated successfully" });
      } else {
        toast({ title: "Error", description: data.error || "Prediction failed", variant: "destructive" });
      }
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const batchPredictMutation = useMutation({
    mutationFn: async (data: { proteinSequence: string; ligandSmilesList: string[]; pipeline: Pipeline; topN: number }) => {
      const response = await apiRequest("POST", "/api/compute/aqaffinity/batch-predict", data);
      return response.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        setBatchResult(data.batchResults);
        toast({ title: "Batch Prediction Complete", description: `${data.batchResults.successfulCount} predictions completed` });
      } else {
        toast({ title: "Error", description: data.error || "Batch prediction failed", variant: "destructive" });
      }
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const screeningMutation = useMutation({
    mutationFn: async (data: { proteinSequence: string; compoundLibrary: Array<{ name: string; smiles: string }>; affinityThresholdNm: number; pipeline: Pipeline }) => {
      const response = await apiRequest("POST", "/api/compute/aqaffinity/screen-library", data);
      return response.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        setScreeningResult(data.screeningResults);
        toast({ title: "Screening Complete", description: `Found ${data.screeningResults.totalHits} hits from ${data.screeningResults.totalScreened} compounds` });
      } else {
        toast({ title: "Error", description: data.error || "Screening failed", variant: "destructive" });
      }
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const handleSinglePredict = () => {
    if (!proteinSequence || proteinSequence.length < 10) {
      toast({ title: "Validation Error", description: "Protein sequence must be at least 10 amino acids", variant: "destructive" });
      return;
    }
    if (!ligandSmiles) {
      toast({ title: "Validation Error", description: "Ligand SMILES is required", variant: "destructive" });
      return;
    }
    setSingleResult(null);
    singlePredictMutation.mutate({ proteinSequence, ligandSmiles, pipeline });
  };

  const handleBatchPredict = () => {
    if (!proteinSequence || proteinSequence.length < 10) {
      toast({ title: "Validation Error", description: "Protein sequence must be at least 10 amino acids", variant: "destructive" });
      return;
    }
    const smilesList = batchSmiles.split("\n").map(s => s.trim()).filter(s => s.length > 0);
    if (smilesList.length === 0) {
      toast({ title: "Validation Error", description: "At least one SMILES is required", variant: "destructive" });
      return;
    }
    setBatchResult(null);
    batchPredictMutation.mutate({ 
      proteinSequence, 
      ligandSmilesList: smilesList, 
      pipeline, 
      topN: parseInt(topN) || 10 
    });
  };

  const handleScreening = () => {
    if (!proteinSequence || proteinSequence.length < 10) {
      toast({ title: "Validation Error", description: "Protein sequence must be at least 10 amino acids", variant: "destructive" });
      return;
    }
    
    const lines = libraryInput.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    const compoundLibrary = lines.map((line, idx) => {
      const parts = line.split(",");
      if (parts.length >= 2) {
        return { name: parts[0].trim(), smiles: parts[1].trim() };
      }
      return { name: `Compound_${idx + 1}`, smiles: line };
    });
    
    if (compoundLibrary.length === 0) {
      toast({ title: "Validation Error", description: "At least one compound is required", variant: "destructive" });
      return;
    }
    
    setScreeningResult(null);
    screeningMutation.mutate({
      proteinSequence,
      compoundLibrary,
      affinityThresholdNm: parseFloat(threshold) || 100,
      pipeline
    });
  };

  const getAffinityColor = (affinity: number) => {
    if (affinity < 10) return "text-green-600 dark:text-green-400";
    if (affinity < 100) return "text-emerald-600 dark:text-emerald-400";
    if (affinity < 1000) return "text-yellow-600 dark:text-yellow-400";
    if (affinity < 10000) return "text-orange-600 dark:text-orange-400";
    return "text-red-600 dark:text-red-400";
  };

  const getAffinityLabel = (affinity: number) => {
    if (affinity < 10) return "Extremely Strong";
    if (affinity < 100) return "Strong";
    if (affinity < 1000) return "Moderate";
    if (affinity < 10000) return "Weak";
    return "Very Weak";
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2" data-testid="text-page-title">
            <Zap className="h-6 w-6 text-primary" />
            AQAffinity Binding Prediction
          </h1>
          <p className="text-muted-foreground mt-1">
            Structure-free protein-ligand binding affinity prediction powered by SandboxAQ
          </p>
        </div>
        <Badge variant="outline" className="gap-1">
          <AlertCircle className="h-3 w-3" />
          Simulation Mode
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <FlaskConical className="h-4 w-4" />
              Drug Discovery
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Screen drug candidates against therapeutic targets</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Target className="h-4 w-4" />
              Vaccine Discovery
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Analyze epitope-MHC binding interactions</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Materials Discovery
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Predict catalyst-substrate and polymer binding</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Target Protein</CardTitle>
          <CardDescription>Enter the amino acid sequence of your target protein</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="protein-sequence">Protein Sequence (single-letter amino acid code)</Label>
            <Textarea
              id="protein-sequence"
              data-testid="input-protein-sequence"
              placeholder="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG..."
              value={proteinSequence}
              onChange={(e) => setProteinSequence(e.target.value.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, ""))}
              className="font-mono text-sm min-h-[100px]"
            />
            <p className="text-xs text-muted-foreground">
              {proteinSequence.length} amino acids
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="pipeline">Discovery Pipeline</Label>
            <Select value={pipeline} onValueChange={(v) => setPipeline(v as Pipeline)}>
              <SelectTrigger id="pipeline" data-testid="select-pipeline">
                <SelectValue placeholder="Select pipeline" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="drug_discovery">Drug Discovery</SelectItem>
                <SelectItem value="vaccine_discovery">Vaccine Discovery</SelectItem>
                <SelectItem value="materials_discovery">Materials Discovery</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="single" data-testid="tab-single">Single Prediction</TabsTrigger>
          <TabsTrigger value="batch" data-testid="tab-batch">Batch Prediction</TabsTrigger>
          <TabsTrigger value="screening" data-testid="tab-screening">Library Screening</TabsTrigger>
        </TabsList>

        <TabsContent value="single" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Single Ligand Prediction</CardTitle>
              <CardDescription>Predict binding affinity for a single protein-ligand pair</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="ligand-smiles">Ligand SMILES</Label>
                <Input
                  id="ligand-smiles"
                  data-testid="input-ligand-smiles"
                  placeholder="CC(=O)Oc1ccccc1C(=O)O"
                  value={ligandSmiles}
                  onChange={(e) => setLigandSmiles(e.target.value)}
                  className="font-mono"
                />
              </div>
              <Button 
                onClick={handleSinglePredict} 
                disabled={singlePredictMutation.isPending}
                data-testid="button-predict-single"
              >
                {singlePredictMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Predict Binding Affinity
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {singleResult && (
            <Card className="border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  Prediction Result
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Predicted IC50</p>
                    <p className={`text-2xl font-bold ${getAffinityColor(singleResult.predictedAffinity)}`} data-testid="text-predicted-affinity">
                      {singleResult.predictedAffinity.toFixed(2)} nM
                    </p>
                    <Badge variant={singleResult.isStrongBinder ? "default" : "secondary"}>
                      {getAffinityLabel(singleResult.predictedAffinity)}
                    </Badge>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Confidence</p>
                    <p className="text-2xl font-bold" data-testid="text-confidence">
                      {(singleResult.confidenceScore * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Strong Binder</p>
                    <p className="text-2xl font-bold" data-testid="text-strong-binder">
                      {singleResult.isStrongBinder ? "Yes" : "No"}
                    </p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Model</p>
                    <p className="text-sm font-medium">{singleResult.predictionMethod}</p>
                    <p className="text-xs text-muted-foreground">v{singleResult.modelVersion}</p>
                  </div>
                </div>
                <div className="pt-4 border-t">
                  <p className="text-xs text-muted-foreground mb-2">Ligand SMILES</p>
                  <code className="text-xs font-mono bg-muted p-2 rounded block overflow-x-auto">
                    {singleResult.ligandSmiles}
                  </code>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="batch" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Batch Ligand Prediction</CardTitle>
              <CardDescription>Predict binding affinity for multiple ligands against the same target</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="batch-smiles">Ligand SMILES (one per line)</Label>
                <Textarea
                  id="batch-smiles"
                  data-testid="input-batch-smiles"
                  placeholder="CC(=O)Oc1ccccc1C(=O)O&#10;c1ccccc1&#10;CCO&#10;CCCC"
                  value={batchSmiles}
                  onChange={(e) => setBatchSmiles(e.target.value)}
                  className="font-mono text-sm min-h-[150px]"
                />
              </div>
              <div className="flex items-center gap-4">
                <div className="space-y-2">
                  <Label htmlFor="top-n">Top N Results</Label>
                  <Input
                    id="top-n"
                    data-testid="input-top-n"
                    type="number"
                    min="1"
                    max="50"
                    value={topN}
                    onChange={(e) => setTopN(e.target.value)}
                    className="w-24"
                  />
                </div>
                <Button 
                  onClick={handleBatchPredict} 
                  disabled={batchPredictMutation.isPending}
                  className="mt-6"
                  data-testid="button-predict-batch"
                >
                  {batchPredictMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      Run Batch Prediction
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {batchResult && (
            <Card className="border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  Batch Results
                </CardTitle>
                <CardDescription>
                  {batchResult.successfulCount} of {batchResult.totalCount} predictions successful
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold">{batchResult.totalCount}</p>
                    <p className="text-xs text-muted-foreground">Total</p>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold">{batchResult.averageAffinity.toFixed(0)}</p>
                    <p className="text-xs text-muted-foreground">Avg IC50 (nM)</p>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold">{batchResult.topBinders.filter(b => b.isStrongBinder).length}</p>
                    <p className="text-xs text-muted-foreground">Strong Binders</p>
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium">Top Binders</p>
                  <div className="space-y-2 max-h-[300px] overflow-y-auto">
                    {batchResult.topBinders.map((binder, idx) => (
                      <div 
                        key={idx} 
                        className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                        data-testid={`row-binder-${idx}`}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium w-6">#{idx + 1}</span>
                          <code className="text-xs font-mono truncate max-w-[200px]">{binder.ligandSmiles}</code>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className={`font-bold ${getAffinityColor(binder.predictedAffinity)}`}>
                            {binder.predictedAffinity.toFixed(2)} nM
                          </span>
                          {binder.isStrongBinder && (
                            <Badge variant="default" className="text-xs">Strong</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="screening" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Library Screening</CardTitle>
              <CardDescription>Screen a compound library against your target with hit classification</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="library-input">Compound Library (name,SMILES per line or just SMILES)</Label>
                <Textarea
                  id="library-input"
                  data-testid="input-library"
                  placeholder="Aspirin,CC(=O)Oc1ccccc1C(=O)O&#10;Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O&#10;c1ccccc1"
                  value={libraryInput}
                  onChange={(e) => setLibraryInput(e.target.value)}
                  className="font-mono text-sm min-h-[150px]"
                />
              </div>
              <div className="flex items-center gap-4">
                <div className="space-y-2">
                  <Label htmlFor="threshold">Hit Threshold (nM)</Label>
                  <Input
                    id="threshold"
                    data-testid="input-threshold"
                    type="number"
                    min="1"
                    max="10000"
                    value={threshold}
                    onChange={(e) => setThreshold(e.target.value)}
                    className="w-32"
                  />
                </div>
                <Button 
                  onClick={handleScreening} 
                  disabled={screeningMutation.isPending}
                  className="mt-6"
                  data-testid="button-screen-library"
                >
                  {screeningMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Screening...
                    </>
                  ) : (
                    <>
                      <Beaker className="h-4 w-4 mr-2" />
                      Screen Library
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {screeningResult && (
            <Card className="border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  Screening Results
                </CardTitle>
                <CardDescription>
                  Found {screeningResult.totalHits} hits from {screeningResult.totalScreened} compounds
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold">{screeningResult.totalScreened}</p>
                    <p className="text-xs text-muted-foreground">Screened</p>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold text-green-600">{screeningResult.totalHits}</p>
                    <p className="text-xs text-muted-foreground">Hits</p>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold">{screeningResult.hitRate.toFixed(1)}%</p>
                    <p className="text-xs text-muted-foreground">Hit Rate</p>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <p className="text-2xl font-bold">{screeningResult.thresholdNm}</p>
                    <p className="text-xs text-muted-foreground">Threshold (nM)</p>
                  </div>
                </div>

                {screeningResult.recommendations.length > 0 && (
                  <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                    <p className="text-sm font-medium mb-1">Recommendations</p>
                    <ul className="text-sm text-muted-foreground">
                      {screeningResult.recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-center gap-2">
                          <ChevronRight className="h-3 w-3" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="space-y-2">
                  <p className="text-sm font-medium">Top 10 Compounds</p>
                  <div className="space-y-2 max-h-[300px] overflow-y-auto">
                    {screeningResult.top10.map((compound, idx) => (
                      <div 
                        key={idx} 
                        className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                        data-testid={`row-compound-${idx}`}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium w-6">#{idx + 1}</span>
                          <div>
                            <p className="text-sm font-medium">{compound.name}</p>
                            <code className="text-xs font-mono text-muted-foreground truncate max-w-[200px] block">
                              {compound.smiles}
                            </code>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className={`font-bold ${getAffinityColor(compound.predictedIc50Nm)}`}>
                            {compound.predictedIc50Nm.toFixed(2)} nM
                          </span>
                          {compound.isHit ? (
                            <Badge variant="default" className="text-xs">Hit</Badge>
                          ) : (
                            <Badge variant="secondary" className="text-xs">Miss</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
