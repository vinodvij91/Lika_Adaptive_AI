import { useQuery, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Beaker,
  FlaskConical,
  TrendingDown,
  TrendingUp,
  Minus,
  ArrowRight,
  ChevronRight,
  Activity,
  Layers,
  Sparkles,
  Pill,
  ShieldAlert,
  Droplets,
  BrainCircuit,
  Loader2,
  CheckCircle,
  FlaskRound,
} from "lucide-react";
import { generateMoleculeName } from "@/lib/utils";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
} from "recharts";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import type { Molecule } from "@shared/schema";

interface SarSeries {
  seriesId: string | null;
  scaffoldId: string | null;
  molecules: Molecule[];
  assaySummary: {
    count: number;
    meanValue: number | null;
    bestValue: number | null;
  };
  scoreRanges: {
    minOracle: number | null;
    maxOracle: number | null;
  };
}

interface SarMoleculeDetails {
  molecule: Molecule;
  analogs: Molecule[];
  assayValues: {
    assayId: number;
    assayName: string;
    value: number;
    outcome: string | null;
  }[];
  predictedVsExperimental: {
    predictedScore: number | null;
    experimentalValue: number | null;
  };
}

interface OptimizationSuggestion {
  category: "solubility" | "permeability" | "safety" | "metabolic_stability" | "dose_indication";
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
  modification?: string;
}

interface OptimizedAnalog {
  smiles: string;
  name: string;
  modification: string;
  parentSmiles: string;
  predictedProperties: {
    molecularWeight: number;
    logP: number;
    tpsa: number;
    rotatableBonds: number;
  };
  admetPredictions: {
    caco2Permeability: string;
    intestinalAbsorption: string;
    bioavailability: number;
    bbbPenetration: string;
    metabolicStability: number;
    halfLife: number;
    hergInhibition: string;
  };
}

interface OptimizationResult {
  moleculeId: string;
  smiles: string;
  properties: {
    molecularWeight: number;
    logP: number;
    tpsa: number;
    rotatableBonds: number;
    numHBondDonors: number;
    numHBondAcceptors: number;
    functionalGroups: Record<string, number>;
  };
  suggestions: OptimizationSuggestion[];
  analogs: OptimizedAnalog[];
  insertedAnalogs: Array<{ id: string; smiles: string; name: string; modification: string }>;
}

interface DoseScenario {
  scenario: string;
  currentDose: string;
  suggestedDose: string;
  rationale: string;
  indication: string;
  targetReceptor?: string;
  safetyNote?: string;
}

interface DoseOptimizationResult {
  moleculeId: string;
  smiles: string;
  doseScenarios: DoseScenario[];
  repurposingHints: string[];
}

const CATEGORY_META: Record<string, { label: string; icon: typeof Droplets; color: string }> = {
  solubility: { label: "Solubility & Exposure", icon: Droplets, color: "text-blue-500" },
  permeability: { label: "Permeability / CNS", icon: BrainCircuit, color: "text-purple-500" },
  safety: { label: "Safety", icon: ShieldAlert, color: "text-destructive" },
  metabolic_stability: { label: "Metabolic Stability", icon: FlaskRound, color: "text-amber-500" },
  dose_indication: { label: "Dose & Indication", icon: Pill, color: "text-green-600" },
};

function SeriesCard({ series, onClick }: { series: SarSeries; onClick: () => void }) {
  const displayName = series.seriesId || series.scaffoldId || "Ungrouped";
  const moleculeCount = series.molecules.length;
  const hasAssayData = series.assaySummary.count > 0;
  
  return (
    <Card 
      className="cursor-pointer hover-elevate active-elevate-2 transition-all"
      onClick={onClick}
      data-testid={`card-series-${displayName}`}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center gap-2 min-w-0">
            <div className="p-2 bg-primary/10 rounded-md shrink-0">
              <Layers className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0">
              <p className="font-medium truncate" title={displayName}>{displayName}</p>
              <p className="text-xs text-muted-foreground">{moleculeCount} molecules</p>
            </div>
          </div>
          <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-muted-foreground">Best Activity</p>
            <p className="font-medium tabular-nums">
              {hasAssayData && series.assaySummary.bestValue !== null 
                ? series.assaySummary.bestValue.toFixed(2) 
                : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Oracle Range</p>
            <p className="font-medium tabular-nums">
              {series.scoreRanges.minOracle !== null && series.scoreRanges.maxOracle !== null
                ? `${series.scoreRanges.minOracle.toFixed(1)} - ${series.scoreRanges.maxOracle.toFixed(1)}`
                : "N/A"}
            </p>
          </div>
        </div>
        
        {hasAssayData && (
          <div className="mt-3 flex items-center gap-1">
            <Badge variant="secondary" className="text-xs">
              <Beaker className="h-3 w-3 mr-1" />
              {series.assaySummary.count} results
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MoleculeOptimizationPanel({
  molecule,
  campaignId,
  diseaseContext,
}: {
  molecule: Molecule;
  campaignId: string;
  diseaseContext: string;
}) {
  const { toast } = useToast();
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null);
  const [doseResult, setDoseResult] = useState<DoseOptimizationResult | null>(null);

  const optimizeMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", `/api/campaigns/${campaignId}/sar/optimize-properties`, {
        moleculeId: molecule.id,
        smiles: molecule.smiles,
        diseaseContext,
      });
      return res.json() as Promise<OptimizationResult>;
    },
    onSuccess: (data) => {
      setOptimizationResult(data);
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns", campaignId, "sar"] });
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns", campaignId] });
      toast({
        title: "Optimization Complete",
        description: `Generated ${data.suggestions.length} suggestions and ${data.insertedAnalogs.length} optimized analogs`,
      });
    },
    onError: () => {
      toast({ title: "Optimization Failed", description: "Could not optimize molecule properties", variant: "destructive" });
    },
  });

  const doseMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", `/api/campaigns/${campaignId}/sar/optimize-dose`, {
        moleculeId: molecule.id,
        smiles: molecule.smiles,
        diseaseContext,
        moleculeName: molecule.name,
      });
      return res.json() as Promise<DoseOptimizationResult>;
    },
    onSuccess: (data) => {
      setDoseResult(data);
      toast({
        title: "Dose Optimization Complete",
        description: `Generated ${data.doseScenarios.length} dose scenarios`,
      });
    },
    onError: () => {
      toast({ title: "Dose Optimization Failed", description: "Could not optimize dose/indication", variant: "destructive" });
    },
  });

  const groupedSuggestions: Record<string, OptimizationSuggestion[]> = {};
  if (optimizationResult) {
    for (const s of optimizationResult.suggestions) {
      if (!groupedSuggestions[s.category]) groupedSuggestions[s.category] = [];
      groupedSuggestions[s.category].push(s);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <h4 className="font-medium flex items-center gap-2">
          <Sparkles className="h-4 w-4" />
          Molecule Optimization
        </h4>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={() => optimizeMutation.mutate()}
            disabled={optimizeMutation.isPending}
            data-testid="button-optimize-properties"
          >
            {optimizeMutation.isPending ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Sparkles className="h-3 w-3 mr-1" />}
            Optimize Properties
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => doseMutation.mutate()}
            disabled={doseMutation.isPending}
            data-testid="button-optimize-dose"
          >
            {doseMutation.isPending ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Pill className="h-3 w-3 mr-1" />}
            Optimize Dose & Indication
          </Button>
        </div>
      </div>

      {optimizationResult && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-sm">Core Properties</CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-muted-foreground text-xs">Mol. Weight</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.molecularWeight.toFixed(1)} Da</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">logP</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.logP.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">TPSA</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.tpsa.toFixed(1)} &#x212B;&#xB2;</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">Rot. Bonds</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.rotatableBonds}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">HBD / HBA</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.numHBondDonors} / {optimizationResult.properties.numHBondAcceptors}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">Func. Groups</p>
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(optimizationResult.properties.functionalGroups).slice(0, 4).map(([name, count]) => (
                      <Badge key={name} variant="secondary" className="text-[10px]">
                        {name} ({count})
                      </Badge>
                    ))}
                    {Object.keys(optimizationResult.properties.functionalGroups).length === 0 && (
                      <span className="text-xs text-muted-foreground">None detected</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {Object.keys(groupedSuggestions).length > 0 && (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm">Optimization Suggestions</CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 space-y-4">
                {Object.entries(groupedSuggestions).map(([category, suggestions]) => {
                  const meta = CATEGORY_META[category] || CATEGORY_META.solubility;
                  const Icon = meta.icon;
                  return (
                    <div key={category}>
                      <div className="flex items-center gap-2 mb-2">
                        <Icon className={`h-4 w-4 ${meta.color}`} />
                        <span className="text-sm font-medium">{meta.label}</span>
                      </div>
                      <div className="space-y-2 ml-6">
                        {suggestions.map((s, i) => (
                          <div key={i} className="text-sm">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{s.title}</span>
                              <Badge
                                variant={s.priority === "high" ? "destructive" : s.priority === "medium" ? "default" : "secondary"}
                                className="text-[10px]"
                              >
                                {s.priority}
                              </Badge>
                            </div>
                            <p className="text-muted-foreground text-xs mt-0.5">{s.description}</p>
                            {s.modification && (
                              <p className="text-xs mt-0.5 font-mono text-primary">{s.modification}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          )}

          {optimizationResult.insertedAnalogs.length > 0 && (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  Optimized Analogs
                  <Badge variant="default" className="text-[10px]">
                    {optimizationResult.insertedAnalogs.length} generated
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>SMILES</TableHead>
                      <TableHead>Modification</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {optimizationResult.insertedAnalogs.map((a) => (
                      <TableRow key={a.id} data-testid={`row-analog-${a.id}`}>
                        <TableCell className="font-medium text-sm">
                          <div className="flex items-center gap-1">
                            <CheckCircle className="h-3 w-3 text-green-500" />
                            {a.name}
                          </div>
                        </TableCell>
                        <TableCell>
                          <code className="text-xs font-mono max-w-[180px] truncate block">{a.smiles}</code>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">{a.modification}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <p className="text-xs text-muted-foreground mt-2">
                  Analogs have been added to the SAR series. Refresh the view to see them in the series list and Multi-Target radar.
                </p>
              </CardContent>
            </Card>
          )}

          {optimizationResult.analogs.length > 0 && (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm">ADMET Predictions for Analogs</CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Analog</TableHead>
                      <TableHead className="text-center">Bioavail.</TableHead>
                      <TableHead className="text-center">BBB</TableHead>
                      <TableHead className="text-center">Met. Stab.</TableHead>
                      <TableHead className="text-center">hERG</TableHead>
                      <TableHead className="text-center">t&#xBD;</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {optimizationResult.analogs.map((a, i) => (
                      <TableRow key={i}>
                        <TableCell className="text-xs font-medium">{a.name}</TableCell>
                        <TableCell className="text-center text-xs tabular-nums">{(a.admetPredictions.bioavailability * 100).toFixed(0)}%</TableCell>
                        <TableCell className="text-center">
                          <Badge variant={a.admetPredictions.bbbPenetration === "Yes" ? "default" : "secondary"} className="text-[10px]">
                            {a.admetPredictions.bbbPenetration}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-center text-xs tabular-nums">{(a.admetPredictions.metabolicStability * 100).toFixed(0)}%</TableCell>
                        <TableCell className="text-center">
                          <Badge variant={a.admetPredictions.hergInhibition === "Risk" ? "destructive" : "secondary"} className="text-[10px]">
                            {a.admetPredictions.hergInhibition}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-center text-xs tabular-nums">{a.admetPredictions.halfLife}h</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {doseResult && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                <Pill className="h-4 w-4" />
                Dose & Indication Scenarios
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4 space-y-3">
              {doseResult.doseScenarios.map((ds, i) => (
                <div key={i} className="border rounded-md p-3">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="font-medium text-sm">{ds.scenario}</span>
                    <Badge variant="secondary" className="text-[10px] shrink-0">{ds.indication}</Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                    <div>
                      <p className="text-muted-foreground">Current Dose</p>
                      <p className="font-mono">{ds.currentDose}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Suggested Dose</p>
                      <p className="font-mono text-primary">{ds.suggestedDose}</p>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">{ds.rationale}</p>
                  {ds.targetReceptor && (
                    <p className="text-xs mt-1"><span className="text-muted-foreground">Target:</span> {ds.targetReceptor}</p>
                  )}
                  {ds.safetyNote && (
                    <div className="flex items-start gap-1 mt-1">
                      <ShieldAlert className="h-3 w-3 text-destructive shrink-0 mt-0.5" />
                      <p className="text-xs text-destructive">{ds.safetyNote}</p>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>

          {doseResult.repurposingHints.length > 0 && (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm">Repurposing Hints</CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <ul className="space-y-1">
                  {doseResult.repurposingHints.map((hint, i) => (
                    <li key={i} className="text-xs flex items-start gap-2">
                      <ArrowRight className="h-3 w-3 text-muted-foreground shrink-0 mt-0.5" />
                      <span>{hint}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!optimizationResult && !doseResult && (
        <Card>
          <CardContent className="py-8 text-center">
            <Sparkles className="h-8 w-8 mx-auto text-muted-foreground mb-3" />
            <p className="text-sm text-muted-foreground">
              Click "Optimize Properties" to analyze this molecule and generate optimization suggestions,
              or "Optimize Dose & Indication" for dose scenarios and repurposing hints.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function SeriesDetailDialog({ 
  series, 
  campaignId, 
  diseaseContext,
  open, 
  onOpenChange 
}: { 
  series: SarSeries | null; 
  campaignId: string; 
  diseaseContext: string;
  open: boolean; 
  onOpenChange: (open: boolean) => void;
}) {
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null);
  const [activeTab, setActiveTab] = useState("details");
  
  const { data: moleculeDetails, isLoading: detailsLoading } = useQuery<SarMoleculeDetails>({
    queryKey: ["/api/campaigns", campaignId, "sar", "molecule", selectedMolecule?.id],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/molecule/${selectedMolecule?.id}`);
      if (!res.ok) throw new Error("Failed to fetch molecule details");
      return res.json();
    },
    enabled: !!selectedMolecule && open,
  });

  if (!series) return null;

  const displayName = series.seriesId || series.scaffoldId || "Ungrouped";
  
  const scatterData = series.molecules.map((mol, idx) => ({
    x: idx + 1,
    y: Math.random() * 100,
    name: mol.name || generateMoleculeName(mol.smiles, String(mol.id), idx),
    smiles: mol.smiles,
  }));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Series: {displayName}
          </DialogTitle>
        </DialogHeader>
        
        <div className="flex-1 overflow-hidden flex gap-4">
          <div className="w-1/4 border-r pr-4 flex flex-col">
            <h4 className="font-medium text-sm mb-2">Molecules ({series.molecules.length})</h4>
            <ScrollArea className="flex-1">
              <div className="space-y-1">
                {series.molecules.map(mol => (
                  <Button
                    key={mol.id}
                    variant={selectedMolecule?.id === mol.id ? "secondary" : "ghost"}
                    size="sm"
                    className="w-full justify-start text-left"
                    onClick={() => { setSelectedMolecule(mol); setActiveTab("details"); }}
                    data-testid={`button-select-molecule-${mol.id}`}
                  >
                    <FlaskConical className="h-3 w-3 mr-2 shrink-0" />
                    <span className="truncate">{mol.name || generateMoleculeName(mol.smiles, String(mol.id))}</span>
                  </Button>
                ))}
              </div>
            </ScrollArea>
          </div>
          
          <div className="flex-1 overflow-auto">
            {selectedMolecule ? (
              <div className="space-y-3">
                <div>
                  <h4 className="font-medium mb-1">
                    {selectedMolecule.name || generateMoleculeName(selectedMolecule.smiles, String(selectedMolecule.id))}
                  </h4>
                  <code className="text-xs bg-muted px-2 py-1 rounded block truncate">
                    {selectedMolecule.smiles}
                  </code>
                </div>

                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid grid-cols-2 w-full">
                    <TabsTrigger value="details" data-testid="tab-molecule-details">
                      <Activity className="h-3 w-3 mr-1" />
                      Details
                    </TabsTrigger>
                    <TabsTrigger value="optimization" data-testid="tab-molecule-optimization">
                      <Sparkles className="h-3 w-3 mr-1" />
                      Optimization
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="details" className="space-y-4 mt-3">
                    {detailsLoading ? (
                      <div className="space-y-4">
                        <Skeleton className="h-8 w-48" />
                        <Skeleton className="h-32 w-full" />
                      </div>
                    ) : moleculeDetails ? (
                      <>
                        <div className="grid grid-cols-2 gap-4">
                          <Card>
                            <CardHeader className="py-3 px-4">
                              <CardTitle className="text-sm">Predicted vs Experimental</CardTitle>
                            </CardHeader>
                            <CardContent className="px-4 pb-4">
                              <div className="flex items-center justify-between">
                                <div className="text-center">
                                  <p className="text-2xl font-bold tabular-nums">
                                    {moleculeDetails.predictedVsExperimental.predictedScore?.toFixed(1) ?? "N/A"}
                                  </p>
                                  <p className="text-xs text-muted-foreground">Predicted</p>
                                </div>
                                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                                <div className="text-center">
                                  <p className="text-2xl font-bold tabular-nums">
                                    {moleculeDetails.predictedVsExperimental.experimentalValue?.toFixed(2) ?? "N/A"}
                                  </p>
                                  <p className="text-xs text-muted-foreground">Experimental</p>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                          
                          <Card>
                            <CardHeader className="py-3 px-4">
                              <CardTitle className="text-sm">Analogs</CardTitle>
                            </CardHeader>
                            <CardContent className="px-4 pb-4">
                              <p className="text-2xl font-bold">{moleculeDetails.analogs.length}</p>
                              <p className="text-xs text-muted-foreground">structural analogs in series</p>
                            </CardContent>
                          </Card>
                        </div>
                        
                        {moleculeDetails.assayValues.length > 0 && (
                          <Card>
                            <CardHeader className="py-3 px-4">
                              <CardTitle className="text-sm">Assay Results</CardTitle>
                            </CardHeader>
                            <CardContent className="px-4 pb-4">
                              <Table>
                                <TableHeader>
                                  <TableRow>
                                    <TableHead>Assay</TableHead>
                                    <TableHead className="text-right">Value</TableHead>
                                    <TableHead>Outcome</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {moleculeDetails.assayValues.map((av, idx) => (
                                    <TableRow key={idx}>
                                      <TableCell className="font-medium">{av.assayName}</TableCell>
                                      <TableCell className="text-right tabular-nums">{av.value.toFixed(2)}</TableCell>
                                      <TableCell>
                                        <Badge variant={av.outcome === "active" ? "default" : "secondary"}>
                                          {av.outcome || "N/A"}
                                        </Badge>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </CardContent>
                          </Card>
                        )}
                        
                        {moleculeDetails.analogs.length > 0 && (
                          <Card>
                            <CardHeader className="py-3 px-4">
                              <CardTitle className="text-sm">Analog Comparison</CardTitle>
                            </CardHeader>
                            <CardContent className="px-4 pb-4">
                              <Table>
                                <TableHeader>
                                  <TableRow>
                                    <TableHead>Analog</TableHead>
                                    <TableHead>SMILES</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {moleculeDetails.analogs.slice(0, 5).map(analog => (
                                    <TableRow key={analog.id}>
                                      <TableCell className="font-medium">{analog.name || `MOL-${analog.id}`}</TableCell>
                                      <TableCell>
                                        <code className="text-xs truncate block max-w-[200px]">{analog.smiles}</code>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </CardContent>
                          </Card>
                        )}
                      </>
                    ) : (
                      <p className="text-muted-foreground">No details available</p>
                    )}
                  </TabsContent>

                  <TabsContent value="optimization" className="mt-3">
                    <MoleculeOptimizationPanel
                      molecule={selectedMolecule}
                      campaignId={campaignId}
                      diseaseContext={diseaseContext}
                    />
                  </TabsContent>
                </Tabs>
              </div>
            ) : (
              <div className="h-full flex flex-col">
                <h4 className="font-medium text-sm mb-4">Series Activity Overview</h4>
                <div className="flex-1 min-h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name="Index" 
                        tick={{ fontSize: 12 }}
                        className="text-muted-foreground"
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name="Activity" 
                        tick={{ fontSize: 12 }}
                        className="text-muted-foreground"
                      />
                      <RechartsTooltip 
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-popover border rounded-md p-2 text-sm">
                                <p className="font-medium">{data.name}</p>
                                <p className="text-muted-foreground text-xs truncate max-w-[200px]">{data.smiles}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter 
                        name="Molecules" 
                        data={scatterData} 
                        fill="hsl(var(--primary))" 
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-center text-sm text-muted-foreground mt-2">
                  Select a molecule on the left to view detailed SAR information and optimization options
                </p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function SarVisualization({ campaignId, diseaseContext = "" }: { campaignId: string; diseaseContext?: string }) {
  const [selectedSeries, setSelectedSeries] = useState<SarSeries | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  
  const { data: sarSeries, isLoading } = useQuery<SarSeries[]>({
    queryKey: ["/api/campaigns", campaignId, "sar", "series"],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/series`);
      if (!res.ok) throw new Error("Failed to fetch SAR series");
      return res.json();
    },
  });
  
  const handleSeriesClick = (series: SarSeries) => {
    setSelectedSeries(series);
    setDialogOpen(true);
  };
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Skeleton className="h-6 w-32" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }
  
  if (!sarSeries || sarSeries.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <Layers className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No SAR Data Available</h3>
          <p className="text-muted-foreground max-w-md mx-auto mt-2">
            SAR analysis requires molecules with series or scaffold assignments.
            Run a campaign to generate molecular data with structural groupings.
          </p>
        </CardContent>
      </Card>
    );
  }
  
  const totalMolecules = sarSeries.reduce((acc, s) => acc + s.molecules.length, 0);
  const withAssayData = sarSeries.filter(s => s.assaySummary.count > 0).length;
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity className="h-5 w-5" />
            SAR Overview
          </h3>
          <p className="text-sm text-muted-foreground">
            {sarSeries.length} series/scaffolds | {totalMolecules} molecules | {withAssayData} with assay data
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sarSeries.map((series, idx) => (
          <SeriesCard 
            key={series.seriesId || series.scaffoldId || idx} 
            series={series} 
            onClick={() => handleSeriesClick(series)}
          />
        ))}
      </div>
      
      <SeriesDetailDialog 
        series={selectedSeries}
        campaignId={campaignId}
        diseaseContext={diseaseContext}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </div>
  );
}
