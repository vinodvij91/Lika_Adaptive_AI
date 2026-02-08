import { useQuery, useMutation } from "@tanstack/react-query";
import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
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
  Zap,
  BarChart3,
  TestTubes,
  ListChecks,
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

interface OptimizationSummary {
  totalOriginal: number;
  totalOptimized: number;
  totalDoseScenarios: number;
  improvements: {
    solubilityImproved: number;
    toxicityReduced: number;
    cnsImproved: number;
    metabolicImproved: number;
  };
  propertyDeltas: Array<{
    moleculeId: string;
    moleculeName: string;
    smiles: string;
    parentSmiles: string;
    modification: string;
    logPDelta: number | null;
    tpsaDelta: number | null;
    mwDelta: number | null;
    admetScore: number | null;
    oracleScore: number | null;
    hergRisk: string | null;
    bbbPenetration: string | null;
  }>;
  doseScenarios: Array<{
    moleculeId: string;
    moleculeName: string;
    smiles: string;
    modification: string;
    originalIndication: string;
    suggestedIndications: string[];
    doseReductionFactor: string;
  }>;
  seriesOptimizationMap: Record<string, { original: number; optimized: number }>;
  optimizedMoleculeIds: string[];
}

const CATEGORY_META: Record<string, { label: string; icon: typeof Droplets; color: string }> = {
  solubility: { label: "Solubility & Exposure", icon: Droplets, color: "text-blue-500" },
  permeability: { label: "Permeability / CNS", icon: BrainCircuit, color: "text-purple-500" },
  safety: { label: "Safety", icon: ShieldAlert, color: "text-destructive" },
  metabolic_stability: { label: "Metabolic Stability", icon: FlaskRound, color: "text-amber-500" },
  dose_indication: { label: "Dose & Indication", icon: Pill, color: "text-green-600" },
};

function OptimizationSummaryPanel({
  summary,
  onOpenMolecule,
}: {
  summary: OptimizationSummary;
  onOpenMolecule?: (moleculeId: string) => void;
}) {
  if (summary.totalOptimized === 0) return null;

  const totalImprovements =
    summary.improvements.solubilityImproved +
    summary.improvements.toxicityReduced +
    summary.improvements.cnsImproved +
    summary.improvements.metabolicImproved;

  return (
    <Card className="border-primary/20 bg-primary/5" data-testid="panel-optimization-summary">
      <CardContent className="p-4 space-y-4">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-primary/10 rounded-md shrink-0">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="font-semibold text-sm">Optimization Summary</h4>
            <p className="text-xs text-muted-foreground mt-0.5">
              Campaign-level overview of molecule optimization results
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="text-center p-2 bg-background rounded-md">
            <p className="text-2xl font-bold tabular-nums" data-testid="text-original-count">
              {summary.totalOriginal}
            </p>
            <p className="text-xs text-muted-foreground">Original</p>
          </div>
          <div className="text-center p-2 bg-background rounded-md">
            <p className="text-2xl font-bold tabular-nums text-primary" data-testid="text-optimized-count">
              {summary.totalOptimized}
            </p>
            <p className="text-xs text-muted-foreground">Optimized</p>
          </div>
          <div className="text-center p-2 bg-background rounded-md">
            <p className="text-2xl font-bold tabular-nums" data-testid="text-dose-scenarios-count">
              {summary.totalDoseScenarios}
            </p>
            <p className="text-xs text-muted-foreground">Dose Scenarios</p>
          </div>
          <div className="text-center p-2 bg-background rounded-md">
            <p className="text-2xl font-bold tabular-nums text-green-600" data-testid="text-improvements-count">
              {totalImprovements}
            </p>
            <p className="text-xs text-muted-foreground">Improvements</p>
          </div>
        </div>

        {totalImprovements > 0 && (
          <div className="flex flex-wrap gap-2">
            {summary.improvements.solubilityImproved > 0 && (
              <Badge variant="secondary" className="gap-1">
                <Droplets className="h-3 w-3" />
                Solubility +{summary.improvements.solubilityImproved}
              </Badge>
            )}
            {summary.improvements.toxicityReduced > 0 && (
              <Badge variant="secondary" className="gap-1">
                <ShieldAlert className="h-3 w-3" />
                Toxicity reduced {summary.improvements.toxicityReduced}
              </Badge>
            )}
            {summary.improvements.cnsImproved > 0 && (
              <Badge variant="secondary" className="gap-1">
                <BrainCircuit className="h-3 w-3" />
                CNS suitability +{summary.improvements.cnsImproved}
              </Badge>
            )}
            {summary.improvements.metabolicImproved > 0 && (
              <Badge variant="secondary" className="gap-1">
                <FlaskRound className="h-3 w-3" />
                Metabolic stability +{summary.improvements.metabolicImproved}
              </Badge>
            )}
          </div>
        )}

        {summary.propertyDeltas.length > 0 && (
          <div>
            <h5 className="text-xs font-medium text-muted-foreground mb-2">Optimized Analogs &mdash; Property Changes</h5>
            <ScrollArea className="max-h-[200px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs">Compound</TableHead>
                    <TableHead className="text-xs">Modification</TableHead>
                    <TableHead className="text-xs text-right">logP</TableHead>
                    <TableHead className="text-xs text-right">ADMET</TableHead>
                    <TableHead className="text-xs text-center">hERG</TableHead>
                    <TableHead className="text-xs text-center">BBB</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {summary.propertyDeltas.map((d) => (
                    <TableRow
                      key={d.moleculeId}
                      className={onOpenMolecule ? "cursor-pointer" : ""}
                      onClick={() => onOpenMolecule?.(d.moleculeId)}
                      data-testid={`row-optimized-${d.moleculeId}`}
                    >
                      <TableCell className="text-xs font-medium max-w-[120px] truncate" title={d.moleculeName}>
                        {d.moleculeName}
                      </TableCell>
                      <TableCell className="text-xs max-w-[150px] truncate" title={d.modification}>
                        {d.modification}
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {d.logPDelta != null ? (
                          <span className={d.logPDelta < 0 ? "text-green-600" : d.logPDelta > 0 ? "text-amber-500" : ""}>
                            {d.logPDelta > 0 ? "+" : ""}{d.logPDelta.toFixed(2)}
                          </span>
                        ) : "-"}
                      </TableCell>
                      <TableCell className="text-xs text-right tabular-nums">
                        {d.admetScore != null ? (
                          <span className={d.admetScore > 0.7 ? "text-green-600" : "text-amber-500"}>
                            {d.admetScore.toFixed(2)}
                          </span>
                        ) : "-"}
                      </TableCell>
                      <TableCell className="text-xs text-center">
                        {d.hergRisk ? (
                          <Badge variant={d.hergRisk === "Low" ? "secondary" : "destructive"} className="text-[10px]">
                            {d.hergRisk}
                          </Badge>
                        ) : "-"}
                      </TableCell>
                      <TableCell className="text-xs text-center">
                        {d.bbbPenetration ? (
                          <Badge variant={d.bbbPenetration === "Yes" ? "default" : "secondary"} className="text-[10px]">
                            {d.bbbPenetration}
                          </Badge>
                        ) : "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </div>
        )}

        {summary.doseScenarios.length > 0 && (
          <div>
            <h5 className="text-xs font-medium text-muted-foreground mb-2">Dose & Repurposing Scenarios</h5>
            <div className="space-y-2">
              {summary.doseScenarios.map((ds, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 bg-background rounded-md hover-elevate cursor-pointer"
                  onClick={() => onOpenMolecule?.(ds.moleculeId)}
                  data-testid={`row-dose-scenario-${idx}`}
                >
                  <Pill className="h-4 w-4 text-green-600 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium truncate">{ds.moleculeName}</p>
                    <p className="text-[10px] text-muted-foreground truncate">
                      {ds.originalIndication} &rarr; {ds.suggestedIndications.join(", ")}
                    </p>
                  </div>
                  <Badge variant="outline" className="text-[10px] shrink-0">
                    {ds.doseReductionFactor}
                  </Badge>
                  <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SeriesCard({
  series,
  optimizationInfo,
  onClick,
  selected,
  onSelectionChange,
}: {
  series: SarSeries;
  optimizationInfo?: { original: number; optimized: number };
  onClick: () => void;
  selected?: boolean;
  onSelectionChange?: (checked: boolean) => void;
}) {
  const displayName = series.seriesId || series.scaffoldId || "Ungrouped";
  const moleculeCount = series.molecules.length;
  const hasAssayData = series.assaySummary.count > 0;
  const hasOptimized = optimizationInfo && optimizationInfo.optimized > 0;

  return (
    <Card
      className="cursor-pointer hover-elevate active-elevate-2 transition-all"
      onClick={onClick}
      data-testid={`card-series-${displayName}`}
    >
      <CardContent className="p-4">
        {onSelectionChange !== undefined && (
          <div className="flex items-center gap-2 mb-2">
            <Checkbox
              checked={selected}
              onCheckedChange={(checked) => {
                onSelectionChange(!!checked);
              }}
              onClick={(e) => e.stopPropagation()}
              data-testid={`checkbox-series-${displayName}`}
            />
            <span className="text-xs text-muted-foreground">
              {moleculeCount} molecule{moleculeCount !== 1 ? "s" : ""}
            </span>
          </div>
        )}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center gap-2 min-w-0">
            <div className={`p-2 rounded-md shrink-0 ${hasOptimized ? "bg-primary/15" : "bg-primary/10"}`}>
              {hasOptimized ? (
                <Sparkles className="h-4 w-4 text-primary" />
              ) : (
                <Layers className="h-4 w-4 text-primary" />
              )}
            </div>
            <div className="min-w-0">
              <p className="font-medium truncate" title={displayName}>{displayName}</p>
              {hasOptimized ? (
                <p className="text-xs text-muted-foreground">
                  {optimizationInfo!.original} original / {optimizationInfo!.optimized} optimized
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">{moleculeCount} molecules</p>
              )}
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

        <div className="mt-3 flex items-center gap-1 flex-wrap">
          {hasAssayData && (
            <Badge variant="secondary" className="text-xs">
              <Beaker className="h-3 w-3 mr-1" />
              {series.assaySummary.count} results
            </Badge>
          )}
          {hasOptimized && (
            <Badge variant="default" className="text-xs">
              <Sparkles className="h-3 w-3 mr-1" />
              Optimized
            </Badge>
          )}
        </div>
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
              <CardTitle className="text-sm flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Core Properties
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div>
                  <p className="text-muted-foreground">MW</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.molecularWeight.toFixed(1)} Da</p>
                </div>
                <div>
                  <p className="text-muted-foreground">logP</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.logP.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">TPSA</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.tpsa.toFixed(1)} &#x212B;&#xB2;</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Rot. Bonds</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.rotatableBonds}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">HBD / HBA</p>
                  <p className="font-mono font-medium tabular-nums">{optimizationResult.properties.numHBondDonors} / {optimizationResult.properties.numHBondAcceptors}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Groups</p>
                  <div className="flex flex-wrap gap-1 mt-0.5">
                    {Object.entries(optimizationResult.properties.functionalGroups).slice(0, 4).map(([name, count]) => (
                      <Badge key={name} variant="outline" className="text-[10px]">{name}: {count}</Badge>
                    ))}
                    {Object.keys(optimizationResult.properties.functionalGroups).length === 0 && (
                      <span className="text-muted-foreground">None detected</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {Object.keys(groupedSuggestions).length > 0 && (
            <div className="space-y-3">
              <h5 className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                Optimization Suggestions
              </h5>
              {Object.entries(groupedSuggestions).map(([cat, suggestions]) => {
                const meta = CATEGORY_META[cat] || { label: cat, icon: Activity, color: "text-foreground" };
                const Icon = meta.icon;
                return (
                  <Card key={cat}>
                    <CardContent className="p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <Icon className={`h-4 w-4 ${meta.color}`} />
                        <span className="text-xs font-medium">{meta.label}</span>
                      </div>
                      <div className="space-y-2">
                        {suggestions.map((s, i) => (
                          <div key={i} className="flex items-start gap-2">
                            <Badge
                              variant={s.priority === "high" ? "destructive" : s.priority === "medium" ? "default" : "secondary"}
                              className="text-[10px] shrink-0 mt-0.5"
                            >
                              {s.priority}
                            </Badge>
                            <div>
                              <p className="text-xs font-medium">{s.title}</p>
                              <p className="text-[11px] text-muted-foreground">{s.description}</p>
                              {s.modification && (
                                <code className="text-[10px] bg-muted px-1 py-0.5 rounded mt-1 inline-block">{s.modification}</code>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}

          {optimizationResult.insertedAnalogs.length > 0 && (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <TestTubes className="h-4 w-4" />
                  Generated Analogs
                  <Badge variant="secondary" className="ml-auto">
                    {optimizationResult.insertedAnalogs.length} generated
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Name</TableHead>
                      <TableHead className="text-xs">Modification</TableHead>
                      <TableHead className="text-xs">SMILES</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {optimizationResult.insertedAnalogs.map((a) => (
                      <TableRow key={a.id}>
                        <TableCell className="text-xs font-medium">{a.name}</TableCell>
                        <TableCell className="text-xs">{a.modification}</TableCell>
                        <TableCell>
                          <code className="text-[10px] truncate block max-w-[150px]">{a.smiles}</code>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {optimizationResult.analogs.length > 0 && (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  ADMET Predictions
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <ScrollArea className="max-h-[250px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="text-xs">Analog</TableHead>
                        <TableHead className="text-xs text-right">Bioavail.</TableHead>
                        <TableHead className="text-xs text-center">Permeab.</TableHead>
                        <TableHead className="text-xs text-center">BBB</TableHead>
                        <TableHead className="text-xs text-right">Metab. Stab.</TableHead>
                        <TableHead className="text-xs text-right">t&#xBD;</TableHead>
                        <TableHead className="text-xs text-center">hERG</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {optimizationResult.analogs.map((a, i) => (
                        <TableRow key={i}>
                          <TableCell className="text-xs font-medium">{a.name}</TableCell>
                          <TableCell className="text-xs text-right tabular-nums">
                            <span className={a.admetPredictions.bioavailability > 0.7 ? "text-green-600" : "text-amber-500"}>
                              {a.admetPredictions.bioavailability.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-xs text-center">
                            <Badge variant={a.admetPredictions.caco2Permeability === "High" ? "default" : "secondary"} className="text-[10px]">
                              {a.admetPredictions.caco2Permeability}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs text-center">
                            <Badge variant={a.admetPredictions.bbbPenetration === "Yes" ? "default" : "secondary"} className="text-[10px]">
                              {a.admetPredictions.bbbPenetration}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs text-right tabular-nums">
                            <span className={a.admetPredictions.metabolicStability > 0.7 ? "text-green-600" : "text-amber-500"}>
                              {a.admetPredictions.metabolicStability.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-xs text-right tabular-nums">{a.admetPredictions.halfLife.toFixed(1)}h</TableCell>
                          <TableCell className="text-xs text-center">
                            <Badge variant={a.admetPredictions.hergInhibition === "Low" ? "secondary" : "destructive"} className="text-[10px]">
                              {a.admetPredictions.hergInhibition}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {doseResult && (
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm flex items-center gap-2">
              <Pill className="h-4 w-4 text-green-600" />
              Dose Scenarios & Repurposing
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4 space-y-3">
            {doseResult.doseScenarios.map((ds, i) => (
              <div key={i} className="p-3 bg-muted/50 rounded-md space-y-1.5">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">{ds.indication}</Badge>
                  <span className="text-xs font-medium">{ds.scenario}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Current: </span>
                    <span className="font-mono">{ds.currentDose}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Suggested: </span>
                    <span className="font-mono text-green-600">{ds.suggestedDose}</span>
                  </div>
                </div>
                <p className="text-[11px] text-muted-foreground">{ds.rationale}</p>
                {ds.safetyNote && (
                  <p className="text-[11px] text-destructive flex items-center gap-1">
                    <ShieldAlert className="h-3 w-3" />
                    {ds.safetyNote}
                  </p>
                )}
              </div>
            ))}
            {doseResult.repurposingHints.length > 0 && (
              <div>
                <p className="text-xs font-medium mb-1 flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" />
                  Repurposing Hints
                </p>
                <ul className="space-y-1">
                  {doseResult.repurposingHints.map((hint, i) => (
                    <li key={i} className="text-xs text-muted-foreground flex items-start gap-1.5">
                      <ArrowRight className="h-3 w-3 mt-0.5 shrink-0" />
                      {hint}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {!optimizationResult && !doseResult && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            <Sparkles className="h-8 w-8 mx-auto mb-3 opacity-50" />
            <p>
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
  onOpenChange,
  optimizedMoleculeIds,
}: {
  series: SarSeries | null;
  campaignId: string;
  diseaseContext: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  optimizedMoleculeIds: string[];
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

  const originalMols = series.molecules.filter(m => !optimizedMoleculeIds.includes(m.id));
  const optimizedMols = series.molecules.filter(m => optimizedMoleculeIds.includes(m.id));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Series: {displayName}
            {optimizedMols.length > 0 && (
              <Badge variant="default" className="text-xs ml-2">
                {optimizedMols.length} optimized
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex gap-4">
          <div className="w-1/4 border-r pr-4 flex flex-col">
            <h4 className="font-medium text-sm mb-2">Molecules ({series.molecules.length})</h4>
            <ScrollArea className="flex-1">
              <div className="space-y-1">
                {originalMols.length > 0 && optimizedMols.length > 0 && (
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider px-2 pt-1">Original</p>
                )}
                {originalMols.map(mol => (
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
                {optimizedMols.length > 0 && (
                  <>
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider px-2 pt-2">Optimized</p>
                    {optimizedMols.map(mol => (
                      <Button
                        key={mol.id}
                        variant={selectedMolecule?.id === mol.id ? "secondary" : "ghost"}
                        size="sm"
                        className="w-full justify-start text-left"
                        onClick={() => { setSelectedMolecule(mol); setActiveTab("details"); }}
                        data-testid={`button-select-molecule-${mol.id}`}
                      >
                        <Sparkles className="h-3 w-3 mr-2 shrink-0 text-primary" />
                        <span className="truncate">{mol.name || generateMoleculeName(mol.smiles, String(mol.id))}</span>
                      </Button>
                    ))}
                  </>
                )}
              </div>
            </ScrollArea>
          </div>

          <div className="flex-1 overflow-auto">
            {selectedMolecule ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div>
                    <h4 className="font-medium mb-1">
                      {selectedMolecule.name || generateMoleculeName(selectedMolecule.smiles, String(selectedMolecule.id))}
                    </h4>
                    <code className="text-xs bg-muted px-2 py-1 rounded block truncate">
                      {selectedMolecule.smiles}
                    </code>
                  </div>
                  {optimizedMoleculeIds.includes(selectedMolecule.id) && (
                    <Badge variant="default" className="shrink-0">Optimized</Badge>
                  )}
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
  const [selectedSeriesKeys, setSelectedSeriesKeys] = useState<Set<string>>(new Set());
  const [bulkMode, setBulkMode] = useState(false);
  const { toast } = useToast();

  const { data: sarSeries, isLoading } = useQuery<SarSeries[]>({
    queryKey: ["/api/campaigns", campaignId, "sar", "series"],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/series`);
      if (!res.ok) throw new Error("Failed to fetch SAR series");
      return res.json();
    },
  });

  const { data: optimizationSummary } = useQuery<OptimizationSummary>({
    queryKey: ["/api/campaigns", campaignId, "sar", "optimization-summary"],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/optimization-summary`);
      if (!res.ok) throw new Error("Failed to fetch optimization summary");
      return res.json();
    },
    enabled: !!sarSeries && sarSeries.length > 0,
  });

  const selectedMoleculeIds = useMemo(() => {
    if (!sarSeries || selectedSeriesKeys.size === 0) return [];
    return sarSeries
      .filter(s => selectedSeriesKeys.has(s.seriesId || s.scaffoldId || "ungrouped"))
      .flatMap(s => s.molecules.map(m => m.id));
  }, [sarSeries, selectedSeriesKeys]);

  const allSelected = useMemo(() => {
    if (!sarSeries || sarSeries.length === 0) return false;
    return sarSeries.every(s => selectedSeriesKeys.has(s.seriesId || s.scaffoldId || "ungrouped"));
  }, [sarSeries, selectedSeriesKeys]);

  const toggleSelectAll = useCallback(() => {
    if (!sarSeries) return;
    if (allSelected) {
      setSelectedSeriesKeys(new Set());
    } else {
      setSelectedSeriesKeys(new Set(sarSeries.map(s => s.seriesId || s.scaffoldId || "ungrouped")));
    }
  }, [sarSeries, allSelected]);

  const toggleSeriesSelection = useCallback((seriesKey: string, checked: boolean) => {
    setSelectedSeriesKeys(prev => {
      const next = new Set(prev);
      if (checked) { next.add(seriesKey); } else { next.delete(seriesKey); }
      return next;
    });
  }, []);

  const invalidateSarQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["/api/campaigns", campaignId, "sar"] });
    queryClient.invalidateQueries({ queryKey: ["/api/campaigns", campaignId] });
  }, [campaignId]);

  const bulkPropertiesMutation = useMutation({
    mutationFn: async () => {
      const idsToOptimize = selectedMoleculeIds.length > 0 ? selectedMoleculeIds : undefined;
      const res = await apiRequest("POST", `/api/campaigns/${campaignId}/sar/bulk-optimize-properties`, {
        moleculeIds: idsToOptimize,
        diseaseContext,
      });
      return res.json() as Promise<{
        totalProcessed: number;
        totalSkipped: number;
        totalAnalogsInserted: number;
        results: Array<{ moleculeId: string; analogsCreated: number }>;
      }>;
    },
    onSuccess: (data) => {
      invalidateSarQueries();
      setBulkMode(false);
      setSelectedSeriesKeys(new Set());
      toast({
        title: "Bulk Property Optimization Complete",
        description: `Processed ${data.totalProcessed} molecules, created ${data.totalAnalogsInserted} optimized analogs${data.totalSkipped > 0 ? ` (${data.totalSkipped} already optimized)` : ""}`,
      });
    },
    onError: () => {
      toast({ title: "Bulk Optimization Failed", description: "Could not complete bulk property optimization", variant: "destructive" });
    },
  });

  const bulkDoseMutation = useMutation({
    mutationFn: async () => {
      const idsToOptimize = selectedMoleculeIds.length > 0 ? selectedMoleculeIds : undefined;
      const res = await apiRequest("POST", `/api/campaigns/${campaignId}/sar/bulk-optimize-dose`, {
        moleculeIds: idsToOptimize,
        diseaseContext,
      });
      return res.json() as Promise<{
        totalProcessed: number;
        totalDoseScenarios: number;
        results: Array<{ moleculeId: string; doseScenarios: number }>;
      }>;
    },
    onSuccess: (data) => {
      invalidateSarQueries();
      setBulkMode(false);
      setSelectedSeriesKeys(new Set());
      toast({
        title: "Bulk Dose Optimization Complete",
        description: `Processed ${data.totalProcessed} molecules, generated ${data.totalDoseScenarios} dose scenarios`,
      });
    },
    onError: () => {
      toast({ title: "Bulk Optimization Failed", description: "Could not complete bulk dose optimization", variant: "destructive" });
    },
  });

  const isBulkRunning = bulkPropertiesMutation.isPending || bulkDoseMutation.isPending;

  const handleSeriesClick = (series: SarSeries) => {
    setSelectedSeries(series);
    setDialogOpen(true);
  };

  const handleOpenMolecule = (moleculeId: string) => {
    if (!sarSeries) return;
    for (const series of sarSeries) {
      const mol = series.molecules.find(m => m.id === moleculeId);
      if (mol) {
        setSelectedSeries(series);
        setDialogOpen(true);
        return;
      }
    }
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
  const seriesOptMap = optimizationSummary?.seriesOptimizationMap || {};
  const optimizedMoleculeIds = optimizationSummary?.optimizedMoleculeIds || [];
  const selectedCount = selectedMoleculeIds.length;

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
            {optimizationSummary && optimizationSummary.totalOptimized > 0 && (
              <span className="text-primary"> | {optimizationSummary.totalOptimized} optimized</span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={bulkMode ? "default" : "outline"}
            onClick={() => {
              setBulkMode(!bulkMode);
              if (bulkMode) setSelectedSeriesKeys(new Set());
            }}
            disabled={isBulkRunning}
            data-testid="button-toggle-bulk-mode"
          >
            <ListChecks className="h-4 w-4 mr-1" />
            Bulk Optimize
          </Button>
        </div>
      </div>

      {bulkMode && (
        <Card className="border-primary/30 bg-primary/5" data-testid="panel-bulk-optimization">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-3">
                <Checkbox
                  checked={allSelected}
                  onCheckedChange={() => toggleSelectAll()}
                  disabled={isBulkRunning}
                  data-testid="checkbox-select-all"
                />
                <span className="text-sm font-medium">
                  {selectedCount > 0
                    ? `${selectedCount} of ${totalMolecules} molecules selected (${selectedSeriesKeys.size} series)`
                    : "Select series below, or click a button to optimize all"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  onClick={() => bulkPropertiesMutation.mutate()}
                  disabled={isBulkRunning}
                  data-testid="button-bulk-optimize-properties"
                >
                  {bulkPropertiesMutation.isPending ? (
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                  ) : (
                    <Sparkles className="h-3 w-3 mr-1" />
                  )}
                  Optimize Properties{selectedCount > 0 ? ` (${selectedCount})` : " (All)"}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => bulkDoseMutation.mutate()}
                  disabled={isBulkRunning}
                  data-testid="button-bulk-optimize-dose"
                >
                  {bulkDoseMutation.isPending ? (
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                  ) : (
                    <Pill className="h-3 w-3 mr-1" />
                  )}
                  Optimize Dose & Indication{selectedCount > 0 ? ` (${selectedCount})` : " (All)"}
                </Button>
              </div>
            </div>
            {isBulkRunning && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  {bulkPropertiesMutation.isPending ? "Running property optimization across molecules..." : "Running dose/indication optimization across molecules..."}
                </p>
                <Progress value={undefined} className="h-1.5" />
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {optimizationSummary && optimizationSummary.totalOptimized > 0 && (
        <OptimizationSummaryPanel
          summary={optimizationSummary}
          onOpenMolecule={handleOpenMolecule}
        />
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sarSeries.map((series, idx) => {
          const seriesKey = series.seriesId || series.scaffoldId || "ungrouped";
          const optInfo = seriesOptMap[seriesKey];
          return (
            <SeriesCard
              key={series.seriesId || series.scaffoldId || idx}
              series={series}
              optimizationInfo={optInfo}
              onClick={() => handleSeriesClick(series)}
              selected={bulkMode ? selectedSeriesKeys.has(seriesKey) : undefined}
              onSelectionChange={bulkMode ? (checked) => toggleSeriesSelection(seriesKey, checked) : undefined}
            />
          );
        })}
      </div>

      <SeriesDetailDialog
        series={selectedSeries}
        campaignId={campaignId}
        diseaseContext={diseaseContext}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        optimizedMoleculeIds={optimizedMoleculeIds}
      />
    </div>
  );
}
