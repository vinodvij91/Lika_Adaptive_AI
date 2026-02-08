import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { ResultsPanel } from "@/components/results-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  Factory,
  Gauge,
  AlertTriangle,
  FlaskConical,
  Rocket,
  Building2,
  TrendingUp,
  BarChart3,
  Settings,
  Thermometer,
  Layers,
  DollarSign,
  RefreshCw,
  Download,
  Zap,
  Loader2,
} from "lucide-react";
import type { MaterialsOracleScore, MaterialEntity } from "@shared/schema";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

type ReadinessTier = "lab-only" | "pilot-ready" | "production-viable";

function getTier(score: number): ReadinessTier {
  if (score >= 0.7) return "production-viable";
  if (score >= 0.4) return "pilot-ready";
  return "lab-only";
}

const TIER_CONFIG: Record<ReadinessTier, { label: string; icon: typeof FlaskConical; color: string }> = {
  "lab-only": { label: "Lab-Only", icon: FlaskConical, color: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30" },
  "pilot-ready": { label: "Pilot-Ready", icon: Rocket, color: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30" },
  "production-viable": { label: "Production-Viable", icon: Building2, color: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30" },
};

interface ScoreGaugeProps {
  label: string;
  value: number;
  icon: typeof Gauge;
  inverted?: boolean;
}

function ScoreGauge({ label, value, icon: Icon, inverted = false }: ScoreGaugeProps) {
  const percent = Math.round(value * 100);
  const displayValue = inverted ? 100 - percent : percent;
  const color = displayValue >= 70 ? "text-green-600 dark:text-green-400" : displayValue >= 40 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400";

  return (
    <div className="text-center">
      <div className="w-10 h-10 mx-auto rounded-full bg-muted/50 flex items-center justify-center mb-1">
        <Icon className={`h-5 w-5 ${color}`} />
      </div>
      <div className={`text-lg font-bold font-mono ${color}`}>{percent}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

function OracleScoreCard({ score, materialName }: { score: MaterialsOracleScore; materialName?: string }) {
  const overall = score.oracleScore || 0;
  const synthesis = score.synthesisFeasibility || 0;
  const costFactor = score.manufacturingCostFactor || 0;
  const tier = getTier(synthesis);
  const tierConfig = TIER_CONFIG[tier];
  const TierIcon = tierConfig.icon;
  const overallPercent = Math.round(overall * 100);
  const overallColor = overallPercent >= 70 ? "text-green-600 dark:text-green-400" : overallPercent >= 40 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400";
  const breakdown = (score.propertyBreakdown || {}) as Record<string, number>;

  return (
    <Card className="hover-elevate" data-testid={`card-score-${score.id}`}>
      <CardContent className="p-4 space-y-4">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="font-medium truncate max-w-[180px]">{materialName || score.materialId}</div>
            <Badge variant="outline" className={`mt-1 ${tierConfig.color}`}>
              <TierIcon className="h-3 w-3 mr-1" />
              {tierConfig.label}
            </Badge>
          </div>
          <div className="text-right">
            <div className={`text-3xl font-bold font-mono ${overallColor}`}>{overallPercent}</div>
            <div className="text-xs text-muted-foreground">Score</div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <ScoreGauge label="Synthesis" value={synthesis} icon={FlaskConical} />
          <ScoreGauge label="Cost" value={costFactor} icon={DollarSign} inverted />
          <ScoreGauge label="Oracle" value={overall} icon={Gauge} />
        </div>

        {Object.keys(breakdown).length > 0 && (
          <div className="pt-2 border-t">
            <div className="text-xs text-muted-foreground mb-2">Property Breakdown</div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(breakdown).slice(0, 4).map(([key, val]) => (
                <Badge key={key} variant="secondary" className="text-xs">
                  {key.replace(/_/g, " ")}: {typeof val === "number" ? (val * 100).toFixed(0) : val}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function ManufacturabilitySccoringPage() {
  const { toast } = useToast();
  const [tierFilter, setTierFilter] = useState<string>("all");
  const [minScore, setMinScore] = useState<number[]>([0]);
  const [showLowRiskOnly, setShowLowRiskOnly] = useState(false);

  const { data: oracleScores = [], isLoading } = useQuery<MaterialsOracleScore[]>({
    queryKey: ["/api/materials-oracle-scores"],
    queryFn: async () => {
      const res = await fetch("/api/materials-oracle-scores?limit=200", { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch scores");
      return res.json();
    },
  });

  const { data: materialsResponse } = useQuery<{ materials: MaterialEntity[], total: number }>({
    queryKey: ["/api/materials"],
  });

  const materialsMap = useMemo(() => {
    const map: Record<string, string> = {};
    (materialsResponse?.materials || []).forEach(m => {
      map[m.id] = m.name || m.id;
    });
    return map;
  }, [materialsResponse]);

  const stats = useMemo(() => {
    const total = oracleScores.length;
    const productionViable = oracleScores.filter(s => getTier(s.synthesisFeasibility || 0) === "production-viable").length;
    const pilotReady = oracleScores.filter(s => getTier(s.synthesisFeasibility || 0) === "pilot-ready").length;
    const labOnly = total - productionViable - pilotReady;
    const yieldRate = total > 0 ? ((productionViable / total) * 100).toFixed(1) : "0.0";
    return { total, productionViable, pilotReady, labOnly, yieldRate };
  }, [oracleScores]);

  const filteredScores = useMemo(() => {
    let result = oracleScores;
    if (tierFilter !== "all") {
      result = result.filter(s => getTier(s.synthesisFeasibility || 0) === tierFilter);
    }
    if (minScore[0] > 0) {
      result = result.filter(s => (s.oracleScore || 0) * 100 >= minScore[0]);
    }
    if (showLowRiskOnly) {
      result = result.filter(s => (s.manufacturingCostFactor || 0) < 0.4);
    }
    return result;
  }, [oracleScores, tierFilter, minScore, showLowRiskOnly]);

  const runScoringMutation = useMutation({
    mutationFn: async () => {
      const materialsData = (materialsResponse?.materials || []).slice(0, 10).map(m => {
        const rep = m.representation as any;
        return {
          type: m.type,
          smiles: rep?.smiles,
          formula: rep?.formula,
        };
      });
      if (materialsData.length === 0) throw new Error("No materials to score");
      const res = await apiRequest("POST", "/api/compute/materials/manufacturability", { materials: materialsData });
      return res.json();
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["/api/materials-oracle-scores"] });
      toast({
        title: "Scoring Complete",
        description: `Scored ${result.results?.length || 0} materials for manufacturability.`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Scoring Failed",
        description: error.message || "Failed to run scoring",
        variant: "destructive",
      });
    },
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Manufacturability Scoring" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              disabled={runScoringMutation.isPending}
              onClick={() => runScoringMutation.mutate()}
              data-testid="button-recalculate"
            >
              {runScoringMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-2" />
              )}
              {runScoringMutation.isPending ? "Scoring..." : "Run Scoring"}
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6 rounded-lg border border-primary/20">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/30">
                <Factory className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-1">Manufacturability Scoring Engine</h2>
                <p className="text-muted-foreground">
                  Evaluate production feasibility at scale. Calculate <strong className="text-foreground">complexity</strong>,
                  <strong className="text-foreground"> cost proxy</strong>, <strong className="text-foreground">scale-up risk</strong>, 
                  and <strong className="text-foreground">process sensitivity</strong> for every variant.
                </p>
              </div>
            </div>
          </div>

          {isLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-4">
                    <Skeleton className="h-8 w-16 mb-1" />
                    <Skeleton className="h-4 w-24" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : oracleScores.length > 0 ? (
            <>
              <Card className="bg-gradient-to-r from-green-500/5 via-green-500/10 to-green-500/5 border-green-500/20">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="flex items-center gap-3">
                      <Building2 className="h-8 w-8 text-green-600 dark:text-green-400" />
                      <div>
                        <div className="text-sm text-muted-foreground">Production-Viable Materials</div>
                        <div className="text-2xl font-bold">
                          <span className="text-green-600 dark:text-green-400 font-mono">{formatNumber(stats.productionViable)}</span>
                          <span className="text-muted-foreground font-normal text-base ml-2">of {formatNumber(stats.total)} scored materials</span>
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="default"
                      onClick={() => setTierFilter("production-viable")}
                      data-testid="button-show-manufacturable"
                    >
                      <Zap className="h-4 w-4 mr-2" />
                      Show Only Production-Viable
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-md bg-green-500/10 flex items-center justify-center">
                        <Building2 className="h-5 w-5 text-green-600 dark:text-green-400" />
                      </div>
                      <div>
                        <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">{formatNumber(stats.productionViable)}</div>
                        <div className="text-xs text-muted-foreground">Production-Viable</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-md bg-blue-500/10 flex items-center justify-center">
                        <Rocket className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <div className="text-2xl font-bold font-mono text-blue-600 dark:text-blue-400">{formatNumber(stats.pilotReady)}</div>
                        <div className="text-xs text-muted-foreground">Pilot-Ready</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-md bg-amber-500/10 flex items-center justify-center">
                        <FlaskConical className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                      </div>
                      <div>
                        <div className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">{formatNumber(stats.labOnly)}</div>
                        <div className="text-xs text-muted-foreground">Lab-Only</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
                        <TrendingUp className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <div className="text-2xl font-bold font-mono">{stats.yieldRate}%</div>
                        <div className="text-xs text-muted-foreground">Yield Rate</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <Card className="p-12 text-center">
              <Factory className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
              <h3 className="text-lg font-medium mb-2">No scores yet</h3>
              <p className="text-muted-foreground mb-4">
                Run the scoring engine on your materials to evaluate manufacturability.
              </p>
              <Button
                disabled={runScoringMutation.isPending || !materialsResponse?.materials?.length}
                onClick={() => runScoringMutation.mutate()}
                data-testid="button-run-first-scoring"
              >
                {runScoringMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Gauge className="h-4 w-4 mr-2" />
                )}
                Run Scoring
              </Button>
            </Card>
          )}

          {oracleScores.length > 0 && (
            <>
              <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Material Scores
                  <Badge variant="secondary" className="ml-2">{filteredScores.length} shown</Badge>
                </h3>
                <div className="flex items-center gap-4 flex-wrap">
                  <div className="flex items-center gap-2">
                    <Label className="text-sm text-muted-foreground">Tier:</Label>
                    <Select value={tierFilter} onValueChange={setTierFilter}>
                      <SelectTrigger className="w-40" data-testid="select-tier-filter">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Tiers</SelectItem>
                        <SelectItem value="production-viable">Production-Viable</SelectItem>
                        <SelectItem value="pilot-ready">Pilot-Ready</SelectItem>
                        <SelectItem value="lab-only">Lab-Only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center gap-2">
                    <Label className="text-sm text-muted-foreground">Min Score:</Label>
                    <Slider
                      value={minScore}
                      onValueChange={setMinScore}
                      max={100}
                      min={0}
                      step={5}
                      className="w-24"
                    />
                    <span className="text-sm font-mono w-8">{minScore[0]}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={showLowRiskOnly}
                      onCheckedChange={setShowLowRiskOnly}
                      data-testid="switch-low-risk"
                    />
                    <Label className="text-sm">Low cost only</Label>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredScores.map(score => (
                  <OracleScoreCard
                    key={score.id}
                    score={score}
                    materialName={materialsMap[score.materialId]}
                  />
                ))}
              </div>

              {filteredScores.length === 0 && (
                <div className="text-center py-12 text-muted-foreground">
                  No materials match current filters. Try adjusting the tier or minimum score.
                </div>
              )}
            </>
          )}

          <ResultsPanel
            materialsCampaignId="demo-campaign"
            title="Scoring Artifacts"
            collapsible={true}
            defaultExpanded={false}
          />
        </div>
      </main>
    </div>
  );
}
