import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRoute, Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  Hexagon,
  Target,
  Activity,
  Zap,
  Layers,
  BarChart3,
  Factory,
  Calculator,
  ArrowLeft,
  CheckCircle,
  Clock,
  Play,
  Pause,
  TrendingUp,
  Sparkles,
  AlertTriangle,
} from "lucide-react";
import type { MaterialsCampaign, MaterialsCampaignAggregate } from "@shared/schema";

function formatNumber(num: number): string {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
  if (num >= 1000) return (num / 1000).toFixed(0) + "K";
  return num.toLocaleString();
}

function formatDate(dateStr: string | Date | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

const statusColors: Record<string, string> = {
  running: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30",
  completed: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30",
  paused: "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/30",
  queued: "bg-muted text-muted-foreground border-border",
  pending: "bg-muted text-muted-foreground border-border",
  failed: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30",
};

const statusIconMap: Record<string, typeof Play> = {
  running: Play,
  completed: CheckCircle,
  paused: Pause,
  queued: Clock,
  pending: Clock,
  failed: Zap,
};

export default function MaterialsCampaignDetailPage() {
  const [, params] = useRoute("/materials-campaigns/:id");
  const campaignId = params?.id;
  const [activeTab, setActiveTab] = useState("overview");

  const { data: campaign, isLoading: campaignLoading } = useQuery<MaterialsCampaign>({
    queryKey: ["/api/materials-campaigns", campaignId],
    queryFn: async () => {
      const res = await fetch(`/api/materials-campaigns/${campaignId}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch campaign");
      return res.json();
    },
    enabled: !!campaignId,
  });

  const { data: aggregate } = useQuery<MaterialsCampaignAggregate>({
    queryKey: ["/api/materials-campaigns", campaignId, "aggregates"],
    queryFn: async () => {
      const res = await fetch(`/api/materials-campaigns/${campaignId}/aggregates`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch aggregates");
      return res.json();
    },
    enabled: !!campaignId,
  });

  const { data: materialsResponse } = useQuery<{ materials: any[]; total: number }>({
    queryKey: ["/api/materials", campaignId],
    queryFn: async () => {
      const res = await fetch(`/api/materials?campaignId=${campaignId}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch materials");
      return res.json();
    },
    enabled: !!campaignId,
  });

  const materials = materialsResponse?.materials || [];
  const materialIds = materials.map((m: any) => m.id);

  const { data: allVariants = [] } = useQuery<any[]>({
    queryKey: ["/api/material-variants", campaignId, materialIds],
    queryFn: async () => {
      const res = await fetch(`/api/material-variants`, { credentials: "include" });
      if (!res.ok) return [];
      const data = await res.json();
      const list = data.variants || data || [];
      return list.filter((v: any) => materialIds.includes(v.materialId));
    },
    enabled: !!campaignId && materialIds.length > 0,
  });
  const variants = allVariants;

  const { data: oracleScores = [] } = useQuery<any[]>({
    queryKey: ["/api/materials-oracle-scores", campaignId],
    enabled: !!campaignId,
  });

  if (campaignLoading) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Materials Campaigns", href: "/materials-campaigns" }, { label: "Loading..." }]} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        </main>
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Materials Campaigns", href: "/materials-campaigns" }, { label: "Not Found" }]} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto text-center py-16">
            <Hexagon className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
            <h2 className="text-xl font-semibold mb-2">Campaign Not Found</h2>
            <p className="text-muted-foreground mb-4">This campaign may have been deleted.</p>
            <Link href="/materials-campaigns">
              <Button data-testid="button-back-to-campaigns">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Campaigns
              </Button>
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const status = (campaign.status || "pending") as string;
  const StatusIcon = statusIconMap[status] || Clock;
  const pipelineConfig = campaign.pipelineConfig as Record<string, any> | null;

  const totalVariants = aggregate?.totalVariants || 0;
  const totalMaterials = aggregate?.totalMaterials || materials.length;
  const avgScore = aggregate?.avgOracleScore || 0;

  const scoreTiers = useMemo(() => {
    const productionViable = oracleScores.filter((s: any) => s.score >= 0.7).length;
    const pilotReady = oracleScores.filter((s: any) => s.score >= 0.4 && s.score < 0.7).length;
    const labOnly = oracleScores.filter((s: any) => s.score < 0.4).length;
    return { productionViable, pilotReady, labOnly, total: oracleScores.length };
  }, [oracleScores]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Materials Campaigns", href: "/materials-campaigns" },
          { label: campaign.name },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Link href="/materials-campaigns">
              <Button variant="outline" data-testid="button-back">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </Link>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-amber-950/30 via-amber-900/20 to-background p-6 rounded-lg border border-amber-500/20">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-lg bg-amber-500/20 flex items-center justify-center border border-amber-500/30">
                  <Hexagon className="h-7 w-7 text-amber-400" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold" data-testid="text-campaign-name">{campaign.name}</h1>
                  <p className="text-sm text-muted-foreground capitalize">
                    {campaign.modality || campaign.domain || "materials"} &middot; Created {formatDate(campaign.createdAt)}
                  </p>
                </div>
              </div>
              <Badge variant="outline" className={statusColors[status] || statusColors.pending}>
                <StatusIcon className="h-3 w-3" />
                <span className="ml-1 capitalize">{status}</span>
              </Badge>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono">{formatNumber(totalMaterials)}</div>
                <div className="text-xs text-muted-foreground">Materials</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono">{formatNumber(totalVariants)}</div>
                <div className="text-xs text-muted-foreground">Variants</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
                  {avgScore > 0 ? avgScore.toFixed(2) : "—"}
                </div>
                <div className="text-xs text-muted-foreground">Avg Score</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">
                  {scoreTiers.productionViable}
                </div>
                <div className="text-xs text-muted-foreground">Production Viable</div>
              </CardContent>
            </Card>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="overview" data-testid="tab-overview">
                <BarChart3 className="h-4 w-4 mr-2" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="variants" data-testid="tab-variants">
                <Layers className="h-4 w-4 mr-2" />
                Variants ({variants.length})
              </TabsTrigger>
              <TabsTrigger value="properties" data-testid="tab-properties">
                <Calculator className="h-4 w-4 mr-2" />
                Properties
              </TabsTrigger>
              <TabsTrigger value="manufacturability" data-testid="tab-manufacturability">
                <Factory className="h-4 w-4 mr-2" />
                Manufacturability
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4 mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Target className="h-4 w-4 text-amber-500" />
                      Campaign Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Domain</span>
                      <span className="capitalize">{campaign.domain || "materials"}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Modality</span>
                      <span className="capitalize">{campaign.modality || "—"}</span>
                    </div>
                    {pipelineConfig?.targetProperty && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Target Property</span>
                        <span className="capitalize">{pipelineConfig.targetProperty.replace(/_/g, " ")}</span>
                      </div>
                    )}
                    {pipelineConfig?.materialType && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Material Type</span>
                        <span className="capitalize">{pipelineConfig.materialType.replace(/_/g, " ")}</span>
                      </div>
                    )}
                    {pipelineConfig?.variantCount && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Variant Count Target</span>
                        <span>{pipelineConfig.variantCount}</span>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-green-500" />
                      Score Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-green-500" />
                          <span>Production Viable (&ge;70%)</span>
                        </div>
                        <span className="font-mono font-medium">{scoreTiers.productionViable}</span>
                      </div>
                      <Progress value={scoreTiers.total > 0 ? (scoreTiers.productionViable / scoreTiers.total) * 100 : 0} className="h-2" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-amber-500" />
                          <span>Pilot Ready (&ge;40%)</span>
                        </div>
                        <span className="font-mono font-medium">{scoreTiers.pilotReady}</span>
                      </div>
                      <Progress value={scoreTiers.total > 0 ? (scoreTiers.pilotReady / scoreTiers.total) * 100 : 0} className="h-2" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-red-500" />
                          <span>Lab Only (&lt;40%)</span>
                        </div>
                        <span className="font-mono font-medium">{scoreTiers.labOnly}</span>
                      </div>
                      <Progress value={scoreTiers.total > 0 ? (scoreTiers.labOnly / scoreTiers.total) * 100 : 0} className="h-2" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Activity className="h-4 w-4 text-blue-500" />
                    Materials in Campaign ({materials.length})
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {materials.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Hexagon className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">No materials assigned to this campaign</p>
                    </div>
                  ) : (
                    <ScrollArea className="h-[250px]">
                      <div className="space-y-2 pr-2">
                        {materials.slice(0, 50).map((mat: any) => (
                          <div key={mat.id} className="flex items-center justify-between p-2 rounded border bg-card">
                            <div className="flex items-center gap-2 min-w-0">
                              <Hexagon className="h-4 w-4 text-amber-500 shrink-0" />
                              <div className="min-w-0">
                                <p className="text-sm font-medium truncate">{mat.name || mat.id}</p>
                                <p className="text-xs text-muted-foreground capitalize">{mat.type?.replace(/_/g, " ")}</p>
                              </div>
                            </div>
                            <Badge variant="secondary" className="text-xs shrink-0">
                              {mat.type?.replace(/_/g, " ")}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="variants" className="mt-4">
              <Card>
                <CardHeader className="pb-3 flex flex-row items-center justify-between gap-2 flex-wrap">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Layers className="h-4 w-4 text-violet-500" />
                    Material Variants
                  </CardTitle>
                  <Link href="/material-variants">
                    <Button variant="outline" size="sm" data-testid="button-view-all-variants">
                      View All Variants
                    </Button>
                  </Link>
                </CardHeader>
                <CardContent>
                  {variants.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Layers className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">No variants generated yet</p>
                      <p className="text-xs mt-1">Generate variants from your materials library</p>
                      <Link href="/materials-library">
                        <Button variant="outline" size="sm" className="mt-3" data-testid="button-go-to-library">
                          Go to Materials Library
                        </Button>
                      </Link>
                    </div>
                  ) : (
                    <ScrollArea className="h-[400px]">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>ID</TableHead>
                            <TableHead>Type</TableHead>
                            <TableHead>Base Material</TableHead>
                            <TableHead className="text-right">Score</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {variants.slice(0, 100).map((v: any, idx: number) => {
                            const params = v.variantParams || v.parameters || {};
                            return (
                              <TableRow key={v.id || idx} data-testid={`row-variant-${v.id || idx}`}>
                                <TableCell className="font-mono text-xs">
                                  {v.externalVariantId || v.id || `V-${idx + 1}`}
                                </TableCell>
                                <TableCell>
                                  <Badge variant="secondary" className="text-xs capitalize">
                                    {params.variant_type || v.variantType || "substitution"}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-sm truncate max-w-[200px]">
                                  {params.base_material || params.baseMaterial || "—"}
                                </TableCell>
                                <TableCell className="text-right font-mono text-sm">
                                  {params.score ? Number(params.score).toFixed(3) : "—"}
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="properties" className="mt-4">
              <Card>
                <CardHeader className="pb-3 flex flex-row items-center justify-between gap-2 flex-wrap">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Calculator className="h-4 w-4 text-emerald-500" />
                    Property Predictions
                  </CardTitle>
                  <Link href="/property-prediction">
                    <Button variant="outline" size="sm" data-testid="button-run-predictions">
                      <Sparkles className="h-3 w-3 mr-1" />
                      Run Predictions
                    </Button>
                  </Link>
                </CardHeader>
                <CardContent>
                  {oracleScores.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Calculator className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">No property predictions available</p>
                      <p className="text-xs mt-1">Run predictions on materials to see results here</p>
                      <Link href="/property-prediction">
                        <Button variant="outline" size="sm" className="mt-3" data-testid="button-go-to-prediction">
                          Go to Property Prediction
                        </Button>
                      </Link>
                    </div>
                  ) : (
                    <ScrollArea className="h-[400px]">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Material ID</TableHead>
                            <TableHead>Property</TableHead>
                            <TableHead className="text-right">Score</TableHead>
                            <TableHead className="text-right">Tier</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {oracleScores.slice(0, 100).map((score: any, idx: number) => {
                            const tier = score.score >= 0.7 ? "Production" : score.score >= 0.4 ? "Pilot" : "Lab";
                            const tierColor = score.score >= 0.7 ? "text-green-600 dark:text-green-400" : score.score >= 0.4 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400";
                            return (
                              <TableRow key={idx} data-testid={`row-score-${idx}`}>
                                <TableCell className="font-mono text-xs truncate max-w-[180px]">
                                  {score.materialId}
                                </TableCell>
                                <TableCell className="capitalize text-sm">
                                  {(score.propertyName || "").replace(/_/g, " ")}
                                </TableCell>
                                <TableCell className="text-right font-mono text-sm">
                                  {Number(score.score).toFixed(3)}
                                </TableCell>
                                <TableCell className="text-right">
                                  <Badge variant="outline" className={`text-xs ${tierColor}`}>
                                    {tier}
                                  </Badge>
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="manufacturability" className="mt-4">
              <Card>
                <CardHeader className="pb-3 flex flex-row items-center justify-between gap-2 flex-wrap">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Factory className="h-4 w-4 text-violet-500" />
                    Manufacturability Assessment
                  </CardTitle>
                  <Link href="/manufacturability-scoring">
                    <Button variant="outline" size="sm" data-testid="button-go-to-manufacturability">
                      Full Assessment
                    </Button>
                  </Link>
                </CardHeader>
                <CardContent>
                  {scoreTiers.total === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Factory className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">No manufacturability scores yet</p>
                      <p className="text-xs mt-1">Run scoring pipeline to assess production viability</p>
                      <Link href="/manufacturability-scoring">
                        <Button variant="outline" size="sm" className="mt-3" data-testid="button-run-scoring">
                          Run Scoring
                        </Button>
                      </Link>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20 text-center">
                          <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">
                            {scoreTiers.productionViable}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">Production Viable</div>
                          <div className="text-xs text-green-600 dark:text-green-400 mt-0.5">
                            {scoreTiers.total > 0 ? ((scoreTiers.productionViable / scoreTiers.total) * 100).toFixed(0) : 0}%
                          </div>
                        </div>
                        <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/20 text-center">
                          <div className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
                            {scoreTiers.pilotReady}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">Pilot Ready</div>
                          <div className="text-xs text-amber-600 dark:text-amber-400 mt-0.5">
                            {scoreTiers.total > 0 ? ((scoreTiers.pilotReady / scoreTiers.total) * 100).toFixed(0) : 0}%
                          </div>
                        </div>
                        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-center">
                          <div className="text-2xl font-bold font-mono text-red-600 dark:text-red-400">
                            {scoreTiers.labOnly}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">Lab Only</div>
                          <div className="text-xs text-red-600 dark:text-red-400 mt-0.5">
                            {scoreTiers.total > 0 ? ((scoreTiers.labOnly / scoreTiers.total) * 100).toFixed(0) : 0}%
                          </div>
                        </div>
                      </div>

                      <div className="p-4 rounded-lg border bg-muted/30">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Overall Average</span>
                          <span className="text-lg font-bold font-mono">
                            {avgScore > 0 ? (avgScore * 100).toFixed(1) + "%" : "—"}
                          </span>
                        </div>
                        <Progress value={avgScore * 100} className="h-2" />
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}
