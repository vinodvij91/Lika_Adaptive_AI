import { useState, useMemo } from "react";
import { Link } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  Hexagon,
  Target,
  Activity,
  Zap,
  Plus,
  ArrowRight,
  Clock,
  CheckCircle,
  Play,
  Pause,
  Settings,
  BarChart3,
  Factory,
  Loader2,
  Trash2,
  RefreshCw,
} from "lucide-react";
import type { MaterialsCampaign, MaterialsCampaignAggregate } from "@shared/schema";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

function formatLargeNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(2) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + "K";
  }
  return num.toLocaleString();
}

function formatDate(dateStr: string | Date | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

type CampaignStatus = "running" | "completed" | "paused" | "queued" | "pending" | "failed";

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

function CampaignCard({ campaign, aggregate }: { campaign: MaterialsCampaign; aggregate?: MaterialsCampaignAggregate }) {
  const totalVariants = aggregate?.totalVariants || 0;
  const totalMaterials = aggregate?.totalMaterials || 0;
  const avgScore = aggregate?.avgOracleScore || 0;
  const status = (campaign.status || "pending") as CampaignStatus;
  const StatusIcon = statusIconMap[status] || Clock;
  const pipelineConfig = campaign.pipelineConfig as Record<string, any> | null;

  return (
    <Link href={`/materials-campaigns/${campaign.id}`}>
      <Card className="hover-elevate cursor-pointer" data-testid={`card-campaign-${campaign.id}`}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-md bg-amber-500/10 flex items-center justify-center border border-amber-500/30">
                <Hexagon className="h-5 w-5 text-amber-400" />
              </div>
              <div>
                <CardTitle className="text-base">{campaign.name}</CardTitle>
                <p className="text-xs text-muted-foreground capitalize">
                  {campaign.modality || campaign.domain || "materials"}
                </p>
              </div>
            </div>
            <Badge variant="outline" className={statusColors[status] || statusColors.pending}>
              <StatusIcon className="h-3 w-3" />
              <span className="ml-1 capitalize">{status}</span>
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-md bg-muted/50 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Activity className="h-3 w-3" />
                <span>Materials</span>
              </div>
              <div className="text-lg font-bold font-mono">
                {formatNumber(totalMaterials)}
              </div>
            </div>
            <div className="p-3 rounded-md bg-muted/50 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Zap className="h-3 w-3" />
                <span>Variants</span>
              </div>
              <div className="text-lg font-bold font-mono">
                {formatNumber(totalVariants)}
              </div>
            </div>
          </div>

          {avgScore > 0 && (
            <div className="p-3 rounded-md bg-green-500/10 border border-green-500/20 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
                <Target className="h-3 w-3" />
                <span>Avg Oracle Score</span>
              </div>
              <div className="text-lg font-bold font-mono text-green-700 dark:text-green-300">
                {avgScore.toFixed(2)}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-2 border-t gap-2">
            <span className="text-xs text-muted-foreground">
              Created {formatDate(campaign.createdAt)}
            </span>
            <div className="flex gap-2">
              {status === "completed" && (
                <Button variant="default" size="sm" data-testid={`button-triage-${campaign.id}`} onClick={(e) => e.stopPropagation()}>
                  <Target className="h-3.5 w-3.5 mr-1" />
                  View Triage
                </Button>
              )}
              <Button variant="ghost" size="sm" data-testid={`button-view-campaign-${campaign.id}`}>
                View Details
                <ArrowRight className="h-3.5 w-3.5 ml-1" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function PlatformMetrics({ campaigns, aggregates }: { campaigns: MaterialsCampaign[]; aggregates: Record<string, MaterialsCampaignAggregate> }) {
  const totals = useMemo(() => {
    let totalMaterials = 0;
    let totalVariants = 0;
    let totalWithScores = 0;
    let sumScores = 0;

    Object.values(aggregates).forEach(agg => {
      totalMaterials += agg.totalMaterials || 0;
      totalVariants += agg.totalVariants || 0;
      if (agg.avgOracleScore && agg.avgOracleScore > 0) {
        sumScores += agg.avgOracleScore;
        totalWithScores++;
      }
    });

    return {
      totalMaterials,
      totalVariants,
      avgScore: totalWithScores > 0 ? sumScores / totalWithScores : 0,
    };
  }, [aggregates]);

  const activeCampaigns = campaigns.filter(c => c.status === "running").length;
  const completedCampaigns = campaigns.filter(c => c.status === "completed").length;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono">{campaigns.length}</div>
          <div className="text-xs text-muted-foreground">Total Campaigns</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">{activeCampaigns}</div>
          <div className="text-xs text-muted-foreground">Running</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono text-blue-600 dark:text-blue-400">{completedCampaigns}</div>
          <div className="text-xs text-muted-foreground">Completed</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono">{formatNumber(totals.totalMaterials)}</div>
          <div className="text-xs text-muted-foreground">Total Materials</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono">{formatNumber(totals.totalVariants)}</div>
          <div className="text-xs text-muted-foreground">Total Variants</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
            {totals.avgScore > 0 ? totals.avgScore.toFixed(2) : "—"}
          </div>
          <div className="text-xs text-muted-foreground">Avg Score</div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function MaterialsCampaignsPage() {
  const { toast } = useToast();
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const { data: campaigns = [], isLoading } = useQuery<MaterialsCampaign[]>({
    queryKey: ["/api/materials-campaigns"],
  });

  const { data: aggregatesMap = {} } = useQuery<Record<string, MaterialsCampaignAggregate>>({
    queryKey: ["/api/materials-campaigns", "all-aggregates"],
    queryFn: async () => {
      const map: Record<string, MaterialsCampaignAggregate> = {};
      await Promise.all(
        campaigns.map(async (c) => {
          try {
            const res = await fetch(`/api/materials-campaigns/${c.id}/aggregates`, { credentials: "include" });
            if (res.ok) {
              const data = await res.json();
              if (data) map[c.id] = data;
            }
          } catch {}
        })
      );
      return map;
    },
    enabled: campaigns.length > 0,
  });

  const filteredCampaigns = useMemo(() => {
    if (statusFilter === "all") return campaigns;
    return campaigns.filter(c => c.status === statusFilter);
  }, [campaigns, statusFilter]);

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/materials-campaigns/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/materials-campaigns"] });
      toast({ title: "Campaign deleted" });
    },
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Materials Campaigns" }]}
        actions={
          <div className="flex items-center gap-2">
            <Link href="/materials/campaigns/new">
              <Button data-testid="button-new-campaign">
                <Plus className="h-4 w-4 mr-2" />
                New Campaign
              </Button>
            </Link>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-amber-950/30 via-amber-900/20 to-background p-6 rounded-lg border border-amber-500/20">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-amber-500/20 flex items-center justify-center border border-amber-500/30">
                <Hexagon className="h-6 w-6 text-amber-400" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-1">Enterprise Materials Discovery</h2>
                <p className="text-muted-foreground">
                  Evaluate <strong className="text-amber-400">millions of material variants</strong>, 
                  execute <strong className="text-amber-400">high-throughput property prediction pipelines</strong>, 
                  and iterate toward real-world performance using <strong className="text-amber-400">simulation-in-the-loop optimization</strong>.
                </p>
              </div>
            </div>
          </div>

          {isLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-4 text-center">
                    <Skeleton className="h-8 w-16 mx-auto mb-1" />
                    <Skeleton className="h-4 w-20 mx-auto" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <PlatformMetrics campaigns={campaigns} aggregates={aggregatesMap} />
          )}

          <div className="flex items-center justify-between flex-wrap gap-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Campaign Portfolio
            </h3>
            <Tabs value={statusFilter} onValueChange={setStatusFilter}>
              <TabsList>
                <TabsTrigger value="all" data-testid="tab-filter-all">All</TabsTrigger>
                <TabsTrigger value="running" data-testid="tab-filter-running">Running</TabsTrigger>
                <TabsTrigger value="completed" data-testid="tab-filter-completed">Completed</TabsTrigger>
                <TabsTrigger value="paused" data-testid="tab-filter-paused">Paused</TabsTrigger>
                <TabsTrigger value="pending" data-testid="tab-filter-pending">Pending</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {isLoading ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {Array.from({ length: 4 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-6 space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-16 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {filteredCampaigns.map(campaign => (
                <CampaignCard
                  key={campaign.id}
                  campaign={campaign}
                  aggregate={aggregatesMap[campaign.id]}
                />
              ))}
            </div>
          )}

          {!isLoading && filteredCampaigns.length === 0 && (
            <Card className="p-12 text-center">
              <Hexagon className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
              <h3 className="text-lg font-medium mb-2">
                {campaigns.length === 0 ? "No campaigns yet" : "No campaigns found"}
              </h3>
              <p className="text-muted-foreground mb-4">
                {campaigns.length === 0
                  ? "Create your first materials discovery campaign to start evaluating variants."
                  : "No campaigns match the current filter."}
              </p>
              {campaigns.length === 0 ? (
                <Link href="/materials/campaigns/new">
                  <Button data-testid="button-new-campaign-empty">
                    <Plus className="h-4 w-4 mr-2" />
                    New Campaign
                  </Button>
                </Link>
              ) : (
                <Button onClick={() => setStatusFilter("all")}>
                  View All Campaigns
                </Button>
              )}
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
