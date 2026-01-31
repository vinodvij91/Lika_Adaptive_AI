import { useState, useMemo } from "react";
import { Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
} from "lucide-react";

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

interface CampaignMetrics {
  id: string;
  name: string;
  domain: string;
  status: "running" | "completed" | "paused" | "queued";
  variantsEvaluated: number;
  totalVariants: number;
  predictionsGenerated: number;
  activeSimulationJobs: number;
  topPercentile: number;
  topCandidatesCount: number;
  manufacturingReadyCount: number;
  createdAt: string;
  estimatedCompletion?: string;
}

function generateMockCampaigns(): CampaignMetrics[] {
  return [
    {
      id: "1",
      name: "High-Permeability Membranes",
      domain: "membrane",
      status: "running",
      variantsEvaluated: 420000,
      totalVariants: 520000,
      predictionsGenerated: 1340000,
      activeSimulationJobs: 12,
      topPercentile: 5,
      topCandidatesCount: 8200,
      manufacturingReadyCount: 1450,
      createdAt: "2025-12-15",
      estimatedCompletion: "4h 32m",
    },
    {
      id: "2",
      name: "Thermal-Stable Polymer Coatings",
      domain: "coating",
      status: "running",
      variantsEvaluated: 285000,
      totalVariants: 350000,
      predictionsGenerated: 892000,
      activeSimulationJobs: 8,
      topPercentile: 3,
      topCandidatesCount: 4250,
      manufacturingReadyCount: 890,
      createdAt: "2025-12-18",
      estimatedCompletion: "2h 15m",
    },
    {
      id: "3",
      name: "High-Conductivity Electrolytes",
      domain: "electrolyte",
      status: "completed",
      variantsEvaluated: 680000,
      totalVariants: 680000,
      predictionsGenerated: 2100000,
      activeSimulationJobs: 0,
      topPercentile: 2,
      topCandidatesCount: 12400,
      manufacturingReadyCount: 3200,
      createdAt: "2025-11-20",
    },
    {
      id: "4",
      name: "Lightweight Structural Composites",
      domain: "composite",
      status: "paused",
      variantsEvaluated: 145000,
      totalVariants: 400000,
      predictionsGenerated: 456000,
      activeSimulationJobs: 0,
      topPercentile: 5,
      topCandidatesCount: 2100,
      manufacturingReadyCount: 450,
      createdAt: "2025-12-10",
    },
    {
      id: "5",
      name: "Next-Gen Catalyst Discovery",
      domain: "catalyst",
      status: "queued",
      variantsEvaluated: 0,
      totalVariants: 750000,
      predictionsGenerated: 0,
      activeSimulationJobs: 0,
      topPercentile: 0,
      topCandidatesCount: 0,
      manufacturingReadyCount: 0,
      createdAt: "2025-12-22",
    },
  ];
}

function CampaignCard({ campaign }: { campaign: CampaignMetrics }) {
  const progress = campaign.totalVariants > 0 
    ? (campaign.variantsEvaluated / campaign.totalVariants) * 100 
    : 0;

  const statusColors = {
    running: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30",
    completed: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30",
    paused: "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/30",
    queued: "bg-muted text-muted-foreground border-border",
  };

  const statusIcons = {
    running: <Play className="h-3 w-3" />,
    completed: <CheckCircle className="h-3 w-3" />,
    paused: <Pause className="h-3 w-3" />,
    queued: <Clock className="h-3 w-3" />,
  };

  return (
    <Card className="hover-elevate" data-testid={`card-campaign-${campaign.id}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-md bg-amber-500/10 flex items-center justify-center border border-amber-500/30">
              <Hexagon className="h-5 w-5 text-amber-400" />
            </div>
            <div>
              <CardTitle className="text-base">{campaign.name}</CardTitle>
              <p className="text-xs text-muted-foreground capitalize">{campaign.domain}</p>
            </div>
          </div>
          <Badge variant="outline" className={statusColors[campaign.status]}>
            {statusIcons[campaign.status]}
            <span className="ml-1 capitalize">{campaign.status}</span>
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {campaign.status !== "queued" && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Variants Evaluated</span>
              <span className="font-mono font-medium">
                {formatNumber(campaign.variantsEvaluated)} / {formatNumber(campaign.totalVariants)}
              </span>
            </div>
            <Progress value={progress} className="h-2" />
            {campaign.estimatedCompletion && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                <span>Est. completion: {campaign.estimatedCompletion}</span>
              </div>
            )}
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-md bg-muted/50 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Activity className="h-3 w-3" />
              <span>Predictions</span>
            </div>
            <div className="text-lg font-bold font-mono">
              {formatLargeNumber(campaign.predictionsGenerated)}
            </div>
          </div>
          <div className="p-3 rounded-md bg-muted/50 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Zap className="h-3 w-3" />
              <span>Active Jobs</span>
            </div>
            <div className="text-lg font-bold font-mono">
              {campaign.activeSimulationJobs}
            </div>
          </div>
        </div>

        {campaign.topCandidatesCount > 0 && (
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-md bg-green-500/10 border border-green-500/20 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
                <Target className="h-3 w-3" />
                <span>Top {campaign.topPercentile}%</span>
              </div>
              <div className="text-lg font-bold font-mono text-green-700 dark:text-green-300">
                {formatNumber(campaign.topCandidatesCount)}
              </div>
              <div className="text-xs text-muted-foreground">candidates</div>
            </div>
            <div className="p-3 rounded-md bg-amber-500/10 border border-amber-500/20 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400">
                <Factory className="h-3 w-3" />
                <span>Mfg Ready</span>
              </div>
              <div className="text-lg font-bold font-mono text-amber-700 dark:text-amber-300">
                {formatNumber(campaign.manufacturingReadyCount)}
              </div>
              <div className="text-xs text-muted-foreground">variants</div>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between pt-2 border-t gap-2">
          <span className="text-xs text-muted-foreground">Created {campaign.createdAt}</span>
          <div className="flex gap-2">
            {campaign.status === "completed" && (
              <Link href={`/materials/campaigns/${campaign.id}/triage`}>
                <Button variant="default" size="sm" data-testid={`button-triage-${campaign.id}`}>
                  <Target className="h-3.5 w-3.5 mr-1" />
                  View Triage
                </Button>
              </Link>
            )}
            <Button variant="ghost" size="sm" data-testid={`button-view-campaign-${campaign.id}`}>
              View Details
              <ArrowRight className="h-3.5 w-3.5 ml-1" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PlatformMetrics({ campaigns }: { campaigns: CampaignMetrics[] }) {
  const totals = useMemo(() => {
    return campaigns.reduce((acc, c) => ({
      variantsEvaluated: acc.variantsEvaluated + c.variantsEvaluated,
      totalVariants: acc.totalVariants + c.totalVariants,
      predictions: acc.predictions + c.predictionsGenerated,
      activeJobs: acc.activeJobs + c.activeSimulationJobs,
      topCandidates: acc.topCandidates + c.topCandidatesCount,
      manufacturingReady: acc.manufacturingReady + c.manufacturingReadyCount,
    }), { variantsEvaluated: 0, totalVariants: 0, predictions: 0, activeJobs: 0, topCandidates: 0, manufacturingReady: 0 });
  }, [campaigns]);

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
          <div className="text-2xl font-bold font-mono">{formatNumber(totals.variantsEvaluated)}</div>
          <div className="text-xs text-muted-foreground">Variants Evaluated</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono">{formatLargeNumber(totals.predictions)}</div>
          <div className="text-xs text-muted-foreground">Predictions</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">{formatNumber(totals.topCandidates)}</div>
          <div className="text-xs text-muted-foreground">Top Candidates</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">{formatNumber(totals.manufacturingReady)}</div>
          <div className="text-xs text-muted-foreground">Mfg Ready</div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function MaterialsCampaignsPage() {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const campaigns = useMemo(() => generateMockCampaigns(), []);

  const filteredCampaigns = useMemo(() => {
    if (statusFilter === "all") return campaigns;
    return campaigns.filter(c => c.status === statusFilter);
  }, [campaigns, statusFilter]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Materials Campaigns" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" data-testid="button-campaign-settings">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
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

          <PlatformMetrics campaigns={campaigns} />

          <div className="flex items-center justify-between">
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
              </TabsList>
            </Tabs>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredCampaigns.map(campaign => (
              <CampaignCard key={campaign.id} campaign={campaign} />
            ))}
          </div>

          {filteredCampaigns.length === 0 && (
            <Card className="p-12 text-center">
              <Hexagon className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
              <h3 className="text-lg font-medium mb-2">No campaigns found</h3>
              <p className="text-muted-foreground mb-4">
                No campaigns match the current filter.
              </p>
              <Button onClick={() => setStatusFilter("all")}>
                View All Campaigns
              </Button>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
