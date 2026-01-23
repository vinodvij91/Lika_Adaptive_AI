import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { MetricCard } from "@/components/metric-card";
import { StatusBadge } from "@/components/status-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Hexagon,
  Workflow,
  Sparkles,
  ArrowRight,
  Plus,
  Layers,
  Calculator,
  Factory,
  Beaker,
} from "lucide-react";
import type { MaterialsCampaign } from "@shared/schema";

interface MaterialsDashboardStats {
  materialVariantsEvaluated: number;
  propertiesPredicted: number;
  manufacturableCandidates: number;
  activePipelines: number;
}

export default function MaterialsDashboardPage() {
  const { data: stats, isLoading: statsLoading } = useQuery<MaterialsDashboardStats>({
    queryKey: ["/api/dashboard/stats", "materials"],
  });

  const { data: recentCampaigns, isLoading: campaignsLoading } = useQuery<MaterialsCampaign[]>({
    queryKey: ["/api/materials-campaigns", { limit: 5 }],
  });

  const ctaCards = [
    {
      title: "Import Materials",
      description: "Upload polymers, crystals, or composites data",
      icon: Hexagon,
      href: "/import/materials/materials",
      color: "chart-2",
    },
    {
      title: "Start Property Pipeline",
      description: "Run property prediction on your materials",
      icon: Calculator,
      href: "/property-pipelines",
      color: "primary",
    },
    {
      title: "Manufacturability Analysis",
      description: "Assess production feasibility of candidates",
      icon: Factory,
      href: "/manufacturability-scoring",
      color: "chart-3",
    },
  ];

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Materials Science" }, { label: "Dashboard" }]}
        actions={
          <Link href="/materials-campaigns">
            <Button className="gap-2" data-testid="button-new-campaign">
              <Plus className="h-4 w-4" />
              New Campaign
            </Button>
          </Link>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {statsLoading ? (
              <>
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
              </>
            ) : (
              <>
                <MetricCard
                  title="Material Variants Evaluated"
                  value={stats?.materialVariantsEvaluated?.toLocaleString() ?? 0}
                  icon={Hexagon}
                  subtitle="Total variants processed"
                />
                <MetricCard
                  title="Properties Predicted"
                  value={stats?.propertiesPredicted?.toLocaleString() ?? 0}
                  icon={Calculator}
                  subtitle="Property predictions run"
                />
                <MetricCard
                  title="Manufacturable Candidates"
                  value={stats?.manufacturableCandidates ?? 0}
                  icon={Factory}
                  subtitle="Production-ready materials"
                />
                <MetricCard
                  title="Active Pipelines"
                  value={stats?.activePipelines ?? 0}
                  icon={Workflow}
                  subtitle="Currently running"
                />
              </>
            )}
          </div>

          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="text-lg flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-chart-2" />
                What should I do next?
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {ctaCards.map((card) => (
                  <Link key={card.title} href={card.href}>
                    <div
                      className="p-4 rounded-md border hover-elevate cursor-pointer h-full"
                      data-testid={`card-cta-${card.title.toLowerCase().replace(/\s+/g, "-")}`}
                    >
                      <div className={`w-10 h-10 rounded-md bg-${card.color}/10 flex items-center justify-center mb-3`}>
                        <card.icon className={`h-5 w-5 text-${card.color}`} />
                      </div>
                      <h3 className="font-medium mb-1">{card.title}</h3>
                      <p className="text-sm text-muted-foreground">{card.description}</p>
                    </div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Layers className="h-5 w-5 text-muted-foreground" />
                  Materials Library
                </CardTitle>
                <Link href="/multi-scale-representations">
                  <Button variant="ghost" size="sm" className="gap-1" data-testid="link-view-materials">
                    View All
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </CardHeader>
              <CardContent className="pt-0">
                <EmptyState
                  icon={Hexagon}
                  title="No materials yet"
                  description="Import materials to start your discovery"
                  action={
                    <Link href="/import/materials/materials">
                      <Button size="sm" className="gap-2">
                        <Plus className="h-4 w-4" />
                        Import Materials
                      </Button>
                    </Link>
                  }
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Workflow className="h-5 w-5 text-muted-foreground" />
                  Recent Campaigns
                </CardTitle>
                <Link href="/materials-campaigns">
                  <Button variant="ghost" size="sm" className="gap-1" data-testid="link-view-all-campaigns">
                    View All
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </CardHeader>
              <CardContent className="pt-0">
                {campaignsLoading ? (
                  <LoadingRows />
                ) : recentCampaigns && recentCampaigns.length > 0 ? (
                  <div className="space-y-3">
                    {recentCampaigns.map((campaign) => (
                      <Link key={campaign.id} href={`/materials-campaigns/${campaign.id}`}>
                        <div
                          className="flex items-center gap-3 p-3 rounded-md hover-elevate cursor-pointer"
                          data-testid={`card-campaign-${campaign.id}`}
                        >
                          <div className="w-10 h-10 rounded-md bg-chart-2/10 flex items-center justify-center flex-shrink-0">
                            <Workflow className="h-5 w-5 text-chart-2" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{campaign.name}</p>
                            <p className="text-sm text-muted-foreground truncate">
                              {campaign.modality || "Materials"}
                            </p>
                          </div>
                          {campaign.status && <StatusBadge status={campaign.status} />}
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    icon={Workflow}
                    title="No campaigns yet"
                    description="Start a campaign to discover new materials"
                    action={
                      <Link href="/materials-campaigns">
                        <Button size="sm" variant="outline" className="gap-2">
                          Start Campaign
                        </Button>
                      </Link>
                    }
                  />
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function MetricCardSkeleton() {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <Skeleton className="h-4 w-24 mb-2" />
            <Skeleton className="h-8 w-16" />
          </div>
          <Skeleton className="h-10 w-10 rounded-md" />
        </div>
      </CardContent>
    </Card>
  );
}

function LoadingRows() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="flex items-center gap-3">
          <Skeleton className="h-10 w-10 rounded-md" />
          <div className="space-y-1.5 flex-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: typeof Hexagon;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-3">
        <Icon className="h-6 w-6 text-muted-foreground" />
      </div>
      <p className="font-medium mb-1">{title}</p>
      <p className="text-sm text-muted-foreground mb-4">{description}</p>
      {action}
    </div>
  );
}
