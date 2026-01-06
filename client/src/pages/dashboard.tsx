import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { MetricCard } from "@/components/metric-card";
import { StatusBadge } from "@/components/status-badge";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  FlaskConical,
  Workflow,
  Target,
  Sparkles,
  ArrowRight,
  Plus,
  FolderKanban,
} from "lucide-react";
import type { Project, Campaign, DiseaseArea } from "@shared/schema";

interface DashboardStats {
  totalMolecules: number;
  totalCampaigns: number;
  activeCampaigns: number;
  campaignsThisWeek: number;
  domainBreakdown: Record<DiseaseArea, number>;
}

export default function DashboardPage() {
  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>({
    queryKey: ["/api/dashboard/stats"],
  });

  const { data: recentProjects, isLoading: projectsLoading } = useQuery<Project[]>({
    queryKey: ["/api/projects", { limit: 5 }],
  });

  const { data: recentCampaigns, isLoading: campaignsLoading } = useQuery<Campaign[]>({
    queryKey: ["/api/campaigns", { limit: 5 }],
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Dashboard" }]}
        actions={
          <Link href="/projects">
            <Button className="gap-2" data-testid="button-new-project">
              <Plus className="h-4 w-4" />
              New Project
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
                  title="Molecules Evaluated"
                  value={stats?.totalMolecules?.toLocaleString() ?? 0}
                  icon={FlaskConical}
                  subtitle="Total across all campaigns"
                />
                <MetricCard
                  title="Total Campaigns"
                  value={stats?.totalCampaigns ?? 0}
                  icon={Workflow}
                />
                <MetricCard
                  title="Active Campaigns"
                  value={stats?.activeCampaigns ?? 0}
                  icon={Sparkles}
                  subtitle="Currently running"
                />
                <MetricCard
                  title="Weekly Throughput"
                  value={stats?.campaignsThisWeek ?? 0}
                  icon={Target}
                  subtitle="Campaigns completed this week"
                />
              </>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4">
                <CardTitle className="text-lg flex items-center gap-2">
                  <FolderKanban className="h-5 w-5 text-muted-foreground" />
                  Recent Projects
                </CardTitle>
                <Link href="/projects">
                  <Button variant="ghost" size="sm" className="gap-1" data-testid="link-view-all-projects">
                    View All
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </CardHeader>
              <CardContent className="pt-0">
                {projectsLoading ? (
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
                ) : recentProjects && recentProjects.length > 0 ? (
                  <div className="space-y-3">
                    {recentProjects.map((project) => (
                      <Link key={project.id} href={`/projects/${project.id}`}>
                        <div
                          className="flex items-center gap-3 p-3 rounded-md hover-elevate cursor-pointer"
                          data-testid={`card-project-${project.id}`}
                        >
                          <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center flex-shrink-0">
                            <FolderKanban className="h-5 w-5 text-primary" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{project.name}</p>
                            <p className="text-sm text-muted-foreground truncate">
                              {project.description || "No description"}
                            </p>
                          </div>
                          {project.diseaseArea && (
                            <DiseaseAreaBadge area={project.diseaseArea} showIcon={false} />
                          )}
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    icon={FolderKanban}
                    title="No projects yet"
                    description="Create your first project to get started"
                    action={
                      <Link href="/projects/new">
                        <Button size="sm" className="gap-2">
                          <Plus className="h-4 w-4" />
                          Create Project
                        </Button>
                      </Link>
                    }
                  />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Workflow className="h-5 w-5 text-muted-foreground" />
                  Recent Campaigns
                </CardTitle>
                <Link href="/campaigns">
                  <Button variant="ghost" size="sm" className="gap-1" data-testid="link-view-all-campaigns">
                    View All
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </CardHeader>
              <CardContent className="pt-0">
                {campaignsLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="flex items-center gap-3">
                        <Skeleton className="h-10 w-10 rounded-md" />
                        <div className="space-y-1.5 flex-1">
                          <Skeleton className="h-4 w-32" />
                          <Skeleton className="h-3 w-48" />
                        </div>
                        <Skeleton className="h-5 w-16 rounded-full" />
                      </div>
                    ))}
                  </div>
                ) : recentCampaigns && recentCampaigns.length > 0 ? (
                  <div className="space-y-3">
                    {recentCampaigns.map((campaign) => (
                      <Link key={campaign.id} href={`/campaigns/${campaign.id}`}>
                        <div
                          className="flex items-center gap-3 p-3 rounded-md hover-elevate cursor-pointer"
                          data-testid={`card-campaign-${campaign.id}`}
                        >
                          <div className="w-10 h-10 rounded-md bg-chart-2/10 flex items-center justify-center flex-shrink-0">
                            <Workflow className="h-5 w-5 text-chart-2" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{campaign.name}</p>
                            <div className="flex items-center gap-2 mt-0.5">
                              {campaign.domainType && (
                                <DiseaseAreaBadge area={campaign.domainType} showIcon={false} />
                              )}
                            </div>
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
                    description="Start a campaign from a project"
                    action={
                      <Link href="/projects">
                        <Button size="sm" variant="outline" className="gap-2">
                          View Projects
                        </Button>
                      </Link>
                    }
                  />
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="text-lg">Domain Distribution</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              {statsLoading ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[1, 2, 3, 4].map((i) => (
                    <Skeleton key={i} className="h-20 rounded-md" />
                  ))}
                </div>
              ) : stats?.domainBreakdown ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(stats.domainBreakdown).map(([domain, count]) => (
                    <div
                      key={domain}
                      className="p-4 rounded-md bg-muted/50 text-center"
                    >
                      <p className="text-2xl font-bold tabular-nums">{count}</p>
                      <DiseaseAreaBadge area={domain as DiseaseArea} className="mt-2" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No campaign data available yet
                </p>
              )}
            </CardContent>
          </Card>
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

function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: typeof FolderKanban;
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
