import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  FlaskConical,
  Workflow,
  Target,
  ArrowRight,
  Plus,
  FolderKanban,
  TestTube2,
  Crosshair,
  Upload,
  Sparkles,
} from "lucide-react";
import type { Project, Campaign } from "@shared/schema";

interface DrugDashboardStats {
  moleculesScreened: number;
  hitsIdentified: number;
  activeTargets: number;
  assaysUploaded: number;
  activeCampaigns: number;
}

export default function DrugDashboardPage() {
  const { data: stats, isLoading: statsLoading } = useQuery<DrugDashboardStats>({
    queryKey: ["/api/dashboard/stats", "drug"],
  });

  const { data: recentProjects } = useQuery<Project[]>({
    queryKey: ["/api/projects", { limit: 5, domain: "drug" }],
  });

  const { data: recentCampaigns } = useQuery<Campaign[]>({
    queryKey: ["/api/campaigns", { limit: 5, domain: "drug" }],
  });

  const metrics = [
    { label: "Molecules Screened", value: stats?.moleculesScreened ?? 0, icon: FlaskConical },
    { label: "Hits Identified", value: stats?.hitsIdentified ?? 0, icon: Crosshair },
    { label: "Active Targets", value: stats?.activeTargets ?? 0, icon: Target },
    { label: "Assays Uploaded", value: stats?.assaysUploaded ?? 0, icon: TestTube2 },
  ];

  const quickActions = [
    {
      title: "Import Molecules",
      description: "Upload SMILES, SDF, or CSV files",
      icon: Upload,
      href: "/import/drug/molecules",
    },
    {
      title: "Start Campaign",
      description: "Run virtual screening",
      icon: Workflow,
      href: "/campaigns/new",
    },
    {
      title: "Upload Assay Data",
      description: "Import experimental results",
      icon: TestTube2,
      href: "/assays",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                <FlaskConical className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Drug Discovery</h1>
                <p className="text-sm text-muted-foreground">Welcome back. Here's your research overview.</p>
              </div>
            </div>
          </header>

          <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metrics.map((metric) => (
              <div
                key={metric.label}
                className="group p-5 rounded-xl border bg-card/50 hover:bg-card transition-colors"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                    <metric.icon className="h-4 w-4 text-primary" />
                  </div>
                </div>
                <p className="text-2xl font-semibold tabular-nums">
                  {statsLoading ? "—" : metric.value.toLocaleString()}
                </p>
                <p className="text-xs text-muted-foreground mt-1">{metric.label}</p>
              </div>
            ))}
          </section>

          <section className="space-y-4">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-medium text-muted-foreground">Quick Actions</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {quickActions.map((action) => (
                <Link key={action.title} href={action.href}>
                  <Card className="group cursor-pointer border-dashed hover:border-solid hover:border-primary/30 transition-all h-full">
                    <CardContent className="p-5 flex items-start gap-4">
                      <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center group-hover:bg-primary/10 transition-colors flex-shrink-0">
                        <action.icon className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm">{action.title}</p>
                        <p className="text-xs text-muted-foreground mt-0.5">{action.description}</p>
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground/50 group-hover:text-primary group-hover:translate-x-0.5 transition-all flex-shrink-0 mt-1" />
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FolderKanban className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-medium text-muted-foreground">Recent Projects</h2>
                </div>
                <Link href="/projects">
                  <Button variant="ghost" size="sm" className="text-xs h-7 gap-1">
                    View all <ArrowRight className="h-3 w-3" />
                  </Button>
                </Link>
              </div>
              <Card>
                <CardContent className="p-0">
                  {recentProjects && recentProjects.length > 0 ? (
                    <div className="divide-y">
                      {recentProjects.slice(0, 4).map((project) => (
                        <Link key={project.id} href={`/projects/${project.id}`}>
                          <div className="flex items-center gap-3 p-4 hover:bg-muted/50 transition-colors cursor-pointer">
                            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                              <FolderKanban className="h-4 w-4 text-primary" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">{project.name}</p>
                              <p className="text-xs text-muted-foreground truncate">
                                {project.description || "No description"}
                              </p>
                            </div>
                          </div>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <div className="p-8 text-center">
                      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
                        <FolderKanban className="h-5 w-5 text-muted-foreground" />
                      </div>
                      <p className="text-sm font-medium mb-1">No projects yet</p>
                      <p className="text-xs text-muted-foreground mb-4">Create your first project to get started</p>
                      <Link href="/projects/new">
                        <Button size="sm" className="h-8 text-xs gap-1.5">
                          <Plus className="h-3.5 w-3.5" />
                          Create Project
                        </Button>
                      </Link>
                    </div>
                  )}
                </CardContent>
              </Card>
            </section>

            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Workflow className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-medium text-muted-foreground">Active Campaigns</h2>
                </div>
                <Link href="/campaigns">
                  <Button variant="ghost" size="sm" className="text-xs h-7 gap-1">
                    View all <ArrowRight className="h-3 w-3" />
                  </Button>
                </Link>
              </div>
              <Card>
                <CardContent className="p-0">
                  {recentCampaigns && recentCampaigns.length > 0 ? (
                    <div className="divide-y">
                      {recentCampaigns.slice(0, 4).map((campaign) => (
                        <Link key={campaign.id} href={`/campaigns/${campaign.id}`}>
                          <div className="flex items-center gap-3 p-4 hover:bg-muted/50 transition-colors cursor-pointer">
                            <div className="w-8 h-8 rounded-lg bg-chart-2/10 flex items-center justify-center flex-shrink-0">
                              <Workflow className="h-4 w-4 text-chart-2" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">{campaign.name}</p>
                            </div>
                            {campaign.status && (
                              <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                                {campaign.status}
                              </span>
                            )}
                          </div>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <div className="p-8 text-center">
                      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
                        <Workflow className="h-5 w-5 text-muted-foreground" />
                      </div>
                      <p className="text-sm font-medium mb-1">No campaigns yet</p>
                      <p className="text-xs text-muted-foreground mb-4">Start a campaign to run virtual screening</p>
                      <Link href="/campaigns/new">
                        <Button size="sm" variant="outline" className="h-8 text-xs">
                          Start Campaign
                        </Button>
                      </Link>
                    </div>
                  )}
                </CardContent>
              </Card>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
