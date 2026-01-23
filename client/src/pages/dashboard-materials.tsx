import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Hexagon,
  Workflow,
  ArrowRight,
  Plus,
  Layers,
  Calculator,
  Factory,
  Upload,
  Sparkles,
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

  const { data: recentCampaigns } = useQuery<MaterialsCampaign[]>({
    queryKey: ["/api/materials-campaigns", { limit: 5 }],
  });

  const metrics = [
    { label: "Variants Evaluated", value: stats?.materialVariantsEvaluated ?? 0, icon: Hexagon },
    { label: "Properties Predicted", value: stats?.propertiesPredicted ?? 0, icon: Calculator },
    { label: "Manufacturable", value: stats?.manufacturableCandidates ?? 0, icon: Factory },
    { label: "Active Pipelines", value: stats?.activePipelines ?? 0, icon: Workflow },
  ];

  const quickActions = [
    {
      title: "Import Materials",
      description: "Upload polymers, crystals, or composites",
      icon: Upload,
      href: "/import/materials/materials",
    },
    {
      title: "Property Pipeline",
      description: "Run property predictions",
      icon: Calculator,
      href: "/property-prediction",
    },
    {
      title: "Manufacturability",
      description: "Assess production feasibility",
      icon: Factory,
      href: "/manufacturability-scoring",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                <Hexagon className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Materials Science</h1>
                <p className="text-sm text-muted-foreground">Welcome back. Here's your discovery overview.</p>
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
                  <div className="w-8 h-8 rounded-lg bg-chart-2/10 flex items-center justify-center">
                    <metric.icon className="h-4 w-4 text-chart-2" />
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
                  <Card className="group cursor-pointer border-dashed hover:border-solid hover:border-chart-2/30 transition-all h-full">
                    <CardContent className="p-5 flex items-start gap-4">
                      <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center group-hover:bg-chart-2/10 transition-colors flex-shrink-0">
                        <action.icon className="h-5 w-5 text-muted-foreground group-hover:text-chart-2 transition-colors" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm">{action.title}</p>
                        <p className="text-xs text-muted-foreground mt-0.5">{action.description}</p>
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground/50 group-hover:text-chart-2 group-hover:translate-x-0.5 transition-all flex-shrink-0 mt-1" />
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
                  <Layers className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-medium text-muted-foreground">Materials Library</h2>
                </div>
                <Link href="/materials-library">
                  <Button variant="ghost" size="sm" className="text-xs h-7 gap-1">
                    View all <ArrowRight className="h-3 w-3" />
                  </Button>
                </Link>
              </div>
              <Card>
                <CardContent className="p-0">
                  <div className="p-8 text-center">
                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
                      <Hexagon className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <p className="text-sm font-medium mb-1">No materials yet</p>
                    <p className="text-xs text-muted-foreground mb-4">Import materials to start your discovery</p>
                    <Link href="/import/materials/materials">
                      <Button size="sm" className="h-8 text-xs gap-1.5">
                        <Plus className="h-3.5 w-3.5" />
                        Import Materials
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </section>

            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Workflow className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-medium text-muted-foreground">Active Pipelines</h2>
                </div>
                <Link href="/simulation-runs">
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
                        <Link key={campaign.id} href={`/materials-campaigns/${campaign.id}`}>
                          <div className="flex items-center gap-3 p-4 hover:bg-muted/50 transition-colors cursor-pointer">
                            <div className="w-8 h-8 rounded-lg bg-chart-2/10 flex items-center justify-center flex-shrink-0">
                              <Workflow className="h-4 w-4 text-chart-2" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">{campaign.name}</p>
                              <p className="text-xs text-muted-foreground truncate">
                                {campaign.modality || "Materials"}
                              </p>
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
                      <p className="text-sm font-medium mb-1">No pipelines yet</p>
                      <p className="text-xs text-muted-foreground mb-4">Start a simulation to analyze materials</p>
                      <Link href="/simulation-runs">
                        <Button size="sm" variant="outline" className="h-8 text-xs">
                          Start Simulation
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
