import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
  TrendingUp,
  Zap,
  Beaker,
} from "lucide-react";
import type { MaterialsCampaign, MaterialEntity } from "@shared/schema";

interface MaterialsDashboardStats {
  materialVariantsEvaluated: number;
  propertiesPredicted: number;
  manufacturableCandidates: number;
  activePipelines: number;
}

export default function MaterialsDashboardPage() {
  const { data: stats, isLoading: statsLoading } = useQuery<MaterialsDashboardStats>({
    queryKey: ["/api/dashboard/stats/materials"],
  });

  const { data: materialsResponse } = useQuery<{ materials: MaterialEntity[], total: number }>({
    queryKey: ["/api/materials"],
  });
  const recentMaterials = materialsResponse?.materials;

  const { data: recentCampaigns } = useQuery<MaterialsCampaign[]>({
    queryKey: ["/api/materials-campaigns"],
  });

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-emerald-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-600 via-teal-500 to-cyan-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxNWwtMTMuMDkgNy41djE1TDMwIDQ1bDEzLjA5LTcuNXYtMTVMMzAgMTV6IiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4xNSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Hexagon className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">Materials Science</h1>
                  <p className="text-emerald-100">Discover next-generation advanced materials</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-emerald-100">
                <Zap className="h-4 w-4" />
                <span>Property-first discovery with manufacturability scoring</span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-emerald-500 to-teal-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Hexagon className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.materialVariantsEvaluated ?? 0).toLocaleString()}
                </p>
                <p className="text-emerald-100 text-sm font-medium">Variants Evaluated</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-blue-500 to-cyan-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Calculator className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.propertiesPredicted ?? 0).toLocaleString()}
                </p>
                <p className="text-blue-100 text-sm font-medium">Properties Predicted</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-violet-500 to-purple-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Factory className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.manufacturableCandidates ?? 0).toLocaleString()}
                </p>
                <p className="text-violet-100 text-sm font-medium">Manufacturable</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-amber-500 to-orange-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Workflow className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.activePipelines ?? 0).toLocaleString()}
                </p>
                <p className="text-amber-100 text-sm font-medium">Active Pipelines</p>
              </CardContent>
            </Card>
          </div>

          <Card className="border-2 border-dashed border-emerald-500/30 bg-emerald-500/5">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                  <Sparkles className="h-5 w-5 text-emerald-500" />
                </div>
                <div>
                  <h2 className="font-semibold text-lg">Get Started</h2>
                  <p className="text-sm text-muted-foreground">Choose an action to begin your discovery workflow</p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Link href="/import/materials/materials">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-emerald-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-4 group-hover:bg-emerald-500/20 transition-colors">
                      <Upload className="h-6 w-6 text-emerald-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Import Materials</h3>
                    <p className="text-sm text-muted-foreground">Upload polymers, crystals, or composites data</p>
                  </div>
                </Link>
                <Link href="/property-prediction">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-blue-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4 group-hover:bg-blue-500/20 transition-colors">
                      <Calculator className="h-6 w-6 text-blue-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Property Pipeline</h3>
                    <p className="text-sm text-muted-foreground">Run property predictions on your materials</p>
                  </div>
                </Link>
                <Link href="/manufacturability-scoring">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-violet-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4 group-hover:bg-violet-500/20 transition-colors">
                      <Factory className="h-6 w-6 text-violet-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Manufacturability</h3>
                    <p className="text-sm text-muted-foreground">Assess production feasibility of candidates</p>
                  </div>
                </Link>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-lg">
              <CardContent className="p-0">
                <div className="flex items-center justify-between p-5 border-b">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                      <Layers className="h-5 w-5 text-emerald-500" />
                    </div>
                    <h2 className="font-semibold">Materials Library</h2>
                  </div>
                  <Link href="/materials-library">
                    <Button variant="ghost" size="sm" className="gap-1" data-testid="link-view-all-materials">
                      View all <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                {recentMaterials && recentMaterials.length > 0 ? (
                  <div className="divide-y">
                    {recentMaterials.slice(0, 5).map((material) => (
                      <div key={material.id} className="flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors" data-testid={`card-material-${material.id}`}>
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 flex items-center justify-center">
                          <Beaker className="h-5 w-5 text-emerald-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="font-medium truncate">{material.name}</p>
                            {material.isDemo && (
                              <Badge variant="outline" className="text-xs bg-amber-500/10 text-amber-600 border-amber-500/30">
                                Demo
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground truncate capitalize">
                            {material.type} - {material.baseFamily || "Unknown family"}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-10 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-4">
                      <Hexagon className="h-7 w-7 text-emerald-500" />
                    </div>
                    <p className="font-semibold mb-1">No materials yet</p>
                    <p className="text-sm text-muted-foreground mb-5">Import materials to start your discovery</p>
                    <Link href="/import/materials/materials">
                      <Button className="gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600" data-testid="button-import-materials">
                        <Plus className="h-4 w-4" />
                        Import Materials
                      </Button>
                    </Link>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="shadow-lg">
              <CardContent className="p-0">
                <div className="flex items-center justify-between p-5 border-b">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
                      <Workflow className="h-5 w-5 text-amber-500" />
                    </div>
                    <h2 className="font-semibold">Active Pipelines</h2>
                  </div>
                  <Link href="/simulation-runs">
                    <Button variant="ghost" size="sm" className="gap-1">
                      View all <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                {recentCampaigns && recentCampaigns.length > 0 ? (
                  <div className="divide-y">
                    {recentCampaigns.slice(0, 4).map((campaign) => (
                      <Link key={campaign.id} href={`/materials-campaigns/${campaign.id}`}>
                        <div className="flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors cursor-pointer">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center">
                            <Workflow className="h-5 w-5 text-amber-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{campaign.name}</p>
                            <p className="text-sm text-muted-foreground truncate">
                              {campaign.modality || "Materials"}
                            </p>
                          </div>
                          {campaign.status && (
                            <span className="text-xs px-3 py-1 rounded-full bg-amber-500/10 text-amber-600 font-medium">
                              {campaign.status}
                            </span>
                          )}
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="p-10 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 flex items-center justify-center mx-auto mb-4">
                      <Workflow className="h-7 w-7 text-amber-500" />
                    </div>
                    <p className="font-semibold mb-1">No pipelines yet</p>
                    <p className="text-sm text-muted-foreground mb-5">Start a simulation to analyze materials</p>
                    <Link href="/simulation-runs">
                      <Button variant="outline" className="gap-2">
                        Start Simulation
                      </Button>
                    </Link>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
