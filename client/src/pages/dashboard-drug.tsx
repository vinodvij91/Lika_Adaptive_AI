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
  TrendingUp,
  Zap,
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

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-blue-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMtOS45NDEgMC0xOCA4LjA1OS0xOCAxOHM4LjA1OSAxOCAxOCAxOCAxOC04LjA1OSAxOC0xOC04LjA1OS0xOC0xOC0xOHptMCAzMmMtNy43MzIgMC0xNC02LjI2OC0xNC0xNHM2LjI2OC0xNCAxNC0xNCAxNCA2LjI2OCAxNCAxNC02LjI2OCAxNC0xNCAxNHoiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4xKSIvPjwvZz48L3N2Zz4=')] opacity-30" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <FlaskConical className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">Drug Discovery</h1>
                  <p className="text-blue-100">Accelerate your therapeutic development pipeline</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-blue-100">
                <Zap className="h-4 w-4" />
                <span>AI-powered virtual screening and lead optimization</span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-blue-500 to-blue-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <FlaskConical className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.moleculesScreened ?? 0).toLocaleString()}
                </p>
                <p className="text-blue-100 text-sm font-medium">Molecules Screened</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-emerald-500 to-teal-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Crosshair className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.hitsIdentified ?? 0).toLocaleString()}
                </p>
                <p className="text-emerald-100 text-sm font-medium">Hits Identified</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-violet-500 to-purple-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Target className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.activeTargets ?? 0).toLocaleString()}
                </p>
                <p className="text-violet-100 text-sm font-medium">Active Targets</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-amber-500 to-orange-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <TestTube2 className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.assaysUploaded ?? 0).toLocaleString()}
                </p>
                <p className="text-amber-100 text-sm font-medium">Assays Uploaded</p>
              </CardContent>
            </Card>
          </div>

          <Card className="border-2 border-dashed border-primary/30 bg-primary/5">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
                  <Sparkles className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h2 className="font-semibold text-lg">Get Started</h2>
                  <p className="text-sm text-muted-foreground">Choose an action to begin your discovery workflow</p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Link href="/import/drug/molecules">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-blue-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4 group-hover:bg-blue-500/20 transition-colors">
                      <Upload className="h-6 w-6 text-blue-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Import Molecules</h3>
                    <p className="text-sm text-muted-foreground">Upload SMILES, SDF, or CSV files to build your library</p>
                  </div>
                </Link>
                <Link href="/campaigns/new">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-emerald-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-4 group-hover:bg-emerald-500/20 transition-colors">
                      <Workflow className="h-6 w-6 text-emerald-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Start Campaign</h3>
                    <p className="text-sm text-muted-foreground">Run virtual screening against your targets</p>
                  </div>
                </Link>
                <Link href="/assays">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-violet-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4 group-hover:bg-violet-500/20 transition-colors">
                      <TestTube2 className="h-6 w-6 text-violet-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Upload Assay Data</h3>
                    <p className="text-sm text-muted-foreground">Import experimental results for SAR analysis</p>
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
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                      <FolderKanban className="h-5 w-5 text-primary" />
                    </div>
                    <h2 className="font-semibold">Recent Projects</h2>
                  </div>
                  <Link href="/projects">
                    <Button variant="ghost" size="sm" className="gap-1">
                      View all <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                {recentProjects && recentProjects.length > 0 ? (
                  <div className="divide-y">
                    {recentProjects.slice(0, 4).map((project) => (
                      <Link key={project.id} href={`/projects/${project.id}`}>
                        <div className="flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors cursor-pointer">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center">
                            <FolderKanban className="h-5 w-5 text-blue-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{project.name}</p>
                            <p className="text-sm text-muted-foreground truncate">
                              {project.description || "No description"}
                            </p>
                          </div>
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="p-10 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/10 to-cyan-500/10 flex items-center justify-center mx-auto mb-4">
                      <FolderKanban className="h-7 w-7 text-blue-500" />
                    </div>
                    <p className="font-semibold mb-1">No projects yet</p>
                    <p className="text-sm text-muted-foreground mb-5">Create your first project to organize your research</p>
                    <Link href="/projects/new">
                      <Button className="gap-2 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600">
                        <Plus className="h-4 w-4" />
                        Create Project
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
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                      <Workflow className="h-5 w-5 text-emerald-500" />
                    </div>
                    <h2 className="font-semibold">Active Campaigns</h2>
                  </div>
                  <Link href="/campaigns">
                    <Button variant="ghost" size="sm" className="gap-1">
                      View all <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                {recentCampaigns && recentCampaigns.length > 0 ? (
                  <div className="divide-y">
                    {recentCampaigns.slice(0, 4).map((campaign) => (
                      <Link key={campaign.id} href={`/campaigns/${campaign.id}`}>
                        <div className="flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors cursor-pointer">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 flex items-center justify-center">
                            <Workflow className="h-5 w-5 text-emerald-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{campaign.name}</p>
                          </div>
                          {campaign.status && (
                            <span className="text-xs px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-600 font-medium">
                              {campaign.status}
                            </span>
                          )}
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="p-10 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-4">
                      <Workflow className="h-7 w-7 text-emerald-500" />
                    </div>
                    <p className="font-semibold mb-1">No campaigns yet</p>
                    <p className="text-sm text-muted-foreground mb-5">Start a campaign to run virtual screening</p>
                    <Link href="/campaigns/new">
                      <Button variant="outline" className="gap-2">
                        Start Campaign
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
