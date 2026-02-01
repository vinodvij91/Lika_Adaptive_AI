import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Syringe,
  Target,
  ArrowRight,
  Plus,
  Dna,
  Upload,
  Sparkles,
  TrendingUp,
  Zap,
  Shield,
  GitBranch,
} from "lucide-react";

interface VaccineDashboardStats {
  targetsAnalyzed: number;
  epitopesIdentified: number;
  vaccineConstructs: number;
  pipelinesRun: number;
}

export default function VaccineDashboardPage() {
  const { data: stats, isLoading: statsLoading } = useQuery<VaccineDashboardStats>({
    queryKey: ["/api/dashboard/stats/vaccine"],
  });

  const { data: targets } = useQuery<any[]>({
    queryKey: ["/api/targets"],
  });

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-purple-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-purple-600 via-violet-500 to-indigo-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMtOS45NDEgMC0xOCA4LjA1OS0xOCAxOHM4LjA1OSAxOCAxOCAxOCAxOC04LjA1OSAxOC0xOC04LjA1OS0xOC0xOC0xOHptMCAzMmMtNy43MzIgMC0xNC02LjI2OC0xNC0xNHM2LjI2OC0xNCAxNC0xNCAxNCA2LjI2OCAxNCAxNC02LjI2OCAxNC0xNCAxNHoiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4xKSIvPjwvZz48L3N2Zz4=')] opacity-30" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Syringe className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">Vaccine Discovery</h1>
                  <p className="text-purple-100">Design next-generation immunotherapies and vaccines</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-purple-100">
                <Zap className="h-4 w-4" />
                <span>AI-powered epitope prediction and vaccine construct design</span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-purple-500 to-purple-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Target className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.targetsAnalyzed ?? 0).toLocaleString()}
                </p>
                <p className="text-purple-100 text-sm font-medium">Targets Analyzed</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-indigo-500 to-indigo-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Shield className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.epitopesIdentified ?? 0).toLocaleString()}
                </p>
                <p className="text-indigo-100 text-sm font-medium">Epitopes Identified</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-violet-500 to-violet-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <Dna className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.vaccineConstructs ?? 0).toLocaleString()}
                </p>
                <p className="text-violet-100 text-sm font-medium">Vaccine Constructs</p>
              </CardContent>
            </Card>

            <Card className="relative overflow-hidden border-0 shadow-lg bg-gradient-to-br from-fuchsia-500 to-fuchsia-600 text-white">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <GitBranch className="h-8 w-8 opacity-80" />
                  <TrendingUp className="h-5 w-5 opacity-60" />
                </div>
                <p className="text-4xl font-bold mb-1">
                  {statsLoading ? "—" : (stats?.pipelinesRun ?? 0).toLocaleString()}
                </p>
                <p className="text-fuchsia-100 text-sm font-medium">Pipelines Run</p>
              </CardContent>
            </Card>
          </div>

          <Card className="border-2 border-dashed border-purple-500/30 bg-purple-500/5">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                  <Sparkles className="h-5 w-5 text-purple-500" />
                </div>
                <div>
                  <h2 className="font-semibold text-lg">Get Started</h2>
                  <p className="text-sm text-muted-foreground">Choose an action to begin your vaccine discovery workflow</p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Link href="/targets">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-purple-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4 group-hover:bg-purple-500/20 transition-colors">
                      <Target className="h-6 w-6 text-purple-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Add Targets</h3>
                    <p className="text-sm text-muted-foreground">Import protein sequences for antigen analysis</p>
                  </div>
                </Link>
                <Link href="/vaccine-discovery">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-indigo-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-indigo-500/10 flex items-center justify-center mb-4 group-hover:bg-indigo-500/20 transition-colors">
                      <Syringe className="h-6 w-6 text-indigo-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Run Pipeline</h3>
                    <p className="text-sm text-muted-foreground">Predict epitopes and design vaccine constructs</p>
                  </div>
                </Link>
                <Link href="/trajectory-analysis">
                  <div className="group p-5 rounded-xl bg-background border-2 border-transparent hover:border-violet-500/50 hover:shadow-lg transition-all cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4 group-hover:bg-violet-500/20 transition-colors">
                      <GitBranch className="h-6 w-6 text-violet-500" />
                    </div>
                    <h3 className="font-semibold mb-1">Trajectory Analysis</h3>
                    <p className="text-sm text-muted-foreground">Identify targets from scRNA-seq data</p>
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
                    <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center">
                      <Target className="h-5 w-5 text-purple-500" />
                    </div>
                    <h2 className="font-semibold">Protein Targets</h2>
                  </div>
                  <Link href="/targets">
                    <Button variant="ghost" size="sm" className="gap-1">
                      View all <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                {targets && targets.length > 0 ? (
                  <div className="divide-y">
                    {targets.slice(0, 4).map((target: any) => (
                      <Link key={target.id} href={`/targets/${target.id}`}>
                        <div className="flex items-center gap-4 p-4 hover:bg-muted/50 transition-colors cursor-pointer" data-testid={`card-target-${target.id}`}>
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-indigo-500/20 flex items-center justify-center">
                            <Target className="h-5 w-5 text-purple-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <p className="font-medium truncate">{target.name}</p>
                              {target.isDemo && (
                                <Badge variant="outline" className="text-xs bg-amber-500/10 text-amber-600 border-amber-500/30">
                                  Demo
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground truncate">
                              {target.organism || target.species || "Unknown organism"}
                            </p>
                          </div>
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="p-10 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500/10 to-indigo-500/10 flex items-center justify-center mx-auto mb-4">
                      <Target className="h-7 w-7 text-purple-500" />
                    </div>
                    <p className="font-semibold mb-1">No targets yet</p>
                    <p className="text-sm text-muted-foreground mb-5">Import protein sequences to begin analysis</p>
                    <Link href="/import">
                      <Button className="gap-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600">
                        <Plus className="h-4 w-4" />
                        Import Targets
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
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center">
                      <Syringe className="h-5 w-5 text-indigo-500" />
                    </div>
                    <h2 className="font-semibold">Vaccine Pipelines</h2>
                  </div>
                  <Link href="/vaccine-discovery">
                    <Button variant="ghost" size="sm" className="gap-1">
                      View all <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                <div className="p-10 text-center">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/10 to-violet-500/10 flex items-center justify-center mx-auto mb-4">
                    <Syringe className="h-7 w-7 text-indigo-500" />
                  </div>
                  <p className="font-semibold mb-1">Ready to design vaccines</p>
                  <p className="text-sm text-muted-foreground mb-5">Run AI-powered epitope prediction and vaccine design</p>
                  <Link href="/vaccine-discovery">
                    <Button variant="outline" className="gap-2">
                      Start Pipeline
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
