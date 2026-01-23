import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Layers, Upload, Play, Box, ArrowRight, Clock, Zap, Database } from "lucide-react";

export default function DockingPage() {
  const features = [
    {
      title: "Upload Structures",
      description: "Import protein structures (PDB) and ligand conformers for docking simulations",
      icon: Upload,
      color: "from-blue-500 to-cyan-500",
      bgColor: "bg-blue-500/10",
      textColor: "text-blue-500",
    },
    {
      title: "Run Docking",
      description: "Execute molecular docking against target proteins with configurable parameters",
      icon: Play,
      color: "from-emerald-500 to-teal-500",
      bgColor: "bg-emerald-500/10",
      textColor: "text-emerald-500",
    },
    {
      title: "3D Visualization",
      description: "Interactive 3D viewer for protein-ligand complexes and binding poses",
      icon: Box,
      color: "from-violet-500 to-purple-500",
      bgColor: "bg-violet-500/10",
      textColor: "text-violet-500",
    },
  ];

  const stats = [
    { label: "Structures Uploaded", value: 0, icon: Database, color: "text-blue-500", bgColor: "bg-blue-500/10" },
    { label: "Docking Jobs", value: 0, icon: Play, color: "text-emerald-500", bgColor: "bg-emerald-500/10" },
    { label: "Poses Generated", value: 0, icon: Box, color: "text-violet-500", bgColor: "bg-violet-500/10" },
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-violet-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxMGwyMCAxMHYyMEwzMCA1MCAxMCA0MFYyMEwzMCAxMHoiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjE1KSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9nPjwvc3ZnPg==')] opacity-40" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Layers className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">Docking & 3D</h1>
                  <p className="text-violet-100">Molecular docking and structure visualization</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-violet-100">
                <Zap className="h-4 w-4" />
                <span>High-throughput virtual screening with GPU acceleration</span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {stats.map((stat) => (
              <Card key={stat.label} className="border-0 shadow-lg">
                <CardContent className="p-6">
                  <div className="flex items-center gap-4">
                    <div className={`w-14 h-14 rounded-2xl ${stat.bgColor} flex items-center justify-center`}>
                      <stat.icon className={`h-7 w-7 ${stat.color}`} />
                    </div>
                    <div>
                      <p className="text-3xl font-bold">{stat.value}</p>
                      <p className="text-sm text-muted-foreground">{stat.label}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {features.map((feature) => (
              <Card key={feature.title} className="group cursor-pointer border-2 border-transparent hover:border-violet-500/30 hover:shadow-xl transition-all overflow-hidden">
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${feature.color}`} />
                  <div className="p-6">
                    <div className={`w-14 h-14 rounded-2xl ${feature.bgColor} flex items-center justify-center mb-5 group-hover:scale-110 transition-transform`}>
                      <feature.icon className={`h-7 w-7 ${feature.textColor}`} />
                    </div>
                    <h3 className="font-semibold text-lg mb-2">{feature.title}</h3>
                    <p className="text-sm text-muted-foreground mb-5">{feature.description}</p>
                    <Button className={`w-full gap-2 bg-gradient-to-r ${feature.color} border-0`}>
                      Get Started
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="shadow-lg">
            <CardContent className="p-0">
              <div className="flex items-center gap-3 p-5 border-b">
                <div className="w-10 h-10 rounded-xl bg-violet-500/10 flex items-center justify-center">
                  <Clock className="h-5 w-5 text-violet-500" />
                </div>
                <h2 className="font-semibold">Recent Docking Jobs</h2>
              </div>
              <div className="p-12 text-center">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-500/10 to-purple-500/10 flex items-center justify-center mx-auto mb-5">
                  <Layers className="h-9 w-9 text-violet-500" />
                </div>
                <p className="font-semibold text-lg mb-2">No docking jobs yet</p>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Upload protein structures and ligands to configure your first docking run
                </p>
                <Button className="gap-2 bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600">
                  <Upload className="h-4 w-4" />
                  Upload Structures
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
