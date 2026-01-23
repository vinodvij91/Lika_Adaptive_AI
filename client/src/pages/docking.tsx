import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Layers, Upload, Play, Box, ArrowRight, Clock } from "lucide-react";
import { Link } from "wouter";

export default function DockingPage() {
  const features = [
    {
      title: "Upload Structures",
      description: "Import protein structures (PDB) and ligand conformers for docking simulations",
      icon: Upload,
      action: "Upload Files",
    },
    {
      title: "Run Docking",
      description: "Execute molecular docking against target proteins with configurable parameters",
      icon: Play,
      action: "Configure Run",
    },
    {
      title: "3D Visualization",
      description: "Interactive 3D viewer for protein-ligand complexes and binding poses",
      icon: Box,
      action: "Open Viewer",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                <Layers className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Docking & 3D</h1>
                <p className="text-sm text-muted-foreground">Molecular docking and 3D structure visualization</p>
              </div>
            </div>
          </header>

          <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {features.map((feature) => (
              <Card key={feature.title} className="group cursor-pointer hover:border-primary/30 transition-all">
                <CardContent className="p-6 space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                    <feature.icon className="h-6 w-6 text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>
                  <div>
                    <h3 className="font-medium mb-1">{feature.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
                  </div>
                  <Button variant="outline" className="w-full gap-2 group-hover:border-primary/30">
                    {feature.action}
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </section>

          <section className="space-y-4">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-medium text-muted-foreground">Recent Jobs</h2>
            </div>
            <Card>
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <Layers className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium mb-1">No docking jobs yet</p>
                  <p className="text-xs text-muted-foreground mb-4 max-w-sm mx-auto">
                    Upload protein structures and ligands to configure your first docking run
                  </p>
                  <Button size="sm" className="h-8 text-xs gap-1.5">
                    <Upload className="h-3.5 w-3.5" />
                    Upload Structures
                  </Button>
                </div>
              </CardContent>
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}
