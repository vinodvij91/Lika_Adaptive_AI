import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Layers, Upload, Play, Box } from "lucide-react";
import { Link } from "wouter";

export default function DockingPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Drug Discovery" }, { label: "Docking & 3D" }]}
      />
      
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="hover-elevate cursor-pointer">
              <CardHeader>
                <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center mb-2">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>Upload Structures</CardTitle>
                <CardDescription>
                  Import protein structures (PDB) and ligand conformers for docking simulations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" data-testid="button-upload-structures">
                  Upload PDB/SDF Files
                </Button>
              </CardContent>
            </Card>

            <Card className="hover-elevate cursor-pointer">
              <CardHeader>
                <div className="w-12 h-12 rounded-md bg-chart-2/10 flex items-center justify-center mb-2">
                  <Play className="h-6 w-6 text-chart-2" />
                </div>
                <CardTitle>Run Docking</CardTitle>
                <CardDescription>
                  Execute molecular docking against target proteins with configurable parameters
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full" data-testid="button-run-docking">
                  Configure Docking Run
                </Button>
              </CardContent>
            </Card>

            <Card className="hover-elevate cursor-pointer">
              <CardHeader>
                <div className="w-12 h-12 rounded-md bg-chart-3/10 flex items-center justify-center mb-2">
                  <Box className="h-6 w-6 text-chart-3" />
                </div>
                <CardTitle>3D Visualization</CardTitle>
                <CardDescription>
                  Interactive 3D viewer for protein-ligand complexes and binding poses
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full" data-testid="button-3d-viewer">
                  Open 3D Viewer
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-muted-foreground" />
                Recent Docking Jobs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Layers className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="font-medium mb-1">No docking jobs yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Upload structures and configure your first docking run
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
