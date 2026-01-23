import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Workflow, Play, Clock, CheckCircle } from "lucide-react";

export default function SimulationRunsPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Materials Science" }, { label: "Simulation Runs" }]}
        actions={
          <Button className="gap-2" data-testid="button-new-simulation">
            <Play className="h-4 w-4" />
            New Simulation
          </Button>
        }
      />
      
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-amber-500/10 flex items-center justify-center">
                    <Clock className="h-6 w-6 text-amber-500" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Running</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-blue-500/10 flex items-center justify-center">
                    <Workflow className="h-6 w-6 text-blue-500" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Queued</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-green-500/10 flex items-center justify-center">
                    <CheckCircle className="h-6 w-6 text-green-500" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Completed</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Workflow className="h-5 w-5 text-muted-foreground" />
                Simulation History
              </CardTitle>
              <CardDescription>
                View and manage molecular dynamics, DFT, and property simulations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Workflow className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="font-medium mb-1">No simulations yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Start a new simulation to analyze material properties
                </p>
                <Button data-testid="button-start-simulation">
                  Start Simulation
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
