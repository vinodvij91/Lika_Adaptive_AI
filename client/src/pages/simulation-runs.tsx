import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Workflow, Play, Clock, CheckCircle, AlertCircle, Plus } from "lucide-react";

export default function SimulationRunsPage() {
  const stats = [
    { label: "Running", value: 0, icon: Clock, color: "text-amber-500", bgColor: "bg-amber-500/10" },
    { label: "Queued", value: 0, icon: Workflow, color: "text-blue-500", bgColor: "bg-blue-500/10" },
    { label: "Completed", value: 0, icon: CheckCircle, color: "text-emerald-500", bgColor: "bg-emerald-500/10" },
    { label: "Failed", value: 0, icon: AlertCircle, color: "text-red-500", bgColor: "bg-red-500/10" },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center">
                <Workflow className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Simulation Runs</h1>
                <p className="text-sm text-muted-foreground">Molecular dynamics, DFT, and property simulations</p>
              </div>
            </div>
            <Button className="gap-2" data-testid="button-new-simulation">
              <Plus className="h-4 w-4" />
              New Simulation
            </Button>
          </header>

          <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="p-5 rounded-xl border bg-card/50"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-8 h-8 rounded-lg ${stat.bgColor} flex items-center justify-center`}>
                    <stat.icon className={`h-4 w-4 ${stat.color}`} />
                  </div>
                </div>
                <p className="text-2xl font-semibold tabular-nums">{stat.value}</p>
                <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
              </div>
            ))}
          </section>

          <section className="space-y-4">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-medium text-muted-foreground">Simulation History</h2>
            </div>
            <Card>
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <Workflow className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium mb-1">No simulations yet</p>
                  <p className="text-xs text-muted-foreground mb-4 max-w-sm mx-auto">
                    Start a new simulation to analyze material properties at molecular scale
                  </p>
                  <Button size="sm" className="h-8 text-xs gap-1.5">
                    <Play className="h-3.5 w-3.5" />
                    Start Simulation
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
