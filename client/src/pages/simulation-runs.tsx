import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Workflow, Play, Clock, CheckCircle, AlertCircle, Plus, Zap, Cpu } from "lucide-react";

export default function SimulationRunsPage() {
  const stats = [
    { label: "Running", value: 0, icon: Clock, color: "from-amber-500 to-orange-500", bgColor: "bg-amber-500", textColor: "text-amber-500" },
    { label: "Queued", value: 0, icon: Workflow, color: "from-blue-500 to-cyan-500", bgColor: "bg-blue-500", textColor: "text-blue-500" },
    { label: "Completed", value: 0, icon: CheckCircle, color: "from-emerald-500 to-teal-500", bgColor: "bg-emerald-500", textColor: "text-emerald-500" },
    { label: "Failed", value: 0, icon: AlertCircle, color: "from-red-500 to-rose-500", bgColor: "bg-red-500", textColor: "text-red-500" },
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-teal-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-teal-600 via-cyan-500 to-blue-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0xMCAxMGg0MHY0MEgxMHoiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIiBzdHJva2Utd2lkdGg9IjIiLz48L2c+PC9zdmc+')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Workflow className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">Simulation Runs</h1>
                    <p className="text-teal-100">Molecular dynamics, DFT, and property simulations</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-teal-100">
                  <Cpu className="h-4 w-4" />
                  <span>Distributed computing across GPU clusters</span>
                </div>
              </div>
              <Button className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0">
                <Plus className="h-4 w-4" />
                New Simulation
              </Button>
            </div>
          </header>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            {stats.map((stat) => (
              <Card key={stat.label} className="border-0 shadow-lg overflow-hidden">
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${stat.color}`} />
                  <div className="p-6">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl ${stat.bgColor}/10 flex items-center justify-center`}>
                        <stat.icon className={`h-6 w-6 ${stat.textColor}`} />
                      </div>
                      <div>
                        <p className="text-3xl font-bold">{stat.value}</p>
                        <p className="text-sm text-muted-foreground">{stat.label}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="shadow-lg">
            <CardContent className="p-0">
              <div className="flex items-center gap-3 p-5 border-b">
                <div className="w-10 h-10 rounded-xl bg-teal-500/10 flex items-center justify-center">
                  <Clock className="h-5 w-5 text-teal-500" />
                </div>
                <h2 className="font-semibold">Simulation History</h2>
              </div>
              <div className="p-12 text-center">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-teal-500/10 to-cyan-500/10 flex items-center justify-center mx-auto mb-5">
                  <Workflow className="h-9 w-9 text-teal-500" />
                </div>
                <p className="font-semibold text-lg mb-2">No simulations yet</p>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Start a new simulation to analyze material properties at molecular scale
                </p>
                <Button className="gap-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600">
                  <Play className="h-4 w-4" />
                  Start Simulation
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
