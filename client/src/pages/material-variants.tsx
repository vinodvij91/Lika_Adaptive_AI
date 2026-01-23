import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Layers, Plus, Search, GitBranch } from "lucide-react";
import { Input } from "@/components/ui/input";

export default function MaterialVariantsPage() {
  const stats = [
    { label: "Total Variants", value: 0, color: "text-chart-2", bgColor: "bg-chart-2/10", icon: Layers },
    { label: "Active Explorations", value: 0, color: "text-primary", bgColor: "bg-primary/10", icon: GitBranch },
    { label: "Evaluated", value: 0, color: "text-emerald-500", bgColor: "bg-emerald-500/10", icon: Layers },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                <Layers className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Material Variants</h1>
                <p className="text-sm text-muted-foreground">Explore structural variations and property predictions</p>
              </div>
            </div>
            <Button className="gap-2" data-testid="button-create-variant">
              <Plus className="h-4 w-4" />
              Create Variant
            </Button>
          </header>

          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search variants..." 
              className="pl-9"
              data-testid="input-search-variants"
            />
          </div>

          <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
            <Card>
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <Layers className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium mb-1">No variants yet</p>
                  <p className="text-xs text-muted-foreground mb-4 max-w-sm mx-auto">
                    Create variants from your materials to explore structural modifications
                  </p>
                  <Button size="sm" className="h-8 text-xs gap-1.5">
                    <Plus className="h-3.5 w-3.5" />
                    Create Variant
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
