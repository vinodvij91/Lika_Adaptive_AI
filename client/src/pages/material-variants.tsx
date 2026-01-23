import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Layers, Plus, Search, GitBranch, Zap, Sparkles, FlaskConical } from "lucide-react";
import { Input } from "@/components/ui/input";

export default function MaterialVariantsPage() {
  const stats = [
    { label: "Total Variants", value: 0, icon: Layers, color: "from-violet-500 to-purple-500", bgColor: "bg-violet-500" },
    { label: "Active Explorations", value: 0, icon: GitBranch, color: "from-blue-500 to-cyan-500", bgColor: "bg-blue-500" },
    { label: "Evaluated", value: 0, icon: Sparkles, color: "from-emerald-500 to-teal-500", bgColor: "bg-emerald-500" },
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-violet-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0xMCAzMGgyME0zMCAxMHYyMCIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Layers className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">Material Variants</h1>
                    <p className="text-violet-100">Explore structural variations and property predictions</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-violet-100">
                  <Zap className="h-4 w-4" />
                  <span>AI-guided structural exploration and optimization</span>
                </div>
              </div>
              <Button className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0">
                <Plus className="h-4 w-4" />
                Create Variant
              </Button>
            </div>
          </header>

          <div className="relative max-w-xl">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input 
              placeholder="Search variants by parent material or properties..." 
              className="pl-12 h-12 text-base"
              data-testid="input-search-variants"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {stats.map((stat) => (
              <Card key={stat.label} className="border-0 shadow-lg overflow-hidden">
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${stat.color}`} />
                  <div className="p-6">
                    <div className="flex items-center gap-4">
                      <div className={`w-14 h-14 rounded-xl ${stat.bgColor}/10 flex items-center justify-center`}>
                        <stat.icon className={`h-7 w-7 ${stat.bgColor.replace('bg-', 'text-')}`} />
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
              <div className="p-12 text-center">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-500/10 to-purple-500/10 flex items-center justify-center mx-auto mb-5">
                  <Layers className="h-9 w-9 text-violet-500" />
                </div>
                <p className="font-semibold text-lg mb-2">No variants yet</p>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Create variants from your materials to explore structural modifications and predict property changes
                </p>
                <Button className="gap-2 bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600">
                  <Plus className="h-4 w-4" />
                  Create Variant
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
