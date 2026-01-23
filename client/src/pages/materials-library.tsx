import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Hexagon, Plus, Upload, Search, Zap, Atom, Box, Layers } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Link } from "wouter";

export default function MaterialsLibraryPage() {
  const categories = [
    { label: "Total Materials", value: 0, icon: Hexagon, color: "from-emerald-500 to-teal-500", bgColor: "bg-emerald-500" },
    { label: "Polymers", value: 0, icon: Layers, color: "from-blue-500 to-cyan-500", bgColor: "bg-blue-500" },
    { label: "Crystals", value: 0, icon: Atom, color: "from-violet-500 to-purple-500", bgColor: "bg-violet-500" },
    { label: "Composites", value: 0, icon: Box, color: "from-amber-500 to-orange-500", bgColor: "bg-amber-500" },
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-emerald-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-600 via-teal-500 to-cyan-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxNWwtMTMuMDkgNy41djE1TDMwIDQ1bDEzLjA5LTcuNXYtMTVMMzAgMTV6IiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4xNSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Hexagon className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">Materials Library</h1>
                    <p className="text-emerald-100">Browse and manage your materials collection</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-emerald-100">
                  <Zap className="h-4 w-4" />
                  <span>Polymers, crystals, composites, catalysts, and coatings</span>
                </div>
              </div>
              <Link href="/import/materials/materials">
                <Button className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0">
                  <Upload className="h-4 w-4" />
                  Import Materials
                </Button>
              </Link>
            </div>
          </header>

          <div className="relative max-w-xl">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input 
              placeholder="Search materials by name, formula, or properties..." 
              className="pl-12 h-12 text-base"
              data-testid="input-search-materials"
            />
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            {categories.map((cat) => (
              <Card key={cat.label} className="border-0 shadow-lg overflow-hidden group cursor-pointer hover:shadow-xl transition-all">
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${cat.color}`} />
                  <div className="p-6">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl ${cat.bgColor}/10 flex items-center justify-center group-hover:scale-110 transition-transform`}>
                        <cat.icon className={`h-6 w-6 ${cat.bgColor.replace('bg-', 'text-')}`} />
                      </div>
                      <div>
                        <p className="text-3xl font-bold">{cat.value}</p>
                        <p className="text-sm text-muted-foreground">{cat.label}</p>
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
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-5">
                  <Hexagon className="h-9 w-9 text-emerald-500" />
                </div>
                <p className="font-semibold text-lg mb-2">No materials yet</p>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Import polymers, crystals, composites, or coatings to build your materials library
                </p>
                <Link href="/import/materials/materials">
                  <Button className="gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600">
                    <Plus className="h-4 w-4" />
                    Import Materials
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
