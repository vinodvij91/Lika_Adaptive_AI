import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Hexagon, Plus, Upload, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Link } from "wouter";

export default function MaterialsLibraryPage() {
  const categories = [
    { label: "Total Materials", value: 0, color: "text-chart-2", bgColor: "bg-chart-2/10" },
    { label: "Polymers", value: 0, color: "text-primary", bgColor: "bg-primary/10" },
    { label: "Crystals", value: 0, color: "text-chart-3", bgColor: "bg-chart-3/10" },
    { label: "Composites", value: 0, color: "text-amber-500", bgColor: "bg-amber-500/10" },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                <Hexagon className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Materials Library</h1>
                <p className="text-sm text-muted-foreground">Browse and manage your materials collection</p>
              </div>
            </div>
            <Link href="/import/materials/materials">
              <Button className="gap-2" data-testid="button-import-materials">
                <Upload className="h-4 w-4" />
                Import Materials
              </Button>
            </Link>
          </header>

          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search materials by name, formula, or properties..." 
              className="pl-9"
              data-testid="input-search-materials"
            />
          </div>

          <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {categories.map((cat) => (
              <div
                key={cat.label}
                className="p-5 rounded-xl border bg-card/50"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-8 h-8 rounded-lg ${cat.bgColor} flex items-center justify-center`}>
                    <Hexagon className={`h-4 w-4 ${cat.color}`} />
                  </div>
                </div>
                <p className="text-2xl font-semibold tabular-nums">{cat.value}</p>
                <p className="text-xs text-muted-foreground mt-1">{cat.label}</p>
              </div>
            ))}
          </section>

          <section className="space-y-4">
            <Card>
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <Hexagon className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium mb-1">No materials yet</p>
                  <p className="text-xs text-muted-foreground mb-4 max-w-sm mx-auto">
                    Import polymers, crystals, composites, or coatings to build your library
                  </p>
                  <Link href="/import/materials/materials">
                    <Button size="sm" className="h-8 text-xs gap-1.5">
                      <Plus className="h-3.5 w-3.5" />
                      Import Materials
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}
