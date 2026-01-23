import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Hexagon, Plus, Upload, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Link } from "wouter";

export default function MaterialsLibraryPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Materials Science" }, { label: "Materials Library" }]}
        actions={
          <Link href="/import/materials/materials">
            <Button className="gap-2" data-testid="button-import-materials">
              <Upload className="h-4 w-4" />
              Import Materials
            </Button>
          </Link>
        }
      />
      
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input 
                placeholder="Search materials by name, formula, or properties..." 
                className="pl-9"
                data-testid="input-search-materials"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-chart-2/10 flex items-center justify-center">
                    <Hexagon className="h-6 w-6 text-chart-2" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Materials</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center">
                    <Hexagon className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Polymers</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-chart-3/10 flex items-center justify-center">
                    <Hexagon className="h-6 w-6 text-chart-3" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Crystals</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Hexagon className="h-5 w-5 text-muted-foreground" />
                All Materials
              </CardTitle>
              <CardDescription>
                Browse and manage your materials library including polymers, crystals, composites, and coatings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Hexagon className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="font-medium mb-1">No materials yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Import materials to start building your library
                </p>
                <Link href="/import/materials/materials">
                  <Button className="gap-2">
                    <Plus className="h-4 w-4" />
                    Import Materials
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
