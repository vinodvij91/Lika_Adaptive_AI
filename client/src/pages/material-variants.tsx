import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Layers, Plus, Search, GitBranch } from "lucide-react";
import { Input } from "@/components/ui/input";

export default function MaterialVariantsPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Materials Science" }, { label: "Variants" }]}
        actions={
          <Button className="gap-2" data-testid="button-create-variant">
            <Plus className="h-4 w-4" />
            Create Variant
          </Button>
        }
      />
      
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input 
                placeholder="Search variants..." 
                className="pl-9"
                data-testid="input-search-variants"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-chart-2/10 flex items-center justify-center">
                    <Layers className="h-6 w-6 text-chart-2" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Variants</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center">
                    <GitBranch className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Active Explorations</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-md bg-chart-3/10 flex items-center justify-center">
                    <Layers className="h-6 w-6 text-chart-3" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Evaluated</p>
                    <p className="text-2xl font-bold">0</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-muted-foreground" />
                Material Variants
              </CardTitle>
              <CardDescription>
                Explore structural variations of materials with property predictions and comparisons
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Layers className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="font-medium mb-1">No variants yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Create variants from your materials to explore structural modifications
                </p>
                <Button className="gap-2">
                  <Plus className="h-4 w-4" />
                  Create Variant
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
