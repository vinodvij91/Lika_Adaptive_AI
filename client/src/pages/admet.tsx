import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Beaker, Activity, Shield, Zap, TrendingUp } from "lucide-react";

export default function AdmetPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Drug Discovery" }, { label: "ADMET" }]}
      />
      
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-md bg-blue-500/10 flex items-center justify-center">
                    <Activity className="h-4 w-4 text-blue-500" />
                  </div>
                  <CardTitle className="text-base">Absorption</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Intestinal absorption, bioavailability, Caco-2 permeability
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-md bg-green-500/10 flex items-center justify-center">
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  </div>
                  <CardTitle className="text-base">Distribution</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  BBB penetration, plasma protein binding, VDss
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-md bg-amber-500/10 flex items-center justify-center">
                    <Zap className="h-4 w-4 text-amber-500" />
                  </div>
                  <CardTitle className="text-base">Metabolism</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  CYP inhibition/induction, metabolic stability
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-md bg-red-500/10 flex items-center justify-center">
                    <Shield className="h-4 w-4 text-red-500" />
                  </div>
                  <CardTitle className="text-base">Toxicity</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  hERG inhibition, hepatotoxicity, mutagenicity
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Run ADMET Predictions</CardTitle>
              <CardDescription>
                Predict absorption, distribution, metabolism, excretion, and toxicity properties for your molecules
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Beaker className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="font-medium mb-1">No ADMET predictions yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Select molecules from your library to run ADMET predictions
                </p>
                <Button data-testid="button-run-admet">
                  Select Molecules
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
