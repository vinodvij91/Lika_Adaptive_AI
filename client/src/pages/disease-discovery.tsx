import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { 
  Brain, 
  Search, 
  Beaker,
  ArrowRight,
  Target,
  Database,
  Loader2,
  FlaskConical,
  Sparkles,
  Activity
} from "lucide-react";

interface DiseaseCondition {
  condition: string;
  count: number;
}

interface SmilesStats {
  success: boolean;
  totalRecords: number;
  categories: { category: string; count: number }[];
  diseaseConditions: DiseaseCondition[];
}

const DISEASE_ICONS: Record<string, typeof Brain> = {
  "Alzheimer's Disease": Brain,
  "Parkinson's Disease": Activity,
  "Huntington's Disease": Brain,
  "ALS (Amyotrophic Lateral Sclerosis)": Activity,
  "Multiple Sclerosis": Brain,
  "Dementia": Brain,
  "Neuroinflammation": FlaskConical,
  "Neurological Disorders": Brain,
  "default": Target
};

export default function DiseaseDiscovery() {
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedDisease, setSelectedDisease] = useState<DiseaseCondition | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const { data: stats, isLoading: statsLoading } = useQuery<SmilesStats>({
    queryKey: ["/api/external-sync/digitalocean/smiles/stats"],
  });

  const startScreeningMutation = useMutation({
    mutationFn: async (disease: string) => {
      const projectRes = await apiRequest("POST", "/api/projects", {
        name: `${disease} Drug Discovery`,
        description: `Automated screening of ChEMBL compounds against ${disease} protein targets`,
        diseaseArea: "CNS",
      });
      const project = await projectRes.json();
      
      const campaignRes = await apiRequest("POST", "/api/campaigns", {
        name: `${disease} SMILES Screening`,
        projectId: project.id,
        domainType: "CNS",
        modality: "small_molecule",
        pipelineConfig: {
          diseaseCondition: disease,
          seedSource: { type: "external_smiles", diseaseFilter: disease },
          filteringRules: ["lipinski", "veber", "pains", "brenk"],
          scoringWeights: { wQsar: 0.3, wAdmet: 0.3, wDocking: 0.4 },
          enableQuantumOptimization: false
        }
      });
      const campaign = await campaignRes.json();
      
      await apiRequest("POST", `/api/campaigns/${campaign.id}/start`, {});
      
      return { project, campaign, disease };
    },
    onSuccess: async ({ project, campaign, disease }) => {
      queryClient.invalidateQueries({ queryKey: ["/api/projects"] });
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns"] });
      
      toast({
        title: "Screening Started",
        description: `Pipeline is now running for ${disease} with ${selectedDisease?.count.toLocaleString()} compounds`,
      });
      
      setDialogOpen(false);
      setLocation(`/campaigns/${campaign.id}`);
    },
    onError: (error: any) => {
      toast({
        title: "Failed to start screening",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    },
  });

  const filteredDiseases = stats?.diseaseConditions.filter(d => 
    d.condition.toLowerCase().includes(searchTerm.toLowerCase()) &&
    d.condition !== "Unknown" &&
    d.condition !== "General"
  ) || [];

  const topDiseases = filteredDiseases.slice(0, 12);

  const handleStartScreening = (disease: DiseaseCondition) => {
    setSelectedDisease(disease);
    setDialogOpen(true);
  };

  const confirmStartScreening = () => {
    if (selectedDisease) {
      startScreeningMutation.mutate(selectedDisease.condition);
    }
  };

  const getIcon = (disease: string) => {
    return DISEASE_ICONS[disease] || DISEASE_ICONS.default;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="text-center space-y-4">
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Brain className="h-10 w-10 text-primary" />
          <h1 className="text-3xl font-bold" data-testid="text-page-title">
            Disease Discovery
          </h1>
        </div>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Choose a disease to start screening compounds against validated protein targets.
          Our database contains <span className="font-semibold text-primary">{stats?.totalRecords?.toLocaleString() || "1.6M+"}</span> curated SMILES compounds.
        </p>
      </div>

      <div className="max-w-md mx-auto">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          <Input
            placeholder="Search diseases..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
            data-testid="input-search-diseases"
          />
        </div>
      </div>

      {statsLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[1, 2, 3, 4, 5, 6, 7, 8].map(i => (
            <Card key={i}>
              <CardContent className="pt-6 space-y-3">
                <Skeleton className="h-8 w-8 rounded-full" />
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-10 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {topDiseases.map((disease) => {
              const Icon = getIcon(disease.condition);
              
              return (
                <Card 
                  key={disease.condition}
                  className="relative overflow-visible bg-card border"
                  data-testid={`card-disease-${disease.condition.replace(/\s+/g, '-').toLowerCase()}`}
                >
                  <CardContent className="pt-6 space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Icon className="h-6 w-6 text-primary" />
                      </div>
                      <Badge variant="secondary" className="font-mono">
                        {disease.count.toLocaleString()}
                      </Badge>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-lg leading-tight">
                        {disease.condition}
                      </h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        compounds available
                      </p>
                    </div>

                    <Button 
                      className="w-full gap-2" 
                      onClick={() => handleStartScreening(disease)}
                      data-testid={`button-start-${disease.condition.replace(/\s+/g, '-').toLowerCase()}`}
                    >
                      <Sparkles className="h-4 w-4" />
                      Start Screening
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {filteredDiseases.length > 12 && (
            <div className="text-center">
              <p className="text-muted-foreground">
                Showing 12 of {filteredDiseases.length} disease categories. 
                Use the search to find more.
              </p>
            </div>
          )}

          {filteredDiseases.length === 0 && searchTerm && (
            <div className="text-center py-12">
              <Database className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-lg text-muted-foreground">
                No diseases found matching "{searchTerm}"
              </p>
            </div>
          )}
        </>
      )}

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-2">
            <Beaker className="h-5 w-5" />
            How It Works
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">1</div>
                <h4 className="font-medium">Choose Disease</h4>
              </div>
              <p className="text-sm text-muted-foreground pl-10">
                Select from our curated disease categories to focus your screening efforts.
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">2</div>
                <h4 className="font-medium">Automated Screening</h4>
              </div>
              <p className="text-sm text-muted-foreground pl-10">
                Our pipeline screens compounds against validated targets for your chosen disease.
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">3</div>
                <h4 className="font-medium">Analyze Results</h4>
              </div>
              <p className="text-sm text-muted-foreground pl-10">
                Review scored hits, docking results, and ADMET predictions for lead optimization.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex flex-wrap items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Start Screening Campaign
            </DialogTitle>
            <DialogDescription>
              You're about to start a drug discovery campaign for {selectedDisease?.condition}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="rounded-lg border p-4 bg-muted/50">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Disease</p>
                  <p className="font-semibold">{selectedDisease?.condition}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Available Compounds</p>
                  <p className="font-semibold">{selectedDisease?.count.toLocaleString()}</p>
                </div>
              </div>
            </div>

            <p className="text-sm text-muted-foreground">
              This will create a new project and screening campaign. The pipeline will automatically 
              screen compounds from our database against protein targets associated with this disease.
            </p>

            <div className="flex flex-wrap gap-3">
              <Button 
                variant="outline" 
                className="flex-1"
                onClick={() => setDialogOpen(false)}
                data-testid="button-cancel-screening"
              >
                Cancel
              </Button>
              <Button 
                className="flex-1 gap-2"
                onClick={confirmStartScreening}
                disabled={startScreeningMutation.isPending}
                data-testid="button-confirm-screening"
              >
                {startScreeningMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <ArrowRight className="h-4 w-4" />
                    Start Screening
                  </>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
