import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  Activity,
  Heart,
  Dna,
  Pill,
  Zap,
  Shield,
  Microscope,
  Flame,
  Eye,
  Bone,
  Baby,
  Droplets,
  Wind,
  Scan
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

const ADDITIONAL_DISEASE_CATEGORIES: DiseaseCondition[] = [
  { condition: "Cancer - Solid Tumors", count: 0 },
  { condition: "Cancer - Hematological", count: 0 },
  { condition: "Cancer - Breast", count: 0 },
  { condition: "Cancer - Lung", count: 0 },
  { condition: "Cancer - Pancreatic", count: 0 },
  { condition: "Mitochondrial Diseases", count: 0 },
  { condition: "Mitochondrial Dysfunction", count: 0 },
  { condition: "Rare Genetic Disorders", count: 0 },
  { condition: "Orphan Diseases", count: 0 },
  { condition: "Cardiovascular Disease", count: 0 },
  { condition: "Heart Failure", count: 0 },
  { condition: "Diabetes Type 2", count: 0 },
  { condition: "Metabolic Syndrome", count: 0 },
  { condition: "Obesity", count: 0 },
  { condition: "NAFLD/NASH", count: 0 },
  { condition: "Autoimmune Disorders", count: 0 },
  { condition: "Rheumatoid Arthritis", count: 0 },
  { condition: "Lupus", count: 0 },
  { condition: "Inflammatory Bowel Disease", count: 0 },
  { condition: "Infectious Disease - Viral", count: 0 },
  { condition: "Infectious Disease - Bacterial", count: 0 },
  { condition: "HIV/AIDS", count: 0 },
  { condition: "Hepatitis", count: 0 },
  { condition: "Tuberculosis", count: 0 },
  { condition: "Respiratory Diseases", count: 0 },
  { condition: "COPD", count: 0 },
  { condition: "Asthma", count: 0 },
  { condition: "Pulmonary Fibrosis", count: 0 },
  { condition: "Kidney Disease", count: 0 },
  { condition: "Liver Disease", count: 0 },
  { condition: "Eye Diseases", count: 0 },
  { condition: "Macular Degeneration", count: 0 },
  { condition: "Glaucoma", count: 0 },
  { condition: "Bone Disorders", count: 0 },
  { condition: "Osteoporosis", count: 0 },
  { condition: "Pediatric Diseases", count: 0 },
  { condition: "Aging & Longevity", count: 0 },
];

const THERAPEUTIC_AREAS = [
  { id: "all", label: "All Diseases" },
  { id: "cns", label: "CNS & Neuro" },
  { id: "oncology", label: "Oncology" },
  { id: "rare", label: "Rare & Genetic" },
  { id: "cardio", label: "Cardiovascular" },
  { id: "metabolic", label: "Metabolic" },
  { id: "immune", label: "Immune & Inflammatory" },
  { id: "infectious", label: "Infectious" },
  { id: "other", label: "Other" },
];

const DISEASE_AREA_MAP: Record<string, string> = {
  "Alzheimer's Disease": "cns",
  "Parkinson's Disease": "cns",
  "Huntington's Disease": "cns",
  "ALS (Amyotrophic Lateral Sclerosis)": "cns",
  "Multiple Sclerosis": "cns",
  "Dementia": "cns",
  "Neuroinflammation": "cns",
  "Neurological Disorders": "cns",
  "Cancer - Solid Tumors": "oncology",
  "Cancer - Hematological": "oncology",
  "Cancer - Breast": "oncology",
  "Cancer - Lung": "oncology",
  "Cancer - Pancreatic": "oncology",
  "Mitochondrial Diseases": "rare",
  "Mitochondrial Dysfunction": "rare",
  "Rare Genetic Disorders": "rare",
  "Orphan Diseases": "rare",
  "Cardiovascular Disease": "cardio",
  "Heart Failure": "cardio",
  "Diabetes Type 2": "metabolic",
  "Metabolic Syndrome": "metabolic",
  "Obesity": "metabolic",
  "NAFLD/NASH": "metabolic",
  "Autoimmune Disorders": "immune",
  "Rheumatoid Arthritis": "immune",
  "Lupus": "immune",
  "Inflammatory Bowel Disease": "immune",
  "Infectious Disease - Viral": "infectious",
  "Infectious Disease - Bacterial": "infectious",
  "HIV/AIDS": "infectious",
  "Hepatitis": "infectious",
  "Tuberculosis": "infectious",
  "Respiratory Diseases": "other",
  "COPD": "other",
  "Asthma": "other",
  "Pulmonary Fibrosis": "other",
  "Kidney Disease": "other",
  "Liver Disease": "other",
  "Eye Diseases": "other",
  "Macular Degeneration": "other",
  "Glaucoma": "other",
  "Bone Disorders": "other",
  "Osteoporosis": "other",
  "Pediatric Diseases": "other",
  "Aging & Longevity": "other",
};

const DISEASE_ICONS: Record<string, typeof Brain> = {
  "Alzheimer's Disease": Brain,
  "Parkinson's Disease": Activity,
  "Huntington's Disease": Brain,
  "ALS (Amyotrophic Lateral Sclerosis)": Activity,
  "Multiple Sclerosis": Brain,
  "Dementia": Brain,
  "Neuroinflammation": FlaskConical,
  "Neurological Disorders": Brain,
  "Cancer - Solid Tumors": Scan,
  "Cancer - Hematological": Droplets,
  "Cancer - Breast": Target,
  "Cancer - Lung": Wind,
  "Cancer - Pancreatic": Target,
  "Mitochondrial Diseases": Zap,
  "Mitochondrial Dysfunction": Zap,
  "Rare Genetic Disorders": Dna,
  "Orphan Diseases": Dna,
  "Cardiovascular Disease": Heart,
  "Heart Failure": Heart,
  "Diabetes Type 2": Pill,
  "Metabolic Syndrome": Flame,
  "Obesity": Flame,
  "NAFLD/NASH": Flame,
  "Autoimmune Disorders": Shield,
  "Rheumatoid Arthritis": Bone,
  "Lupus": Shield,
  "Inflammatory Bowel Disease": Shield,
  "Infectious Disease - Viral": Microscope,
  "Infectious Disease - Bacterial": Microscope,
  "HIV/AIDS": Shield,
  "Hepatitis": Microscope,
  "Tuberculosis": Microscope,
  "Respiratory Diseases": Wind,
  "COPD": Wind,
  "Asthma": Wind,
  "Pulmonary Fibrosis": Wind,
  "Kidney Disease": Droplets,
  "Liver Disease": Flame,
  "Eye Diseases": Eye,
  "Macular Degeneration": Eye,
  "Glaucoma": Eye,
  "Bone Disorders": Bone,
  "Osteoporosis": Bone,
  "Pediatric Diseases": Baby,
  "Aging & Longevity": Activity,
  "default": Target
};

export default function DiseaseDiscovery() {
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedDisease, setSelectedDisease] = useState<DiseaseCondition | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedArea, setSelectedArea] = useState("all");

  const { data: stats, isLoading: statsLoading } = useQuery<SmilesStats>({
    queryKey: ["/api/external-sync/digitalocean/smiles/stats"],
  });

  const startScreeningMutation = useMutation({
    mutationFn: async (disease: string) => {
      const projectRes = await apiRequest("POST", "/api/projects", {
        name: `${disease} Drug Discovery`,
        description: `Automated screening of ChEMBL compounds against ${disease} protein targets`,
        diseaseArea: getDiseaseAreaLabel(disease),
      });
      const project = await projectRes.json();
      
      const campaignRes = await apiRequest("POST", "/api/campaigns", {
        name: `${disease} SMILES Screening`,
        projectId: project.id,
        domainType: getDiseaseAreaLabel(disease),
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
        description: `Pipeline is now running for ${disease}`,
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

  const getDiseaseAreaLabel = (disease: string): string => {
    const area = DISEASE_AREA_MAP[disease];
    const areaObj = THERAPEUTIC_AREAS.find(a => a.id === area);
    return areaObj?.label || "Other";
  };

  const allDiseases = (() => {
    const dbDiseases = stats?.diseaseConditions.filter(d => 
      d.condition !== "Unknown" && d.condition !== "General"
    ) || [];
    
    const dbConditionNames = new Set(dbDiseases.map(d => d.condition));
    const additionalDiseases = ADDITIONAL_DISEASE_CATEGORIES.filter(
      d => !dbConditionNames.has(d.condition)
    );
    
    return [...dbDiseases, ...additionalDiseases];
  })();

  const filteredDiseases = allDiseases.filter(d => {
    const matchesSearch = d.condition.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesArea = selectedArea === "all" || 
      DISEASE_AREA_MAP[d.condition] === selectedArea ||
      (!DISEASE_AREA_MAP[d.condition] && selectedArea === "other");
    return matchesSearch && matchesArea;
  });

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

      <Tabs value={selectedArea} onValueChange={setSelectedArea} className="w-full">
        <TabsList className="flex flex-wrap h-auto gap-1 bg-transparent justify-center">
          {THERAPEUTIC_AREAS.map(area => (
            <TabsTrigger 
              key={area.id} 
              value={area.id}
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
              data-testid={`tab-${area.id}`}
            >
              {area.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

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
            {filteredDiseases.map((disease) => {
              const Icon = getIcon(disease.condition);
              const hasData = disease.count > 0;
              
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
                      {hasData ? (
                        <Badge variant="secondary" className="font-mono">
                          {disease.count.toLocaleString()}
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-muted-foreground">
                          Coming Soon
                        </Badge>
                      )}
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-lg leading-tight">
                        {disease.condition}
                      </h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        {hasData ? "compounds available" : "database pending"}
                      </p>
                    </div>

                    <Button 
                      className="w-full gap-2" 
                      onClick={() => handleStartScreening(disease)}
                      disabled={!hasData}
                      variant={hasData ? "default" : "outline"}
                      data-testid={`button-start-${disease.condition.replace(/\s+/g, '-').toLowerCase()}`}
                    >
                      <Sparkles className="h-4 w-4" />
                      {hasData ? "Start Screening" : "Not Available"}
                      {hasData && <ArrowRight className="h-4 w-4" />}
                    </Button>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {filteredDiseases.length === 0 && (
            <div className="text-center py-12">
              <Database className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-lg text-muted-foreground">
                {searchTerm 
                  ? `No diseases found matching "${searchTerm}"`
                  : "No diseases in this category"}
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
