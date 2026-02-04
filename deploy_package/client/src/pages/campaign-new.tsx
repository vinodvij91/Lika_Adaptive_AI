import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useLocation, useSearch } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Sparkles,
  Upload,
  FlaskConical,
  Target,
  Beaker,
  ChevronRight,
  Play,
  Save,
  CheckCircle,
  Library,
  FileStack,
  Brain,
  Activity,
  Dna,
  Heart,
  Zap,
  Cpu,
  Layers,
  Gauge,
  Eye,
  Clock,
} from "lucide-react";
import type { Project, Target as TargetType, DiseaseArea, PipelineConfig, CuratedLibrary, PipelineTemplate, PipelineTemplateTarget } from "@shared/schema";

const diseaseAreas: DiseaseArea[] = ["CNS", "Oncology", "Rare", "Infectious", "Cardiometabolic", "Autoimmune", "Respiratory", "Other"];

const filteringRules = [
  { id: "lipinski", label: "Lipinski's Rule of Five" },
  { id: "veber", label: "Veber's Rules" },
  { id: "pains", label: "PAINS Filter" },
  { id: "brenk", label: "Brenk Filter" },
];

interface PipelineStep {
  id: number;
  title: string;
  icon: typeof Sparkles;
  completed: boolean;
}

const templateDomainIcons: Record<string, typeof Brain> = {
  alzheimers: Brain,
  oncology: Dna,
  neuroinflammation: Activity,
  metabolic_disease: Heart,
  immunology: Activity,
  infectious_disease: Dna,
  custom: FileStack,
};

const templateDomainLabels: Record<string, string> = {
  alzheimers: "Alzheimer's Disease",
  oncology: "Oncology",
  neuroinflammation: "Neuroinflammation",
  metabolic_disease: "Metabolic Disease",
  immunology: "Immunology",
  infectious_disease: "Infectious Disease",
  custom: "Custom",
};

export default function CampaignNewPage() {
  const [, setLocation] = useLocation();
  const searchParams = useSearch();
  const params = new URLSearchParams(searchParams);
  const projectIdFromUrl = params.get("projectId");
  const { toast } = useToast();

  const [currentStep, setCurrentStep] = useState(0);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [projectId, setProjectId] = useState(projectIdFromUrl || "");
  const [domainType, setDomainType] = useState<DiseaseArea>("CNS");
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [templateTargets, setTemplateTargets] = useState<PipelineTemplateTarget[]>([]);
  const [generator, setGenerator] = useState<"bionemo_molmim" | "upload_library" | "curated_library">("bionemo_molmim");
  const [selectedLibraryId, setSelectedLibraryId] = useState<string>("");
  const [selectedFilters, setSelectedFilters] = useState<string[]>(["lipinski"]);
  const [dockingMethod, setDockingMethod] = useState<"bionemo_diffdock" | "external_docking">("bionemo_diffdock");
  const [wDocking, setWDocking] = useState(0.4);
  const [wAdmet, setWAdmet] = useState(0.3);
  const [wQsar, setWQsar] = useState(0.3);
  const [enableQuantum, setEnableQuantum] = useState(false);
  const [quantumObjective, setQuantumObjective] = useState("maximize_oracle_score");
  const [quantumMaxMolecules, setQuantumMaxMolecules] = useState(200);

  // Virtual Screening Configuration (Step 6)
  const [vsAqaffinity, setVsAqaffinity] = useState(true);
  const [vsAutodock, setVsAutodock] = useState(false);
  const [vsDualScreening, setVsDualScreening] = useState(false);
  const [vsBindingAffinity, setVsBindingAffinity] = useState(0.4);
  const [vsAdmetProperties, setVsAdmetProperties] = useState(0.3);
  const [vsSelectivity, setVsSelectivity] = useState(0.2);
  const [vsSyntheticAccessibility, setVsSyntheticAccessibility] = useState(0.1);

  const { data: projects } = useQuery<Project[]>({
    queryKey: ["/api/projects"],
  });

  const { data: targetsWithDiseases } = useQuery<(TargetType & { diseases: string[] })[]>({
    queryKey: ["/api/targets-with-diseases"],
  });

  const { data: diseases } = useQuery<{ disease: string; count: number }[]>({
    queryKey: ["/api/diseases"],
  });

  const [targetDiseaseFilter, setTargetDiseaseFilter] = useState<string>("all");

  const filteredTargets = targetsWithDiseases?.filter(t => 
    targetDiseaseFilter === "all" || (t.diseases && t.diseases.includes(targetDiseaseFilter))
  ) || [];

  const { data: libraries } = useQuery<CuratedLibrary[]>({
    queryKey: ["/api/libraries"],
  });

  const { data: pipelineTemplates } = useQuery<PipelineTemplate[]>({
    queryKey: ["/api/pipeline-templates"],
  });

  const { data: selectedTemplate } = useQuery<PipelineTemplate & { targets: PipelineTemplateTarget[] }>({
    queryKey: ["/api/pipeline-templates", selectedTemplateId],
    enabled: !!selectedTemplateId,
  });

  useEffect(() => {
    if (selectedTemplateId === null) {
      setTemplateTargets([]);
      setWDocking(0.4);
      setWAdmet(0.3);
      setWQsar(0.3);
      return;
    }
    if (selectedTemplate) {
      const weights = selectedTemplate.scoringWeights as { efficacy?: number; selectivity?: number; safety?: number } | null;
      if (weights) {
        setWDocking(weights.efficacy || 0.4);
        setWAdmet(weights.selectivity || 0.3);
        setWQsar(weights.safety || 0.3);
      }
      if (selectedTemplate.targets) {
        setTemplateTargets(selectedTemplate.targets);
      } else {
        setTemplateTargets([]);
      }
      if (selectedTemplate.domain === "alzheimers" || selectedTemplate.domain === "neuroinflammation") {
        setDomainType("CNS");
      } else if (selectedTemplate.domain === "oncology") {
        setDomainType("Oncology");
      } else if (selectedTemplate.domain === "metabolic_disease") {
        setDomainType("Cardiometabolic");
      }
    }
  }, [selectedTemplate, selectedTemplateId]);

  const curatedLibraries = libraries?.filter(lib => lib.status === "curated" && lib.domainType === domainType) || [];
  const selectedLibrary = libraries?.find(lib => lib.id === selectedLibraryId);

  const createMutation = useMutation({
    mutationFn: async (data: { name: string; projectId: string; domainType: DiseaseArea; pipelineConfig: PipelineConfig }) => {
      const res = await apiRequest("POST", "/api/campaigns", data);
      return res.json();
    },
    onSuccess: (campaign) => {
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns"] });
      queryClient.invalidateQueries({ queryKey: ["/api/projects", projectId] });
      toast({ title: "Campaign created", description: `${campaign.name} is ready to run.` });
      setLocation(`/campaigns/${campaign.id}`);
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to create campaign", variant: "destructive" });
    },
  });

  const steps: PipelineStep[] = [
    { id: 0, title: "Template", icon: FileStack, completed: true },
    { id: 1, title: "Basic Info", icon: Beaker, completed: !!name && !!projectId },
    { id: 2, title: "Targets", icon: Target, completed: selectedTargets.length > 0 || templateTargets.length > 0 },
    { id: 3, title: "Compounds", icon: Sparkles, completed: true },
    { id: 4, title: "Filtering", icon: FlaskConical, completed: true },
    { id: 5, title: "Scoring", icon: Gauge, completed: true },
    { id: 6, title: "Virtual Screening", icon: Zap, completed: vsAqaffinity || vsAutodock },
    { id: 7, title: "Review & Launch", icon: Eye, completed: false },
  ];

  const handleSubmit = () => {
    const pipelineConfig: PipelineConfig = {
      generator: generator === "curated_library" ? "upload_library" : generator,
      filteringRules: selectedFilters,
      dockingMethod,
      scoringWeights: { wDocking, wAdmet, wQsar },
      targetIds: selectedTargets,
      enableQuantumOptimization: enableQuantum,
      quantumParams: enableQuantum ? {
        objective: quantumObjective,
        maxMolecules: quantumMaxMolecules,
      } : undefined,
      seedSource: generator === "curated_library" && selectedLibraryId ? {
        type: "curated_library",
        libraryId: selectedLibraryId,
      } : generator === "upload_library" ? {
        type: "uploaded_set",
      } : {
        type: "generated",
      },
      templateId: selectedTemplateId || undefined,
      templateTargets: templateTargets.length > 0 ? templateTargets.map(t => ({
        name: t.name,
        role: t.role,
        category: t.category,
      })) : undefined,
      virtualScreening: {
        methods: {
          aqaffinity: vsAqaffinity,
          autodock: vsAutodock,
        },
        dualScreeningEnabled: vsDualScreening,
        scoringWeights: {
          bindingAffinity: vsBindingAffinity,
          admetProperties: vsAdmetProperties,
          selectivity: vsSelectivity,
          syntheticAccessibility: vsSyntheticAccessibility,
        },
        estimatedTime: {
          aqaffinity: "4 hours",
          autodock: "48 hours",
        },
      },
    };

    createMutation.mutate({
      name,
      projectId,
      domainType,
      pipelineConfig,
    });
  };

  const canProceed = () => {
    switch (currentStep) {
      case 0: return true;
      case 1: return !!name && !!projectId;
      case 2: return selectedTargets.length > 0 || templateTargets.length > 0;
      case 6: return vsAqaffinity || vsAutodock;
      default: return true;
    }
  };

  const handleDualScreeningToggle = (enabled: boolean) => {
    setVsDualScreening(enabled);
    if (enabled) {
      setVsAqaffinity(true);
      setVsAutodock(true);
    }
  };

  const getTotalVsWeight = () => {
    return vsBindingAffinity + vsAdmetProperties + vsSelectivity + vsSyntheticAccessibility;
  };

  const toggleTarget = (targetId: string) => {
    setSelectedTargets((prev) =>
      prev.includes(targetId)
        ? prev.filter((id) => id !== targetId)
        : [...prev, targetId]
    );
  };

  const toggleFilter = (filterId: string) => {
    setSelectedFilters((prev) =>
      prev.includes(filterId)
        ? prev.filter((id) => id !== filterId)
        : [...prev, filterId]
    );
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Campaigns", href: "/campaigns" },
          { label: "New Campaign" },
        ]}
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex gap-8">
            <div className="w-64 flex-shrink-0">
              <Card>
                <CardContent className="p-4">
                  <nav className="space-y-1">
                    {steps.map((step, index) => {
                      const Icon = step.icon;
                      const isActive = index === currentStep;
                      const isCompleted = step.completed && index < currentStep;

                      return (
                        <button
                          key={step.id}
                          onClick={() => setCurrentStep(index)}
                          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-colors ${
                            isActive
                              ? "bg-primary/10 text-primary"
                              : "text-muted-foreground hover-elevate"
                          }`}
                          data-testid={`step-${step.id}`}
                        >
                          <div
                            className={`w-8 h-8 rounded-md flex items-center justify-center flex-shrink-0 ${
                              isActive
                                ? "bg-primary text-primary-foreground"
                                : isCompleted
                                ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
                                : "bg-muted"
                            }`}
                          >
                            {isCompleted ? (
                              <CheckCircle className="h-4 w-4" />
                            ) : (
                              <Icon className="h-4 w-4" />
                            )}
                          </div>
                          <span className="text-sm font-medium">{step.title}</span>
                        </button>
                      );
                    })}
                  </nav>
                </CardContent>
              </Card>
            </div>

            <div className="flex-1 space-y-6">
              {currentStep === 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Select Pipeline Template</CardTitle>
                    <CardDescription>
                      Choose a disease-specific template to pre-configure targets, assay panels, and scoring weights, or start from scratch.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                      <button
                        onClick={() => setSelectedTemplateId(null)}
                        className={`p-5 rounded-lg border-2 text-left transition-colors ${
                          selectedTemplateId === null
                            ? "border-primary bg-primary/5"
                            : "border-border hover-elevate"
                        }`}
                        data-testid="option-blank-template"
                      >
                        <FileStack className="h-8 w-8 text-muted-foreground mb-3" />
                        <h3 className="font-semibold mb-1">Start from Scratch</h3>
                        <p className="text-sm text-muted-foreground">
                          Configure all pipeline settings manually
                        </p>
                      </button>
                      {pipelineTemplates?.filter(t => t.isBuiltIn).map((template) => {
                        const DomainIcon = templateDomainIcons[template.domain] || FileStack;
                        return (
                          <button
                            key={template.id}
                            onClick={() => setSelectedTemplateId(template.id)}
                            className={`p-5 rounded-lg border-2 text-left transition-colors ${
                              selectedTemplateId === template.id
                                ? "border-primary bg-primary/5"
                                : "border-border hover-elevate"
                            }`}
                            data-testid={`option-template-${template.id}`}
                          >
                            <div className="flex items-start justify-between mb-3">
                              <DomainIcon className="h-8 w-8 text-chart-3" />
                              <Badge variant="secondary" className="text-xs">
                                {templateDomainLabels[template.domain] || template.domain}
                              </Badge>
                            </div>
                            <h3 className="font-semibold mb-1">{template.name}</h3>
                            <p className="text-sm text-muted-foreground line-clamp-2">
                              {template.description}
                            </p>
                          </button>
                        );
                      })}
                    </div>
                    {selectedTemplate && (
                      <div className="p-4 bg-muted/50 rounded-md space-y-3">
                        <div>
                          <p className="font-medium">{selectedTemplate.name}</p>
                          <p className="text-sm text-muted-foreground">{selectedTemplate.description}</p>
                        </div>
                        {templateTargets.length > 0 && (
                          <div>
                            <p className="text-sm font-medium mb-2">Pre-configured Targets:</p>
                            <div className="flex flex-wrap gap-2">
                              {templateTargets.map((t, i) => (
                                <Badge key={i} variant={t.role === "primary" ? "default" : t.role === "safety" ? "destructive" : "secondary"}>
                                  {t.name} ({t.role})
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {currentStep === 1 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Basic Information</CardTitle>
                    <CardDescription>
                      Set up the campaign name and select a project
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Campaign Name</Label>
                      <Input
                        id="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="e.g., BACE1 Screening Round 1"
                        data-testid="input-campaign-name"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="project">Project</Label>
                      <Select value={projectId} onValueChange={setProjectId}>
                        <SelectTrigger data-testid="select-project">
                          <SelectValue placeholder="Select a project" />
                        </SelectTrigger>
                        <SelectContent>
                          {projects?.map((project) => (
                            <SelectItem key={project.id} value={project.id}>
                              {project.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="domain">Domain Type</Label>
                      <Select value={domainType} onValueChange={(v) => setDomainType(v as DiseaseArea)}>
                        <SelectTrigger data-testid="select-domain">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {diseaseAreas.map((area) => (
                            <SelectItem key={area} value={area}>
                              {area}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              )}

              {currentStep === 2 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Select Targets</CardTitle>
                    <CardDescription>
                      {templateTargets.length > 0 
                        ? "Template targets are pre-configured. You can also add targets from your database."
                        : "Choose one or more protein targets for this campaign"}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {templateTargets.length > 0 && (
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium text-muted-foreground">Template Targets (pre-configured)</h4>
                        <div className="space-y-2">
                          {templateTargets.map((t, i) => (
                            <div
                              key={i}
                              className="flex items-center gap-3 p-3 rounded-md bg-muted/50 border border-muted"
                              data-testid={`template-target-${i}`}
                            >
                              <div className={`w-9 h-9 rounded-md flex items-center justify-center ${
                                t.role === "primary" ? "bg-primary/10" : t.role === "safety" ? "bg-destructive/10" : "bg-chart-2/10"
                              }`}>
                                <Target className={`h-4 w-4 ${
                                  t.role === "primary" ? "text-primary" : t.role === "safety" ? "text-destructive" : "text-chart-2"
                                }`} />
                              </div>
                              <div className="flex-1">
                                <p className="font-medium">{t.name}</p>
                                <p className="text-sm text-muted-foreground">{t.description || t.category}</p>
                              </div>
                              <Badge variant={t.role === "primary" ? "default" : t.role === "safety" ? "destructive" : "secondary"}>
                                {t.role}
                              </Badge>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {targetsWithDiseases && targetsWithDiseases.length > 0 && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between gap-4">
                          <h4 className="text-sm font-medium text-muted-foreground">
                            {templateTargets.length > 0 ? "Additional Targets (from database)" : "Select from database"}
                          </h4>
                          <Select value={targetDiseaseFilter} onValueChange={setTargetDiseaseFilter}>
                            <SelectTrigger className="w-[200px]" data-testid="select-target-disease-filter">
                              <SelectValue placeholder="Filter by disease" />
                            </SelectTrigger>
                            <SelectContent className="max-h-[300px]">
                              <SelectItem value="all">All Diseases</SelectItem>
                              {diseases?.slice().sort((a, b) => a.disease.localeCompare(b.disease)).map((d) => (
                                <SelectItem key={d.disease} value={d.disease}>
                                  {d.disease} ({d.count})
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2 max-h-[400px] overflow-y-auto">
                          {filteredTargets.map((target) => (
                            <div
                              key={target.id}
                              className={`flex items-center gap-3 p-3 rounded-md cursor-pointer transition-colors ${
                                selectedTargets.includes(target.id)
                                  ? "bg-primary/10 border border-primary/30"
                                  : "hover-elevate"
                              }`}
                              onClick={() => toggleTarget(target.id)}
                              data-testid={`target-option-${target.id}`}
                            >
                              <Checkbox
                                checked={selectedTargets.includes(target.id)}
                                onCheckedChange={() => toggleTarget(target.id)}
                              />
                              <div className="w-9 h-9 rounded-md bg-chart-3/10 flex items-center justify-center">
                                <Target className="h-4 w-4 text-chart-3" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="font-medium">{target.name}</p>
                                <p className="text-sm text-muted-foreground">
                                  {target.uniprotId || "No UniProt ID"}
                                  {target.diseases && target.diseases.length > 0 && (
                                    <span className="ml-2 text-xs">
                                      {target.diseases.slice(0, 2).join(", ")}
                                      {target.diseases.length > 2 && ` +${target.diseases.length - 2}`}
                                    </span>
                                  )}
                                </p>
                              </div>
                            </div>
                          ))}
                          {filteredTargets.length === 0 && targetDiseaseFilter !== "all" && (
                            <p className="text-muted-foreground text-center py-4 text-sm">
                              No targets found for {targetDiseaseFilter}
                            </p>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {!targetsWithDiseases?.length && !templateTargets.length && (
                      <p className="text-muted-foreground text-center py-8">
                        No targets available. Add targets first.
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}

              {currentStep === 3 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Molecule Generator</CardTitle>
                    <CardDescription>
                      Choose how molecules will be generated or sourced
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      <button
                        onClick={() => setGenerator("bionemo_molmim")}
                        className={`p-6 rounded-lg border-2 text-left transition-colors ${
                          generator === "bionemo_molmim"
                            ? "border-primary bg-primary/5"
                            : "border-border hover-elevate"
                        }`}
                        data-testid="option-bionemo"
                      >
                        <Sparkles className="h-8 w-8 text-primary mb-3" />
                        <h3 className="font-semibold mb-1">BioNeMo MolMIM</h3>
                        <p className="text-sm text-muted-foreground">
                          Generate novel molecules using BioNeMo AI
                        </p>
                      </button>
                      <button
                        onClick={() => setGenerator("curated_library")}
                        className={`p-6 rounded-lg border-2 text-left transition-colors ${
                          generator === "curated_library"
                            ? "border-primary bg-primary/5"
                            : "border-border hover-elevate"
                        }`}
                        data-testid="option-curated-library"
                      >
                        <Library className="h-8 w-8 text-chart-3 mb-3" />
                        <h3 className="font-semibold mb-1">Curated Library</h3>
                        <p className="text-sm text-muted-foreground">
                          Use a domain-specific curated library
                        </p>
                      </button>
                      <button
                        onClick={() => setGenerator("upload_library")}
                        className={`p-6 rounded-lg border-2 text-left transition-colors ${
                          generator === "upload_library"
                            ? "border-primary bg-primary/5"
                            : "border-border hover-elevate"
                        }`}
                        data-testid="option-upload"
                      >
                        <Upload className="h-8 w-8 text-chart-2 mb-3" />
                        <h3 className="font-semibold mb-1">Upload Library</h3>
                        <p className="text-sm text-muted-foreground">
                          Screen an existing molecule library
                        </p>
                      </button>
                    </div>

                    {generator === "curated_library" && (
                      <div className="mt-6 pt-6 border-t">
                        <h4 className="font-medium mb-4">Select Curated Library</h4>
                        {curatedLibraries.length > 0 ? (
                          <div className="space-y-2">
                            {curatedLibraries.map((lib) => (
                              <div
                                key={lib.id}
                                onClick={() => setSelectedLibraryId(lib.id)}
                                className={`flex items-center gap-3 p-3 rounded-md cursor-pointer transition-colors ${
                                  selectedLibraryId === lib.id
                                    ? "bg-primary/10 border border-primary/30"
                                    : "hover-elevate"
                                }`}
                                data-testid={`library-option-${lib.id}`}
                              >
                                <div className="w-9 h-9 rounded-md bg-chart-3/10 flex items-center justify-center">
                                  <Library className="h-4 w-4 text-chart-3" />
                                </div>
                                <div className="flex-1 min-w-0">
                                  <p className="font-medium">{lib.name}</p>
                                  <p className="text-sm text-muted-foreground">
                                    {lib.moleculeCount || 0} molecules, {lib.scaffoldCount || 0} scaffolds
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-muted-foreground text-center py-4">
                            No curated libraries match your selected domain ({domainType}).{" "}
                            <a href="/libraries" className="text-primary underline">Create one</a>
                          </p>
                        )}
                        {selectedLibrary && (
                          <div className="mt-4 p-4 bg-muted rounded-md">
                            <p className="font-medium">{selectedLibrary.name}</p>
                            <p className="text-sm text-muted-foreground mt-1">
                              {selectedLibrary.description || "No description"}
                            </p>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {currentStep === 4 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Filtering Rules</CardTitle>
                    <CardDescription>
                      Select molecular property filters to apply
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {filteringRules.map((rule) => (
                        <div
                          key={rule.id}
                          className={`flex items-center gap-3 p-3 rounded-md cursor-pointer transition-colors ${
                            selectedFilters.includes(rule.id)
                              ? "bg-primary/10 border border-primary/30"
                              : "hover-elevate"
                          }`}
                          onClick={() => toggleFilter(rule.id)}
                          data-testid={`filter-option-${rule.id}`}
                        >
                          <Checkbox
                            checked={selectedFilters.includes(rule.id)}
                            onCheckedChange={() => toggleFilter(rule.id)}
                          />
                          <span className="font-medium">{rule.label}</span>
                        </div>
                      ))}
                    </div>

                    <div className="mt-6 pt-6 border-t">
                      <h4 className="font-medium mb-4">Docking Method</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <button
                          onClick={() => setDockingMethod("bionemo_diffdock")}
                          className={`p-4 rounded-lg border-2 text-left transition-colors ${
                            dockingMethod === "bionemo_diffdock"
                              ? "border-primary bg-primary/5"
                              : "border-border hover-elevate"
                          }`}
                          data-testid="docking-bionemo"
                        >
                          <h3 className="font-semibold mb-1">BioNeMo DiffDock</h3>
                          <p className="text-xs text-muted-foreground">
                            AI-powered docking
                          </p>
                        </button>
                        <button
                          onClick={() => setDockingMethod("external_docking")}
                          className={`p-4 rounded-lg border-2 text-left transition-colors ${
                            dockingMethod === "external_docking"
                              ? "border-primary bg-primary/5"
                              : "border-border hover-elevate"
                          }`}
                          data-testid="docking-external"
                        >
                          <h3 className="font-semibold mb-1">External Service</h3>
                          <p className="text-xs text-muted-foreground">
                            Use external docking
                          </p>
                        </button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {currentStep === 5 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Scoring Weights</CardTitle>
                    <CardDescription>
                      Configure how the oracle score is calculated
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label>Docking Score Weight</Label>
                          <span className="text-sm font-mono">{wDocking.toFixed(2)}</span>
                        </div>
                        <Slider
                          value={[wDocking]}
                          onValueChange={([v]) => setWDocking(v)}
                          min={0}
                          max={1}
                          step={0.05}
                          data-testid="slider-docking"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label>ADMET Score Weight</Label>
                          <span className="text-sm font-mono">{wAdmet.toFixed(2)}</span>
                        </div>
                        <Slider
                          value={[wAdmet]}
                          onValueChange={([v]) => setWAdmet(v)}
                          min={0}
                          max={1}
                          step={0.05}
                          data-testid="slider-admet"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label>QSAR Score Weight</Label>
                          <span className="text-sm font-mono">{wQsar.toFixed(2)}</span>
                        </div>
                        <Slider
                          value={[wQsar]}
                          onValueChange={([v]) => setWQsar(v)}
                          min={0}
                          max={1}
                          step={0.05}
                          data-testid="slider-qsar"
                        />
                      </div>
                    </div>

                    <div className="p-4 bg-muted/50 rounded-md">
                      <p className="text-sm text-muted-foreground">
                        Total: <span className="font-mono font-medium">{(wDocking + wAdmet + wQsar).toFixed(2)}</span>
                        {Math.abs(wDocking + wAdmet + wQsar - 1) > 0.01 && (
                          <span className="text-amber-600 dark:text-amber-400 ml-2">
                            (Weights should sum to 1.0)
                          </span>
                        )}
                      </p>
                    </div>

                    <div className="pt-6 border-t">
                      <div className="flex items-center gap-3 mb-4">
                        <Checkbox
                          id="enable-quantum"
                          checked={enableQuantum}
                          onCheckedChange={(checked) => setEnableQuantum(checked === true)}
                          data-testid="checkbox-enable-quantum"
                        />
                        <div>
                          <Label htmlFor="enable-quantum" className="text-base font-semibold cursor-pointer">
                            Enable Quantum Optimization
                          </Label>
                          <p className="text-xs text-muted-foreground">
                            Use quantum computing for combinatorial molecule selection (Advanced)
                          </p>
                        </div>
                      </div>

                      {enableQuantum && (
                        <div className="ml-6 space-y-4 p-4 border rounded-md bg-muted/30">
                          <div className="space-y-2">
                            <Label htmlFor="quantum-objective">Optimization Objective</Label>
                            <Select value={quantumObjective} onValueChange={setQuantumObjective}>
                              <SelectTrigger data-testid="select-quantum-objective">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="maximize_oracle_score">Maximize Oracle Score</SelectItem>
                                <SelectItem value="maximize_diversity">Maximize Diversity</SelectItem>
                                <SelectItem value="balanced">Balanced Selection</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="quantum-max">Max Molecules to Select</Label>
                            <Input
                              id="quantum-max"
                              type="number"
                              value={quantumMaxMolecules}
                              onChange={(e) => setQuantumMaxMolecules(parseInt(e.target.value) || 200)}
                              min={10}
                              max={1000}
                              data-testid="input-quantum-max"
                            />
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Quantum optimization is most effective for large candidate pools ({">"}100 molecules)
                          </p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {currentStep === 6 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="h-5 w-5 text-primary" />
                      Virtual Screening & Docking
                    </CardTitle>
                    <CardDescription>
                      Configure screening methods and scoring weights for hit identification
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div>
                      <Label className="text-base font-semibold mb-4 block">Select Screening Methods</Label>
                      <div className="space-y-3">
                        <div
                          onClick={() => !vsDualScreening && setVsAqaffinity(!vsAqaffinity)}
                          className={`flex items-start gap-4 p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                            vsAqaffinity
                              ? "border-primary bg-primary/5"
                              : "border-border hover-elevate"
                          } ${vsDualScreening ? "opacity-80" : ""}`}
                          data-testid="vs-option-aqaffinity"
                        >
                          <Checkbox
                            checked={vsAqaffinity}
                            onCheckedChange={(checked) => !vsDualScreening && setVsAqaffinity(checked === true)}
                            disabled={vsDualScreening}
                            data-testid="checkbox-vs-aqaffinity"
                          />
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h3 className="font-semibold">AQAffinity</h3>
                              <Badge variant="secondary" className="text-xs">GPU</Badge>
                              <Badge className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">Structure-free</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">
                              SandboxAQ's AI model for fast binding affinity prediction. No protein structure required.
                            </p>
                            <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              <span>Estimated: 4 hours</span>
                            </div>
                          </div>
                        </div>

                        <div
                          onClick={() => !vsDualScreening && setVsAutodock(!vsAutodock)}
                          className={`flex items-start gap-4 p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                            vsAutodock
                              ? "border-primary bg-primary/5"
                              : "border-border hover-elevate"
                          } ${vsDualScreening ? "opacity-80" : ""}`}
                          data-testid="vs-option-autodock"
                        >
                          <Checkbox
                            checked={vsAutodock}
                            onCheckedChange={(checked) => !vsDualScreening && setVsAutodock(checked === true)}
                            disabled={vsDualScreening}
                            data-testid="checkbox-vs-autodock"
                          />
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h3 className="font-semibold">AutoDock Vina</h3>
                              <Badge variant="secondary" className="text-xs">CPU</Badge>
                              <Badge variant="outline" className="text-xs">Structure-based</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">
                              Physics-based molecular docking. Requires protein structure (PDB).
                            </p>
                            <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              <span>Estimated: 48 hours</span>
                            </div>
                          </div>
                        </div>

                        <div
                          onClick={() => handleDualScreeningToggle(!vsDualScreening)}
                          className={`flex items-start gap-4 p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                            vsDualScreening
                              ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                              : "border-border hover-elevate"
                          }`}
                          data-testid="vs-option-dual"
                        >
                          <Checkbox
                            checked={vsDualScreening}
                            onCheckedChange={(checked) => handleDualScreeningToggle(checked === true)}
                            data-testid="checkbox-vs-dual"
                          />
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h3 className="font-semibold">Dual Screening + Comparison</h3>
                              <Badge className="text-xs bg-emerald-600 text-white">RECOMMENDED</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">
                              Run both methods and compare results. Consensus scoring improves hit quality.
                            </p>
                            <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
                              <Layers className="h-3 w-3" />
                              <span>Parallel execution with consensus analysis</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 border-t">
                      <Label className="text-base font-semibold mb-4 block">Scoring Weights</Label>
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between mb-2">
                            <Label className="flex items-center gap-2">
                              <Target className="h-4 w-4 text-primary" />
                              Binding Affinity
                            </Label>
                            <span className="text-sm font-mono">{(vsBindingAffinity * 100).toFixed(0)}%</span>
                          </div>
                          <Slider
                            value={[vsBindingAffinity]}
                            onValueChange={([v]) => setVsBindingAffinity(v)}
                            min={0}
                            max={1}
                            step={0.05}
                            data-testid="slider-vs-binding"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <Label className="flex items-center gap-2">
                              <FlaskConical className="h-4 w-4 text-chart-2" />
                              ADMET Properties
                            </Label>
                            <span className="text-sm font-mono">{(vsAdmetProperties * 100).toFixed(0)}%</span>
                          </div>
                          <Slider
                            value={[vsAdmetProperties]}
                            onValueChange={([v]) => setVsAdmetProperties(v)}
                            min={0}
                            max={1}
                            step={0.05}
                            data-testid="slider-vs-admet"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <Label className="flex items-center gap-2">
                              <Layers className="h-4 w-4 text-chart-3" />
                              Selectivity
                            </Label>
                            <span className="text-sm font-mono">{(vsSelectivity * 100).toFixed(0)}%</span>
                          </div>
                          <Slider
                            value={[vsSelectivity]}
                            onValueChange={([v]) => setVsSelectivity(v)}
                            min={0}
                            max={1}
                            step={0.05}
                            data-testid="slider-vs-selectivity"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <Label className="flex items-center gap-2">
                              <Beaker className="h-4 w-4 text-chart-4" />
                              Synthetic Accessibility
                            </Label>
                            <span className="text-sm font-mono">{(vsSyntheticAccessibility * 100).toFixed(0)}%</span>
                          </div>
                          <Slider
                            value={[vsSyntheticAccessibility]}
                            onValueChange={([v]) => setVsSyntheticAccessibility(v)}
                            min={0}
                            max={1}
                            step={0.05}
                            data-testid="slider-vs-synthetic"
                          />
                        </div>
                      </div>

                      <div className="mt-4 p-3 bg-muted/50 rounded-md">
                        <p className="text-sm text-muted-foreground">
                          Total: <span className="font-mono font-medium">{(getTotalVsWeight() * 100).toFixed(0)}%</span>
                          {Math.abs(getTotalVsWeight() - 1) > 0.01 && (
                            <span className="text-amber-600 dark:text-amber-400 ml-2">
                              (Weights should sum to 100%)
                            </span>
                          )}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {currentStep === 7 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Eye className="h-5 w-5 text-primary" />
                      Review & Launch
                    </CardTitle>
                    <CardDescription>
                      Review your campaign configuration before launching
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-muted/50">
                        <h4 className="text-sm font-medium text-muted-foreground mb-2">Campaign</h4>
                        <p className="font-semibold">{name || "Untitled"}</p>
                        <p className="text-sm text-muted-foreground mt-1">{domainType}</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <h4 className="text-sm font-medium text-muted-foreground mb-2">Targets</h4>
                        <p className="font-semibold">
                          {templateTargets.length > 0 
                            ? `${templateTargets.length} template targets`
                            : `${selectedTargets.length} selected`}
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <h4 className="text-sm font-medium text-muted-foreground mb-2">Compound Source</h4>
                        <p className="font-semibold">
                          {generator === "bionemo_molmim" ? "BioNeMo MolMIM" : 
                           generator === "curated_library" ? "Curated Library" : "Upload Library"}
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <h4 className="text-sm font-medium text-muted-foreground mb-2">Filters</h4>
                        <p className="font-semibold">{selectedFilters.length} filters applied</p>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-3 flex items-center gap-2">
                        <Zap className="h-4 w-4 text-primary" />
                        Virtual Screening Configuration
                      </h4>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between py-2 border-b">
                          <span className="text-sm">Methods</span>
                          <div className="flex gap-2">
                            {vsAqaffinity && (
                              <Badge variant="secondary" className="text-xs">AQAffinity (GPU)</Badge>
                            )}
                            {vsAutodock && (
                              <Badge variant="secondary" className="text-xs">AutoDock (CPU)</Badge>
                            )}
                            {vsDualScreening && (
                              <Badge className="text-xs bg-emerald-600 text-white">Dual Screening</Badge>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center justify-between py-2 border-b">
                          <span className="text-sm">Estimated Time</span>
                          <span className="text-sm font-medium">
                            {vsDualScreening ? "~48 hours (parallel)" : 
                             vsAutodock ? "~48 hours" : "~4 hours"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between py-2">
                          <span className="text-sm">Scoring Weights</span>
                          <span className="text-sm font-mono">
                            {(vsBindingAffinity * 100).toFixed(0)}% / {(vsAdmetProperties * 100).toFixed(0)}% / {(vsSelectivity * 100).toFixed(0)}% / {(vsSyntheticAccessibility * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {enableQuantum && (
                      <div className="border rounded-lg p-4 bg-purple-50 dark:bg-purple-900/20">
                        <h4 className="font-medium mb-2 flex items-center gap-2">
                          <Cpu className="h-4 w-4 text-purple-600" />
                          Quantum Optimization Enabled
                        </h4>
                        <p className="text-sm text-muted-foreground">
                          Objective: {quantumObjective.replace(/_/g, " ")} | Max molecules: {quantumMaxMolecules}
                        </p>
                      </div>
                    )}

                    <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
                      <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">Ready to Launch</h4>
                      <p className="text-sm text-emerald-700 dark:text-emerald-400">
                        Your campaign is configured and ready. Click "Create & Start" to begin virtual screening.
                        Results will be available in Hit Triage once screening is complete.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="flex items-center justify-between pt-4">
                <Button
                  variant="outline"
                  onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                  disabled={currentStep === 0}
                  data-testid="button-previous"
                >
                  Previous
                </Button>

                <div className="flex gap-2">
                  {currentStep === steps.length - 1 ? (
                    <>
                      <Button
                        variant="outline"
                        onClick={handleSubmit}
                        disabled={createMutation.isPending}
                        className="gap-2"
                        data-testid="button-save"
                      >
                        <Save className="h-4 w-4" />
                        Save Draft
                      </Button>
                      <Button
                        onClick={handleSubmit}
                        disabled={createMutation.isPending}
                        className="gap-2"
                        data-testid="button-create-campaign"
                      >
                        <Play className="h-4 w-4" />
                        {createMutation.isPending ? "Creating..." : "Create & Start"}
                      </Button>
                    </>
                  ) : (
                    <Button
                      onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
                      disabled={!canProceed()}
                      className="gap-2"
                      data-testid="button-next"
                    >
                      Next
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
