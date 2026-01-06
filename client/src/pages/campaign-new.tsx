import { useState } from "react";
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
} from "lucide-react";
import type { Project, Target as TargetType, DiseaseArea, PipelineConfig } from "@shared/schema";

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

export default function CampaignNewPage() {
  const [, setLocation] = useLocation();
  const searchParams = useSearch();
  const params = new URLSearchParams(searchParams);
  const projectIdFromUrl = params.get("projectId");
  const { toast } = useToast();

  const [currentStep, setCurrentStep] = useState(0);
  const [name, setName] = useState("");
  const [projectId, setProjectId] = useState(projectIdFromUrl || "");
  const [domainType, setDomainType] = useState<DiseaseArea>("CNS");
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [generator, setGenerator] = useState<"bionemo_molmim" | "upload_library">("bionemo_molmim");
  const [selectedFilters, setSelectedFilters] = useState<string[]>(["lipinski"]);
  const [dockingMethod, setDockingMethod] = useState<"bionemo_diffdock" | "external_docking">("bionemo_diffdock");
  const [wDocking, setWDocking] = useState(0.4);
  const [wAdmet, setWAdmet] = useState(0.3);
  const [wQsar, setWQsar] = useState(0.3);
  const [enableQuantum, setEnableQuantum] = useState(false);
  const [quantumObjective, setQuantumObjective] = useState("maximize_oracle_score");
  const [quantumMaxMolecules, setQuantumMaxMolecules] = useState(200);

  const { data: projects } = useQuery<Project[]>({
    queryKey: ["/api/projects"],
  });

  const { data: targets } = useQuery<TargetType[]>({
    queryKey: ["/api/targets"],
  });

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
    { id: 0, title: "Basic Info", icon: Beaker, completed: !!name && !!projectId },
    { id: 1, title: "Targets", icon: Target, completed: selectedTargets.length > 0 },
    { id: 2, title: "Generator", icon: Sparkles, completed: true },
    { id: 3, title: "Filtering", icon: FlaskConical, completed: true },
    { id: 4, title: "Scoring", icon: Target, completed: true },
  ];

  const handleSubmit = () => {
    const pipelineConfig: PipelineConfig = {
      generator,
      filteringRules: selectedFilters,
      dockingMethod,
      scoringWeights: { wDocking, wAdmet, wQsar },
      targetIds: selectedTargets,
      enableQuantumOptimization: enableQuantum,
      quantumParams: enableQuantum ? {
        objective: quantumObjective,
        maxMolecules: quantumMaxMolecules,
      } : undefined,
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
      case 0: return !!name && !!projectId;
      case 1: return selectedTargets.length > 0;
      default: return true;
    }
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

              {currentStep === 1 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Select Targets</CardTitle>
                    <CardDescription>
                      Choose one or more protein targets for this campaign
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {targets && targets.length > 0 ? (
                      <div className="space-y-2">
                        {targets.map((target) => (
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
                            <div>
                              <p className="font-medium">{target.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {target.uniprotId || "No UniProt ID"}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No targets available. Add targets first.
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}

              {currentStep === 2 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Molecule Generator</CardTitle>
                    <CardDescription>
                      Choose how molecules will be generated or sourced
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
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
                  </CardContent>
                </Card>
              )}

              {currentStep === 3 && (
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

              {currentStep === 4 && (
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
