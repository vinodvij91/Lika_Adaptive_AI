import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  FileStack,
  Beaker,
  Target,
  Hexagon,
  Layers,
  Zap,
  Activity,
  Factory,
  Box,
  Eye,
  CheckCircle,
  ChevronRight,
  Play,
  Save,
  Upload,
  Cpu,
  Atom,
  Gauge,
  Clock,
  Sparkles,
  Database,
  Settings,
  Calculator,
  Thermometer,
  FlaskConical,
} from "lucide-react";
import type { MaterialsProgram, MaterialEntity } from "@shared/schema";

const materialTemplates = [
  { id: "battery", name: "Battery Materials", description: "High-capacity, fast-charging battery components", icon: Zap, color: "text-yellow-500" },
  { id: "polymer", name: "High-Performance Polymers", description: "Thermal-stable, mechanically robust polymers", icon: Hexagon, color: "text-blue-500" },
  { id: "catalyst", name: "Catalysts", description: "Industrial catalysts for chemical processes", icon: FlaskConical, color: "text-green-500" },
  { id: "semiconductor", name: "Semiconductors", description: "Electronic and optoelectronic materials", icon: Cpu, color: "text-purple-500" },
  { id: "membrane", name: "Separation Membranes", description: "Gas separation and filtration membranes", icon: Layers, color: "text-cyan-500" },
  { id: "coating", name: "Protective Coatings", description: "Corrosion-resistant, thermal barrier coatings", icon: Box, color: "text-orange-500" },
  { id: "alloy", name: "Metal Alloys", description: "High-strength, lightweight metal alloys", icon: Atom, color: "text-slate-500" },
  { id: "custom", name: "Custom Template", description: "Define your own material discovery workflow", icon: Settings, color: "text-muted-foreground" },
];

const materialTypes = [
  { id: "polymer", label: "Polymer" },
  { id: "metal_alloy", label: "Metal Alloy" },
  { id: "composite", label: "Composite" },
  { id: "ceramic", label: "Ceramic" },
  { id: "semiconductor", label: "Semiconductor" },
  { id: "crystal", label: "Crystal" },
  { id: "membrane", label: "Membrane" },
  { id: "coating", label: "Coating" },
];

const industryTypes = [
  "Automotive", "Aerospace", "Energy Storage", "Electronics", 
  "Healthcare", "Construction", "Chemical Processing", "Consumer Goods"
];

const targetProperties = [
  { id: "tensile_strength", label: "Tensile Strength", category: "mechanical", unit: "MPa" },
  { id: "hardness", label: "Hardness", category: "mechanical", unit: "HV" },
  { id: "elasticity", label: "Elastic Modulus", category: "mechanical", unit: "GPa" },
  { id: "conductivity", label: "Electrical Conductivity", category: "electrical", unit: "S/m" },
  { id: "dielectric", label: "Dielectric Constant", category: "electrical", unit: "" },
  { id: "melting_point", label: "Melting Point", category: "thermal", unit: "°C" },
  { id: "thermal_conductivity", label: "Thermal Conductivity", category: "thermal", unit: "W/mK" },
  { id: "stability", label: "Chemical Stability", category: "chemical", unit: "" },
  { id: "corrosion", label: "Corrosion Resistance", category: "chemical", unit: "" },
  { id: "density", label: "Density", category: "physical", unit: "g/cm³" },
];

const multiScaleLevels = [
  { id: "atomic", label: "Atomic Level (DFT)", description: "Quantum mechanical calculations", icon: Atom, time: "8-12h" },
  { id: "molecular", label: "Molecular Level (MD)", description: "Molecular dynamics simulations", icon: Hexagon, time: "4-6h" },
  { id: "macro", label: "Macro Level (FEA)", description: "Finite element analysis mesh", icon: Box, time: "2-4h" },
];

interface PipelineStep {
  id: number;
  title: string;
  icon: typeof FileStack;
  completed: boolean;
}

interface MaterialsPipelineConfig {
  template: string;
  materialType: string;
  industry: string;
  targetProperties: string[];
  propertyTargets: Record<string, { min?: number; max?: number; target?: number }>;
  sourceType: "database" | "generate" | "upload";
  selectedMaterials: string[];
  multiScaleLevels: string[];
  propertyPrediction: {
    aqaffinity: boolean;
    dft: boolean;
    dualPrediction: boolean;
    scoringWeights: {
      mechanical: number;
      electrical: number;
      thermal: number;
      chemical: number;
    };
  };
  structurePropertyAnalysis: boolean;
  manufacturabilityAssessment: boolean;
  feaSimulations: boolean;
  feaConfig?: {
    simulationType: string;
    meshDensity: string;
    loadConditions: string[];
  };
}

export default function MaterialsCampaignNewPage() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const [currentStep, setCurrentStep] = useState(0);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [programId, setProgramId] = useState("");
  const [industry, setIndustry] = useState("");
  const [materialType, setMaterialType] = useState("");
  const [selectedProperties, setSelectedProperties] = useState<string[]>([]);
  const [propertyTargets, setPropertyTargets] = useState<Record<string, { min?: number; max?: number; target?: number }>>({});
  const [sourceType, setSourceType] = useState<"database" | "generate" | "upload">("database");
  const [selectedMaterials, setSelectedMaterials] = useState<string[]>([]);
  const [selectedMultiScale, setSelectedMultiScale] = useState<string[]>(["molecular"]);
  
  const [ppAqaffinity, setPpAqaffinity] = useState(true);
  const [ppDft, setPpDft] = useState(false);
  const [ppDualPrediction, setPpDualPrediction] = useState(false);
  const [wMechanical, setWMechanical] = useState(0.3);
  const [wElectrical, setWElectrical] = useState(0.2);
  const [wThermal, setWThermal] = useState(0.25);
  const [wChemical, setWChemical] = useState(0.25);
  
  const [enableStructureProperty, setEnableStructureProperty] = useState(true);
  const [enableManufacturability, setEnableManufacturability] = useState(true);
  const [enableFea, setEnableFea] = useState(false);
  const [feaSimulationType, setFeaSimulationType] = useState("structural");
  const [feaMeshDensity, setFeaMeshDensity] = useState("medium");

  const { data: programs } = useQuery<MaterialsProgram[]>({
    queryKey: ["/api/materials-programs"],
  });

  const { data: materialsResponse } = useQuery<{ materials: MaterialEntity[], total: number }>({
    queryKey: ["/api/materials"],
  });

  const materials = materialsResponse?.materials || [];

  const createMutation = useMutation({
    mutationFn: async (data: { name: string; description: string | null; programId: string | null; industry: string; pipelineConfig: MaterialsPipelineConfig }) => {
      const res = await apiRequest("POST", "/api/materials-campaigns", data);
      return res.json();
    },
    onSuccess: (campaign) => {
      queryClient.invalidateQueries({ queryKey: ["/api/materials-campaigns"] });
      toast({ title: "Campaign created", description: `${campaign.name} is ready to run.` });
      setLocation(`/materials/campaigns/${campaign.id}`);
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to create campaign", variant: "destructive" });
    },
  });

  const steps: PipelineStep[] = [
    { id: 0, title: "Template", icon: FileStack, completed: !!selectedTemplate },
    { id: 1, title: "Basic Info", icon: Beaker, completed: !!name },
    { id: 2, title: "Material Targets", icon: Target, completed: !!materialType && selectedProperties.length > 0 },
    { id: 3, title: "Material Library", icon: Database, completed: sourceType === "generate" || selectedMaterials.length > 0 },
    { id: 4, title: "Multi-Scale", icon: Layers, completed: selectedMultiScale.length > 0 },
    { id: 5, title: "Property Prediction", icon: Zap, completed: ppAqaffinity || ppDft },
    { id: 6, title: "Structure-Property", icon: Activity, completed: true },
    { id: 7, title: "Manufacturability", icon: Factory, completed: true },
    { id: 8, title: "FEA Simulations", icon: Box, completed: true },
    { id: 9, title: "Review & Launch", icon: Eye, completed: false },
  ];

  const handleSubmit = () => {
    const pipelineConfig: MaterialsPipelineConfig = {
      template: selectedTemplate || "custom",
      materialType,
      industry,
      targetProperties: selectedProperties,
      propertyTargets,
      sourceType,
      selectedMaterials,
      multiScaleLevels: selectedMultiScale,
      propertyPrediction: {
        aqaffinity: ppAqaffinity,
        dft: ppDft,
        dualPrediction: ppDualPrediction,
        scoringWeights: {
          mechanical: wMechanical,
          electrical: wElectrical,
          thermal: wThermal,
          chemical: wChemical,
        },
      },
      structurePropertyAnalysis: enableStructureProperty,
      manufacturabilityAssessment: enableManufacturability,
      feaSimulations: enableFea,
      feaConfig: enableFea ? {
        simulationType: feaSimulationType,
        meshDensity: feaMeshDensity,
        loadConditions: [],
      } : undefined,
    };

    createMutation.mutate({
      name,
      description: description || null,
      programId: programId || null,
      industry,
      pipelineConfig,
    });
  };

  const handleDualPredictionToggle = (enabled: boolean) => {
    setPpDualPrediction(enabled);
    if (enabled) {
      setPpAqaffinity(true);
      setPpDft(true);
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 0: return !!selectedTemplate;
      case 1: return !!name;
      case 2: return !!materialType && selectedProperties.length > 0;
      case 3: return sourceType === "generate" || selectedMaterials.length > 0;
      case 4: return selectedMultiScale.length > 0;
      case 5: return ppAqaffinity || ppDft;
      default: return true;
    }
  };

  const toggleProperty = (propId: string) => {
    setSelectedProperties(prev =>
      prev.includes(propId) ? prev.filter(id => id !== propId) : [...prev, propId]
    );
  };

  const toggleMaterial = (matId: string) => {
    setSelectedMaterials(prev =>
      prev.includes(matId) ? prev.filter(id => id !== matId) : [...prev, matId]
    );
  };

  const toggleMultiScale = (level: string) => {
    setSelectedMultiScale(prev =>
      prev.includes(level) ? prev.filter(l => l !== level) : [...prev, level]
    );
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold mb-2">Select Campaign Template</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Choose a pre-configured template or start with a custom workflow
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {materialTemplates.map(template => {
                const Icon = template.icon;
                const isSelected = selectedTemplate === template.id;
                return (
                  <Card 
                    key={template.id}
                    className={`cursor-pointer hover-elevate ${isSelected ? "ring-2 ring-primary" : ""}`}
                    onClick={() => setSelectedTemplate(template.id)}
                    data-testid={`template-${template.id}`}
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-md bg-muted ${template.color}`}>
                          <Icon className="h-5 w-5" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium">{template.name}</h4>
                          <p className="text-xs text-muted-foreground mt-1">{template.description}</p>
                        </div>
                        {isSelected && <CheckCircle className="h-5 w-5 text-primary" />}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        );

      case 1:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Campaign Information</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Provide basic details about your materials discovery campaign
              </p>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Campaign Name</Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., High-Conductivity Battery Electrolytes"
                  data-testid="input-campaign-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description (optional)</Label>
                <Textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe the goals and objectives of this campaign..."
                  className="resize-none"
                  data-testid="input-description"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="program">Program (optional)</Label>
                <Select value={programId} onValueChange={setProgramId}>
                  <SelectTrigger data-testid="select-program">
                    <SelectValue placeholder="Select a program" />
                  </SelectTrigger>
                  <SelectContent>
                    {programs?.map(prog => (
                      <SelectItem key={prog.id} value={prog.id}>{prog.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="industry">Industry</Label>
                <Select value={industry} onValueChange={setIndustry}>
                  <SelectTrigger data-testid="select-industry">
                    <SelectValue placeholder="Select target industry" />
                  </SelectTrigger>
                  <SelectContent>
                    {industryTypes.map(ind => (
                      <SelectItem key={ind} value={ind}>{ind}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Define Material Requirements</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Specify the material type and target properties for your discovery
              </p>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Material Type</Label>
                <div className="grid grid-cols-4 gap-2">
                  {materialTypes.map(type => (
                    <Button
                      key={type.id}
                      variant={materialType === type.id ? "default" : "outline"}
                      size="sm"
                      onClick={() => setMaterialType(type.id)}
                      data-testid={`material-type-${type.id}`}
                    >
                      {type.label}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Target Properties</Label>
                <p className="text-xs text-muted-foreground">Select properties to optimize</p>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  {targetProperties.map(prop => (
                    <div
                      key={prop.id}
                      className={`flex items-center gap-3 p-3 rounded-md border cursor-pointer hover-elevate ${
                        selectedProperties.includes(prop.id) ? "border-primary bg-primary/5" : ""
                      }`}
                      onClick={() => toggleProperty(prop.id)}
                      data-testid={`property-${prop.id}`}
                    >
                      <Checkbox checked={selectedProperties.includes(prop.id)} />
                      <div className="flex-1">
                        <p className="text-sm font-medium">{prop.label}</p>
                        <p className="text-xs text-muted-foreground">{prop.category} {prop.unit && `(${prop.unit})`}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Upload className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">Structure Upload</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Upload crystal structure (CIF) or molecular structure files in the Material Library step
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Material Library / Generator</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Select materials from existing databases or generate new variants
              </p>
            </div>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <Card 
                  className={`cursor-pointer hover-elevate ${sourceType === "database" ? "ring-2 ring-primary" : ""}`}
                  onClick={() => setSourceType("database")}
                  data-testid="source-database"
                >
                  <CardContent className="pt-4 text-center">
                    <Database className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                    <h4 className="font-medium">Existing Database</h4>
                    <p className="text-xs text-muted-foreground mt-1">Select from materials library</p>
                  </CardContent>
                </Card>
                <Card 
                  className={`cursor-pointer hover-elevate ${sourceType === "generate" ? "ring-2 ring-primary" : ""}`}
                  onClick={() => setSourceType("generate")}
                  data-testid="source-generate"
                >
                  <CardContent className="pt-4 text-center">
                    <Sparkles className="h-8 w-8 mx-auto mb-2 text-purple-500" />
                    <h4 className="font-medium">Generate Variants</h4>
                    <p className="text-xs text-muted-foreground mt-1">AI-generated substitutions & doping</p>
                  </CardContent>
                </Card>
                <Card 
                  className={`cursor-pointer hover-elevate ${sourceType === "upload" ? "ring-2 ring-primary" : ""}`}
                  onClick={() => setSourceType("upload")}
                  data-testid="source-upload"
                >
                  <CardContent className="pt-4 text-center">
                    <Upload className="h-8 w-8 mx-auto mb-2 text-green-500" />
                    <h4 className="font-medium">Upload Custom</h4>
                    <p className="text-xs text-muted-foreground mt-1">Upload material compositions</p>
                  </CardContent>
                </Card>
              </div>

              {sourceType === "database" && materials.length > 0 && (
                <div className="space-y-2">
                  <Label>Select Materials ({selectedMaterials.length} selected)</Label>
                  <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
                    {materials.slice(0, 20).map(mat => (
                      <div
                        key={mat.id}
                        className={`flex items-center gap-2 p-2 rounded-md border cursor-pointer hover-elevate ${
                          selectedMaterials.includes(mat.id) ? "border-primary bg-primary/5" : ""
                        }`}
                        onClick={() => toggleMaterial(mat.id)}
                        data-testid={`material-${mat.id}`}
                      >
                        <Checkbox checked={selectedMaterials.includes(mat.id)} />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{mat.name}</p>
                          <p className="text-xs text-muted-foreground">{mat.type || "Material"}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {sourceType === "generate" && (
                <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10">
                  <CardContent className="pt-4">
                    <div className="flex items-center gap-3">
                      <Sparkles className="h-8 w-8 text-purple-500" />
                      <div>
                        <h4 className="font-medium">AI Variant Generation</h4>
                        <p className="text-sm text-muted-foreground">
                          Will generate ~10,000 variants through element substitution, doping, and compositional optimization
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Multi-Scale Representation</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Generate representations at different scales for comprehensive analysis
              </p>
            </div>
            <div className="space-y-4">
              {multiScaleLevels.map(level => {
                const Icon = level.icon;
                const isSelected = selectedMultiScale.includes(level.id);
                return (
                  <Card 
                    key={level.id}
                    className={`cursor-pointer hover-elevate ${isSelected ? "ring-2 ring-primary" : ""}`}
                    onClick={() => toggleMultiScale(level.id)}
                    data-testid={`scale-${level.id}`}
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-center gap-4">
                        <Checkbox checked={isSelected} />
                        <div className="p-2 rounded-md bg-muted">
                          <Icon className="h-5 w-5" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium">{level.label}</h4>
                          <p className="text-sm text-muted-foreground">{level.description}</p>
                        </div>
                        <Badge variant="secondary">
                          <Clock className="h-3 w-3 mr-1" />
                          {level.time}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Property Prediction (AQAffinity)</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Configure AI and physics-based prediction methods for material properties
              </p>
            </div>

            <Card className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border-yellow-200 dark:border-yellow-800">
              <CardContent className="pt-4">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-yellow-500/20 rounded-md">
                    <Zap className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold">Core Value: Property Prediction</h4>
                    <p className="text-sm text-muted-foreground">Predict properties for ALL material variants using AI</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 border rounded-md">
                <div className="flex items-center gap-3">
                  <Switch 
                    checked={ppDualPrediction} 
                    onCheckedChange={handleDualPredictionToggle}
                    data-testid="switch-dual-prediction"
                  />
                  <div>
                    <p className="font-medium">Dual Prediction (Recommended)</p>
                    <p className="text-sm text-muted-foreground">Run both AQAffinity and DFT, compare results</p>
                  </div>
                </div>
                <Badge variant="default">Best Accuracy</Badge>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <Card className={`${ppAqaffinity ? "ring-2 ring-primary" : ""}`}>
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Checkbox 
                          checked={ppAqaffinity} 
                          onCheckedChange={(checked) => setPpAqaffinity(!!checked)}
                          data-testid="checkbox-aqaffinity"
                        />
                        <span className="font-medium">AQAffinity (AI)</span>
                      </div>
                      <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                        <Cpu className="h-3 w-3 mr-1" />
                        GPU
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Structure-free AI prediction using SandboxAQ technology
                    </p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>~4 hours for 10K materials</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className={`${ppDft ? "ring-2 ring-primary" : ""}`}>
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Checkbox 
                          checked={ppDft} 
                          onCheckedChange={(checked) => setPpDft(!!checked)}
                          data-testid="checkbox-dft"
                        />
                        <span className="font-medium">DFT Calculations</span>
                      </div>
                      <Badge variant="secondary">
                        <Calculator className="h-3 w-3 mr-1" />
                        CPU
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Physics-based density functional theory calculations
                    </p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>~48 hours for 10K materials</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Property Weights</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Mechanical Properties</span>
                      <span className="font-medium">{Math.round(wMechanical * 100)}%</span>
                    </div>
                    <Slider
                      value={[wMechanical]}
                      onValueChange={([v]) => setWMechanical(v)}
                      min={0}
                      max={1}
                      step={0.05}
                      data-testid="slider-mechanical"
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Electrical Properties</span>
                      <span className="font-medium">{Math.round(wElectrical * 100)}%</span>
                    </div>
                    <Slider
                      value={[wElectrical]}
                      onValueChange={([v]) => setWElectrical(v)}
                      min={0}
                      max={1}
                      step={0.05}
                      data-testid="slider-electrical"
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Thermal Properties</span>
                      <span className="font-medium">{Math.round(wThermal * 100)}%</span>
                    </div>
                    <Slider
                      value={[wThermal]}
                      onValueChange={([v]) => setWThermal(v)}
                      min={0}
                      max={1}
                      step={0.05}
                      data-testid="slider-thermal"
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Chemical Properties</span>
                      <span className="font-medium">{Math.round(wChemical * 100)}%</span>
                    </div>
                    <Slider
                      value={[wChemical]}
                      onValueChange={([v]) => setWChemical(v)}
                      min={0}
                      max={1}
                      step={0.05}
                      data-testid="slider-chemical"
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        );

      case 6:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Structure-Property Analysis</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Analyze correlations between material structures and predicted properties
              </p>
            </div>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/20 rounded-md">
                      <Activity className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <h4 className="font-medium">Enable Structure-Property Analysis</h4>
                      <p className="text-sm text-muted-foreground">
                        Identify key structural features that drive performance
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={enableStructureProperty}
                    onCheckedChange={setEnableStructureProperty}
                    data-testid="switch-structure-property"
                  />
                </div>
              </CardContent>
            </Card>

            {enableStructureProperty && (
              <div className="grid grid-cols-3 gap-4">
                <Card className="bg-muted/50">
                  <CardContent className="pt-4 text-center">
                    <Target className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm font-medium">Feature Importance</p>
                    <p className="text-xs text-muted-foreground">Identify key structural features</p>
                  </CardContent>
                </Card>
                <Card className="bg-muted/50">
                  <CardContent className="pt-4 text-center">
                    <Gauge className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm font-medium">Design Rules</p>
                    <p className="text-xs text-muted-foreground">Suggest optimization strategies</p>
                  </CardContent>
                </Card>
                <Card className="bg-muted/50">
                  <CardContent className="pt-4 text-center">
                    <Sparkles className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm font-medium">Next-Gen Candidates</p>
                    <p className="text-xs text-muted-foreground">AI-suggested improvements</p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        );

      case 7:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Manufacturability Assessment</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Evaluate production feasibility, cost, and scalability
              </p>
            </div>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-orange-500/20 rounded-md">
                      <Factory className="h-5 w-5 text-orange-500" />
                    </div>
                    <div>
                      <h4 className="font-medium">Enable Manufacturability Assessment</h4>
                      <p className="text-sm text-muted-foreground">
                        Evaluate if materials can be manufactured at scale
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={enableManufacturability}
                    onCheckedChange={setEnableManufacturability}
                    data-testid="switch-manufacturability"
                  />
                </div>
              </CardContent>
            </Card>

            {enableManufacturability && (
              <div className="grid grid-cols-3 gap-4">
                <Card className="bg-muted/50">
                  <CardContent className="pt-4 text-center">
                    <CheckCircle className="h-6 w-6 mx-auto mb-2 text-green-500" />
                    <p className="text-sm font-medium">Feasibility Check</p>
                    <p className="text-xs text-muted-foreground">Can this be manufactured?</p>
                  </CardContent>
                </Card>
                <Card className="bg-muted/50">
                  <CardContent className="pt-4 text-center">
                    <Gauge className="h-6 w-6 mx-auto mb-2 text-blue-500" />
                    <p className="text-sm font-medium">Cost Estimation</p>
                    <p className="text-xs text-muted-foreground">$/kg production cost</p>
                  </CardContent>
                </Card>
                <Card className="bg-muted/50">
                  <CardContent className="pt-4 text-center">
                    <Activity className="h-6 w-6 mx-auto mb-2 text-purple-500" />
                    <p className="text-sm font-medium">Scalability</p>
                    <p className="text-xs text-muted-foreground">Lab → Pilot → Production</p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        );

      case 8:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">FEA Simulations (Optional)</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Run detailed finite element analysis on top candidates
              </p>
            </div>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-cyan-500/20 rounded-md">
                      <Box className="h-5 w-5 text-cyan-500" />
                    </div>
                    <div>
                      <h4 className="font-medium">Enable FEA Simulations</h4>
                      <p className="text-sm text-muted-foreground">
                        Validate mechanical properties under realistic load conditions
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={enableFea}
                    onCheckedChange={setEnableFea}
                    data-testid="switch-fea"
                  />
                </div>
              </CardContent>
            </Card>

            {enableFea && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Simulation Type</Label>
                    <Select value={feaSimulationType} onValueChange={setFeaSimulationType}>
                      <SelectTrigger data-testid="select-fea-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="structural">Structural Analysis</SelectItem>
                        <SelectItem value="thermal">Thermal Analysis</SelectItem>
                        <SelectItem value="cfd">CFD (Fluid Dynamics)</SelectItem>
                        <SelectItem value="coupled">Coupled Thermo-Mechanical</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Mesh Density</Label>
                    <Select value={feaMeshDensity} onValueChange={setFeaMeshDensity}>
                      <SelectTrigger data-testid="select-mesh-density">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="coarse">Coarse (Fast)</SelectItem>
                        <SelectItem value="medium">Medium (Balanced)</SelectItem>
                        <SelectItem value="fine">Fine (High Accuracy)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Card className="bg-muted/50">
                  <CardContent className="pt-4">
                    <p className="text-sm text-muted-foreground">
                      FEA simulations will run on top 20 candidates after property prediction is complete.
                      Estimated time: 2-8 hours depending on complexity.
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        );

      case 9:
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Review & Launch</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Review your configuration and launch the materials discovery campaign
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Campaign Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Name:</span>
                    <span className="font-medium">{name || "Not set"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Template:</span>
                    <span className="font-medium">{selectedTemplate || "Custom"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Industry:</span>
                    <span className="font-medium">{industry || "Not set"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Material Type:</span>
                    <span className="font-medium">{materialType || "Not set"}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Pipeline Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Properties:</span>
                    <span className="font-medium">{selectedProperties.length} selected</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Multi-Scale:</span>
                    <span className="font-medium">{selectedMultiScale.length} levels</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Prediction:</span>
                    <span className="font-medium">
                      {ppDualPrediction ? "Dual" : ppAqaffinity ? "AQAffinity" : "DFT"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">FEA:</span>
                    <span className="font-medium">{enableFea ? "Enabled" : "Disabled"}</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-green-200 dark:border-green-800">
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-500/20 rounded-md">
                    <Play className="h-5 w-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold">Ready to Launch</h4>
                    <p className="text-sm text-muted-foreground">
                      Estimated completion: {ppDualPrediction ? "52 hours" : ppAqaffinity ? "4 hours" : "48 hours"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Materials", href: "/materials" },
          { label: "Campaigns", href: "/materials/campaigns" },
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

            <div className="flex-1">
              <Card>
                <CardContent className="p-6">
                  {renderStepContent()}

                  <div className="flex justify-between mt-8 pt-4 border-t">
                    <Button
                      variant="outline"
                      onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                      disabled={currentStep === 0}
                      data-testid="button-prev"
                    >
                      Previous
                    </Button>

                    {currentStep < steps.length - 1 ? (
                      <Button
                        onClick={() => setCurrentStep(currentStep + 1)}
                        disabled={!canProceed()}
                        data-testid="button-next"
                      >
                        Continue
                        <ChevronRight className="h-4 w-4 ml-1" />
                      </Button>
                    ) : (
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          onClick={() => {
                            toast({ title: "Draft saved", description: "Campaign configuration saved as draft." });
                          }}
                          data-testid="button-save-draft"
                        >
                          <Save className="h-4 w-4 mr-2" />
                          Save Draft
                        </Button>
                        <Button
                          onClick={handleSubmit}
                          disabled={createMutation.isPending}
                          data-testid="button-launch"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          {createMutation.isPending ? "Launching..." : "Launch Campaign"}
                        </Button>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
