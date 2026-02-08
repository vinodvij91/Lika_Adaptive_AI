import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useLocation } from "wouter";
import {
  Calculator,
  Zap,
  Activity,
  Play,
  CheckCircle2,
  Hexagon,
  Atom,
  Layers,
  Loader2,
  Factory,
  TrendingUp,
  ThermometerSun,
  Gauge,
  Sparkles,
  FlaskConical,
  Eye,
} from "lucide-react";

interface PropertyResult {
  property_name: string;
  value: number;
  unit: string;
  confidence: number;
  method: string;
  percentile: number;
}

interface MaterialPredictionResult {
  material_id: string;
  material_type: string;
  smiles?: string;
  descriptors?: Record<string, number>;
  properties: PropertyResult[];
}

interface ManufacturabilityResult {
  material_id: string;
  material_type: string;
  overall_score: number;
  synthesis_feasibility: number;
  cost_factor: number;
  scalability: number;
  environmental_score: number;
  complexity: number;
  recommendations: string[];
}

const SAMPLE_POLYMERS = [
  { name: "Polyethylene (PE)", type: "polymer", smiles: "CC" },
  { name: "Polystyrene (PS)", type: "polymer", smiles: "c1ccccc1CC" },
  { name: "PMMA", type: "polymer", smiles: "CC(C)(C(=O)OC)C" },
  { name: "Nylon-6", type: "polymer", smiles: "NCCCCCC(=O)O" },
  { name: "PET", type: "polymer", smiles: "c1ccc(C(=O)OCCO)cc1" },
  { name: "Polyimide", type: "polymer", smiles: "c1cc2c(cc1)C(=O)NC2=O" },
];

const SAMPLE_CRYSTALS = [
  { name: "Iron Oxide", type: "crystal", formula: "Fe2O3" },
  { name: "Silicon Dioxide", type: "crystal", formula: "SiO2" },
  { name: "Titanium Dioxide", type: "crystal", formula: "TiO2" },
  { name: "Aluminum Oxide", type: "crystal", formula: "Al2O3" },
  { name: "Zinc Oxide", type: "crystal", formula: "ZnO" },
  { name: "Copper Oxide", type: "crystal", formula: "CuO" },
];

const PROPERTY_ICONS: Record<string, any> = {
  thermal_conductivity: ThermometerSun,
  tensile_strength: Gauge,
  youngs_modulus: Activity,
  density: Hexagon,
  glass_transition: ThermometerSun,
  bandgap: Zap,
};

const PROPERTY_COLORS: Record<string, string> = {
  thermal_conductivity: "from-orange-500 to-red-500",
  tensile_strength: "from-blue-500 to-indigo-500",
  youngs_modulus: "from-violet-500 to-purple-500",
  density: "from-emerald-500 to-teal-500",
  glass_transition: "from-amber-500 to-orange-500",
  bandgap: "from-cyan-500 to-blue-500",
};

function MoleculeViewer({ smiles, name }: { smiles: string; name?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!canvasRef.current || !smiles) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    ctx.fillStyle = "hsl(var(--card))";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const atoms: { x: number; y: number; symbol: string; color: string }[] = [];
    const bonds: { from: number; to: number; order: number }[] = [];
    
    const atomColors: Record<string, string> = {
      C: "#404040",
      O: "#ff4444",
      N: "#4444ff",
      S: "#ffcc00",
      F: "#00ff00",
      Cl: "#00cc00",
      Br: "#aa4400",
      P: "#ff8800",
      Si: "#cccccc",
      default: "#666666"
    };
    
    let currentAtom = 0;
    let ringStart = -1;
    const ringStack: number[] = [];
    let x = 80, y = 120;
    let angle = 0;
    const bondLength = 30;
    
    const parseSmiles = smiles.replace(/\[|\]/g, "");
    let i = 0;
    
    while (i < parseSmiles.length) {
      const char = parseSmiles[i];
      
      if (/[A-Z]/.test(char)) {
        let symbol = char;
        if (i + 1 < parseSmiles.length && /[a-z]/.test(parseSmiles[i + 1])) {
          symbol += parseSmiles[i + 1];
          i++;
        }
        
        atoms.push({
          x: x + (Math.random() - 0.5) * 5,
          y: y + (Math.random() - 0.5) * 5,
          symbol,
          color: atomColors[symbol] || atomColors.default
        });
        
        if (currentAtom > 0) {
          bonds.push({ from: currentAtom - 1, to: currentAtom, order: 1 });
        }
        
        angle += (Math.random() - 0.5) * 1.2 + 0.5;
        x += Math.cos(angle) * bondLength;
        y += Math.sin(angle) * bondLength;
        
        if (x < 40) x = 40;
        if (x > 260) x = 260;
        if (y < 40) y = 40;
        if (y > 200) y = 200;
        
        currentAtom++;
      } else if (char === "(") {
        ringStack.push(currentAtom - 1);
      } else if (char === ")") {
        const branchStart = ringStack.pop();
        if (branchStart !== undefined && atoms[branchStart]) {
          x = atoms[branchStart].x;
          y = atoms[branchStart].y;
        }
      } else if (char === "1" || char === "2" || char === "3") {
        if (ringStart === -1) {
          ringStart = currentAtom - 1;
        } else {
          bonds.push({ from: ringStart, to: currentAtom - 1, order: 1 });
          ringStart = -1;
        }
      } else if (char === "=") {
        if (bonds.length > 0) {
          bonds[bonds.length - 1].order = 2;
        }
      }
      
      i++;
    }
    
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    
    bonds.forEach(bond => {
      const from = atoms[bond.from];
      const to = atoms[bond.to];
      if (!from || !to) return;
      
      ctx.strokeStyle = "#888888";
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      
      if (bond.order === 2) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const nx = -dy / len * 4;
        const ny = dx / len * 4;
        
        ctx.beginPath();
        ctx.moveTo(from.x + nx, from.y + ny);
        ctx.lineTo(to.x + nx, to.y + ny);
        ctx.stroke();
      }
    });
    
    atoms.forEach(atom => {
      ctx.fillStyle = atom.color;
      ctx.beginPath();
      ctx.arc(atom.x, atom.y, atom.symbol === "C" ? 4 : 8, 0, Math.PI * 2);
      ctx.fill();
      
      if (atom.symbol !== "C") {
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(atom.symbol, atom.x, atom.y);
      }
    });
    
  }, [smiles]);

  if (!smiles) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p className="text-sm">No structure to display</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-3">
      <canvas 
        ref={canvasRef} 
        width={300} 
        height={240}
        className="rounded-lg border bg-card"
        data-testid="molecule-canvas"
      />
      {name && (
        <p className="text-sm font-medium text-center">{name}</p>
      )}
      <code className="text-xs text-muted-foreground font-mono bg-muted px-2 py-1 rounded max-w-full truncate">
        {smiles}
      </code>
    </div>
  );
}

function PropertyCard({ prop }: { prop: PropertyResult }) {
  const Icon = PROPERTY_ICONS[prop.property_name] || Calculator;
  const gradient = PROPERTY_COLORS[prop.property_name] || "from-gray-500 to-gray-600";
  
  const formatName = (name: string) => {
    return name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow">
      <div className={`h-1 bg-gradient-to-r ${gradient}`} />
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center text-white`}>
              <Icon className="h-4 w-4" />
            </div>
            <span className="font-medium text-sm">{formatName(prop.property_name)}</span>
          </div>
          <Badge variant="outline" className="text-xs">
            {prop.method.toUpperCase()}
          </Badge>
        </div>
        
        <div className="flex items-baseline gap-2 mb-3">
          <span className="text-2xl font-bold font-mono">
            {typeof prop.value === "number" ? prop.value.toFixed(2) : prop.value}
          </span>
          <span className="text-sm text-muted-foreground">{prop.unit}</span>
        </div>
        
        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 rounded bg-muted/50 text-center">
            <div className="text-sm font-mono font-medium">{Math.round(prop.confidence * 100)}%</div>
            <div className="text-xs text-muted-foreground">Confidence</div>
          </div>
          <div className="p-2 rounded bg-muted/50 text-center">
            <div className="text-sm font-mono font-medium">P{Math.round(prop.percentile)}</div>
            <div className="text-xs text-muted-foreground">Percentile</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ManufacturabilityCard({ result }: { result: ManufacturabilityResult }) {
  const scoreColor = result.overall_score >= 0.8 ? "text-emerald-600" : 
                     result.overall_score >= 0.6 ? "text-amber-600" : "text-red-600";
  
  return (
    <Card className="shadow-lg">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Factory className="h-5 w-5 text-violet-500" />
          Manufacturability Assessment
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-4 rounded-lg bg-gradient-to-r from-violet-500/5 to-purple-500/5 border">
          <div>
            <p className="text-sm text-muted-foreground">Overall Score</p>
            <p className={`text-4xl font-bold ${scoreColor}`}>
              {(result.overall_score * 100).toFixed(0)}%
            </p>
          </div>
          <div className="w-24 h-24 relative">
            <svg className="w-full h-full transform -rotate-90">
              <circle cx="48" cy="48" r="40" fill="none" stroke="currentColor" className="text-muted/20" strokeWidth="8" />
              <circle 
                cx="48" cy="48" r="40" fill="none" stroke="currentColor" 
                className={scoreColor.replace("text-", "stroke-")}
                strokeWidth="8" 
                strokeDasharray={`${result.overall_score * 251} 251`}
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Synthesis</span>
              <span className="text-sm font-mono font-medium">{(result.synthesis_feasibility * 100).toFixed(0)}%</span>
            </div>
            <Progress value={result.synthesis_feasibility * 100} className="h-2" />
          </div>
          <div className="p-3 rounded-lg border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Cost Factor</span>
              <span className="text-sm font-mono font-medium">{(result.cost_factor * 100).toFixed(0)}%</span>
            </div>
            <Progress value={result.cost_factor * 100} className="h-2" />
          </div>
          <div className="p-3 rounded-lg border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Scalability</span>
              <span className="text-sm font-mono font-medium">{(result.scalability * 100).toFixed(0)}%</span>
            </div>
            <Progress value={result.scalability * 100} className="h-2" />
          </div>
          <div className="p-3 rounded-lg border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Environmental</span>
              <span className="text-sm font-mono font-medium">{(result.environmental_score * 100).toFixed(0)}%</span>
            </div>
            <Progress value={result.environmental_score * 100} className="h-2" />
          </div>
        </div>
        
        {result.recommendations.length > 0 && (
          <div className="p-3 rounded-lg bg-muted/30">
            <p className="text-sm font-medium mb-2">Recommendations</p>
            <ul className="space-y-1">
              {result.recommendations.map((rec, i) => (
                <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function PropertyPredictionPage() {
  const [location] = useLocation();
  const [materialType, setMaterialType] = useState<"polymer" | "crystal">("polymer");
  const [inputText, setInputText] = useState("");
  const [activeTab, setActiveTab] = useState("predict");
  const [results, setResults] = useState<MaterialPredictionResult[]>([]);
  const [manufResults, setManufResults] = useState<ManufacturabilityResult[]>([]);
  const [selectedMaterial, setSelectedMaterial] = useState<MaterialPredictionResult | null>(null);
  const [currentMaterialName, setCurrentMaterialName] = useState("");
  const [currentSmiles, setCurrentSmiles] = useState("");
  const [aiAnalysis, setAiAnalysis] = useState<any>(null);
  const [screeningInput, setScreeningInput] = useState("");
  const [screeningConstraints, setScreeningConstraints] = useState<{
    thermal_conductivity?: { min?: number; max?: number };
    tensile_strength?: { min?: number; max?: number };
    youngs_modulus?: { min?: number; max?: number };
    density?: { min?: number; max?: number };
    glass_transition?: { min?: number; max?: number };
  }>({});
  const [screeningResults, setScreeningResults] = useState<{
    passed: MaterialPredictionResult[];
    failed: MaterialPredictionResult[];
  } | null>(null);
  const [isScreening, setIsScreening] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const smiles = params.get("smiles");
    const name = params.get("name");
    const type = params.get("type");
    
    if (smiles) {
      setInputText(smiles);
      setCurrentSmiles(smiles);
    }
    if (name) {
      setCurrentMaterialName(name);
    }
    if (type === "crystal") {
      setMaterialType("crystal");
    } else if (type) {
      setMaterialType("polymer");
    }
  }, [location]);

  const { data: computeNodes } = useQuery<any[]>({
    queryKey: ["/api/compute-nodes"],
  });

  const { data: materialsResponse } = useQuery<{ materials: any[], total: number }>({
    queryKey: ["/api/materials"],
  });
  const userMaterials = materialsResponse?.materials || [];

  const onlineNodes = computeNodes?.filter((n) => n.status === "active") || [];

  const predictMutation = useMutation({
    mutationFn: async (materials: any[]) => {
      const res = await apiRequest("POST", "/api/compute/materials/predict", { materials });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.results) {
        setResults(data.results);
        if (data.results.length > 0) {
          setSelectedMaterial(data.results[0]);
        }
      }
    },
  });

  const manufacturabilityMutation = useMutation({
    mutationFn: async (materials: any[]) => {
      const res = await apiRequest("POST", "/api/compute/materials/manufacturability", { materials });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.results) {
        setManufResults(data.results);
      }
    },
  });

  const aiAnalysisMutation = useMutation({
    mutationFn: async ({ smiles, properties, materialName }: { smiles: string; properties: any[]; materialName?: string }) => {
      const res = await apiRequest("POST", "/api/compute/materials/ai-analysis", { smiles, properties, materialName });
      return res.json();
    },
    onSuccess: (data) => {
      if (data.analysis) {
        setAiAnalysis(data);
      }
    },
  });

  const handlePredict = async () => {
    let materials: any[] = [];
    
    if (inputText.trim()) {
      const lines = inputText.split("\n").filter(l => l.trim());
      materials = lines.map(line => {
        if (materialType === "polymer") {
          return { type: "polymer", smiles: line.trim() };
        } else {
          return { type: "crystal", formula: line.trim() };
        }
      });
    } else {
      const sampleMats = materialType === "polymer" ? SAMPLE_POLYMERS : SAMPLE_CRYSTALS;
      const apiMats = (userMaterials || [])
        .filter((m: any) => m.type === materialType)
        .slice(0, 4)
        .map((m: any) => materialType === "polymer"
          ? { type: m.type, smiles: m.smiles || (m.representation as any)?.smiles }
          : { type: m.type, formula: m.composition || (m.representation as any)?.formula });
      materials = apiMats.length > 0 ? apiMats : sampleMats.slice(0, 4).map(p => materialType === "polymer" ? { type: p.type, smiles: (p as any).smiles } : { type: p.type, formula: (p as any).formula });
    }
    
    setResults([]);
    setManufResults([]);
    await predictMutation.mutateAsync(materials);
    await manufacturabilityMutation.mutateAsync(materials);
  };

  const isLoading = predictMutation.isPending || manufacturabilityMutation.isPending;

  const handleBatchScreen = async () => {
    setIsScreening(true);
    setScreeningResults(null);
    
    let materials: any[] = [];
    
    if (screeningInput.trim()) {
      const lines = screeningInput.split("\n").filter(l => l.trim());
      materials = lines.map(line => {
        if (materialType === "polymer") {
          return { type: "polymer", smiles: line.trim() };
        } else {
          return { type: "crystal", formula: line.trim() };
        }
      });
    } else {
      const sampleMats = materialType === "polymer" ? SAMPLE_POLYMERS : SAMPLE_CRYSTALS;
      const apiMats = (userMaterials || [])
        .filter((m: any) => m.type === materialType)
        .map((m: any) => materialType === "polymer"
          ? { type: m.type, smiles: m.smiles || (m.representation as any)?.smiles }
          : { type: m.type, formula: m.composition || (m.representation as any)?.formula });
      materials = apiMats.length > 0 ? apiMats : sampleMats.map(p => materialType === "polymer" ? { type: p.type, smiles: (p as any).smiles } : { type: p.type, formula: (p as any).formula });
    }
    
    try {
      const res = await apiRequest("POST", "/api/compute/materials/predict", { materials });
      const data = await res.json();
      
      if (data.results) {
        const passed: MaterialPredictionResult[] = [];
        const failed: MaterialPredictionResult[] = [];
        
        data.results.forEach((result: MaterialPredictionResult) => {
          let meetsConstraints = true;
          
          for (const prop of result.properties) {
            const constraint = screeningConstraints[prop.property_name as keyof typeof screeningConstraints];
            if (constraint) {
              if (constraint.min !== undefined && prop.value < constraint.min) {
                meetsConstraints = false;
                break;
              }
              if (constraint.max !== undefined && prop.value > constraint.max) {
                meetsConstraints = false;
                break;
              }
            }
          }
          
          if (meetsConstraints) {
            passed.push(result);
          } else {
            failed.push(result);
          }
        });
        
        setScreeningResults({ passed, failed });
      }
    } catch (error) {
      console.error("Screening error:", error);
    } finally {
      setIsScreening(false);
    }
  };

  const updateConstraint = (property: string, type: "min" | "max", value: string) => {
    const numValue = value === "" ? undefined : parseFloat(value);
    setScreeningConstraints(prev => ({
      ...prev,
      [property]: {
        ...prev[property as keyof typeof prev],
        [type]: numValue
      }
    }));
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-emerald-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-6">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-600 via-teal-500 to-cyan-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxNWwtMTMuMDkgNy41djE1TDMwIDQ1bDEzLjA5LTcuNXYtMTVMMzAgMTV6IiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4xNSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Calculator className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">Property Prediction Engine</h1>
                  <p className="text-emerald-100">ML-powered material property predictions with manufacturability scoring</p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-sm text-emerald-100">
                {onlineNodes.length > 0 ? (
                  <Badge variant="outline" className="bg-emerald-500/20 text-white border-emerald-400/50 gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    {onlineNodes.length} Compute Node{onlineNodes.length !== 1 ? "s" : ""} Online
                  </Badge>
                ) : (
                  <Badge variant="outline" className="bg-amber-500/20 text-white border-amber-400/50">
                    No compute nodes available
                  </Badge>
                )}
                <Badge variant="outline" className="bg-white/20 text-white border-white/30 gap-1">
                  <Sparkles className="h-3 w-3" />
                  Thermal, Mechanical, Electrical
                </Badge>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-4">
              <Card className="shadow-lg">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <FlaskConical className="h-5 w-5 text-emerald-500" />
                    Input Materials
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Tabs value={activeTab} onValueChange={setActiveTab}>
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="predict" data-testid="tab-predict">
                        <Calculator className="h-4 w-4 mr-2" />
                        Properties
                      </TabsTrigger>
                      <TabsTrigger value="screen" data-testid="tab-screen">
                        <TrendingUp className="h-4 w-4 mr-2" />
                        Screen
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="predict" className="space-y-4 mt-4">
                      <div>
                        <Label className="text-sm font-medium mb-2 block">Material Type</Label>
                        <Select value={materialType} onValueChange={(v) => setMaterialType(v as any)}>
                          <SelectTrigger data-testid="select-material-type">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="polymer">
                              <div className="flex items-center gap-2">
                                <Layers className="h-4 w-4" />
                                Polymer (SMILES)
                              </div>
                            </SelectItem>
                            <SelectItem value="crystal">
                              <div className="flex items-center gap-2">
                                <Atom className="h-4 w-4" />
                                Crystal (Formula)
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium mb-2 block">
                          {materialType === "polymer" ? "SMILES (one per line)" : "Formulas (one per line)"}
                        </Label>
                        <Textarea
                          className="h-28 font-mono text-sm"
                          placeholder={materialType === "polymer" 
                            ? "CC\nc1ccccc1CC\nCC(C)(C(=O)OC)C" 
                            : "Fe2O3\nSiO2\nTiO2"}
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          data-testid="input-materials"
                        />
                      </div>

                      {userMaterials.length > 0 && (
                        <div>
                          <Label className="text-xs font-medium text-muted-foreground mb-2 block">Quick Load from Library</Label>
                          <ScrollArea className="h-28 rounded border bg-muted/30">
                            <div className="p-2 space-y-1">
                              {userMaterials.slice(0, 20).map((mat: any) => {
                                const smiles = (mat.representation as any)?.smiles || '';
                                if (!smiles) return null;
                                return (
                                  <div 
                                    key={mat.id}
                                    className="flex items-center justify-between p-1.5 rounded cursor-pointer hover-elevate"
                                    onClick={() => {
                                      setInputText(prev => prev ? prev + '\n' + smiles : smiles);
                                      setCurrentMaterialName(mat.name || '');
                                      setCurrentSmiles(smiles);
                                    }}
                                    data-testid={`button-load-material-${mat.id}`}
                                  >
                                    <div className="flex items-center gap-2 min-w-0">
                                      <FlaskConical className="h-3 w-3 text-emerald-500 shrink-0" />
                                      <span className="text-xs truncate">{mat.name || mat.id}</span>
                                    </div>
                                    <Badge variant="secondary" className="text-[10px] shrink-0">{mat.type?.replace(/_/g, ' ')}</Badge>
                                  </div>
                                );
                              })}
                            </div>
                          </ScrollArea>
                        </div>
                      )}

                      <Button
                        className="w-full gap-2 bg-gradient-to-r from-emerald-500 to-teal-500"
                        onClick={handlePredict}
                        disabled={isLoading || onlineNodes.length === 0}
                        data-testid="button-run-prediction"
                      >
                        {isLoading ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Play className="h-4 w-4" />
                        )}
                        Run Property Prediction
                      </Button>
                    </TabsContent>

                    <TabsContent value="screen" className="space-y-4 mt-4">
                      <p className="text-sm text-muted-foreground">
                        Batch screening mode for high-throughput discovery with target property constraints.
                      </p>
                      
                      <div className="space-y-3">
                        <Label className="text-xs font-medium text-muted-foreground">Property Constraints</Label>
                        <div className="space-y-2">
                          {[
                            { key: "thermal_conductivity", label: "Thermal Conductivity (W/mK)", icon: ThermometerSun },
                            { key: "tensile_strength", label: "Tensile Strength (MPa)", icon: Gauge },
                            { key: "density", label: "Density (g/cm³)", icon: Hexagon },
                          ].map(({ key, label, icon: Icon }) => (
                            <div key={key} className="flex items-center gap-2 p-2 rounded border bg-muted/30">
                              <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
                              <span className="text-xs flex-1 truncate">{label}</span>
                              <input
                                type="number"
                                placeholder="Min"
                                className="w-16 h-7 px-2 text-xs rounded border bg-background"
                                value={screeningConstraints[key as keyof typeof screeningConstraints]?.min ?? ""}
                                onChange={(e) => updateConstraint(key, "min", e.target.value)}
                                data-testid={`input-${key}-min`}
                              />
                              <span className="text-xs text-muted-foreground">-</span>
                              <input
                                type="number"
                                placeholder="Max"
                                className="w-16 h-7 px-2 text-xs rounded border bg-background"
                                value={screeningConstraints[key as keyof typeof screeningConstraints]?.max ?? ""}
                                onChange={(e) => updateConstraint(key, "max", e.target.value)}
                                data-testid={`input-${key}-max`}
                              />
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs font-medium text-muted-foreground">Materials to Screen (one per line)</Label>
                        <Textarea
                          className="h-20 font-mono text-sm"
                          placeholder={materialType === "polymer" 
                            ? "CC\nc1ccccc1CC\nCC(C)(C(=O)OC)C\nNCCCCCC(=O)O" 
                            : "Fe2O3\nSiO2\nTiO2\nAl2O3"}
                          value={screeningInput}
                          onChange={(e) => setScreeningInput(e.target.value)}
                          data-testid="input-screening-materials"
                        />
                      </div>

                      <Button
                        className="w-full gap-2 bg-gradient-to-r from-violet-500 to-purple-500"
                        onClick={handleBatchScreen}
                        disabled={isScreening || onlineNodes.length === 0}
                        data-testid="button-batch-screen"
                      >
                        {isScreening ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <TrendingUp className="h-4 w-4" />
                        )}
                        Run Batch Screening
                      </Button>

                      {screeningResults && (
                        <div className="space-y-2 pt-2 border-t">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">Results:</span>
                            <div className="flex gap-2">
                              <Badge variant="default" className="bg-emerald-500">
                                {screeningResults.passed.length} Passed
                              </Badge>
                              <Badge variant="secondary">
                                {screeningResults.failed.length} Failed
                              </Badge>
                            </div>
                          </div>
                          {screeningResults.passed.length > 0 && (
                            <div className="space-y-1">
                              {screeningResults.passed.map((mat, idx) => (
                                <div key={idx} className="flex items-center gap-2 p-2 rounded bg-emerald-500/10 border border-emerald-500/30">
                                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                                  <span className="text-xs font-mono truncate flex-1">
                                    {mat.smiles || mat.material_id}
                                  </span>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="h-6 text-xs"
                                    onClick={() => {
                                      setResults([mat]);
                                      setSelectedMaterial(mat);
                                      setActiveTab("predict");
                                    }}
                                    data-testid={`button-view-${idx}`}
                                  >
                                    View
                                  </Button>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </TabsContent>
                  </Tabs>

                  <div className="pt-2 border-t">
                    <p className="text-xs text-muted-foreground mb-2">Sample Materials:</p>
                    <div className="flex flex-wrap gap-1">
                      {(materialType === "polymer" ? SAMPLE_POLYMERS : SAMPLE_CRYSTALS).slice(0, 4).map((mat) => (
                        <Button
                          key={mat.name}
                          size="sm"
                          variant="outline"
                          className="text-xs"
                          onClick={() => setInputText(
                            materialType === "polymer" 
                              ? (mat as any).smiles 
                              : (mat as any).formula
                          )}
                          data-testid={`button-demo-${mat.name.toLowerCase().replace(/\s/g, "-")}`}
                        >
                          {mat.name}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {(currentSmiles || inputText) && materialType === "polymer" && (
                <Card className="shadow-lg">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Eye className="h-5 w-5 text-emerald-500" />
                      Structure Visualization
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <MoleculeViewer 
                      smiles={currentSmiles || inputText.split("\n")[0]} 
                      name={currentMaterialName || undefined}
                    />
                  </CardContent>
                </Card>
              )}

              {results.length > 0 && (
                <Card className="shadow-lg">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <Hexagon className="h-5 w-5 text-emerald-500" />
                        Materials
                      </span>
                      <Badge variant="secondary" className="font-mono">
                        {results.length}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-[300px] overflow-auto">
                      {results.map((mat, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            selectedMaterial === mat
                              ? "bg-emerald-500/10 border-emerald-500"
                              : "hover:bg-muted/50"
                          }`}
                          onClick={() => setSelectedMaterial(mat)}
                          data-testid={`result-material-${idx}`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <code className="text-xs font-mono truncate max-w-[160px]">
                              {mat.material_id.slice(0, 12)}...
                            </code>
                            <Badge variant="outline" className="text-xs capitalize">
                              {mat.material_type}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {mat.properties.length} properties predicted
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            <div className="lg:col-span-2 space-y-4">
              {selectedMaterial && selectedMaterial.properties.length > 0 ? (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {selectedMaterial.properties.map((prop, idx) => (
                      <PropertyCard key={idx} prop={prop} />
                    ))}
                  </div>
                  
                  {manufResults.length > 0 && (
                    <ManufacturabilityCard 
                      result={manufResults.find(m => m.material_id === selectedMaterial.material_id) || manufResults[0]} 
                    />
                  )}
                  
                  {materialType === "polymer" && selectedMaterial.smiles && (
                    <Card className="shadow-lg">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg flex items-center gap-2">
                            <Sparkles className="h-5 w-5 text-purple-500" />
                            AI-Powered Analysis
                          </CardTitle>
                          <Button
                            onClick={() => aiAnalysisMutation.mutate({
                              smiles: selectedMaterial.smiles || currentSmiles || inputText.split("\n")[0],
                              properties: selectedMaterial.properties,
                              materialName: currentMaterialName || selectedMaterial.material_id
                            })}
                            disabled={aiAnalysisMutation.isPending}
                            size="sm"
                            className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                            data-testid="button-ai-analysis"
                          >
                            {aiAnalysisMutation.isPending ? (
                              <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Analyzing...
                              </>
                            ) : (
                              <>
                                <Sparkles className="h-4 w-4 mr-2" />
                                Get AI Insights
                              </>
                            )}
                          </Button>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Use GPT-4o to analyze structure-property relationships and get scientific recommendations
                        </p>
                      </CardHeader>
                      {aiAnalysis?.analysis && (
                        <CardContent className="space-y-4">
                          <div className="grid gap-4">
                            {aiAnalysis.analysis.structureAnalysis && (
                              <div className="p-3 bg-purple-500/10 rounded-lg">
                                <p className="font-semibold text-purple-600 dark:text-purple-400 mb-1 flex items-center gap-2">
                                  <Atom className="h-4 w-4" />
                                  Structure Analysis
                                </p>
                                <p className="text-sm">{aiAnalysis.analysis.structureAnalysis}</p>
                              </div>
                            )}
                            {aiAnalysis.analysis.propertyExplanation && (
                              <div className="p-3 bg-blue-500/10 rounded-lg">
                                <p className="font-semibold text-blue-600 dark:text-blue-400 mb-1 flex items-center gap-2">
                                  <Activity className="h-4 w-4" />
                                  Property Explanation
                                </p>
                                <p className="text-sm">{aiAnalysis.analysis.propertyExplanation}</p>
                              </div>
                            )}
                            {aiAnalysis.analysis.applicationSuggestions && (
                              <div className="p-3 bg-emerald-500/10 rounded-lg">
                                <p className="font-semibold text-emerald-600 dark:text-emerald-400 mb-1 flex items-center gap-2">
                                  <Factory className="h-4 w-4" />
                                  Applications
                                </p>
                                <ul className="text-sm list-disc list-inside">
                                  {(Array.isArray(aiAnalysis.analysis.applicationSuggestions) 
                                    ? aiAnalysis.analysis.applicationSuggestions 
                                    : [aiAnalysis.analysis.applicationSuggestions]
                                  ).map((app: string, i: number) => (
                                    <li key={i}>{app}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            {aiAnalysis.analysis.improvementSuggestions && (
                              <div className="p-3 bg-amber-500/10 rounded-lg">
                                <p className="font-semibold text-amber-600 dark:text-amber-400 mb-1 flex items-center gap-2">
                                  <TrendingUp className="h-4 w-4" />
                                  Improvement Suggestions
                                </p>
                                <ul className="text-sm list-disc list-inside">
                                  {(Array.isArray(aiAnalysis.analysis.improvementSuggestions)
                                    ? aiAnalysis.analysis.improvementSuggestions
                                    : [aiAnalysis.analysis.improvementSuggestions]
                                  ).map((sug: string, i: number) => (
                                    <li key={i}>{sug}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground text-right">
                            Powered by {aiAnalysis.model}
                          </p>
                        </CardContent>
                      )}
                    </Card>
                  )}
                </>
              ) : (
                <Card className="shadow-lg">
                  <CardContent className="p-12 text-center">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-5">
                      <Calculator className="h-9 w-9 text-emerald-500" />
                    </div>
                    <p className="font-semibold text-lg mb-2">No predictions yet</p>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Enter materials (SMILES for polymers, formulas for crystals) and run the prediction pipeline to see property results.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
