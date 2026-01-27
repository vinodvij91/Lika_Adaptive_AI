import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Link } from "wouter";
import { 
  Hexagon, Plus, Upload, Search, Zap, Atom, Box, Layers, 
  Calculator, Factory, Sparkles, ArrowRight, CheckCircle2,
  Loader2, Play, Database
} from "lucide-react";
import type { MaterialEntity } from "@shared/schema";

interface QuickPrediction {
  material_id: string;
  properties: Array<{
    property_name: string;
    value: number;
    unit: string;
    confidence: number;
  }>;
}

const DEMO_MATERIALS: Array<{name: string; type: "polymer" | "crystal" | "composite"; smiles?: string; formula?: string}> = [
  { name: "Polyethylene (PE)", type: "polymer", smiles: "CC" },
  { name: "Polystyrene (PS)", type: "polymer", smiles: "c1ccccc1CC" },
  { name: "PMMA", type: "polymer", smiles: "CC(C)(C(=O)OC)C" },
  { name: "Iron Oxide", type: "crystal", formula: "Fe2O3" },
  { name: "Silicon Dioxide", type: "crystal", formula: "SiO2" },
  { name: "Titanium Dioxide", type: "crystal", formula: "TiO2" },
  { name: "Carbon Fiber Composite", type: "composite", formula: "C" },
  { name: "Glass Fiber Composite", type: "composite", formula: "SiO2" },
];

export default function MaterialsLibraryPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("all");
  const [quickPredictions, setQuickPredictions] = useState<QuickPrediction[]>([]);

  const categoryParam = activeTab !== "all" ? `?category=${activeTab}` : "";
  const { data: materialsResponse, isLoading } = useQuery<{ materials: MaterialEntity[], total: number }>({
    queryKey: ["/api/materials", activeTab],
    queryFn: async () => {
      const res = await fetch(`/api/materials${categoryParam}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch materials");
      return res.json();
    },
  });
  const materials = materialsResponse?.materials || [];
  const totalMaterials = materialsResponse?.total || 0;

  const { data: typeCounts } = useQuery<{ total: number; polymers: number; crystals: number; composites: number; thinFilms: number; electrochemical: number }>({
    queryKey: ["/api/materials/type-counts"],
  });

  const { data: computeNodes } = useQuery<any[]>({
    queryKey: ["/api/compute-nodes"],
  });

  const onlineNodes = computeNodes?.filter((n) => n.status === "active") || [];

  const quickPredictMutation = useMutation({
    mutationFn: async (material: typeof DEMO_MATERIALS[0]) => {
      const matData = material.type === "polymer" 
        ? { type: material.type, smiles: material.smiles }
        : { type: material.type, formula: material.formula };
      const res = await apiRequest("POST", "/api/compute/materials/predict", { 
        materials: [matData],
        properties: ["thermal_conductivity", "tensile_strength", "density"]
      });
      return res.json();
    },
  });

  const handleQuickPredict = async (material: typeof DEMO_MATERIALS[0]) => {
    const result = await quickPredictMutation.mutateAsync(material);
    if (result.results?.[0]) {
      setQuickPredictions(prev => [
        ...prev.filter(p => p.material_id !== result.results[0].material_id),
        result.results[0]
      ]);
    }
  };

  // Group related material types for filtering
  const polymerTypes = ["polymer", "homopolymer", "copolymer"];
  const crystalTypes = ["crystal", "perovskite", "double_perovskite", "spinel", "binary_oxide", "binary_chalcogenide", "binary_pnictide", "mxene_2d", "tmd_2d", "2d_material"];
  const compositeTypes = ["composite", "high_entropy_alloy", "binary_alloy", "ternary_alloy"];
  const thinFilmTypes = ["thin_film", "doped_semiconductor"];
  const electrochemicalTypes = ["battery_cathode", "battery_anode", "solid_electrolyte", "catalyst", "coating", "membrane"];

  const isPolymer = (type: string) => polymerTypes.includes(type);
  const isCrystal = (type: string) => crystalTypes.includes(type);
  const isComposite = (type: string) => compositeTypes.includes(type);
  const isThinFilm = (type: string) => thinFilmTypes.includes(type);
  const isElectrochemical = (type: string) => electrochemicalTypes.includes(type);

  const categories = [
    { label: "Total Materials", value: typeCounts?.total || totalMaterials, icon: Hexagon, color: "from-emerald-500 to-teal-500", bgColor: "bg-emerald-500", filter: "all" },
    { label: "Composites/Alloys", value: typeCounts?.composites || 0, icon: Box, color: "from-amber-500 to-orange-500", bgColor: "bg-amber-500", filter: "composite" },
    { label: "Thin Films", value: typeCounts?.thinFilms || 0, icon: Layers, color: "from-pink-500 to-rose-500", bgColor: "bg-pink-500", filter: "thinfilm" },
    { label: "Crystals", value: typeCounts?.crystals || 0, icon: Atom, color: "from-violet-500 to-purple-500", bgColor: "bg-violet-500", filter: "crystal" },
    { label: "Polymers", value: typeCounts?.polymers || 0, icon: Hexagon, color: "from-blue-500 to-cyan-500", bgColor: "bg-blue-500", filter: "polymer" },
    { label: "Electrochemical", value: typeCounts?.electrochemical || 0, icon: Zap, color: "from-yellow-500 to-amber-500", bgColor: "bg-yellow-500", filter: "electrochemical" },
  ];

  // Backend now handles category filtering; only apply search filter client-side
  const filteredMaterials = materials.filter(m => {
    if (searchQuery && !m.name?.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-emerald-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-600 via-teal-500 to-cyan-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxNWwtMTMuMDkgNy41djE1TDMwIDQ1bDEzLjA5LTcuNXYtMTVMMzAgMTV6IiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4xNSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Hexagon className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">Materials Library</h1>
                    <p className="text-emerald-100">Browse and manage your materials collection</p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-3 text-sm text-emerald-100">
                  <Badge variant="outline" className="bg-white/20 text-white border-white/30 gap-1">
                    <Database className="h-3 w-3" />
                    {totalMaterials.toLocaleString()} Total Materials
                  </Badge>
                  <Badge variant="outline" className="bg-white/20 text-white border-white/30 gap-1">
                    <Zap className="h-3 w-3" />
                    Polymers, Crystals, Composites
                  </Badge>
                  {onlineNodes.length > 0 && (
                    <Badge variant="outline" className="bg-emerald-500/20 text-white border-emerald-400/50 gap-1">
                      <CheckCircle2 className="h-3 w-3" />
                      {onlineNodes.length} Compute Node{onlineNodes.length !== 1 ? "s" : ""}
                    </Badge>
                  )}
                </div>
              </div>
              <Link href="/import/materials/materials">
                <Button className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0">
                  <Upload className="h-4 w-4" />
                  Import Materials
                </Button>
              </Link>
            </div>
          </header>

          <div className="relative max-w-xl">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input 
              placeholder="Search materials by name, formula, or properties..." 
              className="pl-12 h-12 text-base"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              data-testid="input-search-materials"
            />
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            {categories.map((cat) => (
              <Card 
                key={cat.label} 
                className={`border-0 shadow-lg overflow-hidden group cursor-pointer hover:shadow-xl transition-all ${activeTab === cat.filter ? "ring-2 ring-primary" : ""}`}
                onClick={() => setActiveTab(cat.filter)}
                data-testid={`card-category-${cat.filter}`}
              >
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${cat.color}`} />
                  <div className="p-6">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl ${cat.bgColor}/10 flex items-center justify-center group-hover:scale-110 transition-transform`}>
                        <cat.icon className={`h-6 w-6 ${cat.bgColor.replace('bg-', 'text-')}`} />
                      </div>
                      <div>
                        <p className="text-3xl font-bold">{cat.value}</p>
                        <p className="text-sm text-muted-foreground">{cat.label}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {materials.length > 0 ? (
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Hexagon className="h-5 w-5 text-emerald-500" />
                  Your Materials ({totalMaterials.toLocaleString()})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {filteredMaterials.map((material) => {
                    const typeCategory = isPolymer(material.type) ? "polymer" :
                      isCrystal(material.type) ? "crystal" :
                      isComposite(material.type) ? "composite" :
                      isThinFilm(material.type) ? "thinfilm" :
                      isElectrochemical(material.type) ? "electrochemical" : "other";
                    
                    const iconConfig = {
                      polymer: { bg: "bg-blue-500/10", icon: Hexagon, color: "text-blue-500" },
                      crystal: { bg: "bg-violet-500/10", icon: Atom, color: "text-violet-500" },
                      composite: { bg: "bg-amber-500/10", icon: Box, color: "text-amber-500" },
                      thinfilm: { bg: "bg-pink-500/10", icon: Layers, color: "text-pink-500" },
                      electrochemical: { bg: "bg-yellow-500/10", icon: Zap, color: "text-yellow-500" },
                      other: { bg: "bg-gray-500/10", icon: Hexagon, color: "text-gray-500" },
                    }[typeCategory];
                    
                    const IconComponent = iconConfig.icon;
                    
                    return (
                      <div
                        key={material.id}
                        className="p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                        data-testid={`material-${material.id}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${iconConfig.bg}`}>
                              <IconComponent className={`h-5 w-5 ${iconConfig.color}`} />
                            </div>
                            <div>
                              <p className="font-medium">{material.name || material.id}</p>
                              <p className="text-sm text-muted-foreground capitalize">{material.type.replace(/_/g, ' ')}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {material.isCurated && (
                              <Badge variant="outline" className="text-xs">Curated</Badge>
                            )}
                            <Link href={`/property-prediction?smiles=${encodeURIComponent((material.representation as any)?.smiles || '')}&name=${encodeURIComponent(material.name || '')}&type=${material.type}`}>
                              <Button size="sm" variant="outline" className="gap-1">
                                <Calculator className="h-3 w-3" />
                                Predict
                              </Button>
                            </Link>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="shadow-lg">
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 flex items-center justify-center mx-auto mb-5">
                    <Hexagon className="h-9 w-9 text-emerald-500" />
                  </div>
                  <p className="font-semibold text-lg mb-2">No materials yet</p>
                  <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                    Import polymers, crystals, composites, or coatings to build your materials library
                  </p>
                  <Link href="/import/materials/materials">
                    <Button className="gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600">
                      <Plus className="h-4 w-4" />
                      Import Materials
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          )}

          <Card className="shadow-lg border-2 border-dashed border-emerald-500/30">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-amber-500" />
                Quick Demo: Try Property Prediction
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Click any demo material below to instantly predict its properties using our ML pipeline:
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {DEMO_MATERIALS.map((mat, idx) => {
                  const prediction = quickPredictions.find(p => 
                    p.properties.some(pr => pr.property_name === "thermal_conductivity")
                  );
                  const typeColor = mat.type === "polymer" ? "border-blue-500/30 hover:border-blue-500" :
                                    mat.type === "crystal" ? "border-violet-500/30 hover:border-violet-500" :
                                    "border-amber-500/30 hover:border-amber-500";
                  
                  return (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border-2 ${typeColor} bg-background hover:shadow-md transition-all cursor-pointer`}
                      onClick={() => handleQuickPredict(mat)}
                      data-testid={`demo-material-${idx}`}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        {mat.type === "polymer" ? <Layers className="h-4 w-4 text-blue-500" /> :
                         mat.type === "crystal" ? <Atom className="h-4 w-4 text-violet-500" /> :
                         <Box className="h-4 w-4 text-amber-500" />}
                        <span className="font-medium text-sm">{mat.name}</span>
                      </div>
                      <code className="text-xs text-muted-foreground block truncate">
                        {mat.smiles || mat.formula}
                      </code>
                      {quickPredictMutation.isPending ? (
                        <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Predicting...
                        </div>
                      ) : (
                        <div className="mt-2 flex items-center gap-1 text-xs text-emerald-600">
                          <Play className="h-3 w-3" />
                          Click to predict
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              
              {quickPredictions.length > 0 && (
                <div className="mt-6 p-4 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                  <p className="text-sm font-medium mb-3 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    Latest Prediction Result
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    {quickPredictions[quickPredictions.length - 1].properties.slice(0, 3).map((prop, idx) => (
                      <div key={idx} className="p-3 rounded bg-background border">
                        <p className="text-xs text-muted-foreground capitalize">
                          {prop.property_name.replace(/_/g, " ")}
                        </p>
                        <p className="text-lg font-bold font-mono">
                          {prop.value.toFixed(2)} <span className="text-xs font-normal text-muted-foreground">{prop.unit}</span>
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Link href="/property-prediction">
              <Card className="group hover:shadow-lg transition-all cursor-pointer border-2 border-transparent hover:border-emerald-500/50">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-4 group-hover:bg-emerald-500/20 transition-colors">
                    <Calculator className="h-6 w-6 text-emerald-500" />
                  </div>
                  <h3 className="font-semibold mb-1">Property Prediction</h3>
                  <p className="text-sm text-muted-foreground mb-3">Run ML predictions on materials</p>
                  <div className="flex items-center text-sm text-emerald-600 font-medium">
                    Open Pipeline <ArrowRight className="h-4 w-4 ml-1" />
                  </div>
                </CardContent>
              </Card>
            </Link>
            <Link href="/manufacturability-scoring">
              <Card className="group hover:shadow-lg transition-all cursor-pointer border-2 border-transparent hover:border-violet-500/50">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4 group-hover:bg-violet-500/20 transition-colors">
                    <Factory className="h-6 w-6 text-violet-500" />
                  </div>
                  <h3 className="font-semibold mb-1">Manufacturability</h3>
                  <p className="text-sm text-muted-foreground mb-3">Score synthesis feasibility</p>
                  <div className="flex items-center text-sm text-violet-600 font-medium">
                    Score Materials <ArrowRight className="h-4 w-4 ml-1" />
                  </div>
                </CardContent>
              </Card>
            </Link>
            <Link href="/materials-campaigns">
              <Card className="group hover:shadow-lg transition-all cursor-pointer border-2 border-transparent hover:border-blue-500/50">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4 group-hover:bg-blue-500/20 transition-colors">
                    <Zap className="h-6 w-6 text-blue-500" />
                  </div>
                  <h3 className="font-semibold mb-1">Discovery Campaigns</h3>
                  <p className="text-sm text-muted-foreground mb-3">Manage research campaigns</p>
                  <div className="flex items-center text-sm text-blue-600 font-medium">
                    View Campaigns <ArrowRight className="h-4 w-4 ml-1" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
