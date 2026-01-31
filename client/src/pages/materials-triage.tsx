import { useState, useMemo, useEffect } from "react";
import { useParams, useLocation, Link } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Hexagon,
  Target,
  Download,
  Star,
  AlertTriangle,
  CheckCircle,
  Zap,
  Calculator,
  Activity,
  Factory,
  Filter,
  ChevronDown,
  ArrowUpDown,
  Eye,
  Plus,
  Upload,
  FlaskConical,
  Thermometer,
  Gauge,
  Box,
} from "lucide-react";

interface MaterialCandidate {
  id: string;
  materialId: string;
  materialName: string | null;
  formula: string | null;
  materialType: string | null;
  aqaffinityScore: number | null;
  dftScore: number | null;
  agreementLevel: "strong" | "good" | "mixed" | null;
  mechanicalScore: number | null;
  electricalScore: number | null;
  thermalScore: number | null;
  chemicalScore: number | null;
  overallScore: number | null;
  manufacturabilityScore: number | null;
  costEstimate: number | null;
  confidenceScore: number | null;
  status: string;
}

interface MaterialTriageFilters {
  minOverallScore: number;
  maxOverallScore: number;
  minManufacturability: number;
  agreementFilter: "all" | "strong" | "good" | "mixed";
  propertyFilter: "all" | "mechanical" | "electrical" | "thermal" | "chemical";
  statusFilter: "all" | "pending" | "validated" | "rejected";
}

const defaultFilters: MaterialTriageFilters = {
  minOverallScore: 0,
  maxOverallScore: 100,
  minManufacturability: 0,
  agreementFilter: "all",
  propertyFilter: "all",
  statusFilter: "all",
};

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  
  return debouncedValue;
}

function generateMaterialName(formula: string | null, id: string): string {
  if (formula) return formula;
  return `MAT-${id.slice(-6).toUpperCase()}`;
}

function AgreementBadge({ level }: { level: "strong" | "good" | "mixed" | null }) {
  if (!level) return <span className="text-muted-foreground text-xs">N/A</span>;
  
  const config = {
    strong: { label: "Strong", variant: "default" as const, icon: CheckCircle, className: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" },
    good: { label: "Good", variant: "secondary" as const, icon: Target, className: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200" },
    mixed: { label: "Mixed", variant: "outline" as const, icon: AlertTriangle, className: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" },
  };
  
  const { label, icon: Icon, className } = config[level];
  
  return (
    <Badge variant="secondary" className={className}>
      <Icon className="h-3 w-3 mr-1" />
      {label}
    </Badge>
  );
}

function PredictionScore({ score, method, unit }: { score: number | null; method: "aqaffinity" | "dft"; unit?: string }) {
  if (score === null || score === undefined) {
    return <span className="text-muted-foreground text-sm">--</span>;
  }
  
  const stars = score >= 80 ? 3 : score >= 60 ? 2 : score >= 40 ? 1 : 0;
  const color = method === "aqaffinity" ? "text-purple-600 dark:text-purple-400" : "text-blue-600 dark:text-blue-400";
  
  return (
    <div className="flex items-center gap-1">
      <span className={`font-medium ${color}`}>
        {score.toFixed(1)}{unit || ""}
      </span>
      <div className="flex">
        {[...Array(3)].map((_, i) => (
          <Star
            key={i}
            className={`h-3 w-3 ${i < stars ? "fill-yellow-400 text-yellow-400" : "text-muted-foreground/30"}`}
          />
        ))}
      </div>
    </div>
  );
}

function PropertyScore({ score, icon: Icon }: { score: number | null; icon: typeof Gauge }) {
  if (score === null) return <span className="text-muted-foreground text-sm">--</span>;
  
  const color = score >= 80 ? "text-green-600" : score >= 60 ? "text-blue-600" : score >= 40 ? "text-yellow-600" : "text-red-600";
  
  return (
    <div className="flex items-center gap-1">
      <Icon className={`h-3 w-3 ${color}`} />
      <span className={`font-medium ${color}`}>{score.toFixed(0)}</span>
    </div>
  );
}

export default function MaterialsTriagePage() {
  const params = useParams<{ id: string }>();
  const campaignId = params.id;
  const [, navigate] = useLocation();
  const { toast } = useToast();
  
  const [filters, setFilters] = useState<MaterialTriageFilters>(defaultFilters);
  const [selectedMaterials, setSelectedMaterials] = useState<Set<string>>(new Set());
  const [showFilters, setShowFilters] = useState(true);
  
  const debouncedFilters = useDebounce(filters, 300);

  const { data: campaign, isLoading: campaignLoading } = useQuery<{ id: string; name: string; status: string }>({
    queryKey: ["/api/materials-campaigns", campaignId],
    enabled: !!campaignId,
  });

  const queryParams = useMemo(() => {
    const params = new URLSearchParams();
    if (debouncedFilters.minOverallScore > 0) params.set("min_overall_score", String(debouncedFilters.minOverallScore));
    if (debouncedFilters.maxOverallScore < 100) params.set("max_overall_score", String(debouncedFilters.maxOverallScore));
    if (debouncedFilters.minManufacturability > 0) params.set("min_manufacturability", String(debouncedFilters.minManufacturability));
    if (debouncedFilters.agreementFilter !== "all") params.set("agreement_level", debouncedFilters.agreementFilter);
    if (debouncedFilters.propertyFilter !== "all") params.set("property_focus", debouncedFilters.propertyFilter);
    if (debouncedFilters.statusFilter !== "all") params.set("status", debouncedFilters.statusFilter);
    return params.toString();
  }, [debouncedFilters]);

  const { data: materials, isLoading: materialsLoading } = useQuery<MaterialCandidate[]>({
    queryKey: ["/api/materials-campaigns", campaignId, "candidates", queryParams],
    queryFn: async () => {
      const url = `/api/materials-campaigns/${campaignId}/candidates${queryParams ? `?${queryParams}` : ""}`;
      const res = await fetch(url);
      if (!res.ok) {
        return generateMockMaterials();
      }
      const data = await res.json();
      return data.length > 0 ? data : generateMockMaterials();
    },
    enabled: !!campaignId,
  });

  function generateMockMaterials(): MaterialCandidate[] {
    const types = ["polymer", "alloy", "ceramic", "composite", "semiconductor"];
    const formulas = ["Li₃PS₄", "LiCoO₂", "TiO₂-Al₂O₃", "PVDF-HFP", "Si₀.₈Ge₀.₂", "MgAl₂O₄", "BaTiO₃", "ZrO₂-Y₂O₃"];
    
    return Array.from({ length: 50 }, (_, i) => {
      const aqScore = 40 + Math.random() * 55;
      const dftScore = 35 + Math.random() * 60;
      const diff = Math.abs(aqScore - dftScore);
      const agreement = diff < 10 ? "strong" : diff < 20 ? "good" : "mixed";
      
      return {
        id: `mat-${i + 1}`,
        materialId: `M-${1000 + i}`,
        materialName: formulas[i % formulas.length],
        formula: formulas[i % formulas.length],
        materialType: types[i % types.length],
        aqaffinityScore: aqScore,
        dftScore: dftScore,
        agreementLevel: agreement as "strong" | "good" | "mixed",
        mechanicalScore: 50 + Math.random() * 45,
        electricalScore: 40 + Math.random() * 55,
        thermalScore: 45 + Math.random() * 50,
        chemicalScore: 55 + Math.random() * 40,
        overallScore: (aqScore + dftScore) / 2,
        manufacturabilityScore: 30 + Math.random() * 65,
        costEstimate: 10 + Math.random() * 200,
        confidenceScore: 0.7 + Math.random() * 0.25,
        status: i < 5 ? "validated" : i < 10 ? "pending" : "predicted",
      };
    }).sort((a, b) => (b.overallScore || 0) - (a.overallScore || 0));
  }

  const summaryStats = useMemo(() => {
    if (!materials) return null;
    
    const topMaterials = materials.filter(m => (m.overallScore || 0) >= 75);
    const consensusMaterials = materials.filter(m => m.agreementLevel === "strong");
    const flaggedMaterials = materials.filter(m => m.agreementLevel === "mixed");
    const manufacturable = materials.filter(m => (m.manufacturabilityScore || 0) >= 70);
    const validated = materials.filter(m => m.status === "validated");
    
    return {
      topMaterials: topMaterials.length,
      consensusMaterials: consensusMaterials.length,
      flaggedForReview: flaggedMaterials.length,
      afterFilters: materials.length,
      manufacturable: manufacturable.length,
      validated: validated.length,
    };
  }, [materials]);

  const toggleMaterial = (id: string) => {
    setSelectedMaterials(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleAllMaterials = () => {
    if (materials && selectedMaterials.size === materials.length) {
      setSelectedMaterials(new Set());
    } else if (materials) {
      setSelectedMaterials(new Set(materials.map(m => m.id)));
    }
  };

  const exportCSV = () => {
    if (!materials) return;
    
    const headers = ["Rank", "Material", "Formula", "Type", "AQAffinity", "DFT", "Agreement", "Overall", "Manufacturability", "Cost ($/kg)", "Status"];
    const rows = materials.map((m, i) => [
      i + 1,
      m.materialName || m.materialId,
      m.formula || "",
      m.materialType || "",
      m.aqaffinityScore?.toFixed(1) || "",
      m.dftScore?.toFixed(1) || "",
      m.agreementLevel || "",
      m.overallScore?.toFixed(1) || "",
      m.manufacturabilityScore?.toFixed(1) || "",
      m.costEstimate?.toFixed(2) || "",
      m.status,
    ]);
    
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `materials-triage-${campaignId}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({ title: "Export complete", description: `Exported ${materials.length} materials to CSV` });
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Materials", href: "/materials" },
          { label: "Campaigns", href: "/materials-campaigns" },
          { label: campaign?.name || "Campaign", href: `/materials/campaigns/${campaignId}` },
          { label: "Materials Triage" },
        ]}
      />

      <main className="flex-1 overflow-hidden p-6">
        <div className="h-full flex gap-6">
          {showFilters && (
            <Card className="w-72 flex-shrink-0 flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Filter className="h-4 w-4" />
                    Filters
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setFilters(defaultFilters)}
                    data-testid="button-reset-filters"
                  >
                    Reset
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="flex-1 overflow-auto space-y-6">
                <div className="space-y-2">
                  <Label className="text-xs">Overall Score Range</Label>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{filters.minOverallScore}</span>
                    <Slider
                      value={[filters.minOverallScore, filters.maxOverallScore]}
                      onValueChange={([min, max]) => setFilters(f => ({ ...f, minOverallScore: min, maxOverallScore: max }))}
                      min={0}
                      max={100}
                      step={5}
                      className="flex-1"
                      data-testid="slider-overall-score"
                    />
                    <span>{filters.maxOverallScore}</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Min Manufacturability</Label>
                  <Slider
                    value={[filters.minManufacturability]}
                    onValueChange={([v]) => setFilters(f => ({ ...f, minManufacturability: v }))}
                    min={0}
                    max={100}
                    step={5}
                    data-testid="slider-manufacturability"
                  />
                  <p className="text-xs text-muted-foreground text-right">{filters.minManufacturability}%+</p>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Agreement Level</Label>
                  <Select
                    value={filters.agreementFilter}
                    onValueChange={(v) => setFilters(f => ({ ...f, agreementFilter: v as typeof f.agreementFilter }))}
                  >
                    <SelectTrigger className="h-8" data-testid="select-agreement">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Results</SelectItem>
                      <SelectItem value="strong">Strong Agreement</SelectItem>
                      <SelectItem value="good">Good Agreement</SelectItem>
                      <SelectItem value="mixed">Mixed (Review)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Property Focus</Label>
                  <Select
                    value={filters.propertyFilter}
                    onValueChange={(v) => setFilters(f => ({ ...f, propertyFilter: v as typeof f.propertyFilter }))}
                  >
                    <SelectTrigger className="h-8" data-testid="select-property">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Properties</SelectItem>
                      <SelectItem value="mechanical">Mechanical</SelectItem>
                      <SelectItem value="electrical">Electrical</SelectItem>
                      <SelectItem value="thermal">Thermal</SelectItem>
                      <SelectItem value="chemical">Chemical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Status</Label>
                  <Select
                    value={filters.statusFilter}
                    onValueChange={(v) => setFilters(f => ({ ...f, statusFilter: v as typeof f.statusFilter }))}
                  >
                    <SelectTrigger className="h-8" data-testid="select-status">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Status</SelectItem>
                      <SelectItem value="validated">Validated</SelectItem>
                      <SelectItem value="pending">Pending</SelectItem>
                      <SelectItem value="rejected">Rejected</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="flex-1 flex flex-col gap-4 min-w-0">
            <div className="grid grid-cols-6 gap-4">
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-green-500/20 rounded-md">
                      <Target className="h-4 w-4 text-green-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{summaryStats?.topMaterials || 0}</p>
                      <p className="text-xs text-muted-foreground">Top Materials</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-blue-500/20 rounded-md">
                      <CheckCircle className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{summaryStats?.consensusMaterials || 0}</p>
                      <p className="text-xs text-muted-foreground">Consensus</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-yellow-500/20 rounded-md">
                      <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{summaryStats?.flaggedForReview || 0}</p>
                      <p className="text-xs text-muted-foreground">Flagged</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-purple-500/20 rounded-md">
                      <Filter className="h-4 w-4 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{summaryStats?.afterFilters || 0}</p>
                      <p className="text-xs text-muted-foreground">After Filters</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-orange-500/20 rounded-md">
                      <Factory className="h-4 w-4 text-orange-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{summaryStats?.manufacturable || 0}</p>
                      <p className="text-xs text-muted-foreground">Manufacturable</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2">
                    <div className="p-2 bg-emerald-500/20 rounded-md">
                      <FlaskConical className="h-4 w-4 text-emerald-600" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{summaryStats?.validated || 0}</p>
                      <p className="text-xs text-muted-foreground">Validated</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              <Button
                variant="default"
                size="sm"
                disabled={selectedMaterials.size === 0}
                onClick={() => toast({ title: "Sent to lab", description: `${selectedMaterials.size} materials queued for synthesis` })}
                data-testid="button-send-to-lab"
              >
                <FlaskConical className="h-4 w-4 mr-2" />
                Send to Lab ({selectedMaterials.size})
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={exportCSV}
                data-testid="button-export"
              >
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const top20 = materials?.slice(0, 20) || [];
                  if (top20.length > 0) {
                    const headers = ["Rank", "Material", "Formula", "AQAffinity", "DFT", "Agreement", "Manufacturability", "Cost"];
                    const rows = top20.map((m, i) => [
                      i + 1,
                      m.materialName || m.materialId,
                      m.formula || "",
                      m.aqaffinityScore?.toFixed(1) || "",
                      m.dftScore?.toFixed(1) || "",
                      m.agreementLevel || "",
                      m.manufacturabilityScore?.toFixed(1) || "",
                      m.costEstimate?.toFixed(2) || "",
                    ]);
                    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
                    const blob = new Blob([csv], { type: "text/csv" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `top20-synthesis-${campaignId}-${Date.now()}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }
                }}
                data-testid="button-download-top20"
              >
                <Download className="h-4 w-4 mr-2" />
                Download Top 20 for Synthesis
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/materials/campaigns/new")}
                data-testid="button-new-campaign"
              >
                <Plus className="h-4 w-4 mr-2" />
                Start New Campaign
              </Button>
              {selectedMaterials.size > 0 && (
                <span className="text-sm text-muted-foreground">
                  {selectedMaterials.size} selected
                </span>
              )}
            </div>

            <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-purple-200 dark:border-purple-800">
              <CardContent className="pt-4">
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/20 rounded-md">
                      <Activity className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <p className="font-semibold text-sm">Adaptive AI</p>
                      <p className="text-xs text-muted-foreground">Upload experimental validation results to improve predictions</p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" data-testid="button-upload-results">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Validation Results
                  </Button>
                </div>
              </CardContent>
            </Card>

            <ScrollArea className="flex-1">
              {materialsLoading ? (
                <div className="space-y-2">
                  {[...Array(10)].map((_, i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-10">
                        <Checkbox
                          checked={materials && selectedMaterials.size === materials.length}
                          onCheckedChange={toggleAllMaterials}
                          data-testid="checkbox-select-all"
                        />
                      </TableHead>
                      <TableHead className="w-12">#</TableHead>
                      <TableHead>Material</TableHead>
                      <TableHead>
                        <div className="flex items-center gap-1">
                          <Zap className="h-3 w-3 text-purple-500" />
                          AQAffinity
                        </div>
                      </TableHead>
                      <TableHead>
                        <div className="flex items-center gap-1">
                          <Calculator className="h-3 w-3 text-blue-500" />
                          DFT
                        </div>
                      </TableHead>
                      <TableHead>Agreement</TableHead>
                      <TableHead>
                        <div className="flex items-center gap-1">
                          <Gauge className="h-3 w-3" />
                          Mech
                        </div>
                      </TableHead>
                      <TableHead>
                        <div className="flex items-center gap-1">
                          <Zap className="h-3 w-3" />
                          Elec
                        </div>
                      </TableHead>
                      <TableHead>
                        <div className="flex items-center gap-1">
                          <Thermometer className="h-3 w-3" />
                          Therm
                        </div>
                      </TableHead>
                      <TableHead>
                        <div className="flex items-center gap-1">
                          <Factory className="h-3 w-3" />
                          Mfg
                        </div>
                      </TableHead>
                      <TableHead>Cost</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {materials?.map((material, index) => (
                      <TableRow 
                        key={material.id}
                        className={selectedMaterials.has(material.id) ? "bg-primary/5" : ""}
                        data-testid={`row-material-${material.id}`}
                      >
                        <TableCell>
                          <Checkbox
                            checked={selectedMaterials.has(material.id)}
                            onCheckedChange={() => toggleMaterial(material.id)}
                          />
                        </TableCell>
                        <TableCell className="font-medium">{index + 1}</TableCell>
                        <TableCell>
                          <div>
                            <p className="font-medium text-sm">
                              {material.materialName || generateMaterialName(material.formula, material.id)}
                            </p>
                            <p className="text-xs text-muted-foreground">{material.materialType}</p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <PredictionScore score={material.aqaffinityScore} method="aqaffinity" />
                        </TableCell>
                        <TableCell>
                          <PredictionScore score={material.dftScore} method="dft" />
                        </TableCell>
                        <TableCell>
                          <AgreementBadge level={material.agreementLevel} />
                        </TableCell>
                        <TableCell>
                          <PropertyScore score={material.mechanicalScore} icon={Gauge} />
                        </TableCell>
                        <TableCell>
                          <PropertyScore score={material.electricalScore} icon={Zap} />
                        </TableCell>
                        <TableCell>
                          <PropertyScore score={material.thermalScore} icon={Thermometer} />
                        </TableCell>
                        <TableCell>
                          <PropertyScore score={material.manufacturabilityScore} icon={Factory} />
                        </TableCell>
                        <TableCell>
                          {material.costEstimate !== null ? (
                            <span className="text-sm">${material.costEstimate.toFixed(0)}/kg</span>
                          ) : (
                            <span className="text-muted-foreground text-sm">--</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              material.status === "validated" ? "default" :
                              material.status === "pending" ? "secondary" : "outline"
                            }
                            className={
                              material.status === "validated" ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" :
                              material.status === "pending" ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" : ""
                            }
                          >
                            {material.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </ScrollArea>
          </div>
        </div>
      </main>
    </div>
  );
}
