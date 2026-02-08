import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Layers, 
  Plus, 
  Search, 
  GitBranch, 
  Sparkles, 
  Zap,
  RefreshCw,
  Database,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Columns,
  Eye,
  ChevronDown
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { Link } from "wouter";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface MaterialVariant {
  id: string;
  materialId: string;
  variantName?: string;
  variantType?: string;
  variantParams?: Record<string, any>;
  parameters?: Record<string, any>;
  predictedProperties: Record<string, any> | null;
  status?: string;
  generatedBy?: string;
  simulationState?: string | null;
  manufacturabilityScore?: number | null;
  createdAt: string;
}

interface VariantsResponse {
  variants: MaterialVariant[];
  total: number;
}

const ALL_COLUMNS = [
  { key: "external_variant_id", label: "Variant ID", group: "Core" },
  { key: "variant_type", label: "Variant Type", group: "Core" },
  { key: "base_material", label: "Base Material", group: "Core" },
  { key: "description", label: "Description", group: "Core" },
  { key: "source", label: "Source", group: "Core" },
  { key: "substrate", label: "Substrate", group: "Deposition" },
  { key: "deposition_method", label: "Deposition Method", group: "Deposition" },
  { key: "film_material", label: "Film Material", group: "Deposition" },
  { key: "target_thickness_nm", label: "Thickness (nm)", group: "Deposition" },
  { key: "ald_cycles", label: "ALD Cycles", group: "Deposition" },
  { key: "power_w", label: "Power (W)", group: "Deposition" },
  { key: "pressure_mtorr", label: "Pressure (mTorr)", group: "Deposition" },
  { key: "substrate_temperature_c", label: "Substrate Temp (°C)", group: "Deposition" },
  { key: "dopant", label: "Dopant", group: "Chemistry" },
  { key: "doping_method", label: "Doping Method", group: "Chemistry" },
  { key: "concentration_wt_percent", label: "Concentration (wt%)", group: "Chemistry" },
  { key: "concentration_cm3", label: "Concentration (cm⁻³)", group: "Chemistry" },
  { key: "solvent", label: "Solvent", group: "Chemistry" },
  { key: "initiator", label: "Initiator", group: "Chemistry" },
  { key: "polymer", label: "Polymer", group: "Polymer" },
  { key: "target_mn", label: "Target Mn", group: "Polymer" },
  { key: "target_pdi", label: "Target PDI", group: "Polymer" },
  { key: "polymerization_temperature_c", label: "Polymerization Temp (°C)", group: "Polymer" },
  { key: "polymerization_time_hours", label: "Polymerization Time (h)", group: "Polymer" },
  { key: "binder", label: "Binder", group: "Composite" },
  { key: "binder_wt_percent", label: "Binder (wt%)", group: "Composite" },
  { key: "matrix", label: "Matrix", group: "Composite" },
  { key: "reinforcement", label: "Reinforcement", group: "Composite" },
  { key: "fiber_volume_percent", label: "Fiber Volume (%)", group: "Composite" },
  { key: "fiber_architecture", label: "Fiber Architecture", group: "Composite" },
  { key: "layup_sequence", label: "Layup Sequence", group: "Composite" },
  { key: "active_material", label: "Active Material", group: "Battery" },
  { key: "active_material_wt_percent", label: "Active Material (wt%)", group: "Battery" },
  { key: "conductive_additive", label: "Conductive Additive", group: "Battery" },
  { key: "conductive_additive_wt_percent", label: "Conductive Additive (wt%)", group: "Battery" },
  { key: "synthesis_method", label: "Synthesis Method", group: "Processing" },
  { key: "processing_method", label: "Processing Method", group: "Processing" },
  { key: "treatment_type", label: "Treatment Type", group: "Processing" },
  { key: "treatment_temperature_c", label: "Treatment Temp (°C)", group: "Processing" },
  { key: "atmosphere", label: "Atmosphere", group: "Processing" },
  { key: "annealing_temperature_c", label: "Annealing Temp (°C)", group: "Thermal" },
  { key: "annealing_time_hours", label: "Annealing Time (h)", group: "Thermal" },
  { key: "annealing_time_min", label: "Annealing Time (min)", group: "Thermal" },
  { key: "sintering_temperature_c", label: "Sintering Temp (°C)", group: "Thermal" },
  { key: "sintering_time_hours", label: "Sintering Time (h)", group: "Thermal" },
  { key: "heating_rate_c_min", label: "Heating Rate (°C/min)", group: "Thermal" },
  { key: "cooling_rate_c_min", label: "Cooling Rate (°C/min)", group: "Thermal" },
  { key: "cooling_method", label: "Cooling Method", group: "Thermal" },
  { key: "quench_medium", label: "Quench Medium", group: "Thermal" },
  { key: "curing_temperature_c", label: "Curing Temp (°C)", group: "Curing" },
  { key: "curing_time_hours", label: "Curing Time (h)", group: "Curing" },
  { key: "curing_pressure_mpa", label: "Curing Pressure (MPa)", group: "Curing" },
  { key: "drying_temperature_c", label: "Drying Temp (°C)", group: "Curing" },
  { key: "drying_time_hours", label: "Drying Time (h)", group: "Curing" },
  { key: "print_speed_mm_s", label: "Print Speed (mm/s)", group: "Additive" },
  { key: "nozzle_temperature_c", label: "Nozzle Temp (°C)", group: "Additive" },
  { key: "layer_thickness_um", label: "Layer Thickness (μm)", group: "Additive" },
  { key: "infill_density_percent", label: "Infill Density (%)", group: "Additive" },
  { key: "build_orientation", label: "Build Orientation", group: "Additive" },
  { key: "laser_power_w", label: "Laser Power (W)", group: "Additive" },
  { key: "scan_speed_mm_s", label: "Scan Speed (mm/s)", group: "Additive" },
  { key: "coating_thickness_um", label: "Coating Thickness (μm)", group: "Surface" },
  { key: "implantation_energy_kev", label: "Implantation Energy (keV)", group: "Surface" },
  { key: "reduction_percent", label: "Reduction (%)", group: "Metallurgy" },
  { key: "grain_size_target_um", label: "Grain Size Target (μm)", group: "Metallurgy" },
  { key: "calendering_pressure_mpa", label: "Calendering Pressure (MPa)", group: "Metallurgy" },
  { key: "holding_time_hours", label: "Holding Time (h)", group: "Other" },
  { key: "support_type", label: "Support Type", group: "Other" },
  { key: "temperature_regime", label: "Temperature Regime", group: "Other" },
];

const DEFAULT_VISIBLE_COLUMNS = [
  "external_variant_id", "variant_type", "base_material", "description",
  "deposition_method", "substrate", "target_thickness_nm", "substrate_temperature_c"
];

export default function MaterialVariantsPage() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 50;
  const [visibleColumns, setVisibleColumns] = useState<string[]>(DEFAULT_VISIBLE_COLUMNS);
  const [selectedVariant, setSelectedVariant] = useState<MaterialVariant | null>(null);

  const { data, isLoading, error } = useQuery<VariantsResponse>({
    queryKey: ["/api/material-variants", page, pageSize],
    queryFn: async () => {
      const response = await fetch(`/api/material-variants?limit=${pageSize}&offset=${page * pageSize}`, {
        credentials: 'include'
      });
      if (!response.ok) throw new Error('Failed to fetch variants');
      return response.json();
    }
  });

  const columnGroups = useMemo(() => {
    const groups: Record<string, typeof ALL_COLUMNS> = {};
    ALL_COLUMNS.forEach(col => {
      if (!groups[col.group]) groups[col.group] = [];
      groups[col.group].push(col);
    });
    return groups;
  }, []);

  const toggleColumn = (key: string) => {
    setVisibleColumns(prev => 
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  const toggleGroup = (group: string) => {
    const groupKeys = columnGroups[group].map(c => c.key);
    const allVisible = groupKeys.every(k => visibleColumns.includes(k));
    if (allVisible) {
      setVisibleColumns(prev => prev.filter(k => !groupKeys.includes(k)));
    } else {
      setVisibleColumns(prev => {
        const combined = [...prev, ...groupKeys];
        return combined.filter((key, index) => combined.indexOf(key) === index);
      });
    }
  };

  const getVariantValue = (variant: MaterialVariant, key: string): string => {
    const params = variant.variantParams || variant.parameters || {};
    const value = params[key];
    if (value === null || value === undefined || value === '') return '-';
    return String(value);
  };

  const syncMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/external-sync/sync/variants", {
        source: "digitalocean",
        tableName: "variants_formulations_massive"
      });
      return response.json();
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["/api/material-variants"] });
      toast({
        title: "Sync Complete",
        description: `Synced ${result.recordsInserted} new variants from DigitalOcean database.`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Sync Failed",
        description: error.message || "Failed to sync variants from DigitalOcean",
        variant: "destructive",
      });
    },
  });

  const variants = data?.variants || [];
  const total = data?.total || 0;
  const totalPages = Math.ceil(total / pageSize);

  const filteredVariants = searchQuery 
    ? variants.filter(v => {
        const query = searchQuery.toLowerCase();
        const params = v.variantParams || v.parameters || {};
        const searchableFields = [
          params.description, params.variant_type, params.base_material,
          params.external_variant_id, params.substrate, params.deposition_method,
          v.variantName, v.variantType
        ].filter(Boolean);
        return searchableFields.some(field => String(field).toLowerCase().includes(query));
      })
    : variants;

  const evaluatedCount = variants.filter(v => v.predictedProperties && Object.keys(v.predictedProperties).length > 0).length;

  const stats = [
    { label: "Total Variants", value: total, icon: Layers, color: "from-violet-500 to-purple-500", bgColor: "bg-violet-500" },
    { label: "Active Explorations", value: variants.filter(v => v.status === "exploring" || v.status === "active").length, icon: GitBranch, color: "from-blue-500 to-cyan-500", bgColor: "bg-blue-500" },
    { label: "Evaluated", value: evaluatedCount, icon: Sparkles, color: "from-emerald-500 to-teal-500", bgColor: "bg-emerald-500" },
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
      case "exploring":
        return <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/30">Active</Badge>;
      case "evaluated":
        return <Badge className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30">Evaluated</Badge>;
      case "pending":
        return <Badge className="bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/30">Pending</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-violet-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0xMCAzMGgyME0zMCAxMHYyMCIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Layers className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">Material Variants</h1>
                    <p className="text-violet-100">Explore structural variations and property predictions</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-violet-100">
                  <Zap className="h-4 w-4" />
                  <span>AI-guided structural exploration and optimization</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Button 
                  className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0"
                  onClick={() => syncMutation.mutate()}
                  disabled={syncMutation.isPending}
                  data-testid="button-sync-variants"
                >
                  {syncMutation.isPending ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Database className="h-4 w-4" />
                  )}
                  {syncMutation.isPending ? "Syncing..." : "Sync from DigitalOcean"}
                </Button>
                <Link href="/property-prediction">
                  <Button 
                    className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0"
                    data-testid="button-send-to-pipeline"
                  >
                    <Sparkles className="h-4 w-4" />
                    Send to Property Pipeline
                  </Button>
                </Link>
                <Button 
                  className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0"
                  data-testid="button-create-variant"
                >
                  <Plus className="h-4 w-4" />
                  Create Variant
                </Button>
              </div>
            </div>
          </header>

          <div className="relative max-w-xl">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input 
              placeholder="Search variants by name or type..." 
              className="pl-12 h-12 text-base"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              data-testid="input-search-variants"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {stats.map((stat) => (
              <Card key={stat.label} className="border-0 shadow-lg overflow-hidden">
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${stat.color}`} />
                  <div className="p-6">
                    <div className="flex items-center gap-4">
                      <div className={`w-14 h-14 rounded-xl ${stat.bgColor}/10 flex items-center justify-center`}>
                        <stat.icon className={`h-7 w-7 ${stat.bgColor.replace('bg-', 'text-')}`} />
                      </div>
                      <div>
                        {isLoading ? (
                          <Skeleton className="h-9 w-20" />
                        ) : (
                          <p className="text-3xl font-bold">{stat.value.toLocaleString()}</p>
                        )}
                        <p className="text-sm text-muted-foreground">{stat.label}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {isLoading ? (
            <Card className="shadow-lg">
              <CardContent className="p-6">
                <div className="space-y-4">
                  {[1, 2, 3, 4, 5].map(i => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : error ? (
            <Card className="shadow-lg">
              <CardContent className="p-12 text-center">
                <div className="w-20 h-20 rounded-3xl bg-destructive/10 flex items-center justify-center mx-auto mb-5">
                  <Database className="h-9 w-9 text-destructive" />
                </div>
                <p className="font-semibold text-lg mb-2">Failed to load variants</p>
                <p className="text-muted-foreground mb-6">
                  {(error as Error).message || "An error occurred while fetching variants"}
                </p>
                <Button 
                  variant="outline" 
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/material-variants"] })}
                  data-testid="button-retry-load"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry
                </Button>
              </CardContent>
            </Card>
          ) : filteredVariants.length === 0 ? (
            <Card className="shadow-lg">
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-500/10 to-purple-500/10 flex items-center justify-center mx-auto mb-5">
                    <Layers className="h-9 w-9 text-violet-500" />
                  </div>
                  <p className="font-semibold text-lg mb-2">No variants yet</p>
                  <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                    Sync variants from your DigitalOcean database or create new variants from your materials
                  </p>
                  <div className="flex items-center justify-center gap-3">
                    <Button 
                      variant="outline"
                      className="gap-2"
                      onClick={() => syncMutation.mutate()}
                      disabled={syncMutation.isPending}
                      data-testid="button-sync-empty"
                    >
                      <Database className="h-4 w-4" />
                      Sync from DigitalOcean
                    </Button>
                    <Button className="gap-2 bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600">
                      <Plus className="h-4 w-4" />
                      Create Variant
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="shadow-lg">
              <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4 flex-wrap">
                <div className="flex items-center gap-4">
                  <CardTitle className="text-lg">Material Variants</CardTitle>
                  <Badge variant="outline" className="text-xs">
                    {visibleColumns.length} of {ALL_COLUMNS.length} columns
                  </Badge>
                </div>
                <div className="flex items-center gap-3">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm" className="gap-2" data-testid="button-column-picker">
                        <Columns className="h-4 w-4" />
                        Columns
                        <ChevronDown className="h-3 w-3" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-72 max-h-[400px] overflow-y-auto">
                      <DropdownMenuLabel className="flex items-center justify-between">
                        <span>Column Visibility</span>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-6 text-xs"
                          onClick={() => setVisibleColumns(DEFAULT_VISIBLE_COLUMNS)}
                        >
                          Reset
                        </Button>
                      </DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      {Object.entries(columnGroups).map(([group, cols]) => (
                        <div key={group}>
                          <DropdownMenuCheckboxItem
                            checked={cols.every(c => visibleColumns.includes(c.key))}
                            onCheckedChange={() => toggleGroup(group)}
                            className="font-semibold"
                          >
                            {group} ({cols.length})
                          </DropdownMenuCheckboxItem>
                          {cols.map(col => (
                            <DropdownMenuCheckboxItem
                              key={col.key}
                              checked={visibleColumns.includes(col.key)}
                              onCheckedChange={() => toggleColumn(col.key)}
                              className="pl-6"
                            >
                              {col.label}
                            </DropdownMenuCheckboxItem>
                          ))}
                          <DropdownMenuSeparator />
                        </div>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <span className="text-sm text-muted-foreground">
                    Showing {page * pageSize + 1}-{Math.min((page + 1) * pageSize, total)} of {total.toLocaleString()}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <ScrollArea className="h-[500px]">
                    <Table style={{ minWidth: `${visibleColumns.length * 150 + 100}px` }}>
                      <TableHeader>
                        <TableRow>
                          {ALL_COLUMNS.filter(c => visibleColumns.includes(c.key)).map(col => (
                            <TableHead key={col.key} className="min-w-[130px] whitespace-nowrap">
                              {col.label}
                            </TableHead>
                          ))}
                          <TableHead className="w-[80px] sticky right-0 bg-background">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredVariants.map((variant) => (
                          <TableRow key={variant.id} data-testid={`row-variant-${variant.id}`}>
                            {ALL_COLUMNS.filter(c => visibleColumns.includes(c.key)).map(col => (
                              <TableCell key={col.key} className="whitespace-nowrap max-w-[200px] truncate">
                                {col.key === "variant_type" ? (
                                  <Badge variant="secondary" className="text-xs">
                                    {getVariantValue(variant, col.key)}
                                  </Badge>
                                ) : (
                                  getVariantValue(variant, col.key)
                                )}
                              </TableCell>
                            ))}
                            <TableCell className="sticky right-0 bg-background">
                              <Button 
                                variant="ghost" 
                                size="icon"
                                onClick={() => setSelectedVariant(variant)}
                                data-testid={`button-view-variant-${variant.id}`}
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ScrollArea>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between px-6 py-4 border-t">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.max(0, p - 1))}
                      disabled={page === 0}
                      data-testid="button-prev-page"
                    >
                      <ChevronLeft className="h-4 w-4 mr-1" />
                      Previous
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      Page {page + 1} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1}
                      data-testid="button-next-page"
                    >
                      Next
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Variant Detail Dialog */}
          <Dialog open={!!selectedVariant} onOpenChange={(open) => !open && setSelectedVariant(null)}>
            <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-3">
                  <Layers className="h-5 w-5 text-violet-500" />
                  Variant Details
                </DialogTitle>
              </DialogHeader>
              {selectedVariant && (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="p-3 rounded-md bg-muted/50">
                      <p className="text-xs text-muted-foreground mb-1">Variant ID</p>
                      <p className="font-mono text-sm">{getVariantValue(selectedVariant, "external_variant_id")}</p>
                    </div>
                    <div className="p-3 rounded-md bg-muted/50">
                      <p className="text-xs text-muted-foreground mb-1">Type</p>
                      <Badge variant="secondary">{getVariantValue(selectedVariant, "variant_type")}</Badge>
                    </div>
                    <div className="p-3 rounded-md bg-muted/50">
                      <p className="text-xs text-muted-foreground mb-1">Base Material</p>
                      <p className="font-medium">{getVariantValue(selectedVariant, "base_material")}</p>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3 text-sm">All Parameters ({ALL_COLUMNS.length} columns)</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                      {ALL_COLUMNS.map(col => {
                        const value = getVariantValue(selectedVariant, col.key);
                        if (value === '-') return null;
                        return (
                          <div key={col.key} className="p-2 rounded-md border bg-card">
                            <p className="text-xs text-muted-foreground truncate">{col.label}</p>
                            <p className="font-medium text-sm truncate" title={value}>{value}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3 text-sm">Raw Data</h4>
                    <pre className="p-4 rounded-md bg-muted text-xs overflow-auto max-h-[200px]">
                      {JSON.stringify(selectedVariant.variantParams || selectedVariant.parameters, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>
        </div>
      </div>
    </div>
  );
}
