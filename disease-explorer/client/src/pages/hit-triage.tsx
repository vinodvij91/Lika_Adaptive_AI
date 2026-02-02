import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams, useLocation } from "wouter";
import { useState, useMemo, useCallback } from "react";
import { ResultsPanel } from "@/components/results-panel";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowLeft,
  Download,
  Beaker,
  Filter,
  AlertTriangle,
  CheckCircle,
  XCircle,
  MinusCircle,
  HelpCircle,
  Target,
  FlaskConical,
  Activity,
  Gauge,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { generateMoleculeName } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { useDebounce } from "@/hooks/use-debounce";

interface HitCandidate {
  id: number;
  moleculeId: number;
  smiles: string;
  moleculeName: string | null;
  oracleScore: number | null;
  dockingScore: number | null;
  admetScore: number | null;
  synthesisScore: number | null;
  translationalScore: number | null;
  ipRiskFlag: boolean | null;
  dockingUncertainty: number | null;
  admetUncertainty: number | null;
  overallUncertainty: number | null;
  lastAssayOutcome: string | null;
  bestAssayValue: number | null;
  aqaffinityScore: number | null;
  autodockScore: number | null;
  agreementLevel: "strong" | "good" | "mixed" | null;
}

interface HitTriageFilters {
  minOracleScore: number;
  maxOracleScore: number;
  minDockingScore: number;
  minAdmetScore: number;
  minSynthesisScore: number;
  maxUncertainty: number;
  ipSafeOnly: boolean;
  assayFilter: "all" | "tested" | "untested";
  agreementFilter: "all" | "strong" | "good" | "mixed";
}

interface Campaign {
  id: number;
  name: string;
  status: string;
}

interface Assay {
  id: number;
  name: string;
  type: string;
  readoutType: string;
}

const defaultFilters: HitTriageFilters = {
  minOracleScore: 0,
  maxOracleScore: 100,
  minDockingScore: 0,
  minAdmetScore: 0,
  minSynthesisScore: 0,
  maxUncertainty: 100,
  ipSafeOnly: false,
  assayFilter: "all",
  agreementFilter: "all",
};

function ScoreBar({ score, max = 100, color = "primary" }: { score: number | null; max?: number; color?: string }) {
  if (score === null) return <span className="text-muted-foreground text-xs">N/A</span>;
  const percentage = Math.min(100, Math.max(0, (score / max) * 100));
  const colorClass = color === "success" ? "bg-green-500" : color === "warning" ? "bg-amber-500" : "bg-primary";
  
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
        <div 
          className={`h-full ${colorClass} rounded-full`} 
          style={{ width: `${percentage}%` }} 
        />
      </div>
      <span className="text-xs tabular-nums w-8">{score.toFixed(1)}</span>
    </div>
  );
}

function UncertaintyIndicator({ uncertainty }: { uncertainty: number | null }) {
  if (uncertainty === null) return <HelpCircle className="h-4 w-4 text-muted-foreground" />;
  
  if (uncertainty < 0.2) {
    return (
      <Tooltip>
        <TooltipTrigger>
          <CheckCircle className="h-4 w-4 text-green-600" />
        </TooltipTrigger>
        <TooltipContent>Low uncertainty: {(uncertainty * 100).toFixed(0)}%</TooltipContent>
      </Tooltip>
    );
  }
  if (uncertainty < 0.5) {
    return (
      <Tooltip>
        <TooltipTrigger>
          <MinusCircle className="h-4 w-4 text-amber-500" />
        </TooltipTrigger>
        <TooltipContent>Medium uncertainty: {(uncertainty * 100).toFixed(0)}%</TooltipContent>
      </Tooltip>
    );
  }
  return (
    <Tooltip>
      <TooltipTrigger>
        <AlertTriangle className="h-4 w-4 text-red-500" />
      </TooltipTrigger>
      <TooltipContent>High uncertainty: {(uncertainty * 100).toFixed(0)}%</TooltipContent>
    </Tooltip>
  );
}

function AssayStatusBadge({ outcome }: { outcome: string | null }) {
  if (!outcome) {
    return <Badge variant="outline" className="text-xs">Not tested</Badge>;
  }
  
  const config: Record<string, { color: string; icon: typeof CheckCircle }> = {
    active: { color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200", icon: CheckCircle },
    inactive: { color: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200", icon: XCircle },
    toxic: { color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200", icon: AlertTriangle },
    inconclusive: { color: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200", icon: HelpCircle },
  };
  
  const cfg = config[outcome] || config.inconclusive;
  const Icon = cfg.icon;
  
  return (
    <Badge variant="secondary" className={`text-xs ${cfg.color}`}>
      <Icon className="h-3 w-3 mr-1" />
      {outcome.charAt(0).toUpperCase() + outcome.slice(1)}
    </Badge>
  );
}

function AgreementBadge({ level }: { level: "strong" | "good" | "mixed" | null }) {
  if (!level) return <span className="text-muted-foreground text-xs">-</span>;
  
  const config = {
    strong: { 
      color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200", 
      label: "STRONG",
      icon: CheckCircle,
    },
    good: { 
      color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200", 
      label: "GOOD",
      icon: CheckCircle,
    },
    mixed: { 
      color: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200", 
      label: "MIXED",
      icon: AlertTriangle,
    },
  };
  
  const cfg = config[level];
  const Icon = cfg.icon;
  
  return (
    <Badge variant="secondary" className={`text-xs ${cfg.color}`}>
      <Icon className="h-3 w-3 mr-1" />
      {cfg.label}
    </Badge>
  );
}

function AffinityScore({ score, unit = "nM" }: { score: number | null; unit?: string }) {
  if (score === null) return <span className="text-muted-foreground text-xs">N/A</span>;
  
  const stars = score < 10 ? 3 : score < 50 ? 2 : 1;
  const starDisplay = "★".repeat(stars) + "☆".repeat(3 - stars);
  
  return (
    <div className="flex items-center gap-1">
      <span className="text-xs tabular-nums font-medium">{score.toFixed(1)} {unit}</span>
      <span className="text-xs text-amber-500">{starDisplay}</span>
    </div>
  );
}

function DockingScore({ score }: { score: number | null }) {
  if (score === null) return <span className="text-muted-foreground text-xs">N/A</span>;
  
  const absScore = Math.abs(score);
  const stars = absScore > 8 ? 3 : absScore > 6 ? 2 : 1;
  const starDisplay = "★".repeat(stars) + "☆".repeat(3 - stars);
  
  return (
    <div className="flex items-center gap-1">
      <span className="text-xs tabular-nums font-medium">{score.toFixed(1)}</span>
      <span className="text-xs text-amber-500">{starDisplay}</span>
    </div>
  );
}

export default function HitTriagePage() {
  const params = useParams<{ id: string }>();
  const campaignId = params.id;
  const [, navigate] = useLocation();
  const { toast } = useToast();
  
  const [filters, setFilters] = useState<HitTriageFilters>(defaultFilters);
  const [selectedHits, setSelectedHits] = useState<Set<number>>(new Set());
  const [showFilters, setShowFilters] = useState(true);
  const [sendToAssayOpen, setSendToAssayOpen] = useState(false);
  const [selectedAssayId, setSelectedAssayId] = useState<string>("");
  
  const debouncedFilters = useDebounce(filters, 300);
  
  const { data: campaign, isLoading: campaignLoading } = useQuery<Campaign>({
    queryKey: ["/api/campaigns", campaignId],
    enabled: !!campaignId,
  });
  
  const queryParams = useMemo(() => {
    const params = new URLSearchParams();
    if (debouncedFilters.minOracleScore > 0) params.set("min_oracle_score", String(debouncedFilters.minOracleScore));
    if (debouncedFilters.maxOracleScore < 100) params.set("max_oracle_score", String(debouncedFilters.maxOracleScore));
    if (debouncedFilters.minDockingScore > 0) params.set("min_docking_score", String(debouncedFilters.minDockingScore));
    if (debouncedFilters.minAdmetScore > 0) params.set("min_admet_score", String(debouncedFilters.minAdmetScore));
    if (debouncedFilters.minSynthesisScore > 0) params.set("min_synthesis_score", String(debouncedFilters.minSynthesisScore));
    if (debouncedFilters.maxUncertainty < 100) params.set("max_uncertainty", String(debouncedFilters.maxUncertainty / 100));
    if (debouncedFilters.ipSafeOnly) params.set("ip_safe_only", "true");
    if (debouncedFilters.assayFilter === "tested") params.set("has_assay_data", "true");
    if (debouncedFilters.assayFilter === "untested") params.set("has_assay_data", "false");
    if (debouncedFilters.agreementFilter !== "all") params.set("agreement_level", debouncedFilters.agreementFilter);
    return params.toString();
  }, [debouncedFilters]);
  
  const { data: hits, isLoading: hitsLoading } = useQuery<HitCandidate[]>({
    queryKey: ["/api/campaigns", campaignId, "hits", queryParams],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/hits?${queryParams}`);
      if (!res.ok) throw new Error("Failed to fetch hits");
      return res.json();
    },
    enabled: !!campaignId,
  });
  
  const { data: assays } = useQuery<Assay[]>({
    queryKey: ["/api/assays"],
  });
  
  const sendToAssayMutation = useMutation({
    mutationFn: async ({ assayId, moleculeIds }: { assayId: number; moleculeIds: number[] }) => {
      return apiRequest("POST", `/api/assays/${assayId}/add-molecules`, { moleculeIds });
    },
    onSuccess: () => {
      toast({ title: "Molecules sent to assay", description: `${selectedHits.size} molecules queued for testing.` });
      setSelectedHits(new Set());
      setSendToAssayOpen(false);
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns", campaignId, "hits"] });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to send molecules to assay", variant: "destructive" });
    },
  });
  
  const exportCSV = useCallback(() => {
    if (!hits) return;
    
    const selectedList = Array.from(selectedHits);
    const toExport = selectedList.length > 0 
      ? hits.filter(h => selectedHits.has(h.moleculeId))
      : hits;
    
    const headers = ["Molecule ID", "Name", "SMILES", "Oracle Score", "Docking", "ADMET", "Synthesis", "Translational", "IP Risk", "Assay Status"];
    const rows = toExport.map(h => [
      h.moleculeId,
      h.moleculeName || "",
      h.smiles,
      h.oracleScore ?? "",
      h.dockingScore ?? "",
      h.admetScore ?? "",
      h.synthesisScore ?? "",
      h.translationalScore ?? "",
      h.ipRiskFlag ? "Yes" : "No",
      h.lastAssayOutcome || "Not tested",
    ]);
    
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hit-triage-${campaignId}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [hits, selectedHits, campaignId]);
  
  const toggleSelectAll = useCallback(() => {
    if (!hits) return;
    if (selectedHits.size === hits.length) {
      setSelectedHits(new Set());
    } else {
      setSelectedHits(new Set(hits.map(h => h.moleculeId)));
    }
  }, [hits, selectedHits]);
  
  const toggleSelectHit = useCallback((moleculeId: number) => {
    setSelectedHits(prev => {
      const next = new Set(prev);
      if (next.has(moleculeId)) {
        next.delete(moleculeId);
      } else {
        next.add(moleculeId);
      }
      return next;
    });
  }, []);
  
  const summary = useMemo(() => {
    if (!hits) return { total: 0, filtered: 0, tested: 0, active: 0, consensusHits: 0, flaggedForReview: 0 };
    return {
      total: hits.length,
      filtered: hits.length,
      tested: hits.filter(h => h.lastAssayOutcome).length,
      active: hits.filter(h => h.lastAssayOutcome === "active").length,
      consensusHits: hits.filter(h => h.agreementLevel === "strong" || h.agreementLevel === "good").length,
      flaggedForReview: hits.filter(h => h.agreementLevel === "mixed").length,
    };
  }, [hits]);
  
  if (campaignLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-[600px] w-full" />
      </div>
    );
  }
  
  if (!campaign) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Target className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Campaign not found</h3>
            <Button onClick={() => navigate("/campaigns")} className="mt-4" data-testid="button-back-to-campaigns">
              Back to Campaigns
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }
  
  return (
    <div className="flex h-full overflow-hidden">
      {showFilters && (
        <div className="w-72 border-r bg-muted/30 p-4 flex flex-col gap-4 overflow-auto">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold flex items-center gap-2">
              <Filter className="h-4 w-4" />
              Filters
            </h2>
            <Button variant="ghost" size="icon" onClick={() => setShowFilters(false)} data-testid="button-hide-filters">
              <ChevronLeft className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="space-y-5">
            <div className="space-y-2">
              <Label className="text-xs">Oracle Score Range</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={filters.minOracleScore}
                  onChange={e => setFilters(f => ({ ...f, minOracleScore: Number(e.target.value) }))}
                  className="w-16 text-xs"
                  min={0}
                  max={100}
                  data-testid="input-min-oracle"
                />
                <span className="text-muted-foreground text-xs">to</span>
                <Input
                  type="number"
                  value={filters.maxOracleScore}
                  onChange={e => setFilters(f => ({ ...f, maxOracleScore: Number(e.target.value) }))}
                  className="w-16 text-xs"
                  min={0}
                  max={100}
                  data-testid="input-max-oracle"
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs">Min Docking Score</Label>
              <Slider
                value={[filters.minDockingScore]}
                onValueChange={([v]) => setFilters(f => ({ ...f, minDockingScore: v }))}
                max={100}
                step={1}
                data-testid="slider-docking"
              />
              <span className="text-xs text-muted-foreground">{filters.minDockingScore}</span>
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs">Min ADMET Score</Label>
              <Slider
                value={[filters.minAdmetScore]}
                onValueChange={([v]) => setFilters(f => ({ ...f, minAdmetScore: v }))}
                max={100}
                step={1}
                data-testid="slider-admet"
              />
              <span className="text-xs text-muted-foreground">{filters.minAdmetScore}</span>
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs">Min Synthesis Score</Label>
              <Slider
                value={[filters.minSynthesisScore]}
                onValueChange={([v]) => setFilters(f => ({ ...f, minSynthesisScore: v }))}
                max={100}
                step={1}
                data-testid="slider-synthesis"
              />
              <span className="text-xs text-muted-foreground">{filters.minSynthesisScore}</span>
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs">Max Uncertainty</Label>
              <Slider
                value={[filters.maxUncertainty]}
                onValueChange={([v]) => setFilters(f => ({ ...f, maxUncertainty: v }))}
                max={100}
                step={5}
                data-testid="slider-uncertainty"
              />
              <span className="text-xs text-muted-foreground">{filters.maxUncertainty}%</span>
            </div>
            
            <div className="flex items-center justify-between">
              <Label className="text-xs">IP-Safe Only</Label>
              <Switch
                checked={filters.ipSafeOnly}
                onCheckedChange={v => setFilters(f => ({ ...f, ipSafeOnly: v }))}
                data-testid="switch-ip-safe"
              />
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs">Assay Status</Label>
              <Select
                value={filters.assayFilter}
                onValueChange={(v: "all" | "tested" | "untested") => setFilters(f => ({ ...f, assayFilter: v }))}
              >
                <SelectTrigger data-testid="select-assay-filter">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All molecules</SelectItem>
                  <SelectItem value="tested">Tested only</SelectItem>
                  <SelectItem value="untested">Untested only</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs">Method Agreement</Label>
              <Select
                value={filters.agreementFilter}
                onValueChange={(v: "all" | "strong" | "good" | "mixed") => setFilters(f => ({ ...f, agreementFilter: v }))}
              >
                <SelectTrigger data-testid="select-agreement-filter">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All compounds</SelectItem>
                  <SelectItem value="strong">Strong agreement</SelectItem>
                  <SelectItem value="good">Good agreement</SelectItem>
                  <SelectItem value="mixed">Mixed (review)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setFilters(defaultFilters)}
              className="w-full"
              data-testid="button-reset-filters"
            >
              Reset Filters
            </Button>
          </div>
        </div>
      )}
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="p-4 border-b space-y-4">
          <div className="flex items-center gap-4 flex-wrap">
            {!showFilters && (
              <Button variant="ghost" size="icon" onClick={() => setShowFilters(true)} data-testid="button-show-filters">
                <ChevronRight className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={() => navigate(`/campaigns/${campaignId}`)} data-testid="button-back">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-xl font-semibold" data-testid="text-page-title">Hit Triage</h1>
              <p className="text-sm text-muted-foreground">{campaign.name}</p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary/10 rounded-md">
                    <FlaskConical className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold tabular-nums" data-testid="stat-total">{summary.total}</p>
                    <p className="text-xs text-muted-foreground">Top Hits</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-500/10 rounded-md">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold tabular-nums" data-testid="stat-consensus">{summary.consensusHits}</p>
                    <p className="text-xs text-muted-foreground">Consensus Hits</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-amber-500/10 rounded-md">
                    <AlertTriangle className="h-5 w-5 text-amber-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold tabular-nums" data-testid="stat-flagged">{summary.flaggedForReview}</p>
                    <p className="text-xs text-muted-foreground">Flagged for Review</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-500/10 rounded-md">
                    <Filter className="h-5 w-5 text-blue-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold tabular-nums" data-testid="stat-filtered">{summary.filtered}</p>
                    <p className="text-xs text-muted-foreground">After Filters</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-500/10 rounded-md">
                    <Beaker className="h-5 w-5 text-purple-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold tabular-nums" data-testid="stat-tested">{summary.tested}</p>
                    <p className="text-xs text-muted-foreground">Assay Tested</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-emerald-500/10 rounded-md">
                    <Activity className="h-5 w-5 text-emerald-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold tabular-nums" data-testid="stat-active">{summary.active}</p>
                    <p className="text-xs text-muted-foreground">Confirmed Actives</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
          
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant="default"
              size="sm"
              disabled={selectedHits.size === 0 || !assays || assays.length === 0}
              onClick={() => setSendToAssayOpen(true)}
              data-testid="button-send-to-assay"
            >
              <Beaker className="h-4 w-4 mr-2" />
              Send to Lab ({selectedHits.size})
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
                const top20 = hits?.slice(0, 20) || [];
                if (top20.length > 0) {
                  const headers = ["Rank", "Compound", "SMILES", "AQAffinity (nM)", "AutoDock", "Agreement", "ADMET", "Synthesis"];
                  const rows = top20.map((h, i) => [
                    i + 1,
                    h.moleculeName || generateMoleculeName(h.smiles, String(h.moleculeId)),
                    h.smiles || "",
                    h.aqaffinityScore ?? "",
                    h.autodockScore ?? "",
                    h.agreementLevel || "",
                    h.admetScore ?? "",
                    h.synthesisScore ?? "",
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
              onClick={() => navigate("/campaigns/new")}
              data-testid="button-new-campaign"
            >
              Start New Campaign
            </Button>
            {selectedHits.size > 0 && (
              <span className="text-sm text-muted-foreground">
                {selectedHits.size} selected
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
                    <p className="text-xs text-muted-foreground">Upload lab validation results to improve predictions</p>
                  </div>
                </div>
                <Button variant="outline" size="sm" data-testid="button-upload-assay">
                  Upload Assay Results
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
        
        <ScrollArea className="flex-1">
          {hitsLoading ? (
            <div className="p-4 space-y-2">
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : !hits || hits.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <FlaskConical className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No hit candidates found</h3>
              <p className="text-muted-foreground text-center max-w-md">
                Try adjusting your filters or run a new campaign to generate hit candidates.
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader className="sticky top-0 bg-background z-10">
                <TableRow>
                  <TableHead className="w-10">
                    <Checkbox
                      checked={hits.length > 0 && selectedHits.size === hits.length}
                      onCheckedChange={toggleSelectAll}
                      data-testid="checkbox-select-all"
                    />
                  </TableHead>
                  <TableHead>Rank</TableHead>
                  <TableHead>Compound</TableHead>
                  <TableHead>AQAffinity</TableHead>
                  <TableHead>AutoDock</TableHead>
                  <TableHead>Agreement</TableHead>
                  <TableHead>ADMET</TableHead>
                  <TableHead>Synthesis</TableHead>
                  <TableHead className="w-10">Conf.</TableHead>
                  <TableHead>Assay Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hits.map((hit, index) => (
                  <TableRow 
                    key={hit.moleculeId} 
                    className={selectedHits.has(hit.moleculeId) ? "bg-muted/50" : ""}
                    data-testid={`row-hit-${hit.moleculeId}`}
                  >
                    <TableCell>
                      <Checkbox
                        checked={selectedHits.has(hit.moleculeId)}
                        onCheckedChange={() => toggleSelectHit(hit.moleculeId)}
                        data-testid={`checkbox-hit-${hit.moleculeId}`}
                      />
                    </TableCell>
                    <TableCell>
                      <span className="font-bold text-lg tabular-nums">{index + 1}</span>
                    </TableCell>
                    <TableCell>
                      <div>
                        <span className="font-medium">{hit.moleculeName || generateMoleculeName(hit.smiles, String(hit.moleculeId))}</span>
                        {hit.ipRiskFlag && (
                          <Tooltip>
                            <TooltipTrigger>
                              <AlertTriangle className="h-3 w-3 text-amber-500 ml-1 inline" />
                            </TooltipTrigger>
                            <TooltipContent>Potential IP risk</TooltipContent>
                          </Tooltip>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <AffinityScore score={hit.aqaffinityScore} unit="nM" />
                    </TableCell>
                    <TableCell>
                      <DockingScore score={hit.autodockScore} />
                    </TableCell>
                    <TableCell>
                      <AgreementBadge level={hit.agreementLevel} />
                    </TableCell>
                    <TableCell>
                      <ScoreBar score={hit.admetScore} color="success" />
                    </TableCell>
                    <TableCell>
                      <ScoreBar score={hit.synthesisScore} color="warning" />
                    </TableCell>
                    <TableCell>
                      <UncertaintyIndicator uncertainty={hit.overallUncertainty} />
                    </TableCell>
                    <TableCell>
                      <AssayStatusBadge outcome={hit.lastAssayOutcome} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </ScrollArea>
      </div>
      
      <Dialog open={sendToAssayOpen} onOpenChange={setSendToAssayOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Send to Assay</DialogTitle>
            <DialogDescription>
              Select an assay to queue {selectedHits.size} molecule(s) for testing.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Target Assay</Label>
              <Select value={selectedAssayId} onValueChange={setSelectedAssayId}>
                <SelectTrigger data-testid="select-target-assay">
                  <SelectValue placeholder="Select an assay" />
                </SelectTrigger>
                <SelectContent>
                  {assays?.map(assay => (
                    <SelectItem key={assay.id} value={String(assay.id)}>
                      {assay.name} ({assay.type})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div className="bg-muted rounded-md p-3">
              <p className="text-sm">
                <strong>{selectedHits.size}</strong> molecules will be added to the selected assay for testing.
              </p>
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setSendToAssayOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (selectedAssayId) {
                  sendToAssayMutation.mutate({
                    assayId: Number(selectedAssayId),
                    moleculeIds: Array.from(selectedHits),
                  });
                }
              }}
              disabled={!selectedAssayId || sendToAssayMutation.isPending}
              data-testid="button-confirm-send"
            >
              {sendToAssayMutation.isPending ? "Sending..." : "Confirm"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
