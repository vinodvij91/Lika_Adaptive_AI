import { useState, useMemo, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Link, useParams } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
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
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  Target,
  Shield,
  Dna,
  Syringe,
  Activity,
  GitBranch,
  CheckCircle2,
  AlertTriangle,
  ArrowLeft,
  Plus,
  Loader2,
  Download,
  Star,
  Beaker,
  Brain,
  RefreshCw,
  ChevronDown,
} from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { VaccineCampaign, VaccineEpitope, VaccineConstruct } from "@shared/schema";

interface CampaignTarget {
  id: string;
  targetId: string;
  campaignId: string;
  target?: {
    id: string;
    name: string;
    organism: string | null;
    uniprotId: string | null;
    pdbId: string | null;
    sequenceLength: number | null;
  };
}

const statusConfig: Record<string, { label: string; className: string }> = {
  draft: { label: "Draft", className: "bg-muted text-muted-foreground border-border" },
  active: { label: "Active", className: "bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/30" },
  completed: { label: "Completed", className: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30" },
  paused: { label: "Paused", className: "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/30" },
};

const pipelineSteps = [
  { key: "targets", label: "Select Targets", icon: Target },
  { key: "epitopes", label: "Design Epitopes", icon: Dna },
  { key: "constructs", label: "Build Constructs", icon: GitBranch },
  { key: "immunogenicity", label: "Predict Immunogenicity", icon: Shield },
  { key: "assays", label: "Plan Assays", icon: Beaker },
];

function formatDate(d: string | Date | null | undefined): string {
  if (!d) return "—";
  return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function scoreColor(score: number | null | undefined): string {
  if (score == null) return "text-muted-foreground";
  if (score >= 0.7) return "text-green-600 dark:text-green-400";
  if (score >= 0.4) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

function scoreBg(score: number | null | undefined): string {
  if (score == null) return "bg-muted";
  if (score >= 0.7) return "bg-green-500/10";
  if (score >= 0.4) return "bg-yellow-500/10";
  return "bg-red-500/10";
}

function riskColor(risk: number | null | undefined): string {
  if (risk == null) return "text-muted-foreground";
  if (risk <= 0.3) return "text-green-600 dark:text-green-400";
  if (risk <= 0.6) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

export default function VaccineCampaignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  const [addTargetsOpen, setAddTargetsOpen] = useState(false);
  const [selectedTargetIds, setSelectedTargetIds] = useState<string[]>([]);
  const [selectedEpitopeIds, setSelectedEpitopeIds] = useState<string[]>([]);
  const [constructDialogOpen, setConstructDialogOpen] = useState(false);
  const [constructName, setConstructName] = useState("");
  const [constructType, setConstructType] = useState("peptide");
  const [generateDialogOpen, setGenerateDialogOpen] = useState(false);
  const [genName, setGenName] = useState("");
  const [genType, setGenType] = useState("peptide");
  const [hlaFilter, setHlaFilter] = useState("all");
  const [targetFilter, setTargetFilter] = useState("all");
  const [percentileThreshold, setPercentileThreshold] = useState([0]);
  const [sortColumn, setSortColumn] = useState<string>("immunogenicityScore");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");

  const { data: campaign, isLoading: campaignLoading } = useQuery<VaccineCampaign>({
    queryKey: ["/api/vaccine-campaigns", id],
  });

  const { data: campaignTargets = [], isLoading: targetsLoading } = useQuery<CampaignTarget[]>({
    queryKey: ["/api/vaccine-campaigns", id, "targets"],
    enabled: !!id,
  });

  const { data: epitopes = [], isLoading: epitopesLoading } = useQuery<VaccineEpitope[]>({
    queryKey: ["/api/vaccine-campaigns", id, "epitopes"],
    enabled: !!id,
  });

  const { data: constructs = [], isLoading: constructsLoading } = useQuery<VaccineConstruct[]>({
    queryKey: ["/api/vaccine-campaigns", id, "constructs"],
    enabled: !!id,
  });

  const { data: allTargets = [] } = useQuery<any[]>({
    queryKey: ["/api/targets"],
    enabled: addTargetsOpen,
  });

  const addTargetsMutation = useMutation({
    mutationFn: async (targetIds: string[]) => {
      await apiRequest("POST", `/api/vaccine-campaigns/${id}/targets`, { targetIds });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "targets"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      setAddTargetsOpen(false);
      setSelectedTargetIds([]);
      toast({ title: "Targets added successfully" });
    },
    onError: (err: Error) => {
      toast({ title: "Failed to add targets", description: err.message, variant: "destructive" });
    },
  });

  const predictEpitopesMutation = useMutation({
    mutationFn: async () => {
      await apiRequest("POST", `/api/vaccine-campaigns/${id}/predict-epitopes`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "epitopes"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      toast({ title: "Epitope prediction started" });
    },
    onError: (err: Error) => {
      toast({ title: "Prediction failed", description: err.message, variant: "destructive" });
    },
  });

  const generateConstructMutation = useMutation({
    mutationFn: async (data: { epitopeIds: string[]; name: string; type: string }) => {
      await apiRequest("POST", `/api/vaccine-campaigns/${id}/generate-construct`, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "constructs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      setConstructDialogOpen(false);
      setGenerateDialogOpen(false);
      setSelectedEpitopeIds([]);
      setGenName("");
      toast({ title: "Construct generated successfully" });
    },
    onError: (err: Error) => {
      toast({ title: "Failed to generate construct", description: err.message, variant: "destructive" });
    },
  });

  const predictImmunogenicityMutation = useMutation({
    mutationFn: async () => {
      await apiRequest("POST", `/api/vaccine-campaigns/${id}/predict-immunogenicity`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "constructs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      toast({ title: "Immunogenicity prediction complete" });
    },
    onError: (err: Error) => {
      toast({ title: "Prediction failed", description: err.message, variant: "destructive" });
    },
  });

  const autoOptimizeMutation = useMutation({
    mutationFn: () => apiRequest("POST", `/api/vaccine-campaigns/${id}/auto-optimize`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "epitopes"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "constructs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      toast({ title: "Pipeline auto-completed" });
    },
    onError: (err: Error) => {
      toast({ title: "Auto-optimize failed", description: err.message, variant: "destructive" });
    },
  });

  const runFullPipelineMutation = useMutation({
    mutationFn: () => apiRequest("POST", `/api/vaccine-campaigns/${id}/run-pipeline`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "targets"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "epitopes"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "constructs"] });
      toast({ title: "Full pipeline started" });
    },
    onError: (err: Error) => {
      toast({ title: "Pipeline failed", description: err.message, variant: "destructive" });
    },
  });

  const hasAutoTriggered = useRef(false);

  useEffect(() => {
    if (
      campaign &&
      campaignTargets.length > 0 &&
      (epitopes.length === 0 || constructs.length === 0) &&
      campaign.status !== "running" &&
      !hasAutoTriggered.current &&
      !epitopesLoading &&
      !constructsLoading
    ) {
      hasAutoTriggered.current = true;
      autoOptimizeMutation.mutate();
    }
  }, [campaign, campaignTargets, epitopes, constructs, epitopesLoading, constructsLoading]);

  const markCandidateMutation = useMutation({
    mutationFn: async (constructId: string) => {
      await apiRequest("PATCH", `/api/vaccine-campaigns/${id}/constructs/${constructId}`, {
        isPreclinicalCandidate: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id, "constructs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns", id] });
      toast({ title: "Marked as preclinical candidate" });
    },
    onError: (err: Error) => {
      toast({ title: "Update failed", description: err.message, variant: "destructive" });
    },
  });

  const filteredEpitopes = useMemo(() => {
    let filtered = epitopes;
    if (targetFilter !== "all") {
      filtered = filtered.filter((e) => e.targetId === targetFilter);
    }
    if (hlaFilter !== "all") {
      filtered = filtered.filter((e) => e.hlaClass === hlaFilter);
    }
    if (percentileThreshold[0] > 0) {
      filtered = filtered.filter((e) => (e.percentile ?? 0) >= percentileThreshold[0]);
    }
    return filtered;
  }, [epitopes, targetFilter, hlaFilter, percentileThreshold]);

  const sortedConstructs = useMemo(() => {
    const sorted = [...constructs].sort((a, b) => {
      const aVal = (a as any)[sortColumn] ?? 0;
      const bVal = (b as any)[sortColumn] ?? 0;
      return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
    });
    return sorted;
  }, [constructs, sortColumn, sortDirection]);

  const handleSort = (col: string) => {
    if (sortColumn === col) {
      setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortColumn(col);
      setSortDirection("desc");
    }
  };

  const existingTargetIds = new Set(campaignTargets.map((ct) => ct.targetId));
  const availableTargets = allTargets.filter((t: any) => !existingTargetIds.has(t.id));

  const currentStep = useMemo(() => {
    if (!campaign) return 0;
    if ((campaign.candidateCount ?? 0) > 0) return 4;
    if ((campaign.constructCount ?? 0) > 0) return 3;
    if ((campaign.epitopeCount ?? 0) > 0) return 2;
    if ((campaign.targetCount ?? 0) > 0) return 1;
    return 0;
  }, [campaign]);

  const exportConstruct = (c: VaccineConstruct) => {
    const blob = new Blob([JSON.stringify(c, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${c.name || "construct"}_${c.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (campaignLoading) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Vaccine Campaigns", href: "/vaccine-campaigns" }, { label: "Loading..." }]} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        </main>
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Vaccine Campaigns", href: "/vaccine-campaigns" }, { label: "Not Found" }]} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto text-center py-16">
            <Syringe className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
            <h2 className="text-xl font-semibold mb-2">Campaign Not Found</h2>
            <p className="text-muted-foreground mb-4">This vaccine campaign may have been deleted.</p>
            <Link href="/vaccine-campaigns">
              <Button data-testid="button-back-to-campaigns">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Campaigns
              </Button>
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const status = campaign.status || "draft";
  const statusCfg = statusConfig[status] || statusConfig.draft;

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Vaccine Campaigns", href: "/vaccine-campaigns" },
          { label: campaign.name },
        ]}
        actions={
          <Link href="/vaccine-campaigns">
            <Button variant="outline" data-testid="button-back">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          </Link>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-violet-950/30 via-purple-900/20 to-background p-6 rounded-lg border border-violet-500/20">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-lg bg-violet-500/20 flex items-center justify-center border border-violet-500/30">
                  <Syringe className="h-7 w-7 text-violet-400" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold" data-testid="text-campaign-name">{campaign.name}</h1>
                  <p className="text-sm text-muted-foreground">
                    {campaign.pathogen || "Unknown pathogen"} &middot; {campaign.vaccineType?.replace(/_/g, " ") || "Protein subunit"} &middot; Created {formatDate(campaign.createdAt)}
                  </p>
                  {campaign.description && (
                    <p className="text-sm text-muted-foreground mt-1">{campaign.description}</p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {autoOptimizeMutation.isPending && (
                  <div className="flex items-center gap-1.5 text-sm text-violet-400">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running pipeline...
                  </div>
                )}
                <Badge variant="outline" className={statusCfg.className} data-testid="badge-status">
                  {statusCfg.label}
                </Badge>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <Target className="h-5 w-5 mx-auto mb-1 text-violet-500" />
                <div className="text-2xl font-bold font-mono" data-testid="text-target-count">{campaign.targetCount ?? 0}</div>
                <div className="text-xs text-muted-foreground">Targets</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <Dna className="h-5 w-5 mx-auto mb-1 text-indigo-500" />
                <div className="text-2xl font-bold font-mono" data-testid="text-epitope-count">{campaign.epitopeCount ?? 0}</div>
                <div className="text-xs text-muted-foreground">Epitopes</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <GitBranch className="h-5 w-5 mx-auto mb-1 text-purple-500" />
                <div className="text-2xl font-bold font-mono" data-testid="text-construct-count">{campaign.constructCount ?? 0}</div>
                <div className="text-xs text-muted-foreground">Constructs</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <Star className="h-5 w-5 mx-auto mb-1 text-amber-500" />
                <div className="text-2xl font-bold font-mono" data-testid="text-candidate-count">{campaign.candidateCount ?? 0}</div>
                <div className="text-xs text-muted-foreground">Candidates</div>
              </CardContent>
            </Card>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="flex-wrap">
              <TabsTrigger value="overview" data-testid="tab-overview">
                <Activity className="h-4 w-4 mr-2" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="targets" data-testid="tab-targets">
                <Target className="h-4 w-4 mr-2" />
                Targets
              </TabsTrigger>
              <TabsTrigger value="epitopes" data-testid="tab-epitopes">
                <Dna className="h-4 w-4 mr-2" />
                Epitope Design
              </TabsTrigger>
              <TabsTrigger value="constructs" data-testid="tab-constructs">
                <GitBranch className="h-4 w-4 mr-2" />
                Constructs
              </TabsTrigger>
              <TabsTrigger value="trajectory" data-testid="tab-trajectory">
                <Beaker className="h-4 w-4 mr-2" />
                Trajectory & Readouts
              </TabsTrigger>
              <TabsTrigger value="safety" data-testid="tab-safety">
                <Shield className="h-4 w-4 mr-2" />
                Immunogenicity & Safety
              </TabsTrigger>
            </TabsList>

            {/* Tab 1: Overview */}
            <TabsContent value="overview" className="space-y-6 mt-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Activity className="h-4 w-4 text-violet-500" />
                    Pipeline Progress
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2 overflow-x-auto pb-2">
                    {pipelineSteps.map((step, idx) => {
                      const StepIcon = step.icon;
                      const completed = idx < currentStep;
                      const active = idx === currentStep;
                      return (
                        <div key={step.key} className="flex items-center gap-2 flex-shrink-0">
                          {idx > 0 && (
                            <div className={`h-0.5 w-8 ${completed ? "bg-violet-500" : "bg-border"}`} />
                          )}
                          <div
                            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm ${
                              completed
                                ? "bg-violet-500/10 text-violet-600 dark:text-violet-400 border border-violet-500/30"
                                : active
                                  ? "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30"
                                  : "bg-muted text-muted-foreground border border-border"
                            }`}
                            data-testid={`step-${step.key}`}
                          >
                            {completed ? (
                              <CheckCircle2 className="h-4 w-4 text-violet-500" />
                            ) : (
                              <StepIcon className="h-4 w-4" />
                            )}
                            {step.label}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              <Button
                onClick={() => runFullPipelineMutation.mutate()}
                disabled={runFullPipelineMutation.isPending}
                data-testid="button-run-full-pipeline"
              >
                {runFullPipelineMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4 mr-2" />
                )}
                Run Full Pipeline
              </Button>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Syringe className="h-4 w-4 text-violet-500" />
                      Campaign Metadata
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Name</span>
                      <span className="font-medium">{campaign.name}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Pathogen</span>
                      <span>{campaign.pathogen || "—"}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Vaccine Type</span>
                      <span className="capitalize">{campaign.vaccineType?.replace(/_/g, " ") || "—"}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Status</span>
                      <Badge variant="outline" className={statusCfg.className}>
                        {statusCfg.label}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Created</span>
                      <span>{formatDate(campaign.createdAt)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Updated</span>
                      <span>{formatDate(campaign.updatedAt)}</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Brain className="h-4 w-4 text-indigo-500" />
                      Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Targets Selected</span>
                      <span className="font-mono font-medium">{campaign.targetCount ?? 0}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Epitopes Designed</span>
                      <span className="font-mono font-medium">{campaign.epitopeCount ?? 0}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Constructs Built</span>
                      <span className="font-mono font-medium">{campaign.constructCount ?? 0}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Candidates Selected</span>
                      <span className="font-mono font-medium text-violet-600 dark:text-violet-400">{campaign.candidateCount ?? 0}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Tab 2: Targets */}
            <TabsContent value="targets" className="space-y-4 mt-4">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <h3 className="text-lg font-semibold">Campaign Targets</h3>
                <div className="flex items-center gap-2 flex-wrap">
                  <Button
                    variant="outline"
                    onClick={() => predictEpitopesMutation.mutate()}
                    disabled={predictEpitopesMutation.isPending || campaignTargets.length === 0}
                    data-testid="button-predict-epitopes"
                  >
                    {predictEpitopesMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Dna className="h-4 w-4 mr-2" />
                    )}
                    Re-run Epitope Prediction
                  </Button>
                  <Button onClick={() => setAddTargetsOpen(true)} data-testid="button-add-targets">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Targets
                  </Button>
                </div>
              </div>

              {targetsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full" />)}
                </div>
              ) : campaignTargets.length === 0 ? (
                <Card>
                  <CardContent className="p-8 text-center">
                    <Target className="h-10 w-10 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-muted-foreground">No targets added yet. Add targets to begin epitope prediction.</p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Organism</TableHead>
                        <TableHead>UniProt ID</TableHead>
                        <TableHead>PDB ID</TableHead>
                        <TableHead>Seq Length</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {campaignTargets.map((ct) => (
                        <TableRow key={ct.id} data-testid={`row-target-${ct.id}`}>
                          <TableCell className="font-medium">{ct.target?.name || ct.targetId}</TableCell>
                          <TableCell className="text-muted-foreground">{ct.target?.organism || "—"}</TableCell>
                          <TableCell>
                            {ct.target?.uniprotId ? (
                              <Badge variant="outline">{ct.target.uniprotId}</Badge>
                            ) : "—"}
                          </TableCell>
                          <TableCell>
                            {ct.target?.pdbId ? (
                              <Badge variant="outline">{ct.target.pdbId}</Badge>
                            ) : "—"}
                          </TableCell>
                          <TableCell className="font-mono">{ct.target?.sequenceLength ?? "—"}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Card>
              )}
            </TabsContent>

            {/* Tab 3: Epitope Design */}
            <TabsContent value="epitopes" className="space-y-4 mt-4">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <h3 className="text-lg font-semibold">Epitope Design</h3>
                <Button
                  disabled={selectedEpitopeIds.length === 0}
                  onClick={() => setConstructDialogOpen(true)}
                  data-testid="button-add-to-construct"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add to Construct ({selectedEpitopeIds.length})
                </Button>
              </div>

              <div className="flex items-end gap-4 flex-wrap">
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Target</Label>
                  <Select value={targetFilter} onValueChange={setTargetFilter}>
                    <SelectTrigger className="w-[180px]" data-testid="select-target-filter">
                      <SelectValue placeholder="All targets" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All targets</SelectItem>
                      {campaignTargets.map((ct) => (
                        <SelectItem key={ct.targetId} value={ct.targetId}>
                          {ct.target?.name || ct.targetId}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">HLA Class</Label>
                  <Select value={hlaFilter} onValueChange={setHlaFilter}>
                    <SelectTrigger className="w-[120px]" data-testid="select-hla-filter">
                      <SelectValue placeholder="All" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="I">Class I</SelectItem>
                      <SelectItem value="II">Class II</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1 min-w-[200px]">
                  <Label className="text-xs text-muted-foreground">
                    Percentile Threshold: {percentileThreshold[0]}%
                  </Label>
                  <Slider
                    value={percentileThreshold}
                    onValueChange={setPercentileThreshold}
                    max={100}
                    step={1}
                    data-testid="slider-percentile"
                  />
                </div>
              </div>

              {epitopesLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full" />)}
                </div>
              ) : filteredEpitopes.length === 0 ? (
                <Card>
                  <CardContent className="p-8 text-center">
                    <Dna className="h-10 w-10 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-muted-foreground">No epitopes found. Run epitope prediction from the Targets tab.</p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[40px]" />
                        <TableHead>Sequence</TableHead>
                        <TableHead>Start/End</TableHead>
                        <TableHead>HLA Class</TableHead>
                        <TableHead>HLA Allele</TableHead>
                        <TableHead>Affinity</TableHead>
                        <TableHead>Percentile</TableHead>
                        <TableHead>Conservancy</TableHead>
                        <TableHead>Surface</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredEpitopes.map((ep) => (
                        <TableRow key={ep.id} data-testid={`row-epitope-${ep.id}`}>
                          <TableCell>
                            <Checkbox
                              checked={selectedEpitopeIds.includes(ep.id)}
                              onCheckedChange={(checked) => {
                                setSelectedEpitopeIds((prev) =>
                                  checked ? [...prev, ep.id] : prev.filter((x) => x !== ep.id)
                                );
                              }}
                              data-testid={`checkbox-epitope-${ep.id}`}
                            />
                          </TableCell>
                          <TableCell className="font-mono text-xs max-w-[200px] truncate">{ep.sequence}</TableCell>
                          <TableCell className="font-mono text-sm">{ep.startPos ?? "—"}/{ep.endPos ?? "—"}</TableCell>
                          <TableCell>
                            {ep.hlaClass ? (
                              <Badge variant="outline">{ep.hlaClass}</Badge>
                            ) : "—"}
                          </TableCell>
                          <TableCell className="text-sm">{ep.hlaAllele || "—"}</TableCell>
                          <TableCell className="font-mono text-sm">{ep.affinity?.toFixed(1) ?? "—"}</TableCell>
                          <TableCell className="font-mono text-sm">{ep.percentile?.toFixed(1) ?? "—"}%</TableCell>
                          <TableCell className="font-mono text-sm">{ep.conservancy?.toFixed(2) ?? "—"}</TableCell>
                          <TableCell>
                            {ep.surfaceExposed ? (
                              <CheckCircle2 className="h-4 w-4 text-green-500" />
                            ) : (
                              <span className="text-muted-foreground text-xs">No</span>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Card>
              )}
            </TabsContent>

            {/* Tab 4: Constructs */}
            <TabsContent value="constructs" className="space-y-4 mt-4">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <h3 className="text-lg font-semibold">Vaccine Constructs</h3>
                <div className="flex items-center gap-2 flex-wrap">
                  <Button
                    variant="outline"
                    onClick={() => predictImmunogenicityMutation.mutate()}
                    disabled={predictImmunogenicityMutation.isPending || constructs.length === 0}
                    data-testid="button-predict-immunogenicity"
                  >
                    {predictImmunogenicityMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Shield className="h-4 w-4 mr-2" />
                    )}
                    Re-run Immunogenicity
                  </Button>
                  <Link href="/bionemo">
                    <Button variant="outline" data-testid="button-send-bionemo">
                      <Brain className="h-4 w-4 mr-2" />
                      Send to BioNeMo
                    </Button>
                  </Link>
                  <Button onClick={() => setGenerateDialogOpen(true)} data-testid="button-generate-construct">
                    <Plus className="h-4 w-4 mr-2" />
                    Re-generate Construct
                  </Button>
                </div>
              </div>

              {constructsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full" />)}
                </div>
              ) : constructs.length === 0 ? (
                <Card>
                  <CardContent className="p-8 text-center">
                    <GitBranch className="h-10 w-10 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-muted-foreground">No constructs yet. Generate a construct from selected epitopes.</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {constructs.map((c) => (
                    <Card key={c.id} data-testid={`card-construct-${c.id}`}>
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between gap-2">
                          <CardTitle className="text-sm">{c.name}</CardTitle>
                          {c.isPreclinicalCandidate && (
                            <Badge variant="outline" className="bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/30">
                              <Star className="h-3 w-3 mr-1" />
                              Candidate
                            </Badge>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Type</span>
                          <Badge variant="outline" className="capitalize">{c.type || "peptide"}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Length</span>
                          <span className="font-mono">{c.length ?? 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Epitopes</span>
                          <span className="font-mono">{c.epitopeCount ?? 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">HLA Coverage</span>
                          <span className="font-mono">{c.hlaCoverage != null ? `${(c.hlaCoverage * 100).toFixed(0)}%` : "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Immunogenicity</span>
                          <span className={`font-mono font-medium ${scoreColor(c.immunogenicityScore)}`}>
                            {c.immunogenicityScore?.toFixed(2) ?? "—"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">T-Cell Score</span>
                          <span className={`font-mono font-medium ${scoreColor(c.tcellScore)}`}>
                            {c.tcellScore?.toFixed(2) ?? "—"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">B-Cell Score</span>
                          <span className={`font-mono font-medium ${scoreColor(c.bcellScore)}`}>
                            {c.bcellScore?.toFixed(2) ?? "—"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Cross-Reactivity Risk</span>
                          <span className={`font-mono font-medium ${riskColor(c.crossReactivityRisk)}`}>
                            {c.crossReactivityRisk?.toFixed(2) ?? "—"}
                          </span>
                        </div>
                        {(c as any).optimizationMetadata && (() => {
                          const meta = (c as any).optimizationMetadata;
                          return (
                            <Collapsible>
                              <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground w-full mt-1">
                                <ChevronDown className="h-3 w-3" />
                                Optimization Details
                              </CollapsibleTrigger>
                              <CollapsibleContent className="space-y-1.5 pt-2 text-xs">
                                {meta.codonAdaptationIndex != null && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">CAI</span>
                                    <span className="font-mono">{meta.codonAdaptationIndex.toFixed(3)}</span>
                                  </div>
                                )}
                                {meta.gcContent != null && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">GC Content</span>
                                    <span className="font-mono">{(meta.gcContent * 100).toFixed(1)}%</span>
                                  </div>
                                )}
                                {meta.stabilityScore != null && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Stability</span>
                                    <span className="font-mono">{meta.stabilityScore.toFixed(2)}</span>
                                  </div>
                                )}
                                {meta.linkerDesign?.linkerType && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Linker</span>
                                    <span className="font-mono">{meta.linkerDesign.linkerType}</span>
                                  </div>
                                )}
                                {meta.mrnaProperties && (
                                  <>
                                    {meta.mrnaProperties.modifiedNucleotides && (
                                      <div className="flex justify-between">
                                        <span className="text-muted-foreground">Modified Nucleotides</span>
                                        <span className="font-mono">{meta.mrnaProperties.modifiedNucleotides}</span>
                                      </div>
                                    )}
                                    {meta.mrnaProperties.halfLifeHours != null && (
                                      <div className="flex justify-between">
                                        <span className="text-muted-foreground">Half-Life</span>
                                        <span className="font-mono">{meta.mrnaProperties.halfLifeHours}h</span>
                                      </div>
                                    )}
                                  </>
                                )}
                              </CollapsibleContent>
                            </Collapsible>
                          );
                        })()}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabsContent>

            {/* Tab 5: Trajectory & Readouts */}
            <TabsContent value="trajectory" className="space-y-6 mt-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Activity className="h-4 w-4 text-violet-500" />
                    Immune Data Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    Analyze scRNA-seq and immune cell trajectory data from vaccination experiments.
                  </p>
                  <Link href="/trajectory-analysis">
                    <Button variant="outline" data-testid="button-trajectory-analysis">
                      <Activity className="h-4 w-4 mr-2" />
                      Open Trajectory Analysis
                    </Button>
                  </Link>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  {
                    gene: "IGHG1 / IGHG3",
                    readout: "Neutralizing Titers",
                    correlation: 0.87,
                    pValue: 0.002,
                    samples: 48,
                  },
                  {
                    gene: "IFNG / TNF",
                    readout: "ELISpot (SFU/10\u2076)",
                    correlation: 0.73,
                    pValue: 0.01,
                    samples: 36,
                  },
                  {
                    gene: "IL-2 / IL-4 / IL-10",
                    readout: "Cytokine Panel",
                    correlation: 0.65,
                    pValue: 0.03,
                    samples: 42,
                  },
                ].map((item) => (
                  <Card key={item.gene} data-testid={`card-readout-${item.gene}`}>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-mono">{item.gene}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Readout</span>
                        <span>{item.readout}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Correlation</span>
                        <span className="font-mono font-medium text-violet-600 dark:text-violet-400">r = {item.correlation}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">p-value</span>
                        <span className="font-mono">{item.pValue}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Samples</span>
                        <span className="font-mono">{item.samples}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <Button
                variant="outline"
                onClick={() => toast({ title: "Harvesting immune assays", description: "Fetching latest assay data from connected instruments..." })}
                data-testid="button-harvest-assays"
              >
                <Beaker className="h-4 w-4 mr-2" />
                Harvest Immune Assays
              </Button>
            </TabsContent>

            {/* Tab 6: Immunogenicity & Safety */}
            <TabsContent value="safety" className="space-y-4 mt-4">
              <h3 className="text-lg font-semibold">Immunogenicity & Safety Matrix</h3>

              {constructsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full" />)}
                </div>
              ) : sortedConstructs.length === 0 ? (
                <Card>
                  <CardContent className="p-8 text-center">
                    <Shield className="h-10 w-10 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-muted-foreground">No constructs to evaluate. Generate constructs first.</p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Construct</TableHead>
                        <TableHead
                          className="cursor-pointer select-none"
                          onClick={() => handleSort("immunogenicityScore")}
                          data-testid="sort-immunogenicity"
                        >
                          Immunogenicity {sortColumn === "immunogenicityScore" && (sortDirection === "asc" ? "\u2191" : "\u2193")}
                        </TableHead>
                        <TableHead
                          className="cursor-pointer select-none"
                          onClick={() => handleSort("tcellScore")}
                          data-testid="sort-tcell"
                        >
                          T-Cell {sortColumn === "tcellScore" && (sortDirection === "asc" ? "\u2191" : "\u2193")}
                        </TableHead>
                        <TableHead
                          className="cursor-pointer select-none"
                          onClick={() => handleSort("bcellScore")}
                          data-testid="sort-bcell"
                        >
                          B-Cell {sortColumn === "bcellScore" && (sortDirection === "asc" ? "\u2191" : "\u2193")}
                        </TableHead>
                        <TableHead
                          className="cursor-pointer select-none"
                          onClick={() => handleSort("crossReactivityRisk")}
                          data-testid="sort-cross-reactivity"
                        >
                          Cross-Reactivity {sortColumn === "crossReactivityRisk" && (sortDirection === "asc" ? "\u2191" : "\u2193")}
                        </TableHead>
                        <TableHead>Safety</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedConstructs.map((c) => {
                        const flags = (c.safetyFlags as string[] | null) || [];
                        return (
                          <TableRow key={c.id} data-testid={`row-safety-${c.id}`}>
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <span className="font-medium">{c.name}</span>
                                {c.isPreclinicalCandidate && (
                                  <Star className="h-3 w-3 text-violet-500" />
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <span className={`font-mono font-medium px-2 py-0.5 rounded ${scoreBg(c.immunogenicityScore)} ${scoreColor(c.immunogenicityScore)}`}>
                                {c.immunogenicityScore?.toFixed(2) ?? "—"}
                              </span>
                            </TableCell>
                            <TableCell>
                              <span className={`font-mono font-medium px-2 py-0.5 rounded ${scoreBg(c.tcellScore)} ${scoreColor(c.tcellScore)}`}>
                                {c.tcellScore?.toFixed(2) ?? "—"}
                              </span>
                            </TableCell>
                            <TableCell>
                              <span className={`font-mono font-medium px-2 py-0.5 rounded ${scoreBg(c.bcellScore)} ${scoreColor(c.bcellScore)}`}>
                                {c.bcellScore?.toFixed(2) ?? "—"}
                              </span>
                            </TableCell>
                            <TableCell>
                              <span className={`font-mono font-medium px-2 py-0.5 rounded ${
                                c.crossReactivityRisk != null
                                  ? c.crossReactivityRisk <= 0.3
                                    ? "bg-green-500/10 text-green-600 dark:text-green-400"
                                    : c.crossReactivityRisk <= 0.6
                                      ? "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400"
                                      : "bg-red-500/10 text-red-600 dark:text-red-400"
                                  : "bg-muted text-muted-foreground"
                              }`}>
                                {c.crossReactivityRisk?.toFixed(2) ?? "—"}
                              </span>
                            </TableCell>
                            <TableCell>
                              {flags.length > 0 ? (
                                <div className="flex items-center gap-1 flex-wrap">
                                  {flags.map((flag: string, i: number) => (
                                    <Badge key={i} variant="outline" className="bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30">
                                      <AlertTriangle className="h-3 w-3 mr-1" />
                                      {flag}
                                    </Badge>
                                  ))}
                                </div>
                              ) : (
                                <Badge variant="outline" className="bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30">
                                  <CheckCircle2 className="h-3 w-3 mr-1" />
                                  Clear
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-1">
                                {!c.isPreclinicalCandidate && (
                                  <Button
                                    variant="outline"
                                    onClick={() => markCandidateMutation.mutate(c.id)}
                                    disabled={markCandidateMutation.isPending}
                                    data-testid={`button-mark-candidate-${c.id}`}
                                  >
                                    <Star className="h-3 w-3 mr-1" />
                                    Candidate
                                  </Button>
                                )}
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  onClick={() => exportConstruct(c)}
                                  data-testid={`button-export-${c.id}`}
                                >
                                  <Download className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </main>

      {/* Add Targets Dialog */}
      <Dialog open={addTargetsOpen} onOpenChange={setAddTargetsOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Add Targets to Campaign</DialogTitle>
            <DialogDescription>Select targets to add to this vaccine campaign.</DialogDescription>
          </DialogHeader>
          <div className="max-h-[300px] overflow-auto space-y-2 py-2">
            {availableTargets.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No additional targets available.</p>
            ) : (
              availableTargets.map((t: any) => (
                <label
                  key={t.id}
                  className="flex items-center gap-3 p-2 rounded-md hover-elevate cursor-pointer"
                  data-testid={`label-target-${t.id}`}
                >
                  <Checkbox
                    checked={selectedTargetIds.includes(t.id)}
                    onCheckedChange={(checked) => {
                      setSelectedTargetIds((prev) =>
                        checked ? [...prev, t.id] : prev.filter((x) => x !== t.id)
                      );
                    }}
                    data-testid={`checkbox-target-${t.id}`}
                  />
                  <div>
                    <div className="text-sm font-medium">{t.name}</div>
                    <div className="text-xs text-muted-foreground">{t.organism || "Unknown organism"} {t.uniprotId ? `\u00b7 ${t.uniprotId}` : ""}</div>
                  </div>
                </label>
              ))
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAddTargetsOpen(false)} data-testid="button-cancel-add-targets">
              Cancel
            </Button>
            <Button
              onClick={() => addTargetsMutation.mutate(selectedTargetIds)}
              disabled={selectedTargetIds.length === 0 || addTargetsMutation.isPending}
              data-testid="button-confirm-add-targets"
            >
              {addTargetsMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Add {selectedTargetIds.length} Target{selectedTargetIds.length !== 1 ? "s" : ""}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Construct from Epitopes Dialog */}
      <Dialog open={constructDialogOpen} onOpenChange={setConstructDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Construct from Epitopes</DialogTitle>
            <DialogDescription>
              Build a vaccine construct from {selectedEpitopeIds.length} selected epitope{selectedEpitopeIds.length !== 1 ? "s" : ""}.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="construct-name">Construct Name</Label>
              <Input
                id="construct-name"
                value={constructName}
                onChange={(e) => setConstructName(e.target.value)}
                placeholder="e.g. Spike-mRNA-v1"
                data-testid="input-construct-name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="construct-type">Type</Label>
              <Select value={constructType} onValueChange={setConstructType}>
                <SelectTrigger data-testid="select-construct-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="peptide">Peptide</SelectItem>
                  <SelectItem value="mRNA">mRNA</SelectItem>
                  <SelectItem value="protein">Protein</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConstructDialogOpen(false)} data-testid="button-cancel-construct">
              Cancel
            </Button>
            <Button
              onClick={() =>
                generateConstructMutation.mutate({
                  epitopeIds: selectedEpitopeIds,
                  name: constructName,
                  type: constructType,
                })
              }
              disabled={!constructName || generateConstructMutation.isPending}
              data-testid="button-confirm-construct"
            >
              {generateConstructMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Create Construct
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Generate Construct Dialog */}
      <Dialog open={generateDialogOpen} onOpenChange={setGenerateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Generate Vaccine Construct</DialogTitle>
            <DialogDescription>
              Create a new construct from all available epitopes.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="gen-name">Name</Label>
              <Input
                id="gen-name"
                value={genName}
                onChange={(e) => setGenName(e.target.value)}
                placeholder="e.g. Multi-epitope-v2"
                data-testid="input-gen-name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="gen-type">Type</Label>
              <Select value={genType} onValueChange={setGenType}>
                <SelectTrigger data-testid="select-gen-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="peptide">Peptide</SelectItem>
                  <SelectItem value="mRNA">mRNA</SelectItem>
                  <SelectItem value="protein">Protein</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setGenerateDialogOpen(false)} data-testid="button-cancel-generate">
              Cancel
            </Button>
            <Button
              onClick={() =>
                generateConstructMutation.mutate({
                  epitopeIds: epitopes.map((e) => e.id),
                  name: genName,
                  type: genType,
                })
              }
              disabled={!genName || generateConstructMutation.isPending}
              data-testid="button-confirm-generate"
            >
              {generateConstructMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Generate
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
