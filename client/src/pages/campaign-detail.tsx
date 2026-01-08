import { useQuery, useMutation } from "@tanstack/react-query";
import { Link, useParams } from "wouter";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Play,
  Download,
  Workflow,
  FlaskConical,
  CheckCircle,
  Clock,
  AlertCircle,
  Loader2,
  ArrowRight,
  Sparkles,
  Target,
  Beaker,
  Atom,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Campaign, Job, MoleculeScore, PipelineConfig } from "@shared/schema";

interface CampaignWithDetails extends Campaign {
  jobs?: Job[];
  moleculeScores?: (MoleculeScore & { molecule?: { smiles: string } })[];
}

const jobTypeLabels: Record<string, string> = {
  generation: "Hit Generation",
  filtering: "Screening Cascade",
  docking: "Virtual Docking",
  scoring: "Hit Prioritization",
  quantum_optimization: "Quantum Optimization",
  quantum_scoring: "Quantum Scoring",
  assay_validation: "Assay Validation",
  sar_feedback: "SAR Feedback",
};

const jobTypeIcons: Record<string, typeof Sparkles> = {
  generation: Sparkles,
  filtering: Beaker,
  docking: Target,
  scoring: FlaskConical,
  quantum_optimization: Atom,
  quantum_scoring: Atom,
};

export default function CampaignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { toast } = useToast();

  const { data: campaign, isLoading } = useQuery<CampaignWithDetails>({
    queryKey: ["/api/campaigns", id],
    refetchInterval: (query) => {
      const data = query.state.data as CampaignWithDetails | undefined;
      return data?.status === "running" ? 3000 : false;
    },
  });

  const startMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", `/api/campaigns/${id}/start`, {});
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/campaigns", id] });
      toast({ title: "Campaign started", description: "The pipeline is now running." });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to start campaign", variant: "destructive" });
    },
  });

  const handleExport = () => {
    const scores = campaign?.moleculeScores || [];
    if (scores.length === 0) {
      toast({ title: "No data", description: "No molecules to export", variant: "destructive" });
      return;
    }

    const csv = [
      ["SMILES", "Docking Score", "ADMET Score", "QSAR Score", "Oracle Score"].join(","),
      ...scores.map((s) =>
        [
          `"${s.molecule?.smiles || ""}"`,
          s.dockingScore?.toFixed(4) || "",
          s.admetScore?.toFixed(4) || "",
          s.qsarScore?.toFixed(4) || "",
          s.oracleScore?.toFixed(4) || "",
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${campaign?.name || "campaign"}-molecules.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Campaigns", href: "/campaigns" }, { label: "Loading..." }]} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-64 rounded-lg" />
          </div>
        </main>
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Campaigns", href: "/campaigns" }, { label: "Not Found" }]} />
        <main className="flex-1 overflow-auto p-6">
          <Card>
            <CardContent className="py-16 text-center">
              <p className="text-muted-foreground">Campaign not found</p>
              <Link href="/campaigns">
                <Button variant="outline" className="mt-4">Back to Campaigns</Button>
              </Link>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  const pipelineConfig = campaign.pipelineConfig as PipelineConfig | null;

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Campaigns", href: "/campaigns" },
          { label: campaign.name },
        ]}
        actions={
          <div className="flex items-center gap-2">
            {campaign.status === "pending" && (
              <Button
                onClick={() => startMutation.mutate()}
                disabled={startMutation.isPending}
                className="gap-2"
                data-testid="button-start-campaign"
              >
                {startMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                Start Campaign
              </Button>
            )}
            <Button variant="outline" onClick={handleExport} className="gap-2" data-testid="button-export">
              <Download className="h-4 w-4" />
              Export CSV
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-start justify-between gap-4 flex-wrap">
              <div className="space-y-1">
                <CardTitle className="text-2xl" data-testid="text-campaign-name">
                  {campaign.name}
                </CardTitle>
                <div className="flex items-center gap-2 flex-wrap">
                  {campaign.domainType && <DiseaseAreaBadge area={campaign.domainType} />}
                  <StatusBadge status={campaign.status || "pending"} />
                </div>
              </div>
              {campaign.status === "running" && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Pipeline running...
                </div>
              )}
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Pipeline Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between gap-4 overflow-x-auto pb-2">
                {["generation", "filtering", "docking", "scoring"].map((type, index) => {
                  const job = campaign.jobs?.find((j) => j.type === type);
                  const Icon = jobTypeIcons[type] || Workflow;
                  
                  return (
                    <div key={type} className="flex items-center gap-2">
                      <div className="flex flex-col items-center min-w-[120px]">
                        <div
                          className={`w-14 h-14 rounded-lg flex items-center justify-center ${
                            job?.status === "completed"
                              ? "bg-emerald-100 dark:bg-emerald-900/30"
                              : job?.status === "running"
                              ? "bg-blue-100 dark:bg-blue-900/30"
                              : job?.status === "failed"
                              ? "bg-red-100 dark:bg-red-900/30"
                              : "bg-muted"
                          }`}
                        >
                          {job?.status === "running" ? (
                            <Loader2 className="h-6 w-6 animate-spin text-blue-600 dark:text-blue-400" />
                          ) : job?.status === "completed" ? (
                            <CheckCircle className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
                          ) : job?.status === "failed" ? (
                            <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
                          ) : (
                            <Icon className="h-6 w-6 text-muted-foreground" />
                          )}
                        </div>
                        <span className="text-sm font-medium mt-2 text-center">
                          {jobTypeLabels[type]}
                        </span>
                        {job && (
                          <Badge
                            variant="outline"
                            className="mt-1 text-xs capitalize no-default-hover-elevate no-default-active-elevate"
                          >
                            {job.status}
                          </Badge>
                        )}
                      </div>
                      {index < 3 && (
                        <ArrowRight className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                      )}
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="molecules" className="space-y-4">
            <TabsList>
              <TabsTrigger value="molecules" className="gap-2" data-testid="tab-molecules">
                <FlaskConical className="h-4 w-4" />
                Hit Candidates
              </TabsTrigger>
              <TabsTrigger value="assay" className="gap-2" data-testid="tab-assay">
                <Beaker className="h-4 w-4" />
                Assay Validation
              </TabsTrigger>
              <TabsTrigger value="jobs" className="gap-2" data-testid="tab-jobs">
                <Workflow className="h-4 w-4" />
                Hit-to-Lead Progress
              </TabsTrigger>
              <TabsTrigger value="config" className="gap-2" data-testid="tab-config">
                <Target className="h-4 w-4" />
                Configuration
              </TabsTrigger>
            </TabsList>

            <TabsContent value="molecules">
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                      <CardTitle className="text-lg">Virtual Hit Candidates</CardTitle>
                      <p className="text-sm text-muted-foreground">Prioritized molecules from screening cascade</p>
                    </div>
                    <Badge variant="outline" className="text-xs no-default-hover-elevate no-default-active-elevate">
                      {campaign.moleculeScores?.length || 0} hits identified
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  {campaign.moleculeScores && campaign.moleculeScores.length > 0 ? (
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>SMILES</TableHead>
                            <TableHead className="text-right">Docking Score</TableHead>
                            <TableHead className="text-right">ADMET Score</TableHead>
                            <TableHead className="text-right">SAR Score</TableHead>
                            <TableHead className="text-right">Hit Score</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {campaign.moleculeScores
                            .sort((a, b) => (b.oracleScore || 0) - (a.oracleScore || 0))
                            .slice(0, 20)
                            .map((score) => (
                              <TableRow key={score.id} data-testid={`row-score-${score.id}`}>
                                <TableCell>
                                  <code className="text-xs font-mono max-w-xs truncate block">
                                    {score.molecule?.smiles}
                                  </code>
                                </TableCell>
                                <TableCell className="text-right font-mono">
                                  {score.dockingScore?.toFixed(3) || "-"}
                                </TableCell>
                                <TableCell className="text-right font-mono">
                                  {score.admetScore?.toFixed(3) || "-"}
                                </TableCell>
                                <TableCell className="text-right font-mono">
                                  {score.qsarScore?.toFixed(3) || "-"}
                                </TableCell>
                                <TableCell className="text-right font-mono font-semibold">
                                  {score.oracleScore?.toFixed(3) || "-"}
                                </TableCell>
                              </TableRow>
                            ))}
                        </TableBody>
                      </Table>
                    </div>
                  ) : (
                    <div className="py-16 text-center">
                      <FlaskConical className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        {campaign.status === "pending"
                          ? "Start the campaign to begin hit identification"
                          : campaign.status === "running"
                          ? "Virtual hits will appear here as they are screened"
                          : "No hit candidates found"}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="assay">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                      <CardTitle className="text-lg">Assay Validation Queue</CardTitle>
                      <p className="text-sm text-muted-foreground">Hits pending experimental validation</p>
                    </div>
                    <Badge variant="outline" className="text-xs no-default-hover-elevate no-default-active-elevate">
                      Bioassay tracking
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="p-4 rounded-lg bg-muted/50 border">
                      <p className="text-sm text-muted-foreground mb-1">Awaiting Assay</p>
                      <p className="text-2xl font-semibold">0</p>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/50 border">
                      <p className="text-sm text-muted-foreground mb-1">In Testing</p>
                      <p className="text-2xl font-semibold">0</p>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/50 border">
                      <p className="text-sm text-muted-foreground mb-1">Validated Hits</p>
                      <p className="text-2xl font-semibold">0</p>
                    </div>
                  </div>
                  <div className="py-8 text-center border rounded-lg">
                    <Beaker className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-muted-foreground mb-2">No assay results yet</p>
                    <p className="text-sm text-muted-foreground">
                      Submit top hits for dose-response and selectivity assays
                    </p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="jobs">
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                      <CardTitle className="text-lg">Hit-to-Lead Progress</CardTitle>
                      <p className="text-sm text-muted-foreground">Screening cascade and optimization pipeline status</p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  {campaign.jobs && campaign.jobs.length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Pipeline Step</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Started</TableHead>
                          <TableHead>Completed</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {campaign.jobs.map((job) => (
                          <TableRow key={job.id}>
                            <TableCell className="font-medium capitalize">
                              {jobTypeLabels[job.type] || job.type}
                            </TableCell>
                            <TableCell>
                              <StatusBadge status={job.status || "pending"} />
                            </TableCell>
                            <TableCell className="text-muted-foreground">
                              {job.startedAt
                                ? formatDistanceToNow(new Date(job.startedAt), { addSuffix: true })
                                : "-"}
                            </TableCell>
                            <TableCell className="text-muted-foreground">
                              {job.finishedAt
                                ? formatDistanceToNow(new Date(job.finishedAt), { addSuffix: true })
                                : "-"}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="py-16 text-center">
                      <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        No pipeline steps yet. Start the campaign to begin hit discovery.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="config">
              <Card>
                <CardContent className="pt-6">
                  {pipelineConfig ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Generator</p>
                          <p className="font-medium capitalize">
                            {pipelineConfig.generator?.replace("_", " ")}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Docking Method</p>
                          <p className="font-medium capitalize">
                            {pipelineConfig.dockingMethod?.replace("_", " ")}
                          </p>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground mb-2">Scoring Weights</p>
                        <div className="flex gap-4">
                          <Badge variant="outline" className="no-default-hover-elevate no-default-active-elevate">
                            Docking: {pipelineConfig.scoringWeights?.wDocking}
                          </Badge>
                          <Badge variant="outline" className="no-default-hover-elevate no-default-active-elevate">
                            ADMET: {pipelineConfig.scoringWeights?.wAdmet}
                          </Badge>
                          <Badge variant="outline" className="no-default-hover-elevate no-default-active-elevate">
                            QSAR: {pipelineConfig.scoringWeights?.wQsar}
                          </Badge>
                        </div>
                      </div>
                      {pipelineConfig.filteringRules && pipelineConfig.filteringRules.length > 0 && (
                        <div>
                          <p className="text-sm text-muted-foreground mb-2">Filtering Rules</p>
                          <div className="flex gap-2 flex-wrap">
                            {pipelineConfig.filteringRules.map((rule) => (
                              <Badge key={rule} variant="secondary" className="capitalize">
                                {rule}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-muted-foreground">No configuration available</p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}
