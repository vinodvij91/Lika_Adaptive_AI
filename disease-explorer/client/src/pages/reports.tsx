import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  BarChart3,
  TrendingUp,
  CheckCircle,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  Dna,
  Atom,
  Play,
  AlertCircle,
} from "lucide-react";
import type { DiseaseArea, ProcessingJob } from "@shared/schema";

interface ReportsData {
  oracleDistribution: { range: string; count: number }[];
  admetPassRate: { passed: number; failed: number };
  domainBreakdown: Record<DiseaseArea, number>;
  recentCampaigns: { name: string; molecules: number; avgScore: number }[];
}

interface PipelineHistoryResponse {
  jobs: ProcessingJob[];
  total: number;
}

const JOB_TYPE_LABELS: Record<string, { label: string; icon: typeof FlaskConical }> = {
  vaccine_discovery: { label: "Vaccine Discovery", icon: Dna },
  drug: { label: "Drug Discovery", icon: FlaskConical },
  materials: { label: "Materials Science", icon: Atom },
  full_pipeline: { label: "Full Pipeline", icon: Play },
};

const STATUS_CONFIG: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; label: string }> = {
  succeeded: { variant: "default", label: "Completed" },
  failed: { variant: "destructive", label: "Failed" },
  running: { variant: "secondary", label: "Running" },
  queued: { variant: "outline", label: "Queued" },
};

function formatDate(date: string | Date | null): string {
  if (!date) return "N/A";
  const d = new Date(date);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function ReportsPage() {
  const [selectedJob, setSelectedJob] = useState<ProcessingJob | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  const { data: reports, isLoading: reportsLoading } = useQuery<ReportsData>({
    queryKey: ["/api/reports"],
  });

  const { data: pipelineHistory, isLoading: historyLoading } = useQuery<PipelineHistoryResponse>({
    queryKey: ["/api/reports/pipeline-history"],
  });

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader breadcrumbs={[{ label: "Reports" }]} />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Clock className="h-5 w-5 text-muted-foreground" />
                Pipeline Run History
              </CardTitle>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : pipelineHistory?.jobs && pipelineHistory.jobs.length > 0 ? (
                <ScrollArea className="h-[400px]">
                  <div className="space-y-2">
                    {pipelineHistory.jobs.map((job) => {
                      const typeConfig = JOB_TYPE_LABELS[job.type] || { label: job.type, icon: Play };
                      const jobStatus = job.status || "queued";
                      const statusConfig = STATUS_CONFIG[jobStatus] || { variant: "outline" as const, label: jobStatus };
                      const Icon = typeConfig.icon;
                      const input = job.inputPayload as any;

                      return (
                        <div
                          key={job.id}
                          data-testid={`pipeline-job-${job.id}`}
                          onClick={() => setSelectedJob(job)}
                          className="flex items-center gap-4 p-4 rounded-md border hover-elevate cursor-pointer"
                        >
                          <div className="flex-shrink-0">
                            <Icon className="h-5 w-5 text-muted-foreground" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium">{typeConfig.label}</span>
                              <Badge variant={statusConfig.variant} className="text-xs">
                                {statusConfig.label}
                              </Badge>
                            </div>
                            <div className="text-sm text-muted-foreground truncate">
                              {input?.sequence ? (
                                <>
                                  Sequence: {input.sequence.substring(0, 30)}
                                  {input.sequence.length > 30 ? "..." : ""} 
                                  ({input.sequenceLength || input.sequence.length} aa)
                                </>
                              ) : input?.vaccineType ? (
                                `Type: ${input.vaccineType}`
                              ) : (
                                "Pipeline run"
                              )}
                            </div>
                          </div>
                          <div className="flex-shrink-0 text-sm text-muted-foreground">
                            {formatDate(job.completedAt || job.createdAt)}
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      );
                    })}
                  </div>
                </ScrollArea>
              ) : (
                <EmptyState message="No pipeline runs yet. Run a vaccine or drug discovery pipeline to see results here." />
              )}
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-muted-foreground" />
                  Oracle Score Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                {reportsLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <div key={i} className="flex items-center gap-3">
                        <Skeleton className="h-4 w-16" />
                        <Skeleton className="h-6 flex-1" />
                        <Skeleton className="h-4 w-8" />
                      </div>
                    ))}
                  </div>
                ) : reports?.oracleDistribution ? (
                  <div className="space-y-3">
                    {reports.oracleDistribution.map((item) => {
                      const max = Math.max(...reports.oracleDistribution.map((d) => d.count));
                      const percentage = max > 0 ? (item.count / max) * 100 : 0;

                      return (
                        <div key={item.range} className="flex items-center gap-3">
                          <span className="text-sm text-muted-foreground w-20">
                            {item.range}
                          </span>
                          <div className="flex-1 h-6 bg-muted rounded-md overflow-hidden">
                            <div
                              className="h-full bg-primary/80 rounded-md transition-all"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono w-10 text-right">
                            {item.count}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <EmptyState message="No score data available" />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-muted-foreground" />
                  ADMET Pass Rate
                </CardTitle>
              </CardHeader>
              <CardContent>
                {reportsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Skeleton className="h-32 w-32 rounded-full" />
                  </div>
                ) : reports?.admetPassRate ? (
                  <div className="flex items-center justify-center gap-8 py-4">
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center mb-2">
                        <CheckCircle className="h-10 w-10 text-emerald-600 dark:text-emerald-400" />
                      </div>
                      <p className="text-2xl font-bold">{reports.admetPassRate.passed}</p>
                      <p className="text-sm text-muted-foreground">Passed</p>
                    </div>
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mb-2">
                        <XCircle className="h-10 w-10 text-red-600 dark:text-red-400" />
                      </div>
                      <p className="text-2xl font-bold">{reports.admetPassRate.failed}</p>
                      <p className="text-sm text-muted-foreground">Failed</p>
                    </div>
                  </div>
                ) : (
                  <EmptyState message="No ADMET data available" />
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Domain Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              {reportsLoading ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[1, 2, 3, 4].map((i) => (
                    <Skeleton key={i} className="h-24 rounded-md" />
                  ))}
                </div>
              ) : reports?.domainBreakdown ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(reports.domainBreakdown).map(([domain, count]) => (
                    <div
                      key={domain}
                      className="p-4 rounded-md bg-muted/50 text-center hover-elevate"
                    >
                      <p className="text-3xl font-bold tabular-nums mb-2">{count}</p>
                      <DiseaseAreaBadge area={domain as DiseaseArea} />
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState message="No domain data available" />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Recent Campaign Performance</CardTitle>
            </CardHeader>
            <CardContent>
              {reportsLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex items-center justify-between">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-12" />
                    </div>
                  ))}
                </div>
              ) : reports?.recentCampaigns && reports.recentCampaigns.length > 0 ? (
                <div className="space-y-4">
                  {reports.recentCampaigns.map((campaign, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 rounded-md hover-elevate"
                    >
                      <span className="font-medium">{campaign.name}</span>
                      <span className="text-sm text-muted-foreground">
                        {campaign.molecules} molecules
                      </span>
                      <span className="text-sm font-mono">
                        Avg: {campaign.avgScore.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState message="No campaign data available" />
              )}
            </CardContent>
          </Card>
        </div>
      </main>

      <Dialog open={!!selectedJob} onOpenChange={() => setSelectedJob(null)}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedJob && (() => {
                const typeConfig = JOB_TYPE_LABELS[selectedJob.type] || { label: selectedJob.type, icon: Play };
                const jobStatus = selectedJob.status || "queued";
                const statusConfig = STATUS_CONFIG[jobStatus] || { variant: "outline" as const, label: jobStatus };
                const IconComponent = typeConfig.icon;
                return (
                  <>
                    <IconComponent className="h-5 w-5" />
                    {typeConfig.label} Results
                    <Badge variant={statusConfig.variant} className="ml-2">
                      {statusConfig.label}
                    </Badge>
                  </>
                );
              })()}
            </DialogTitle>
          </DialogHeader>
          <ScrollArea className="flex-1">
            {selectedJob && <JobDetailsContent job={selectedJob} expandedSections={expandedSections} toggleSection={toggleSection} />}
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function JobDetailsContent({ 
  job, 
  expandedSections, 
  toggleSection 
}: { 
  job: ProcessingJob; 
  expandedSections: Record<string, boolean>;
  toggleSection: (key: string) => void;
}) {
  const input = job.inputPayload as any;
  const output = job.outputPayload as any;

  if (job.status === "failed") {
    return (
      <div className="p-4 bg-destructive/10 rounded-md">
        <div className="flex items-center gap-2 text-destructive mb-2">
          <AlertCircle className="h-5 w-5" />
          <span className="font-medium">Pipeline Failed</span>
        </div>
        <p className="text-sm">{job.errorMessage || "Unknown error"}</p>
      </div>
    );
  }

  const stages = output?.stages || {};
  const mhcEpitopes = stages.mhc1_epitopes;
  const conservation = stages.conservation;
  const vaccineDesign = stages.vaccine_design;

  return (
    <div className="space-y-4 pr-4">
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-muted-foreground">Created:</span>
          <span className="ml-2">{formatDate(job.createdAt)}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Completed:</span>
          <span className="ml-2">{formatDate(job.completedAt)}</span>
        </div>
        {input?.sequence && (
          <div className="col-span-2">
            <span className="text-muted-foreground">Sequence:</span>
            <code className="ml-2 text-xs bg-muted px-2 py-1 rounded break-all">
              {input.sequence}
            </code>
          </div>
        )}
      </div>

      {mhcEpitopes && (
        <Collapsible 
          open={expandedSections["mhc1"]} 
          onOpenChange={() => toggleSection("mhc1")}
        >
          <CollapsibleTrigger asChild>
            <Button 
              variant="ghost" 
              className="w-full justify-between p-4 h-auto"
              data-testid="toggle-mhc1-epitopes"
            >
              <div className="flex items-center gap-2">
                <Dna className="h-4 w-4" />
                <span className="font-medium">T-Cell Epitopes (MHC-I)</span>
                <Badge variant="secondary" className="ml-2">
                  {mhcEpitopes.total_strong_binders || 0} strong binders
                </Badge>
              </div>
              {expandedSections["mhc1"] ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-4 border rounded-md mt-2 space-y-4">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Method:</span>
                  <span className="ml-2">{mhcEpitopes.method}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Peptides tested:</span>
                  <span className="ml-2">{mhcEpitopes.total_peptides_tested}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Threshold:</span>
                  <span className="ml-2">{mhcEpitopes.threshold_nm} nM</span>
                </div>
              </div>

              {Object.entries(mhcEpitopes.predictions || {}).map(([allele, epitopes]: [string, any]) => {
                if (!epitopes || epitopes.length === 0) return null;
                return (
                  <div key={allele} className="space-y-2">
                    <h4 className="font-medium text-sm">{allele}</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-muted-foreground">
                            <th className="pr-4 py-1">Peptide</th>
                            <th className="pr-4 py-1">Position</th>
                            <th className="pr-4 py-1">Affinity (nM)</th>
                            <th className="pr-4 py-1">Rank %</th>
                            <th className="py-1">Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(epitopes as any[]).slice(0, 10).map((e: any, i: number) => (
                            <tr key={i} className="border-t">
                              <td className="pr-4 py-2 font-mono">{e.peptide}</td>
                              <td className="pr-4 py-2">{e.position}</td>
                              <td className="pr-4 py-2">{e.affinity_nm?.toFixed(2)}</td>
                              <td className="pr-4 py-2">{e.percentile_rank?.toFixed(2)}%</td>
                              <td className="py-2">{e.presentation_score?.toFixed(3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {conservation && (
        <Collapsible 
          open={expandedSections["conservation"]} 
          onOpenChange={() => toggleSection("conservation")}
        >
          <CollapsibleTrigger asChild>
            <Button 
              variant="ghost" 
              className="w-full justify-between p-4 h-auto"
              data-testid="toggle-conservation"
            >
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                <span className="font-medium">Conservation Analysis</span>
                <Badge variant="secondary" className="ml-2">
                  {conservation.conserved_regions?.length || 0} regions
                </Badge>
              </div>
              {expandedSections["conservation"] ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-4 border rounded-md mt-2">
              <div className="text-sm text-muted-foreground mb-2">
                Alignment length: {conservation.alignment_length} | 
                Sequences: {conservation.num_sequences}
              </div>
              {conservation.conserved_regions?.length > 0 && (
                <div className="space-y-1">
                  {conservation.conserved_regions.map((region: [number, number], i: number) => (
                    <div key={i} className="text-sm">
                      Region {i + 1}: positions {region[0]} - {region[1]}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {vaccineDesign && (
        <Collapsible 
          open={expandedSections["design"]} 
          onOpenChange={() => toggleSection("design")}
        >
          <CollapsibleTrigger asChild>
            <Button 
              variant="ghost" 
              className="w-full justify-between p-4 h-auto"
              data-testid="toggle-vaccine-design"
            >
              <div className="flex items-center gap-2">
                <FlaskConical className="h-4 w-4" />
                <span className="font-medium">Vaccine Construct Design</span>
                <Badge variant="secondary" className="ml-2">
                  {vaccineDesign.num_epitopes || 0} epitopes
                </Badge>
              </div>
              {expandedSections["design"] ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-4 border rounded-md mt-2 space-y-4">
              <div className="text-sm">
                <span className="text-muted-foreground">Total length:</span>
                <span className="ml-2">{vaccineDesign.total_length} aa</span>
              </div>
              {vaccineDesign.sequence && (
                <div>
                  <span className="text-sm text-muted-foreground">Construct sequence:</span>
                  <code className="block mt-1 text-xs bg-muted p-2 rounded break-all font-mono">
                    {vaccineDesign.sequence}
                  </code>
                </div>
              )}
              {vaccineDesign.annotations?.length > 0 && (
                <div>
                  <span className="text-sm text-muted-foreground">Annotations:</span>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {vaccineDesign.annotations.map((ann: [string, number, number], i: number) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {ann[0]} ({ann[1]}-{ann[2]})
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {output?.raw && (
        <div className="p-4 border rounded-md">
          <span className="text-sm text-muted-foreground">Raw output:</span>
          <pre className="mt-2 text-xs bg-muted p-2 rounded overflow-x-auto max-h-48">
            {output.raw}
          </pre>
        </div>
      )}
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-8">
      <p className="text-muted-foreground">{message}</p>
    </div>
  );
}
