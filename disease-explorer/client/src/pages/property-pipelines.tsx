import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { MetricCard } from "@/components/metric-card";
import { StatusBadge } from "@/components/status-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Zap,
  Activity,
  TrendingUp,
  Clock,
  CheckCircle2,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
  Filter,
  ChevronDown,
  ChevronUp,
  Layers,
} from "lucide-react";
import type { ProcessingJob } from "@shared/schema";

interface ProcessingJobsResponse {
  jobs: ProcessingJob[];
  total: number;
}

interface PipelineMetrics {
  totalVariantsToday: number;
  predictionsToday: number;
  activeJobs: number;
  avgThroughput: number;
}

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

function JobProgressCard({ job }: { job: ProcessingJob }) {
  const progress = job.progressPercent || 0;
  const isRunning = job.status === "running";
  const isQueued = job.status === "queued";
  const isFailed = job.status === "failed";
  const isSucceeded = job.status === "succeeded";

  const getStatusColor = () => {
    if (isFailed) return "text-destructive";
    if (isSucceeded) return "text-green-600 dark:text-green-400";
    if (isRunning) return "text-blue-600 dark:text-blue-400";
    return "text-muted-foreground";
  };

  const getStatusIcon = () => {
    if (isFailed) return <AlertCircle className="h-4 w-4" />;
    if (isSucceeded) return <CheckCircle2 className="h-4 w-4" />;
    if (isRunning) return <Activity className="h-4 w-4 animate-pulse" />;
    if (isQueued) return <Clock className="h-4 w-4" />;
    return <Pause className="h-4 w-4" />;
  };

  return (
    <div 
      className="p-4 border rounded-md bg-card hover-elevate"
      data-testid={`job-card-${job.id}`}
    >
      <div className="flex items-center justify-between gap-4 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className={getStatusColor()}>{getStatusIcon()}</span>
          <span className="font-medium truncate">{job.type.replace(/_/g, " ")}</span>
          <Badge variant="outline" className="text-xs">
            {job.status}
          </Badge>
        </div>
        <div className="text-sm text-muted-foreground whitespace-nowrap">
          {formatNumber(job.itemsCompleted || 0)} / {formatNumber(job.itemsTotal || 0)}
        </div>
      </div>
      
      <Progress value={progress} className="h-2 mb-2" />
      
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{progress.toFixed(1)}% complete</span>
        {isRunning && job.itemsTotal && job.itemsCompleted && (
          <span>
            ~{Math.ceil(((job.itemsTotal - job.itemsCompleted) / Math.max(job.itemsCompleted, 1)) * 60)}s remaining
          </span>
        )}
      </div>
    </div>
  );
}

function ScaleIndicator({ 
  label, 
  value, 
  unit,
  animated = false 
}: { 
  label: string; 
  value: number; 
  unit?: string;
  animated?: boolean;
}) {
  const [displayValue, setDisplayValue] = useState(0);
  
  useEffect(() => {
    if (animated && value > 0) {
      const duration = 1500;
      const steps = 60;
      const increment = value / steps;
      let current = 0;
      const timer = setInterval(() => {
        current += increment;
        if (current >= value) {
          setDisplayValue(value);
          clearInterval(timer);
        } else {
          setDisplayValue(Math.floor(current));
        }
      }, duration / steps);
      return () => clearInterval(timer);
    } else {
      setDisplayValue(value);
    }
  }, [value, animated]);

  return (
    <div className="text-center p-4">
      <div className="text-3xl font-bold font-mono tabular-nums">
        {formatNumber(displayValue)}
        {unit && <span className="text-lg text-muted-foreground ml-1">{unit}</span>}
      </div>
      <div className="text-sm text-muted-foreground mt-1">{label}</div>
    </div>
  );
}

function FilterPanel({
  filters,
  onFiltersChange,
}: {
  filters: {
    topPercentile: number;
    propertyThreshold: number;
    manufacturability: number;
  };
  onFiltersChange: (filters: any) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Card>
      <CardHeader 
        className="pb-2 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between gap-4">
          <CardTitle className="text-base flex items-center gap-2">
            <Filter className="h-4 w-4" />
            Filter Live Results
          </CardTitle>
          <Button variant="ghost" size="icon" data-testid="button-toggle-filters">
            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>
        </div>
      </CardHeader>
      {isExpanded && (
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Top Percentile</Label>
              <span className="text-sm font-mono">{filters.topPercentile}%</span>
            </div>
            <Slider
              value={[filters.topPercentile]}
              onValueChange={([v]) => onFiltersChange({ ...filters, topPercentile: v })}
              min={1}
              max={100}
              step={1}
              data-testid="slider-top-percentile"
            />
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Property Threshold</Label>
              <span className="text-sm font-mono">{filters.propertyThreshold.toFixed(1)}</span>
            </div>
            <Slider
              value={[filters.propertyThreshold]}
              onValueChange={([v]) => onFiltersChange({ ...filters, propertyThreshold: v })}
              min={0}
              max={1}
              step={0.1}
              data-testid="slider-property-threshold"
            />
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Manufacturability Cutoff</Label>
              <span className="text-sm font-mono">{filters.manufacturability.toFixed(1)}</span>
            </div>
            <Slider
              value={[filters.manufacturability]}
              onValueChange={([v]) => onFiltersChange({ ...filters, manufacturability: v })}
              min={0}
              max={1}
              step={0.1}
              data-testid="slider-manufacturability"
            />
          </div>
        </CardContent>
      )}
    </Card>
  );
}

export default function PropertyPipelinesPage() {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [filters, setFilters] = useState({
    topPercentile: 10,
    propertyThreshold: 0.5,
    manufacturability: 0.3,
  });

  const { data: jobsResponse, isLoading: jobsLoading, refetch } = useQuery<ProcessingJobsResponse>({
    queryKey: ["/api/processing-jobs", { status: statusFilter !== "all" ? statusFilter : undefined, limit: 20 }],
    refetchInterval: 3000,
  });

  const jobs = jobsResponse?.jobs || [];
  const totalJobs = jobsResponse?.total || 0;

  const runningJobs = jobs.filter(j => j.status === "running");
  const queuedJobs = jobs.filter(j => j.status === "queued");
  const completedJobs = jobs.filter(j => j.status === "succeeded");

  const totalVariantsProcessing = runningJobs.reduce((acc, j) => acc + (j.itemsTotal || 0), 0);
  const totalVariantsCompleted = runningJobs.reduce((acc, j) => acc + (j.itemsCompleted || 0), 0);
  const overallProgress = totalVariantsProcessing > 0 
    ? (totalVariantsCompleted / totalVariantsProcessing) * 100 
    : 0;

  const todayPredictions = completedJobs.reduce((acc, j) => acc + (j.itemsTotal || 0), 0) + totalVariantsCompleted;

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Property Pipelines" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={() => refetch()} data-testid="button-refresh">
              <RotateCcw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button data-testid="button-new-pipeline">
              <Play className="h-4 w-4 mr-2" />
              New Pipeline
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <Card className="bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 border-none">
            <CardContent className="p-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <ScaleIndicator 
                  label="Variants Evaluating" 
                  value={totalVariantsProcessing}
                  animated
                />
                <ScaleIndicator 
                  label="Predictions Today" 
                  value={todayPredictions}
                  animated
                />
                <ScaleIndicator 
                  label="Active Pipelines" 
                  value={runningJobs.length}
                />
                <ScaleIndicator 
                  label="In Queue" 
                  value={queuedJobs.length}
                />
              </div>
            </CardContent>
          </Card>

          {runningJobs.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between gap-4">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Activity className="h-5 w-5 text-blue-500" />
                    Overall Progress
                  </CardTitle>
                  <div className="text-sm text-muted-foreground">
                    {formatNumber(totalVariantsCompleted)} / {formatNumber(totalVariantsProcessing)} variants
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <Progress value={overallProgress} className="h-4" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xs font-medium text-foreground/80">
                      {overallProgress.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
                  <span>Processing at high throughput</span>
                  <span className="flex items-center gap-1">
                    <Zap className="h-3 w-3 text-yellow-500" />
                    Streaming results
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="flex items-center justify-between gap-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Layers className="h-5 w-5" />
                  Pipeline Jobs
                  <Badge variant="secondary" className="ml-2">{totalJobs}</Badge>
                </h2>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-40" data-testid="select-status-filter">
                    <SelectValue placeholder="Filter by status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="running">Running</SelectItem>
                    <SelectItem value="queued">Queued</SelectItem>
                    <SelectItem value="succeeded">Completed</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {jobsLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : jobs.length > 0 ? (
                <div className="space-y-3">
                  {jobs.map((job) => (
                    <JobProgressCard key={job.id} job={job} />
                  ))}
                </div>
              ) : (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                    <Activity className="h-12 w-12 text-muted-foreground/50 mb-4" />
                    <p className="text-muted-foreground">No pipeline jobs found</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Start a new pipeline to see progress here
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>

            <div className="space-y-4">
              <FilterPanel filters={filters} onFiltersChange={setFilters} />
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <TrendingUp className="h-4 w-4" />
                    Throughput Stats
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Avg. variants/min</span>
                    <span className="font-mono font-medium">~2,500</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Peak throughput</span>
                    <span className="font-mono font-medium">~8,400/min</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Batch size</span>
                    <span className="font-mono font-medium">10,000</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Parallel workers</span>
                    <span className="font-mono font-medium">8</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button variant="outline" className="w-full justify-start" data-testid="button-pause-all">
                    <Pause className="h-4 w-4 mr-2" />
                    Pause All Pipelines
                  </Button>
                  <Button variant="outline" className="w-full justify-start" data-testid="button-export-results">
                    <TrendingUp className="h-4 w-4 mr-2" />
                    Export Current Results
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
