import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState } from "react";
import { 
  ArrowLeft, 
  FileSpreadsheet, 
  CheckCircle2, 
  AlertCircle, 
  Clock,
  Loader2,
  Download,
  ExternalLink
} from "lucide-react";
import type { ImportJob } from "@shared/schema";

const STATUS_CONFIG: Record<string, { color: string; icon: typeof CheckCircle2 }> = {
  pending: { color: "bg-gray-500", icon: Clock },
  parsing: { color: "bg-blue-500", icon: Loader2 },
  validating: { color: "bg-blue-500", icon: Loader2 },
  ingesting: { color: "bg-blue-500", icon: Loader2 },
  succeeded: { color: "bg-green-500", icon: CheckCircle2 },
  failed: { color: "bg-red-500", icon: AlertCircle },
  cancelled: { color: "bg-gray-500", icon: AlertCircle },
};

const IMPORT_TYPE_LABELS: Record<string, string> = {
  compound_library: "Compound Library",
  hit_list: "Hit Lists",
  assay_results: "Assay Results",
  target_structures: "Targets / Structures",
  sar_annotation: "SAR Annotation",
  materials_library: "Materials Library",
  material_variants: "Variants / Formulations",
  properties_dataset: "Properties Dataset",
  simulation_summaries: "Simulation Summaries",
  imaging_spectroscopy: "Imaging / Spectroscopy",
};

export default function ImportHistoryPage() {
  const [, setLocation] = useLocation();
  const [domainFilter, setDomainFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const { data: importJobs = [], isLoading } = useQuery<ImportJob[]>({
    queryKey: ["/api/import-jobs", { 
      domain: domainFilter !== "all" ? domainFilter : undefined,
      status: statusFilter !== "all" ? statusFilter : undefined,
    }],
  });

  const formatDate = (date: Date | string | null) => {
    if (!date) return "—";
    return new Date(date).toLocaleString();
  };

  const formatSize = (bytes: number | null) => {
    if (!bytes) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex-1 overflow-auto p-6" data-testid="import-history-page">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setLocation("/import")}
              data-testid="button-back-to-hub"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-semibold" data-testid="text-page-title">Import History</h1>
              <p className="text-muted-foreground">View past imports and their status</p>
            </div>
          </div>
          <Button onClick={() => setLocation("/import")} data-testid="button-new-import">
            New Import
          </Button>
        </div>

        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Filters</CardTitle>
              <div className="flex gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Domain:</span>
                  <Select value={domainFilter} onValueChange={setDomainFilter}>
                    <SelectTrigger className="w-40" data-testid="select-domain-filter">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Domains</SelectItem>
                      <SelectItem value="drug">Drug Discovery</SelectItem>
                      <SelectItem value="materials">Materials Science</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Status:</span>
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger className="w-40" data-testid="select-status-filter">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Status</SelectItem>
                      <SelectItem value="succeeded">Succeeded</SelectItem>
                      <SelectItem value="failed">Failed</SelectItem>
                      <SelectItem value="pending">Pending</SelectItem>
                      <SelectItem value="ingesting">In Progress</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </CardHeader>
        </Card>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : importJobs.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <FileSpreadsheet className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No imports yet</h3>
              <p className="text-muted-foreground mt-1">
                Start by importing some data into your platform
              </p>
              <Button className="mt-4" onClick={() => setLocation("/import")} data-testid="button-start-import">
                Start Import
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {importJobs.map((job) => {
              const statusConfig = STATUS_CONFIG[job.status || "pending"];
              const StatusIcon = statusConfig?.icon || Clock;
              const createdObjects = job.createdObjects as { type: string; count: number }[] | null;
              const validationSummary = job.validationSummary as { totalRows?: number; validRows?: number; invalidRows?: number } | null;

              return (
                <Card key={job.id} className="hover-elevate" data-testid={`card-import-job-${job.id}`}>
                  <CardContent className="py-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div className={`p-2 rounded-md ${statusConfig?.color || "bg-gray-500"}/10`}>
                          <FileSpreadsheet className="h-6 w-6 text-primary" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-medium">{job.fileName}</p>
                            <Badge variant="outline" className="text-xs">
                              {job.domain === "drug" ? "Drug" : "Materials"}
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {IMPORT_TYPE_LABELS[job.importType] || job.importType}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">
                            {formatSize(job.fileSize)} • {formatDate(job.createdAt)}
                          </p>
                          {validationSummary && (
                            <p className="text-sm text-muted-foreground">
                              {validationSummary.validRows?.toLocaleString()} of {validationSummary.totalRows?.toLocaleString()} rows imported
                            </p>
                          )}
                          {createdObjects && createdObjects.length > 0 && (
                            <div className="flex gap-2 mt-2">
                              {createdObjects.map((obj, i) => (
                                <Badge key={i} variant="outline">
                                  {obj.count.toLocaleString()} {obj.type}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                          <StatusIcon className={`h-4 w-4 ${
                            job.status === "succeeded" ? "text-green-500" :
                            job.status === "failed" ? "text-red-500" :
                            job.status?.includes("ing") ? "text-blue-500 animate-spin" :
                            "text-muted-foreground"
                          }`} />
                          <span className="text-sm capitalize">{job.status}</span>
                        </div>
                        <Button variant="ghost" size="sm" data-testid={`button-view-job-${job.id}`}>
                          <ExternalLink className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    {job.errorMessage && (
                      <div className="mt-3 p-2 bg-red-500/10 rounded-md">
                        <p className="text-sm text-red-600">{job.errorMessage}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
