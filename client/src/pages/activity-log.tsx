import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Activity, 
  ChevronLeft, 
  ChevronRight,
  Clock, 
  FileText, 
  FlaskConical, 
  LogIn, 
  MousePointer, 
  Navigation, 
  Server, 
  Target, 
  Upload, 
  XCircle,
  Beaker,
  Workflow
} from "lucide-react";
import { formatDistanceToNow, format } from "date-fns";

interface ActivityLog {
  id: string;
  userId: string;
  activityType: string;
  action: string;
  description: string | null;
  metadata: Record<string, any> | null;
  entityType: string | null;
  entityId: string | null;
  createdAt: string;
}

interface ActivityLogsResponse {
  logs: ActivityLog[];
  total: number;
  limit: number;
  offset: number;
}

interface ActivityStats {
  type: string;
  count: number;
}

const ACTIVITY_TYPES = [
  { value: "all", label: "All Activities" },
  { value: "user_action", label: "User Actions" },
  { value: "system_response", label: "System Responses" },
  { value: "navigation", label: "Navigation" },
  { value: "data_import", label: "Data Imports" },
  { value: "analysis_run", label: "Analysis Runs" },
  { value: "campaign_action", label: "Campaign Actions" },
  { value: "molecule_action", label: "Molecule Actions" },
  { value: "target_action", label: "Target Actions" },
  { value: "pipeline_action", label: "Pipeline Actions" },
  { value: "error", label: "Errors" },
  { value: "auth", label: "Authentication" },
];

function getActivityIcon(type: string) {
  switch (type) {
    case "user_action":
      return <MousePointer className="h-4 w-4" />;
    case "system_response":
      return <Server className="h-4 w-4" />;
    case "navigation":
      return <Navigation className="h-4 w-4" />;
    case "data_import":
      return <Upload className="h-4 w-4" />;
    case "analysis_run":
      return <FlaskConical className="h-4 w-4" />;
    case "campaign_action":
      return <Beaker className="h-4 w-4" />;
    case "molecule_action":
      return <FileText className="h-4 w-4" />;
    case "target_action":
      return <Target className="h-4 w-4" />;
    case "pipeline_action":
      return <Workflow className="h-4 w-4" />;
    case "error":
      return <XCircle className="h-4 w-4" />;
    case "auth":
      return <LogIn className="h-4 w-4" />;
    default:
      return <Activity className="h-4 w-4" />;
  }
}

function getActivityColor(type: string) {
  switch (type) {
    case "user_action":
      return "bg-blue-500/10 text-blue-500 border-blue-500/20";
    case "system_response":
      return "bg-green-500/10 text-green-500 border-green-500/20";
    case "navigation":
      return "bg-slate-500/10 text-slate-500 border-slate-500/20";
    case "data_import":
      return "bg-purple-500/10 text-purple-500 border-purple-500/20";
    case "analysis_run":
      return "bg-amber-500/10 text-amber-500 border-amber-500/20";
    case "campaign_action":
      return "bg-cyan-500/10 text-cyan-500 border-cyan-500/20";
    case "molecule_action":
      return "bg-pink-500/10 text-pink-500 border-pink-500/20";
    case "target_action":
      return "bg-orange-500/10 text-orange-500 border-orange-500/20";
    case "pipeline_action":
      return "bg-indigo-500/10 text-indigo-500 border-indigo-500/20";
    case "error":
      return "bg-red-500/10 text-red-500 border-red-500/20";
    case "auth":
      return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
    default:
      return "bg-muted text-muted-foreground";
  }
}

function ActivityLogItem({ log }: { log: ActivityLog }) {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div 
      className="flex gap-4 p-4 border-b last:border-b-0 hover-elevate cursor-pointer"
      onClick={() => setExpanded(!expanded)}
      data-testid={`activity-log-item-${log.id}`}
    >
      <div className={`flex items-center justify-center w-10 h-10 rounded-full shrink-0 ${getActivityColor(log.activityType)}`}>
        {getActivityIcon(log.activityType)}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="font-medium">{log.action}</span>
            <Badge variant="outline" className="text-xs">
              {log.activityType.replace(/_/g, " ")}
            </Badge>
          </div>
          <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
            <Clock className="h-3 w-3" />
            <span title={format(new Date(log.createdAt), "PPpp")}>
              {formatDistanceToNow(new Date(log.createdAt), { addSuffix: true })}
            </span>
          </div>
        </div>
        {log.description && (
          <p className="text-sm text-muted-foreground mt-1">{log.description}</p>
        )}
        {log.entityType && (
          <div className="flex items-center gap-2 mt-2">
            <Badge variant="secondary" className="text-xs">
              {log.entityType}
            </Badge>
            {log.entityId && (
              <span className="text-xs text-muted-foreground font-mono">{log.entityId}</span>
            )}
          </div>
        )}
        {expanded && log.metadata && Object.keys(log.metadata).length > 0 && (
          <div className="mt-3 p-3 bg-muted/50 rounded-md">
            <p className="text-xs font-medium mb-2">Metadata</p>
            <pre className="text-xs text-muted-foreground overflow-x-auto">
              {JSON.stringify(log.metadata, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

function ActivityLogSkeleton() {
  return (
    <div className="flex gap-4 p-4 border-b">
      <Skeleton className="w-10 h-10 rounded-full shrink-0" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-4 w-72" />
      </div>
    </div>
  );
}

export default function ActivityLogPage() {
  const [selectedType, setSelectedType] = useState("all");
  const [page, setPage] = useState(0);
  const limit = 20;
  
  const { data: logsData, isLoading: logsLoading } = useQuery<ActivityLogsResponse>({
    queryKey: ["/api/activity-logs", selectedType, page],
    queryFn: async () => {
      const params = new URLSearchParams({
        limit: String(limit),
        offset: String(page * limit),
      });
      if (selectedType !== "all") {
        params.set("type", selectedType);
      }
      const res = await fetch(`/api/activity-logs?${params}`);
      if (!res.ok) throw new Error("Failed to fetch logs");
      return res.json();
    },
  });

  const { data: statsData } = useQuery<ActivityStats[]>({
    queryKey: ["/api/activity-logs/stats"],
  });

  const logs = logsData?.logs || [];
  const total = logsData?.total || 0;
  const totalPages = Math.ceil(total / limit);

  return (
    <div className="container mx-auto p-6 space-y-6" data-testid="activity-log-page">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="h-6 w-6" />
            Activity Log
          </h1>
          <p className="text-muted-foreground">
            Track all your actions and system responses in chronological order
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {statsData?.map((stat) => (
          <Card 
            key={stat.type} 
            className="cursor-pointer hover-elevate"
            onClick={() => setSelectedType(stat.type)}
            data-testid={`stat-card-${stat.type}`}
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <div className={`p-2 rounded-md ${getActivityColor(stat.type)}`}>
                  {getActivityIcon(stat.type)}
                </div>
                <div>
                  <p className="text-lg font-bold">{stat.count}</p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {stat.type.replace(/_/g, " ")}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4 space-y-0 pb-4">
          <CardTitle className="text-lg">Recent Activity</CardTitle>
          <Select value={selectedType} onValueChange={(v) => { setSelectedType(v); setPage(0); }}>
            <SelectTrigger className="w-48" data-testid="filter-activity-type">
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              {ACTIVITY_TYPES.map((type) => (
                <SelectItem key={type.value} value={type.value}>
                  {type.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardHeader>
        <CardContent className="p-0">
          {logsLoading ? (
            <div>
              {Array.from({ length: 5 }).map((_, i) => (
                <ActivityLogSkeleton key={i} />
              ))}
            </div>
          ) : logs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Activity className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No activity logs found</p>
              <p className="text-sm text-muted-foreground">
                Your actions will appear here as you use the platform
              </p>
            </div>
          ) : (
            <div>
              {logs.map((log) => (
                <ActivityLogItem key={log.id} log={log} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {page * limit + 1} - {Math.min((page + 1) * limit, total)} of {total} entries
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
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
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              data-testid="button-next-page"
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
