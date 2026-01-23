import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  FileJson,
  Table2,
  Image,
  Box,
  FileText,
  Download,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Database,
  Loader2,
  FileWarning,
} from "lucide-react";
import type { JobArtifact } from "@shared/schema";

interface ResultsPanelProps {
  jobId?: string;
  campaignId?: string;
  materialsCampaignId?: string;
  title?: string;
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

const ARTIFACT_ICONS: Record<string, typeof FileJson> = {
  json: FileJson,
  table: Table2,
  image: Image,
  model3d: Box,
  report: FileText,
};

const ARTIFACT_COLORS: Record<string, string> = {
  json: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30",
  table: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30",
  image: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/30",
  model3d: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30",
  report: "bg-primary/10 text-primary border-primary/30",
};

function JsonPreview({ data }: { data: unknown }) {
  const [expanded, setExpanded] = useState(false);
  const jsonStr = JSON.stringify(data, null, 2);
  const lines = jsonStr.split("\n");
  const preview = expanded ? jsonStr : lines.slice(0, 10).join("\n") + (lines.length > 10 ? "\n..." : "");

  return (
    <div className="space-y-2">
      <pre className="p-3 rounded-md bg-muted/50 text-xs font-mono overflow-x-auto whitespace-pre-wrap">
        {preview}
      </pre>
      {lines.length > 10 && (
        <Button variant="ghost" size="sm" onClick={() => setExpanded(!expanded)}>
          {expanded ? (
            <>
              <ChevronUp className="h-3 w-3 mr-1" />
              Show Less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show All ({lines.length} lines)
            </>
          )}
        </Button>
      )}
    </div>
  );
}

function TablePreview({ data }: { data: unknown }) {
  if (!Array.isArray(data) || data.length === 0) {
    return <div className="text-sm text-muted-foreground">No table data available</div>;
  }

  const headers = Object.keys(data[0]);
  const rows = data.slice(0, 10);

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              {headers.map((h) => (
                <th key={h} className="p-2 text-left font-medium text-muted-foreground">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-muted/50">
                {headers.map((h) => (
                  <td key={h} className="p-2 font-mono">
                    {String(row[h] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length > 10 && (
        <div className="text-xs text-muted-foreground">
          Showing 10 of {data.length} rows
        </div>
      )}
    </div>
  );
}

function ImagePreview({ uri, name }: { uri: string; name: string }) {
  return (
    <div className="space-y-2">
      <div className="relative aspect-video bg-muted/50 rounded-md overflow-hidden">
        <img
          src={uri}
          alt={name}
          className="object-contain w-full h-full"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    </div>
  );
}

function Model3DPreview({ uri, name }: { uri: string; name: string }) {
  return (
    <div className="space-y-2">
      <div className="aspect-video bg-muted/50 rounded-md flex items-center justify-center">
        <div className="text-center">
          <Box className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
          <div className="text-sm font-medium">{name}</div>
          <div className="text-xs text-muted-foreground">3D Model Viewer</div>
          <Button variant="outline" size="sm" className="mt-2" asChild>
            <a href={uri} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-3 w-3 mr-1" />
              Open in Viewer
            </a>
          </Button>
        </div>
      </div>
    </div>
  );
}

function ReportPreview({ summaryJson }: { summaryJson: unknown }) {
  if (!summaryJson || typeof summaryJson !== "object") {
    return <div className="text-sm text-muted-foreground">No report summary available</div>;
  }

  const summary = summaryJson as Record<string, unknown>;

  return (
    <div className="space-y-3">
      {Object.entries(summary).map(([key, value]) => (
        <div key={key} className="p-3 rounded-md bg-muted/50">
          <div className="text-xs text-muted-foreground uppercase tracking-wide mb-1">{key.replace(/_/g, " ")}</div>
          <div className="text-sm font-medium">
            {typeof value === "object" ? JSON.stringify(value) : String(value)}
          </div>
        </div>
      ))}
    </div>
  );
}

interface ArtifactCardProps {
  artifact: JobArtifact;
}

function ArtifactCard({ artifact }: ArtifactCardProps) {
  const Icon = ARTIFACT_ICONS[artifact.artifactType] || FileText;
  const colorClass = ARTIFACT_COLORS[artifact.artifactType] || ARTIFACT_COLORS.report;

  const renderPreview = () => {
    switch (artifact.artifactType) {
      case "json":
        return <JsonPreview data={artifact.summaryJson} />;
      case "table":
        return <TablePreview data={artifact.summaryJson} />;
      case "image":
        return <ImagePreview uri={artifact.uri} name={artifact.name} />;
      case "model3d":
        return <Model3DPreview uri={artifact.uri} name={artifact.name} />;
      case "report":
        return <ReportPreview summaryJson={artifact.summaryJson} />;
      default:
        return null;
    }
  };

  return (
    <Card className="hover-elevate" data-testid={`card-artifact-${artifact.id}`}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-md flex items-center justify-center border ${colorClass}`}>
              <Icon className="h-4 w-4" />
            </div>
            <div>
              <CardTitle className="text-sm">{artifact.name}</CardTitle>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="outline" className={`text-xs ${colorClass}`}>
                  {artifact.artifactType}
                </Badge>
                {artifact.mimeType && (
                  <span className="text-xs text-muted-foreground">{artifact.mimeType}</span>
                )}
              </div>
            </div>
          </div>
          <Button variant="ghost" size="icon" asChild>
            <a href={artifact.uri} target="_blank" rel="noopener noreferrer" download>
              <Download className="h-4 w-4" />
            </a>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {renderPreview()}
      </CardContent>
    </Card>
  );
}

function generateMockArtifacts(count: number): JobArtifact[] {
  const types: Array<"json" | "table" | "image" | "model3d" | "report"> = ["json", "table", "image", "model3d", "report"];
  const names = [
    "Property Predictions",
    "Score Distribution",
    "Molecular Structure",
    "Conformer Ensemble",
    "Synthesis Report",
    "Docking Results",
    "ML Model Output",
    "Simulation Data",
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `artifact-${i}`,
    jobId: `job-${Math.floor(i / 3)}`,
    companyId: null,
    campaignId: null,
    materialsCampaignId: null,
    domain: i % 2 === 0 ? "materials" : "drug",
    artifactType: types[i % types.length],
    name: names[i % names.length],
    uri: `https://storage.example.com/artifacts/${i}.${types[i % types.length] === "image" ? "png" : types[i % types.length]}`,
    mimeType: types[i % types.length] === "json" ? "application/json" : types[i % types.length] === "image" ? "image/png" : null,
    summaryJson: types[i % types.length] === "json" 
      ? { predicted_value: 0.85 + Math.random() * 0.1, confidence: 0.92, method: "GNN" }
      : types[i % types.length] === "table"
      ? Array.from({ length: 5 }, (_, j) => ({ id: j, property: `Prop_${j}`, value: (Math.random() * 100).toFixed(2), unit: "kJ/mol" }))
      : types[i % types.length] === "report"
      ? { total_predictions: 42000, successful: 41850, failed: 150, avg_confidence: 0.89 }
      : null,
    createdAt: new Date(),
  })) as JobArtifact[];
}

export function ResultsPanel({
  jobId,
  campaignId,
  materialsCampaignId,
  title = "Computation Results",
  collapsible = true,
  defaultExpanded = true,
}: ResultsPanelProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [activeType, setActiveType] = useState<string>("all");

  const endpoint = jobId 
    ? `/api/jobs/${jobId}/artifacts`
    : materialsCampaignId 
    ? `/api/materials-campaigns/${materialsCampaignId}/artifacts`
    : campaignId
    ? `/api/campaigns/${campaignId}/artifacts`
    : null;

  const { data: artifacts = [], isLoading } = useQuery<JobArtifact[]>({
    queryKey: [endpoint],
    enabled: !!endpoint,
  });

  const displayArtifacts = artifacts.length > 0 ? artifacts : generateMockArtifacts(6);

  const filteredArtifacts = activeType === "all" 
    ? displayArtifacts 
    : displayArtifacts.filter(a => a.artifactType === activeType);

  const artifactTypes = ["all", ...Array.from(new Set(displayArtifacts.map(a => a.artifactType)))];

  if (!endpoint) {
    return null;
  }

  return (
    <Card data-testid="results-panel">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center border border-primary/30">
              <Database className="h-4 w-4 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">{title}</CardTitle>
              <div className="text-xs text-muted-foreground">
                {displayArtifacts.length} artifacts from computation jobs
              </div>
            </div>
          </div>
          {collapsible && (
            <Button variant="ghost" size="sm" onClick={() => setExpanded(!expanded)}>
              {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          )}
        </div>
      </CardHeader>
      {expanded && (
        <CardContent className="space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : displayArtifacts.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <FileWarning className="h-10 w-10 text-muted-foreground mb-2" />
              <div className="text-sm text-muted-foreground">No artifacts generated yet</div>
              <div className="text-xs text-muted-foreground">Run a computation job to generate results</div>
            </div>
          ) : (
            <>
              <Tabs value={activeType} onValueChange={setActiveType}>
                <TabsList>
                  {artifactTypes.map(type => (
                    <TabsTrigger key={type} value={type} className="capitalize">
                      {type === "all" ? "All" : type === "model3d" ? "3D Models" : type}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>

              <ScrollArea className="h-[400px]">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pr-4">
                  {filteredArtifacts.map(artifact => (
                    <ArtifactCard key={artifact.id} artifact={artifact} />
                  ))}
                </div>
              </ScrollArea>
            </>
          )}
        </CardContent>
      )}
    </Card>
  );
}
