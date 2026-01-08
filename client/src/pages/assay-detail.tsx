import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams, useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { 
  ArrowLeft, 
  FlaskConical, 
  Upload, 
  FileSpreadsheet, 
  Activity,
  CheckCircle,
  XCircle,
  AlertCircle,
  Download,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Assay, AssayResult, Molecule } from "@shared/schema";
import { useState, useCallback } from "react";

type AssayTypeValue = "binding" | "functional" | "in_vivo" | "pk" | "admet" | "other";
type AssayWithResultsCount = Assay & { resultsCount: number };
type AssayResultWithMolecule = AssayResult & { molecule: Molecule | null };

const typeLabels: Record<AssayTypeValue, { label: string; color: string }> = {
  binding: { label: "Binding", color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200" },
  functional: { label: "Functional", color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" },
  in_vivo: { label: "In Vivo", color: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200" },
  pk: { label: "PK", color: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200" },
  admet: { label: "ADMET", color: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200" },
  other: { label: "Other", color: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200" },
};

const outcomeConfig: Record<string, { icon: typeof CheckCircle; color: string }> = {
  active: { icon: CheckCircle, color: "text-green-600" },
  inactive: { icon: XCircle, color: "text-muted-foreground" },
  toxic: { icon: AlertCircle, color: "text-red-600" },
  inconclusive: { icon: AlertCircle, color: "text-yellow-600" },
  no_effect: { icon: XCircle, color: "text-muted-foreground" },
};

export default function AssayDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [csvData, setCsvData] = useState("");
  const [parsedRows, setParsedRows] = useState<any[]>([]);
  const [parseError, setParseError] = useState<string | null>(null);
  const [campaignId, setCampaignId] = useState("");

  const { data: assay, isLoading: assayLoading } = useQuery<AssayWithResultsCount>({
    queryKey: ["/api/assays", id, "details"],
    queryFn: async () => {
      const res = await fetch(`/api/assays/${id}/details`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch assay");
      return res.json();
    },
  });

  const { data: results, isLoading: resultsLoading } = useQuery<AssayResultWithMolecule[]>({
    queryKey: ["/api/assays", id, "results"],
    queryFn: async () => {
      const res = await fetch(`/api/assays/${id}/results`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch results");
      return res.json();
    },
  });

  const uploadMutation = useMutation({
    mutationFn: async (rows: any[]) => {
      const res = await apiRequest("POST", `/api/assays/${id}/upload`, { 
        rows, 
        campaignId: campaignId || undefined 
      });
      return res.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/assays", id, "details"] });
      queryClient.invalidateQueries({ queryKey: ["/api/assays", id, "results"] });
      toast({ 
        title: "Upload Complete",
        description: `Imported ${data.imported} results. ${data.moleculesCreated} new molecules created. ${data.errors?.length || 0} errors.`
      });
      setCsvData("");
      setParsedRows([]);
    },
    onError: () => {
      toast({ title: "Upload failed", variant: "destructive" });
    },
  });

  const parseCSV = useCallback((text: string) => {
    setParseError(null);
    const lines = text.trim().split("\n");
    if (lines.length < 2) {
      setParseError("CSV must have a header row and at least one data row");
      setParsedRows([]);
      return;
    }

    const headers = lines[0].split(",").map(h => h.trim().toLowerCase().replace(/\s+/g, "_"));
    const requiredFields = ["smiles", "value"];
    const hasRequired = requiredFields.every(f => headers.includes(f) || headers.includes("molecule_id"));

    if (!hasRequired && !headers.includes("molecule_id")) {
      setParseError("CSV must have either 'smiles' and 'value' columns, or 'molecule_id' and 'value' columns");
      setParsedRows([]);
      return;
    }

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const values = lines[i].split(",").map(v => v.trim());
      const row: Record<string, string> = {};
      headers.forEach((h, j) => {
        row[h] = values[j] || "";
      });
      rows.push(row);
    }

    setParsedRows(rows);
  }, []);

  const handleCSVChange = (text: string) => {
    setCsvData(text);
    if (text.trim()) {
      parseCSV(text);
    } else {
      setParsedRows([]);
      setParseError(null);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      setCsvData(text);
      parseCSV(text);
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const handleUpload = () => {
    if (parsedRows.length === 0) {
      toast({ title: "No data to upload", variant: "destructive" });
      return;
    }
    uploadMutation.mutate(parsedRows);
  };

  const downloadTemplate = () => {
    const template = "smiles,value,outcome_label,concentration,replicate_id\nCCO,15.5,active,10,1\nCC(=O)O,45.2,inactive,10,1";
    const blob = new Blob([template], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "assay_upload_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (assayLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center gap-4">
          <Skeleton className="h-9 w-9" />
          <div>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-32 mt-1" />
          </div>
        </div>
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!assay) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FlaskConical className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Assay not found</h3>
            <Button onClick={() => navigate("/assays")} className="mt-4" data-testid="button-back-to-assays">
              Back to Assays
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4 flex-wrap">
        <Button variant="ghost" size="icon" onClick={() => navigate("/assays")} data-testid="button-back">
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-semibold truncate" data-testid="text-assay-name">{assay.name}</h1>
            {assay.type && (
              <Badge variant="secondary" className={typeLabels[assay.type as AssayTypeValue]?.color}>
                {typeLabels[assay.type as AssayTypeValue]?.label}
              </Badge>
            )}
            <Badge variant="outline">{assay.readoutType}</Badge>
          </div>
          <p className="text-muted-foreground flex items-center gap-4 mt-1 flex-wrap">
            <span className="flex items-center gap-1">
              <Activity className="h-4 w-4" />
              {assay.units}
            </span>
            <span>{assay.resultsCount} results</span>
            {assay.createdAt && (
              <span>Created {formatDistanceToNow(new Date(assay.createdAt), { addSuffix: true })}</span>
            )}
          </p>
        </div>
      </div>

      {assay.description && (
        <Card>
          <CardContent className="py-4">
            <p className="text-sm text-muted-foreground">{assay.description}</p>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="results" className="space-y-4">
        <TabsList>
          <TabsTrigger value="results" data-testid="tab-results">
            Results ({assay.resultsCount})
          </TabsTrigger>
          <TabsTrigger value="upload" data-testid="tab-upload">
            Upload Data
          </TabsTrigger>
        </TabsList>

        <TabsContent value="results" className="space-y-4">
          {resultsLoading ? (
            <Card>
              <CardContent className="py-8">
                <Skeleton className="h-64 w-full" />
              </CardContent>
            </Card>
          ) : !results || results.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <FileSpreadsheet className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium">No results yet</h3>
                <p className="text-muted-foreground text-center">
                  Upload experimental data to see results here
                </p>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Experimental Results</CardTitle>
                <CardDescription>
                  {results.length} data points from wet-lab experiments
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Molecule</TableHead>
                        <TableHead>SMILES</TableHead>
                        <TableHead className="text-right">Value</TableHead>
                        <TableHead>Units</TableHead>
                        <TableHead>Outcome</TableHead>
                        <TableHead>Conc.</TableHead>
                        <TableHead>Replicate</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.map((result) => {
                        const OutcomeIcon = result.outcomeLabel ? outcomeConfig[result.outcomeLabel]?.icon : null;
                        const outcomeColor = result.outcomeLabel ? outcomeConfig[result.outcomeLabel]?.color : "";
                        
                        return (
                          <TableRow key={result.id} data-testid={`row-result-${result.id}`}>
                            <TableCell className="font-medium">
                              {result.molecule?.name || "Unnamed"}
                            </TableCell>
                            <TableCell className="font-mono text-xs max-w-[200px] truncate">
                              {result.molecule?.smiles || "-"}
                            </TableCell>
                            <TableCell className="text-right font-medium">
                              {result.value?.toFixed(2) ?? "-"}
                            </TableCell>
                            <TableCell>{result.units || assay.units}</TableCell>
                            <TableCell>
                              {result.outcomeLabel ? (
                                <div className={`flex items-center gap-1 ${outcomeColor}`}>
                                  {OutcomeIcon && <OutcomeIcon className="h-4 w-4" />}
                                  <span className="capitalize">{result.outcomeLabel}</span>
                                </div>
                              ) : "-"}
                            </TableCell>
                            <TableCell>
                              {result.concentration ? `${result.concentration}` : "-"}
                            </TableCell>
                            <TableCell>{result.replicateId || "-"}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Assay Data
              </CardTitle>
              <CardDescription>
                Import experimental results from CSV. Required columns: smiles (or molecule_id), value.
                Optional: outcome_label, concentration, replicate_id.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-sm font-medium mb-1">Campaign ID (optional)</label>
                  <Input
                    placeholder="Link results to a campaign..."
                    value={campaignId}
                    onChange={(e) => setCampaignId(e.target.value)}
                    data-testid="input-campaign-id"
                  />
                </div>
                <div className="flex items-end gap-2">
                  <div>
                    <label htmlFor="file-upload" className="block text-sm font-medium mb-1">Upload CSV</label>
                    <Input
                      id="file-upload"
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      data-testid="input-file-upload"
                    />
                  </div>
                  <Button variant="outline" onClick={downloadTemplate} data-testid="button-download-template">
                    <Download className="mr-2 h-4 w-4" />
                    Template
                  </Button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Or paste CSV data</label>
                <Textarea
                  placeholder="smiles,value,outcome_label&#10;CCO,15.5,active&#10;CC(=O)O,45.2,inactive"
                  value={csvData}
                  onChange={(e) => handleCSVChange(e.target.value)}
                  className="font-mono text-sm min-h-[150px]"
                  data-testid="textarea-csv-data"
                />
              </div>

              {parseError && (
                <div className="p-3 bg-destructive/10 text-destructive rounded-md text-sm" data-testid="text-parse-error">
                  {parseError}
                </div>
              )}

              {parsedRows.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Preview ({parsedRows.length} rows)</h4>
                  <div className="border rounded-md overflow-x-auto max-h-[300px]">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          {Object.keys(parsedRows[0]).map((key) => (
                            <TableHead key={key} className="text-xs capitalize">
                              {key.replace(/_/g, " ")}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {parsedRows.slice(0, 10).map((row, i) => (
                          <TableRow key={i}>
                            {Object.values(row).map((val, j) => (
                              <TableCell key={j} className="text-xs font-mono">
                                {String(val).substring(0, 30)}
                                {String(val).length > 30 ? "..." : ""}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                        {parsedRows.length > 10 && (
                          <TableRow>
                            <TableCell 
                              colSpan={Object.keys(parsedRows[0]).length}
                              className="text-center text-muted-foreground text-xs"
                            >
                              ... and {parsedRows.length - 10} more rows
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}

              <div className="flex justify-end gap-2 pt-4">
                <Button
                  variant="outline"
                  onClick={() => {
                    setCsvData("");
                    setParsedRows([]);
                    setParseError(null);
                  }}
                  disabled={!csvData}
                  data-testid="button-clear"
                >
                  Clear
                </Button>
                <Button
                  onClick={handleUpload}
                  disabled={parsedRows.length === 0 || uploadMutation.isPending}
                  data-testid="button-upload"
                >
                  {uploadMutation.isPending ? "Uploading..." : `Upload ${parsedRows.length} Results`}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
