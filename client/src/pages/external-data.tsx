import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Database, RefreshCw, CheckCircle2, XCircle, Loader2, Cloud, Server, FlaskConical, Hexagon, Layers } from "lucide-react";

interface ConnectionResult {
  success: boolean;
  message: string;
  tables?: string[];
  source: string;
}

interface SyncResult {
  success: boolean;
  table: string;
  recordsProcessed: number;
  recordsInserted: number;
  recordsSkipped: number;
  errors: string[];
}

interface PreviewResult {
  success: boolean;
  data?: Record<string, any>[];
  columns?: string[];
  error?: string;
  source: string;
}

export default function ExternalDataPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("connections");
  const [previewTable, setPreviewTable] = useState<{ name: string; source: 'supabase' | 'digitalocean' } | null>(null);

  const { data: connections, isLoading: connectionsLoading, refetch: refetchConnections } = useQuery<{
    supabase: ConnectionResult;
    digitalocean: ConnectionResult;
  }>({
    queryKey: ["/api/external-sync/test-all-connections"],
    staleTime: 60000,
  });

  const { data: preview, isLoading: previewLoading } = useQuery<PreviewResult>({
    queryKey: ["/api/external-sync/preview", previewTable?.name, previewTable?.source],
    enabled: !!previewTable,
    queryFn: async () => {
      const response = await fetch(`/api/external-sync/preview/${previewTable!.name}?source=${previewTable!.source}&limit=10`);
      return response.json();
    },
  });

  const syncSmilesMutation = useMutation({
    mutationFn: async (): Promise<SyncResult> => {
      const res = await apiRequest("POST", "/api/external-sync/sync/smiles", { tableName: "SMILES" });
      return res.json();
    },
    onSuccess: (data: SyncResult) => {
      toast({
        title: "SMILES Sync Complete",
        description: `Processed ${data.recordsProcessed} records. Inserted: ${data.recordsInserted}, Skipped: ${data.recordsSkipped}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/molecules"] });
    },
    onError: (error: any) => {
      toast({
        title: "Sync Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const syncMaterialPropertiesMutation = useMutation({
    mutationFn: async (): Promise<SyncResult> => {
      const res = await apiRequest("POST", "/api/external-sync/sync/material-properties", { tableName: "Material_Properties" });
      return res.json();
    },
    onSuccess: (data: SyncResult) => {
      toast({
        title: "Material Properties Sync Complete",
        description: `Processed ${data.recordsProcessed} records. Inserted: ${data.recordsInserted}, Skipped: ${data.recordsSkipped}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/materials"] });
    },
    onError: (error: any) => {
      toast({
        title: "Sync Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const syncVariantsMutation = useMutation({
    mutationFn: async (): Promise<SyncResult> => {
      const res = await apiRequest("POST", "/api/external-sync/sync/variants", { tableName: "Variants_Formulations", source: "digitalocean" });
      return res.json();
    },
    onSuccess: (data: SyncResult) => {
      toast({
        title: "Variants Sync Complete",
        description: `Processed ${data.recordsProcessed} records. Inserted: ${data.recordsInserted}, Skipped: ${data.recordsSkipped}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/material-variants"] });
    },
    onError: (error: any) => {
      toast({
        title: "Sync Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const syncAllMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/external-sync/sync/all");
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Full Sync Complete",
        description: "All tables have been synchronized",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/molecules"] });
      queryClient.invalidateQueries({ queryKey: ["/api/materials"] });
      queryClient.invalidateQueries({ queryKey: ["/api/material-variants"] });
    },
    onError: (error: any) => {
      toast({
        title: "Sync Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const ConnectionCard = ({ source, data, icon: Icon }: { source: string; data?: ConnectionResult; icon: any }) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
        <div className="flex items-center gap-2">
          <Icon className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-sm font-medium capitalize">{source}</CardTitle>
        </div>
        {data?.success ? (
          <Badge variant="default" className="bg-green-600">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Connected
          </Badge>
        ) : (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Disconnected
          </Badge>
        )}
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground">{data?.message}</p>
        {data?.tables && data.tables.length > 0 && (
          <div className="mt-2">
            <p className="text-xs font-medium mb-1">Available Tables:</p>
            <div className="flex flex-wrap gap-1">
              {data.tables.slice(0, 10).map((table) => (
                <Badge
                  key={table}
                  variant="outline"
                  className="text-xs cursor-pointer hover-elevate"
                  onClick={() => setPreviewTable({ name: table, source: source as 'supabase' | 'digitalocean' })}
                  data-testid={`badge-table-${table}`}
                >
                  {table}
                </Badge>
              ))}
              {data.tables.length > 10 && (
                <Badge variant="outline" className="text-xs">
                  +{data.tables.length - 10} more
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );

  const SyncCard = ({ title, description, icon: Icon, onSync, isPending, source }: {
    title: string;
    description: string;
    icon: any;
    onSync: () => void;
    isPending: boolean;
    source: string;
  }) => (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Icon className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">{title}</CardTitle>
          </div>
          <Badge variant="outline" className="text-xs">{source}</Badge>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <Button
          onClick={onSync}
          disabled={isPending}
          className="w-full"
          data-testid={`button-sync-${title.toLowerCase().replace(/\s/g, '-')}`}
        >
          {isPending ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Syncing...
            </>
          ) : (
            <>
              <RefreshCw className="h-4 w-4 mr-2" />
              Sync Now
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Database className="h-6 w-6" />
            External Data Sources
          </h1>
          <p className="text-muted-foreground">
            Connect and sync data from Supabase and DigitalOcean databases
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => refetchConnections()}
          disabled={connectionsLoading}
          data-testid="button-refresh-connections"
        >
          {connectionsLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="h-4 w-4" />
          )}
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="connections" data-testid="tab-connections">Connections</TabsTrigger>
          <TabsTrigger value="sync" data-testid="tab-sync">Sync Data</TabsTrigger>
          <TabsTrigger value="preview" data-testid="tab-preview">Preview</TabsTrigger>
        </TabsList>

        <TabsContent value="connections" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <ConnectionCard
              source="supabase"
              data={connections?.supabase}
              icon={Cloud}
            />
            <ConnectionCard
              source="digitalocean"
              data={connections?.digitalocean}
              icon={Server}
            />
          </div>
        </TabsContent>

        <TabsContent value="sync" className="space-y-4">
          <Card className="mb-4">
            <CardHeader>
              <CardTitle>Sync All Tables</CardTitle>
              <CardDescription>
                Sync SMILES and Material Properties from Supabase, and Variants from DigitalOcean
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                onClick={() => syncAllMutation.mutate()}
                disabled={syncAllMutation.isPending}
                size="lg"
                data-testid="button-sync-all"
              >
                {syncAllMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Syncing All...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Sync All Tables
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-3">
            <SyncCard
              title="SMILES"
              description="Drug names, disease conditions, ChEMBL IDs, and therapeutic classes"
              icon={FlaskConical}
              onSync={() => syncSmilesMutation.mutate()}
              isPending={syncSmilesMutation.isPending}
              source="Supabase"
            />
            <SyncCard
              title="Material Properties"
              description="Material IDs, property names, values, units, and temperature data"
              icon={Hexagon}
              onSync={() => syncMaterialPropertiesMutation.mutate()}
              isPending={syncMaterialPropertiesMutation.isPending}
              source="Supabase"
            />
            <SyncCard
              title="Variants & Formulations"
              description="Material variants with formulation parameters and processing methods"
              icon={Layers}
              onSync={() => syncVariantsMutation.mutate()}
              isPending={syncVariantsMutation.isPending}
              source="DigitalOcean"
            />
          </div>
        </TabsContent>

        <TabsContent value="preview" className="space-y-4">
          {!previewTable ? (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                Click on a table name in the Connections tab to preview its data
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="flex items-center gap-2">
                    Preview: {previewTable.name}
                    <Badge variant="outline">{previewTable.source}</Badge>
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setPreviewTable(null)}
                    data-testid="button-close-preview"
                  >
                    Close
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {previewLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin" />
                  </div>
                ) : preview?.success && preview.data ? (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          {preview.columns?.map((col) => (
                            <TableHead key={col} className="whitespace-nowrap">
                              {col}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {preview.data.map((row, i) => (
                          <TableRow key={i}>
                            {preview.columns?.map((col) => (
                              <TableCell key={col} className="max-w-xs truncate">
                                {typeof row[col] === 'object' ? JSON.stringify(row[col]) : String(row[col] ?? '')}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <div className="py-8 text-center text-destructive">
                    {preview?.error || "Failed to load preview"}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
