import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Layers, 
  Plus, 
  Search, 
  GitBranch, 
  Zap, 
  Sparkles, 
  RefreshCw,
  Database,
  Upload,
  ChevronLeft,
  ChevronRight,
  ExternalLink
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
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

interface MaterialVariant {
  id: string;
  materialId: string;
  variantName: string;
  variantType: string;
  parameters: Record<string, any>;
  predictedProperties: Record<string, any> | null;
  status: string;
  createdAt: string;
}

interface VariantsResponse {
  variants: MaterialVariant[];
  total: number;
}

export default function MaterialVariantsPage() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const { data, isLoading, error } = useQuery<VariantsResponse>({
    queryKey: ["/api/material-variants", { limit: pageSize, offset: page * pageSize }],
  });

  const syncMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/external-sync/sync/variants", {
        source: "digitalocean",
        tableName: "variants_formulations_massive"
      });
      return response.json();
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["/api/material-variants"] });
      toast({
        title: "Sync Complete",
        description: `Synced ${result.recordsInserted} new variants from DigitalOcean database.`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Sync Failed",
        description: error.message || "Failed to sync variants from DigitalOcean",
        variant: "destructive",
      });
    },
  });

  const variants = data?.variants || [];
  const total = data?.total || 0;
  const totalPages = Math.ceil(total / pageSize);

  const filteredVariants = searchQuery 
    ? variants.filter(v => 
        v.variantName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        v.variantType.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : variants;

  const evaluatedCount = variants.filter(v => v.predictedProperties && Object.keys(v.predictedProperties).length > 0).length;

  const stats = [
    { label: "Total Variants", value: total, icon: Layers, color: "from-violet-500 to-purple-500", bgColor: "bg-violet-500" },
    { label: "Active Explorations", value: variants.filter(v => v.status === "exploring" || v.status === "active").length, icon: GitBranch, color: "from-blue-500 to-cyan-500", bgColor: "bg-blue-500" },
    { label: "Evaluated", value: evaluatedCount, icon: Sparkles, color: "from-emerald-500 to-teal-500", bgColor: "bg-emerald-500" },
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
      case "exploring":
        return <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/30">Active</Badge>;
      case "evaluated":
        return <Badge className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30">Evaluated</Badge>;
      case "pending":
        return <Badge className="bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/30">Pending</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-violet-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0xMCAzMGgyME0zMCAxMHYyMCIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10 flex items-center justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Layers className="h-7 w-7" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold">Material Variants</h1>
                    <p className="text-violet-100">Explore structural variations and property predictions</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-violet-100">
                  <Zap className="h-4 w-4" />
                  <span>AI-guided structural exploration and optimization</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Button 
                  className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0"
                  onClick={() => syncMutation.mutate()}
                  disabled={syncMutation.isPending}
                  data-testid="button-sync-variants"
                >
                  {syncMutation.isPending ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Database className="h-4 w-4" />
                  )}
                  {syncMutation.isPending ? "Syncing..." : "Sync from DigitalOcean"}
                </Button>
                <Button 
                  className="gap-2 bg-white/20 backdrop-blur hover:bg-white/30 text-white border-0"
                  data-testid="button-create-variant"
                >
                  <Plus className="h-4 w-4" />
                  Create Variant
                </Button>
              </div>
            </div>
          </header>

          <div className="relative max-w-xl">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input 
              placeholder="Search variants by name or type..." 
              className="pl-12 h-12 text-base"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              data-testid="input-search-variants"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {stats.map((stat) => (
              <Card key={stat.label} className="border-0 shadow-lg overflow-hidden">
                <CardContent className="p-0">
                  <div className={`h-2 bg-gradient-to-r ${stat.color}`} />
                  <div className="p-6">
                    <div className="flex items-center gap-4">
                      <div className={`w-14 h-14 rounded-xl ${stat.bgColor}/10 flex items-center justify-center`}>
                        <stat.icon className={`h-7 w-7 ${stat.bgColor.replace('bg-', 'text-')}`} />
                      </div>
                      <div>
                        {isLoading ? (
                          <Skeleton className="h-9 w-20" />
                        ) : (
                          <p className="text-3xl font-bold">{stat.value.toLocaleString()}</p>
                        )}
                        <p className="text-sm text-muted-foreground">{stat.label}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {isLoading ? (
            <Card className="shadow-lg">
              <CardContent className="p-6">
                <div className="space-y-4">
                  {[1, 2, 3, 4, 5].map(i => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : error ? (
            <Card className="shadow-lg">
              <CardContent className="p-12 text-center">
                <div className="w-20 h-20 rounded-3xl bg-destructive/10 flex items-center justify-center mx-auto mb-5">
                  <Database className="h-9 w-9 text-destructive" />
                </div>
                <p className="font-semibold text-lg mb-2">Failed to load variants</p>
                <p className="text-muted-foreground mb-6">
                  {(error as Error).message || "An error occurred while fetching variants"}
                </p>
                <Button 
                  variant="outline" 
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/material-variants"] })}
                  data-testid="button-retry-load"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry
                </Button>
              </CardContent>
            </Card>
          ) : filteredVariants.length === 0 ? (
            <Card className="shadow-lg">
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-500/10 to-purple-500/10 flex items-center justify-center mx-auto mb-5">
                    <Layers className="h-9 w-9 text-violet-500" />
                  </div>
                  <p className="font-semibold text-lg mb-2">No variants yet</p>
                  <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                    Sync variants from your DigitalOcean database or create new variants from your materials
                  </p>
                  <div className="flex items-center justify-center gap-3">
                    <Button 
                      variant="outline"
                      className="gap-2"
                      onClick={() => syncMutation.mutate()}
                      disabled={syncMutation.isPending}
                      data-testid="button-sync-empty"
                    >
                      <Database className="h-4 w-4" />
                      Sync from DigitalOcean
                    </Button>
                    <Button className="gap-2 bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600">
                      <Plus className="h-4 w-4" />
                      Create Variant
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="shadow-lg">
              <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4">
                <CardTitle className="text-lg">Material Variants</CardTitle>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>Showing {page * pageSize + 1}-{Math.min((page + 1) * pageSize, total)} of {total.toLocaleString()}</span>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[500px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Variant Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Parameters</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="w-[100px]">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredVariants.map((variant) => (
                        <TableRow key={variant.id} data-testid={`row-variant-${variant.id}`}>
                          <TableCell className="font-medium">{variant.variantName}</TableCell>
                          <TableCell>
                            <Badge variant="secondary">{variant.variantType}</Badge>
                          </TableCell>
                          <TableCell className="max-w-[200px] truncate text-muted-foreground">
                            {variant.parameters ? Object.entries(variant.parameters).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(", ") : "-"}
                          </TableCell>
                          <TableCell>{getStatusBadge(variant.status)}</TableCell>
                          <TableCell className="text-muted-foreground">
                            {new Date(variant.createdAt).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <Button 
                              variant="ghost" 
                              size="icon"
                              data-testid={`button-view-variant-${variant.id}`}
                            >
                              <ExternalLink className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between px-6 py-4 border-t">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.max(0, p - 1))}
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
                      onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1}
                      data-testid="button-next-page"
                    >
                      Next
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
