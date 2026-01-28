import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
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
  Database, 
  Search, 
  Download, 
  Filter, 
  ChevronLeft, 
  ChevronRight,
  FlaskConical,
  Plus,
  Check,
  Loader2,
  ArrowRight,
  Brain
} from "lucide-react";
import type { Project, CuratedLibrary } from "@shared/schema";

interface SmilesRow {
  drug_name: string;
  disease_condition: string;
  smiles: string;
  chembl_id: string;
  category: string;
}

interface SmilesQueryResult {
  success: boolean;
  totalCount: number;
  rows: SmilesRow[];
  categories?: string[];
  diseaseConditions?: string[];
}

interface SmilesStats {
  success: boolean;
  totalRecords: number;
  categories: { category: string; count: number }[];
  diseaseConditions: { condition: string; count: number }[];
}

export default function ExternalSmilesLibrary() {
  const { toast } = useToast();
  const [search, setSearch] = useState("");
  const [diseaseCondition, setDiseaseCondition] = useState<string>("all");
  const [category, setCategory] = useState<string>("all");
  const [page, setPage] = useState(0);
  const [selectedSmiles, setSelectedSmiles] = useState<Set<string>>(new Set());
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<{ type: "project" | "library"; id: string } | null>(null);
  const pageSize = 50;

  const { data: stats, isLoading: statsLoading } = useQuery<SmilesStats>({
    queryKey: ["/api/external-sync/digitalocean/smiles/stats"],
  });

  const { data: smilesData, isLoading: smilesLoading, isFetching } = useQuery<SmilesQueryResult>({
    queryKey: ["/api/external-sync/digitalocean/smiles", { 
      limit: pageSize, 
      offset: page * pageSize,
      diseaseCondition: diseaseCondition !== "all" ? diseaseCondition : undefined,
      category: category !== "all" ? category : undefined,
      search: search.length >= 3 ? search : undefined
    }],
    enabled: true,
  });

  const { data: projects } = useQuery<Project[]>({
    queryKey: ["/api/projects"],
  });

  const { data: libraries } = useQuery<CuratedLibrary[]>({
    queryKey: ["/api/libraries"],
  });

  const importMutation = useMutation({
    mutationFn: async ({ smilesList, targetType, targetId }: { 
      smilesList: string[]; 
      targetType: "project" | "library";
      targetId: string;
    }) => {
      if (targetType === "library") {
        const res = await apiRequest("POST", `/api/libraries/${targetId}/import-smiles`, {
          smilesList,
          tags: ["imported", "external-db"]
        });
        return res.json();
      } else {
        const res = await apiRequest("POST", `/api/projects/${targetId}/molecules/bulk`, {
          smilesList: smilesList.map(s => ({ smiles: s, source: "external-db" }))
        });
        return res.json();
      }
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["/api/libraries"] });
      queryClient.invalidateQueries({ queryKey: ["/api/projects"] });
      toast({ 
        title: "Import successful", 
        description: `Imported ${variables.smilesList.length} SMILES to ${variables.targetType}` 
      });
      setSelectedSmiles(new Set());
      setImportDialogOpen(false);
      setSelectedTarget(null);
    },
    onError: (error: any) => {
      toast({ 
        title: "Import failed", 
        description: error.message || "Failed to import SMILES",
        variant: "destructive" 
      });
    },
  });

  const toggleSmiles = (smiles: string) => {
    const newSelected = new Set(selectedSmiles);
    if (newSelected.has(smiles)) {
      newSelected.delete(smiles);
    } else {
      newSelected.add(smiles);
    }
    setSelectedSmiles(newSelected);
  };

  const selectAll = () => {
    if (!smilesData?.rows) return;
    const allOnPage = new Set(smilesData.rows.map(r => r.smiles));
    if (smilesData.rows.every(r => selectedSmiles.has(r.smiles))) {
      const newSelected = new Set(selectedSmiles);
      allOnPage.forEach(s => newSelected.delete(s));
      setSelectedSmiles(newSelected);
    } else {
      setSelectedSmiles(new Set([...Array.from(selectedSmiles), ...Array.from(allOnPage)]));
    }
  };

  const totalPages = Math.ceil((smilesData?.totalCount || 0) / pageSize);

  const handleImport = () => {
    if (!selectedTarget || selectedSmiles.size === 0) return;
    importMutation.mutate({
      smilesList: Array.from(selectedSmiles),
      targetType: selectedTarget.type,
      targetId: selectedTarget.id
    });
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2" data-testid="text-page-title">
            <Database className="h-6 w-6" />
            External SMILES Library
          </h1>
          <p className="text-muted-foreground">
            Browse 1.6M+ categorized SMILES from ChEMBL database
          </p>
        </div>
        {selectedSmiles.size > 0 && (
          <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
            <DialogTrigger asChild>
              <Button data-testid="button-import-selected">
                <Download className="h-4 w-4 mr-2" />
                Import {selectedSmiles.size} Selected
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Import SMILES</DialogTitle>
                <DialogDescription>
                  Choose where to import the {selectedSmiles.size} selected SMILES
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <h3 className="font-medium flex items-center gap-2">
                    <FlaskConical className="h-4 w-4" />
                    Import to Library
                  </h3>
                  <div className="grid gap-2 max-h-40 overflow-y-auto">
                    {libraries?.map((lib) => (
                      <Button
                        key={lib.id}
                        variant={selectedTarget?.type === "library" && selectedTarget.id === lib.id ? "default" : "outline"}
                        className="justify-start"
                        onClick={() => setSelectedTarget({ type: "library", id: lib.id })}
                        data-testid={`button-select-library-${lib.id}`}
                      >
                        {selectedTarget?.type === "library" && selectedTarget.id === lib.id && (
                          <Check className="h-4 w-4 mr-2" />
                        )}
                        {lib.name}
                        <Badge variant="secondary" className="ml-auto">{lib.moleculeCount || 0}</Badge>
                      </Button>
                    ))}
                    {(!libraries || libraries.length === 0) && (
                      <p className="text-sm text-muted-foreground">No libraries available. Create one first.</p>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <h3 className="font-medium flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    Import to Project
                  </h3>
                  <div className="grid gap-2 max-h-40 overflow-y-auto">
                    {projects?.map((proj) => (
                      <Button
                        key={proj.id}
                        variant={selectedTarget?.type === "project" && selectedTarget.id === proj.id ? "default" : "outline"}
                        className="justify-start"
                        onClick={() => setSelectedTarget({ type: "project", id: proj.id })}
                        data-testid={`button-select-project-${proj.id}`}
                      >
                        {selectedTarget?.type === "project" && selectedTarget.id === proj.id && (
                          <Check className="h-4 w-4 mr-2" />
                        )}
                        {proj.name}
                      </Button>
                    ))}
                    {(!projects || projects.length === 0) && (
                      <p className="text-sm text-muted-foreground">No projects available. Create one first.</p>
                    )}
                  </div>
                </div>
                <Button 
                  className="w-full" 
                  disabled={!selectedTarget || importMutation.isPending}
                  onClick={handleImport}
                  data-testid="button-confirm-import"
                >
                  {importMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Importing...
                    </>
                  ) : (
                    <>
                      <ArrowRight className="h-4 w-4 mr-2" />
                      Import to {selectedTarget?.type === "library" ? "Library" : selectedTarget?.type === "project" ? "Project" : "..."}
                    </>
                  )}
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {statsLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <Card key={i}>
              <CardContent className="pt-4">
                <Skeleton className="h-8 w-24 mb-2" />
                <Skeleton className="h-4 w-32" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : stats?.success ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-4">
              <div className="text-3xl font-bold text-primary">
                {stats.totalRecords.toLocaleString()}
              </div>
              <p className="text-sm text-muted-foreground">Total SMILES</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-3xl font-bold text-primary">
                {stats.diseaseConditions.length}
              </div>
              <p className="text-sm text-muted-foreground">Disease Conditions</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-3xl font-bold text-primary">
                {stats.categories.length}
              </div>
              <p className="text-sm text-muted-foreground">Categories</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-3xl font-bold text-primary">
                {selectedSmiles.size}
              </div>
              <p className="text-sm text-muted-foreground">Selected for Import</p>
            </CardContent>
          </Card>
        </div>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Filter SMILES
          </CardTitle>
          <CardDescription>
            Search by drug name, SMILES, ChEMBL ID, or filter by disease condition
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search drug name, SMILES, ChEMBL ID..."
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setPage(0);
                  }}
                  className="pl-10"
                  data-testid="input-search-smiles"
                />
              </div>
            </div>
            <div className="w-64">
              <Select 
                value={diseaseCondition} 
                onValueChange={(val) => {
                  setDiseaseCondition(val);
                  setPage(0);
                }}
              >
                <SelectTrigger data-testid="select-disease-condition">
                  <SelectValue placeholder="All Disease Conditions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Disease Conditions</SelectItem>
                  {stats?.diseaseConditions.map((dc) => (
                    <SelectItem key={dc.condition} value={dc.condition}>
                      {dc.condition} ({dc.count.toLocaleString()})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="w-64">
              <Select 
                value={category} 
                onValueChange={(val) => {
                  setCategory(val);
                  setPage(0);
                }}
              >
                <SelectTrigger data-testid="select-category">
                  <SelectValue placeholder="All Categories" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Categories</SelectItem>
                  {stats?.categories.map((cat) => (
                    <SelectItem key={cat.category} value={cat.category}>
                      {cat.category} ({cat.count.toLocaleString()})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {smilesLoading ? (
            <div className="space-y-2">
              {[1, 2, 3, 4, 5].map(i => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : smilesData?.success ? (
            <>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>
                  Showing {page * pageSize + 1} - {Math.min((page + 1) * pageSize, smilesData.totalCount)} of {smilesData.totalCount.toLocaleString()}
                  {isFetching && <Loader2 className="h-4 w-4 ml-2 inline animate-spin" />}
                </span>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(p => Math.max(0, p - 1))}
                    disabled={page === 0}
                    data-testid="button-prev-page"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span>Page {page + 1} of {totalPages.toLocaleString()}</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(p => p + 1)}
                    disabled={page >= totalPages - 1}
                    data-testid="button-next-page"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <div className="border rounded-md overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">
                        <Checkbox
                          checked={smilesData.rows.length > 0 && smilesData.rows.every(r => selectedSmiles.has(r.smiles))}
                          onCheckedChange={selectAll}
                          data-testid="checkbox-select-all"
                        />
                      </TableHead>
                      <TableHead>Drug Name / ChEMBL ID</TableHead>
                      <TableHead>Disease Condition</TableHead>
                      <TableHead>Category</TableHead>
                      <TableHead className="max-w-md">SMILES</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {smilesData.rows.map((row, idx) => (
                      <TableRow 
                        key={`${row.chembl_id}-${idx}`}
                        className={selectedSmiles.has(row.smiles) ? "bg-primary/5" : ""}
                      >
                        <TableCell>
                          <Checkbox
                            checked={selectedSmiles.has(row.smiles)}
                            onCheckedChange={() => toggleSmiles(row.smiles)}
                            data-testid={`checkbox-smiles-${idx}`}
                          />
                        </TableCell>
                        <TableCell>
                          <div className="font-medium">{row.drug_name}</div>
                          <div className="text-xs text-muted-foreground font-mono">{row.chembl_id}</div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="secondary">{row.disease_condition}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{row.category || "—"}</Badge>
                        </TableCell>
                        <TableCell className="max-w-md">
                          <code className="text-xs break-all line-clamp-2">{row.smiles}</code>
                        </TableCell>
                      </TableRow>
                    ))}
                    {smilesData.rows.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                          No SMILES found matching your criteria
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              Failed to load SMILES data. Please check database connection.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
