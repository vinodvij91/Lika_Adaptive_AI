import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Plus,
  Search,
  Target,
  ArrowRight,
  CheckCircle,
  XCircle,
  Filter,
} from "lucide-react";
import type { Target as TargetType } from "@shared/schema";

type TargetWithDiseases = TargetType & { diseases: string[] };

export default function TargetsPage() {
  const [search, setSearch] = useState("");
  const [diseaseFilter, setDiseaseFilter] = useState<string>("all");
  const [dialogOpen, setDialogOpen] = useState(false);
  const { toast } = useToast();

  const { data: diseases } = useQuery<{ disease: string; count: number }[]>({
    queryKey: ["/api/diseases"],
  });

  const { data: targets, isLoading } = useQuery<TargetWithDiseases[]>({
    queryKey: ["/api/targets-with-diseases", diseaseFilter],
    queryFn: async () => {
      const url = diseaseFilter && diseaseFilter !== "all" 
        ? `/api/targets-with-diseases?disease=${encodeURIComponent(diseaseFilter)}`
        : "/api/targets-with-diseases";
      const res = await fetch(url, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch targets");
      return res.json();
    },
  });

  const createMutation = useMutation({
    mutationFn: async (data: {
      name: string;
      uniprotId?: string;
      sequence?: string;
      structureSource?: string;
    }) => {
      const res = await apiRequest("POST", "/api/targets", data);
      return res.json();
    },
    onSuccess: (target) => {
      queryClient.invalidateQueries({ queryKey: ["/api/targets-with-diseases"] });
      setDialogOpen(false);
      toast({ title: "Target created", description: `${target.name} has been added.` });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to create target", variant: "destructive" });
    },
  });

  const filteredTargets = targets?.filter(
    (t) =>
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.uniprotId?.toLowerCase().includes(search.toLowerCase()) ||
      t.geneName?.toLowerCase().includes(search.toLowerCase()) ||
      t.organism?.toLowerCase().includes(search.toLowerCase())
  );

  const handleCreate = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    createMutation.mutate({
      name: formData.get("name") as string,
      uniprotId: formData.get("uniprotId") as string || undefined,
      sequence: formData.get("sequence") as string || undefined,
      structureSource: formData.get("structureSource") as string || undefined,
    });
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Targets" }]}
        actions={
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button className="gap-2" data-testid="button-new-target">
                <Plus className="h-4 w-4" />
                Add Target
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <form onSubmit={handleCreate}>
                <DialogHeader>
                  <DialogTitle>Add New Target</DialogTitle>
                  <DialogDescription>
                    Add a protein target for docking and screening campaigns.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Target Name</Label>
                    <Input
                      id="name"
                      name="name"
                      placeholder="e.g., BACE1, EGFR"
                      required
                      data-testid="input-target-name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="uniprotId">UniProt ID</Label>
                    <Input
                      id="uniprotId"
                      name="uniprotId"
                      placeholder="e.g., P56817"
                      data-testid="input-uniprot-id"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="sequence">Sequence (optional)</Label>
                    <Textarea
                      id="sequence"
                      name="sequence"
                      placeholder="Paste protein sequence..."
                      rows={4}
                      className="font-mono text-xs"
                      data-testid="input-sequence"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="structureSource">Structure Source</Label>
                    <Select name="structureSource" defaultValue="uploaded">
                      <SelectTrigger data-testid="select-structure-source">
                        <SelectValue placeholder="Select source" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="uploaded">Uploaded</SelectItem>
                        <SelectItem value="bionemo_predicted">BioNeMo Predicted</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <DialogFooter>
                  <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createMutation.isPending} data-testid="button-create-target">
                    {createMutation.isPending ? "Adding..." : "Add Target"}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by name, gene, organism..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
                data-testid="input-search-targets"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <Select value={diseaseFilter} onValueChange={setDiseaseFilter}>
                <SelectTrigger className="w-[200px]" data-testid="select-disease-filter">
                  <SelectValue placeholder="Filter by disease" />
                </SelectTrigger>
                <SelectContent className="max-h-[300px]">
                  <SelectItem value="all">All Diseases</SelectItem>
                  {diseases?.map((d) => (
                    <SelectItem key={d.disease} value={d.disease}>
                      {d.disease} ({d.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {targets && (
              <Badge variant="outline" className="no-default-hover-elevate no-default-active-elevate">
                {targets.length.toLocaleString()} targets
              </Badge>
            )}
          </div>

          {isLoading ? (
            <Card>
              <CardContent className="p-0 overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Protein Name</TableHead>
                      <TableHead>Gene</TableHead>
                      <TableHead>UniProt ID</TableHead>
                      <TableHead>Organism</TableHead>
                      <TableHead>Disease(s)</TableHead>
                      <TableHead>SMILES</TableHead>
                      <TableHead>Seq Length</TableHead>
                      <TableHead>Structures</TableHead>
                      <TableHead className="w-[80px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[1, 2, 3].map((i) => (
                      <TableRow key={i}>
                        <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-12" /></TableCell>
                        <TableCell><Skeleton className="h-5 w-8" /></TableCell>
                        <TableCell><Skeleton className="h-8 w-8 rounded-md" /></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : filteredTargets && filteredTargets.length > 0 ? (
            <Card>
              <CardContent className="p-0 overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Protein Name</TableHead>
                      <TableHead>Gene</TableHead>
                      <TableHead>UniProt ID</TableHead>
                      <TableHead>Organism</TableHead>
                      <TableHead>Disease(s)</TableHead>
                      <TableHead>SMILES</TableHead>
                      <TableHead>Seq Length</TableHead>
                      <TableHead>Structures</TableHead>
                      <TableHead className="w-[80px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredTargets.map((target) => (
                      <TableRow key={target.id} data-testid={`row-target-${target.id}`}>
                        <TableCell>
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-md bg-chart-3/10 flex items-center justify-center flex-shrink-0">
                              <Target className="h-4 w-4 text-chart-3" />
                            </div>
                            <span className="font-medium text-sm max-w-[200px] truncate" title={target.name}>
                              {target.name}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                            {target.geneName || "-"}
                          </code>
                        </TableCell>
                        <TableCell>
                          <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                            {target.uniprotId || "-"}
                          </code>
                        </TableCell>
                        <TableCell>
                          <span className="text-xs text-muted-foreground max-w-[100px] truncate block" title={target.organism || ""}>
                            {target.organism || "-"}
                          </span>
                        </TableCell>
                        <TableCell>
                          {target.diseases && target.diseases.length > 0 ? (
                            <div className="flex flex-wrap gap-1 max-w-[150px]" title={target.diseases.join(", ")}>
                              <Badge variant="secondary" className="text-xs no-default-hover-elevate no-default-active-elevate">
                                {target.diseases[0]}
                              </Badge>
                              {target.diseases.length > 1 && (
                                <Badge variant="outline" className="text-xs no-default-hover-elevate no-default-active-elevate">
                                  +{target.diseases.length - 1}
                                </Badge>
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {target.smiles ? (
                            <code className="text-xs bg-muted px-1.5 py-0.5 rounded max-w-[100px] truncate block" title={target.smiles}>
                              {target.smiles.length > 15 ? target.smiles.slice(0, 15) + "..." : target.smiles}
                            </code>
                          ) : (
                            <span className="text-xs text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <span className="text-xs">
                            {target.sequenceLength ? target.sequenceLength.toLocaleString() : "-"}
                          </span>
                        </TableCell>
                        <TableCell>
                          {target.hasStructure ? (
                            <Badge variant="outline" className="text-xs no-default-hover-elevate no-default-active-elevate">
                              {target.numStructures || 1}
                            </Badge>
                          ) : (
                            <span className="text-xs text-muted-foreground">0</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Link href={`/targets/${target.id}`}>
                            <Button variant="ghost" size="icon" data-testid={`button-view-target-${target.id}`}>
                              <ArrowRight className="h-4 w-4" />
                            </Button>
                          </Link>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="py-16">
                <div className="flex flex-col items-center justify-center text-center">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <Target className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No targets found</h3>
                  <p className="text-muted-foreground mb-6 max-w-sm">
                    {search
                      ? "No targets match your search."
                      : "Add protein targets to use in your screening campaigns."}
                  </p>
                  {!search && (
                    <Button onClick={() => setDialogOpen(true)} className="gap-2">
                      <Plus className="h-4 w-4" />
                      Add Target
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
