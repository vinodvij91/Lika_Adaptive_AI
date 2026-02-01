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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
  MoreVertical,
  Boxes,
  Loader2,
} from "lucide-react";
import type { Target as TargetType } from "@shared/schema";

type TargetWithDiseases = TargetType & { diseases: string[] };

interface StructurePrediction {
  id: string;
  name: string;
  pdbData: string;
  confidenceScore: number;
  metrics: {
    pLDDT: number;
    pTM: number;
    iPTM?: number;
    numResidues: number;
    numAtoms: number;
  };
  ligandBindingSite?: {
    residues: string[];
    bindingPocketVolume: number;
    interactionType: string[];
  };
  modelVersion: string;
  isSimulated: boolean;
  fromCache?: boolean;
}

export default function TargetsPage() {
  const [search, setSearch] = useState("");
  const [diseaseFilter, setDiseaseFilter] = useState<string>("all");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [predictionDialogOpen, setPredictionDialogOpen] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<TargetWithDiseases | null>(null);
  const [ligandSmiles, setLigandSmiles] = useState("");
  const [manualSequence, setManualSequence] = useState("");
  const [predictionResult, setPredictionResult] = useState<StructurePrediction | null>(null);
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

  const structurePredictionMutation = useMutation({
    mutationFn: async (data: {
      proteinSequence: string;
      ligandSmiles?: string;
      name?: string;
      targetId?: string;
    }) => {
      const res = await apiRequest("POST", "/api/structures/openfold3/predict", data);
      return res.json();
    },
    onSuccess: (result: StructurePrediction) => {
      setPredictionResult(result);
      toast({ 
        title: result.fromCache ? "Structure Retrieved from Cache" : "3D Structure Predicted", 
        description: `pLDDT: ${result.metrics.pLDDT.toFixed(1)}, pTM: ${result.metrics.pTM.toFixed(2)}${result.isSimulated ? " (simulated)" : ""}` 
      });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to predict structure", variant: "destructive" });
    },
  });

  const handlePredictStructure = () => {
    const sequence = selectedTarget?.sequence || manualSequence.trim();
    if (!sequence) {
      toast({ title: "Error", description: "Please provide a protein sequence for structure prediction", variant: "destructive" });
      return;
    }
    structurePredictionMutation.mutate({
      proteinSequence: sequence,
      ligandSmiles: ligandSmiles || undefined,
      name: `${selectedTarget?.name || "Unknown"} Complex`,
      targetId: selectedTarget?.id,
    });
  };

  const openPredictionDialog = (target: TargetWithDiseases) => {
    setSelectedTarget(target);
    setLigandSmiles("");
    setManualSequence("");
    setPredictionResult(null);
    setPredictionDialogOpen(true);
  };

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
                  {diseases?.slice().sort((a, b) => a.disease.localeCompare(b.disease)).map((d) => (
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
                          <div className="flex items-center gap-1">
                            <Button 
                              variant="ghost" 
                              size="icon" 
                              onClick={() => openPredictionDialog(target)}
                              title="Predict 3D Complex"
                              data-testid={`button-predict-structure-${target.id}`}
                            >
                              <Boxes className="h-4 w-4" />
                            </Button>
                            <Link href={`/targets/${target.id}`}>
                              <Button variant="ghost" size="icon" data-testid={`button-view-target-${target.id}`}>
                                <ArrowRight className="h-4 w-4" />
                              </Button>
                            </Link>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon" data-testid={`button-target-actions-${target.id}`}>
                                  <MoreVertical className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem 
                                  onClick={() => openPredictionDialog(target)}
                                  data-testid={`button-predict-structure-menu-${target.id}`}
                                >
                                  <Boxes className="h-4 w-4 mr-2" />
                                  Predict 3D Complex
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
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

      <Dialog open={predictionDialogOpen} onOpenChange={setPredictionDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Predict 3D Complex - OpenFold3 NIM</DialogTitle>
            <DialogDescription>
              {selectedTarget ? (
                <>Predict the 3D structure of <strong>{selectedTarget.name}</strong> with optional ligand using NVIDIA OpenFold3 NIM (AlphaFold3-compatible).</>
              ) : (
                "Select a target to predict its 3D structure."
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {selectedTarget && (
              <div className="space-y-2">
                <Label>Target Sequence</Label>
                {selectedTarget.sequence ? (
                  <div className="p-3 bg-muted rounded-md">
                    <code className="text-xs font-mono break-all">
                      {selectedTarget.sequence.length > 200 
                        ? selectedTarget.sequence.slice(0, 200) + "..." 
                        : selectedTarget.sequence}
                    </code>
                    <p className="text-xs text-muted-foreground mt-2">
                      {selectedTarget.sequence.length} amino acids
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <p className="text-sm text-amber-600 bg-amber-50 dark:bg-amber-900/20 p-2 rounded">
                      This target has no sequence stored. Enter a protein sequence below to predict its structure.
                    </p>
                    <textarea
                      value={manualSequence}
                      onChange={(e) => setManualSequence(e.target.value)}
                      placeholder="Enter amino acid sequence (e.g., MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF...)"
                      className="w-full h-24 p-3 font-mono text-xs border rounded-md bg-background resize-none"
                      data-testid="textarea-manual-sequence"
                    />
                    <p className="text-xs text-muted-foreground">
                      {manualSequence.trim().length > 0 && `${manualSequence.trim().length} characters`}
                    </p>
                  </div>
                )}
              </div>
            )}
            <div className="space-y-2">
              <Label htmlFor="ligandSmiles">Ligand SMILES (optional)</Label>
              <Input
                id="ligandSmiles"
                placeholder="e.g., CC(=O)Nc1ccc(O)cc1"
                value={ligandSmiles}
                onChange={(e) => setLigandSmiles(e.target.value)}
                className="font-mono"
                data-testid="input-ligand-smiles"
              />
              <p className="text-xs text-muted-foreground">
                Enter a SMILES string to predict protein-ligand complex structure
              </p>
            </div>
            {predictionResult && (
              <div className="space-y-3 p-4 bg-muted/50 rounded-lg border">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Prediction Results</h4>
                  {predictionResult.fromCache && (
                    <Badge variant="secondary" className="text-xs">From Cache</Badge>
                  )}
                  {predictionResult.isSimulated && (
                    <Badge variant="outline" className="text-xs">Simulated</Badge>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-muted-foreground">pLDDT Score:</span>
                    <span className="ml-2 font-medium">{predictionResult.metrics.pLDDT.toFixed(1)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">pTM Score:</span>
                    <span className="ml-2 font-medium">{predictionResult.metrics.pTM.toFixed(3)}</span>
                  </div>
                  {predictionResult.metrics.iPTM !== undefined && (
                    <div>
                      <span className="text-muted-foreground">iPTM Score:</span>
                      <span className="ml-2 font-medium">{predictionResult.metrics.iPTM.toFixed(3)}</span>
                    </div>
                  )}
                  <div>
                    <span className="text-muted-foreground">Atoms:</span>
                    <span className="ml-2 font-medium">{predictionResult.metrics.numAtoms}</span>
                  </div>
                </div>
                {predictionResult.ligandBindingSite && (
                  <div className="pt-2 border-t">
                    <p className="text-sm font-medium mb-1">Binding Site</p>
                    <div className="text-xs text-muted-foreground space-y-1">
                      <p>Residues: {predictionResult.ligandBindingSite.residues.slice(0, 5).join(", ")}{predictionResult.ligandBindingSite.residues.length > 5 ? "..." : ""}</p>
                      <p>Pocket Volume: {predictionResult.ligandBindingSite.bindingPocketVolume.toFixed(0)} \u00c5\u00b3</p>
                      <p>Interactions: {predictionResult.ligandBindingSite.interactionType.join(", ")}</p>
                    </div>
                  </div>
                )}
                <div className="pt-2 border-t">
                  <p className="text-xs text-muted-foreground">Model: {predictionResult.modelVersion}</p>
                </div>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPredictionDialogOpen(false)}>
              Close
            </Button>
            <Button 
              onClick={handlePredictStructure}
              disabled={(!selectedTarget?.sequence && !manualSequence.trim()) || structurePredictionMutation.isPending}
              data-testid="button-run-prediction"
            >
              {structurePredictionMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Boxes className="h-4 w-4 mr-2" />
                  Predict Structure
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
