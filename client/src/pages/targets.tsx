import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Link, useLocation } from "wouter";
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
  Eye,
  RotateCw,
  Syringe,
} from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";

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

const VACCINE_TEMPLATES = [
  { id: "nipah-f", name: "Nipah F Protein", organism: "Nipah virus", uniprotId: "Q9IH63", geneName: "F", sequence: "MLARFDALNLFEQPIKGVFSLIALSTSFILYSIAFSIAGAVIQQTAGNLITQ" },
  { id: "sars-spike", name: "SARS-CoV-2 Spike", organism: "SARS-CoV-2", uniprotId: "P0DTC2", geneName: "S", sequence: "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHST" },
  { id: "influenza-ha", name: "Influenza HA", organism: "Influenza A", uniprotId: "P03437", geneName: "HA" },
  { id: "rsv-f", name: "RSV F Protein", organism: "RSV", uniprotId: "P03420", geneName: "F" },
  { id: "hiv-env", name: "HIV Env (gp160)", organism: "HIV-1", uniprotId: "P04578", geneName: "env" },
  { id: "ebola-gp", name: "Ebola GP", organism: "Ebola virus", uniprotId: "Q05320", geneName: "GP" },
];

export default function TargetsPage() {
  const [search, setSearch] = useState("");
  const [diseaseFilter, setDiseaseFilter] = useState<string>("all");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);
  const [selectedTemplates, setSelectedTemplates] = useState<string[]>([]);
  const [predictionDialogOpen, setPredictionDialogOpen] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<TargetWithDiseases | null>(null);
  const [ligandSmiles, setLigandSmiles] = useState("");
  const [manualSequence, setManualSequence] = useState("");
  const [fetchingSequence, setFetchingSequence] = useState(false);
  const [predictionResult, setPredictionResult] = useState<StructurePrediction | null>(null);
  const [show3DViewer, setShow3DViewer] = useState(false);
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const viewerInstanceRef = useRef<any>(null);
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  const iframeSrcDoc = predictionResult?.pdbData ? `<!DOCTYPE html>
<html><head>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>body{margin:0;overflow:hidden;background:#1a1a2e}#viewer{width:100vw;height:100vh}</style>
</head><body>
<div id="viewer"></div>
<script>
var pdbData = ${JSON.stringify(predictionResult.pdbData)};
function init(){
  var v = $3Dmol.createViewer("viewer",{backgroundColor:"0x1a1a2e",antialias:true});
  v.addModel(pdbData,"pdb");
  v.setStyle({},{cartoon:{color:"spectrum"}});
  v.zoomTo();
  v.spin(true);
  v.render();
  window._viewer = v;
  window.addEventListener("message",function(e){
    if(e.data==="spin" && window._viewer) window._viewer.spin(true);
  });
}
if(window.$3Dmol) init();
else document.querySelector("script").onload = init;
</script>
</body></html>` : null;

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

  const templateImportMutation = useMutation({
    mutationFn: async (templateIds: string[]) => {
      const templates = VACCINE_TEMPLATES.filter((t) => templateIds.includes(t.id));
      const results = [];
      for (const template of templates) {
        const res = await apiRequest("POST", "/api/targets", {
          name: template.name,
          uniprotId: template.uniprotId,
          organism: template.organism,
          geneName: template.geneName,
          sequence: template.sequence,
        });
        results.push(await res.json());
      }
      return results;
    },
    onSuccess: (results) => {
      queryClient.invalidateQueries({ queryKey: ["/api/targets-with-diseases"] });
      setTemplateDialogOpen(false);
      setSelectedTemplates([]);
      toast({ title: "Templates imported", description: `${results.length} vaccine antigen target(s) added.` });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to import templates", variant: "destructive" });
    },
  });

  const handleSendToPipeline = async (target: TargetWithDiseases) => {
    try {
      const campaignRes = await apiRequest("POST", "/api/vaccine-campaigns", {
        name: `Campaign - ${target.name}`,
        pathogen: target.organism,
      });
      const campaign = await campaignRes.json();
      await apiRequest("POST", `/api/vaccine-campaigns/${campaign.id}/targets`, {
        targetIds: [target.id],
      });
      toast({ title: "Sent to Pipeline", description: `Created vaccine campaign for ${target.name}` });
      setLocation(`/vaccine-campaigns/${campaign.id}`);
    } catch {
      toast({ title: "Error", description: "Failed to send target to pipeline", variant: "destructive" });
    }
  };

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

  const openPredictionDialog = async (target: TargetWithDiseases) => {
    setSelectedTarget(target);
    setLigandSmiles("");
    setManualSequence("");
    setPredictionResult(null);
    setPredictionDialogOpen(true);
    
    // Auto-fetch sequence from UniProt if target has UniProt ID but no sequence
    if (!target.sequence && target.uniprotId) {
      setFetchingSequence(true);
      try {
        const response = await fetch(`https://rest.uniprot.org/uniprotkb/${target.uniprotId}.fasta`);
        if (response.ok) {
          const fastaText = await response.text();
          // Parse FASTA - skip the header line and join remaining lines
          const lines = fastaText.split('\n');
          const sequence = lines.slice(1).join('').trim();
          if (sequence) {
            setManualSequence(sequence);
            toast({ title: "Sequence loaded", description: `Fetched ${sequence.length} amino acids from UniProt` });
          }
        }
      } catch (error) {
        console.error("Failed to fetch sequence from UniProt:", error);
      } finally {
        setFetchingSequence(false);
      }
    }
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
          <div className="flex items-center gap-2">
            <Button variant="outline" className="gap-2" onClick={() => { setSelectedTemplates([]); setTemplateDialogOpen(true); }} data-testid="button-add-from-templates">
              <Boxes className="h-4 w-4" />
              Add from Templates
            </Button>
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
          </div>
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
                                <DropdownMenuItem
                                  onClick={() => handleSendToPipeline(target)}
                                  data-testid={`button-send-to-pipeline-${target.id}`}
                                >
                                  <Syringe className="h-4 w-4 mr-2" />
                                  Send to Pipeline
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

      <Dialog open={templateDialogOpen} onOpenChange={setTemplateDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add from Vaccine Antigen Templates</DialogTitle>
            <DialogDescription>
              Select vaccine antigen targets to import. These are common pathogen surface proteins used in vaccine development.
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 py-4">
            {VACCINE_TEMPLATES.map((template) => (
              <Card
                key={template.id}
                className={`cursor-pointer transition-colors ${selectedTemplates.includes(template.id) ? "border-primary" : ""}`}
                onClick={() => {
                  setSelectedTemplates((prev) =>
                    prev.includes(template.id)
                      ? prev.filter((id) => id !== template.id)
                      : [...prev, template.id]
                  );
                }}
                data-testid={`card-template-${template.id}`}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <Checkbox
                      checked={selectedTemplates.includes(template.id)}
                      onCheckedChange={(checked) => {
                        setSelectedTemplates((prev) =>
                          checked
                            ? [...prev, template.id]
                            : prev.filter((id) => id !== template.id)
                        );
                      }}
                      data-testid={`checkbox-template-${template.id}`}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm">{template.name}</p>
                      <p className="text-xs text-muted-foreground">{template.organism}</p>
                      <div className="flex items-center gap-2 mt-1 flex-wrap">
                        <Badge variant="outline" className="text-xs no-default-hover-elevate no-default-active-elevate">{template.uniprotId}</Badge>
                        <Badge variant="secondary" className="text-xs no-default-hover-elevate no-default-active-elevate">{template.geneName}</Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setTemplateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              disabled={selectedTemplates.length === 0 || templateImportMutation.isPending}
              onClick={() => templateImportMutation.mutate(selectedTemplates)}
              data-testid="button-import-selected"
            >
              {templateImportMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Importing...
                </>
              ) : (
                `Import Selected (${selectedTemplates.length})`
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

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
                    {fetchingSequence ? (
                      <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-sm">Fetching sequence from UniProt...</span>
                      </div>
                    ) : manualSequence ? (
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-md">
                        <code className="text-xs font-mono break-all">
                          {manualSequence.length > 200 
                            ? manualSequence.slice(0, 200) + "..." 
                            : manualSequence}
                        </code>
                        <p className="text-xs text-muted-foreground mt-2">
                          {manualSequence.length} amino acids (from UniProt)
                        </p>
                      </div>
                    ) : (
                      <>
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
                      </>
                    )}
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
                <div className="pt-2 border-t flex items-center justify-between">
                  <p className="text-xs text-muted-foreground">Model: {predictionResult.modelVersion}</p>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setShow3DViewer(true)}
                    data-testid="button-view-3d"
                  >
                    <Eye className="h-4 w-4 mr-1" />
                    View 3D
                  </Button>
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
              disabled={(!selectedTarget?.sequence && !manualSequence.trim()) || fetchingSequence || structurePredictionMutation.isPending}
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

      <Dialog open={show3DViewer} onOpenChange={setShow3DViewer}>
        <DialogContent className="max-w-4xl max-h-[90vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Boxes className="h-5 w-5" />
              3D Structure Viewer
            </DialogTitle>
            <DialogDescription>
              Predicted 3D structure for {selectedTarget?.name || "Unknown Target"}
              {predictionResult?.isSimulated && " (Simulated)"}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {iframeSrcDoc ? (
              <iframe
                ref={viewerContainerRef as any}
                srcDoc={iframeSrcDoc}
                className="w-full h-[500px] rounded-lg border-0"
                sandbox="allow-scripts allow-same-origin"
                data-testid="3d-viewer-container"
              />
            ) : (
              <div className="w-full h-[500px] bg-muted rounded-lg flex items-center justify-center">
                <p className="text-muted-foreground text-sm">No structure data available</p>
              </div>
            )}
            {predictionResult && (
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-4 text-sm">
                  <Badge variant="secondary">pLDDT: {predictionResult.metrics.pLDDT.toFixed(1)}</Badge>
                  <Badge variant="secondary">pTM: {predictionResult.metrics.pTM.toFixed(3)}</Badge>
                  <Badge variant="outline">{predictionResult.metrics.numAtoms} atoms</Badge>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const iframe = viewerContainerRef.current as HTMLIFrameElement | null;
                    if (iframe?.contentWindow) {
                      iframe.contentWindow.postMessage("spin", "*");
                    }
                  }}
                >
                  <RotateCw className="h-4 w-4 mr-1" />
                  Spin
                </Button>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShow3DViewer(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
