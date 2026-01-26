import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { 
  Layers, Play, Box, Atom, Zap, 
  Loader2, FlaskConical, Target, Activity, Dna, Sparkles, CheckCircle2
} from "lucide-react";

declare global {
  interface Window {
    $3Dmol: any;
  }
}

interface BioNemoPropertyPrediction {
  smiles: string;
  qed: number;
  plogP: number;
  molecularWeight: number;
  synthesizability: number;
  drugLikeness: string;
  confidence: number;
}

interface BioNemoDockingPrediction {
  moleculeSmiles: string;
  targetId: string;
  bindingAffinity: number;
  poseScore: number;
  confidence: number;
}

interface MoleculeResult {
  smiles: string;
  valid: boolean;
  molBlock?: string;
  properties?: {
    mw: number;
    logp: number;
    tpsa?: number;
    hbd: number;
    hba: number;
    numAtoms?: number;
  };
  docking?: {
    bindingEnergy: number;
    affinityNM?: number;
  };
  bionemo?: BioNemoPropertyPrediction;
  bionemoDocking?: BioNemoDockingPrediction;
}

const DEMO_MOLECULES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1cnc2c1c(=O)n(c(=O)n2C)C" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" },
  { name: "Naproxen", smiles: "COc1ccc2cc(C(C)C(=O)O)ccc2c1" },
  { name: "Acetaminophen", smiles: "CC(=O)Nc1ccc(O)cc1" },
  { name: "Diazepam", smiles: "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21" },
];

const DEFAULT_TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH";

function Molecule3DViewer({ molBlock, style = "stick" }: { molBlock: string; style?: string }) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (!viewerRef.current || !molBlock) return;

    const loadViewer = async () => {
      if (!window.$3Dmol) {
        const script = document.createElement("script");
        script.src = "https://3dmol.org/build/3Dmol-min.js";
        script.async = true;
        await new Promise((resolve) => {
          script.onload = resolve;
          document.head.appendChild(script);
        });
      }

      if (viewerInstanceRef.current) {
        viewerInstanceRef.current.clear();
      } else {
        viewerInstanceRef.current = window.$3Dmol.createViewer(viewerRef.current, {
          backgroundColor: "0x1a1a2e",
        });
      }

      const viewer = viewerInstanceRef.current;
      viewer.addModel(molBlock, "mol");

      if (style === "stick") {
        viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });
      } else if (style === "sphere") {
        viewer.setStyle({}, { sphere: { scale: 0.8 } });
      } else if (style === "cartoon") {
        viewer.setStyle({}, { stick: { colorscheme: "Jmol" } });
      }

      viewer.zoomTo();
      viewer.render();
      viewer.spin("y", 0.5);
    };

    loadViewer();
  }, [molBlock, style]);

  return (
    <div
      ref={viewerRef}
      className="w-full h-[400px] rounded-lg overflow-hidden"
      style={{ position: "relative" }}
      data-testid="viewer-3d-molecule"
    />
  );
}

export default function DockingPage() {
  const [smilesInput, setSmilesInput] = useState("");
  const [targetSequence, setTargetSequence] = useState(DEFAULT_TARGET);
  const [selectedMolecule, setSelectedMolecule] = useState<MoleculeResult | null>(null);
  const [viewStyle, setViewStyle] = useState<"stick" | "sphere" | "cartoon">("stick");
  const [activeTab, setActiveTab] = useState("docking");
  const [results, setResults] = useState<MoleculeResult[]>([]);

  const { data: bionemoStatus } = useQuery<{configured: boolean; provider: string; capabilities: string[]}>({
    queryKey: ["/api/bionemo/status"],
  });

  const { data: computeNodes } = useQuery<any[]>({
    queryKey: ["/api/compute-nodes"],
  });

  const onlineNodes = computeNodes?.filter((n) => n.status === "active") || [];

  const generate3DMutation = useMutation({
    mutationFn: async (smiles: string[]) => {
      const res = await apiRequest("POST", "/api/compute/generate-3d", { smiles });
      return res.json();
    },
  });

  const bionemoPropertyMutation = useMutation({
    mutationFn: async (smilesList: string[]) => {
      const res = await apiRequest("POST", "/api/bionemo/predict/batch", { smilesList });
      return res.json();
    },
  });

  const bionemoDockingMutation = useMutation({
    mutationFn: async ({ smiles, targetSequence }: { smiles: string; targetSequence?: string }) => {
      const res = await apiRequest("POST", "/api/bionemo/predict/docking", { smiles, targetSequence });
      return res.json();
    },
  });

  const handleFullPipeline = async () => {
    const smilesList = smilesInput.trim()
      ? smilesInput.split("\n").map((s) => s.trim()).filter(Boolean)
      : DEMO_MOLECULES.slice(0, 4).map((m) => m.smiles);

    setResults([]);
    
    const gen3dResult = await generate3DMutation.mutateAsync(smilesList);
    
    if (gen3dResult.results) {
      const moleculeResults: MoleculeResult[] = gen3dResult.results.map((r: any) => ({
        smiles: r.smiles,
        valid: r.valid,
        molBlock: r.molBlock,
        properties: r.properties,
        docking: r.docking,
      }));

      setResults(moleculeResults);
      if (moleculeResults.length > 0) {
        setSelectedMolecule(moleculeResults[0]);
      }

      if (bionemoStatus?.configured) {
        try {
          const bionemoProps = await bionemoPropertyMutation.mutateAsync(smilesList);
          
          const updatedResults = moleculeResults.map((mol, idx) => ({
            ...mol,
            bionemo: bionemoProps[idx],
          }));
          setResults(updatedResults);

          for (let i = 0; i < smilesList.length; i++) {
            try {
              const dockResult = await bionemoDockingMutation.mutateAsync({ 
                smiles: smilesList[i], 
                targetSequence 
              });
              
              setResults(prev => prev.map((mol, idx) => 
                idx === i ? { ...mol, bionemoDocking: dockResult } : mol
              ));
            } catch (e) {
              console.error("Docking failed for", smilesList[i]);
            }
          }
        } catch (e) {
          console.error("BioNemo prediction failed:", e);
        }
      }
    }
  };

  const isLoading = generate3DMutation.isPending || bionemoPropertyMutation.isPending || bionemoDockingMutation.isPending;

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-violet-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-6">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zMCAxMGwyMCAxMHYyMEwzMCA1MCAxMCA0MFYyMEwzMCAxMHoiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjE1KSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9nPjwvc3ZnPg==')] opacity-40" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Layers className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">BioNemo Docking & 3D</h1>
                  <p className="text-violet-100">NVIDIA DiffDock-powered molecular docking with 3D visualization</p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-sm text-violet-100">
                {bionemoStatus?.configured && (
                  <Badge variant="outline" className="bg-emerald-500/20 text-white border-emerald-400/50 gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    BioNemo Connected
                  </Badge>
                )}
                <Badge variant="outline" className="bg-white/20 text-white border-white/30 gap-1">
                  <Sparkles className="h-3 w-3" />
                  DiffDock
                </Badge>
                <Badge variant="outline" className="bg-white/20 text-white border-white/30 gap-1">
                  <Dna className="h-3 w-3" />
                  MolMIM Embeddings
                </Badge>
                {onlineNodes.length > 0 && (
                  <Badge variant="outline" className="bg-white/20 text-white border-white/30">
                    <Zap className="h-3 w-3 mr-1" />
                    {onlineNodes.length} GPU node{onlineNodes.length !== 1 ? "s" : ""}
                  </Badge>
                )}
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-4">
              <Card className="shadow-lg">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <FlaskConical className="h-5 w-5 text-violet-500" />
                    Input Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Tabs value={activeTab} onValueChange={setActiveTab}>
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="docking" data-testid="tab-docking">
                        <Target className="h-4 w-4 mr-2" />
                        Docking
                      </TabsTrigger>
                      <TabsTrigger value="3d" data-testid="tab-3d">
                        <Box className="h-4 w-4 mr-2" />
                        3D View
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="docking" className="space-y-4 mt-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">Ligand SMILES (one per line)</label>
                        <Textarea
                          className="h-24 font-mono text-sm"
                          placeholder="Enter SMILES or leave empty for demo..."
                          value={smilesInput}
                          onChange={(e) => setSmilesInput(e.target.value)}
                          data-testid="input-smiles"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">Target Protein Sequence</label>
                        <Textarea
                          className="h-20 font-mono text-xs"
                          placeholder="Protein sequence (FASTA)..."
                          value={targetSequence}
                          onChange={(e) => setTargetSequence(e.target.value)}
                          data-testid="input-target"
                        />
                      </div>
                      <Button
                        className="w-full gap-2 bg-gradient-to-r from-emerald-500 to-teal-500"
                        onClick={handleFullPipeline}
                        disabled={isLoading}
                        data-testid="button-run-docking"
                      >
                        {isLoading ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Play className="h-4 w-4" />
                        )}
                        Run BioNemo Docking Pipeline
                      </Button>
                    </TabsContent>

                    <TabsContent value="3d" className="space-y-4 mt-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">SMILES for 3D Generation</label>
                        <Textarea
                          className="h-32 font-mono text-sm"
                          placeholder="Enter SMILES for 3D conformer generation..."
                          value={smilesInput}
                          onChange={(e) => setSmilesInput(e.target.value)}
                          data-testid="input-smiles-3d"
                        />
                      </div>
                      <Button
                        className="w-full gap-2 bg-gradient-to-r from-violet-500 to-purple-500"
                        onClick={handleFullPipeline}
                        disabled={isLoading}
                        data-testid="button-generate-3d"
                      >
                        {isLoading ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Atom className="h-4 w-4" />
                        )}
                        Generate 3D + Properties
                      </Button>
                    </TabsContent>
                  </Tabs>

                  <div className="pt-2 border-t">
                    <p className="text-xs text-muted-foreground mb-2">Quick Demo Molecules:</p>
                    <div className="flex flex-wrap gap-1">
                      {DEMO_MOLECULES.slice(0, 4).map((mol) => (
                        <Button
                          key={mol.name}
                          size="sm"
                          variant="outline"
                          className="text-xs"
                          onClick={() => setSmilesInput(mol.smiles)}
                          data-testid={`button-demo-${mol.name.toLowerCase()}`}
                        >
                          {mol.name}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {results.length > 0 && (
                <Card className="shadow-lg">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <Activity className="h-5 w-5 text-emerald-500" />
                        Results
                      </span>
                      <Badge variant="secondary" className="font-mono">
                        {results.length} molecules
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-[350px] overflow-auto">
                      {results.map((mol, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            selectedMolecule === mol
                              ? "bg-violet-500/10 border-violet-500"
                              : "hover:bg-muted/50"
                          }`}
                          onClick={() => setSelectedMolecule(mol)}
                          data-testid={`result-molecule-${idx}`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <code className="text-xs font-mono truncate max-w-[160px]">
                              {mol.smiles}
                            </code>
                            {mol.valid && (
                              <Badge variant="outline" className="text-emerald-500 border-emerald-500/30 text-xs">
                                Valid
                              </Badge>
                            )}
                          </div>
                          
                          {mol.bionemoDocking && (
                            <div className="flex items-center gap-2 text-xs mt-1">
                              <Badge className="bg-emerald-500/20 text-emerald-600 border-0">
                                {mol.bionemoDocking.bindingAffinity.toFixed(2)} kcal/mol
                              </Badge>
                              <span className="text-muted-foreground">
                                Pose: {(mol.bionemoDocking.poseScore * 100).toFixed(0)}%
                              </span>
                            </div>
                          )}
                          
                          {mol.bionemo && (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                              <span>QED: {mol.bionemo.qed.toFixed(2)}</span>
                              <span>pLogP: {mol.bionemo.plogP.toFixed(1)}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            <div className="lg:col-span-2 space-y-4">
              <Card className="shadow-lg overflow-hidden">
                <CardHeader className="pb-3 flex flex-row items-center justify-between">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Box className="h-5 w-5 text-violet-500" />
                    3D Molecular Viewer
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant={viewStyle === "stick" ? "default" : "outline"}
                      onClick={() => setViewStyle("stick")}
                      data-testid="button-style-stick"
                    >
                      Stick
                    </Button>
                    <Button
                      size="sm"
                      variant={viewStyle === "sphere" ? "default" : "outline"}
                      onClick={() => setViewStyle("sphere")}
                      data-testid="button-style-sphere"
                    >
                      Sphere
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  {selectedMolecule?.molBlock ? (
                    <Molecule3DViewer molBlock={selectedMolecule.molBlock} style={viewStyle} />
                  ) : (
                    <div className="h-[400px] flex items-center justify-center bg-muted/30">
                      <div className="text-center">
                        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-500/10 to-purple-500/10 flex items-center justify-center mx-auto mb-4">
                          <Atom className="h-9 w-9 text-violet-500" />
                        </div>
                        <p className="font-medium mb-2">No molecule selected</p>
                        <p className="text-sm text-muted-foreground max-w-sm">
                          Run the BioNemo docking pipeline to generate 3D structures
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {selectedMolecule && (selectedMolecule.bionemo || selectedMolecule.bionemoDocking || selectedMolecule.properties) && (
                <Card className="shadow-lg">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-amber-500" />
                      BioNemo Predictions
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {selectedMolecule.bionemoDocking && (
                      <div className="p-4 rounded-lg border bg-gradient-to-r from-emerald-500/5 to-teal-500/5">
                        <h4 className="font-medium mb-3 flex items-center gap-2">
                          <Target className="h-4 w-4 text-emerald-500" />
                          DiffDock Docking Results
                        </h4>
                        <div className="grid grid-cols-3 gap-4">
                          <div>
                            <p className="text-3xl font-bold text-emerald-600">
                              {selectedMolecule.bionemoDocking.bindingAffinity.toFixed(2)}
                            </p>
                            <p className="text-sm text-muted-foreground">Binding Affinity (kcal/mol)</p>
                          </div>
                          <div>
                            <p className="text-3xl font-bold text-teal-600">
                              {(selectedMolecule.bionemoDocking.poseScore * 100).toFixed(1)}%
                            </p>
                            <p className="text-sm text-muted-foreground">Pose Confidence</p>
                          </div>
                          <div>
                            <p className="text-3xl font-bold text-blue-600">
                              {(selectedMolecule.bionemoDocking.confidence * 100).toFixed(0)}%
                            </p>
                            <p className="text-sm text-muted-foreground">Model Confidence</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {selectedMolecule.bionemo && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-3 rounded-lg bg-blue-500/10">
                          <p className="text-2xl font-bold text-blue-600">{selectedMolecule.bionemo.qed.toFixed(3)}</p>
                          <p className="text-xs text-muted-foreground">QED Score</p>
                        </div>
                        <div className="p-3 rounded-lg bg-emerald-500/10">
                          <p className="text-2xl font-bold text-emerald-600">{selectedMolecule.bionemo.plogP.toFixed(2)}</p>
                          <p className="text-xs text-muted-foreground">pLogP</p>
                        </div>
                        <div className="p-3 rounded-lg bg-violet-500/10">
                          <p className="text-2xl font-bold text-violet-600">{selectedMolecule.bionemo.molecularWeight.toFixed(0)}</p>
                          <p className="text-xs text-muted-foreground">MW (Da)</p>
                        </div>
                        <div className="p-3 rounded-lg bg-amber-500/10">
                          <p className="text-2xl font-bold text-amber-600">{(selectedMolecule.bionemo.synthesizability * 100).toFixed(0)}%</p>
                          <p className="text-xs text-muted-foreground">Synthesizability</p>
                        </div>
                      </div>
                    )}

                    {selectedMolecule.bionemo && (
                      <div className="flex items-center gap-3 p-3 rounded-lg border">
                        <span className="text-sm font-medium">Drug-likeness:</span>
                        <Badge 
                          className={
                            selectedMolecule.bionemo.drugLikeness === "High" 
                              ? "bg-emerald-500" 
                              : selectedMolecule.bionemo.drugLikeness === "Moderate"
                              ? "bg-amber-500"
                              : "bg-red-500"
                          }
                        >
                          {selectedMolecule.bionemo.drugLikeness}
                        </Badge>
                        <span className="text-sm text-muted-foreground ml-auto">
                          Confidence: {(selectedMolecule.bionemo.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}

                    {selectedMolecule.properties && !selectedMolecule.bionemo && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-3 rounded-lg bg-blue-500/10">
                          <p className="text-2xl font-bold text-blue-600">{selectedMolecule.properties.mw}</p>
                          <p className="text-xs text-muted-foreground">Molecular Weight</p>
                        </div>
                        <div className="p-3 rounded-lg bg-emerald-500/10">
                          <p className="text-2xl font-bold text-emerald-600">{selectedMolecule.properties.logp}</p>
                          <p className="text-xs text-muted-foreground">LogP</p>
                        </div>
                        <div className="p-3 rounded-lg bg-violet-500/10">
                          <p className="text-2xl font-bold text-violet-600">{selectedMolecule.properties.hba}</p>
                          <p className="text-xs text-muted-foreground">H-Bond Acceptors</p>
                        </div>
                        <div className="p-3 rounded-lg bg-amber-500/10">
                          <p className="text-2xl font-bold text-amber-600">{selectedMolecule.properties.hbd}</p>
                          <p className="text-xs text-muted-foreground">H-Bond Donors</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
