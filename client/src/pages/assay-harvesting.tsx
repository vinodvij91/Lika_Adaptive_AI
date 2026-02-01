import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
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
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import {
  FlaskConical,
  Target,
  Beaker,
  Shield,
  Activity,
  Atom,
  Download,
  RefreshCw,
  FileJson,
  ClipboardCopy,
  Check,
  ExternalLink,
} from "lucide-react";

interface DiseaseTarget {
  name: string;
  symbol: string;
  uniprotId?: string;
  chemblId?: string;
  role: "primary" | "secondary" | "safety";
}

interface RecommendedAssay {
  id: string;
  name: string;
  description: string;
  source: "PubChem" | "ChEMBL" | "Manual";
  category: "binding" | "functional" | "adme" | "safety" | "physicochemical";
  targetName?: string;
  confidence: number;
}

interface DiseaseTemplate {
  disease: string;
  therapeuticArea: string;
  targets: DiseaseTarget[];
  recommendedAssays: Record<string, RecommendedAssay[]>;
  lastUpdated: string;
}

const categoryConfig: Record<string, { label: string; icon: typeof FlaskConical; color: string }> = {
  binding: { label: "Binding", icon: Target, color: "bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300" },
  functional: { label: "Functional", icon: Activity, color: "bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300" },
  adme: { label: "ADME", icon: Beaker, color: "bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300" },
  safety: { label: "Safety", icon: Shield, color: "bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300" },
  physicochemical: { label: "Physicochemical", icon: Atom, color: "bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-300" },
};

const sourceColors: Record<string, string> = {
  PubChem: "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300",
  ChEMBL: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300",
  Manual: "bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-300",
};

export default function AssayHarvestingPage() {
  const { toast } = useToast();
  const [selectedDisease, setSelectedDisease] = useState<string>("Alzheimer's Disease");
  const [selectedAssays, setSelectedAssays] = useState<Set<string>>(new Set());
  const [protocolDialogOpen, setProtocolDialogOpen] = useState(false);
  const [campaignName, setCampaignName] = useState("");
  const [generatedProtocol, setGeneratedProtocol] = useState<any>(null);

  const { data: diseasesData, isLoading: diseasesLoading } = useQuery<{ diseases: string[] }>({
    queryKey: ["/api/assays/diseases"],
  });

  const { data: template, isLoading: templateLoading, refetch: refetchTemplate } = useQuery<DiseaseTemplate>({
    queryKey: ["/api/assays", selectedDisease],
    enabled: !!selectedDisease,
  });

  const harvestMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/assays/harvest", {
        disease: selectedDisease,
        targets: template?.targets?.map(t => t.symbol) || [],
      });
      return res.json();
    },
    onSuccess: () => {
      refetchTemplate();
      toast({
        title: "Assays Harvested",
        description: "Successfully refreshed assay data from PubChem and ChEMBL",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Harvest Failed",
        description: error.message || "Failed to harvest assays",
        variant: "destructive",
      });
    },
  });

  const protocolMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/campaign/protocol", {
        disease: selectedDisease,
        selectedAssays: Array.from(selectedAssays),
        campaignName: campaignName || `${selectedDisease} Discovery Campaign`,
      });
      return res.json();
    },
    onSuccess: (data) => {
      setGeneratedProtocol(data.protocol);
      toast({
        title: "Protocol Generated",
        description: `Created protocol with ${data.protocol.assays.length} assays`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Protocol Generation Failed",
        description: error.message || "Failed to generate protocol",
        variant: "destructive",
      });
    },
  });

  const handleAssayToggle = (assayId: string) => {
    setSelectedAssays(prev => {
      const next = new Set(prev);
      if (next.has(assayId)) {
        next.delete(assayId);
      } else {
        next.add(assayId);
      }
      return next;
    });
  };

  const handleSelectAllInCategory = (category: string) => {
    if (!template) return;
    const assaysInCategory = template.recommendedAssays[category] || [];
    setSelectedAssays(prev => {
      const next = new Set(prev);
      for (const assay of assaysInCategory) {
        next.add(assay.id);
      }
      return next;
    });
  };

  const handleClearCategory = (category: string) => {
    if (!template) return;
    const assaysInCategory = template.recommendedAssays[category] || [];
    setSelectedAssays(prev => {
      const next = new Set(prev);
      for (const assay of assaysInCategory) {
        next.delete(assay.id);
      }
      return next;
    });
  };

  const copyProtocolToClipboard = () => {
    navigator.clipboard.writeText(JSON.stringify(generatedProtocol, null, 2));
    toast({
      title: "Copied",
      description: "Protocol JSON copied to clipboard",
    });
  };

  const downloadProtocol = () => {
    const blob = new Blob([JSON.stringify(generatedProtocol, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${generatedProtocol?.name?.replace(/\s+/g, "_") || "protocol"}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const totalAssays = template
    ? Object.values(template.recommendedAssays).reduce((sum, arr) => sum + arr.length, 0)
    : 0;

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Assay Harvesting</h1>
          <p className="text-muted-foreground">
            Discover and select assays from PubChem and ChEMBL for your discovery campaigns
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={selectedDisease} onValueChange={setSelectedDisease}>
            <SelectTrigger className="w-[220px]" data-testid="select-disease">
              <SelectValue placeholder="Select disease" />
            </SelectTrigger>
            <SelectContent>
              {diseasesLoading ? (
                <SelectItem value="loading" disabled>Loading...</SelectItem>
              ) : (
                diseasesData?.diseases.map(disease => (
                  <SelectItem key={disease} value={disease}>{disease}</SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            onClick={() => harvestMutation.mutate()}
            disabled={harvestMutation.isPending}
            data-testid="button-harvest-assays"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${harvestMutation.isPending ? "animate-spin" : ""}`} />
            {harvestMutation.isPending ? "Harvesting..." : "Harvest Fresh"}
          </Button>
        </div>
      </div>

      {templateLoading ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Skeleton className="h-96" />
          </div>
          <Skeleton className="h-96" />
        </div>
      ) : template ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <FlaskConical className="w-5 h-5" />
                    Recommended Assays
                    <Badge variant="secondary">{totalAssays} found</Badge>
                  </CardTitle>
                  <CardDescription>
                    Curated assays for {template.disease} ({template.therapeuticArea})
                  </CardDescription>
                </div>
                <Badge variant="outline">{selectedAssays.size} selected</Badge>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="binding" className="w-full">
                  <TabsList className="grid w-full grid-cols-5">
                    {Object.entries(categoryConfig).map(([key, config]) => (
                      <TabsTrigger key={key} value={key} className="gap-1" data-testid={`tab-${key}`}>
                        <config.icon className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">{config.label}</span>
                        <Badge variant="secondary" className="ml-1 text-xs">
                          {template.recommendedAssays[key]?.length || 0}
                        </Badge>
                      </TabsTrigger>
                    ))}
                  </TabsList>

                  {Object.entries(categoryConfig).map(([category, config]) => (
                    <TabsContent key={category} value={category}>
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm text-muted-foreground">
                          {config.label} assays for {template.disease}
                        </span>
                        <div className="flex gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleSelectAllInCategory(category)}
                            data-testid={`button-select-all-${category}`}
                          >
                            Select All
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleClearCategory(category)}
                            data-testid={`button-clear-${category}`}
                          >
                            Clear
                          </Button>
                        </div>
                      </div>
                      <ScrollArea className="h-[400px] pr-4">
                        <div className="space-y-3">
                          {(template.recommendedAssays[category] || []).map((assay) => (
                            <div
                              key={assay.id}
                              className={`p-3 rounded-lg border transition-colors ${
                                selectedAssays.has(assay.id)
                                  ? "border-primary bg-primary/5"
                                  : "border-border hover:border-primary/50"
                              }`}
                            >
                              <div className="flex items-start gap-3">
                                <Checkbox
                                  checked={selectedAssays.has(assay.id)}
                                  onCheckedChange={() => handleAssayToggle(assay.id)}
                                  data-testid={`checkbox-assay-${assay.id}`}
                                />
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span className="font-medium">{assay.name}</span>
                                    <Badge variant="outline" className="text-xs">
                                      {assay.id}
                                    </Badge>
                                    <Badge className={`text-xs ${sourceColors[assay.source]}`}>
                                      {assay.source}
                                    </Badge>
                                    {assay.id.startsWith("AID") && (
                                      <a
                                        href={`https://pubchem.ncbi.nlm.nih.gov/bioassay/${assay.id.replace("AID ", "")}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-muted-foreground hover:text-primary"
                                      >
                                        <ExternalLink className="w-3.5 h-3.5" />
                                      </a>
                                    )}
                                    {assay.id.startsWith("CHEMBL") && (
                                      <a
                                        href={`https://www.ebi.ac.uk/chembl/assay_report_card/${assay.id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-muted-foreground hover:text-primary"
                                      >
                                        <ExternalLink className="w-3.5 h-3.5" />
                                      </a>
                                    )}
                                  </div>
                                  <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                                    {assay.description}
                                  </p>
                                  <div className="flex items-center gap-2 mt-2">
                                    {assay.targetName && (
                                      <Badge variant="outline" className="text-xs">
                                        <Target className="w-3 h-3 mr-1" />
                                        {assay.targetName}
                                      </Badge>
                                    )}
                                    <span className="text-xs text-muted-foreground">
                                      Confidence: {(assay.confidence * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                          {(template.recommendedAssays[category] || []).length === 0 && (
                            <div className="text-center py-8 text-muted-foreground">
                              No {config.label.toLowerCase()} assays found
                            </div>
                          )}
                        </div>
                      </ScrollArea>
                    </TabsContent>
                  ))}
                </Tabs>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Disease Targets
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {template.targets.map((target) => (
                    <div
                      key={target.symbol}
                      className="flex items-center justify-between p-2 rounded border"
                    >
                      <div>
                        <div className="font-medium text-sm">{target.symbol}</div>
                        <div className="text-xs text-muted-foreground">{target.name}</div>
                      </div>
                      <Badge
                        variant={target.role === "primary" ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {target.role}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Selection Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(categoryConfig).map(([category, config]) => {
                    const categoryAssays = template.recommendedAssays[category] || [];
                    const selectedCount = categoryAssays.filter(a => selectedAssays.has(a.id)).length;
                    return (
                      <div key={category} className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <config.icon className="w-4 h-4" />
                          <span>{config.label}</span>
                        </div>
                        <Badge variant="outline">
                          {selectedCount} / {categoryAssays.length}
                        </Badge>
                      </div>
                    );
                  })}
                  <div className="border-t pt-2 mt-2">
                    <div className="flex items-center justify-between font-medium">
                      <span>Total Selected</span>
                      <Badge>{selectedAssays.size}</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Generate Protocol</CardTitle>
                <CardDescription>
                  Create an experiment protocol from selected assays
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Label htmlFor="campaign-name">Campaign Name</Label>
                  <Input
                    id="campaign-name"
                    placeholder={`${template.disease} Discovery Campaign`}
                    value={campaignName}
                    onChange={(e) => setCampaignName(e.target.value)}
                    data-testid="input-campaign-name"
                  />
                </div>
                <Button
                  className="w-full"
                  onClick={() => {
                    protocolMutation.mutate();
                    setProtocolDialogOpen(true);
                  }}
                  disabled={selectedAssays.size === 0 || protocolMutation.isPending}
                  data-testid="button-generate-protocol"
                >
                  <FileJson className="w-4 h-4 mr-2" />
                  {protocolMutation.isPending ? "Generating..." : "Generate Protocol"}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        <Card className="p-8 text-center">
          <CardDescription>Select a disease to view recommended assays</CardDescription>
        </Card>
      )}

      <Dialog open={protocolDialogOpen} onOpenChange={setProtocolDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Check className="w-5 h-5 text-green-500" />
              Protocol Generated
            </DialogTitle>
            <DialogDescription>
              Your experiment protocol is ready for export
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 overflow-auto">
            {generatedProtocol && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Campaign</Label>
                    <p className="font-medium">{generatedProtocol.name}</p>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Disease</Label>
                    <p className="font-medium">{generatedProtocol.disease}</p>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Therapeutic Area</Label>
                    <p className="font-medium">{generatedProtocol.therapeuticArea}</p>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Total Assays</Label>
                    <p className="font-medium">{generatedProtocol.assays?.length || 0}</p>
                  </div>
                </div>
                <div className="border rounded p-3 bg-muted/30">
                  <div className="text-xs font-medium mb-2">Workflow Stages</div>
                  <div className="space-y-2">
                    {generatedProtocol.workflow &&
                      Object.entries(generatedProtocol.workflow).map(([key, stage]: [string, any]) => (
                        <div key={key} className="flex items-center gap-2 text-sm">
                          <Badge variant="outline">{key.toUpperCase()}</Badge>
                          <span>{stage.name}</span>
                          <span className="text-muted-foreground">
                            ({stage.assays?.length || 0} assays)
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground mb-2 block">Protocol JSON</Label>
                  <ScrollArea className="h-[200px] border rounded p-3 bg-muted/30">
                    <pre className="text-xs whitespace-pre-wrap">
                      {JSON.stringify(generatedProtocol, null, 2)}
                    </pre>
                  </ScrollArea>
                </div>
              </div>
            )}
          </div>
          <div className="flex justify-end gap-2 pt-4 border-t">
            <Button variant="outline" onClick={copyProtocolToClipboard} data-testid="button-copy-protocol">
              <ClipboardCopy className="w-4 h-4 mr-2" />
              Copy to Clipboard
            </Button>
            <Button onClick={downloadProtocol} data-testid="button-download-protocol">
              <Download className="w-4 h-4 mr-2" />
              Download JSON
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
