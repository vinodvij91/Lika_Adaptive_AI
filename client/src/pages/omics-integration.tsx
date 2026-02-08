import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import {
  Dna,
  Activity,
  Loader2,
  Brain,
  BarChart3,
  Sparkles,
  Info,
  Search,
  ArrowUpDown,
  FlaskConical,
  Microscope,
  Atom,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface OmicsTarget {
  disease_id: string;
  target_id: string;
  integrated_score: number;
  gwas_score: number;
  variant_count: number;
  pathogenic_variants: number;
  tx_log2_fc: number;
  tx_p_value: number;
  tx_tissues: string;
  protein_fc: number;
  detection_evidence: string;
  ptm_flags: string;
  metab_pathway_score: number;
}

interface OmicsBundleData {
  context: {
    disease_or_indication: string;
    vaccine_or_therapeutic: string;
  };
  omics_table: OmicsTarget[];
  sequence_properties?: Record<string, Record<string, number>>;
  ui_text?: {
    panel_title?: string;
    panel_subtitle?: string;
    tooltips?: {
      table?: string;
      genomics?: string;
      transcriptomics?: string;
      proteomics?: string;
      metabolomics?: string;
    };
    narrative?: string;
  };
}

type SortField = "integrated_score" | "gwas_score" | "tx_log2_fc" | "protein_fc" | "metab_pathway_score";

function ScoreBar({ value, maxVal, color }: { value: number; maxVal: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (Math.abs(value) / maxVal) * 100));
  return (
    <div className="flex-1 bg-muted rounded-full h-3 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function OmicsLayerBars({ target }: { target: OmicsTarget }) {
  const layers = [
    { label: "Genomics", value: target.gwas_score, max: 1, color: "bg-blue-500 dark:bg-blue-400" },
    { label: "Transcriptomics", value: Math.abs(target.tx_log2_fc), max: 4, color: "bg-green-500 dark:bg-green-400" },
    { label: "Proteomics", value: target.protein_fc, max: 3, color: "bg-purple-500 dark:bg-purple-400" },
    { label: "Metabolomics", value: target.metab_pathway_score, max: 1, color: "bg-amber-500 dark:bg-amber-400" },
  ];

  return (
    <div className="space-y-3" data-testid={`omics-bars-${target.target_id}`}>
      {layers.map((l) => (
        <div key={l.label} className="space-y-1">
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs text-muted-foreground w-28">{l.label}</span>
            <span className="text-xs font-mono">{l.value.toFixed(2)}</span>
          </div>
          <ScoreBar value={l.value} maxVal={l.max} color={l.color} />
        </div>
      ))}
    </div>
  );
}

function IntegratedScoreBadge({ score }: { score: number }) {
  const pct = (score * 100).toFixed(0);
  let variant: "default" | "secondary" | "outline" = "secondary";
  if (score >= 0.7) variant = "default";
  else if (score >= 0.5) variant = "secondary";
  else variant = "outline";

  return (
    <Badge variant={variant} className="font-mono text-xs" data-testid={`badge-score-${pct}`}>
      {pct}%
    </Badge>
  );
}

const EXAMPLE_DISEASES = [
  { label: "Inflammatory Disease", value: "Inflammatory disease" },
  { label: "COVID-19", value: "COVID-19" },
  { label: "Breast Cancer", value: "Breast cancer" },
  { label: "Alzheimer's Disease", value: "Alzheimer disease" },
  { label: "Type 2 Diabetes", value: "Type 2 diabetes" },
  { label: "Rheumatoid Arthritis", value: "Rheumatoid arthritis" },
  { label: "Influenza A", value: "Influenza A" },
  { label: "HIV-1", value: "HIV-1" },
];

const DEFAULT_TARGETS = ["TNF", "IL6", "FCGR2A", "JAK2", "STAT3", "EGFR", "BRAF", "TP53", "VEGFA", "KRAS"];

export default function OmicsIntegrationPage() {
  const { toast } = useToast();
  const [disease, setDisease] = useState("Inflammatory disease");
  const [targetInput, setTargetInput] = useState(DEFAULT_TARGETS.join(", "));
  const [pipelineType, setPipelineType] = useState("therapeutic_antibody");
  const [sortField, setSortField] = useState<SortField>("integrated_score");
  const [sortAsc, setSortAsc] = useState(false);
  const [filterSearch, setFilterSearch] = useState("");
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);

  const bundleMutation = useMutation({
    mutationFn: async () => {
      const targets = targetInput.split(",").map(t => t.trim()).filter(Boolean);
      const res = await apiRequest("POST", "/api/omics/bundle", {
        disease_or_indication: disease,
        target_ids: targets,
        vaccine_or_therapeutic: pipelineType,
      });
      return res.json() as Promise<OmicsBundleData>;
    },
    onSuccess: () => {
      toast({ title: "Omics Bundle Generated", description: "Multi-omics evidence computed for all targets" });
    },
    onError: (error: any) => {
      toast({ title: "Omics Error", description: error.message, variant: "destructive" });
    },
  });

  const bundle = bundleMutation.data;
  const omicsTable = bundle?.omics_table || [];

  const sortedFiltered = useMemo(() => {
    let rows = [...omicsTable];
    if (filterSearch) {
      const q = filterSearch.toLowerCase();
      rows = rows.filter(r => r.target_id.toLowerCase().includes(q));
    }
    rows.sort((a, b) => {
      const av = a[sortField];
      const bv = b[sortField];
      return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
    return rows;
  }, [omicsTable, filterSearch, sortField, sortAsc]);

  const selectedData = selectedTarget
    ? omicsTable.find(t => t.target_id === selectedTarget)
    : null;

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <TableHead
      className="cursor-pointer select-none"
      onClick={() => handleSort(field)}
      data-testid={`sort-${field}`}
    >
      <div className="flex items-center gap-1">
        {children}
        {sortField === field && (
          <ArrowUpDown className="h-3 w-3" />
        )}
      </div>
    </TableHead>
  );

  return (
    <div className="flex flex-col h-full">
      <div className="border-b p-4">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <Dna className="h-5 w-5 text-emerald-500" />
            <h1 className="text-xl font-semibold" data-testid="text-page-title">Omics Integration</h1>
          </div>
          <Badge variant="outline" className="text-xs">Disease-Agnostic</Badge>
          <Badge variant="secondary" className="text-xs">CPU + Optional GPU</Badge>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Multi-omics evidence aggregation for drug discovery and vaccine pipelines
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        <Tabs defaultValue="analysis" className="w-full">
          <TabsList data-testid="omics-tabs">
            <TabsTrigger value="analysis" data-testid="tab-analysis">
              <BarChart3 className="h-4 w-4 mr-2" />
              Analysis
            </TabsTrigger>
            <TabsTrigger value="detail" data-testid="tab-detail">
              <Microscope className="h-4 w-4 mr-2" />
              Target Detail
            </TabsTrigger>
            <TabsTrigger value="enrichment" data-testid="tab-enrichment">
              <Atom className="h-4 w-4 mr-2" />
              Enrichment
            </TabsTrigger>
          </TabsList>

          <TabsContent value="analysis" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FlaskConical className="h-5 w-5 text-emerald-500" />
                  Configure Omics Analysis
                </CardTitle>
                <CardDescription>
                  Select a disease or indication and specify target genes/proteins to analyze
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label>Disease / Indication</Label>
                    <Select value={disease} onValueChange={setDisease}>
                      <SelectTrigger data-testid="select-disease">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {EXAMPLE_DISEASES.map(d => (
                          <SelectItem key={d.value} value={d.value}>{d.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Pipeline Type</Label>
                    <Select value={pipelineType} onValueChange={setPipelineType}>
                      <SelectTrigger data-testid="select-pipeline-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="therapeutic_antibody">Therapeutic Antibody</SelectItem>
                        <SelectItem value="small_molecule">Small Molecule</SelectItem>
                        <SelectItem value="vaccine">Vaccine</SelectItem>
                        <SelectItem value="gene_therapy">Gene Therapy</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Target Genes / Proteins</Label>
                    <Input
                      value={targetInput}
                      onChange={(e) => setTargetInput(e.target.value)}
                      placeholder="TNF, IL6, JAK2..."
                      data-testid="input-targets"
                    />
                  </div>
                </div>
                <Button
                  onClick={() => bundleMutation.mutate()}
                  disabled={bundleMutation.isPending || !disease || !targetInput.trim()}
                  data-testid="button-run-omics"
                >
                  {bundleMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Activity className="h-4 w-4 mr-2" />
                  )}
                  Run Omics Analysis
                </Button>
              </CardContent>
            </Card>

            {bundle?.ui_text?.narrative && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-amber-500" />
                    {bundle.ui_text.panel_title || "AI-Generated Omics Guidance"}
                  </CardTitle>
                  {bundle.ui_text.panel_subtitle && (
                    <CardDescription>{bundle.ui_text.panel_subtitle}</CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground" data-testid="text-narrative">
                    {bundle.ui_text.narrative}
                  </p>
                </CardContent>
              </Card>
            )}

            {omicsTable.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-blue-500" />
                    Multi-Omics Target Table
                    {bundle?.ui_text?.tooltips?.table && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          <p className="text-xs">{bundle.ui_text.tooltips.table}</p>
                        </TooltipContent>
                      </Tooltip>
                    )}
                  </CardTitle>
                  <CardDescription>
                    {bundle?.context.disease_or_indication} — {sortedFiltered.length} targets ranked by integrated omics score
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2 mb-4">
                    <Search className="h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Filter targets..."
                      value={filterSearch}
                      onChange={(e) => setFilterSearch(e.target.value)}
                      className="max-w-xs"
                      data-testid="input-filter-targets"
                    />
                  </div>
                  <div className="overflow-x-auto">
                    <Table data-testid="omics-table">
                      <TableHeader>
                        <TableRow>
                          <TableHead className="min-w-[100px]">Target</TableHead>
                          <SortHeader field="integrated_score">Score</SortHeader>
                          <SortHeader field="gwas_score">
                            <span className="flex items-center gap-1">
                              Genomics
                              {bundle?.ui_text?.tooltips?.genomics && (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Info className="h-3 w-3 text-muted-foreground" />
                                  </TooltipTrigger>
                                  <TooltipContent className="max-w-xs">
                                    <p className="text-xs">{bundle.ui_text.tooltips.genomics}</p>
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </span>
                          </SortHeader>
                          <SortHeader field="tx_log2_fc">
                            <span className="flex items-center gap-1">
                              Tx (log2FC)
                              {bundle?.ui_text?.tooltips?.transcriptomics && (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Info className="h-3 w-3 text-muted-foreground" />
                                  </TooltipTrigger>
                                  <TooltipContent className="max-w-xs">
                                    <p className="text-xs">{bundle.ui_text.tooltips.transcriptomics}</p>
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </span>
                          </SortHeader>
                          <SortHeader field="protein_fc">
                            <span className="flex items-center gap-1">
                              Proteomics
                              {bundle?.ui_text?.tooltips?.proteomics && (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Info className="h-3 w-3 text-muted-foreground" />
                                  </TooltipTrigger>
                                  <TooltipContent className="max-w-xs">
                                    <p className="text-xs">{bundle.ui_text.tooltips.proteomics}</p>
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </span>
                          </SortHeader>
                          <SortHeader field="metab_pathway_score">
                            <span className="flex items-center gap-1">
                              Metabolomics
                              {bundle?.ui_text?.tooltips?.metabolomics && (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Info className="h-3 w-3 text-muted-foreground" />
                                  </TooltipTrigger>
                                  <TooltipContent className="max-w-xs">
                                    <p className="text-xs">{bundle.ui_text.tooltips.metabolomics}</p>
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </span>
                          </SortHeader>
                          <TableHead>Evidence</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedFiltered.map((t, idx) => {
                          let tissues: string[] = [];
                          let ptms: string[] = [];
                          try { tissues = JSON.parse(t.tx_tissues); } catch {}
                          try { ptms = JSON.parse(t.ptm_flags); } catch {}
                          return (
                            <TableRow
                              key={t.target_id}
                              className="cursor-pointer"
                              onClick={() => setSelectedTarget(t.target_id)}
                              data-testid={`row-target-${t.target_id}`}
                            >
                              <TableCell className="font-medium font-mono">
                                <div className="flex items-center gap-2">
                                  <span className="text-xs text-muted-foreground w-5">{idx + 1}</span>
                                  {t.target_id}
                                </div>
                              </TableCell>
                              <TableCell>
                                <IntegratedScoreBadge score={t.integrated_score} />
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <ScoreBar value={t.gwas_score} maxVal={1} color="bg-blue-500 dark:bg-blue-400" />
                                  <span className="text-xs font-mono w-10 text-right">{t.gwas_score.toFixed(2)}</span>
                                </div>
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <span className={`text-xs font-mono ${t.tx_log2_fc > 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                                    {t.tx_log2_fc > 0 ? "+" : ""}{t.tx_log2_fc.toFixed(2)}
                                  </span>
                                  {tissues.length > 0 && (
                                    <Badge variant="outline" className="text-[10px]">{tissues[0]}</Badge>
                                  )}
                                </div>
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <ScoreBar value={t.protein_fc} maxVal={3} color="bg-purple-500 dark:bg-purple-400" />
                                  <span className="text-xs font-mono w-10 text-right">{t.protein_fc.toFixed(2)}</span>
                                </div>
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <ScoreBar value={t.metab_pathway_score} maxVal={1} color="bg-amber-500 dark:bg-amber-400" />
                                  <span className="text-xs font-mono w-10 text-right">{t.metab_pathway_score.toFixed(2)}</span>
                                </div>
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-1.5 flex-wrap">
                                  <Badge variant="secondary" className="text-[10px]">
                                    {t.variant_count} var
                                  </Badge>
                                  <Badge variant="outline" className="text-[10px]">
                                    {t.detection_evidence}
                                  </Badge>
                                  {ptms.length > 0 && (
                                    <Badge variant="outline" className="text-[10px]">
                                      {ptms.length} PTM
                                    </Badge>
                                  )}
                                </div>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedData && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Microscope className="h-5 w-5 text-cyan-500" />
                    Per-Target Detail: {selectedData.target_id}
                  </CardTitle>
                  <CardDescription>
                    Stacked omics layer contributions for {selectedData.target_id} in {bundle?.context.disease_or_indication}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <OmicsLayerBars target={selectedData} />
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <span className="text-xs font-semibold text-muted-foreground">Integrated Score</span>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-muted rounded-full h-5 overflow-hidden">
                            <div
                              className="h-full bg-emerald-500 dark:bg-emerald-400 rounded-full transition-all duration-500"
                              style={{ width: `${selectedData.integrated_score * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-sm font-semibold">{(selectedData.integrated_score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-muted-foreground">Pathogenic Variants</span>
                          <span className="font-mono">{selectedData.pathogenic_variants} / {selectedData.variant_count}</span>
                        </div>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-muted-foreground">Tx p-value</span>
                          <span className="font-mono">{selectedData.tx_p_value.toExponential(2)}</span>
                        </div>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-muted-foreground">Detection Evidence</span>
                          <Badge variant="outline" className="text-xs">{selectedData.detection_evidence}</Badge>
                        </div>
                        {(() => {
                          let tissues: string[] = [];
                          try { tissues = JSON.parse(selectedData.tx_tissues); } catch {}
                          return tissues.length > 0 ? (
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-muted-foreground">Tissues</span>
                              <div className="flex gap-1 flex-wrap">
                                {tissues.map(t => (
                                  <Badge key={t} variant="secondary" className="text-[10px]">{t}</Badge>
                                ))}
                              </div>
                            </div>
                          ) : null;
                        })()}
                        {(() => {
                          let ptms: string[] = [];
                          try { ptms = JSON.parse(selectedData.ptm_flags); } catch {}
                          return ptms.length > 0 ? (
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-muted-foreground">PTM Flags</span>
                              <div className="flex gap-1 flex-wrap">
                                {ptms.map(p => (
                                  <Badge key={p} variant="outline" className="text-[10px]">{p}</Badge>
                                ))}
                              </div>
                            </div>
                          ) : null;
                        })()}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {!bundleMutation.isPending && omicsTable.length === 0 && !bundleMutation.isError && (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                  <Dna className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold text-lg mb-2">No Omics Data Yet</h3>
                  <p className="text-sm text-muted-foreground max-w-md">
                    Configure your disease, targets, and pipeline type above, then click "Run Omics Analysis" to generate multi-omics evidence.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="detail" className="space-y-6">
            {omicsTable.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {omicsTable.map((t) => {
                  let tissues: string[] = [];
                  let ptms: string[] = [];
                  try { tissues = JSON.parse(t.tx_tissues); } catch {}
                  try { ptms = JSON.parse(t.ptm_flags); } catch {}
                  return (
                    <Card key={t.target_id} data-testid={`card-target-${t.target_id}`}>
                      <CardHeader className="pb-2">
                        <CardTitle className="flex items-center justify-between gap-2 text-base">
                          <span className="font-mono">{t.target_id}</span>
                          <IntegratedScoreBadge score={t.integrated_score} />
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <OmicsLayerBars target={t} />
                        <div className="mt-3 flex gap-1 flex-wrap">
                          {tissues.map(ti => (
                            <Badge key={ti} variant="secondary" className="text-[10px]">{ti}</Badge>
                          ))}
                          {ptms.map(p => (
                            <Badge key={p} variant="outline" className="text-[10px]">{p}</Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            ) : (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                  <Microscope className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold text-lg mb-2">Run Analysis First</h3>
                  <p className="text-sm text-muted-foreground">
                    Go to the Analysis tab and run an omics analysis to see per-target detail cards.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="enrichment" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-500" />
                  BioNeMo Sequence Enrichment
                  <Badge variant="outline" className="text-xs">GPU Optional</Badge>
                </CardTitle>
                <CardDescription>
                  Enrich omics evidence with structure-/sequence-based properties from BioNeMo (stability, disorder, aggregation)
                </CardDescription>
              </CardHeader>
              <CardContent>
                {bundle?.sequence_properties && Object.keys(bundle.sequence_properties).length > 0 ? (
                  <div className="space-y-4">
                    {Object.entries(bundle.sequence_properties).map(([tid, props]) => (
                      <div key={tid} className="space-y-2">
                        <span className="text-sm font-medium font-mono">{tid}</span>
                        <div className="grid grid-cols-3 gap-3">
                          {Object.entries(props).map(([prop, val]) => (
                            <div key={prop} className="space-y-1">
                              <span className="text-xs text-muted-foreground capitalize">{prop}</span>
                              <ScoreBar value={val as number} maxVal={1} color="bg-indigo-500 dark:bg-indigo-400" />
                              <span className="text-xs font-mono">{(val as number).toFixed(3)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Atom className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground">
                      No BioNeMo enrichment data available. Provide protein sequences and ensure BIONEMO_API_KEY is configured to enable sequence property prediction.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
