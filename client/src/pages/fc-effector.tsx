import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import {
  Shield,
  Activity,
  Loader2,
  Dna,
  Brain,
  BarChart3,
  Sparkles,
  Info,
  Cpu,
  Zap,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface AtlasData {
  cell_types: string[];
  receptors: string[];
  matrix: number[][];
}

interface VariantData {
  variant_id: string;
  ADCC_score: number;
  CDC_score: number;
}

interface SpeciesData {
  model_id: string;
  label: string;
  similarity_to_human_NK_FcR: number;
}

interface BundleData {
  context: {
    disease_or_indication: string;
    vaccine_or_therapeutic: string;
  };
  atlas: AtlasData;
  variants: VariantData[];
  species_similarity: SpeciesData[];
  ui_text?: {
    panel_title?: string;
    panel_subtitle?: string;
    tooltips?: {
      atlas?: string;
      effector?: string;
      species?: string;
    };
    narrative?: string;
  };
  bionemo_affinities?: Record<string, number>;
}

function ExpressionHeatmap({ atlas }: { atlas: AtlasData }) {
  const getColor = (value: number) => {
    if (value >= 0.8) return "bg-red-500 dark:bg-red-600 text-white";
    if (value >= 0.6) return "bg-orange-400 dark:bg-orange-500 text-white";
    if (value >= 0.4) return "bg-yellow-300 dark:bg-yellow-500 text-black dark:text-white";
    if (value >= 0.2) return "bg-blue-200 dark:bg-blue-700 text-black dark:text-white";
    return "bg-blue-50 dark:bg-blue-900 text-muted-foreground";
  };

  return (
    <div className="overflow-x-auto" data-testid="atlas-heatmap">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="font-semibold min-w-[160px]">Cell Type</TableHead>
            {atlas.receptors.map((r) => (
              <TableHead key={r} className="text-center font-mono text-xs min-w-[90px]">
                {r}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {atlas.cell_types.map((cellType, i) => (
            <TableRow key={cellType}>
              <TableCell className="font-medium text-sm">{cellType}</TableCell>
              {atlas.matrix[i].map((val, j) => (
                <TableCell key={j} className="p-1 text-center">
                  <div
                    className={`rounded-md px-2 py-1.5 text-xs font-mono font-semibold ${getColor(val)}`}
                    data-testid={`cell-${i}-${j}`}
                  >
                    {val.toFixed(1)}
                  </div>
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function VariantScoreChart({ variants }: { variants: VariantData[] }) {
  return (
    <div className="space-y-4" data-testid="variant-scores">
      {variants.map((v) => (
        <div key={v.variant_id} className="space-y-2">
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm font-medium min-w-[160px]">{v.variant_id}</span>
            <div className="flex gap-2">
              <Badge variant="secondary" className="font-mono text-xs">
                ADCC {(v.ADCC_score * 100).toFixed(0)}%
              </Badge>
              <Badge variant="outline" className="font-mono text-xs">
                CDC {(v.CDC_score * 100).toFixed(0)}%
              </Badge>
            </div>
          </div>
          <div className="flex gap-2 items-center">
            <span className="text-xs text-muted-foreground w-10">ADCC</span>
            <div className="flex-1 bg-muted rounded-full h-3 overflow-hidden">
              <div
                className="h-full bg-cyan-500 dark:bg-cyan-400 rounded-full transition-all duration-500"
                style={{ width: `${v.ADCC_score * 100}%` }}
              />
            </div>
          </div>
          <div className="flex gap-2 items-center">
            <span className="text-xs text-muted-foreground w-10">CDC</span>
            <div className="flex-1 bg-muted rounded-full h-3 overflow-hidden">
              <div
                className="h-full bg-purple-500 dark:bg-purple-400 rounded-full transition-all duration-500"
                style={{ width: `${v.CDC_score * 100}%` }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function SpeciesSimilarityChart({ species }: { species: SpeciesData[] }) {
  return (
    <div className="space-y-4" data-testid="species-similarity">
      {species.map((s) => {
        const pct = (s.similarity_to_human_NK_FcR * 100).toFixed(1);
        const isHigh = s.similarity_to_human_NK_FcR >= 0.95;
        return (
          <div key={s.model_id} className="space-y-2">
            <div className="flex items-center justify-between gap-4">
              <span className="text-sm font-medium">{s.label}</span>
              <Badge variant={isHigh ? "default" : "secondary"} className="font-mono text-xs">
                {pct}% match
              </Badge>
            </div>
            <div className="flex-1 bg-muted rounded-full h-4 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isHigh
                    ? "bg-green-500 dark:bg-green-400"
                    : "bg-amber-500 dark:bg-amber-400"
                }`}
                style={{ width: `${s.similarity_to_human_NK_FcR * 100}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

const FC_DISEASE_INDICATIONS = [
  { group: "Oncology", items: [
    { label: "Breast Cancer (HER2+)", value: "Breast cancer HER2+" },
    { label: "Non-Small Cell Lung Cancer", value: "Non-small cell lung cancer" },
    { label: "Colorectal Cancer", value: "Colorectal cancer" },
    { label: "B-Cell Lymphoma", value: "B-cell lymphoma" },
    { label: "Multiple Myeloma", value: "Multiple myeloma" },
    { label: "Melanoma", value: "Melanoma" },
  ]},
  { group: "Autoimmune & Inflammatory", items: [
    { label: "Rheumatoid Arthritis", value: "Rheumatoid arthritis" },
    { label: "Systemic Lupus Erythematosus", value: "Systemic lupus erythematosus" },
    { label: "Crohn's Disease", value: "Crohn's disease" },
    { label: "Multiple Sclerosis", value: "Multiple sclerosis" },
    { label: "Psoriasis", value: "Psoriasis" },
  ]},
  { group: "Infectious Disease (Vaccines)", items: [
    { label: "COVID-19 / SARS-CoV-2", value: "COVID-19" },
    { label: "Influenza A/B", value: "Influenza" },
    { label: "RSV", value: "RSV" },
    { label: "HIV-1", value: "HIV-1" },
    { label: "Malaria (P. falciparum)", value: "Malaria" },
    { label: "Ebola Virus", value: "Ebola" },
    { label: "Nipah Virus", value: "Nipah" },
    { label: "Yellow Fever", value: "Yellow fever" },
    { label: "HPV", value: "HPV" },
    { label: "TB / BCG", value: "Tuberculosis" },
  ]},
  { group: "Hematology", items: [
    { label: "Hemophilia A", value: "Hemophilia A" },
    { label: "Paroxysmal Nocturnal Hemoglobinuria", value: "PNH" },
    { label: "Immune Thrombocytopenia", value: "Immune thrombocytopenia" },
  ]},
  { group: "Neurology", items: [
    { label: "Alzheimer's Disease", value: "Alzheimer disease" },
    { label: "Myasthenia Gravis", value: "Myasthenia gravis" },
    { label: "Migraine (CGRP)", value: "Migraine" },
  ]},
  { group: "Other", items: [
    { label: "Asthma (Severe Eosinophilic)", value: "Severe eosinophilic asthma" },
    { label: "Atopic Dermatitis", value: "Atopic dermatitis" },
    { label: "Osteoporosis", value: "Osteoporosis" },
    { label: "General / Custom", value: "general" },
  ]},
];

export default function FcEffectorPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  const [disease, setDisease] = useState("COVID-19");
  const [therapeuticType, setTherapeuticType] = useState("therapeutic_antibody");
  const [fcSequence, setFcSequence] = useState("");
  const [includeOpenAI, setIncludeOpenAI] = useState(true);
  const [includeBioNeMo, setIncludeBioNeMo] = useState(true);

  const { data: atlasData, isLoading: atlasLoading } = useQuery<AtlasData>({
    queryKey: ["/api/fc-effector/atlas"],
  });

  const { data: variantsData, isLoading: variantsLoading } = useQuery<VariantData[]>({
    queryKey: ["/api/fc-effector/variants"],
  });

  const { data: speciesData, isLoading: speciesLoading } = useQuery<SpeciesData[]>({
    queryKey: ["/api/fc-effector/species-similarity"],
  });

  const bundleMutation = useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      const response = await apiRequest("POST", "/api/fc-effector/bundle", payload);
      return response as unknown as BundleData;
    },
    onSuccess: (data) => {
      toast({
        title: "Fc Bundle Generated",
        description: data.ui_text?.panel_title || "Analysis complete",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Generation Failed",
        description: error.message || "Failed to generate Fc bundle",
        variant: "destructive",
      });
    },
  });

  const handleGenerateBundle = () => {
    bundleMutation.mutate({
      disease_or_indication: disease,
      vaccine_or_therapeutic: therapeuticType,
      fc_sequence: fcSequence || null,
      include_openai: includeOpenAI,
      include_bionemo: includeBioNeMo,
    });
  };

  const bundleResult = bundleMutation.data;

  return (
    <div className="flex flex-col h-full" data-testid="fc-effector-page">
      <div className="bg-gradient-to-r from-purple-600/10 via-indigo-600/10 to-cyan-600/10 dark:from-purple-900/30 dark:via-indigo-900/30 dark:to-cyan-900/30 border-b px-6 py-5">
        <div className="container mx-auto">
          <div className="flex items-center gap-3">
            <div className="bg-purple-500/20 dark:bg-purple-500/30 p-2.5 rounded-lg">
              <Shield className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold" data-testid="text-page-title">
                Fc Effector Modeling
              </h1>
              <p className="text-sm text-muted-foreground">
                FcgammaR/FcRn atlas, ADCC/CDC scoring, species translation, and AI-powered guidance
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6 flex-1 overflow-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5" data-testid="fc-tabs">
            <TabsTrigger value="overview" data-testid="tab-overview">
              <BarChart3 className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="atlas" data-testid="tab-atlas">
              <Activity className="h-4 w-4 mr-2" />
              Atlas
            </TabsTrigger>
            <TabsTrigger value="effector" data-testid="tab-effector">
              <Zap className="h-4 w-4 mr-2" />
              Effector
            </TabsTrigger>
            <TabsTrigger value="species" data-testid="tab-species">
              <Dna className="h-4 w-4 mr-2" />
              Species
            </TabsTrigger>
            <TabsTrigger value="generate" data-testid="tab-generate">
              <Sparkles className="h-4 w-4 mr-2" />
              Generate
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Cell Types</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" data-testid="text-cell-count">
                    {atlasLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : atlasData?.cell_types.length || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">Immune cell types profiled</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Fc Receptors</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" data-testid="text-receptor-count">
                    {atlasLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : atlasData?.receptors.length || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">FcgammaR + FcRn tracked</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Fc Variants</CardTitle>
                  <Zap className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" data-testid="text-variant-count">
                    {variantsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : variantsData?.length || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">ADCC/CDC scored variants</p>
                </CardContent>
              </Card>
            </div>

            {atlasData && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    FcgammaR / FcRn Expression Atlas
                  </CardTitle>
                  <CardDescription>
                    Relative expression levels across key immune cell types (0 = absent, 1 = high)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ExpressionHeatmap atlas={atlasData} />
                </CardContent>
              </Card>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {variantsData && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="h-5 w-5" />
                      ADCC / CDC Scores
                    </CardTitle>
                    <CardDescription>
                      Predicted effector function scores for Fc variants
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <VariantScoreChart variants={variantsData} />
                  </CardContent>
                </Card>
              )}
              {speciesData && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Dna className="h-5 w-5" />
                      Species Translation
                    </CardTitle>
                    <CardDescription>
                      FcgammaR profile cosine similarity to human NK cells
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <SpeciesSimilarityChart species={speciesData} />
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="atlas" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Human FcgammaR / FcRn Expression Atlas
                </CardTitle>
                <CardDescription>
                  Simplified expression atlas showing relative receptor levels on major immune cell populations.
                  Values represent normalised expression from 0 (absent) to 1 (high).
                </CardDescription>
              </CardHeader>
              <CardContent>
                {atlasLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : atlasData ? (
                  <ExpressionHeatmap atlas={atlasData} />
                ) : (
                  <p className="text-muted-foreground text-center py-8">Failed to load atlas data</p>
                )}
              </CardContent>
            </Card>

            {atlasData && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Legend</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-3">
                    {[
                      { label: "High (0.8-1.0)", cls: "bg-red-500 text-white" },
                      { label: "Medium-High (0.6-0.8)", cls: "bg-orange-400 text-white" },
                      { label: "Medium (0.4-0.6)", cls: "bg-yellow-300 text-black" },
                      { label: "Low (0.2-0.4)", cls: "bg-blue-200 text-black" },
                      { label: "Minimal (0-0.2)", cls: "bg-blue-50 dark:bg-blue-900 text-muted-foreground" },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center gap-1.5">
                        <div className={`w-4 h-4 rounded ${item.cls}`} />
                        <span className="text-xs text-muted-foreground">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="effector" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Fc Variant ADCC / CDC Scores
                </CardTitle>
                <CardDescription>
                  Predicted antibody-dependent cellular cytotoxicity (ADCC) and complement-dependent
                  cytotoxicity (CDC) scores for common Fc engineering variants.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {variantsLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : variantsData ? (
                  <VariantScoreChart variants={variantsData} />
                ) : (
                  <p className="text-muted-foreground text-center py-8">Failed to load variant data</p>
                )}
              </CardContent>
            </Card>

            {variantsData && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Variant Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Variant</TableHead>
                        <TableHead className="text-center">ADCC Score</TableHead>
                        <TableHead className="text-center">CDC Score</TableHead>
                        <TableHead className="text-center">Primary Use</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {variantsData.map((v) => (
                        <TableRow key={v.variant_id}>
                          <TableCell className="font-medium">{v.variant_id}</TableCell>
                          <TableCell className="text-center font-mono">{v.ADCC_score.toFixed(2)}</TableCell>
                          <TableCell className="text-center font-mono">{v.CDC_score.toFixed(2)}</TableCell>
                          <TableCell className="text-center">
                            <Badge variant="secondary" className="text-xs">
                              {v.ADCC_score > v.CDC_score ? "ADCC-driven" : "CDC-driven"}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="species" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Dna className="h-5 w-5" />
                  Species Translation Similarity
                </CardTitle>
                <CardDescription>
                  Cosine similarity of FcgammaR expression profiles between animal models and human NK cells.
                  Higher similarity means better translation of preclinical Fc-dependent efficacy data.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {speciesLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : speciesData ? (
                  <SpeciesSimilarityChart species={speciesData} />
                ) : (
                  <p className="text-muted-foreground text-center py-8">Failed to load species data</p>
                )}
              </CardContent>
            </Card>

            {speciesData && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Translation Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Animal Model</TableHead>
                        <TableHead className="text-center">Similarity to Human NK</TableHead>
                        <TableHead className="text-center">Translatability</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {speciesData.map((s) => (
                        <TableRow key={s.model_id}>
                          <TableCell className="font-medium">{s.label}</TableCell>
                          <TableCell className="text-center font-mono">
                            {(s.similarity_to_human_NK_FcR * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell className="text-center">
                            <Badge
                              variant={s.similarity_to_human_NK_FcR >= 0.95 ? "default" : "secondary"}
                              className="text-xs"
                            >
                              {s.similarity_to_human_NK_FcR >= 0.95 ? "Excellent" : "Moderate"}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="generate" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Generate Fc Bundle
                  </CardTitle>
                  <CardDescription>
                    Build a complete Fc effector analysis bundle with optional BioNeMo affinity prediction
                    and AI-generated guidance text.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="disease">Disease or Indication</Label>
                    <Select value={disease} onValueChange={setDisease}>
                      <SelectTrigger data-testid="select-disease">
                        <SelectValue placeholder="Select disease or indication" />
                      </SelectTrigger>
                      <SelectContent>
                        {FC_DISEASE_INDICATIONS.map((group) => (
                          <div key={group.group}>
                            <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">{group.group}</div>
                            {group.items.map((item) => (
                              <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                            ))}
                          </div>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="therapeutic-type">Therapeutic Type</Label>
                    <Select value={therapeuticType} onValueChange={setTherapeuticType}>
                      <SelectTrigger data-testid="select-therapeutic-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="therapeutic_antibody">Therapeutic Antibody</SelectItem>
                        <SelectItem value="vaccine">Vaccine</SelectItem>
                        <SelectItem value="bispecific">Bispecific Antibody</SelectItem>
                        <SelectItem value="adc">Antibody-Drug Conjugate</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="fc-sequence">Fc Sequence (optional)</Label>
                    <Textarea
                      id="fc-sequence"
                      data-testid="input-fc-sequence"
                      placeholder="Paste Fc amino acid sequence for BioNeMo affinity prediction..."
                      value={fcSequence}
                      onChange={(e) => setFcSequence(e.target.value)}
                      className="font-mono text-xs resize-none"
                      rows={4}
                    />
                  </div>

                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                      <Switch
                        id="include-openai"
                        checked={includeOpenAI}
                        onCheckedChange={setIncludeOpenAI}
                        data-testid="switch-include-openai"
                      />
                      <Label htmlFor="include-openai" className="text-sm flex items-center gap-1">
                        <Brain className="h-3.5 w-3.5" />
                        AI Guidance
                      </Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        id="include-bionemo"
                        checked={includeBioNeMo}
                        onCheckedChange={setIncludeBioNeMo}
                        data-testid="switch-include-bionemo"
                      />
                      <Label htmlFor="include-bionemo" className="text-sm flex items-center gap-1">
                        <Cpu className="h-3.5 w-3.5" />
                        BioNeMo
                      </Label>
                    </div>
                  </div>

                  <Button
                    onClick={handleGenerateBundle}
                    disabled={bundleMutation.isPending}
                    className="w-full"
                    data-testid="button-generate-bundle"
                  >
                    {bundleMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    {bundleMutation.isPending ? "Generating..." : "Generate Fc Bundle"}
                  </Button>
                </CardContent>
              </Card>

              {bundleResult ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {bundleResult.ui_text?.panel_title ? (
                        <>
                          <Brain className="h-5 w-5 text-purple-500" />
                          {bundleResult.ui_text.panel_title}
                        </>
                      ) : (
                        <>
                          <BarChart3 className="h-5 w-5" />
                          Bundle Results
                        </>
                      )}
                    </CardTitle>
                    {bundleResult.ui_text?.panel_subtitle && (
                      <CardDescription>{bundleResult.ui_text.panel_subtitle}</CardDescription>
                    )}
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs">
                          {bundleResult.context.disease_or_indication}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {bundleResult.context.vaccine_or_therapeutic}
                        </Badge>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-bold">{bundleResult.atlas.cell_types.length}</div>
                        <div className="text-xs text-muted-foreground">Cell Types</div>
                      </div>
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-bold">{bundleResult.variants.length}</div>
                        <div className="text-xs text-muted-foreground">Variants</div>
                      </div>
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-bold">{bundleResult.species_similarity.length}</div>
                        <div className="text-xs text-muted-foreground">Models</div>
                      </div>
                    </div>

                    {bundleResult.bionemo_affinities && Object.keys(bundleResult.bionemo_affinities).length > 0 && (
                      <div className="space-y-1">
                        <span className="text-sm font-medium flex items-center gap-1">
                          <Cpu className="h-3.5 w-3.5" />
                          BioNeMo Affinities
                        </span>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(bundleResult.bionemo_affinities).map(([receptor, score]) => (
                            <Badge key={receptor} variant="outline" className="font-mono text-xs">
                              {receptor}: {typeof score === "number" ? score.toFixed(3) : String(score)}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {bundleResult.ui_text?.narrative && (
                      <div className="bg-purple-500/5 dark:bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
                        <div className="flex items-start gap-2">
                          <Brain className="h-4 w-4 text-purple-500 mt-0.5 shrink-0" />
                          <p className="text-sm leading-relaxed">{bundleResult.ui_text.narrative}</p>
                        </div>
                      </div>
                    )}

                    {bundleResult.ui_text?.tooltips && (
                      <div className="space-y-2">
                        <span className="text-sm font-medium">Tooltips</span>
                        <div className="space-y-1.5">
                          {Object.entries(bundleResult.ui_text.tooltips).map(([key, value]) => (
                            <div key={key} className="flex items-start gap-2">
                              <Tooltip>
                                <TooltipTrigger>
                                  <Badge variant="outline" className="text-xs capitalize">{key}</Badge>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="max-w-xs text-xs">{String(value)}</p>
                                </TooltipContent>
                              </Tooltip>
                              <span className="text-xs text-muted-foreground">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-16 text-center">
                    <Info className="h-12 w-12 text-muted-foreground/50 mb-4" />
                    <h3 className="text-lg font-medium mb-1">No Bundle Generated</h3>
                    <p className="text-sm text-muted-foreground max-w-sm">
                      Configure your analysis parameters and click Generate to build a comprehensive
                      Fc effector bundle with optional AI guidance.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
