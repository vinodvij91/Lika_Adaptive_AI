import { useState, useMemo } from "react";
import { PageHeader } from "@/components/page-header";
import { ResultsPanel } from "@/components/results-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Factory,
  Gauge,
  AlertTriangle,
  FlaskConical,
  Rocket,
  Building2,
  TrendingUp,
  BarChart3,
  Settings,
  Thermometer,
  Layers,
  DollarSign,
  RefreshCw,
  Download,
  Zap,
} from "lucide-react";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

type ReadinessTier = "lab-only" | "pilot-ready" | "production-viable";

interface ManufacturabilityScore {
  id: string;
  materialName: string;
  overallScore: number;
  complexityScore: number;
  costProxy: number;
  scaleUpRisk: number;
  processVariationSensitivity: number;
  readinessTier: ReadinessTier;
  processParams: {
    temperature: string;
    pressure: string;
    steps: number;
  };
}

const TIER_CONFIG: Record<ReadinessTier, { label: string; icon: typeof FlaskConical; color: string }> = {
  "lab-only": { label: "Lab-Only", icon: FlaskConical, color: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30" },
  "pilot-ready": { label: "Pilot-Ready", icon: Rocket, color: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30" },
  "production-viable": { label: "Production-Viable", icon: Building2, color: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30" },
};

function generateMockScores(count: number): ManufacturabilityScore[] {
  const materialPrefixes = ["Poly", "Nano", "Bio", "Thermo", "Electro", "Hybrid"];
  const materialSuffixes = ["amide", "ester", "urethane", "carbonate", "oxide", "silicate", "composite"];

  return Array.from({ length: count }, (_, i) => {
    const seed = i * 137;
    const overallScore = Math.round(seededRandom(seed) * 100);
    const readinessTier: ReadinessTier = overallScore >= 70 ? "production-viable" : overallScore >= 40 ? "pilot-ready" : "lab-only";

    return {
      id: `mat-${i}`,
      materialName: `${materialPrefixes[i % materialPrefixes.length]}${materialSuffixes[i % materialSuffixes.length]}-${1000 + i}`,
      overallScore,
      complexityScore: Math.round(20 + seededRandom(seed + 1) * 80),
      costProxy: Math.round(10 + seededRandom(seed + 2) * 90),
      scaleUpRisk: Math.round(seededRandom(seed + 3) * 100),
      processVariationSensitivity: Math.round(seededRandom(seed + 4) * 100),
      readinessTier,
      processParams: {
        temperature: `${Math.round(50 + seededRandom(seed + 5) * 300)}°C`,
        pressure: `${(1 + seededRandom(seed + 6) * 9).toFixed(1)} bar`,
        steps: Math.round(2 + seededRandom(seed + 7) * 8),
      },
    };
  });
}

interface ScoreGaugeProps {
  label: string;
  value: number;
  icon: typeof Gauge;
  inverted?: boolean;
}

function ScoreGauge({ label, value, icon: Icon, inverted = false }: ScoreGaugeProps) {
  const displayValue = inverted ? 100 - value : value;
  const color = displayValue >= 70 ? "text-green-600 dark:text-green-400" : displayValue >= 40 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400";

  return (
    <div className="text-center">
      <div className="w-10 h-10 mx-auto rounded-full bg-muted/50 flex items-center justify-center mb-1">
        <Icon className={`h-5 w-5 ${color}`} />
      </div>
      <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

interface MaterialScoreCardProps {
  score: ManufacturabilityScore;
}

function MaterialScoreCard({ score }: MaterialScoreCardProps) {
  const tierConfig = TIER_CONFIG[score.readinessTier];
  const TierIcon = tierConfig.icon;

  const overallColor = score.overallScore >= 70 ? "text-green-600 dark:text-green-400" : score.overallScore >= 40 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400";

  return (
    <Card className="hover-elevate" data-testid={`card-material-${score.id}`}>
      <CardContent className="p-4 space-y-4">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="font-medium">{score.materialName}</div>
            <Badge variant="outline" className={`mt-1 ${tierConfig.color}`}>
              <TierIcon className="h-3 w-3 mr-1" />
              {tierConfig.label}
            </Badge>
          </div>
          <div className="text-right">
            <div className={`text-3xl font-bold font-mono ${overallColor}`}>{score.overallScore}</div>
            <div className="text-xs text-muted-foreground">Score</div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2">
          <ScoreGauge label="Complexity" value={score.complexityScore} icon={Layers} inverted />
          <ScoreGauge label="Cost" value={score.costProxy} icon={DollarSign} inverted />
          <ScoreGauge label="Scale Risk" value={score.scaleUpRisk} icon={AlertTriangle} inverted />
          <ScoreGauge label="Sensitivity" value={score.processVariationSensitivity} icon={Gauge} inverted />
        </div>

        <div className="pt-2 border-t">
          <div className="text-xs text-muted-foreground mb-2">Process Parameters</div>
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1">
              <Thermometer className="h-3 w-3 text-muted-foreground" />
              {score.processParams.temperature}
            </div>
            <div className="flex items-center gap-1">
              <Gauge className="h-3 w-3 text-muted-foreground" />
              {score.processParams.pressure}
            </div>
            <div className="flex items-center gap-1">
              <Layers className="h-3 w-3 text-muted-foreground" />
              {score.processParams.steps} steps
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ManufacturabilitySccoringPage() {
  const [tierFilter, setTierFilter] = useState<string>("all");
  const [minScore, setMinScore] = useState<number[]>([0]);
  const [showLowRiskOnly, setShowLowRiskOnly] = useState(false);

  const allScores = useMemo(() => generateMockScores(24), []);

  const totalVariants = 420000;
  const manufacturableCount = 12450;
  const pilotReadyCount = 45230;
  const labOnlyCount = totalVariants - manufacturableCount - pilotReadyCount;

  const filteredScores = useMemo(() => {
    let result = allScores;
    if (tierFilter !== "all") {
      result = result.filter(s => s.readinessTier === tierFilter);
    }
    if (minScore[0] > 0) {
      result = result.filter(s => s.overallScore >= minScore[0]);
    }
    if (showLowRiskOnly) {
      result = result.filter(s => s.scaleUpRisk < 40);
    }
    return result;
  }, [allScores, tierFilter, minScore, showLowRiskOnly]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Manufacturability Scoring" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" data-testid="button-recalculate">
              <RefreshCw className="h-4 w-4 mr-2" />
              Recalculate All
            </Button>
            <Button variant="outline" data-testid="button-export">
              <Download className="h-4 w-4 mr-2" />
              Export Scores
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6 rounded-lg border border-primary/20">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/30">
                <Factory className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-1">Manufacturability Scoring Engine</h2>
                <p className="text-muted-foreground">
                  Evaluate production feasibility at scale. Calculate <strong className="text-foreground">complexity</strong>,
                  <strong className="text-foreground"> cost proxy</strong>, <strong className="text-foreground">scale-up risk</strong>, 
                  and <strong className="text-foreground">process sensitivity</strong> for every variant.
                </p>
              </div>
            </div>
          </div>

          <Card className="bg-gradient-to-r from-green-500/5 via-green-500/10 to-green-500/5 border-green-500/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Building2 className="h-8 w-8 text-green-600 dark:text-green-400" />
                  <div>
                    <div className="text-sm text-muted-foreground">Production-Viable Variants</div>
                    <div className="text-2xl font-bold">
                      <span className="text-green-600 dark:text-green-400 font-mono">{formatNumber(manufacturableCount)}</span>
                      <span className="text-muted-foreground font-normal text-base ml-2">of {formatNumber(totalVariants)} variants are manufacturable</span>
                    </div>
                  </div>
                </div>
                <Button variant="default" data-testid="button-show-manufacturable">
                  <Zap className="h-4 w-4 mr-2" />
                  Show Only Production-Viable
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-md bg-green-500/10 flex items-center justify-center">
                    <Building2 className="h-5 w-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold font-mono text-green-600 dark:text-green-400">{formatNumber(manufacturableCount)}</div>
                    <div className="text-xs text-muted-foreground">Production-Viable</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-md bg-blue-500/10 flex items-center justify-center">
                    <Rocket className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold font-mono text-blue-600 dark:text-blue-400">{formatNumber(pilotReadyCount)}</div>
                    <div className="text-xs text-muted-foreground">Pilot-Ready</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-md bg-amber-500/10 flex items-center justify-center">
                    <FlaskConical className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">{formatNumber(labOnlyCount)}</div>
                    <div className="text-xs text-muted-foreground">Lab-Only</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
                    <TrendingUp className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold font-mono">{((manufacturableCount / totalVariants) * 100).toFixed(1)}%</div>
                    <div className="text-xs text-muted-foreground">Yield Rate</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Scoring Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Process Temperature Range</Label>
                  <Select defaultValue="ambient-300">
                    <SelectTrigger data-testid="select-temperature">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ambient-100">Ambient - 100°C</SelectItem>
                      <SelectItem value="ambient-300">Ambient - 300°C</SelectItem>
                      <SelectItem value="high-temp">High Temperature (&gt;300°C)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Pressure Constraints</Label>
                  <Select defaultValue="standard">
                    <SelectTrigger data-testid="select-pressure">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="vacuum">Vacuum Required</SelectItem>
                      <SelectItem value="standard">Standard (1-10 bar)</SelectItem>
                      <SelectItem value="high">High Pressure (&gt;10 bar)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Solvent Availability</Label>
                  <Select defaultValue="common">
                    <SelectTrigger data-testid="select-solvent">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="water">Water-Based Only</SelectItem>
                      <SelectItem value="common">Common Solvents</SelectItem>
                      <SelectItem value="specialty">Specialty Solvents</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex items-center justify-between pt-2 border-t">
                <div className="text-sm text-muted-foreground">
                  Scoring <strong>{formatNumber(totalVariants)}</strong> variants against process constraints
                </div>
                <Button data-testid="button-run-scoring">
                  <Gauge className="h-4 w-4 mr-2" />
                  Run Scoring
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Material Scores
              <Badge variant="secondary" className="ml-2">{filteredScores.length} shown</Badge>
            </h3>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground">Tier:</Label>
                <Select value={tierFilter} onValueChange={setTierFilter}>
                  <SelectTrigger className="w-40" data-testid="select-tier-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Tiers</SelectItem>
                    <SelectItem value="production-viable">Production-Viable</SelectItem>
                    <SelectItem value="pilot-ready">Pilot-Ready</SelectItem>
                    <SelectItem value="lab-only">Lab-Only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground">Min Score:</Label>
                <Slider
                  value={minScore}
                  onValueChange={setMinScore}
                  max={100}
                  min={0}
                  step={5}
                  className="w-24"
                />
                <span className="text-sm font-mono w-8">{minScore[0]}</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={showLowRiskOnly}
                  onCheckedChange={setShowLowRiskOnly}
                  data-testid="switch-low-risk"
                />
                <Label className="text-sm">Low risk only</Label>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredScores.map(score => (
              <MaterialScoreCard key={score.id} score={score} />
            ))}
          </div>

          {filteredScores.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No materials match current filters. Try adjusting the tier or minimum score.
            </div>
          )}

          <ResultsPanel
            materialsCampaignId="demo-campaign"
            title="Scoring Artifacts"
            collapsible={true}
            defaultExpanded={false}
          />
        </div>
      </main>
    </div>
  );
}
