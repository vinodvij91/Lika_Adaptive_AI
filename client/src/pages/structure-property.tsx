import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
  BarChart,
  Bar,
  LineChart,
  Line,
  Legend,
} from "recharts";
import {
  Layers,
  TrendingUp,
  BarChart3,
  Grid3X3,
  Hexagon,
  Target,
  Filter,
  Download,
  ZoomIn,
  Activity,
} from "lucide-react";

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + "K";
  }
  return num.toLocaleString();
}

const GROUP_CONFIGS: Record<string, { labels: string[]; paramLabel: string }> = {
  family: { 
    labels: ["Polyamide", "Polyester", "Polyethylene", "Polypropylene", "PEEK", "PPS", "PTFE"],
    paramLabel: "Material Family"
  },
  scaffold: { 
    labels: ["Linear Chain", "Branched", "Cross-linked", "Dendritic", "Star", "Ladder"],
    paramLabel: "Scaffold Type"
  },
  chain_length: { 
    labels: ["10-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"],
    paramLabel: "Chain Length Bin"
  },
  dopant: { 
    labels: ["None", "Carbon", "Silicon", "Nitrogen", "Fluorine", "Metal Oxide"],
    paramLabel: "Dopant Type"
  },
};

const PROPERTY_OFFSETS: Record<string, number> = {
  thermal_stability: 0,
  tensile_strength: 0.1,
  conductivity: 0.2,
  flexibility: 0.15,
};

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function generateMockPercentileData(totalVariants: number, groupBy: string, seed: number) {
  const rand = seededRandom(seed);
  const percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99];
  const groupOffset = Object.keys(GROUP_CONFIGS).indexOf(groupBy) * 0.05;
  return percentiles.map(p => ({
    percentile: p,
    score: rand() * 0.2 + (100 - p) / 100 * 0.7 + groupOffset,
    count: Math.floor(totalVariants * (p / 100)),
    label: `P${p}`,
  }));
}

function generateMockDensityData(groupBy: string, propertyX: string, seed: number) {
  const rand = seededRandom(seed);
  const bins = 20;
  const data = [];
  const propOffset = PROPERTY_OFFSETS[propertyX] || 0;
  const groupSeed = Object.keys(GROUP_CONFIGS).indexOf(groupBy) * 0.1;
  const peakPos = 0.4 + groupSeed + propOffset;
  
  for (let i = 0; i < bins; i++) {
    const x = i / bins;
    const density = Math.exp(-Math.pow((x - peakPos) * 3, 2)) * 0.7 + 
                   Math.exp(-Math.pow((x - 0.8 - propOffset) * 5, 2)) * 0.3;
    data.push({
      bin: (x * 100).toFixed(0),
      density: density * 1000 + rand() * 100,
      topPerformers: density > 0.5 ? density * 500 : 0,
    });
  }
  return data;
}

function generateMockBinnedScatter(propertyX: string, propertyY: string, seed: number) {
  const rand = seededRandom(seed);
  const data = [];
  const xOffset = PROPERTY_OFFSETS[propertyX] || 0;
  const yOffset = PROPERTY_OFFSETS[propertyY] || 0;
  const centerX = 0.5 + xOffset * 0.5;
  const centerY = 0.6 + yOffset * 0.5;
  
  for (let i = 0; i < 15; i++) {
    for (let j = 0; j < 15; j++) {
      const x = i / 15;
      const y = j / 15;
      const count = Math.floor(Math.exp(-Math.pow(x - centerX, 2) * 4 - Math.pow(y - centerY, 2) * 4) * 5000 + rand() * 200);
      if (count > 100) {
        data.push({
          x: x * 100,
          y: y * 100,
          count,
          avgScore: 0.3 + (x + y) / 4 + rand() * 0.1,
        });
      }
    }
  }
  return data;
}

function generateMockGroupData(groupBy: string, seed: number) {
  const rand = seededRandom(seed);
  const config = GROUP_CONFIGS[groupBy] || GROUP_CONFIGS.family;
  return config.labels.map((label, idx) => ({
    group: label,
    variantCount: Math.floor(rand() * 50000) + 5000 + idx * 3000,
    avgScore: rand() * 0.3 + 0.45 + idx * 0.02,
    topPerformers: Math.floor(rand() * 2000) + 500 + idx * 100,
    p95Score: rand() * 0.15 + 0.78 + idx * 0.015,
  }));
}

function generateMockTrendData(groupBy: string, propertyX: string, seed: number) {
  const rand = seededRandom(seed);
  const data = [];
  const propOffset = PROPERTY_OFFSETS[propertyX] || 0;
  const config = GROUP_CONFIGS[groupBy] || GROUP_CONFIGS.family;
  
  for (let i = 0; i < 10; i++) {
    const param = (i + 1) * 10;
    const baseP50 = 0.4 + i * 0.03 + propOffset;
    data.push({
      parameter: param,
      parameterLabel: groupBy === "chain_length" ? `${param}` : config.labels[i % config.labels.length],
      p50: baseP50 + rand() * 0.05,
      p75: baseP50 + 0.1 + rand() * 0.05,
      p95: baseP50 + 0.25 + rand() * 0.05,
      count: Math.floor(rand() * 15000) + 5000,
    });
  }
  return data;
}

interface InsightBadgeProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  variant?: "default" | "success" | "warning";
}

function InsightBadge({ icon, label, value, variant = "default" }: InsightBadgeProps) {
  const bgClass = variant === "success" 
    ? "bg-green-500/10 border-green-500/30 text-green-700 dark:text-green-400"
    : variant === "warning"
    ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-700 dark:text-yellow-400"
    : "bg-muted border-border";

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md border ${bgClass}`}>
      {icon}
      <span className="text-sm font-medium">{label}:</span>
      <span className="text-sm font-mono">{value}</span>
    </div>
  );
}

function DensityHeatmap({ data }: { data: any[] }) {
  const maxCount = Math.max(...data.map(d => d.count));
  
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-15 gap-0.5">
        {data.map((cell, i) => {
          const intensity = cell.count / maxCount;
          const hue = 220 - intensity * 160;
          return (
            <div
              key={i}
              className="aspect-square rounded-sm cursor-pointer transition-transform hover:scale-110"
              style={{
                backgroundColor: `hsl(${hue}, 70%, ${40 + intensity * 30}%)`,
                opacity: 0.3 + intensity * 0.7,
              }}
              title={`Count: ${formatNumber(cell.count)}\nAvg Score: ${cell.avgScore.toFixed(2)}`}
            />
          );
        })}
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Low density</span>
        <div className="flex items-center gap-1">
          {[0.2, 0.4, 0.6, 0.8, 1].map((v, i) => (
            <div
              key={i}
              className="w-4 h-3 rounded-sm"
              style={{
                backgroundColor: `hsl(${220 - v * 160}, 70%, ${40 + v * 30}%)`,
              }}
            />
          ))}
        </div>
        <span>High density</span>
      </div>
    </div>
  );
}

export default function StructurePropertyPage() {
  const { toast } = useToast();
  const [groupBy, setGroupBy] = useState<string>("family");
  const [propertyX, setPropertyX] = useState<string>("thermal_stability");
  const [propertyY, setPropertyY] = useState<string>("tensile_strength");
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false);
  const [analysisSeed, setAnalysisSeed] = useState(0);

  const handleRunAnalysis = () => {
    setIsRunningAnalysis(true);
    toast({
      title: "Running Analysis",
      description: "Processing 127K+ material variants...",
    });
    // Simulate analysis with new random seed
    setTimeout(() => {
      setAnalysisSeed(prev => prev + 1);
      setIsRunningAnalysis(false);
      toast({
        title: "Analysis Complete",
        description: "Updated structure-property correlations with latest data.",
      });
    }, 1500);
  };

  const totalVariants = 127450 + analysisSeed * 150;
  const topPerformersCount = 12450 + analysisSeed * 25;
  const topPercentile = 5;

  const groupConfig = GROUP_CONFIGS[groupBy] || GROUP_CONFIGS.family;
  const seed = groupBy.length + propertyX.length + propertyY.length + analysisSeed;

  const percentileData = useMemo(() => generateMockPercentileData(totalVariants, groupBy, seed), [groupBy, seed]);
  const densityData = useMemo(() => generateMockDensityData(groupBy, propertyX, seed), [groupBy, propertyX, seed]);
  const binnedScatterData = useMemo(() => generateMockBinnedScatter(propertyX, propertyY, seed), [propertyX, propertyY, seed]);
  const groupData = useMemo(() => generateMockGroupData(groupBy, seed), [groupBy, seed]);
  const trendData = useMemo(() => generateMockTrendData(groupBy, propertyX, seed), [groupBy, propertyX, seed]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Structure-Property Analytics" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" data-testid="button-export-analytics">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button onClick={handleRunAnalysis} disabled={isRunningAnalysis} data-testid="button-run-analysis">
              {isRunningAnalysis ? (
                <>
                  <div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4 mr-2" />
                  Run Analysis
                </>
              )}
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <InsightBadge
              icon={<Layers className="h-4 w-4" />}
              label="Total Variants"
              value={formatNumber(totalVariants)}
            />
            <InsightBadge
              icon={<Target className="h-4 w-4" />}
              label={`Top ${topPercentile}% performers`}
              value={`${formatNumber(topPerformersCount)} variants`}
              variant="success"
            />
            <InsightBadge
              icon={<TrendingUp className="h-4 w-4" />}
              label="Performance plateau"
              value="detected at scale"
              variant="warning"
            />
          </div>

          <Card>
            <CardHeader className="pb-2">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Filter className="h-5 w-5" />
                  Analysis Controls
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Group By</Label>
                  <Select value={groupBy} onValueChange={setGroupBy}>
                    <SelectTrigger data-testid="select-group-by">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="family">Material Family</SelectItem>
                      <SelectItem value="scaffold">Scaffold / Lattice</SelectItem>
                      <SelectItem value="chain_length">Chain Length Bins</SelectItem>
                      <SelectItem value="dopant">Dopant Type</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>X-Axis Property</Label>
                  <Select value={propertyX} onValueChange={setPropertyX}>
                    <SelectTrigger data-testid="select-property-x">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="thermal_stability">Thermal Stability</SelectItem>
                      <SelectItem value="tensile_strength">Tensile Strength</SelectItem>
                      <SelectItem value="conductivity">Conductivity</SelectItem>
                      <SelectItem value="flexibility">Flexibility</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Y-Axis Property</Label>
                  <Select value={propertyY} onValueChange={setPropertyY}>
                    <SelectTrigger data-testid="select-property-y">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="thermal_stability">Thermal Stability</SelectItem>
                      <SelectItem value="tensile_strength">Tensile Strength</SelectItem>
                      <SelectItem value="conductivity">Conductivity</SelectItem>
                      <SelectItem value="flexibility">Flexibility</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="density" className="space-y-4">
            <TabsList>
              <TabsTrigger value="density" data-testid="tab-density">
                <Grid3X3 className="h-4 w-4 mr-2" />
                Density Plot
              </TabsTrigger>
              <TabsTrigger value="percentiles" data-testid="tab-percentiles">
                <TrendingUp className="h-4 w-4 mr-2" />
                Percentile Curves
              </TabsTrigger>
              <TabsTrigger value="heatmap" data-testid="tab-heatmap">
                <Hexagon className="h-4 w-4 mr-2" />
                Property Heatmap
              </TabsTrigger>
              <TabsTrigger value="families" data-testid="tab-families">
                <BarChart3 className="h-4 w-4 mr-2" />
                Family Analysis
              </TabsTrigger>
            </TabsList>

            <TabsContent value="density">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Score Distribution Density</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Aggregated view of {formatNumber(totalVariants)} variants - density shows concentration of results
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={densityData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis 
                          dataKey="bin" 
                          className="text-xs"
                          label={{ value: 'Score Percentile', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          className="text-xs"
                          label={{ value: 'Variant Count', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px',
                          }}
                          formatter={(value: number) => [formatNumber(value), 'Variants']}
                        />
                        <Area
                          type="monotone"
                          dataKey="density"
                          stroke="hsl(var(--primary))"
                          fill="hsl(var(--primary))"
                          fillOpacity={0.3}
                          name="All Variants"
                        />
                        <Area
                          type="monotone"
                          dataKey="topPerformers"
                          stroke="hsl(142, 76%, 36%)"
                          fill="hsl(142, 76%, 36%)"
                          fillOpacity={0.5}
                          name="Top Performers"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 flex items-center justify-center gap-6 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm bg-primary/50" />
                      <span>All Variants</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(142, 76%, 36%)' }} />
                      <span>Top 10% Performers</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="percentiles">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Performance Envelope by Parameter</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Percentile curves showing performance distribution across chain length parameter
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis 
                          dataKey="parameter" 
                          className="text-xs"
                          label={{ value: 'Chain Length', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          className="text-xs"
                          domain={[0.3, 1]}
                          label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px',
                          }}
                          formatter={(value: number, name: string) => [
                            value.toFixed(3),
                            name === 'p50' ? 'Median' : name === 'p75' ? '75th Percentile' : '95th Percentile'
                          ]}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="p95"
                          stroke="hsl(142, 76%, 36%)"
                          strokeWidth={2}
                          dot={false}
                          name="P95 (Top 5%)"
                        />
                        <Line
                          type="monotone"
                          dataKey="p75"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                          name="P75"
                        />
                        <Line
                          type="monotone"
                          dataKey="p50"
                          stroke="hsl(var(--muted-foreground))"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                          name="Median"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 p-3 rounded-md bg-muted/50 text-sm">
                    <span className="font-medium">Insight:</span> Performance plateau detected at chain length 70-80. 
                    Top 5% performers ({formatNumber(topPerformersCount)} variants) concentrate in this range.
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="heatmap">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Binned Property Scatter</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    {propertyX.replace(/_/g, ' ')} vs {propertyY.replace(/_/g, ' ')} - color intensity shows variant density
                  </p>
                </CardHeader>
                <CardContent>
                  <DensityHeatmap data={binnedScatterData} />
                  <div className="mt-6 grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-md bg-muted/50">
                      <div className="text-sm font-medium mb-1">High-Density Region</div>
                      <div className="text-xs text-muted-foreground">
                        X: 40-60, Y: 50-70 contains ~{formatNumber(45000)} variants (35%)
                      </div>
                    </div>
                    <div className="p-3 rounded-md bg-green-500/10 border border-green-500/30">
                      <div className="text-sm font-medium mb-1 text-green-700 dark:text-green-400">Optimal Zone</div>
                      <div className="text-xs text-muted-foreground">
                        X: 70-90, Y: 60-80 - highest avg scores (0.82)
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="families">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Performance by {groupConfig.paramLabel}</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Aggregated metrics across {groupData.length} {groupConfig.paramLabel.toLowerCase()}s
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={groupData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis type="number" domain={[0, 1]} className="text-xs" />
                        <YAxis type="category" dataKey="group" className="text-xs" width={100} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '6px',
                          }}
                          formatter={(value: number, name: string) => [
                            name === 'variantCount' ? formatNumber(value) : value.toFixed(3),
                            name === 'avgScore' ? 'Avg Score' : name === 'p95Score' ? 'P95 Score' : 'Variant Count'
                          ]}
                        />
                        <Legend />
                        <Bar dataKey="avgScore" fill="hsl(var(--primary))" name="Avg Score" />
                        <Bar dataKey="p95Score" fill="hsl(142, 76%, 36%)" name="P95 Score" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 font-medium">{groupConfig.paramLabel}</th>
                          <th className="text-right py-2 font-medium">Variants</th>
                          <th className="text-right py-2 font-medium">Top Performers</th>
                          <th className="text-right py-2 font-medium">Avg Score</th>
                          <th className="text-right py-2 font-medium">P95 Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {groupData.sort((a, b) => b.p95Score - a.p95Score).map((row) => (
                          <tr key={row.group} className="border-b border-muted hover-elevate">
                            <td className="py-2 font-medium">{row.group}</td>
                            <td className="text-right py-2 font-mono">{formatNumber(row.variantCount)}</td>
                            <td className="text-right py-2 font-mono text-green-600 dark:text-green-400">
                              {formatNumber(row.topPerformers)}
                            </td>
                            <td className="text-right py-2 font-mono">{row.avgScore.toFixed(3)}</td>
                            <td className="text-right py-2 font-mono font-medium">{row.p95Score.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}
