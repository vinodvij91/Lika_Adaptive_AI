import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Target,
  Layers,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  BarChart3,
  Sparkles,
} from "lucide-react";
import { generateMoleculeName } from "@/lib/utils";
import {
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  Cell,
} from "recharts";

interface TargetScore {
  targetId: string;
  targetName: string;
  predictedScore: number | null;
  experimentalValue: number | null;
  safetyFlag: boolean;
}

interface MultiTargetMolecule {
  id: string;
  smiles: string;
  seriesId: string | null;
  scaffoldId: string | null;
  oracleScore: number | null;
  targetScores: TargetScore[];
  compositeScore: number | null;
}

interface MultiTargetTarget {
  id: string;
  name: string;
  role: string;
}

interface MultiTargetSeries {
  seriesId: string;
  improvesMultipleTargets: boolean;
  degradesSafety: boolean;
}

interface MultiTargetSarData {
  molecules: MultiTargetMolecule[];
  targets: MultiTargetTarget[];
  series: MultiTargetSeries[];
}

type ProfileFilter = "all" | "original" | "optimized";

function computeAvgRadar(
  mols: MultiTargetMolecule[],
  targets: MultiTargetTarget[],
) {
  return targets.map(target => {
    const avgValue = mols.reduce((acc, mol) => {
      const ts = mol.targetScores.find(t => t.targetId === target.id);
      const val = ts?.experimentalValue ?? ts?.predictedScore ?? null;
      if (val !== null) {
        acc.sum += val;
        acc.count++;
      }
      return acc;
    }, { sum: 0, count: 0 });

    return {
      target: target.name.slice(0, 12),
      fullName: target.name,
      role: target.role,
      value: avgValue.count > 0 ? Math.max(0, 100 - avgValue.sum / avgValue.count / 10) : 0,
      rawValue: avgValue.count > 0 ? avgValue.sum / avgValue.count : null,
    };
  });
}

function RadarSarView({
  molecules,
  targets,
  selectedMoleculeId,
  onSelectMolecule,
  profileFilter,
  onProfileChange,
  optimizedMoleculeIds,
}: {
  molecules: MultiTargetMolecule[];
  targets: MultiTargetTarget[];
  selectedMoleculeId: string | null;
  onSelectMolecule: (id: string | null) => void;
  profileFilter: ProfileFilter;
  onProfileChange: (f: ProfileFilter) => void;
  optimizedMoleculeIds: string[];
}) {
  const hasOptimized = optimizedMoleculeIds.length > 0;
  const selectedMolecule = molecules.find(m => m.id === selectedMoleculeId);

  const originalMols = molecules.filter(m => !optimizedMoleculeIds.includes(m.id));
  const optimizedMols = molecules.filter(m => optimizedMoleculeIds.includes(m.id));

  const showDualTrace = profileFilter === "all" && !selectedMolecule && hasOptimized && optimizedMols.length > 0;

  const filteredMols =
    profileFilter === "original" ? originalMols :
    profileFilter === "optimized" ? optimizedMols :
    molecules;

  const radarData = targets.map(target => {
    const result: Record<string, unknown> = {
      target: target.name.slice(0, 12),
      fullName: target.name,
      role: target.role,
    };

    if (selectedMolecule) {
      const targetScore = selectedMolecule.targetScores.find(ts => ts.targetId === target.id);
      result.value = targetScore?.experimentalValue !== null
        ? Math.max(0, 100 - (targetScore?.experimentalValue ?? 0) / 10)
        : (targetScore?.predictedScore !== null ? (targetScore?.predictedScore ?? 0) * 10 : 0);
      result.rawValue = targetScore?.experimentalValue ?? targetScore?.predictedScore ?? null;
    } else if (showDualTrace) {
      const origAvg = originalMols.reduce((acc, mol) => {
        const ts = mol.targetScores.find(t => t.targetId === target.id);
        const val = ts?.experimentalValue ?? ts?.predictedScore ?? null;
        if (val !== null) { acc.sum += val; acc.count++; }
        return acc;
      }, { sum: 0, count: 0 });

      const optAvg = optimizedMols.reduce((acc, mol) => {
        const ts = mol.targetScores.find(t => t.targetId === target.id);
        const val = ts?.experimentalValue ?? ts?.predictedScore ?? null;
        if (val !== null) { acc.sum += val; acc.count++; }
        return acc;
      }, { sum: 0, count: 0 });

      result.original = origAvg.count > 0 ? Math.max(0, 100 - origAvg.sum / origAvg.count / 10) : 0;
      result.optimized = optAvg.count > 0 ? Math.max(0, 100 - optAvg.sum / optAvg.count / 10) : 0;
      result.rawOriginal = origAvg.count > 0 ? origAvg.sum / origAvg.count : null;
      result.rawOptimized = optAvg.count > 0 ? optAvg.sum / optAvg.count : null;
    } else {
      const avgValue = filteredMols.reduce((acc, mol) => {
        const ts = mol.targetScores.find(t => t.targetId === target.id);
        const val = ts?.experimentalValue ?? ts?.predictedScore ?? null;
        if (val !== null) { acc.sum += val; acc.count++; }
        return acc;
      }, { sum: 0, count: 0 });
      result.value = avgValue.count > 0 ? Math.max(0, 100 - avgValue.sum / avgValue.count / 10) : 0;
      result.rawValue = avgValue.count > 0 ? avgValue.sum / avgValue.count : null;
    }

    return result;
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h4 className="font-medium">Target Activity Radar</h4>
          <p className="text-sm text-muted-foreground">
            {selectedMolecule
              ? `Viewing: ${selectedMolecule.smiles.slice(0, 30)}...`
              : showDualTrace
                ? `Comparing original (${originalMols.length}) vs optimized (${optimizedMols.length})`
                : `Average across ${filteredMols.length} molecules`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasOptimized && (
            <Select value={profileFilter} onValueChange={(v) => onProfileChange(v as ProfileFilter)}>
              <SelectTrigger className="w-[140px]" data-testid="select-profile-filter">
                <SelectValue placeholder="Profile" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Both</SelectItem>
                <SelectItem value="original">Original</SelectItem>
                <SelectItem value="optimized">Optimized</SelectItem>
              </SelectContent>
            </Select>
          )}
          <Select
            value={selectedMoleculeId || "all"}
            onValueChange={(v) => onSelectMolecule(v === "all" ? null : v)}
          >
            <SelectTrigger className="w-[200px]" data-testid="select-molecule-radar">
              <SelectValue placeholder="Select molecule" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All (Average)</SelectItem>
              {filteredMols.slice(0, 50).map((mol, idx) => (
                <SelectItem key={mol.id} value={mol.id}>
                  {optimizedMoleculeIds.includes(mol.id) ? "* " : ""}
                  {generateMoleculeName(mol.smiles, mol.id, idx)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid className="stroke-muted" />
            <PolarAngleAxis
              dataKey="target"
              tick={{ fontSize: 11 }}
              className="text-muted-foreground"
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={{ fontSize: 10 }}
              className="text-muted-foreground"
            />
            {showDualTrace ? (
              <>
                <Radar
                  name="Original"
                  dataKey="original"
                  stroke="hsl(var(--primary))"
                  fill="hsl(var(--primary))"
                  fillOpacity={0.15}
                  strokeWidth={2}
                  strokeDasharray="6 3"
                />
                <Radar
                  name="Optimized"
                  dataKey="optimized"
                  stroke="hsl(150 80% 40%)"
                  fill="hsl(150 80% 40%)"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
                <Legend />
              </>
            ) : (
              <Radar
                name="Activity"
                dataKey="value"
                stroke="hsl(var(--primary))"
                fill="hsl(var(--primary))"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            )}
            <RechartsTooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-popover border rounded-md p-2 text-sm shadow-lg">
                      <p className="font-medium">{data.fullName}</p>
                      <p className="text-xs text-muted-foreground capitalize">Role: {data.role}</p>
                      {showDualTrace ? (
                        <>
                          <p className="text-xs">Original: {data.rawOriginal?.toFixed(2) ?? "N/A"}</p>
                          <p className="text-xs text-green-600">Optimized: {data.rawOptimized?.toFixed(2) ?? "N/A"}</p>
                        </>
                      ) : (
                        <p className="text-xs">Value: {data.rawValue?.toFixed(2) ?? "N/A"}</p>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground flex-wrap">
        {targets.map(target => (
          <div key={target.id} className="flex items-center gap-1">
            <Badge
              variant={target.role === "safety" ? "destructive" : target.role === "secondary" ? "secondary" : "default"}
              className="text-xs"
            >
              {target.name.slice(0, 10)}
            </Badge>
            <span className="capitalize">{target.role}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TradeOffScatter({
  molecules,
  targets,
}: {
  molecules: MultiTargetMolecule[];
  targets: MultiTargetTarget[];
}) {
  const [xTarget, setXTarget] = useState<string>(targets[0]?.id || "");
  const [yTarget, setYTarget] = useState<string>(targets[1]?.id || targets[0]?.id || "");

  const xTargetName = targets.find(t => t.id === xTarget)?.name || "Target 1";
  const yTargetName = targets.find(t => t.id === yTarget)?.name || "Target 2";

  const scatterData = molecules.map(mol => {
    const xScore = mol.targetScores.find(ts => ts.targetId === xTarget);
    const yScore = mol.targetScores.find(ts => ts.targetId === yTarget);

    return {
      id: mol.id,
      smiles: mol.smiles,
      x: xScore?.experimentalValue ?? xScore?.predictedScore ?? null,
      y: yScore?.experimentalValue ?? yScore?.predictedScore ?? null,
      seriesId: mol.seriesId,
      compositeScore: mol.compositeScore,
    };
  }).filter(d => d.x !== null && d.y !== null);

  const seriesColors = [
    "hsl(var(--primary))",
    "hsl(210 80% 60%)",
    "hsl(150 80% 40%)",
    "hsl(30 80% 50%)",
    "hsl(280 80% 60%)",
    "hsl(0 80% 50%)",
  ];

  const uniqueSeries = Array.from(new Set(scatterData.map(d => d.seriesId || "ungrouped")));
  const seriesColorMap = new Map(uniqueSeries.map((s, i) => [s, seriesColors[i % seriesColors.length]]));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h4 className="font-medium">Target Trade-Off Analysis</h4>
          <p className="text-sm text-muted-foreground">
            Compare activity across two targets to identify balanced compounds
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={xTarget} onValueChange={setXTarget}>
            <SelectTrigger className="w-[150px]" data-testid="select-x-target">
              <SelectValue placeholder="X-axis target" />
            </SelectTrigger>
            <SelectContent>
              {targets.map(t => (
                <SelectItem key={t.id} value={t.id}>{t.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-muted-foreground">vs</span>
          <Select value={yTarget} onValueChange={setYTarget}>
            <SelectTrigger className="w-[150px]" data-testid="select-y-target">
              <SelectValue placeholder="Y-axis target" />
            </SelectTrigger>
            <SelectContent>
              {targets.map(t => (
                <SelectItem key={t.id} value={t.id}>{t.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              type="number"
              dataKey="x"
              name={xTargetName}
              tick={{ fontSize: 11 }}
              label={{ value: xTargetName, position: "bottom", offset: 0, fontSize: 12 }}
              className="text-muted-foreground"
            />
            <YAxis
              type="number"
              dataKey="y"
              name={yTargetName}
              tick={{ fontSize: 11 }}
              label={{ value: yTargetName, angle: -90, position: "insideLeft", fontSize: 12 }}
              className="text-muted-foreground"
            />
            <RechartsTooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-popover border rounded-md p-2 text-sm shadow-lg">
                      <code className="text-xs block truncate max-w-[200px]">{data.smiles}</code>
                      <div className="mt-1 space-y-0.5 text-xs">
                        <p>{xTargetName}: {data.x?.toFixed(2)}</p>
                        <p>{yTargetName}: {data.y?.toFixed(2)}</p>
                        {data.seriesId && <p className="text-muted-foreground">Series: {data.seriesId}</p>}
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend />
            <Scatter name="Molecules" data={scatterData}>
              {scatterData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={seriesColorMap.get(entry.seriesId || "ungrouped")}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="text-center text-sm text-muted-foreground">
        Lower values indicate better activity. Points in the bottom-left quadrant are balanced actives.
      </div>
    </div>
  );
}

function SeriesAnalysis({
  molecules,
  series,
  targets,
}: {
  molecules: MultiTargetMolecule[];
  series: MultiTargetSeries[];
  targets: MultiTargetTarget[];
}) {
  const seriesStats = series.map(s => {
    const seriesMols = molecules.filter(m => m.seriesId === s.seriesId);

    const targetStats = targets.map(target => {
      const values = seriesMols
        .map(m => m.targetScores.find(ts => ts.targetId === target.id)?.experimentalValue)
        .filter((v): v is number => v !== null);

      return {
        targetId: target.id,
        targetName: target.name,
        role: target.role,
        count: values.length,
        mean: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null,
        best: values.length > 0 ? Math.min(...values) : null,
      };
    });

    return {
      ...s,
      moleculeCount: seriesMols.length,
      targetStats,
    };
  });

  return (
    <div className="space-y-4">
      <div>
        <h4 className="font-medium">Series-Level Multi-Target Analysis</h4>
        <p className="text-sm text-muted-foreground">
          Evaluate chemical series for balanced multi-target activity
        </p>
      </div>

      <ScrollArea className="h-[500px]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Series</TableHead>
              <TableHead className="text-center">Molecules</TableHead>
              <TableHead className="text-center">Multi-Target</TableHead>
              <TableHead className="text-center">Safety</TableHead>
              {targets.slice(0, 4).map(t => (
                <TableHead key={t.id} className="text-center text-xs">
                  {t.name.slice(0, 8)}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {seriesStats.map(s => (
              <TableRow key={s.seriesId} data-testid={`row-series-${s.seriesId}`}>
                <TableCell className="font-mono text-xs">{s.seriesId}</TableCell>
                <TableCell className="text-center">{s.moleculeCount}</TableCell>
                <TableCell className="text-center">
                  {s.improvesMultipleTargets ? (
                    <CheckCircle className="h-4 w-4 text-green-500 mx-auto" />
                  ) : (
                    <XCircle className="h-4 w-4 text-muted-foreground mx-auto" />
                  )}
                </TableCell>
                <TableCell className="text-center">
                  {s.degradesSafety ? (
                    <AlertTriangle className="h-4 w-4 text-destructive mx-auto" />
                  ) : (
                    <CheckCircle className="h-4 w-4 text-green-500 mx-auto" />
                  )}
                </TableCell>
                {s.targetStats.slice(0, 4).map(ts => (
                  <TableCell key={ts.targetId} className="text-center">
                    {ts.best !== null ? (
                      <span className="text-xs tabular-nums">{ts.best.toFixed(1)}</span>
                    ) : (
                      <span className="text-xs text-muted-foreground">-</span>
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>

      <div className="flex items-center gap-4 text-xs text-muted-foreground justify-center">
        <div className="flex items-center gap-1">
          <CheckCircle className="h-3 w-3 text-green-500" />
          <span>Active on multiple targets</span>
        </div>
        <div className="flex items-center gap-1">
          <AlertTriangle className="h-3 w-3 text-destructive" />
          <span>Safety liability</span>
        </div>
      </div>
    </div>
  );
}

function MoleculeMatrix({
  molecules,
  targets,
  optimizedMoleculeIds,
}: {
  molecules: MultiTargetMolecule[];
  targets: MultiTargetTarget[];
  optimizedMoleculeIds: string[];
}) {
  const getActivityColor = (value: number | null, isSafety: boolean) => {
    if (value === null) return "bg-muted";
    if (isSafety) {
      return value < 100 ? "bg-destructive/80" : value < 1000 ? "bg-yellow-500/50" : "bg-green-500/30";
    }
    return value < 100 ? "bg-green-500/80" : value < 1000 ? "bg-green-500/30" : "bg-muted";
  };

  return (
    <div className="space-y-4">
      <div>
        <h4 className="font-medium">Activity Matrix</h4>
        <p className="text-sm text-muted-foreground">
          Heatmap of activity values across all targets
        </p>
      </div>

      <ScrollArea className="h-[500px]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="sticky left-0 bg-background z-10">Molecule</TableHead>
              <TableHead className="text-center">Type</TableHead>
              <TableHead className="text-center">Composite</TableHead>
              {targets.map(t => (
                <TableHead key={t.id} className="text-center">
                  <div className="flex flex-col items-center gap-0.5">
                    <span className="text-xs">{t.name.slice(0, 8)}</span>
                    <Badge variant={t.role === "safety" ? "destructive" : t.role === "secondary" ? "secondary" : "outline"} className="text-[10px] px-1">
                      {t.role}
                    </Badge>
                  </div>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {molecules.slice(0, 100).map(mol => {
              const isOpt = optimizedMoleculeIds.includes(mol.id);
              return (
                <TableRow key={mol.id}>
                  <TableCell className="font-mono text-xs sticky left-0 bg-background max-w-[150px] truncate">
                    {generateMoleculeName(mol.smiles, mol.id)}
                  </TableCell>
                  <TableCell className="text-center">
                    {isOpt ? (
                      <Badge variant="default" className="text-[10px]">
                        <Sparkles className="h-2.5 w-2.5 mr-0.5" />
                        Opt
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="text-[10px]">Orig</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-center tabular-nums text-xs">
                    {mol.compositeScore?.toFixed(1) ?? "-"}
                  </TableCell>
                  {targets.map(target => {
                    const ts = mol.targetScores.find(s => s.targetId === target.id);
                    const value = ts?.experimentalValue ?? ts?.predictedScore ?? null;
                    const isSafety = target.role === "safety";

                    return (
                      <TableCell key={target.id} className="text-center p-1">
                        <div className={`rounded px-2 py-1 text-xs tabular-nums ${getActivityColor(value, isSafety)}`}>
                          {value !== null ? value.toFixed(1) : "-"}
                        </div>
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </ScrollArea>

      <div className="flex items-center gap-4 text-xs text-muted-foreground justify-center flex-wrap">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500/80" />
          <span>Potent (&lt;100)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500/30" />
          <span>Active (&lt;1000)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-destructive/80" />
          <span>Safety concern</span>
        </div>
        <div className="flex items-center gap-1">
          <Badge variant="default" className="text-[10px]">
            <Sparkles className="h-2.5 w-2.5 mr-0.5" />
            Opt
          </Badge>
          <span>Optimized analog</span>
        </div>
      </div>
    </div>
  );
}

interface OptimizationSummary {
  totalOptimized: number;
  optimizedMoleculeIds: string[];
}

export function MultiTargetSar({ campaignId }: { campaignId: string }) {
  const [selectedMoleculeId, setSelectedMoleculeId] = useState<string | null>(null);
  const [profileFilter, setProfileFilter] = useState<ProfileFilter>("all");

  const { data, isLoading, error } = useQuery<MultiTargetSarData>({
    queryKey: ["/api/campaigns", campaignId, "sar", "multi-target"],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/multi-target`);
      if (!res.ok) throw new Error("Failed to fetch multi-target SAR data");
      return res.json();
    },
  });

  const { data: optSummary } = useQuery<OptimizationSummary>({
    queryKey: ["/api/campaigns", campaignId, "sar", "optimization-summary"],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/optimization-summary`);
      if (!res.ok) throw new Error("Failed to fetch optimization summary");
      return res.json();
    },
    enabled: !!data && data.molecules.length > 0,
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">Failed to Load Multi-Target SAR Data</h3>
          <p className="text-muted-foreground max-w-md mx-auto mt-2">
            {error instanceof Error ? error.message : "An error occurred while loading the data."}
          </p>
        </CardContent>
      </Card>
    );
  }

  if (data.molecules.length === 0 || data.targets.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <Target className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Multi-Target SAR Data</h3>
          <p className="text-muted-foreground max-w-md mx-auto mt-2">
            Multi-target SAR analysis requires molecules scored against multiple targets.
            Configure an assay panel with multiple targets to enable this view.
          </p>
        </CardContent>
      </Card>
    );
  }

  const optimizedMoleculeIds = optSummary?.optimizedMoleculeIds || [];
  const hasOptimized = optimizedMoleculeIds.length > 0;

  const balancedCount = data.molecules.filter(m =>
    m.targetScores.filter(ts => !ts.safetyFlag && ts.experimentalValue !== null && ts.experimentalValue < 1000).length >= 2
  ).length;

  const safetyIssues = data.molecules.filter(m =>
    m.targetScores.some(ts => ts.safetyFlag && ts.experimentalValue !== null && ts.experimentalValue < 100)
  ).length;

  return (
    <div className="space-y-6">
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-primary/10 rounded-md shrink-0">
              <Target className="h-5 w-5 text-primary" />
            </div>
            <div className="space-y-1">
              <h4 className="font-medium text-sm">Understanding Multi-Target SAR</h4>
              <p className="text-sm text-muted-foreground">
                This analysis helps identify molecules with balanced activity across multiple therapeutic targets.
                <strong className="text-foreground"> Balanced compounds</strong> (bottom-left quadrant in Trade-Off view)
                show good activity against both primary and secondary targets while avoiding safety liabilities.
              </p>
              <div className="flex flex-wrap gap-4 mt-2 text-xs text-muted-foreground">
                <span><strong>Radar:</strong> Compare activity profiles across all targets</span>
                <span><strong>Trade-Off:</strong> Visualize selectivity between target pairs</span>
                <span><strong>Series:</strong> Group molecules by chemical scaffold</span>
                <span><strong>Matrix:</strong> Full scoring data for all molecule-target pairs</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Target className="h-5 w-5" />
            Multi-Target SAR Analysis
          </h3>
          <p className="text-sm text-muted-foreground">
            {data.molecules.length} molecules | {data.targets.length} targets | {data.series.length} series
            {hasOptimized && (
              <span className="text-primary"> | {optimizedMoleculeIds.length} optimized</span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant="outline" className="gap-1">
            <TrendingUp className="h-3 w-3" />
            {balancedCount} balanced
          </Badge>
          {safetyIssues > 0 && (
            <Badge variant="destructive" className="gap-1">
              <AlertTriangle className="h-3 w-3" />
              {safetyIssues} safety flags
            </Badge>
          )}
          {hasOptimized && (
            <Badge variant="default" className="gap-1">
              <Sparkles className="h-3 w-3" />
              {optimizedMoleculeIds.length} optimized
            </Badge>
          )}
        </div>
      </div>

      <Tabs defaultValue="radar" className="w-full">
        <TabsList className="grid w-full grid-cols-4 max-w-lg">
          <TabsTrigger value="radar" data-testid="tab-radar">
            <Activity className="h-4 w-4 mr-1" />
            Radar
          </TabsTrigger>
          <TabsTrigger value="tradeoff" data-testid="tab-tradeoff">
            <BarChart3 className="h-4 w-4 mr-1" />
            Trade-Off
          </TabsTrigger>
          <TabsTrigger value="series" data-testid="tab-series">
            <Layers className="h-4 w-4 mr-1" />
            Series
          </TabsTrigger>
          <TabsTrigger value="matrix" data-testid="tab-matrix">
            <Target className="h-4 w-4 mr-1" />
            Matrix
          </TabsTrigger>
        </TabsList>

        <TabsContent value="radar" className="mt-4">
          <Card>
            <CardContent className="p-6">
              <RadarSarView
                molecules={data.molecules}
                targets={data.targets}
                selectedMoleculeId={selectedMoleculeId}
                onSelectMolecule={setSelectedMoleculeId}
                profileFilter={profileFilter}
                onProfileChange={setProfileFilter}
                optimizedMoleculeIds={optimizedMoleculeIds}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tradeoff" className="mt-4">
          <Card>
            <CardContent className="p-6">
              <TradeOffScatter
                molecules={data.molecules}
                targets={data.targets}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="series" className="mt-4">
          <Card>
            <CardContent className="p-6">
              <SeriesAnalysis
                molecules={data.molecules}
                series={data.series}
                targets={data.targets}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="matrix" className="mt-4">
          <Card>
            <CardContent className="p-6">
              <MoleculeMatrix
                molecules={data.molecules}
                targets={data.targets}
                optimizedMoleculeIds={optimizedMoleculeIds}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
