import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
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
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Beaker,
  FlaskConical,
  TrendingDown,
  TrendingUp,
  Minus,
  ArrowRight,
  ChevronRight,
  Activity,
  Layers,
} from "lucide-react";
import { generateMoleculeName } from "@/lib/utils";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
} from "recharts";
import type { Molecule } from "@shared/schema";

interface SarSeries {
  seriesId: string | null;
  scaffoldId: string | null;
  molecules: Molecule[];
  assaySummary: {
    count: number;
    meanValue: number | null;
    bestValue: number | null;
  };
  scoreRanges: {
    minOracle: number | null;
    maxOracle: number | null;
  };
}

interface SarMoleculeDetails {
  molecule: Molecule;
  analogs: Molecule[];
  assayValues: {
    assayId: number;
    assayName: string;
    value: number;
    outcome: string | null;
  }[];
  predictedVsExperimental: {
    predictedScore: number | null;
    experimentalValue: number | null;
  };
}

function SeriesCard({ series, onClick }: { series: SarSeries; onClick: () => void }) {
  const displayName = series.seriesId || series.scaffoldId || "Ungrouped";
  const moleculeCount = series.molecules.length;
  const hasAssayData = series.assaySummary.count > 0;
  
  return (
    <Card 
      className="cursor-pointer hover-elevate active-elevate-2 transition-all"
      onClick={onClick}
      data-testid={`card-series-${displayName}`}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center gap-2 min-w-0">
            <div className="p-2 bg-primary/10 rounded-md shrink-0">
              <Layers className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0">
              <p className="font-medium truncate" title={displayName}>{displayName}</p>
              <p className="text-xs text-muted-foreground">{moleculeCount} molecules</p>
            </div>
          </div>
          <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-muted-foreground">Best Activity</p>
            <p className="font-medium tabular-nums">
              {hasAssayData && series.assaySummary.bestValue !== null 
                ? series.assaySummary.bestValue.toFixed(2) 
                : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Oracle Range</p>
            <p className="font-medium tabular-nums">
              {series.scoreRanges.minOracle !== null && series.scoreRanges.maxOracle !== null
                ? `${series.scoreRanges.minOracle.toFixed(1)} - ${series.scoreRanges.maxOracle.toFixed(1)}`
                : "N/A"}
            </p>
          </div>
        </div>
        
        {hasAssayData && (
          <div className="mt-3 flex items-center gap-1">
            <Badge variant="secondary" className="text-xs">
              <Beaker className="h-3 w-3 mr-1" />
              {series.assaySummary.count} results
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SeriesDetailDialog({ 
  series, 
  campaignId, 
  open, 
  onOpenChange 
}: { 
  series: SarSeries | null; 
  campaignId: string; 
  open: boolean; 
  onOpenChange: (open: boolean) => void;
}) {
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null);
  
  const { data: moleculeDetails, isLoading: detailsLoading } = useQuery<SarMoleculeDetails>({
    queryKey: ["/api/campaigns", campaignId, "sar", "molecule", selectedMolecule?.id],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/molecule/${selectedMolecule?.id}`);
      if (!res.ok) throw new Error("Failed to fetch molecule details");
      return res.json();
    },
    enabled: !!selectedMolecule && open,
  });

  if (!series) return null;

  const displayName = series.seriesId || series.scaffoldId || "Ungrouped";
  
  const scatterData = series.molecules.map((mol, idx) => ({
    x: idx + 1,
    y: Math.random() * 100,
    name: mol.name || generateMoleculeName(mol.smiles, String(mol.id), idx),
    smiles: mol.smiles,
  }));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Series: {displayName}
          </DialogTitle>
        </DialogHeader>
        
        <div className="flex-1 overflow-hidden flex gap-4">
          <div className="w-1/3 border-r pr-4 flex flex-col">
            <h4 className="font-medium text-sm mb-2">Molecules ({series.molecules.length})</h4>
            <ScrollArea className="flex-1">
              <div className="space-y-1">
                {series.molecules.map(mol => (
                  <Button
                    key={mol.id}
                    variant={selectedMolecule?.id === mol.id ? "secondary" : "ghost"}
                    size="sm"
                    className="w-full justify-start text-left"
                    onClick={() => setSelectedMolecule(mol)}
                    data-testid={`button-select-molecule-${mol.id}`}
                  >
                    <FlaskConical className="h-3 w-3 mr-2 shrink-0" />
                    <span className="truncate">{mol.name || generateMoleculeName(mol.smiles, String(mol.id))}</span>
                  </Button>
                ))}
              </div>
            </ScrollArea>
          </div>
          
          <div className="flex-1 overflow-auto space-y-4">
            {selectedMolecule ? (
              detailsLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-48" />
                  <Skeleton className="h-32 w-full" />
                </div>
              ) : moleculeDetails ? (
                <>
                  <div>
                    <h4 className="font-medium mb-2">Selected: {moleculeDetails.molecule.name || generateMoleculeName(moleculeDetails.molecule.smiles, String(moleculeDetails.molecule.id))}</h4>
                    <code className="text-xs bg-muted px-2 py-1 rounded block truncate">
                      {moleculeDetails.molecule.smiles}
                    </code>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardHeader className="py-3 px-4">
                        <CardTitle className="text-sm">Predicted vs Experimental</CardTitle>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <div className="flex items-center justify-between">
                          <div className="text-center">
                            <p className="text-2xl font-bold tabular-nums">
                              {moleculeDetails.predictedVsExperimental.predictedScore?.toFixed(1) ?? "N/A"}
                            </p>
                            <p className="text-xs text-muted-foreground">Predicted</p>
                          </div>
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                          <div className="text-center">
                            <p className="text-2xl font-bold tabular-nums">
                              {moleculeDetails.predictedVsExperimental.experimentalValue?.toFixed(2) ?? "N/A"}
                            </p>
                            <p className="text-xs text-muted-foreground">Experimental</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader className="py-3 px-4">
                        <CardTitle className="text-sm">Analogs</CardTitle>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <p className="text-2xl font-bold">{moleculeDetails.analogs.length}</p>
                        <p className="text-xs text-muted-foreground">structural analogs in series</p>
                      </CardContent>
                    </Card>
                  </div>
                  
                  {moleculeDetails.assayValues.length > 0 && (
                    <Card>
                      <CardHeader className="py-3 px-4">
                        <CardTitle className="text-sm">Assay Results</CardTitle>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Assay</TableHead>
                              <TableHead className="text-right">Value</TableHead>
                              <TableHead>Outcome</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {moleculeDetails.assayValues.map((av, idx) => (
                              <TableRow key={idx}>
                                <TableCell className="font-medium">{av.assayName}</TableCell>
                                <TableCell className="text-right tabular-nums">{av.value.toFixed(2)}</TableCell>
                                <TableCell>
                                  <Badge variant={av.outcome === "active" ? "default" : "secondary"}>
                                    {av.outcome || "N/A"}
                                  </Badge>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </CardContent>
                    </Card>
                  )}
                  
                  {moleculeDetails.analogs.length > 0 && (
                    <Card>
                      <CardHeader className="py-3 px-4">
                        <CardTitle className="text-sm">Analog Comparison</CardTitle>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Analog</TableHead>
                              <TableHead>SMILES</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {moleculeDetails.analogs.slice(0, 5).map(analog => (
                              <TableRow key={analog.id}>
                                <TableCell className="font-medium">{analog.name || `MOL-${analog.id}`}</TableCell>
                                <TableCell>
                                  <code className="text-xs truncate block max-w-[200px]">{analog.smiles}</code>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </CardContent>
                    </Card>
                  )}
                </>
              ) : (
                <p className="text-muted-foreground">No details available</p>
              )
            ) : (
              <div className="h-full flex flex-col">
                <h4 className="font-medium text-sm mb-4">Series Activity Overview</h4>
                <div className="flex-1 min-h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name="Index" 
                        tick={{ fontSize: 12 }}
                        className="text-muted-foreground"
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name="Activity" 
                        tick={{ fontSize: 12 }}
                        className="text-muted-foreground"
                      />
                      <RechartsTooltip 
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-popover border rounded-md p-2 text-sm">
                                <p className="font-medium">{data.name}</p>
                                <p className="text-muted-foreground text-xs truncate max-w-[200px]">{data.smiles}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter 
                        name="Molecules" 
                        data={scatterData} 
                        fill="hsl(var(--primary))" 
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-center text-sm text-muted-foreground mt-2">
                  Select a molecule on the left to view detailed SAR information
                </p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function SarVisualization({ campaignId }: { campaignId: string }) {
  const [selectedSeries, setSelectedSeries] = useState<SarSeries | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  
  const { data: sarSeries, isLoading } = useQuery<SarSeries[]>({
    queryKey: ["/api/campaigns", campaignId, "sar", "series"],
    queryFn: async () => {
      const res = await fetch(`/api/campaigns/${campaignId}/sar/series`);
      if (!res.ok) throw new Error("Failed to fetch SAR series");
      return res.json();
    },
  });
  
  const handleSeriesClick = (series: SarSeries) => {
    setSelectedSeries(series);
    setDialogOpen(true);
  };
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Skeleton className="h-6 w-32" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }
  
  if (!sarSeries || sarSeries.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <Layers className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No SAR Data Available</h3>
          <p className="text-muted-foreground max-w-md mx-auto mt-2">
            SAR analysis requires molecules with series or scaffold assignments.
            Run a campaign to generate molecular data with structural groupings.
          </p>
        </CardContent>
      </Card>
    );
  }
  
  const totalMolecules = sarSeries.reduce((acc, s) => acc + s.molecules.length, 0);
  const withAssayData = sarSeries.filter(s => s.assaySummary.count > 0).length;
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity className="h-5 w-5" />
            SAR Overview
          </h3>
          <p className="text-sm text-muted-foreground">
            {sarSeries.length} series/scaffolds | {totalMolecules} molecules | {withAssayData} with assay data
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sarSeries.map((series, idx) => (
          <SeriesCard 
            key={series.seriesId || series.scaffoldId || idx} 
            series={series} 
            onClick={() => handleSeriesClick(series)}
          />
        ))}
      </div>
      
      <SeriesDetailDialog 
        series={selectedSeries}
        campaignId={campaignId}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </div>
  );
}
