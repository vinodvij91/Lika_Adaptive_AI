import { useQuery } from "@tanstack/react-query";
import { useParams, Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  ArrowLeft, 
  Hexagon,
  Activity,
  Beaker,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Molecule, MoleculeScore, AssayResult } from "@shared/schema";
import { MoleculeStructureViewer } from "@/components/molecule-structure-viewer";
import { ExternalDatabasePanel } from "@/components/external-database-panel";
import { AIPredictionsPanel } from "@/components/ai-predictions-panel";

type MoleculeWithDetails = Molecule & {
  scores: MoleculeScore[];
  assayResults: (AssayResult & { assayName?: string })[];
};

export default function MoleculeDetailPage() {
  const { id } = useParams<{ id: string }>();

  const { data: molecule, isLoading } = useQuery<MoleculeWithDetails>({
    queryKey: ["/api/molecules", id, "details"],
    queryFn: async () => {
      const res = await fetch(`/api/molecules/${id}/details`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch molecule");
      return res.json();
    },
  });

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
        </div>
      </div>
    );
  }

  if (!molecule) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <Hexagon className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-xl font-semibold mb-2">Molecule Not Found</h2>
          <p className="text-muted-foreground mb-4">The molecule you're looking for doesn't exist.</p>
          <Link href="/molecules">
            <Button variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Molecules
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  const latestScore = molecule.scores?.[0];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/molecules">
          <Button variant="ghost" size="icon" data-testid="button-back-molecules">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <Hexagon className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold" data-testid="text-molecule-name">{molecule.name || "Unnamed Molecule"}</h1>
              <p className="text-muted-foreground text-sm font-mono">{molecule.seriesId || molecule.id}</p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {molecule.isDemo && (
            <Badge variant="secondary" className="bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
              Demo
            </Badge>
          )}
          {molecule.source && (
            <Badge variant="outline">{molecule.source}</Badge>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <MoleculeStructureViewer 
          smiles={molecule.smiles} 
          moleculeName={molecule.name || "Molecule Structure"} 
          size="lg"
        />
        <ExternalDatabasePanel 
          smiles={molecule.smiles} 
          moleculeName={molecule.name || undefined}
        />
        <AIPredictionsPanel 
          smiles={molecule.smiles} 
          moleculeName={molecule.name || undefined}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">SMILES</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="font-mono text-sm bg-muted p-4 rounded-md overflow-x-auto break-all">
            {molecule.smiles}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">MW</CardTitle>
          </CardHeader>
          <CardContent>
            <span className="text-lg font-bold">{molecule.molecularWeight?.toFixed(1) || "—"}</span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">LogP</CardTitle>
          </CardHeader>
          <CardContent>
            <span className="text-lg font-bold">{molecule.logP?.toFixed(2) || "—"}</span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">H-Bond Donors</CardTitle>
          </CardHeader>
          <CardContent>
            <span className="text-lg font-bold">{molecule.numHBondDonors ?? "—"}</span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">H-Bond Acceptors</CardTitle>
          </CardHeader>
          <CardContent>
            <span className="text-lg font-bold">{molecule.numHBondAcceptors ?? "—"}</span>
          </CardContent>
        </Card>
      </div>

      {latestScore && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Oracle Scores
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {latestScore.oracleScore != null && (
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-blue-600">
                    {(latestScore.oracleScore * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Oracle Score</div>
                </div>
              )}
              {latestScore.dockingScore != null && (
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-green-600">
                    {latestScore.dockingScore.toFixed(2)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Docking</div>
                </div>
              )}
              {latestScore.admetScore != null && (
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-purple-600">
                    {(latestScore.admetScore * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">ADMET</div>
                </div>
              )}
              {latestScore.synthesisScore != null && (
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-orange-600">
                    {(latestScore.synthesisScore * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Synthesizability</div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Beaker className="h-5 w-5" />
            Assay Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          {molecule.assayResults && molecule.assayResults.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Assay</TableHead>
                  <TableHead>Value</TableHead>
                  <TableHead>Units</TableHead>
                  <TableHead>Outcome</TableHead>
                  <TableHead>Date</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {molecule.assayResults.map((result) => (
                  <TableRow key={result.id}>
                    <TableCell>
                      <Link href={`/assays/${result.assayId}`}>
                        <span className="font-medium hover:underline cursor-pointer">
                          {result.assayName || result.assayId}
                        </span>
                      </Link>
                    </TableCell>
                    <TableCell className="font-mono">{result.value?.toFixed(3) ?? "—"}</TableCell>
                    <TableCell>{result.units || "—"}</TableCell>
                    <TableCell>
                      {result.outcomeLabel && (
                        <Badge variant="outline">{result.outcomeLabel}</Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {result.createdAt ? formatDistanceToNow(new Date(result.createdAt), { addSuffix: true }) : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No assay results for this molecule yet.
            </div>
          )}
        </CardContent>
      </Card>

      <div className="text-xs text-muted-foreground">
        {molecule.createdAt && `Created ${formatDistanceToNow(new Date(molecule.createdAt), { addSuffix: true })}`}
      </div>
    </div>
  );
}
