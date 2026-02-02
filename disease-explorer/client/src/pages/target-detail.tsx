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
  Target,
  Dna,
  ExternalLink,
  CheckCircle,
  XCircle,
  FlaskConical,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Target as TargetType, Assay } from "@shared/schema";

type TargetWithDetails = TargetType & {
  assays: Assay[];
};

export default function TargetDetailPage() {
  const { id } = useParams<{ id: string }>();

  const { data: target, isLoading } = useQuery<TargetWithDetails>({
    queryKey: ["/api/targets", id, "details"],
    queryFn: async () => {
      const res = await fetch(`/api/targets/${id}/details`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch target");
      return res.json();
    },
  });

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
        </div>
      </div>
    );
  }

  if (!target) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <Target className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-xl font-semibold mb-2">Target Not Found</h2>
          <p className="text-muted-foreground mb-4">The target you're looking for doesn't exist.</p>
          <Link href="/targets">
            <Button variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Targets
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/targets">
          <Button variant="ghost" size="icon" data-testid="button-back-targets">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center">
              <Target className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold" data-testid="text-target-name">{target.name}</h1>
              <p className="text-muted-foreground text-sm">
                {target.uniprotId && `UniProt: ${target.uniprotId}`}
                {target.pdbId && ` • PDB: ${target.pdbId}`}
              </p>
            </div>
          </div>
        </div>
        {target.isDemo && (
          <Badge variant="secondary" className="bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
            Demo
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">UniProt ID</CardTitle>
          </CardHeader>
          <CardContent>
            {target.uniprotId ? (
              <a 
                href={`https://www.uniprot.org/uniprotkb/${target.uniprotId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xl font-bold text-blue-600 hover:underline flex items-center gap-1"
                data-testid="link-uniprot"
              >
                {target.uniprotId}
                <ExternalLink className="h-4 w-4" />
              </a>
            ) : (
              <span className="text-xl font-bold text-muted-foreground">—</span>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">PDB ID</CardTitle>
          </CardHeader>
          <CardContent>
            {target.pdbId ? (
              <a 
                href={`https://www.rcsb.org/structure/${target.pdbId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xl font-bold text-blue-600 hover:underline flex items-center gap-1"
                data-testid="link-pdb"
              >
                {target.pdbId}
                <ExternalLink className="h-4 w-4" />
              </a>
            ) : (
              <span className="text-xl font-bold text-muted-foreground">—</span>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Has Structure</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {target.hasStructure ? (
                <>
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <span className="text-xl font-bold">Yes</span>
                </>
              ) : (
                <>
                  <XCircle className="h-5 w-5 text-muted-foreground" />
                  <span className="text-xl font-bold text-muted-foreground">No</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {target.sequence && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Dna className="h-4 w-4" />
              Sequence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="font-mono text-xs bg-muted p-4 rounded-md overflow-x-auto whitespace-pre-wrap break-all">
              {target.sequence}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5" />
            Related Assays
          </CardTitle>
        </CardHeader>
        <CardContent>
          {target.assays && target.assays.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {target.assays.map((assay) => (
                  <TableRow key={assay.id}>
                    <TableCell>
                      <Link href={`/assays/${assay.id}`}>
                        <span className="font-medium hover:underline cursor-pointer" data-testid={`link-assay-${assay.id}`}>
                          {assay.name}
                        </span>
                      </Link>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{assay.type}</Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {assay.createdAt ? formatDistanceToNow(new Date(assay.createdAt), { addSuffix: true }) : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No assays linked to this target yet.
            </div>
          )}
        </CardContent>
      </Card>

      <div className="text-xs text-muted-foreground">
        {target.createdAt && `Created ${formatDistanceToNow(new Date(target.createdAt), { addSuffix: true })}`}
      </div>
    </div>
  );
}
