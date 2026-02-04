import { useQuery, useMutation } from "@tanstack/react-query";
import { CsvFormatGuide, CSV_FORMATS } from "@/components/csv-format-guide";
import { useParams, Link } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  ArrowLeft,
  Library,
  FlaskConical,
  Layers,
  Upload,
  CheckCircle,
  Clock,
  AlertCircle,
  Plus,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { CuratedLibrary, LibraryMolecule, Scaffold, LibraryAnnotation, Molecule } from "@shared/schema";
import { useState } from "react";

interface LibraryWithDetails extends CuratedLibrary {
  molecules?: (LibraryMolecule & { molecule: Molecule | null })[];
  scaffolds?: Scaffold[];
  annotations?: LibraryAnnotation[];
}

const statusConfig: Record<string, { icon: typeof CheckCircle; color: string; label: string }> = {
  curated: { icon: CheckCircle, color: "text-green-600", label: "Curated" },
  processing: { icon: Clock, color: "text-yellow-600", label: "Processing" },
  draft: { icon: AlertCircle, color: "text-muted-foreground", label: "Draft" },
  deprecated: { icon: AlertCircle, color: "text-red-600", label: "Deprecated" },
};

const cleaningStatusColors: Record<string, string> = {
  pending: "bg-yellow-500/20 text-yellow-700",
  cleaning: "bg-blue-500/20 text-blue-700",
  validated: "bg-green-500/20 text-green-700",
  failed: "bg-red-500/20 text-red-700",
};

export default function LibraryDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { toast } = useToast();
  const [isImportOpen, setIsImportOpen] = useState(false);
  const [smilesText, setSmilesText] = useState("");

  const { data: library, isLoading } = useQuery<LibraryWithDetails>({
    queryKey: ["/api/libraries", id],
  });

  const importMutation = useMutation({
    mutationFn: async (smilesList: string[]) => {
      const res = await apiRequest("POST", `/api/libraries/${id}/import-smiles`, { smilesList });
      return res.json();
    },
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ["/api/libraries", id] });
      toast({ title: `Imported ${data.imported} molecules` });
      setIsImportOpen(false);
      setSmilesText("");
    },
    onError: () => {
      toast({ title: "Failed to import SMILES", variant: "destructive" });
    },
  });

  const handleImport = () => {
    const smilesList = smilesText
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (smilesList.length === 0) {
      toast({ title: "No SMILES to import", variant: "destructive" });
      return;
    }
    importMutation.mutate(smilesList);
  };

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

  if (!library) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">Library not found</p>
            <Link href="/libraries">
              <Button variant="outline" className="mt-4">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Libraries
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const statusInfo = statusConfig[library.status || "draft"];
  const StatusIcon = statusInfo.icon;

  const molecules = library.molecules || [];
  const scaffolds = library.scaffolds || [];
  const annotations = library.annotations || [];

  const validatedCount = molecules.filter((m) => m.cleaningStatus === "validated").length;
  const pendingCount = molecules.filter((m) => m.cleaningStatus === "pending").length;
  const failedCount = molecules.filter((m) => m.cleaningStatus === "failed").length;
  const validationProgress = molecules.length > 0 ? (validatedCount / molecules.length) * 100 : 0;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4 flex-wrap">
        <Link href="/libraries">
          <Button variant="ghost" size="icon" data-testid="button-back">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <Library className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-semibold truncate" data-testid="text-library-name">
              {library.name}
            </h1>
            <Badge variant="outline">{library.domainType}</Badge>
            <div className={`flex items-center gap-1 ${statusInfo.color}`}>
              <StatusIcon className="h-4 w-4" />
              <span className="text-sm">{statusInfo.label}</span>
            </div>
          </div>
          {library.description && (
            <p className="text-muted-foreground mt-1">{library.description}</p>
          )}
        </div>
        <Dialog open={isImportOpen} onOpenChange={setIsImportOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-import-smiles">
              <Upload className="h-4 w-4 mr-2" />
              Import SMILES
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Import SMILES</DialogTitle>
              <DialogDescription>
                Paste SMILES strings, one per line. They will be cleaned and validated.
              </DialogDescription>
            </DialogHeader>
            <Textarea
              placeholder="CCO\nCCCO\nCCCCO"
              value={smilesText}
              onChange={(e) => setSmilesText(e.target.value)}
              rows={10}
              className="font-mono text-sm"
              data-testid="textarea-smiles-import"
            />
            <CsvFormatGuide
              title="Expected format:"
              columns={CSV_FORMATS.smiles.columns}
              exampleRows={CSV_FORMATS.smiles.exampleRows}
              templateFilename={CSV_FORMATS.smiles.templateFilename}
              templateContent={CSV_FORMATS.smiles.templateContent}
            />
            <Button
              onClick={handleImport}
              disabled={importMutation.isPending}
              data-testid="button-confirm-import"
            >
              {importMutation.isPending ? "Importing..." : "Import"}
            </Button>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Molecules</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              <FlaskConical className="h-5 w-5 text-primary" />
              {library.moleculeCount || molecules.length}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Scaffolds</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Layers className="h-5 w-5 text-primary" />
              {library.scaffoldCount || scaffolds.length}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Validated</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              {validatedCount}
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <Progress value={validationProgress} className="h-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Pending</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Clock className="h-5 w-5 text-yellow-600" />
              {pendingCount}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <Tabs defaultValue="molecules">
        <TabsList>
          <TabsTrigger value="molecules" data-testid="tab-molecules">
            Molecules ({molecules.length})
          </TabsTrigger>
          <TabsTrigger value="scaffolds" data-testid="tab-scaffolds">
            Scaffolds ({scaffolds.length})
          </TabsTrigger>
          <TabsTrigger value="annotations" data-testid="tab-annotations">
            Annotations ({annotations.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="molecules" className="mt-4">
          {molecules.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <FlaskConical className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No molecules yet</h3>
                <p className="text-muted-foreground mb-4">
                  Import SMILES to populate this library.
                </p>
                <Button onClick={() => setIsImportOpen(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Import SMILES
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>SMILES</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Scaffold</TableHead>
                    <TableHead>Tags</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {molecules.slice(0, 50).map((libMol) => (
                    <TableRow key={libMol.id} data-testid={`row-molecule-${libMol.id}`}>
                      <TableCell className="font-mono text-xs max-w-xs truncate">
                        {libMol.canonicalSmiles || libMol.molecule?.smiles}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="secondary"
                          className={cleaningStatusColors[libMol.cleaningStatus || "pending"]}
                        >
                          {libMol.cleaningStatus || "pending"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {libMol.scaffoldId || "-"}
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1 flex-wrap">
                          {libMol.tags?.slice(0, 2).map((tag) => (
                            <Badge key={tag} variant="outline" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {molecules.length > 50 && (
                <CardContent className="border-t">
                  <p className="text-sm text-muted-foreground text-center">
                    Showing 50 of {molecules.length} molecules
                  </p>
                </CardContent>
              )}
            </Card>
          )}
        </TabsContent>

        <TabsContent value="scaffolds" className="mt-4">
          {scaffolds.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <Layers className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No scaffolds detected</h3>
                <p className="text-muted-foreground">
                  Scaffolds will be generated during library processing.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {scaffolds.map((scaffold) => (
                <Card key={scaffold.id} data-testid={`card-scaffold-${scaffold.id}`}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">
                      {scaffold.name || "Unnamed Scaffold"}
                    </CardTitle>
                    <CardDescription>
                      {scaffold.memberCount || 0} members
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="font-mono text-xs text-muted-foreground truncate">
                      {scaffold.coreSmiles}
                    </p>
                    <Badge variant="outline" className="mt-2">
                      {scaffold.scaffoldType || "murcko"}
                    </Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="annotations" className="mt-4">
          {annotations.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground">
                  No annotations yet. Agent validation will add annotations here.
                </p>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Type</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Created</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {annotations.map((annotation) => (
                    <TableRow key={annotation.id}>
                      <TableCell>{annotation.annotationType}</TableCell>
                      <TableCell>{annotation.annotationValue}</TableCell>
                      <TableCell>
                        <Badge variant={annotation.source === "agent" ? "default" : "secondary"}>
                          {annotation.source}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {annotation.confidence ? `${(annotation.confidence * 100).toFixed(0)}%` : "-"}
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDistanceToNow(new Date(annotation.createdAt || new Date()), { addSuffix: true })}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Library Details</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Type</p>
            <p className="font-medium">{library.libraryType}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Version</p>
            <p className="font-medium">{library.version || 1}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Public</p>
            <p className="font-medium">{library.isPublic ? "Yes" : "No"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Created</p>
            <p className="font-medium">
              {formatDistanceToNow(new Date(library.createdAt || new Date()), { addSuffix: true })}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
