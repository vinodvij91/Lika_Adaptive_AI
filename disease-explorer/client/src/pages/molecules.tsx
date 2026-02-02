import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
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
  Search,
  FlaskConical,
  ArrowRight,
  Filter,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Molecule } from "@shared/schema";

const sourceColors: Record<string, string> = {
  generated: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400",
  uploaded: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
  screened: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400",
};

export default function MoleculesPage() {
  const [search, setSearch] = useState("");
  const [sourceFilter, setSourceFilter] = useState<string>("all");

  const { data: molecules, isLoading } = useQuery<Molecule[]>({
    queryKey: ["/api/molecules"],
  });

  const filteredMolecules = molecules?.filter((m) => {
    const matchesSearch = m.smiles.toLowerCase().includes(search.toLowerCase());
    const matchesSource = sourceFilter === "all" || m.source === sourceFilter;
    return matchesSearch && matchesSource;
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader breadcrumbs={[{ label: "Molecules" }]} />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="relative flex-1 min-w-[200px] max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by SMILES..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
                data-testid="input-search-molecules"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <Select value={sourceFilter} onValueChange={setSourceFilter}>
                <SelectTrigger className="w-[140px]" data-testid="select-source-filter">
                  <SelectValue placeholder="Source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sources</SelectItem>
                  <SelectItem value="generated">Generated</SelectItem>
                  <SelectItem value="uploaded">Uploaded</SelectItem>
                  <SelectItem value="screened">Screened</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {isLoading ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>SMILES</TableHead>
                      <TableHead>Source</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-[100px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[1, 2, 3, 4, 5].map((i) => (
                      <TableRow key={i}>
                        <TableCell><Skeleton className="h-4 w-64" /></TableCell>
                        <TableCell><Skeleton className="h-5 w-20 rounded-full" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                        <TableCell><Skeleton className="h-8 w-8 rounded-md" /></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : filteredMolecules && filteredMolecules.length > 0 ? (
            <Card>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>SMILES</TableHead>
                        <TableHead>Source</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="w-[100px]"></TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredMolecules.map((molecule) => (
                        <TableRow key={molecule.id} data-testid={`row-molecule-${molecule.id}`}>
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <div className="w-9 h-9 rounded-md bg-chart-4/10 flex items-center justify-center flex-shrink-0">
                                <FlaskConical className="h-4 w-4 text-chart-4" />
                              </div>
                              <code className="text-sm font-mono max-w-md truncate block">
                                {molecule.smiles}
                              </code>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="outline"
                              className={`capitalize no-default-hover-elevate no-default-active-elevate ${sourceColors[molecule.source || "generated"]}`}
                            >
                              {molecule.source}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {molecule.createdAt
                              ? formatDistanceToNow(new Date(molecule.createdAt), { addSuffix: true })
                              : "-"}
                          </TableCell>
                          <TableCell>
                            <Link href={`/molecules/${molecule.id}`}>
                              <Button variant="ghost" size="icon" data-testid={`button-view-molecule-${molecule.id}`}>
                                <ArrowRight className="h-4 w-4" />
                              </Button>
                            </Link>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="py-16">
                <div className="flex flex-col items-center justify-center text-center">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <FlaskConical className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No molecules found</h3>
                  <p className="text-muted-foreground max-w-sm">
                    {search || sourceFilter !== "all"
                      ? "No molecules match your filters."
                      : "Molecules will appear here after running campaigns."}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
