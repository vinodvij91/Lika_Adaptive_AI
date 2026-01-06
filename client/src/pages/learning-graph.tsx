import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
  Brain,
  Download,
  Filter,
} from "lucide-react";
import { useState } from "react";
import type { LearningGraphEntry, DiseaseArea } from "@shared/schema";

const outcomeColors: Record<string, string> = {
  promising: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400",
  hit: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
  dropped: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
  unknown: "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400",
};

interface LearningGraphEntryWithMolecule extends LearningGraphEntry {
  molecule?: { smiles: string };
}

export default function LearningGraphPage() {
  const [domainFilter, setDomainFilter] = useState<string>("all");
  const [outcomeFilter, setOutcomeFilter] = useState<string>("all");

  const { data: entries, isLoading } = useQuery<LearningGraphEntryWithMolecule[]>({
    queryKey: ["/api/learning-graph"],
  });

  const filteredEntries = entries?.filter((e) => {
    const matchesDomain = domainFilter === "all" || e.domainType === domainFilter;
    const matchesOutcome = outcomeFilter === "all" || e.outcomeLabel === outcomeFilter;
    return matchesDomain && matchesOutcome;
  });

  const handleExport = () => {
    if (!filteredEntries || filteredEntries.length === 0) return;

    const csv = [
      ["SMILES", "Domain", "Outcome", "Oracle Score"].join(","),
      ...filteredEntries.map((e) =>
        [
          `"${e.molecule?.smiles || ""}"`,
          e.domainType || "",
          e.outcomeLabel || "",
          e.oracleScore?.toFixed(4) || "",
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "learning-graph.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Learning Graph" }]}
        actions={
          <Button variant="outline" onClick={handleExport} className="gap-2" data-testid="button-export">
            <Download className="h-4 w-4" />
            Export
          </Button>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-5 w-5 text-muted-foreground" />
                Internal Learning Graph
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Track molecule outcomes across campaigns to inform future discovery
              </p>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center gap-4 mb-6">
                <div className="flex items-center gap-2">
                  <Filter className="h-4 w-4 text-muted-foreground" />
                  <Select value={domainFilter} onValueChange={setDomainFilter}>
                    <SelectTrigger className="w-[160px]" data-testid="select-domain-filter">
                      <SelectValue placeholder="Domain" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Domains</SelectItem>
                      <SelectItem value="CNS">CNS</SelectItem>
                      <SelectItem value="Oncology">Oncology</SelectItem>
                      <SelectItem value="Rare">Rare</SelectItem>
                      <SelectItem value="Infectious">Infectious</SelectItem>
                      <SelectItem value="Cardiometabolic">Cardiometabolic</SelectItem>
                      <SelectItem value="Autoimmune">Autoimmune</SelectItem>
                      <SelectItem value="Respiratory">Respiratory</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Select value={outcomeFilter} onValueChange={setOutcomeFilter}>
                  <SelectTrigger className="w-[140px]" data-testid="select-outcome-filter">
                    <SelectValue placeholder="Outcome" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Outcomes</SelectItem>
                    <SelectItem value="promising">Promising</SelectItem>
                    <SelectItem value="hit">Hit</SelectItem>
                    <SelectItem value="dropped">Dropped</SelectItem>
                    <SelectItem value="unknown">Unknown</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {isLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="flex items-center gap-4">
                      <Skeleton className="h-4 w-64" />
                      <Skeleton className="h-5 w-20 rounded-full" />
                      <Skeleton className="h-5 w-20 rounded-full" />
                      <Skeleton className="h-4 w-16" />
                    </div>
                  ))}
                </div>
              ) : filteredEntries && filteredEntries.length > 0 ? (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>SMILES</TableHead>
                        <TableHead>Domain</TableHead>
                        <TableHead>Outcome</TableHead>
                        <TableHead className="text-right">Oracle Score</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredEntries.map((entry) => (
                        <TableRow key={entry.id} data-testid={`row-entry-${entry.id}`}>
                          <TableCell>
                            <code className="text-xs font-mono max-w-xs truncate block">
                              {entry.molecule?.smiles}
                            </code>
                          </TableCell>
                          <TableCell>
                            {entry.domainType && (
                              <DiseaseAreaBadge area={entry.domainType} showIcon={false} />
                            )}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="outline"
                              className={`capitalize no-default-hover-elevate no-default-active-elevate ${
                                outcomeColors[entry.outcomeLabel || "unknown"]
                              }`}
                            >
                              {entry.outcomeLabel || "unknown"}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {entry.oracleScore?.toFixed(4) || "-"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="text-center py-16">
                  <Brain className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No learning data yet</h3>
                  <p className="text-muted-foreground max-w-sm mx-auto">
                    {domainFilter !== "all" || outcomeFilter !== "all"
                      ? "No entries match your filters."
                      : "Run campaigns to populate the learning graph with molecule outcomes."}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
