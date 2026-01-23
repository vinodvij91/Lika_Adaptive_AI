import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Database, 
  ExternalLink, 
  Search, 
  AlertCircle, 
  Activity,
  FileText
} from "lucide-react";

interface ExternalDatabasePanelProps {
  smiles: string;
  moleculeName?: string;
}

async function fetchChemblData(smiles: string) {
  const response = await fetch(`/api/lookup/chembl/smiles?smiles=${encodeURIComponent(smiles)}`, {
    credentials: "include",
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Failed to fetch ChEMBL data");
  return response.json();
}

async function fetchPubchemData(smiles: string) {
  const response = await fetch(`/api/lookup/pubchem/smiles?smiles=${encodeURIComponent(smiles)}`, {
    credentials: "include",
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Failed to fetch PubChem data");
  return response.json();
}

export function ExternalDatabasePanel({ smiles, moleculeName }: ExternalDatabasePanelProps) {
  const [activeTab, setActiveTab] = useState("chembl");

  const { data: chemblData, isLoading: chemblLoading, error: chemblError, refetch: refetchChembl } = useQuery({
    queryKey: ["/api/lookup/chembl/smiles", smiles],
    queryFn: () => fetchChemblData(smiles),
    enabled: activeTab === "chembl",
    staleTime: 1000 * 60 * 10,
    retry: 1,
  });

  const { data: pubchemData, isLoading: pubchemLoading, error: pubchemError, refetch: refetchPubchem } = useQuery({
    queryKey: ["/api/lookup/pubchem/smiles", smiles],
    queryFn: () => fetchPubchemData(smiles),
    enabled: activeTab === "pubchem",
    staleTime: 1000 * 60 * 10,
    retry: 1,
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Database className="h-4 w-4 text-emerald-500" />
            External Databases
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full h-8 mb-3">
            <TabsTrigger value="chembl" className="flex-1 text-xs" data-testid="tab-chembl">
              ChEMBL
            </TabsTrigger>
            <TabsTrigger value="pubchem" className="flex-1 text-xs" data-testid="tab-pubchem">
              PubChem
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chembl" className="mt-0">
            {chemblLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : chemblError ? (
              <div className="text-center py-4 text-muted-foreground">
                <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                <p className="text-sm">Failed to load ChEMBL data</p>
                <Button variant="outline" size="sm" className="mt-2" onClick={() => refetchChembl()} data-testid="button-retry-chembl">
                  Retry
                </Button>
              </div>
            ) : chemblData === null ? (
              <div className="text-center py-4 text-muted-foreground">
                <Search className="h-8 w-8 mx-auto mb-2" />
                <p className="text-sm">Not found in ChEMBL database</p>
              </div>
            ) : (
              <ScrollArea className="h-64">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{chemblData.preferredName}</div>
                      <div className="text-xs text-muted-foreground font-mono">{chemblData.chemblId}</div>
                    </div>
                    <a 
                      href={`https://www.ebi.ac.uk/chembl/compound_report_card/${chemblData.chemblId}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Button variant="outline" size="sm" data-testid="button-view-chembl">
                        <ExternalLink className="h-3 w-3" />
                      </Button>
                    </a>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">MW</div>
                      <div className="font-medium">{chemblData.molecularWeight?.toFixed(2) || "N/A"}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">Max Phase</div>
                      <div className="font-medium flex items-center gap-1">
                        {chemblData.maxPhase || 0}
                        {chemblData.maxPhase >= 4 && <Badge variant="default" className="text-[10px] px-1">Approved</Badge>}
                      </div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">Type</div>
                      <div className="font-medium">{chemblData.moleculeType || "N/A"}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">Natural Product</div>
                      <div className="font-medium">{chemblData.naturalProduct ? "Yes" : "No"}</div>
                    </div>
                  </div>

                  {chemblData.activities && chemblData.activities.length > 0 && (
                    <div>
                      <div className="text-xs font-medium mb-2 flex items-center gap-1">
                        <Activity className="h-3 w-3" />
                        Bioactivity Data
                      </div>
                      <div className="space-y-1.5">
                        {chemblData.activities.slice(0, 3).map((act: any, i: number) => (
                          <div key={i} className="bg-muted/30 p-2 rounded text-xs">
                            <div className="flex items-center justify-between">
                              <span className="font-medium truncate flex-1">{act.targetPrefName}</span>
                              <Badge variant="outline" className="text-[10px] ml-2">
                                {act.activityType}: {act.value?.toFixed(2)} {act.units}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </TabsContent>

          <TabsContent value="pubchem" className="mt-0">
            {pubchemLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : pubchemError ? (
              <div className="text-center py-4 text-muted-foreground">
                <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                <p className="text-sm">Failed to load PubChem data</p>
                <Button variant="outline" size="sm" className="mt-2" onClick={() => refetchPubchem()} data-testid="button-retry-pubchem">
                  Retry
                </Button>
              </div>
            ) : pubchemData === null ? (
              <div className="text-center py-4 text-muted-foreground">
                <Search className="h-8 w-8 mx-auto mb-2" />
                <p className="text-sm">Not found in PubChem database</p>
              </div>
            ) : (
              <ScrollArea className="h-64">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-sm">{pubchemData.iupacName || "Unknown"}</div>
                      <div className="text-xs text-muted-foreground font-mono">CID: {pubchemData.cid}</div>
                    </div>
                    <a 
                      href={`https://pubchem.ncbi.nlm.nih.gov/compound/${pubchemData.cid}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Button variant="outline" size="sm" data-testid="button-view-pubchem">
                        <ExternalLink className="h-3 w-3" />
                      </Button>
                    </a>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">MW</div>
                      <div className="font-medium">{pubchemData.molecularWeight?.toFixed(2)}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">XLogP</div>
                      <div className="font-medium">{pubchemData.xlogp?.toFixed(2) || "N/A"}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">TPSA</div>
                      <div className="font-medium">{pubchemData.tpsa?.toFixed(1) || "N/A"} A²</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">Complexity</div>
                      <div className="font-medium">{pubchemData.complexity?.toFixed(0)}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">HBD</div>
                      <div className="font-medium">{pubchemData.hbondDonorCount}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">HBA</div>
                      <div className="font-medium">{pubchemData.hbondAcceptorCount}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">Rotatable Bonds</div>
                      <div className="font-medium">{pubchemData.rotatableBondCount}</div>
                    </div>
                    <div className="bg-muted/50 p-2 rounded">
                      <div className="text-muted-foreground">Heavy Atoms</div>
                      <div className="font-medium">{pubchemData.heavyAtomCount}</div>
                    </div>
                  </div>

                  {pubchemData.synonyms && pubchemData.synonyms.length > 0 && (
                    <div>
                      <div className="text-xs font-medium mb-2 flex items-center gap-1">
                        <FileText className="h-3 w-3" />
                        Synonyms
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {pubchemData.synonyms.slice(0, 5).map((syn: string, i: number) => (
                          <Badge key={i} variant="secondary" className="text-[10px]">
                            {syn}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
