import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { PageHeader } from "@/components/page-header";
import { LikaAgentChat } from "@/components/lika-agent-chat";
import { 
  Brain, 
  Sparkles, 
  FlaskConical, 
  Beaker,
  Target,
  BarChart3,
  Lightbulb,
  FileText,
  Search,
  Hexagon
} from "lucide-react";
import type { Molecule } from "@shared/schema";

const AGENT_CAPABILITIES = [
  {
    icon: FlaskConical,
    title: "SMILES Analysis",
    description: "Parse and validate SMILES, explain functional groups, identify liabilities",
  },
  {
    icon: Beaker,
    title: "ADMET Profiling",
    description: "Assess absorption, distribution, metabolism, excretion, and toxicity risks",
  },
  {
    icon: Target,
    title: "SAR Interpretation",
    description: "Analyze structure-activity relationships and recommend modifications",
  },
  {
    icon: BarChart3,
    title: "Compound Ranking",
    description: "Score and prioritize compounds based on multi-parameter optimization",
  },
  {
    icon: Lightbulb,
    title: "Workflow Guidance",
    description: "Recommend next steps in hit-finding, lead optimization, or screening",
  },
  {
    icon: FileText,
    title: "Scientific Summaries",
    description: "Generate memos, reports, and slide-ready content from your data",
  },
];

export default function LikaAgentPage() {
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const { data: molecules } = useQuery<Molecule[]>({
    queryKey: ["/api/molecules"],
  });

  const filteredMolecules = molecules?.filter(m => 
    (m.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
     m.smiles?.toLowerCase().includes(searchQuery.toLowerCase())) &&
    searchQuery.length > 0
  ).slice(0, 5);

  const moleculeContext = selectedMolecule ? {
    smiles: selectedMolecule.smiles,
    name: selectedMolecule.name || undefined,
    molecularWeight: selectedMolecule.molecularWeight || undefined,
    logP: selectedMolecule.logP || undefined,
  } : undefined;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <PageHeader 
        title="Lika Agent" 
        breadcrumbs={[
          { label: "Drug Discovery" },
          { label: "Lika Agent" }
        ]}
      />

      <div className="flex-1 flex gap-6 p-6 overflow-hidden">
        <div className="w-80 flex flex-col gap-4 flex-shrink-0 overflow-auto">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Hexagon className="h-4 w-4 text-violet-500" />
                Molecule Context
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search molecules..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                  data-testid="input-molecule-search"
                />
              </div>
              
              {filteredMolecules && filteredMolecules.length > 0 && (
                <div className="space-y-1 border rounded-md p-1">
                  {filteredMolecules.map((mol) => (
                    <Button
                      key={mol.id}
                      variant={selectedMolecule?.id === mol.id ? "secondary" : "ghost"}
                      className="w-full justify-start text-left h-auto py-2"
                      onClick={() => {
                        setSelectedMolecule(mol);
                        setSearchQuery("");
                      }}
                      data-testid={`button-select-molecule-${mol.id}`}
                    >
                      <div className="truncate">
                        <div className="font-medium text-sm truncate">{mol.name || "Unnamed"}</div>
                        <div className="text-xs text-muted-foreground font-mono truncate">
                          {mol.smiles?.substring(0, 30)}...
                        </div>
                      </div>
                    </Button>
                  ))}
                </div>
              )}

              {selectedMolecule ? (
                <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm">{selectedMolecule.name || "Selected"}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs"
                      onClick={() => setSelectedMolecule(null)}
                      data-testid="button-clear-molecule"
                    >
                      Clear
                    </Button>
                  </div>
                  <div className="text-xs font-mono text-muted-foreground break-all">
                    {selectedMolecule.smiles?.substring(0, 60)}...
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">MW: </span>
                      <span className="font-medium">{selectedMolecule.molecularWeight?.toFixed(1) || "—"}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">LogP: </span>
                      <span className="font-medium">{selectedMolecule.logP?.toFixed(2) || "—"}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground text-center py-2">
                  Search and select a molecule to provide context for the agent
                </p>
              )}
            </CardContent>
          </Card>

          <Card className="flex-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-amber-500" />
                Agent Capabilities
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {AGENT_CAPABILITIES.map((cap, i) => {
                  const Icon = cap.icon;
                  return (
                    <div key={i} className="flex gap-3">
                      <div className="w-8 h-8 rounded-md bg-muted flex items-center justify-center flex-shrink-0">
                        <Icon className="h-4 w-4 text-muted-foreground" />
                      </div>
                      <div>
                        <div className="text-sm font-medium">{cap.title}</div>
                        <div className="text-xs text-muted-foreground">{cap.description}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex-1 min-w-0">
          <LikaAgentChat 
            moleculeContext={moleculeContext} 
            className="h-full"
          />
        </div>
      </div>
    </div>
  );
}
