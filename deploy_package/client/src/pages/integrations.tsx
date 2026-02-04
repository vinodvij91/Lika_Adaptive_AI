import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { PageHeader } from "@/components/page-header";
import { 
  Database, 
  Brain, 
  ExternalLink, 
  CheckCircle2, 
  XCircle,
  Beaker,
  Target,
  FileText,
  Zap,
  Settings,
  Key,
  Atom
} from "lucide-react";
import { Link } from "wouter";

interface IntegrationStatus {
  openai: {
    configured: boolean;
    capabilities: string[];
  };
  chembl: {
    configured: boolean;
    capabilities: string[];
  };
  pubchem: {
    configured: boolean;
    capabilities: string[];
  };
  uniprot: {
    configured: boolean;
    capabilities: string[];
  };
  quantum: {
    configured: boolean;
    capabilities: string[];
    status: string;
    providers?: { available: number; total: number };
  };
}

export default function IntegrationsPage() {
  const { data: status, isLoading } = useQuery<IntegrationStatus>({
    queryKey: ["/api/integrations/status"],
  });

  const integrations = [
    {
      id: "openai",
      name: "OpenAI GPT-4",
      icon: Brain,
      color: "text-violet-500",
      bgColor: "from-violet-500/20 to-purple-500/10",
      borderColor: "border-violet-500/30",
      description: "AI-powered molecular property predictions including ADMET, drug-likeness, target predictions, and synthesizability scoring.",
      externalUrl: "https://platform.openai.com",
      capabilities: status?.openai.capabilities || [],
      configured: status?.openai.configured || false,
      requiresKey: true,
    },
    {
      id: "chembl",
      name: "ChEMBL Database",
      icon: Beaker,
      color: "text-cyan-500",
      bgColor: "from-cyan-500/20 to-teal-500/10",
      borderColor: "border-cyan-500/30",
      description: "Access 2.4M+ bioactive molecules with bioactivity data, target information, and clinical development status from EMBL-EBI.",
      externalUrl: "https://www.ebi.ac.uk/chembl/",
      capabilities: status?.chembl.capabilities || [],
      configured: status?.chembl.configured || false,
      requiresKey: false,
    },
    {
      id: "pubchem",
      name: "PubChem",
      icon: Database,
      color: "text-emerald-500",
      bgColor: "from-emerald-500/20 to-green-500/10",
      borderColor: "border-emerald-500/30",
      description: "NIH's open chemistry database with 116M+ compounds, molecular properties, synonyms, and computed descriptors.",
      externalUrl: "https://pubchem.ncbi.nlm.nih.gov/",
      capabilities: status?.pubchem.capabilities || [],
      configured: status?.pubchem.configured || false,
      requiresKey: false,
    },
    {
      id: "uniprot",
      name: "UniProt",
      icon: Target,
      color: "text-amber-500",
      bgColor: "from-amber-500/20 to-orange-500/10",
      borderColor: "border-amber-500/30",
      description: "Comprehensive protein sequence and functional information database with 570K+ reviewed entries.",
      externalUrl: "https://www.uniprot.org/",
      capabilities: status?.uniprot.capabilities || [],
      configured: status?.uniprot.configured || false,
      requiresKey: false,
    },
    {
      id: "quantum",
      name: "Quantum Compute",
      icon: Atom,
      color: "text-fuchsia-500",
      bgColor: "from-fuchsia-500/20 to-purple-500/10",
      borderColor: "border-fuchsia-500/30",
      description: "Quantum simulation for molecular energy calculations using VQE and QAOA algorithms. Supports multiple quantum providers.",
      externalUrl: null,
      internalUrl: "/quantum",
      capabilities: status?.quantum?.capabilities || [],
      configured: status?.quantum?.configured || false,
      requiresKey: false,
      providerInfo: status?.quantum?.providers,
    },
  ];

  return (
    <div className="flex-1 overflow-auto">
      <PageHeader 
        title="Integrations" 
        breadcrumbs={[
          { label: "Settings", href: "/settings" },
          { label: "Integrations" }
        ]}
      />

      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Connected Services</h2>
            <p className="text-sm text-muted-foreground">
              These integrations power the advanced features across the platform
            </p>
          </div>
        </div>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <Card key={i}>
                <CardContent className="p-6">
                  <Skeleton className="h-12 w-12 rounded-lg mb-4" />
                  <Skeleton className="h-5 w-32 mb-2" />
                  <Skeleton className="h-4 w-full mb-4" />
                  <Skeleton className="h-8 w-24" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {integrations.map((integration) => {
              const Icon = integration.icon;
              return (
                <Card 
                  key={integration.id} 
                  className={`overflow-hidden bg-gradient-to-br ${integration.bgColor} backdrop-blur-sm`}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-12 h-12 rounded-lg bg-background/50 flex items-center justify-center border ${integration.borderColor}`}>
                          <Icon className={`h-6 w-6 ${integration.color}`} />
                        </div>
                        <div>
                          <CardTitle className="text-lg">{integration.name}</CardTitle>
                          <div className="flex items-center gap-2 mt-1">
                            {integration.configured ? (
                              <Badge variant="default" className="text-xs bg-emerald-500 hover:bg-emerald-600">
                                <CheckCircle2 className="h-3 w-3 mr-1" />
                                Connected
                              </Badge>
                            ) : (
                              <Badge variant="secondary" className="text-xs">
                                <XCircle className="h-3 w-3 mr-1" />
                                Not Configured
                              </Badge>
                            )}
                            {integration.requiresKey && (
                              <Badge variant="outline" className="text-xs">
                                <Key className="h-3 w-3 mr-1" />
                                API Key Required
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="mb-4">
                      {integration.description}
                    </CardDescription>

                    {integration.capabilities.length > 0 && (
                      <div className="mb-4">
                        <div className="text-xs font-medium text-muted-foreground mb-2">Capabilities</div>
                        <div className="flex flex-wrap gap-1">
                          {integration.capabilities.map((cap) => (
                            <Badge key={cap} variant="outline" className="text-[10px]">
                              {cap.replace(/_/g, " ")}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex items-center gap-2">
                      {integration.externalUrl ? (
                        <a href={integration.externalUrl} target="_blank" rel="noopener noreferrer">
                          <Button variant="outline" size="sm" data-testid={`button-view-${integration.id}`}>
                            <ExternalLink className="h-3 w-3 mr-1" />
                            Visit {integration.name.split(" ")[0]}
                          </Button>
                        </a>
                      ) : integration.internalUrl ? (
                        <Link href={integration.internalUrl}>
                          <Button variant="outline" size="sm" data-testid={`button-open-${integration.id}`}>
                            Open {integration.name.split(" ")[0]}
                          </Button>
                        </Link>
                      ) : null}
                      {integration.requiresKey && !integration.configured && (
                        <Button variant="ghost" size="sm" data-testid={`button-configure-${integration.id}`}>
                          <Settings className="h-3 w-3 mr-1" />
                          Configure
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        <Card className="bg-gradient-to-r from-slate-900/50 to-slate-800/50 border-slate-700">
          <CardContent className="p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30">
                <Zap className="h-6 w-6 text-cyan-400" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold mb-1">How Integrations Work</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Lika Sciences connects to external databases and AI services to provide comprehensive molecular analysis. 
                  ChEMBL, PubChem, and UniProt are free public databases that work automatically. 
                  OpenAI requires an API key to enable AI-powered predictions.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5" />
                    <div>
                      <div className="font-medium">Molecule Detail Pages</div>
                      <div className="text-xs text-muted-foreground">Structure visualization, external data lookup</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5" />
                    <div>
                      <div className="font-medium">AI Predictions</div>
                      <div className="text-xs text-muted-foreground">ADMET, drug-likeness, target predictions</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5" />
                    <div>
                      <div className="font-medium">Target Research</div>
                      <div className="text-xs text-muted-foreground">Protein data, bioactivity information</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
