import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Atom,
  Cpu,
  Server,
  Play,
  Activity,
  Zap,
  TrendingDown,
  CheckCircle,
  AlertCircle,
  Clock,
  Gauge
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

interface QuantumProvider {
  id: string;
  name: string;
  type: "simulator" | "hardware";
  qubits: number;
  status: "available" | "unavailable" | "maintenance";
  capabilities: string[];
}

interface VQEResult {
  groundStateEnergy: number;
  optimizedAngles: number[];
  convergenceHistory: number[];
  chemicalAccuracy: boolean;
}

export default function QuantumComputePage() {
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [smiles, setSmiles] = useState("O=O");
  const [basis, setBasis] = useState<"sto-3g" | "6-31g" | "cc-pvdz">("sto-3g");
  const [vqeResult, setVqeResult] = useState<VQEResult | null>(null);

  const { data: providersData, isLoading: providersLoading } = useQuery<{ providers: QuantumProvider[] }>({
    queryKey: ["/api/quantum/providers"],
  });

  const { data: statusData } = useQuery<{
    configured: boolean;
    status: string;
    message: string;
    providers: { available: number; total: number };
  }>({
    queryKey: ["/api/quantum/status"],
  });

  const estimateQubitsMutation = useMutation({
    mutationFn: async (params: { smiles: string; basis: string }) => {
      const response = await apiRequest("POST", "/api/quantum/estimate-qubits", params);
      return response.json();
    },
  });

  const runVQEMutation = useMutation({
    mutationFn: async (params: { smiles: string; basis: string; maxIterations: number }) => {
      const response = await apiRequest("POST", "/api/quantum/vqe-simulation", params);
      return response.json();
    },
    onSuccess: (data) => {
      setVqeResult(data);
    },
  });

  const handleEstimateQubits = () => {
    estimateQubitsMutation.mutate({ smiles, basis });
  };

  const handleRunVQE = () => {
    runVQEMutation.mutate({ smiles, basis, maxIterations: 100 });
  };

  const availableProviders = providersData?.providers?.filter(p => p.status === "available") || [];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <PageHeader 
        title="Quantum Compute" 
        breadcrumbs={[
          { label: "Integrations" },
          { label: "Quantum Compute" }
        ]}
      />

      <div className="flex-1 overflow-auto p-6">
        <div className="mb-6">
          <Card className="bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10 border-violet-500/20">
            <CardContent className="pt-6">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Atom className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg">Quantum Compute Integration</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    {statusData?.message || "Quantum simulation for molecular energy calculations using VQE and QAOA algorithms"}
                  </p>
                  <div className="flex items-center gap-4 mt-3">
                    <Badge variant={statusData?.configured ? "default" : "secondary"}>
                      {statusData?.status === "demo_mode" ? "Demo Mode" : statusData?.status || "Loading..."}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {statusData?.providers?.available || 0} of {statusData?.providers?.total || 0} providers available
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="providers" className="space-y-4">
          <TabsList>
            <TabsTrigger value="providers" data-testid="tab-providers">
              <Server className="h-4 w-4 mr-2" />
              Providers
            </TabsTrigger>
            <TabsTrigger value="simulation" data-testid="tab-simulation">
              <Activity className="h-4 w-4 mr-2" />
              VQE Simulation
            </TabsTrigger>
          </TabsList>

          <TabsContent value="providers">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {providersLoading ? (
                <>
                  <Skeleton className="h-48" />
                  <Skeleton className="h-48" />
                  <Skeleton className="h-48" />
                </>
              ) : (
                providersData?.providers?.map((provider) => (
                  <Card 
                    key={provider.id}
                    className={`${selectedProvider === provider.id ? "border-primary" : ""}`}
                  >
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base">{provider.name}</CardTitle>
                        <Badge 
                          variant={
                            provider.status === "available" ? "default" :
                            provider.status === "maintenance" ? "secondary" : "outline"
                          }
                        >
                          {provider.status}
                        </Badge>
                      </div>
                      <CardDescription className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {provider.type}
                        </Badge>
                        <span>{provider.qubits} qubits</span>
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-wrap gap-1 mb-3">
                        {provider.capabilities.map((cap) => (
                          <Badge key={cap} variant="secondary" className="text-[10px]">
                            {cap.toUpperCase()}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        {provider.type === "simulator" ? (
                          <Cpu className="h-3 w-3" />
                        ) : (
                          <Zap className="h-3 w-3" />
                        )}
                        <span>
                          {provider.type === "simulator" 
                            ? "Simulated quantum computing" 
                            : "Real quantum hardware"}
                        </span>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button 
                        variant={selectedProvider === provider.id ? "default" : "outline"}
                        size="sm"
                        className="w-full"
                        disabled={provider.status !== "available"}
                        onClick={() => setSelectedProvider(provider.id)}
                        data-testid={`button-select-provider-${provider.id}`}
                      >
                        {selectedProvider === provider.id ? "Selected" : "Select Provider"}
                      </Button>
                    </CardFooter>
                  </Card>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="simulation">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Atom className="h-4 w-4 text-violet-500" />
                    VQE Simulation
                  </CardTitle>
                  <CardDescription>
                    Variational Quantum Eigensolver for ground state energy calculation
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="smiles">SMILES Structure</Label>
                    <Input
                      id="smiles"
                      value={smiles}
                      onChange={(e) => setSmiles(e.target.value)}
                      placeholder="Enter SMILES (e.g., O=O for O2)"
                      data-testid="input-smiles"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="basis">Basis Set</Label>
                    <Select value={basis} onValueChange={(v: any) => setBasis(v)}>
                      <SelectTrigger id="basis" data-testid="select-basis">
                        <SelectValue placeholder="Select basis set" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sto-3g">STO-3G (minimal)</SelectItem>
                        <SelectItem value="6-31g">6-31G (split-valence)</SelectItem>
                        <SelectItem value="cc-pvdz">cc-pVDZ (correlation consistent)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={handleEstimateQubits}
                      disabled={!smiles || estimateQubitsMutation.isPending}
                      data-testid="button-estimate-qubits"
                    >
                      <Gauge className="h-4 w-4 mr-2" />
                      Estimate Qubits
                    </Button>
                    <Button
                      onClick={handleRunVQE}
                      disabled={!smiles || runVQEMutation.isPending}
                      data-testid="button-run-vqe"
                    >
                      {runVQEMutation.isPending ? (
                        <>
                          <Activity className="h-4 w-4 mr-2 animate-pulse" />
                          Running...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-2" />
                          Run Simulation
                        </>
                      )}
                    </Button>
                  </div>
                  {estimateQubitsMutation.data && (
                    <div className="bg-muted/50 p-3 rounded-lg">
                      <div className="text-sm font-medium">Qubit Estimate</div>
                      <div className="text-2xl font-bold text-violet-500">
                        {estimateQubitsMutation.data.estimatedQubits} qubits
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Required for {estimateQubitsMutation.data.basis} basis set
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <TrendingDown className="h-4 w-4 text-emerald-500" />
                    Simulation Results
                  </CardTitle>
                  <CardDescription>
                    Ground state energy and convergence analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {!vqeResult && !runVQEMutation.isPending && (
                    <div className="text-center py-8 text-muted-foreground">
                      <Atom className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm">Run a VQE simulation to see results</p>
                    </div>
                  )}
                  {runVQEMutation.isPending && (
                    <div className="text-center py-8">
                      <Activity className="h-12 w-12 mx-auto mb-3 animate-pulse text-violet-500" />
                      <p className="text-sm text-muted-foreground">Computing ground state energy...</p>
                    </div>
                  )}
                  {vqeResult && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gradient-to-br from-violet-500/10 to-purple-500/10 p-4 rounded-lg border border-violet-500/20">
                          <div className="text-xs text-muted-foreground mb-1">Ground State Energy</div>
                          <div className="text-2xl font-bold text-violet-500">
                            {vqeResult.groundStateEnergy.toFixed(4)} Ha
                          </div>
                        </div>
                        <div className="bg-muted/50 p-4 rounded-lg">
                          <div className="text-xs text-muted-foreground mb-1">Chemical Accuracy</div>
                          <div className="flex items-center gap-2">
                            {vqeResult.chemicalAccuracy ? (
                              <CheckCircle className="h-5 w-5 text-emerald-500" />
                            ) : (
                              <AlertCircle className="h-5 w-5 text-amber-500" />
                            )}
                            <span className="font-medium">
                              {vqeResult.chemicalAccuracy ? "Achieved" : "Not achieved"}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <div className="text-sm font-medium mb-2">Convergence History</div>
                        <div className="h-24 bg-muted/50 rounded-lg p-2 flex items-end gap-px">
                          {vqeResult.convergenceHistory.slice(-30).map((energy, i) => {
                            const min = Math.min(...vqeResult.convergenceHistory);
                            const max = Math.max(...vqeResult.convergenceHistory);
                            const range = max - min || 1;
                            const height = ((energy - min) / range) * 100;
                            return (
                              <div
                                key={i}
                                className="flex-1 bg-violet-500/70 rounded-t"
                                style={{ height: `${100 - height}%` }}
                              />
                            );
                          })}
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground mt-1">
                          <span>Iteration 1</span>
                          <span>Iteration {vqeResult.convergenceHistory.length}</span>
                        </div>
                      </div>

                      <div>
                        <div className="text-sm font-medium mb-2">Optimized Angles</div>
                        <div className="flex flex-wrap gap-2">
                          {vqeResult.optimizedAngles.map((angle, i) => (
                            <Badge key={i} variant="outline" className="font-mono">
                              θ{i + 1} = {angle.toFixed(4)} rad
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
