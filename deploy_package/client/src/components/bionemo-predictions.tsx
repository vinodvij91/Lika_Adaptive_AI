import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Brain,
  Sparkles,
  Target,
  FlaskConical,
  Activity,
  Beaker,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
  Zap,
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface BioNemoPredictionsProps {
  smiles: string;
  moleculeName?: string;
  showDocking?: boolean;
  compact?: boolean;
}

interface PropertyPrediction {
  smiles: string;
  qed: number;
  plogP: number;
  molecularWeight: number;
  synthesizability: number;
  drugLikeness: string;
  confidence: number;
}

interface DockingPrediction {
  moleculeSmiles: string;
  targetId: string;
  bindingAffinity: number;
  poseScore: number;
  confidence: number;
}

export function BioNemoPredictions({ 
  smiles, 
  moleculeName,
  showDocking = true,
  compact = false 
}: BioNemoPredictionsProps) {
  const { toast } = useToast();
  const [prediction, setPrediction] = useState<PropertyPrediction | null>(null);
  const [dockingPrediction, setDockingPrediction] = useState<DockingPrediction | null>(null);

  const statusQuery = useQuery<{ configured: boolean; provider: string; capabilities: string[] }>({
    queryKey: ["/api/bionemo/status"],
  });

  const propertyMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/bionemo/predict/properties", { smiles });
      return response.json();
    },
    onSuccess: (data) => {
      setPrediction(data);
      toast({
        title: "Prediction Complete",
        description: "BioNemo property predictions generated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Prediction Failed",
        description: error.message || "Failed to generate predictions",
        variant: "destructive",
      });
    },
  });

  const dockingMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/bionemo/predict/docking", { smiles });
      return response.json();
    },
    onSuccess: (data) => {
      setDockingPrediction(data);
      toast({
        title: "Docking Complete",
        description: "BioNemo docking prediction generated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Docking Failed",
        description: error.message || "Failed to generate docking prediction",
        variant: "destructive",
      });
    },
  });

  const isConfigured = statusQuery.data?.configured;
  const isLoading = propertyMutation.isPending || dockingMutation.isPending;

  const runPredictions = () => {
    propertyMutation.mutate();
    if (showDocking) {
      dockingMutation.mutate();
    }
  };

  const getQedColor = (qed: number) => {
    if (qed >= 0.7) return "text-green-600 dark:text-green-400";
    if (qed >= 0.5) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const getDrugLikenessVariant = (likeness: string): "default" | "secondary" | "destructive" => {
    if (likeness === "High") return "default";
    if (likeness === "Moderate") return "secondary";
    return "destructive";
  };

  if (statusQuery.isLoading) {
    return (
      <Card className={compact ? "p-3" : ""}>
        <CardContent className="py-4">
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-muted-foreground">Checking BioNemo status...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!isConfigured) {
    return (
      <Card className={compact ? "p-3" : ""}>
        <CardContent className="py-4">
          <div className="flex items-center gap-2 text-muted-foreground">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">BioNemo API not configured</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">NVIDIA BioNemo</span>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={runPredictions}
            disabled={isLoading}
            data-testid="button-run-bionemo-compact"
          >
            {isLoading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Zap className="h-3 w-3" />
            )}
          </Button>
        </div>
        
        {prediction && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">QED:</span>
              <span className={getQedColor(prediction.qed)}>{prediction.qed.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">LogP:</span>
              <span>{prediction.plogP.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">MW:</span>
              <span>{prediction.molecularWeight.toFixed(0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Synth:</span>
              <span>{(prediction.synthesizability * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Brain className="h-5 w-5 text-primary" />
            NVIDIA BioNemo AI Predictions
          </CardTitle>
          <div className="flex items-center gap-2">
            {prediction && (
              <Badge variant="outline" className="text-xs">
                <CheckCircle className="h-3 w-3 mr-1 text-green-500" />
                {(prediction.confidence * 100).toFixed(0)}% confidence
              </Badge>
            )}
            <Button
              size="sm"
              onClick={runPredictions}
              disabled={isLoading}
              data-testid="button-run-bionemo-predictions"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : prediction ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Run Predictions
                </>
              )}
            </Button>
          </div>
        </div>
        {moleculeName && (
          <p className="text-sm text-muted-foreground">
            Analyzing {moleculeName}
          </p>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {!prediction && !isLoading && (
          <div className="text-center py-6 text-muted-foreground">
            <FlaskConical className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Click "Run Predictions" to analyze this molecule using NVIDIA BioNemo AI</p>
          </div>
        )}

        {isLoading && (
          <div className="space-y-3">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
            <div className="grid grid-cols-2 gap-4 mt-4">
              <Skeleton className="h-20" />
              <Skeleton className="h-20" />
            </div>
          </div>
        )}

        {prediction && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 space-y-1" data-testid="card-qed-score">
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Activity className="h-3 w-3" />
                      QED Score
                    </div>
                    <div className={`text-xl font-bold ${getQedColor(prediction.qed)}`} data-testid="text-qed-value">
                      {prediction.qed.toFixed(2)}
                    </div>
                    <Progress value={prediction.qed * 100} className="h-1" />
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Quantitative Estimate of Drug-likeness</p>
                  <p className="text-xs text-muted-foreground">Higher is better (0-1)</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 space-y-1" data-testid="card-logp-score">
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Beaker className="h-3 w-3" />
                      LogP
                    </div>
                    <div className="text-xl font-bold" data-testid="text-logp-value">
                      {prediction.plogP.toFixed(2)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {prediction.plogP >= -0.4 && prediction.plogP <= 5.6 ? "Optimal range" : "Outside optimal"}
                    </p>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Partition Coefficient (lipophilicity)</p>
                  <p className="text-xs text-muted-foreground">Optimal: -0.4 to 5.6</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 space-y-1">
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <FlaskConical className="h-3 w-3" />
                      Mol Weight
                    </div>
                    <div className="text-xl font-bold">
                      {prediction.molecularWeight.toFixed(0)}
                    </div>
                    <p className="text-xs text-muted-foreground">Da</p>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Molecular Weight in Daltons</p>
                  <p className="text-xs text-muted-foreground">Drug-like: &lt;500 Da</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 space-y-1">
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Sparkles className="h-3 w-3" />
                      Synthesizability
                    </div>
                    <div className="text-xl font-bold">
                      {(prediction.synthesizability * 100).toFixed(0)}%
                    </div>
                    <Progress value={prediction.synthesizability * 100} className="h-1" />
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Ease of synthesis score</p>
                  <p className="text-xs text-muted-foreground">Higher means easier to synthesize</p>
                </TooltipContent>
              </Tooltip>
            </div>

            <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/30" data-testid="card-drug-likeness">
              <Target className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Drug-Likeness Assessment:</span>
              <Badge variant={getDrugLikenessVariant(prediction.drugLikeness)} data-testid="badge-drug-likeness">
                {prediction.drugLikeness}
              </Badge>
            </div>
          </div>
        )}

        {showDocking && dockingPrediction && (
          <div className="mt-4 pt-4 border-t">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Target className="h-4 w-4" />
              Docking Prediction
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-xs text-muted-foreground mb-1">Binding Affinity</div>
                <div className="text-lg font-bold">
                  {dockingPrediction.bindingAffinity.toFixed(2)} kcal/mol
                </div>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-xs text-muted-foreground mb-1">Pose Score</div>
                <div className="text-lg font-bold">
                  {(dockingPrediction.poseScore * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="text-xs text-muted-foreground flex items-center gap-1 pt-2">
          <Brain className="h-3 w-3" />
          Powered by NVIDIA BioNemo MolMIM
        </div>
      </CardContent>
    </Card>
  );
}

export function BioNemoStatusBadge() {
  const statusQuery = useQuery<{ configured: boolean; provider: string }>({
    queryKey: ["/api/bionemo/status"],
  });

  if (statusQuery.isLoading) {
    return <Badge variant="outline"><Loader2 className="h-3 w-3 animate-spin" /></Badge>;
  }

  if (!statusQuery.data?.configured) {
    return (
      <Badge variant="secondary" className="text-xs">
        <AlertCircle className="h-3 w-3 mr-1" />
        BioNemo Offline
      </Badge>
    );
  }

  return (
    <Badge variant="default" className="text-xs">
      <CheckCircle className="h-3 w-3 mr-1" />
      BioNemo Active
    </Badge>
  );
}
