import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Brain, 
  Sparkles, 
  AlertCircle, 
  RefreshCw,
  Target,
  Beaker,
  Shield,
  FlaskConical,
  Zap,
  CheckCircle2,
  XCircle,
  AlertTriangle
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

interface AIPredictionsPanelProps {
  smiles: string;
  moleculeName?: string;
}

interface MoleculePrediction {
  smiles: string;
  predictions: {
    drugLikeness: {
      score: number;
      lipinskiViolations: number;
      molecularWeight: string;
      logP: string;
      hbdCount: number;
      hbaCount: number;
      verdict: string;
    };
    admet: {
      absorption: { score: number; details: string };
      distribution: { score: number; details: string };
      metabolism: { score: number; details: string };
      excretion: { score: number; details: string };
      toxicity: { score: number; details: string; alerts: string[] };
    };
    targetPredictions: Array<{
      targetName: string;
      confidence: number;
      mechanism: string;
    }>;
    synthesizability: {
      score: number;
      complexity: string;
      estimatedSteps: number;
    };
    summary: string;
  };
  confidence: number;
  generatedAt: string;
}

function ScoreIndicator({ score, label }: { score: number; label: string }) {
  const getColor = (s: number) => {
    if (s >= 80) return "text-emerald-500";
    if (s >= 60) return "text-amber-500";
    return "text-red-500";
  };

  const getIcon = (s: number) => {
    if (s >= 80) return <CheckCircle2 className="h-3 w-3" />;
    if (s >= 60) return <AlertTriangle className="h-3 w-3" />;
    return <XCircle className="h-3 w-3" />;
  };

  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className={`flex items-center gap-1 ${getColor(score)}`}>
        {getIcon(score)}
        <span className="text-xs font-medium">{score}</span>
      </div>
    </div>
  );
}

export function AIPredictionsPanel({ smiles, moleculeName }: AIPredictionsPanelProps) {
  const [prediction, setPrediction] = useState<MoleculePrediction | null>(null);

  const predictMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/predictions/molecule", { smiles });
      return response.json();
    },
    onSuccess: (data) => {
      setPrediction(data);
    },
  });

  const hasRun = prediction !== null || predictMutation.isPending || predictMutation.isError;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Brain className="h-4 w-4 text-violet-500" />
            AI Property Predictions
          </CardTitle>
          {prediction && (
            <Badge variant="secondary" className="text-[10px]">
              {Math.round(prediction.confidence)}% confidence
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!hasRun ? (
          <div className="text-center py-6">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-500/10 flex items-center justify-center border border-violet-500/30">
              <Sparkles className="h-8 w-8 text-violet-400" />
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Generate AI-powered predictions for drug-likeness, ADMET properties, and target predictions
            </p>
            <Button 
              onClick={() => predictMutation.mutate()}
              className="gap-2"
              data-testid="button-generate-predictions"
            >
              <Zap className="h-4 w-4" />
              Generate Predictions
            </Button>
          </div>
        ) : predictMutation.isPending ? (
          <div className="py-6 space-y-4">
            <div className="flex items-center justify-center gap-2 text-muted-foreground">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span>Analyzing molecular structure...</span>
            </div>
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          </div>
        ) : predictMutation.isError ? (
          <div className="text-center py-6 text-muted-foreground">
            <AlertCircle className="h-8 w-8 mx-auto mb-2 text-red-500" />
            <p className="text-sm mb-2">Failed to generate predictions</p>
            <p className="text-xs text-red-400 mb-4">
              {(predictMutation.error as Error)?.message || "OpenAI API key may not be configured"}
            </p>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => predictMutation.mutate()}
              data-testid="button-retry-predictions"
            >
              <RefreshCw className="h-3 w-3 mr-1" />
              Retry
            </Button>
          </div>
        ) : prediction ? (
          <ScrollArea className="h-80">
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-violet-500/10 to-purple-500/10 p-3 rounded-lg border border-violet-500/20">
                <p className="text-sm">{prediction.predictions.summary}</p>
              </div>

              <div>
                <div className="text-xs font-medium mb-2 flex items-center gap-1">
                  <FlaskConical className="h-3 w-3 text-cyan-500" />
                  Drug-Likeness
                </div>
                <div className="bg-muted/30 p-3 rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{prediction.predictions.drugLikeness.verdict}</span>
                    <Badge 
                      variant={prediction.predictions.drugLikeness.score >= 70 ? "default" : "secondary"}
                    >
                      Score: {prediction.predictions.drugLikeness.score}
                    </Badge>
                  </div>
                  <Progress value={prediction.predictions.drugLikeness.score} className="h-1" />
                  <div className="grid grid-cols-2 gap-2 text-xs mt-2">
                    <div>
                      <span className="text-muted-foreground">Lipinski Violations: </span>
                      <span className="font-medium">{prediction.predictions.drugLikeness.lipinskiViolations}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">MW: </span>
                      <span className="font-medium">{prediction.predictions.drugLikeness.molecularWeight}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">LogP: </span>
                      <span className="font-medium">{prediction.predictions.drugLikeness.logP}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">HBD/HBA: </span>
                      <span className="font-medium">
                        {prediction.predictions.drugLikeness.hbdCount}/{prediction.predictions.drugLikeness.hbaCount}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <div className="text-xs font-medium mb-2 flex items-center gap-1">
                  <Shield className="h-3 w-3 text-emerald-500" />
                  ADMET Profile
                </div>
                <div className="bg-muted/30 p-3 rounded-lg space-y-2">
                  <ScoreIndicator score={prediction.predictions.admet.absorption.score} label="Absorption" />
                  <ScoreIndicator score={prediction.predictions.admet.distribution.score} label="Distribution" />
                  <ScoreIndicator score={prediction.predictions.admet.metabolism.score} label="Metabolism" />
                  <ScoreIndicator score={prediction.predictions.admet.excretion.score} label="Excretion" />
                  <ScoreIndicator score={prediction.predictions.admet.toxicity.score} label="Toxicity (higher = safer)" />
                  
                  {prediction.predictions.admet.toxicity.alerts.length > 0 && (
                    <div className="pt-2 border-t border-border mt-2">
                      <div className="text-xs text-red-400 mb-1">Structural Alerts:</div>
                      <div className="flex flex-wrap gap-1">
                        {prediction.predictions.admet.toxicity.alerts.map((alert, i) => (
                          <Badge key={i} variant="destructive" className="text-[10px]">
                            {alert}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {prediction.predictions.targetPredictions.length > 0 && (
                <div>
                  <div className="text-xs font-medium mb-2 flex items-center gap-1">
                    <Target className="h-3 w-3 text-amber-500" />
                    Predicted Targets
                  </div>
                  <div className="space-y-1.5">
                    {prediction.predictions.targetPredictions.map((target, i) => (
                      <div key={i} className="bg-muted/30 p-2 rounded text-xs flex items-center justify-between">
                        <div>
                          <span className="font-medium">{target.targetName}</span>
                          <span className="text-muted-foreground ml-2">({target.mechanism})</span>
                        </div>
                        <Badge variant="outline">{target.confidence}%</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <div className="text-xs font-medium mb-2 flex items-center gap-1">
                  <Beaker className="h-3 w-3 text-blue-500" />
                  Synthesizability
                </div>
                <div className="bg-muted/30 p-3 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm">{prediction.predictions.synthesizability.complexity} Complexity</span>
                    <Badge variant={prediction.predictions.synthesizability.score >= 70 ? "default" : "secondary"}>
                      Score: {prediction.predictions.synthesizability.score}
                    </Badge>
                  </div>
                  <Progress value={prediction.predictions.synthesizability.score} className="h-1" />
                  <div className="text-xs text-muted-foreground mt-2">
                    Estimated {prediction.predictions.synthesizability.estimatedSteps} synthetic steps
                  </div>
                </div>
              </div>

              <div className="text-xs text-muted-foreground text-center pt-2">
                Generated {new Date(prediction.generatedAt).toLocaleString()}
              </div>

              <Button 
                variant="outline" 
                size="sm" 
                className="w-full"
                onClick={() => predictMutation.mutate()}
                data-testid="button-regenerate-predictions"
              >
                <RefreshCw className="h-3 w-3 mr-1" />
                Regenerate Predictions
              </Button>
            </div>
          </ScrollArea>
        ) : null}
      </CardContent>
    </Card>
  );
}
