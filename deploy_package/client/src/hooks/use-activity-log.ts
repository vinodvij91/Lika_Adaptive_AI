import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";

type ActivityType = 
  | "user_action"
  | "system_response"
  | "navigation"
  | "data_import"
  | "analysis_run"
  | "campaign_action"
  | "molecule_action"
  | "target_action"
  | "pipeline_action"
  | "error"
  | "auth";

interface LogActivityParams {
  activityType: ActivityType;
  action: string;
  description?: string;
  metadata?: Record<string, any>;
  entityType?: string;
  entityId?: string;
}

export function useActivityLog() {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: async (params: LogActivityParams) => {
      const response = await apiRequest("POST", "/api/activity-logs", params);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/activity-logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/activity-logs/stats"] });
    },
  });

  const logActivity = (params: LogActivityParams) => {
    mutation.mutate(params);
  };

  const logUserAction = (action: string, description?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "user_action", action, description, metadata });
  };

  const logNavigation = (path: string, pageName?: string) => {
    logActivity({
      activityType: "navigation",
      action: `Navigated to ${pageName || path}`,
      description: `User visited ${path}`,
      metadata: { path, pageName },
    });
  };

  const logDataImport = (action: string, description?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "data_import", action, description, metadata });
  };

  const logAnalysisRun = (action: string, description?: string, entityType?: string, entityId?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "analysis_run", action, description, entityType, entityId, metadata });
  };

  const logCampaignAction = (action: string, campaignId?: string, description?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "campaign_action", action, description, entityType: "campaign", entityId: campaignId, metadata });
  };

  const logMoleculeAction = (action: string, moleculeId?: string, description?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "molecule_action", action, description, entityType: "molecule", entityId: moleculeId, metadata });
  };

  const logTargetAction = (action: string, targetId?: string, description?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "target_action", action, description, entityType: "target", entityId: targetId, metadata });
  };

  const logPipelineAction = (action: string, jobId?: string, description?: string, metadata?: Record<string, any>) => {
    logActivity({ activityType: "pipeline_action", action, description, entityType: "pipeline_job", entityId: jobId, metadata });
  };

  const logError = (action: string, error: Error | string, metadata?: Record<string, any>) => {
    const errorMessage = error instanceof Error ? error.message : error;
    logActivity({
      activityType: "error",
      action,
      description: errorMessage,
      metadata: { ...metadata, errorStack: error instanceof Error ? error.stack : undefined },
    });
  };

  const logAuth = (action: string, description?: string) => {
    logActivity({ activityType: "auth", action, description });
  };

  return {
    logActivity,
    logUserAction,
    logNavigation,
    logDataImport,
    logAnalysisRun,
    logCampaignAction,
    logMoleculeAction,
    logTargetAction,
    logPipelineAction,
    logError,
    logAuth,
    isLogging: mutation.isPending,
  };
}
