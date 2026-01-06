import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { CampaignStatus, JobStatus } from "@shared/schema";

interface StatusBadgeProps {
  status: CampaignStatus | JobStatus;
  className?: string;
}

const statusStyles: Record<string, string> = {
  pending: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
  completed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400",
  failed: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "text-xs font-medium capitalize no-default-hover-elevate no-default-active-elevate",
        statusStyles[status],
        className
      )}
      data-testid={`badge-status-${status}`}
    >
      {status}
    </Badge>
  );
}
