import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { DiseaseArea } from "@shared/schema";
import { Brain, HeartPulse, Microscope, Bug, Wind, Flame, Dna, HelpCircle } from "lucide-react";

interface DiseaseAreaBadgeProps {
  area: DiseaseArea;
  className?: string;
  showIcon?: boolean;
}

const areaStyles: Record<DiseaseArea, { bg: string; icon: typeof Brain }> = {
  CNS: { bg: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400", icon: Brain },
  Oncology: { bg: "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-400", icon: Microscope },
  Rare: { bg: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-400", icon: Dna },
  Infectious: { bg: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400", icon: Bug },
  Cardiometabolic: { bg: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400", icon: HeartPulse },
  Autoimmune: { bg: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400", icon: Flame },
  Respiratory: { bg: "bg-sky-100 text-sky-800 dark:bg-sky-900/30 dark:text-sky-400", icon: Wind },
  Other: { bg: "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400", icon: HelpCircle },
};

export function DiseaseAreaBadge({ area, className, showIcon = true }: DiseaseAreaBadgeProps) {
  const { bg, icon: Icon } = areaStyles[area] || areaStyles.Other;

  return (
    <Badge
      variant="outline"
      className={cn(
        "text-xs font-medium no-default-hover-elevate no-default-active-elevate gap-1",
        bg,
        className
      )}
      data-testid={`badge-disease-${area}`}
    >
      {showIcon && <Icon className="h-3 w-3" />}
      {area}
    </Badge>
  );
}
