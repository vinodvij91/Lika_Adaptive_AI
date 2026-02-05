import { useEffect, useState } from "react";
import { Activity, Cpu, Database, Zap } from "lucide-react";

interface MetricProps {
  icon: typeof Activity;
  value: number;
  suffix: string;
  label: string;
  duration?: number;
}

function AnimatedMetric({ icon: Icon, value, suffix, label, duration = 2000 }: MetricProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const startTime = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(Math.floor(value * eased));
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  }, [value, duration]);

  return (
    <div className="flex items-center gap-3 px-6 py-4">
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/20">
        <Icon className="h-5 w-5 text-cyan-400" />
      </div>
      <div>
        <div className="text-2xl font-bold text-foreground">
          {displayValue.toLocaleString()}{suffix}
        </div>
        <div className="text-xs text-muted-foreground">{label}</div>
      </div>
    </div>
  );
}

export function MetricsStrip() {
  return (
    <div className="relative border-y bg-gradient-to-r from-background via-muted/30 to-background">
      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-transparent to-amber-500/5" />
      <div className="relative max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-border/50">
          <AnimatedMetric
            icon={Database}
            value={1700000}
            suffix="+"
            label="SMILES Library"
          />
          <AnimatedMetric
            icon={Activity}
            value={500}
            suffix="K+"
            label="Materials Database"
            duration={1800}
          />
          <AnimatedMetric
            icon={Cpu}
            value={750}
            suffix="+"
            label="Disease Targets"
            duration={1500}
          />
          <AnimatedMetric
            icon={Zap}
            value={94}
            suffix=".7%"
            label="Model Accuracy"
            duration={2200}
          />
        </div>
      </div>
    </div>
  );
}
