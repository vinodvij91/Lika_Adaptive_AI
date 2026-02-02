import { Badge } from "@/components/ui/badge";

interface IntegrationBadgeProps {
  name: string;
  category: string;
}

function IntegrationBadge({ name, category }: IntegrationBadgeProps) {
  return (
    <div className="group relative px-6 py-4 rounded-lg bg-gradient-to-br from-muted/50 to-transparent border border-border/50 hover-elevate transition-all duration-300">
      <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-cyan-500/5 to-amber-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="relative flex flex-col items-center gap-2">
        <span className="text-sm font-semibold text-foreground">{name}</span>
        <Badge variant="secondary" className="text-xs">
          {category}
        </Badge>
      </div>
    </div>
  );
}

export function IntegrationLogos() {
  const integrations = [
    { name: "NVIDIA BioNeMo", category: "AI/ML" },
    { name: "IonQ Quantum", category: "Quantum" },
    { name: "IBM Quantum", category: "Quantum" },
    { name: "RDKit", category: "Chemistry" },
    { name: "OpenMM", category: "Simulation" },
    { name: "DeepChem", category: "ML" },
  ];

  return (
    <section className="py-16 border-t bg-gradient-to-b from-background to-muted/20">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-10">
          <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-2">
            Powered by Industry Leaders
          </p>
          <h3 className="text-xl font-semibold">Integrated Technologies</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {integrations.map((integration) => (
            <IntegrationBadge
              key={integration.name}
              name={integration.name}
              category={integration.category}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

export function TrustSignals() {
  return (
    <section className="py-12 border-t">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-wrap items-center justify-center gap-8 text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm">SOC 2 Compliant</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" style={{ animationDelay: '0.5s' }} />
            <span className="text-sm">HIPAA Ready</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" style={{ animationDelay: '1s' }} />
            <span className="text-sm">Enterprise Security</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" style={{ animationDelay: '1.5s' }} />
            <span className="text-sm">99.9% Uptime SLA</span>
          </div>
        </div>
      </div>
    </section>
  );
}
