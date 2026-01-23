import { useLocation } from "wouter";
import { useDomain, type DiscoveryDomain } from "@/contexts/domain-context";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FlaskConical, Hexagon, ArrowRight } from "lucide-react";

const domains: {
  id: DiscoveryDomain;
  title: string;
  description: string;
  icon: typeof FlaskConical;
  features: string[];
}[] = [
  {
    id: "drug",
    title: "Drug Discovery",
    description: "Small-molecule design, screening, docking, multi-target campaigns, assays, SAR.",
    icon: FlaskConical,
    features: [
      "Small molecule & PROTAC design",
      "Virtual screening workflows",
      "Multi-target SAR analysis",
      "ADMET prediction",
      "Assay data management",
    ],
  },
  {
    id: "materials",
    title: "Materials Science",
    description: "Polymers, crystals, composites, property-first pipelines, multi-scale representations.",
    icon: Hexagon,
    features: [
      "Polymer & crystal design",
      "Property-first discovery",
      "Multi-scale representations",
      "Manufacturability scoring",
      "Structure-property analysis",
    ],
  },
];

export default function DomainSelectionPage() {
  const { setDomain } = useDomain();
  const [, navigate] = useLocation();

  const handleSelectDomain = (domain: DiscoveryDomain) => {
    setDomain(domain);
    navigate(`/dashboard/${domain}`);
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="w-full max-w-4xl">
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="w-12 h-12 rounded-lg bg-primary flex items-center justify-center">
              <FlaskConical className="h-7 w-7 text-primary-foreground" />
            </div>
          </div>
          <h1 className="text-3xl font-bold mb-2" data-testid="text-welcome-title">
            Welcome to Lika Sciences
          </h1>
          <p className="text-muted-foreground text-lg max-w-xl mx-auto">
            Choose your discovery domain to get started. You can switch between domains anytime.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {domains.map((domain) => (
            <Card
              key={domain.id}
              className="relative overflow-visible group cursor-pointer hover-elevate transition-all"
              onClick={() => handleSelectDomain(domain.id)}
              data-testid={`card-domain-${domain.id}`}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3 mb-2">
                  <div className={`w-11 h-11 rounded-lg flex items-center justify-center ${
                    domain.id === "drug" ? "bg-primary/10" : "bg-chart-2/10"
                  }`}>
                    <domain.icon className={`h-6 w-6 ${
                      domain.id === "drug" ? "text-primary" : "text-chart-2"
                    }`} />
                  </div>
                  <div>
                    <CardTitle className="text-xl">{domain.title}</CardTitle>
                  </div>
                </div>
                <CardDescription className="text-base">
                  {domain.description}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-2 mb-4">
                  {domain.features.map((feature) => (
                    <li key={feature} className="flex items-center gap-2 text-sm text-muted-foreground">
                      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50" />
                      {feature}
                    </li>
                  ))}
                </ul>
                <Button className="w-full gap-2" data-testid={`button-select-${domain.id}`}>
                  Select {domain.title}
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
