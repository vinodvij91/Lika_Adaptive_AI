import { useLocation } from "wouter";
import { useDomain, type DiscoveryDomain } from "@/contexts/domain-context";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FlaskConical, Hexagon, ArrowRight, Check, Syringe } from "lucide-react";
import { LikaLogoLeafGradient } from "@/components/lika-logo";

const domains: {
  id: DiscoveryDomain;
  title: string;
  subtitle: string;
  description: string;
  icon: typeof FlaskConical;
  features: string[];
  gradient: string;
  accentColor: string;
}[] = [
  {
    id: "drug",
    title: "Drug Discovery",
    subtitle: "Accelerate therapeutic development",
    description: "Design and screen small molecules, PROTACs, and peptides with AI-powered virtual screening and multi-target SAR analysis.",
    icon: FlaskConical,
    features: [
      "Small molecule & PROTAC design",
      "Virtual screening workflows",
      "Multi-target SAR analysis",
      "ADMET prediction",
      "Assay data management",
    ],
    gradient: "from-blue-600 via-blue-500 to-cyan-500",
    accentColor: "text-blue-500",
  },
  {
    id: "vaccine",
    title: "Vaccine Discovery",
    subtitle: "Design next-gen immunogens",
    description: "Predict epitopes, optimize MHC binding, and design multi-epitope constructs with integrated bioinformatics tools.",
    icon: Syringe,
    features: [
      "Epitope prediction & MHC binding",
      "B-cell & T-cell analysis",
      "mRNA vaccine design",
      "Codon optimization",
      "Immunogenicity simulation",
    ],
    gradient: "from-purple-600 via-violet-500 to-indigo-500",
    accentColor: "text-purple-500",
  },
  {
    id: "materials",
    title: "Materials Science",
    subtitle: "Discover next-gen materials",
    description: "Explore polymers, crystals, and composites with property-first pipelines, PFAS replacement, and manufacturability scoring.",
    icon: Hexagon,
    features: [
      "PFAS replacement screening",
      "Spider silk & biomaterials",
      "Battery & photovoltaic design",
      "Property-first discovery",
      "OpenFold3 structure prediction",
    ],
    gradient: "from-emerald-600 via-teal-500 to-cyan-500",
    accentColor: "text-emerald-500",
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
    <div className="min-h-screen bg-background relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-chart-2/5" />
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-chart-2/10 rounded-full blur-3xl" />
      
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-6">
        <div className="w-full max-w-5xl">
          <div className="text-center mb-12">
            <div className="flex items-center justify-center gap-3 mb-6">
              <LikaLogoLeafGradient size={72} />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-3 bg-gradient-to-r from-foreground via-foreground to-muted-foreground bg-clip-text" data-testid="text-welcome-title">
              Welcome to Lika Sciences
            </h1>
            <p className="text-muted-foreground text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
              AI-powered discovery platform for scientific breakthroughs. 
              Choose your domain to begin.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {domains.map((domain) => (
              <Card
                key={domain.id}
                className="relative overflow-hidden group cursor-pointer border-2 border-transparent hover:border-primary/20 transition-all duration-300"
                onClick={() => handleSelectDomain(domain.id)}
                data-testid={`card-domain-${domain.id}`}
              >
                <div className={`absolute top-0 left-0 right-0 h-2 bg-gradient-to-r ${domain.gradient}`} />
                
                <CardContent className="p-8">
                  <div className="flex items-start gap-4 mb-6">
                    <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${domain.gradient} flex items-center justify-center shadow-lg flex-shrink-0`}>
                      <domain.icon className="h-7 w-7 text-white" />
                    </div>
                    <div className="flex-1">
                      <h2 className="text-2xl font-bold mb-1">{domain.title}</h2>
                      <p className={`text-sm font-medium ${domain.accentColor}`}>
                        {domain.subtitle}
                      </p>
                    </div>
                  </div>
                  
                  <p className="text-muted-foreground mb-6 leading-relaxed">
                    {domain.description}
                  </p>
                  
                  <div className="space-y-3 mb-8">
                    {domain.features.map((feature) => (
                      <div key={feature} className="flex items-center gap-3">
                        <div className={`w-5 h-5 rounded-full bg-gradient-to-br ${domain.gradient} flex items-center justify-center flex-shrink-0`}>
                          <Check className="h-3 w-3 text-white" />
                        </div>
                        <span className="text-sm">{feature}</span>
                      </div>
                    ))}
                  </div>
                  
                  <Button 
                    className={`w-full gap-2 bg-gradient-to-r ${domain.gradient} hover:opacity-90 transition-opacity border-0`}
                    data-testid={`button-select-${domain.id}`}
                  >
                    Get Started with {domain.title}
                    <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="mt-12 text-center">
            <p className="text-sm text-muted-foreground">
              You can switch between domains anytime from the sidebar
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
