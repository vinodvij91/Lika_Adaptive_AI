import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  Beaker,
  Sparkles,
  Zap,
  Users,
  GitBranch,
  Brain,
  ArrowRight,
  FlaskConical,
  Target,
  BarChart3,
  Atom,
  Layers,
  Settings,
  Shield,
  Hexagon,
  Cpu,
} from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="fixed top-0 left-0 right-0 z-50 border-b bg-background/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-md bg-primary flex items-center justify-center">
              <FlaskConical className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-semibold">Lika Sciences</span>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <a href="/api/login">
              <Button data-testid="button-login">Sign In</Button>
            </a>
          </div>
        </div>
      </header>

      <main>
        <section className="relative pt-32 pb-24 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-chart-3/5" />
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-chart-2/10 rounded-full blur-3xl" />

          <div className="relative max-w-7xl mx-auto px-6">
            <div className="max-w-4xl mx-auto text-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
                <Sparkles className="h-4 w-4" />
                Adaptive AI Discovery Platform
              </div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
                Lika Sciences
                <span className="block text-primary mt-2">Drug Discovery + Advanced Materials</span>
              </h1>
              <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
                Generative design, simulation-in-the-loop, BioNeMo + Molecular ML + Quantum pipelines
                across pharmaceuticals, polymers, catalysts, energy materials, coatings, membranes,
                and next-generation engineered materials.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <a href="/api/login">
                  <Button size="lg" className="gap-2" data-testid="button-explore-drug">
                    Explore Drug Discovery
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </a>
                <a href="/api/login">
                  <Button size="lg" variant="outline" className="gap-2" data-testid="button-explore-materials">
                    Explore Materials Discovery
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </a>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 border-t bg-muted/20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card className="overflow-hidden">
                <CardContent className="p-8">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center">
                      <FlaskConical className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="text-2xl font-bold">Drug Discovery</h3>
                  </div>
                  <p className="text-muted-foreground">
                    End-to-end AI discovery for small molecules, PROTACs, peptides, ADMET, docking,
                    variant-aware scoring, multi-modality pipelines, and simulation-driven iteration.
                  </p>
                  <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-center gap-2"><Target className="h-4 w-4 text-primary" /> BioNeMo Integration</li>
                    <li className="flex items-center gap-2"><Brain className="h-4 w-4 text-primary" /> ML-Guided Docking</li>
                    <li className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-primary" /> ADMET Predictions</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="overflow-hidden">
                <CardContent className="p-8">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 rounded-md bg-chart-2/10 flex items-center justify-center">
                      <Hexagon className="h-6 w-6 text-chart-2" />
                    </div>
                    <h3 className="text-2xl font-bold">Materials Discovery</h3>
                  </div>
                  <p className="text-muted-foreground">
                    AI-guided materials design for polymers, crystals, composites, coatings, membranes,
                    catalysts, and functional materials driven by property-first pipelines and physics-aware simulation.
                  </p>
                  <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-center gap-2"><Layers className="h-4 w-4 text-chart-2" /> Multi-Scale Representations</li>
                    <li className="flex items-center gap-2"><Atom className="h-4 w-4 text-chart-2" /> Property Prediction</li>
                    <li className="flex items-center gap-2"><Settings className="h-4 w-4 text-chart-2" /> Manufacturability Scoring</li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-20 border-t">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">Platform Capabilities</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Unified infrastructure for both drug and materials discovery workflows
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <FeatureCard
                icon={Beaker}
                title="Data & Entity Management"
                description="Organize molecules, polymers, crystals, and compounds with comprehensive metadata tracking"
              />
              <FeatureCard
                icon={Target}
                title="Virtual Screening"
                description="Prioritize candidates with configurable scoring and domain-specific filtering workflows"
              />
              <FeatureCard
                icon={FlaskConical}
                title="Docking & Property Tools"
                description="BioNeMo DiffDock for drugs, coarse-grained MD for materials, and property prediction APIs"
              />
              <FeatureCard
                icon={BarChart3}
                title="ML-Powered Predictions"
                description="ADMET, QSAR, tensile strength, thermal stability, ionic conductivity, and more"
              />
              <FeatureCard
                icon={GitBranch}
                title="Workflow Automation"
                description="Execute campaigns with automated job orchestration across both domains"
              />
              <FeatureCard
                icon={Cpu}
                title="Quantum Computing Ready"
                description="IonQ/IBM Quantum integration for optimization and electronic structure calculations"
              />
            </div>
          </div>
        </section>

        <section className="py-20 bg-muted/30 border-t">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">Drug Discovery Advantages</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                What sets Lika Sciences apart for pharmaceutical research
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <USPCard
                icon={Zap}
                title="Adaptive, Configurable Pipelines"
                description="Design custom discovery workflows with flexible pipeline configuration. Choose generators, filters, docking methods, and scoring functions."
              />
              <USPCard
                icon={Brain}
                title="Design + Simulation in the Loop"
                description="Iterate with BioNeMo molecule generation, docking, and ML predictions in a continuous feedback loop."
              />
              <USPCard
                icon={GitBranch}
                title="Fast Iteration & Parallel Campaigns"
                description="Run multiple campaigns simultaneously with real-time status tracking and job orchestration."
              />
              <USPCard
                icon={Sparkles}
                title="Internal Learning Graph"
                description="Build a self-improving platform that learns from outcomes to inform future experiments."
              />
              <USPCard
                icon={Users}
                title="Collaboration-First UX"
                description="Projects, roles, comments, and shared views for team-based drug discovery."
              />
              <USPCard
                icon={Target}
                title="Domain-Specialized Pipelines"
                description="Pre-configured workflows for CNS, Oncology, Rare Disease, and more with domain-specific oracles."
              />
            </div>
          </div>
        </section>

        <section className="py-20 border-t">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">Materials Discovery Advantages</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Purpose-built capabilities for advanced materials research
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <USPCard
                icon={Layers}
                title="Property-Driven, Simulation-in-the-Loop"
                description="Candidates optimized not just theoretically, but for real-world engineering constraints with physics-aware + ML-guided scoring loops."
              />
              <USPCard
                icon={Atom}
                title="Multi-Scale Representation"
                description="Supports monomer units, polymer chains, lattice structures, and bulk property inference from atom to material scale."
              />
              <USPCard
                icon={Settings}
                title="Adaptive Domain Pipelines"
                description="Tailored workflows for polymers, coatings, membranes, energy materials, catalysts, and composites."
              />
              <USPCard
                icon={Hexagon}
                title="Curated Materials Libraries"
                description="Structure-property archetypes with generative expansion for novel material discovery."
              />
              <USPCard
                icon={Shield}
                title="Manufacturability Awareness"
                description="Production feasibility, cost estimation, and scale-up considerations as first-class scores."
              />
              <USPCard
                icon={Brain}
                title="Cross-Domain Intelligence"
                description="Shared insights and reusable scientific knowledge between drug and materials discovery."
              />
            </div>
          </div>
        </section>

        <section className="py-20 border-t">
          <div className="max-w-7xl mx-auto px-6 text-center">
            <h2 className="text-3xl font-bold mb-4">Ready to Transform Discovery?</h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Get started with Lika Sciences today and accelerate your drug discovery and materials research programs.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <a href="/api/login">
                <Button size="lg" className="gap-2" data-testid="button-get-started">
                  Get Started
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </a>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t py-8">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center">
                <FlaskConical className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="font-semibold">Lika Sciences</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Adaptive AI Discovery Platform for Drug Discovery and Advanced Materials
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Beaker;
  title: string;
  description: string;
}) {
  return (
    <Card className="hover-elevate">
      <CardContent className="p-6">
        <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center mb-4">
          <Icon className="h-5 w-5 text-primary" />
        </div>
        <h3 className="font-semibold mb-2">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}

function USPCard({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Zap;
  title: string;
  description: string;
}) {
  return (
    <Card className="hover-elevate">
      <CardContent className="p-6 flex gap-4">
        <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center flex-shrink-0">
          <Icon className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold mb-2">{title}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </CardContent>
    </Card>
  );
}
