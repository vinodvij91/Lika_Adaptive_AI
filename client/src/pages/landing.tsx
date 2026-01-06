import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ThemeToggle } from "@/components/theme-toggle";
import { DualDomainBackground } from "@/components/scientific-background";
import { AnimatedGrid, FloatingParticles, GlowingOrbs } from "@/components/animated-grid";
import { MetricsStrip } from "@/components/metrics-strip";
import { IntegrationLogos, TrustSignals } from "@/components/integration-logos";
import { LikaLogo, LikaLogoIcon } from "@/components/lika-logo";
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
  Play,
} from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="fixed top-0 left-0 right-0 z-50 border-b bg-background/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
          <LikaLogo size="default" />
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <a href="/api/login">
              <Button data-testid="button-login">Sign In</Button>
            </a>
          </div>
        </div>
      </header>

      <main>
        <section className="relative pt-32 pb-28 overflow-hidden min-h-[90vh] flex items-center">
          <DualDomainBackground />
          <AnimatedGrid />
          <FloatingParticles />
          
          <div className="relative max-w-7xl mx-auto px-6 w-full">
            <div className="max-w-4xl mx-auto text-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-medium mb-6 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                Adaptive AI Discovery Platform
              </div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-light tracking-tight mb-6 text-white">
                <span className="tracking-[0.15em] uppercase">Lika Sciences</span>
                <span className="block bg-gradient-to-r from-cyan-400 via-teal-400 to-amber-400 bg-clip-text text-transparent mt-2 font-semibold tracking-normal normal-case">
                  Drug Discovery + Materials Science
                </span>
              </h1>
              <p className="text-lg md:text-xl text-slate-300 mb-10 max-w-3xl mx-auto leading-relaxed">
                Generative design, simulation-in-the-loop, BioNeMo + Molecular ML + Quantum pipelines
                across pharmaceuticals, polymers, catalysts, energy materials, coatings, membranes,
                and next-generation engineered materials.
              </p>
              
              <div className="flex flex-wrap items-center justify-center gap-4 mb-12">
                <a href="/api/login">
                  <Button size="lg" className="gap-2 bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 border-0 shadow-lg shadow-cyan-500/25" data-testid="button-explore-drug">
                    <FlaskConical className="h-4 w-4" />
                    Explore Drug Discovery
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </a>
                <a href="/api/login">
                  <Button size="lg" variant="outline" className="gap-2 border-amber-500/50 text-amber-400 hover:bg-amber-500/10 backdrop-blur-sm" data-testid="button-explore-materials">
                    <Hexagon className="h-4 w-4" />
                    Explore Materials Science
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </a>
              </div>

              <div className="relative max-w-2xl mx-auto">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-transparent to-amber-500/20 rounded-xl blur-xl" />
                <div className="relative grid grid-cols-3 gap-4 p-6 rounded-xl bg-slate-900/60 backdrop-blur-md border border-white/10">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">2.4M+</div>
                    <div className="text-xs text-slate-400">Compounds Scored</div>
                  </div>
                  <div className="text-center border-x border-white/10">
                    <div className="text-2xl font-bold text-white">847K</div>
                    <div className="text-xs text-slate-400">Daily Predictions</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">156</div>
                    <div className="text-xs text-slate-400">Active Campaigns</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
            <div className="w-6 h-10 rounded-full border-2 border-white/30 flex items-start justify-center pt-2">
              <div className="w-1.5 h-3 rounded-full bg-white/50" />
            </div>
          </div>
        </section>

        <MetricsStrip />

        <section className="py-20 border-t relative overflow-hidden">
          <GlowingOrbs />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card className="overflow-hidden border-cyan-500/20 bg-gradient-to-br from-cyan-950/40 to-slate-950/40 backdrop-blur-sm group">
                <CardContent className="p-8 relative">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl group-hover:bg-cyan-500/20 transition-colors" />
                  <div className="relative">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 shadow-lg shadow-cyan-500/10">
                        <FlaskConical className="h-6 w-6 text-cyan-400" />
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold">Drug Discovery</h3>
                        <p className="text-xs text-cyan-400">Small Molecules, PROTACs, Peptides</p>
                      </div>
                    </div>
                    <p className="text-muted-foreground mb-4">
                      End-to-end AI discovery with ADMET, docking, variant-aware scoring, 
                      multi-modality pipelines, and simulation-driven iteration.
                    </p>
                    <ul className="space-y-3 text-sm">
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center">
                          <Target className="h-3.5 w-3.5 text-cyan-400" />
                        </div>
                        <span className="text-muted-foreground">BioNeMo Integration</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center">
                          <Brain className="h-3.5 w-3.5 text-cyan-400" />
                        </div>
                        <span className="text-muted-foreground">ML-Guided Docking</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center">
                          <BarChart3 className="h-3.5 w-3.5 text-cyan-400" />
                        </div>
                        <span className="text-muted-foreground">ADMET Predictions</span>
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              <Card className="overflow-hidden border-amber-500/20 bg-gradient-to-br from-amber-950/40 to-slate-950/40 backdrop-blur-sm group">
                <CardContent className="p-8 relative">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-full blur-3xl group-hover:bg-amber-500/20 transition-colors" />
                  <div className="relative">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 flex items-center justify-center border border-amber-500/30 shadow-lg shadow-amber-500/10">
                        <Hexagon className="h-6 w-6 text-amber-400" />
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold">Materials Science</h3>
                        <p className="text-xs text-amber-400">Polymers, Crystals, Composites</p>
                      </div>
                    </div>
                    <p className="text-muted-foreground mb-4">
                      AI-guided materials design for coatings, membranes, catalysts, 
                      and functional materials with property-first pipelines.
                    </p>
                    <ul className="space-y-3 text-sm">
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                          <Layers className="h-3.5 w-3.5 text-amber-400" />
                        </div>
                        <span className="text-muted-foreground">Multi-Scale Representations</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                          <Atom className="h-3.5 w-3.5 text-amber-400" />
                        </div>
                        <span className="text-muted-foreground">Property Prediction</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                          <Settings className="h-3.5 w-3.5 text-amber-400" />
                        </div>
                        <span className="text-muted-foreground">Manufacturability Scoring</span>
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-20 border-t bg-gradient-to-b from-muted/10 to-background">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <p className="text-sm font-medium text-cyan-500 uppercase tracking-wider mb-2">Capabilities</p>
              <h2 className="text-3xl font-bold mb-4">Platform Features</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Unified infrastructure for both drug discovery and materials science workflows
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

        <IntegrationLogos />

        <section className="py-20 bg-gradient-to-br from-cyan-950/20 via-background to-background border-t relative overflow-hidden">
          <div className="absolute top-0 left-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-500 text-xs font-medium mb-4 border border-cyan-500/20">
                <FlaskConical className="h-3 w-3" />
                Drug Discovery
              </div>
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
                accentColor="cyan"
              />
              <USPCard
                icon={Brain}
                title="Design + Simulation in the Loop"
                description="Iterate with BioNeMo molecule generation, docking, and ML predictions in a continuous feedback loop."
                accentColor="cyan"
              />
              <USPCard
                icon={GitBranch}
                title="Fast Iteration & Parallel Campaigns"
                description="Run multiple campaigns simultaneously with real-time status tracking and job orchestration."
                accentColor="cyan"
              />
              <USPCard
                icon={Sparkles}
                title="Internal Learning Graph"
                description="Build a self-improving platform that learns from outcomes to inform future experiments."
                accentColor="cyan"
              />
              <USPCard
                icon={Users}
                title="Collaboration-First UX"
                description="Projects, roles, comments, and shared views for team-based drug discovery."
                accentColor="cyan"
              />
              <USPCard
                icon={Target}
                title="Domain-Specialized Pipelines"
                description="Pre-configured workflows for CNS, Oncology, Rare Disease, and more with domain-specific oracles."
                accentColor="cyan"
              />
            </div>
          </div>
        </section>

        <section className="py-20 bg-gradient-to-br from-amber-950/20 via-background to-background border-t relative overflow-hidden">
          <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 text-amber-500 text-xs font-medium mb-4 border border-amber-500/20">
                <Hexagon className="h-3 w-3" />
                Materials Science
              </div>
              <h2 className="text-3xl font-bold mb-4">Materials Science Advantages</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Purpose-built capabilities for advanced materials research
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <USPCard
                icon={Layers}
                title="Property-Driven, Simulation-in-the-Loop"
                description="Candidates optimized not just theoretically, but for real-world engineering constraints with physics-aware + ML-guided scoring loops."
                accentColor="amber"
              />
              <USPCard
                icon={Atom}
                title="Multi-Scale Representation"
                description="Supports monomer units, polymer chains, lattice structures, and bulk property inference from atom to material scale."
                accentColor="amber"
              />
              <USPCard
                icon={Settings}
                title="Adaptive Domain Pipelines"
                description="Tailored workflows for polymers, coatings, membranes, energy materials, catalysts, and composites."
                accentColor="amber"
              />
              <USPCard
                icon={Hexagon}
                title="Curated Materials Libraries"
                description="Structure-property archetypes with generative expansion for novel material discovery."
                accentColor="amber"
              />
              <USPCard
                icon={Shield}
                title="Manufacturability Awareness"
                description="Production feasibility, cost estimation, and scale-up considerations as first-class scores."
                accentColor="amber"
              />
              <USPCard
                icon={Brain}
                title="Cross-Domain Intelligence"
                description="Shared insights and reusable scientific knowledge between drug discovery and materials science."
                accentColor="amber"
              />
            </div>
          </div>
        </section>

        <TrustSignals />

        <section className="py-24 border-t relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-transparent to-amber-500/5" />
          <div className="relative max-w-7xl mx-auto px-6 text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to Transform Discovery?</h2>
            <p className="text-muted-foreground mb-10 max-w-xl mx-auto text-lg">
              Get started with Lika Sciences today and accelerate your drug discovery and materials science programs.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <a href="/api/login">
                <Button size="lg" className="gap-2 bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 border-0 shadow-lg shadow-cyan-500/25" data-testid="button-get-started">
                  Get Started Free
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </a>
              <Button size="lg" variant="outline" className="gap-2" data-testid="button-watch-demo">
                <Play className="h-4 w-4" />
                Watch Demo
              </Button>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t py-8 bg-muted/20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <LikaLogo size="sm" />
            <p className="text-sm text-muted-foreground">
              Adaptive AI Discovery Platform for Drug Discovery and Materials Science
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
    <Card className="hover-elevate group relative overflow-visible">
      <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      <CardContent className="p-6 relative">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center mb-4 border border-primary/20">
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
  accentColor = "cyan",
}: {
  icon: typeof Zap;
  title: string;
  description: string;
  accentColor?: "cyan" | "amber";
}) {
  const colorClasses = {
    cyan: {
      bg: "from-cyan-500/20 to-teal-500/5",
      border: "border-cyan-500/30",
      icon: "text-cyan-400",
      glow: "bg-cyan-500/10",
    },
    amber: {
      bg: "from-amber-500/20 to-orange-500/5",
      border: "border-amber-500/30",
      icon: "text-amber-400",
      glow: "bg-amber-500/10",
    },
  };

  const colors = colorClasses[accentColor];

  return (
    <Card className="hover-elevate group relative overflow-visible">
      <div className={`absolute top-4 left-4 w-20 h-20 ${colors.glow} rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity`} />
      <CardContent className="p-6 flex gap-4 relative">
        <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${colors.bg} flex items-center justify-center flex-shrink-0 border ${colors.border}`}>
          <Icon className={`h-6 w-6 ${colors.icon}`} />
        </div>
        <div>
          <h3 className="font-semibold mb-2">{title}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </CardContent>
    </Card>
  );
}
