import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { DualDomainBackground } from "@/components/scientific-background";
import { AnimatedGrid, FloatingParticles, GlowingOrbs } from "@/components/animated-grid";
import { MetricsStrip } from "@/components/metrics-strip";
import { IntegrationLogos, TrustSignals } from "@/components/integration-logos";
import { LikaLogo, LikaLogoLeafGradient } from "@/components/lika-logo";
import { Navbar } from "@/components/navbar";
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
  Filter,
  TestTube2,
  Activity,
  Network,
  Radar,
  Scale,
  Boxes,
  Thermometer,
  TrendingUp,
  Syringe,
  Dna,
  Microscope,
  HeartPulse,
  MousePointer2,
  Database,
  PieChart,
  Workflow,
} from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main>
        <section className="relative pt-32 pb-24 overflow-hidden min-h-[85vh] flex items-center">
          <DualDomainBackground />
          <AnimatedGrid />
          <FloatingParticles />
          
          <div className="relative max-w-7xl mx-auto px-6 w-full">
            <div className="max-w-3xl mx-auto text-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-medium mb-6 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                Lika Sciences
              </div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-semibold tracking-tight mb-6 text-white leading-tight">
                Adaptive AI platform for molecular R&D
              </h1>
              <p className="text-lg md:text-xl text-slate-300 mb-4 max-w-2xl mx-auto leading-relaxed">
                Design and rank drugs, vaccines, and advanced materials using simulation-in-the-loop AI, BioNeMo, and multi-omics integration.
              </p>
              <p className="text-sm text-slate-400 mb-10">
                Modules for drug discovery, vaccine design, and materials science.
              </p>
              
              <div className="flex flex-wrap items-center justify-center gap-3 mb-6">
                <a href="/login">
                  <Button size="lg" variant="outline" className="gap-2 border-cyan-500/40 text-cyan-300 backdrop-blur-sm" data-testid="button-explore-drug">
                    <FlaskConical className="h-4 w-4" />
                    Drug Discovery
                  </Button>
                </a>
                <a href="/login">
                  <Button size="lg" variant="outline" className="gap-2 border-emerald-500/40 text-emerald-300 backdrop-blur-sm" data-testid="button-explore-vaccine">
                    <Syringe className="h-4 w-4" />
                    Vaccine Discovery
                  </Button>
                </a>
                <a href="/login">
                  <Button size="lg" variant="outline" className="gap-2 border-amber-500/40 text-amber-300 backdrop-blur-sm" data-testid="button-explore-materials">
                    <Hexagon className="h-4 w-4" />
                    Materials Science
                  </Button>
                </a>
              </div>

              <p className="text-sm text-slate-400/80 max-w-2xl mx-auto mb-10">
                Built for biotech, pharma, and materials innovators who need in-silico evidence to de-risk drugs, vaccines, and advanced materials before committing lab spend.
              </p>

              <div className="relative max-w-3xl mx-auto">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-emerald-500/5 to-amber-500/10 rounded-lg blur-xl" />
                <div className="relative flex flex-wrap items-center justify-center gap-6 md:gap-0 md:divide-x divide-white/10 py-4 px-6 rounded-lg bg-slate-900/50 backdrop-blur-md border border-white/10">
                  <div className="text-center md:px-6">
                    <div className="text-xl font-bold text-white">1.7M+</div>
                    <div className="text-xs text-slate-400">molecules</div>
                  </div>
                  <div className="text-center md:px-6">
                    <div className="text-xl font-bold text-white">500K+</div>
                    <div className="text-xs text-slate-400">materials</div>
                  </div>
                  <div className="text-center md:px-6">
                    <div className="text-xl font-bold text-white">3,700+</div>
                    <div className="text-xs text-slate-400">disease targets</div>
                  </div>
                  <div className="text-center md:px-6">
                    <div className="text-xl font-bold text-white">94.7%</div>
                    <div className="text-xs text-slate-400">model AUC</div>
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

        <section className="py-16 border-t bg-gradient-to-b from-muted/20 to-background">
          <div className="max-w-5xl mx-auto px-6">
            <div className="text-center mb-10">
              <p className="text-sm font-medium text-cyan-500 uppercase tracking-wider mb-2">Under the hood</p>
              <h2 className="text-2xl md:text-3xl font-bold">How Lika works</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="p-5 rounded-lg border bg-card">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-3">
                  <Sparkles className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-1.5">Generative design & virtual screening</h3>
                <p className="text-sm text-muted-foreground">AI-guided hit discovery, scoring cascades, and candidate prioritization across modalities.</p>
              </div>
              <div className="p-5 rounded-lg border bg-card">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-500/20 to-green-500/10 flex items-center justify-center border border-emerald-500/30 mb-3">
                  <Brain className="h-5 w-5 text-emerald-400" />
                </div>
                <h3 className="font-semibold mb-1.5">BioNeMo-powered structure & sequence models</h3>
                <p className="text-sm text-muted-foreground">ESMFold, DiffDock, and OpenFold3 for protein structure prediction and molecular docking.</p>
              </div>
              <div className="p-5 rounded-lg border bg-card">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500/20 to-purple-500/10 flex items-center justify-center border border-violet-500/30 mb-3">
                  <Microscope className="h-5 w-5 text-violet-400" />
                </div>
                <h3 className="font-semibold mb-1.5">Fc effector & multi-omics engines</h3>
                <p className="text-sm text-muted-foreground">ADCC/CDC scoring, species translation, and genomics-to-metabolomics evidence integration.</p>
              </div>
              <div className="p-5 rounded-lg border bg-card">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 flex items-center justify-center border border-amber-500/30 mb-3">
                  <Atom className="h-5 w-5 text-amber-400" />
                </div>
                <h3 className="font-semibold mb-1.5">Quantum & materials optimization</h3>
                <p className="text-sm text-muted-foreground">IonQ/IBM Quantum integration, simulation-in-the-loop property prediction, and scale-up scoring.</p>
              </div>
            </div>
          </div>
        </section>

        <MetricsStrip />

        <section className="py-20 border-t relative overflow-hidden">
          <GlowingOrbs />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card className="overflow-hidden border-cyan-500/20 bg-gradient-to-br from-cyan-950/40 to-slate-950/40 backdrop-blur-sm group">
                <CardContent className="p-8 relative">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl group-hover:bg-cyan-500/20 transition-colors" />
                  <div className="relative">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 shadow-lg shadow-cyan-500/10">
                        <FlaskConical className="h-6 w-6 text-cyan-400" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold">Drug Discovery</h3>
                        <p className="text-xs text-cyan-400">Small Molecules, PROTACs, Peptides</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      From virtual hit discovery to assay-validated candidates with ADMET, docking, 
                      variant-aware scoring, and SAR-driven iteration.
                    </p>
                    <ul className="space-y-3 text-sm">
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center">
                          <Target className="h-3.5 w-3.5 text-cyan-400" />
                        </div>
                        <span className="text-muted-foreground">Hit Identification & Triage</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center">
                          <Brain className="h-3.5 w-3.5 text-cyan-400" />
                        </div>
                        <span className="text-muted-foreground">Screening Cascades & Prioritization</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center">
                          <BarChart3 className="h-3.5 w-3.5 text-cyan-400" />
                        </div>
                        <span className="text-muted-foreground">Assay Validation & SAR Feedback</span>
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              <Card className="overflow-hidden border-emerald-500/20 bg-gradient-to-br from-emerald-950/40 to-slate-950/40 backdrop-blur-sm group">
                <CardContent className="p-8 relative">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl group-hover:bg-emerald-500/20 transition-colors" />
                  <div className="relative">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-emerald-500/20 to-green-500/10 flex items-center justify-center border border-emerald-500/30 shadow-lg shadow-emerald-500/10">
                        <Syringe className="h-6 w-6 text-emerald-400" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold">Vaccine Discovery</h3>
                        <p className="text-xs text-emerald-400">Epitopes, Constructs, Immunogenicity</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      End-to-end vaccine design from antigen selection through epitope prediction,
                      multi-epitope construct assembly, and immunogenicity optimization.
                    </p>
                    <ul className="space-y-3 text-sm">
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-emerald-500/10 flex items-center justify-center">
                          <Dna className="h-3.5 w-3.5 text-emerald-400" />
                        </div>
                        <span className="text-muted-foreground">T-cell & B-cell Epitope Prediction</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-emerald-500/10 flex items-center justify-center">
                          <Shield className="h-3.5 w-3.5 text-emerald-400" />
                        </div>
                        <span className="text-muted-foreground">Construct Assembly & Codon Optimization</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-emerald-500/10 flex items-center justify-center">
                          <HeartPulse className="h-3.5 w-3.5 text-emerald-400" />
                        </div>
                        <span className="text-muted-foreground">mRNA Design & Stability Prediction</span>
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
                        <h3 className="text-xl font-bold">Materials Science</h3>
                        <p className="text-xs text-amber-400">Polymers, Crystals, Composites</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      Evaluate millions of material variants, run hundreds of thousands of property 
                      predictions per day, and manage dozens of concurrent campaigns.
                    </p>
                    <ul className="space-y-3 text-sm">
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                          <Layers className="h-3.5 w-3.5 text-amber-400" />
                        </div>
                        <span className="text-muted-foreground">100K-5M+ variants per campaign</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                          <Atom className="h-3.5 w-3.5 text-amber-400" />
                        </div>
                        <span className="text-muted-foreground">High-throughput property prediction</span>
                      </li>
                      <li className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                          <Settings className="h-3.5 w-3.5 text-amber-400" />
                        </div>
                        <span className="text-muted-foreground">Simulation-in-the-loop optimization</span>
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
                Unified infrastructure for drug discovery, vaccine discovery, and materials science workflows
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <FeatureCard
                icon={Beaker}
                title="Data & Entity Management"
                description="Organize molecules, polymers, crystals, antigens, and compounds with comprehensive metadata tracking"
              />
              <FeatureCard
                icon={Syringe}
                title="Vaccine Design Pipeline"
                description="Epitope prediction, construct assembly, codon optimization, and mRNA stability analysis"
              />
              <FeatureCard
                icon={Microscope}
                title="Fc Effector Modeling"
                description="Humanized mouse models with ADCC/CDC scoring, FcR affinity prediction, and species translation"
              />
              <FeatureCard
                icon={Database}
                title="Multi-Omics Integration"
                description="Genomics, transcriptomics, proteomics, and metabolomics evidence aggregation with weighted scoring"
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
                description="ADMET, QSAR, tensile strength, thermal stability, ionic conductivity, and immunogenicity"
              />
              <FeatureCard
                icon={GitBranch}
                title="Workflow Automation"
                description="Execute campaigns with automated job orchestration across all three domains"
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

        <section className="py-20 bg-gradient-to-br from-cyan-950/30 via-teal-950/20 to-background border-t relative overflow-hidden">
          <div className="absolute top-0 left-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-0 w-64 h-64 bg-teal-500/5 rounded-full blur-3xl" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-500 text-xs font-medium mb-4 border border-cyan-500/20">
                <FlaskConical className="h-3 w-3" />
                Hit Discovery Pipeline
              </div>
              <h2 className="text-3xl font-bold mb-4">From Virtual Hits to Assay-Validated Candidates</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Rigorous, scientifically-grounded workflows for modern drug discovery
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
              <div className="p-6 rounded-lg bg-slate-900/40 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-4">
                  <Target className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-2 text-white">AI-Guided Hit Discovery & Prioritization</h3>
                <p className="text-sm text-slate-400">Identify virtual hits through generative design and ML-powered prioritization scoring.</p>
              </div>
              <div className="p-6 rounded-lg bg-slate-900/40 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-4">
                  <Layers className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-2 text-white">Docking + ML Screening Cascades</h3>
                <p className="text-sm text-slate-400">Tiered virtual screening with physics-based docking and ML property filters.</p>
              </div>
              <div className="p-6 rounded-lg bg-slate-900/40 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-4">
                  <Beaker className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-2 text-white">Assay Workflows for Experimental Validation</h3>
                <p className="text-sm text-slate-400">Integrated bioassay tracking with dose-response curves and hit triage queues.</p>
              </div>
              <div className="p-6 rounded-lg bg-slate-900/40 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-4">
                  <BarChart3 className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-2 text-white">SAR-Aware Feedback Loops</h3>
                <p className="text-sm text-slate-400">Structure-activity insights refine scoring models and inform next-round design.</p>
              </div>
              <div className="p-6 rounded-lg bg-slate-900/40 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-4">
                  <Sparkles className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-2 text-white">Hit-to-Lead Optimization</h3>
                <p className="text-sm text-slate-400">Multi-modality support for small molecules, PROTACs, peptides, and fragments.</p>
              </div>
              <div className="p-6 rounded-lg bg-slate-900/40 border border-cyan-500/20 backdrop-blur-sm">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center border border-cyan-500/30 mb-4">
                  <Brain className="h-5 w-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold mb-2 text-white">Continuous Learning Platform</h3>
                <p className="text-sm text-slate-400">Self-improving models that learn from outcomes to boost future campaign success.</p>
              </div>
            </div>
          </div>
        </section>

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
                icon={Target}
                title="AI-Guided Hit Discovery"
                description="Virtual screening cascades identify and prioritize hit candidates with docking + ML scoring pipelines."
                accentColor="cyan"
              />
              <USPCard
                icon={Beaker}
                title="Assay Validation Workflows"
                description="Integrated bioassay tracking for experimental validation with dose-response and hit triage support."
                accentColor="cyan"
              />
              <USPCard
                icon={BarChart3}
                title="SAR-Aware Feedback Loops"
                description="Structure-activity relationship insights that refine models and scoring with each iteration."
                accentColor="cyan"
              />
              <USPCard
                icon={Sparkles}
                title="Hit-to-Lead Optimization"
                description="Systematic refinement from virtual hits to lead candidates across multiple modalities."
                accentColor="cyan"
              />
              <USPCard
                icon={GitBranch}
                title="Screening Cascade Management"
                description="Tiered filtering from primary screens through selectivity and ADMET to final hit selection."
                accentColor="cyan"
              />
              <USPCard
                icon={Brain}
                title="Self-Improving Platform"
                description="Internal learning graph captures outcomes and continuously improves prediction accuracy."
                accentColor="cyan"
              />
            </div>
          </div>
        </section>

        <section className="py-20 bg-gradient-to-br from-emerald-950/20 via-background to-background border-t relative overflow-hidden">
          <div className="absolute top-0 left-0 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-0 w-64 h-64 bg-green-500/5 rounded-full blur-3xl" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 text-xs font-medium mb-4 border border-emerald-500/20">
                <Syringe className="h-3 w-3" />
                Vaccine Discovery
              </div>
              <h2 className="text-3xl font-bold mb-4">Vaccine Discovery Advantages</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                End-to-end computational vaccine design with automated optimization pipelines
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <USPCard
                icon={Dna}
                title="Epitope Prediction Engine"
                description="T-cell and B-cell epitope prediction with HLA allele coverage, conservation analysis, and immunogenicity scoring across pathogen variants."
                accentColor="emerald"
              />
              <USPCard
                icon={Shield}
                title="Multi-Epitope Construct Assembly"
                description="Automated assembly of multi-epitope vaccine constructs with optimized linker design, codon adaptation (CAI), and GC content balancing."
                accentColor="emerald"
              />
              <USPCard
                icon={HeartPulse}
                title="mRNA Vaccine Design"
                description="mRNA sequence optimization with stability prediction, secondary structure analysis, and delivery formulation guidance."
                accentColor="emerald"
              />
              <USPCard
                icon={Activity}
                title="Immunogenicity Prediction"
                description="Population-level immunogenicity assessment with HLA coverage analysis and immune response modeling."
                accentColor="emerald"
              />
              <USPCard
                icon={Workflow}
                title="Automated Optimization Pipeline"
                description="Auto-running pipeline from target assignment through epitope prediction, construct assembly, and immunogenicity analysis."
                accentColor="emerald"
              />
              <USPCard
                icon={Brain}
                title="Cross-Pathogen Intelligence"
                description="Reusable epitope libraries and conservation data across 360+ disease and 10+ vaccine pipeline configurations."
                accentColor="emerald"
              />
            </div>
          </div>
        </section>

        <section className="py-20 bg-gradient-to-br from-violet-950/20 via-background to-background border-t relative overflow-hidden">
          <div className="absolute top-0 right-0 w-96 h-96 bg-violet-500/5 rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-rose-500/5 rounded-full blur-3xl" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-500/10 text-violet-400 text-xs font-medium mb-4 border border-violet-500/20">
                <Microscope className="h-3 w-3" />
                Fc Effector & Omics
              </div>
              <h2 className="text-3xl font-bold mb-4">Fc Effector Modeling & Multi-Omics Integration</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Humanized mouse models for antibody engineering and disease-agnostic multi-omics evidence aggregation
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <USPCard
                icon={Microscope}
                title="Humanized Mouse Models"
                description="Fc-gamma receptor and FcRn atlas with ADCC/CDC effector scoring for antibody and vaccine Fc engineering."
                accentColor="violet"
              />
              <USPCard
                icon={PieChart}
                title="Species Translation"
                description="Cross-species similarity modeling between human, mouse, rat, and NHP Fc receptor systems for translational confidence."
                accentColor="violet"
              />
              <USPCard
                icon={Database}
                title="Multi-Omics Evidence Layers"
                description="Aggregate genomics, transcriptomics, proteomics, and metabolomics evidence per target with weighted integrated scoring."
                accentColor="violet"
              />
              <USPCard
                icon={Dna}
                title="BioNeMo Sequence Enrichment"
                description="GPU-accelerated sequence property enrichment including stability, disorder, and aggregation prediction with CPU fallback."
                accentColor="violet"
              />
              <USPCard
                icon={Target}
                title="Disease-Agnostic Architecture"
                description="Fc Effector and Omics modules work across all 360+ disease pipelines and 10+ vaccine pipelines without configuration."
                accentColor="violet"
              />
              <USPCard
                icon={Sparkles}
                title="AI-Structured Guidance"
                description="OpenAI-powered structured UI text generation with panel titles, tooltips, and narrative summaries for every analysis."
                accentColor="violet"
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
                title="Million-Scale Variant Exploration"
                description="Explore millions of structural variants with high-throughput property prediction pipelines processing 100K-500K variants per batch."
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
                title="Simulation-in-the-Loop Optimization"
                description="Execute high-throughput property prediction pipelines and iterate toward real-world performance with distributed compute."
                accentColor="amber"
              />
              <USPCard
                icon={Hexagon}
                title="Enterprise-Scale Campaign Management"
                description="Manage dozens of concurrent discovery campaigns, each handling 420K+ variants with millions of property predictions."
                accentColor="amber"
              />
              <USPCard
                icon={Shield}
                title="Manufacturing-Ready Candidates"
                description="Production feasibility, cost estimation, and scale-up considerations as first-class scores for real-world deployment."
                accentColor="amber"
              />
              <USPCard
                icon={Brain}
                title="Cross-Domain Intelligence"
                description="Shared insights and reusable scientific knowledge between drug discovery and materials science at enterprise scale."
                accentColor="amber"
              />
            </div>
          </div>
        </section>

        <TrustSignals />

        {/* Drug Discovery Section: From Virtual Hits to Assay-Validated Leads */}
        <section className="py-24 border-t relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-teal-500/5" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-medium mb-6 border border-cyan-500/20">
                <FlaskConical className="h-4 w-4" />
                Drug Discovery
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                From Virtual Hits to Assay-Validated Leads
              </h2>
              <p className="text-muted-foreground max-w-3xl mx-auto text-lg leading-relaxed">
                Lika unifies virtual screening, hit triage, assay validation, and SAR-driven 
                optimization into one adaptive AI discovery workflow.
              </p>
            </div>
            
            <div className="mb-12 max-w-4xl mx-auto">
              <p className="text-center text-muted-foreground leading-relaxed">
                Lika doesn't stop at virtual ranking. The platform provides comprehensive tools for 
                identifying and prioritizing virtual hits, selecting the most promising compounds for assays, 
                integrating experimental results back into the discovery loop, analyzing structure-activity 
                relationships, and iterating toward stronger hit-to-lead candidates.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              <Card className="border-cyan-500/20 bg-gradient-to-br from-cyan-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center mb-4 border border-cyan-500/30">
                    <Filter className="h-6 w-6 text-cyan-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Hit Triage</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Refine virtual hit lists with docking, ADMET, uncertainty filters, synthesis 
                    feasibility, and translational relevance — creating ranked shortlists ready 
                    for assay testing.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-cyan-400/80">hit identification</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-cyan-400/80">prioritization</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-cyan-400/80">screening cascades</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-cyan-500/20 bg-gradient-to-br from-cyan-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center mb-4 border border-cyan-500/30">
                    <TestTube2 className="h-6 w-6 text-cyan-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Assay Feedback</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Upload in-vitro and cell-based assay results into structured workflows. 
                    Track dose-response behavior, experimental outcomes, and validation signals 
                    that inform the next iteration.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-cyan-400/80">assay validation</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-cyan-400/80">dose-response</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-cyan-400/80">active vs inactive</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-cyan-500/20 bg-gradient-to-br from-cyan-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-teal-500/10 flex items-center justify-center mb-4 border border-cyan-500/30">
                    <Activity className="h-6 w-6 text-cyan-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">SAR Insight</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Visualize structure-activity relationships across series, scaffolds, and 
                    analogs. Compare predicted vs experimental activity to drive rational 
                    hit-to-lead refinement.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-cyan-400/80">SAR</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-cyan-400/80">analog comparison</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-cyan-400/80">hit-to-lead</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="text-center">
              <p className="text-muted-foreground mb-6">
                See how Lika connects screening, assays, and SAR into a continuous discovery loop.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-cyan-500/50 text-cyan-400" data-testid="button-explore-hit-triage">
                    <Filter className="h-4 w-4" />
                    Explore Hit Triage
                  </Button>
                </a>
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-cyan-500/50 text-cyan-400" data-testid="button-see-assay-feedback">
                    <TestTube2 className="h-4 w-4" />
                    See Assay Feedback
                  </Button>
                </a>
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-cyan-500/50 text-cyan-400" data-testid="button-view-sar-tools">
                    <Activity className="h-4 w-4" />
                    View SAR Tools
                  </Button>
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* Vaccine Discovery Section: From Antigen to Optimized Construct */}
        <section className="py-24 border-t relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 via-transparent to-green-500/5" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 text-emerald-400 text-sm font-medium mb-6 border border-emerald-500/20">
                <Syringe className="h-4 w-4" />
                Vaccine Discovery
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                From Antigen to Optimized Vaccine Construct
              </h2>
              <p className="text-muted-foreground max-w-3xl mx-auto text-lg leading-relaxed">
                Lika automates the entire vaccine design workflow from target identification through 
                epitope prediction, construct assembly, and immunogenicity optimization.
              </p>
            </div>

            <div className="mb-12 max-w-4xl mx-auto">
              <p className="text-center text-muted-foreground leading-relaxed">
                The vaccine pipeline predicts T-cell and B-cell epitopes with HLA allele coverage, 
                filters and ranks candidates by conservation and immunogenicity, assembles multi-epitope 
                constructs with optimized linkers, and performs codon optimization and mRNA stability analysis 
                for manufacturability.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              <Card className="border-emerald-500/20 bg-gradient-to-br from-emerald-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-emerald-500/20 to-green-500/10 flex items-center justify-center mb-4 border border-emerald-500/30">
                    <Dna className="h-6 w-6 text-emerald-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Epitope Prediction</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Predict T-cell and B-cell epitopes using HLA allele binding analysis, 
                    conservation scoring across pathogen variants, and immunogenicity ranking.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-emerald-400/80">T-cell epitopes</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-emerald-400/80">B-cell epitopes</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-emerald-400/80">HLA coverage</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-emerald-500/20 bg-gradient-to-br from-emerald-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-emerald-500/20 to-green-500/10 flex items-center justify-center mb-4 border border-emerald-500/30">
                    <Shield className="h-6 w-6 text-emerald-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Construct Assembly</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Assemble multi-epitope vaccine constructs with optimized linker sequences, 
                    codon adaptation index (CAI) optimization, and GC content balancing.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-emerald-400/80">linker design</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-emerald-400/80">codon optimization</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-emerald-400/80">CAI scoring</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-emerald-500/20 bg-gradient-to-br from-emerald-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-emerald-500/20 to-green-500/10 flex items-center justify-center mb-4 border border-emerald-500/30">
                    <HeartPulse className="h-6 w-6 text-emerald-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">mRNA & Immunogenicity</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Optimize mRNA sequences for stability and manufacturability. Predict 
                    population-level immunogenicity with HLA coverage analysis and immune 
                    response modeling.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-emerald-400/80">mRNA design</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-emerald-400/80">stability</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-emerald-400/80">immunogenicity</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="text-center">
              <p className="text-muted-foreground mb-6">
                See how Lika accelerates vaccine development from antigen to optimized construct.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-emerald-500/50 text-emerald-400" data-testid="button-explore-epitope-design">
                    <Dna className="h-4 w-4" />
                    Explore Epitope Design
                  </Button>
                </a>
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-emerald-500/50 text-emerald-400" data-testid="button-view-construct-tools">
                    <Shield className="h-4 w-4" />
                    View Construct Tools
                  </Button>
                </a>
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-emerald-500/50 text-emerald-400" data-testid="button-mrna-optimization">
                    <HeartPulse className="h-4 w-4" />
                    mRNA Optimization
                  </Button>
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* Materials Science Section: From Structure to Performance */}
        <section className="py-24 border-t relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 via-transparent to-orange-500/5" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 text-amber-400 text-sm font-medium mb-6 border border-amber-500/20">
                <Hexagon className="h-4 w-4" />
                Materials Science
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                From Molecular Structure to Real-World Performance
              </h2>
              <p className="text-muted-foreground max-w-3xl mx-auto text-lg leading-relaxed">
                Lika brings structure-property understanding, simulation feedback, and 
                iterative materials optimization into a unified discovery environment.
              </p>
            </div>

            <div className="mb-12 max-w-4xl mx-auto">
              <p className="text-center text-muted-foreground leading-relaxed">
                Lika enables materials teams to explore polymer, crystal, and composite structures, 
                predict and simulate material properties, analyze structure-property relationships, 
                and iterate toward higher-performance materials for real applications — with focus on 
                engineering performance, manufacturability, and applied R&D.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              <Card className="border-amber-500/20 bg-gradient-to-br from-amber-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 flex items-center justify-center mb-4 border border-amber-500/30">
                    <Boxes className="h-6 w-6 text-amber-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Structure Libraries & Design</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Manage curated libraries of polymers, crystals, and composite formulations. 
                    Generate and explore new structural variants aligned to performance goals.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-amber-400/80">materials design</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-amber-400/80">polymers</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-amber-400/80">composites</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-amber-500/20 bg-gradient-to-br from-amber-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 flex items-center justify-center mb-4 border border-amber-500/30">
                    <Thermometer className="h-6 w-6 text-amber-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Property & Simulation Feedback</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Run property prediction and simulation workflows to evaluate thermal, 
                    mechanical, transport, and surface behavior — feeding results back into 
                    the discovery loop.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-amber-400/80">simulation-in-the-loop</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-amber-400/80">thermal stability</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-amber-400/80">conductivity</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-amber-500/20 bg-gradient-to-br from-amber-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 flex items-center justify-center mb-4 border border-amber-500/30">
                    <TrendingUp className="h-6 w-6 text-amber-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Structure-Property Insight</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Analyze structure-property relationships across material families. 
                    Compare variants, visualize performance trends, and guide the next 
                    iteration of material design.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-amber-400/80">structure-property mapping</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-amber-400/80">optimization</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-amber-400/80">analytics</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="text-center">
              <p className="text-muted-foreground mb-6">
                See how Lika accelerates property-driven materials innovation.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-amber-500/50 text-amber-400" data-testid="button-explore-materials-workspace">
                    <Boxes className="h-4 w-4" />
                    Explore Materials Workspace
                  </Button>
                </a>
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-amber-500/50 text-amber-400" data-testid="button-view-property-pipelines">
                    <Thermometer className="h-4 w-4" />
                    View Property Pipelines
                  </Button>
                </a>
                <a href="/login">
                  <Button variant="outline" className="gap-2 border-amber-500/50 text-amber-400" data-testid="button-structure-property-tools">
                    <TrendingUp className="h-4 w-4" />
                    Structure-Property Tools
                  </Button>
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* Multi-Target Discovery Section */}
        <section className="py-24 border-t relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-cyan-500/5" />
          <div className="relative max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 text-purple-400 text-sm font-medium mb-6 border border-purple-500/20">
                <Network className="h-4 w-4" />
                Network-Level Discovery
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Designed for Multi-Target and Network-Level Discovery
              </h2>
              <p className="text-muted-foreground max-w-3xl mx-auto text-lg leading-relaxed">
                Complex diseases rarely have a single-target solution. Lika evaluates compounds 
                across multiple biological targets, pathways, and safety panels — enabling rational, 
                poly-pharmacology-aware design.
              </p>
            </div>

            <div className="mb-12 max-w-4xl mx-auto">
              <p className="text-center text-muted-foreground leading-relaxed">
                Lika allows researchers to define campaigns with multiple primary and secondary targets, 
                include compensatory and safety-relevant off-targets, compute composite multi-objective scores, 
                visualize network-level effects instead of single-endpoint potency, and iterate using 
                experimental assay feedback.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              <Card className="border-purple-500/20 bg-gradient-to-br from-purple-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500/20 to-indigo-500/10 flex items-center justify-center mb-4 border border-purple-500/30">
                    <Network className="h-6 w-6 text-purple-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Pathway-Aware Campaigns</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Campaigns can include multiple efficacy, synergy, and safety targets — 
                    reflecting the biological reality of complex diseases such as Alzheimer's, 
                    oncology, and metabolic disorders.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-purple-500/20 bg-gradient-to-br from-purple-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500/20 to-indigo-500/10 flex items-center justify-center mb-4 border border-purple-500/30">
                    <Scale className="h-6 w-6 text-purple-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Multi-Objective Scoring</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Weighted composite scoring integrates target potency, safety signals, 
                    ADMET behavior, and uncertainty to reveal compounds with the strongest 
                    network-level performance.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-purple-500/20 bg-gradient-to-br from-purple-950/20 to-slate-950/20">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500/20 to-indigo-500/10 flex items-center justify-center mb-4 border border-purple-500/30">
                    <Radar className="h-6 w-6 text-purple-400" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">Radar-Plot Molecule Profiles</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Each molecule can be visualized across all targets simultaneously via 
                    radar-plot activity profiles — enabling intuitive trade-off decisions and 
                    balanced optimization.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="text-center">
              <div className="inline-block px-6 py-3 rounded-lg bg-purple-500/10 border border-purple-500/20 mb-6">
                <p className="text-purple-300 font-medium">
                  Beyond single-target screening — Lika enables network-aware, poly-pharmacology discovery.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-24 border-t relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-emerald-500/5 to-amber-500/5" />
          <div className="relative max-w-7xl mx-auto px-6 text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to Transform Discovery?</h2>
            <p className="text-muted-foreground mb-10 max-w-xl mx-auto text-lg">
              Get started with Lika Sciences today and accelerate your drug discovery, vaccine development, and materials science programs.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <a href="/login">
                <Button size="lg" className="gap-2 bg-gradient-to-r from-cyan-600 to-teal-600 border-0 shadow-lg shadow-cyan-500/25" data-testid="button-explore-platform">
                  Explore Platform
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
              Adaptive AI Discovery Platform for Drug Discovery, Vaccine Discovery, and Materials Science
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
  accentColor?: "cyan" | "amber" | "emerald" | "violet";
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
    emerald: {
      bg: "from-emerald-500/20 to-green-500/5",
      border: "border-emerald-500/30",
      icon: "text-emerald-400",
      glow: "bg-emerald-500/10",
    },
    violet: {
      bg: "from-violet-500/20 to-purple-500/5",
      border: "border-violet-500/30",
      icon: "text-violet-400",
      glow: "bg-violet-500/10",
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
