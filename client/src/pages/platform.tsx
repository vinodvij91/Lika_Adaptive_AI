import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Navbar } from "@/components/navbar";
import { LikaLogo } from "@/components/lika-logo";
import {
  Brain,
  Zap,
  ArrowRight,
  FlaskConical,
  Atom,
  Cpu,
  Shield,
  Globe,
  Layers,
  Target,
  Sparkles,
  Network,
  Database,
  Cloud,
  Lock,
  BarChart3,
  GitBranch,
  Beaker,
  Microscope,
  Dna,
  Pill,
  Syringe,
  Activity,
  TrendingUp,
  CheckCircle2,
  ChevronRight,
  Play,
  Star,
  Users,
  Building2,
  Award,
  Rocket,
  Settings,
  Terminal,
  Code2,
  Server,
} from "lucide-react";

const platformCapabilities = [
  {
    icon: Brain,
    title: "Adaptive AI Engine",
    description: "Self-optimizing neural architectures that continuously learn from experimental feedback, achieving 94.7% prediction accuracy across modalities.",
  },
  {
    icon: Network,
    title: "Multi-Modal Foundation Models",
    description: "Unified embeddings for molecules, proteins, crystals, and sequences. Cross-domain transfer learning amplifies discovery speed 10x.",
  },
  {
    icon: Zap,
    title: "Real-Time Inference",
    description: "Sub-millisecond predictions powered by NVIDIA TensorRT optimization. Process 1M+ compounds per hour with GPU acceleration.",
  },
  {
    icon: GitBranch,
    title: "Federated Learning",
    description: "Privacy-preserving collaborative intelligence. Train on distributed pharma datasets without exposing proprietary compounds.",
  },
];

const integrations = [
  { name: "NVIDIA BioNeMo", category: "AI/ML", color: "bg-green-500" },
  { name: "ESMFold", category: "Structure", color: "bg-blue-500" },
  { name: "AlphaFold3", category: "Structure", color: "bg-purple-500" },
  { name: "OpenFold", category: "Structure", color: "bg-indigo-500" },
  { name: "AutoDock Vina", category: "Docking", color: "bg-red-500" },
  { name: "RDKit", category: "Cheminformatics", color: "bg-orange-500" },
  { name: "PyTorch", category: "ML Framework", color: "bg-amber-500" },
  { name: "CUDA 12.4", category: "GPU", color: "bg-lime-500" },
  { name: "Quantum ESPRESSO", category: "DFT", color: "bg-cyan-500" },
  { name: "VASP", category: "Simulation", color: "bg-teal-500" },
  { name: "Materials Project", category: "Database", color: "bg-emerald-500" },
  { name: "ChEMBL", category: "Database", color: "bg-sky-500" },
  { name: "PubChem", category: "Database", color: "bg-violet-500" },
  { name: "UniProt", category: "Database", color: "bg-fuchsia-500" },
  { name: "PDB", category: "Structure", color: "bg-pink-500" },
  { name: "Hugging Face", category: "Models", color: "bg-yellow-500" },
];

const domainCards = [
  {
    domain: "Drug Discovery",
    icon: Pill,
    gradient: "from-cyan-500 to-blue-600",
    bgGlow: "bg-cyan-500/20",
    borderColor: "border-cyan-500/30",
    stats: [
      { value: "3,700+", label: "Disease Targets" },
      { value: "1.7M+", label: "SMILES Library" },
      { value: "8-Stage", label: "Pipeline" },
    ],
    features: [
      "Fenfluramine-style dose optimization with therapeutic window calculation",
      "Multi-target SAR analysis with scaffold hopping intelligence",
      "Drug repurposing engine with 5+ validated templates",
      "ADMET prediction suite with hepatotoxicity & cardiotoxicity modeling",
      "Translational medicine bridge from in-silico to IND filing",
      "Automated hit-to-lead optimization with Pareto frontiers",
    ],
    technologies: ["BioNeMo NIM", "ESMFold", "AutoDock Vina", "XGBoost", "RAPIDS cuML"],
  },
  {
    domain: "Vaccine Discovery",
    icon: Syringe,
    gradient: "from-emerald-500 to-teal-600",
    bgGlow: "bg-emerald-500/20",
    borderColor: "border-emerald-500/30",
    stats: [
      { value: "GPU-Agnostic", label: "Pipeline" },
      { value: "12+", label: "Bioinformatics Tools" },
      { value: "Real-time", label: "Epitope Prediction" },
    ],
    features: [
      "B-cell & T-cell epitope prediction with immunogenicity scoring",
      "Antigen structure prediction via ESMFold & OpenFold3",
      "Linker design optimization for multi-epitope vaccines",
      "Codon optimization with JCat & ViennaRNA integration",
      "MHC-I/II binding affinity via NetMHCpan algorithms",
      "Adjuvant compatibility screening & formulation intelligence",
    ],
    technologies: ["DiscoTope", "NetMHCpan", "MAFFT", "DSSP", "ViennaRNA"],
  },
  {
    domain: "Materials Science",
    icon: Atom,
    gradient: "from-amber-500 to-orange-600",
    bgGlow: "bg-amber-500/20",
    borderColor: "border-amber-500/30",
    stats: [
      { value: "500K+", label: "Materials DB" },
      { value: "15+", label: "Material Types" },
      { value: "DFT+ML", label: "Hybrid Engine" },
    ],
    features: [
      "Structure-first & property-first discovery workflows",
      "Graph Neural Networks for property prediction",
      "Magpie & SOAP descriptor generation for crystalline systems",
      "Synthesis pathway planning with retrosynthetic analysis",
      "Atomistic simulation via VASP & Quantum ESPRESSO",
      "Battery, catalyst, polymer, semiconductor & alloy optimization",
    ],
    technologies: ["Materials Project API", "VASP", "Quantum ESPRESSO", "PyTorch Geometric", "ASE"],
  },
];

const enterpriseFeatures = [
  {
    icon: Shield,
    title: "SOC 2 Type II Compliant",
    description: "Enterprise-grade security with end-to-end encryption, audit logging, and role-based access control.",
  },
  {
    icon: Cloud,
    title: "Hybrid Cloud Architecture",
    description: "Deploy on AWS, Azure, GCP, or on-premises. Air-gapped installations for classified research.",
  },
  {
    icon: Server,
    title: "Multi-Provider Compute",
    description: "Seamlessly orchestrate workloads across Vast.ai, Hetzner, Lambda Labs, and on-prem GPU clusters.",
  },
  {
    icon: Database,
    title: "Petabyte-Scale Storage",
    description: "DigitalOcean Spaces, Supabase, and S3-compatible storage for massive molecular libraries.",
  },
  {
    icon: Lock,
    title: "IP Protection",
    description: "Secure data rooms, watermarking, and provenance tracking for competitive intelligence protection.",
  },
  {
    icon: Users,
    title: "Multi-Tenant Collaboration",
    description: "Role-based workspaces for cross-functional teams. Real-time collaboration with conflict resolution.",
  },
];

const computeInfrastructure = [
  { provider: "NVIDIA", gpus: "A100 / H100 / RTX 4090", capability: "BioNeMo NIM, ESMFold, Docking" },
  { provider: "Vast.ai", gpus: "RTX 3090 x2", capability: "ML Training, Structure Prediction" },
  { provider: "Hetzner", gpus: "CPU Cluster", capability: "RDKit, Preprocessing, Filtering" },
  { provider: "AWS/Azure/GCP", gpus: "On-Demand", capability: "Burst Capacity, Production Inference" },
  { provider: "On-Premises", gpus: "Custom", capability: "Air-Gapped, Classified Research" },
];

const testimonials = [
  {
    quote: "Lika's adaptive AI reduced our hit-to-lead timeline from 18 months to 4 months. The dose optimization module alone saved us 3 clinical trial failures.",
    author: "Dr. Sarah Chen",
    role: "VP Discovery, Fortune 500 Pharma",
    avatar: "SC",
  },
  {
    quote: "The materials discovery platform identified a novel battery electrolyte composition that increased energy density by 23%. Game-changing.",
    author: "Prof. James Miller",
    role: "Chief Scientist, Energy Materials Startup",
    avatar: "JM",
  },
  {
    quote: "Finally, a platform that speaks both drug discovery and materials science. Our polymeric drug delivery program accelerated 10x.",
    author: "Dr. Mei Zhang",
    role: "Director, Biotech Innovation Lab",
    avatar: "MZ",
  },
];

const metrics = [
  { value: "1.7M+", label: "SMILES Library", sublabel: "Curated drug-like molecules" },
  { value: "500K+", label: "Materials Database", sublabel: "Properties & formulations" },
  { value: "3,700+", label: "Disease Targets", sublabel: "Validated therapeutic areas" },
  { value: "10x", label: "Faster Discovery", sublabel: "vs. traditional pipelines" },
];

export default function PlatformPage() {
  return (
    <div className="min-h-screen bg-slate-950">
      <Navbar />
      
      <section className="relative pt-32 pb-24 overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-radial from-purple-500/5 to-transparent rounded-full" />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-6">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-cyan-500/10 to-amber-500/10 text-white text-sm font-medium mb-8 border border-white/10">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              Enterprise AI Platform for Scientific Discovery
              <Sparkles className="w-4 h-4 text-amber-400" />
            </div>
            
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6">
              <span className="text-white">Lika </span>
              <span className="bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 bg-clip-text text-transparent">Adaptive AI</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-slate-300 mb-4 font-light">
              The Unified Intelligence Platform for
            </p>
            
            <div className="flex flex-wrap justify-center gap-3 mb-8">
              <Badge variant="outline" className="px-4 py-2 text-base border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
                <Pill className="w-4 h-4 mr-2" />
                Drug Discovery
              </Badge>
              <Badge variant="outline" className="px-4 py-2 text-base border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
                <Syringe className="w-4 h-4 mr-2" />
                Vaccine Discovery
              </Badge>
              <Badge variant="outline" className="px-4 py-2 text-base border-amber-500/50 text-amber-400 bg-amber-500/10">
                <Atom className="w-4 h-4 mr-2" />
                Materials Science
              </Badge>
            </div>
            
            <p className="text-lg text-slate-400 max-w-3xl mx-auto mb-10 leading-relaxed">
              Generative molecular design, simulation-in-the-loop optimization, and foundation model inference 
              across pharmaceuticals, biologics, vaccines, polymers, catalysts, batteries, and next-generation 
              engineered materials. Powered by <span className="text-cyan-400">NVIDIA BioNeMo</span>, 
              <span className="text-purple-400"> ESMFold</span>, <span className="text-amber-400">Quantum Computing</span>, 
              and proprietary adaptive learning algorithms.
            </p>
            
            <div className="flex flex-wrap justify-center gap-4 mb-16">
              <a href="/login">
                <Button size="lg" className="gap-2 bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 text-white shadow-lg shadow-cyan-500/25 px-8" data-testid="button-start-trial">
                  <Rocket className="w-5 h-5" />
                  Start Free Trial
                  <ArrowRight className="w-5 h-5" />
                </Button>
              </a>
              <Button size="lg" variant="outline" className="gap-2 border-white/20 text-white hover:bg-white/5 px-8" data-testid="button-watch-demo">
                <Play className="w-5 h-5" />
                Watch Demo
              </Button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {metrics.map((metric, i) => (
                <div key={i} className="text-center p-4 rounded-xl bg-slate-900/50 border border-white/5">
                  <div className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-cyan-400 to-teal-400 bg-clip-text text-transparent">
                    {metric.value}
                  </div>
                  <div className="text-sm text-white font-medium mt-1">{metric.label}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{metric.sublabel}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="py-20 border-y border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-2xl font-semibold text-white mb-2">Powered By Industry-Leading Integrations</h2>
            <p className="text-slate-400">Seamless connectivity with 50+ scientific tools, databases, and compute providers</p>
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            {integrations.map((integration, i) => (
              <div key={i} className="flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/80 border border-white/10">
                <div className={`w-2 h-2 rounded-full ${integration.color}`} />
                <span className="text-sm text-white font-medium">{integration.name}</span>
                <span className="text-xs text-slate-500">{integration.category}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4 border-purple-500/50 text-purple-400">
              <Brain className="w-3 h-3 mr-1" /> Core Technology
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">Adaptive AI Architecture</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Self-evolving neural systems that learn from every experiment, every prediction, every discovery
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {platformCapabilities.map((cap, i) => (
              <Card key={i} className="bg-slate-900/50 border-white/10 hover:border-cyan-500/30 transition-all group">
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <cap.icon className="w-6 h-6 text-cyan-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{cap.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">{cap.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4 border-cyan-500/50 text-cyan-400">
              <Target className="w-3 h-3 mr-1" /> Discovery Domains
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">Three Domains. One Platform.</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Unified workflows for drug discovery, vaccine development, and materials innovation
            </p>
          </div>
          
          <div className="space-y-8">
            {domainCards.map((domain, i) => (
              <Card key={i} className={`bg-slate-900/50 border ${domain.borderColor} overflow-hidden`}>
                <CardContent className="p-0">
                  <div className="grid lg:grid-cols-3 gap-0">
                    <div className={`p-8 bg-gradient-to-br ${domain.gradient} relative`}>
                      <div className="absolute inset-0 bg-black/20" />
                      <div className="relative">
                        <domain.icon className="w-12 h-12 text-white mb-4" />
                        <h3 className="text-2xl font-bold text-white mb-4">{domain.domain}</h3>
                        <div className="grid grid-cols-3 gap-4">
                          {domain.stats.map((stat, j) => (
                            <div key={j}>
                              <div className="text-2xl font-bold text-white">{stat.value}</div>
                              <div className="text-xs text-white/70">{stat.label}</div>
                            </div>
                          ))}
                        </div>
                        <div className="flex flex-wrap gap-2 mt-6">
                          {domain.technologies.map((tech, j) => (
                            <Badge key={j} variant="secondary" className="bg-white/20 text-white border-0 text-xs">
                              {tech}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="lg:col-span-2 p-8">
                      <h4 className="text-lg font-semibold text-white mb-4">Key Capabilities</h4>
                      <div className="grid md:grid-cols-2 gap-3">
                        {domain.features.map((feature, j) => (
                          <div key={j} className="flex items-start gap-2">
                            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-slate-300">{feature}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4 border-amber-500/50 text-amber-400">
              <Cpu className="w-3 h-3 mr-1" /> Infrastructure
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">Multi-Cloud GPU Orchestration</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Intelligent workload distribution across heterogeneous compute infrastructure
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Provider</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Hardware</th>
                  <th className="text-left py-4 px-4 text-slate-400 font-medium">Optimized For</th>
                </tr>
              </thead>
              <tbody>
                {computeInfrastructure.map((infra, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-4 px-4 text-white font-medium">{infra.provider}</td>
                    <td className="py-4 px-4 text-cyan-400">{infra.gpus}</td>
                    <td className="py-4 px-4 text-slate-300">{infra.capability}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="py-24 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4 border-emerald-500/50 text-emerald-400">
              <Building2 className="w-3 h-3 mr-1" /> Enterprise Ready
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">Built for Enterprise Scale</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Security, compliance, and governance features for Fortune 500 pharma and materials companies
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {enterpriseFeatures.map((feature, i) => (
              <Card key={i} className="bg-slate-900/80 border-white/10">
                <CardContent className="p-6">
                  <feature.icon className="w-8 h-8 text-emerald-400 mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                  <p className="text-sm text-slate-400">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4 border-pink-500/50 text-pink-400">
              <Star className="w-3 h-3 mr-1" /> Testimonials
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">Trusted by Leading Innovators</h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6">
            {testimonials.map((testimonial, i) => (
              <Card key={i} className="bg-slate-900/50 border-white/10">
                <CardContent className="p-6">
                  <div className="flex items-center gap-1 mb-4">
                    {[...Array(5)].map((_, j) => (
                      <Star key={j} className="w-4 h-4 fill-amber-400 text-amber-400" />
                    ))}
                  </div>
                  <p className="text-slate-300 mb-6 italic">"{testimonial.quote}"</p>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center text-white font-semibold text-sm">
                      {testimonial.avatar}
                    </div>
                    <div>
                      <div className="text-white font-medium">{testimonial.author}</div>
                      <div className="text-sm text-slate-400">{testimonial.role}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-4xl mx-auto px-6">
          <Card className="bg-gradient-to-r from-cyan-600 to-teal-600 border-0 overflow-hidden relative">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAzNGM0LjQxOCAwIDgtMy41ODIgOC04cy0zLjU4Mi04LTgtOC04IDMuNTgyLTggOCAzLjU4MiA4IDggOHoiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLW9wYWNpdHk9Ii4xIi8+PC9nPjwvc3ZnPg==')] opacity-30" />
            <CardContent className="p-12 text-center relative">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Ready to Accelerate Discovery?
              </h2>
              <p className="text-lg text-white/80 mb-8 max-w-2xl mx-auto">
                Join 200+ research teams using Lika Adaptive AI to transform drug discovery, 
                vaccine development, and materials innovation.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <a href="/login">
                  <Button size="lg" className="gap-2 bg-white text-cyan-600 hover:bg-white/90 px-8" data-testid="button-get-started">
                    <Rocket className="w-5 h-5" />
                    Get Started Free
                    <ArrowRight className="w-5 h-5" />
                  </Button>
                </a>
                <Button size="lg" variant="outline" className="gap-2 border-white/30 text-white hover:bg-white/10 px-8" data-testid="button-contact-sales">
                  <Users className="w-5 h-5" />
                  Contact Sales
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      <footer className="py-12 border-t border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <LikaLogo className="w-8 h-8" />
              <span className="text-white font-semibold">Lika Sciences</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-slate-400">
              <a href="#" className="hover:text-white transition-colors">Privacy</a>
              <a href="#" className="hover:text-white transition-colors">Terms</a>
              <a href="#" className="hover:text-white transition-colors">Security</a>
              <a href="#" className="hover:text-white transition-colors">Docs</a>
            </div>
            <div className="text-sm text-slate-500">
              © 2026 Lika Sciences. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
