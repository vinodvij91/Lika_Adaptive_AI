import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  FlaskConical,
  Hexagon,
  Zap,
  Sun,
  Atom,
  Beaker,
  Shield,
  Plane,
  Heart,
  Cpu,
  Building2,
  Lightbulb,
  Magnet,
  Battery,
  Droplets,
  Cloud,
  Target,
  Activity,
  Layers,
  ArrowRight,
  CheckCircle,
  Sparkles,
} from "lucide-react";

interface UseCase {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  icon: typeof FlaskConical;
  image: string;
  benefits: string[];
  industry: string;
  domain: "drug" | "materials";
}

const drugDiscoveryUseCases: UseCase[] = [
  {
    id: "small-molecule",
    title: "Small Molecule Drug Design",
    subtitle: "Accelerate hit-to-lead optimization",
    description: "Design and optimize small molecule therapeutics with AI-powered virtual screening, QSAR modeling, and multi-parameter optimization. Reduce time-to-candidate by up to 60%.",
    icon: FlaskConical,
    image: "/use-cases/drug-research.jpg",
    benefits: ["10x faster lead optimization", "Reduced synthesis cycles", "Higher success rates"],
    industry: "Pharmaceutical",
    domain: "drug",
  },
  {
    id: "protac",
    title: "PROTAC & Targeted Degraders",
    subtitle: "Next-generation therapeutics",
    description: "Design PROTACs, molecular glues, and other targeted protein degraders with specialized linker optimization and E3 ligase binding prediction.",
    icon: Target,
    image: "/use-cases/protac-design.jpg",
    benefits: ["Linker optimization", "Degradation prediction", "Selectivity profiling"],
    industry: "Biotech",
    domain: "drug",
  },
  {
    id: "virtual-screening",
    title: "High-Throughput Virtual Screening",
    subtitle: "Screen millions of compounds",
    description: "GPU-accelerated docking and ML-based scoring enable screening of billion-compound libraries in days. Identify novel scaffolds and reduce false positives.",
    icon: Activity,
    image: "/use-cases/virtual-screening.jpg",
    benefits: ["1M+ compounds/day", "GPU acceleration", "Novel scaffold discovery"],
    industry: "Drug Discovery",
    domain: "drug",
  },
  {
    id: "admet",
    title: "ADMET & Safety Prediction",
    subtitle: "De-risk candidates early",
    description: "Predict absorption, distribution, metabolism, excretion, and toxicity properties early in discovery. Identify liabilities before expensive experiments.",
    icon: Shield,
    image: "/use-cases/admet-safety.jpg",
    benefits: ["Early liability detection", "Reduced attrition", "Cost savings"],
    industry: "Pharmaceutical",
    domain: "drug",
  },
  {
    id: "sar-analysis",
    title: "SAR & Lead Optimization",
    subtitle: "Data-driven medicinal chemistry",
    description: "Analyze structure-activity relationships across assays, identify key pharmacophores, and generate optimized analogs with AI-suggested modifications.",
    icon: Layers,
    image: "/use-cases/sar-analysis.jpg",
    benefits: ["Automated SAR tables", "AI analog suggestions", "Multi-target optimization"],
    industry: "Medicinal Chemistry",
    domain: "drug",
  },
  {
    id: "oncology",
    title: "Oncology Drug Discovery",
    subtitle: "Cancer therapeutics innovation",
    description: "Specialized workflows for kinase inhibitors, antibody-drug conjugates, and immuno-oncology targets with tumor-specific ADMET models.",
    icon: Heart,
    image: "/use-cases/oncology.jpg",
    benefits: ["Kinase selectivity panels", "ADC optimization", "Resistance prediction"],
    industry: "Oncology",
    domain: "drug",
  },
];

const materialsUseCases: UseCase[] = [
  {
    id: "battery",
    title: "Next-Gen Battery Materials",
    subtitle: "Power the future of energy storage",
    description: "Discover high-capacity cathodes, stable anodes, and solid-state electrolytes for Li-ion, Na-ion, and beyond. Optimize for energy density, cycle life, and safety.",
    icon: Battery,
    image: "/use-cases/battery-materials.jpg",
    benefits: ["2x energy density targets", "10,000+ cycle stability", "Solid-state solutions"],
    industry: "Energy Storage",
    domain: "materials",
  },
  {
    id: "solar",
    title: "Photovoltaic & Solar Materials",
    subtitle: "Harvest clean energy efficiently",
    description: "Design perovskites, organic photovoltaics, and tandem solar absorbers with optimal band gaps, stability, and manufacturability for next-generation solar cells.",
    icon: Sun,
    image: "/use-cases/solar-panels.jpg",
    benefits: ["30%+ efficiency targets", "Improved stability", "Lower manufacturing costs"],
    industry: "Renewable Energy",
    domain: "materials",
  },
  {
    id: "superconductor",
    title: "Superconductor Discovery",
    subtitle: "Room-temperature superconductivity",
    description: "Explore cuprates, iron-based, and hydride superconductors with DFT validation. Target higher Tc materials for energy transmission and quantum computing.",
    icon: Zap,
    image: "/use-cases/superconductor.jpg",
    benefits: ["High-Tc discovery", "DFT validation", "Synthesis feasibility"],
    industry: "Advanced Electronics",
    domain: "materials",
  },
  {
    id: "catalyst",
    title: "Catalyst Design",
    subtitle: "Accelerate chemical reactions",
    description: "Discover HER and ORR catalysts for fuel cells and electrolyzers. Optimize activity, selectivity, and durability for green hydrogen production.",
    icon: Atom,
    image: "/use-cases/catalyst.jpg",
    benefits: ["Fuel cell catalysts", "Electrolyzer efficiency", "Reduced platinum use"],
    industry: "Clean Energy",
    domain: "materials",
  },
  {
    id: "thermoelectric",
    title: "Thermoelectric Materials",
    subtitle: "Waste heat to electricity",
    description: "Find high-ZT thermoelectric materials for waste heat recovery in automotive, industrial, and aerospace applications.",
    icon: Activity,
    image: "/use-cases/thermoelectric.jpg",
    benefits: ["High ZT discovery", "Waste heat recovery", "Automotive applications"],
    industry: "Energy Efficiency",
    domain: "materials",
  },
  {
    id: "pfas-replacement",
    title: "PFAS-Free Alternatives",
    subtitle: "Sustainable coating solutions",
    description: "Discover fluorine-free alternatives for coatings, textiles, and packaging that meet EPA compliance while maintaining hydrophobic and chemical resistance properties.",
    icon: Shield,
    image: "/use-cases/pfas-free.jpg",
    benefits: ["EPA compliant", "Biodegradable options", "Performance matching"],
    industry: "Consumer Products",
    domain: "materials",
  },
  {
    id: "aerospace",
    title: "Aerospace & Defense Materials",
    subtitle: "Lightweight, high-performance alloys",
    description: "Design Ti-Al alloys, SiC composites, and carbon fiber materials with optimal strength-to-weight ratios for aircraft, spacecraft, and defense applications.",
    icon: Plane,
    image: "/use-cases/aerospace.jpg",
    benefits: ["40% weight reduction", "High-temperature stability", "Fatigue resistance"],
    industry: "Aerospace",
    domain: "materials",
  },
  {
    id: "biomedical",
    title: "Biomedical Implant Materials",
    subtitle: "Biocompatible solutions",
    description: "Discover materials for orthopedic implants, dental prosthetics, and medical devices with bone-matching modulus and excellent biocompatibility.",
    icon: Heart,
    image: "/use-cases/biomedical.jpg",
    benefits: ["Bone modulus matching", "Osseointegration", "Toxicity-free"],
    industry: "Medical Devices",
    domain: "materials",
  },
  {
    id: "semiconductor",
    title: "Wide-Gap Semiconductors",
    subtitle: "Power electronics revolution",
    description: "Explore SiC, GaN, and novel wide-bandgap materials for power electronics, 5G communications, and electric vehicle inverters.",
    icon: Cpu,
    image: "/use-cases/semiconductor.jpg",
    benefits: ["Higher breakdown voltage", "Thermal efficiency", "EV power systems"],
    industry: "Electronics",
    domain: "materials",
  },
  {
    id: "construction",
    title: "Sustainable Construction",
    subtitle: "Low-carbon building materials",
    description: "Develop geopolymers, fly ash cement, and other low-carbon alternatives to traditional Portland cement, reducing construction's carbon footprint.",
    icon: Building2,
    image: "/use-cases/construction.jpg",
    benefits: ["80% CO2 reduction", "Industrial waste use", "Equal strength"],
    industry: "Construction",
    domain: "materials",
  },
  {
    id: "transparent-conductor",
    title: "Transparent Conductors",
    subtitle: "ITO-free display technology",
    description: "Discover graphene, silver nanowires, and AZO alternatives to indium tin oxide for flexible displays, touchscreens, and solar cells.",
    icon: Lightbulb,
    image: "/use-cases/transparent-conductor.jpg",
    benefits: ["Flexible electronics", "Cost reduction", "Indium-free"],
    industry: "Displays & Electronics",
    domain: "materials",
  },
  {
    id: "magnets",
    title: "Rare-Earth-Free Magnets",
    subtitle: "Sustainable permanent magnets",
    description: "Design permanent magnets without rare-earth elements for electric vehicles, wind turbines, and consumer electronics, reducing supply chain risks.",
    icon: Magnet,
    image: "/use-cases/magnets.jpg",
    benefits: ["Supply chain security", "Cost reduction", "EV motors"],
    industry: "Automotive & Energy",
    domain: "materials",
  },
  {
    id: "solid-electrolyte",
    title: "Solid-State Electrolytes",
    subtitle: "Safe, high-energy batteries",
    description: "Discover LGPS, LLZO, and argyrodite-type solid electrolytes for safer, higher-energy-density solid-state batteries.",
    icon: Battery,
    image: "/use-cases/solid-electrolyte.jpg",
    benefits: ["Fire-safe batteries", "Higher energy density", "Faster charging"],
    industry: "Energy Storage",
    domain: "materials",
  },
  {
    id: "water-purification",
    title: "Water Purification Membranes",
    subtitle: "Clean water for all",
    description: "Design graphene oxide, MXene, and polymer membranes for desalination, heavy metal removal, and organic contaminant filtration.",
    icon: Droplets,
    image: "/use-cases/water-purification.jpg",
    benefits: ["99%+ salt rejection", "Low energy desalination", "Fouling resistance"],
    industry: "Water Treatment",
    domain: "materials",
  },
  {
    id: "carbon-capture",
    title: "Carbon Capture & Storage",
    subtitle: "Combat climate change",
    description: "Discover MOFs, zeolites, and amine sorbents for direct air capture and flue gas CO2 capture with high capacity and low regeneration energy.",
    icon: Cloud,
    image: "/use-cases/carbon-capture.jpg",
    benefits: ["High CO2 capacity", "Low regeneration energy", "Long cycle life"],
    industry: "Climate Tech",
    domain: "materials",
  },
];

export default function UseCasesPage() {
  const [, navigate] = useLocation();

  return (
    <div className="min-h-screen bg-background">
      <section className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-chart-2/10" />
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-chart-2/10 rounded-full blur-3xl" />
        
        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          <Badge className="mb-4" variant="secondary">
            <Sparkles className="h-3 w-3 mr-1" />
            Enterprise Applications
          </Badge>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6" data-testid="text-use-cases-title">
            Powering Discovery Across Industries
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
            From life-saving drugs to sustainable materials, LIKA Sciences accelerates scientific breakthroughs for the world's most challenging problems.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Button size="lg" onClick={() => navigate("/domain")}>
              Start Your Discovery
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button size="lg" variant="outline" onClick={() => navigate("/")}>
              Learn More
            </Button>
          </div>
        </div>
      </section>

      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
              <FlaskConical className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-bold">Drug Discovery</h2>
              <p className="text-muted-foreground">Accelerate therapeutic development</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {drugDiscoveryUseCases.map((useCase) => (
              <UseCaseCard key={useCase.id} useCase={useCase} />
            ))}
          </div>
        </div>
      </section>

      <section className="py-16 px-6 bg-muted/30">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-600 to-teal-500 flex items-center justify-center">
              <Hexagon className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-bold">Materials Science</h2>
              <p className="text-muted-foreground">Discover next-generation materials</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {materialsUseCases.map((useCase) => (
              <UseCaseCard key={useCase.id} useCase={useCase} />
            ))}
          </div>
        </div>
      </section>

      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Accelerate Your Discovery?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join leading pharmaceutical companies, materials science labs, and research institutions using LIKA Sciences.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Button size="lg" onClick={() => navigate("/domain")}>
              Get Started Free
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button size="lg" variant="outline">
              Contact Sales
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}

function UseCaseCard({ useCase }: { useCase: UseCase }) {
  const Icon = useCase.icon;
  const gradientClass = useCase.domain === "drug" 
    ? "from-blue-600 to-cyan-500" 
    : "from-emerald-600 to-teal-500";
  
  return (
    <Card className="overflow-hidden hover-elevate transition-all group" data-testid={`card-use-case-${useCase.id}`}>
      <div className="relative h-48 bg-gradient-to-br from-muted to-muted/50 overflow-hidden">
        <img 
          src={useCase.image} 
          alt={useCase.title}
          className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          onError={(e) => {
            e.currentTarget.style.display = 'none';
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent" />
        <div className={`absolute top-3 left-3 px-2 py-1 rounded-md bg-gradient-to-r ${gradientClass} text-white text-xs font-medium shadow-lg`}>
          {useCase.industry}
        </div>
      </div>
      <CardContent className="p-5">
        <div className="flex items-start gap-3 mb-3">
          <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${gradientClass} flex items-center justify-center flex-shrink-0`}>
            <Icon className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold">{useCase.title}</h3>
            <p className="text-sm text-muted-foreground">{useCase.subtitle}</p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground mb-4 line-clamp-3">
          {useCase.description}
        </p>
        <div className="space-y-2">
          {useCase.benefits.map((benefit, i) => (
            <div key={i} className="flex items-center gap-2 text-sm">
              <CheckCircle className={`h-4 w-4 ${useCase.domain === "drug" ? "text-blue-500" : "text-emerald-500"}`} />
              <span>{benefit}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
