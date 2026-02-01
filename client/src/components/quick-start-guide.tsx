import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  FlaskConical,
  Microscope,
  Target,
  Database,
  Brain,
  FileText,
  Rocket,
  ChevronRight,
  ChevronLeft,
  X,
  Check,
  Sparkles,
  Zap,
  Beaker,
  Activity,
  HelpCircle,
} from "lucide-react";

interface GuideStep {
  id: string;
  title: string;
  description: string;
  icon: typeof FlaskConical;
  features: string[];
  route?: string;
  color: string;
}

const GUIDE_STEPS: GuideStep[] = [
  {
    id: "welcome",
    title: "Welcome to Lika Sciences",
    description: "Your comprehensive platform for accelerating drug discovery, vaccine development, and materials science research.",
    icon: Rocket,
    features: [
      "Multi-domain discovery platform",
      "AI-powered predictions with BioNeMo",
      "Integrated assay harvesting from PubChem & ChEMBL",
      "End-to-end campaign management"
    ],
    color: "text-primary"
  },
  {
    id: "discovery",
    title: "Discovery Campaigns",
    description: "Launch and manage research campaigns across drug discovery, vaccine development, and materials science domains.",
    icon: Microscope,
    features: [
      "Create campaigns for specific disease targets",
      "Track progress through discovery stages",
      "Manage compound libraries",
      "Generate comprehensive reports"
    ],
    route: "/drug-discovery",
    color: "text-blue-500"
  },
  {
    id: "assays",
    title: "Assay Harvesting",
    description: "Discover and curate assays from PubChem and ChEMBL databases to build your experimental protocol.",
    icon: FlaskConical,
    features: [
      "Browse assays by disease area",
      "5-category classification (Binding, Functional, ADME, Safety, Physicochemical)",
      "Select and export protocol JSON",
      "Link to original assay sources"
    ],
    route: "/assay-harvesting",
    color: "text-green-500"
  },
  {
    id: "predictions",
    title: "BioNeMo Predictions",
    description: "Use NVIDIA's BioNeMo models to predict IC50 values and molecular properties for your compound library.",
    icon: Brain,
    features: [
      "MegaMolBART for small molecules",
      "ESM2 for protein-based predictions",
      "Context-aware assay predictions",
      "Top hits identification"
    ],
    route: "/assay-harvesting",
    color: "text-purple-500"
  },
  {
    id: "molecules",
    title: "Molecule Registry",
    description: "Manage your compound library with SMILES validation, duplicate detection, and property calculations.",
    icon: Beaker,
    features: [
      "Import SMILES in batch",
      "3D molecular visualization",
      "Property calculations",
      "Project organization"
    ],
    route: "/molecules",
    color: "text-orange-500"
  },
  {
    id: "compute",
    title: "Compute Infrastructure",
    description: "Run computational pipelines on distributed GPU infrastructure for docking, ML predictions, and simulations.",
    icon: Zap,
    features: [
      "Multi-provider GPU nodes",
      "AutoDock Vina integration",
      "AQAffinity binding predictions",
      "Vaccine discovery pipelines"
    ],
    route: "/compute-nodes",
    color: "text-amber-500"
  },
  {
    id: "ready",
    title: "Ready to Discover!",
    description: "You're all set to start your research journey. Explore the platform and accelerate your discoveries.",
    icon: Sparkles,
    features: [
      "Start with a disease target",
      "Harvest relevant assays",
      "Run BioNeMo predictions",
      "Export your findings"
    ],
    color: "text-primary"
  }
];

const STORAGE_KEY = "lika-quickstart-completed";
const STORAGE_SEEN_KEY = "lika-quickstart-seen";

export function QuickStartGuide() {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [hasCompleted, setHasCompleted] = useState(false);

  useEffect(() => {
    const completed = localStorage.getItem(STORAGE_KEY);
    const seen = localStorage.getItem(STORAGE_SEEN_KEY);
    
    if (completed === "true") {
      setHasCompleted(true);
    } else if (!seen) {
      setTimeout(() => {
        setIsOpen(true);
        localStorage.setItem(STORAGE_SEEN_KEY, "true");
      }, 1000);
    }
  }, []);

  const handleNext = () => {
    if (currentStep < GUIDE_STEPS.length - 1) {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentStep(prev => prev + 1);
        setIsAnimating(false);
      }, 150);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentStep(prev => prev - 1);
        setIsAnimating(false);
      }, 150);
    }
  };

  const handleComplete = () => {
    localStorage.setItem(STORAGE_KEY, "true");
    setHasCompleted(true);
    setIsOpen(false);
    setCurrentStep(0);
  };

  const handleSkip = () => {
    setIsOpen(false);
    setCurrentStep(0);
  };

  const handleRestart = () => {
    setCurrentStep(0);
    setIsOpen(true);
  };

  const progress = ((currentStep + 1) / GUIDE_STEPS.length) * 100;
  const step = GUIDE_STEPS[currentStep];

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        onClick={handleRestart}
        className="relative"
        data-testid="button-quick-start"
      >
        <HelpCircle className="w-5 h-5" />
        {!hasCompleted && (
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-primary rounded-full animate-pulse" />
        )}
      </Button>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-lg p-0 overflow-hidden">
          <div className="relative">
            <div className="absolute top-0 left-0 right-0 h-1">
              <Progress value={progress} className="h-1 rounded-none" />
            </div>
            
            <div className="pt-6 px-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  {GUIDE_STEPS.map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => setCurrentStep(idx)}
                      className={`w-2 h-2 rounded-full transition-all ${
                        idx === currentStep
                          ? "bg-primary w-6"
                          : idx < currentStep
                          ? "bg-primary/50"
                          : "bg-muted"
                      }`}
                      data-testid={`step-indicator-${idx}`}
                    />
                  ))}
                </div>
                <Badge variant="secondary" className="text-xs">
                  {currentStep + 1} / {GUIDE_STEPS.length}
                </Badge>
              </div>
            </div>

            <div
              className={`px-6 pb-6 transition-all duration-150 ${
                isAnimating ? "opacity-0 translate-x-4" : "opacity-100 translate-x-0"
              }`}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-3 rounded-lg bg-muted ${step.color}`}>
                  <step.icon className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg">{step.title}</h3>
                  <p className="text-sm text-muted-foreground">{step.description}</p>
                </div>
              </div>

              <Card className="mb-4">
                <CardContent className="p-4">
                  <ul className="space-y-2">
                    {step.features.map((feature, idx) => (
                      <li
                        key={idx}
                        className="flex items-center gap-2 text-sm"
                        style={{ animationDelay: `${idx * 100}ms` }}
                      >
                        <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {step.route && (
                <a
                  href={step.route}
                  className="text-sm text-primary hover:underline flex items-center gap-1"
                  onClick={() => setIsOpen(false)}
                >
                  Go to {step.title}
                  <ChevronRight className="w-3 h-3" />
                </a>
              )}
            </div>

            <div className="flex items-center justify-between p-4 border-t bg-muted/30">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSkip}
                data-testid="button-skip-guide"
              >
                Skip
              </Button>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePrevious}
                  disabled={currentStep === 0}
                  data-testid="button-previous-step"
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  Back
                </Button>
                <Button
                  size="sm"
                  onClick={handleNext}
                  data-testid="button-next-step"
                >
                  {currentStep === GUIDE_STEPS.length - 1 ? (
                    <>
                      Get Started
                      <Sparkles className="w-4 h-4 ml-1" />
                    </>
                  ) : (
                    <>
                      Next
                      <ChevronRight className="w-4 h-4 ml-1" />
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function useQuickStart() {
  const [hasCompleted, setHasCompleted] = useState(false);

  useEffect(() => {
    const completed = localStorage.getItem(STORAGE_KEY);
    setHasCompleted(completed === "true");
  }, []);

  const resetGuide = () => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_SEEN_KEY);
    setHasCompleted(false);
  };

  return { hasCompleted, resetGuide };
}
