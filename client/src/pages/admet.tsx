import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Beaker, Activity, Shield, Zap, TrendingUp, FlaskConical, ArrowRight } from "lucide-react";

export default function AdmetPage() {
  const admetCategories = [
    {
      title: "Absorption",
      description: "Intestinal absorption, bioavailability, Caco-2 permeability",
      icon: Activity,
      color: "from-blue-500 to-cyan-500",
      bgColor: "bg-blue-500/10",
      textColor: "text-blue-500",
    },
    {
      title: "Distribution",
      description: "BBB penetration, plasma protein binding, VDss",
      icon: TrendingUp,
      color: "from-emerald-500 to-green-500",
      bgColor: "bg-emerald-500/10",
      textColor: "text-emerald-500",
    },
    {
      title: "Metabolism",
      description: "CYP inhibition/induction, metabolic stability",
      icon: Zap,
      color: "from-amber-500 to-orange-500",
      bgColor: "bg-amber-500/10",
      textColor: "text-amber-500",
    },
    {
      title: "Toxicity",
      description: "hERG inhibition, hepatotoxicity, mutagenicity",
      icon: Shield,
      color: "from-red-500 to-rose-500",
      bgColor: "bg-red-500/10",
      textColor: "text-red-500",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-10">
          <header className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                <Beaker className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">ADMET Predictions</h1>
                <p className="text-sm text-muted-foreground">Absorption, Distribution, Metabolism, Excretion & Toxicity</p>
              </div>
            </div>
          </header>

          <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {admetCategories.map((category) => (
              <div
                key={category.title}
                className="group p-5 rounded-xl border bg-card/50 hover:bg-card transition-colors cursor-pointer"
              >
                <div className={`w-10 h-10 rounded-xl ${category.bgColor} flex items-center justify-center mb-4`}>
                  <category.icon className={`h-5 w-5 ${category.textColor}`} />
                </div>
                <h3 className="font-medium mb-1">{category.title}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{category.description}</p>
              </div>
            ))}
          </section>

          <section className="space-y-4">
            <div className="flex items-center gap-2">
              <FlaskConical className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-medium text-muted-foreground">Run Predictions</h2>
            </div>
            <Card>
              <CardContent className="p-0">
                <div className="p-12 text-center">
                  <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <Beaker className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium mb-1">No ADMET predictions yet</p>
                  <p className="text-xs text-muted-foreground mb-4 max-w-sm mx-auto">
                    Select molecules from your library to run comprehensive ADMET predictions
                  </p>
                  <Button size="sm" className="h-8 text-xs gap-1.5">
                    Select Molecules
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}
