import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Beaker, Activity, Shield, Zap, TrendingUp, FlaskConical, ArrowRight } from "lucide-react";

export default function AdmetPage() {
  const admetCategories = [
    {
      title: "Absorption",
      description: "Intestinal absorption, bioavailability, Caco-2 permeability, P-gp substrate",
      icon: Activity,
      color: "from-blue-500 to-cyan-500",
      bgColor: "bg-blue-500",
      stats: "12 properties",
    },
    {
      title: "Distribution",
      description: "BBB penetration, plasma protein binding, volume of distribution",
      icon: TrendingUp,
      color: "from-emerald-500 to-teal-500",
      bgColor: "bg-emerald-500",
      stats: "8 properties",
    },
    {
      title: "Metabolism",
      description: "CYP450 inhibition, CYP450 substrates, metabolic stability",
      icon: Zap,
      color: "from-amber-500 to-orange-500",
      bgColor: "bg-amber-500",
      stats: "15 properties",
    },
    {
      title: "Toxicity",
      description: "hERG inhibition, AMES mutagenicity, hepatotoxicity, skin sensitization",
      icon: Shield,
      color: "from-red-500 to-rose-500",
      bgColor: "bg-red-500",
      stats: "10 properties",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-amber-500/5">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-8 py-10 space-y-8">
          <header className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 p-8 text-white shadow-xl">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxjaXJjbGUgY3g9IjMwIiBjeT0iMzAiIHI9IjIwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4xNSkiIHN0cm9rZS13aWR0aD0iMiIvPjwvZz48L3N2Zz4=')] opacity-40" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <Beaker className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">ADMET Predictions</h1>
                  <p className="text-amber-100">Absorption, Distribution, Metabolism, Excretion & Toxicity</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-amber-100">
                <Zap className="h-4 w-4" />
                <span>45+ pharmacokinetic and toxicity endpoints</span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            {admetCategories.map((category) => (
              <Card key={category.title} className="group cursor-pointer border-0 shadow-lg hover:shadow-xl transition-all overflow-hidden">
                <CardContent className="p-0">
                  <div className={`h-24 bg-gradient-to-br ${category.color} p-5 flex items-end`}>
                    <div className="w-12 h-12 rounded-xl bg-white/20 backdrop-blur flex items-center justify-center">
                      <category.icon className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div className="p-5">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-lg">{category.title}</h3>
                      <span className="text-xs px-2 py-1 rounded-full bg-muted text-muted-foreground">{category.stats}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{category.description}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="shadow-lg">
            <CardContent className="p-0">
              <div className="flex items-center gap-3 p-5 border-b">
                <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
                  <FlaskConical className="h-5 w-5 text-amber-500" />
                </div>
                <h2 className="font-semibold">Run Predictions</h2>
              </div>
              <div className="p-12 text-center">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 flex items-center justify-center mx-auto mb-5">
                  <Beaker className="h-9 w-9 text-amber-500" />
                </div>
                <p className="font-semibold text-lg mb-2">No ADMET predictions yet</p>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Select molecules from your library to run comprehensive pharmacokinetic and toxicity predictions
                </p>
                <Button className="gap-2 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600">
                  Select Molecules
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
