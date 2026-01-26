import { useLocation } from "wouter";
import { useDomain } from "@/contexts/domain-context";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Beaker, 
  FileSpreadsheet, 
  Target, 
  Atom, 
  FlaskConical,
  Layers,
  Database,
  BarChart3,
  Image,
  History,
  ArrowRight
} from "lucide-react";

const DRUG_IMPORT_TYPES = [
  {
    id: "compound_library",
    title: "Compound Library",
    description: "Import molecules with SMILES notation",
    formats: ["CSV", "SDF", "SMI", "XLSX"],
    creates: "molecules",
    icon: Beaker,
  },
  {
    id: "hit_list",
    title: "Hit Lists",
    description: "Import screening hits and scores",
    formats: ["CSV", "SDF"],
    creates: "hit records",
    icon: Target,
  },
  {
    id: "assay_results",
    title: "Assay Results",
    description: "Import experimental assay data",
    formats: ["CSV", "XLSX"],
    creates: "assay_results",
    icon: FlaskConical,
  },
  {
    id: "target_structures",
    title: "Targets / Structures",
    description: "Import protein structures",
    formats: ["PDB", "mmCIF"],
    creates: "targets",
    icon: Atom,
  },
  {
    id: "sar_annotation",
    title: "SAR Annotation",
    description: "Import structure-activity data",
    formats: ["CSV"],
    creates: "learning_graph",
    icon: FileSpreadsheet,
  },
];

const MATERIALS_IMPORT_TYPES = [
  {
    id: "materials_library",
    title: "Materials Library",
    description: "Import polymers, crystals, composites",
    formats: ["CIF", "XYZ", "POSCAR", "CSV", "BigSMILES"],
    creates: "materials",
    icon: Layers,
  },
  {
    id: "material_variants",
    title: "Variants / Formulations",
    description: "Import material variants and formulations",
    formats: ["CSV", "XLSX"],
    creates: "material_variants",
    icon: Database,
  },
  {
    id: "properties_dataset",
    title: "Properties Dataset",
    description: "Import property measurements",
    formats: ["CSV", "XLSX"],
    creates: "material_properties",
    icon: BarChart3,
  },
  {
    id: "simulation_summaries",
    title: "Simulation Summaries",
    description: "Import simulation results",
    formats: ["CSV", "JSON"],
    creates: "oracle_scores",
    icon: FileSpreadsheet,
  },
  {
    id: "imaging_spectroscopy",
    title: "Imaging / Spectroscopy",
    description: "Import imaging and spectroscopy data",
    formats: ["PNG", "TIFF", "CSV"],
    creates: "artifacts",
    icon: Image,
  },
];

export default function ImportHub() {
  const [, setLocation] = useLocation();
  const { domain: globalDomain, setDomain: setGlobalDomain } = useDomain();
  
  // Map global domain to local tab value
  const domain = globalDomain === "materials" ? "materials" : "drug";
  const setDomain = (val: "drug" | "materials") => {
    setGlobalDomain(val === "materials" ? "materials" : "drug");
  };

  const importTypes = domain === "drug" ? DRUG_IMPORT_TYPES : MATERIALS_IMPORT_TYPES;

  const handleImportTypeSelect = (importType: string) => {
    setLocation(`/import/${domain}/${importType}`);
  };

  return (
    <div className="flex-1 overflow-auto p-6" data-testid="import-hub-page">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold" data-testid="text-page-title">Import Hub</h1>
            <p className="text-muted-foreground mt-1">
              Import data from files into your research platform
            </p>
          </div>
          <Button 
            variant="outline" 
            onClick={() => setLocation("/import/history")}
            data-testid="button-import-history"
          >
            <History className="h-4 w-4 mr-2" />
            Import History
          </Button>
        </div>

        <Tabs value={domain} onValueChange={(v) => setDomain(v as "drug" | "materials")}>
          <TabsList className="grid w-full max-w-md grid-cols-2" data-testid="domain-tabs">
            <TabsTrigger value="drug" data-testid="tab-drug-discovery">
              <Beaker className="h-4 w-4 mr-2" />
              Drug Discovery
            </TabsTrigger>
            <TabsTrigger value="materials" data-testid="tab-materials-science">
              <Layers className="h-4 w-4 mr-2" />
              Materials Science
            </TabsTrigger>
          </TabsList>

          <TabsContent value={domain} className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {importTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <Card 
                    key={type.id} 
                    className="cursor-pointer hover-elevate transition-all"
                    onClick={() => handleImportTypeSelect(type.id)}
                    data-testid={`card-import-${type.id}`}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-md bg-primary/10">
                          <Icon className="h-5 w-5 text-primary" />
                        </div>
                        <CardTitle className="text-base">{type.title}</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <CardDescription>{type.description}</CardDescription>
                      <div className="flex flex-wrap gap-1">
                        {type.formats.map((format) => (
                          <Badge key={format} variant="secondary" className="text-xs">
                            {format}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex items-center justify-between pt-2">
                        <span className="text-xs text-muted-foreground">
                          Creates: <span className="font-mono">{type.creates}</span>
                        </span>
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
        </Tabs>

        <Card className="bg-muted/30">
          <CardContent className="py-6">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-md bg-primary/10">
                <FileSpreadsheet className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1">
                <h3 className="font-medium">Enterprise-Grade Import</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Our import system handles files of any size with automatic format detection, 
                  validation, duplicate checking, and column mapping. Save templates for 
                  recurring imports and track all imports in the history.
                </p>
                <div className="flex gap-4 mt-3">
                  <div className="text-sm">
                    <span className="text-muted-foreground">Batch size:</span>{" "}
                    <span className="font-medium">Up to 5M+ records</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Formats:</span>{" "}
                    <span className="font-medium">CSV, Excel, SDF, CIF, XYZ, POSCAR</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
